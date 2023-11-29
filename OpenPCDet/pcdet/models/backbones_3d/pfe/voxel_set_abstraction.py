import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


def bilinear_interpolate_torch(im, x, y): 
    """
    Args:
        im: (H, W, C) [y, x] [200, 176, 256]
        x: (N)
        y: (N)

        双线性插值是一种在二维网格中估计某个位置的方法，通过对四个相邻的点进行加权平均来计算目标点的值

        假设有一个二维网格，网格上的每个点都有一个特定的值,要在这个网格中的某个位置进行双线性插值,
        需要找到目标点附近的四个最近邻点(通常是左上、右上、左下、右下的四个点),
        然后根据目标点在水平和垂直方向上的相对位置来确定权重,对四个点的值进行加权平均
    Returns:

    """
    x0 = torch.floor(x).long() # 点云所在feature_grid的左上角坐标 [2048]
    x1 = x0 + 1 # 点云所在feature_grid的右下角坐标

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0] # 从图像 im 中提取位于 (y0, x0) 处的特征 y0 是行索引，x0 是列索引 [2048, 256]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd) # t() tranpose [2048, 256]
    return ans


class VoxelSetAbstraction(nn.Module): # 注释为pv_rcnn
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)): # mlps: [[16, 16], [16, 16]]
                mlps[k] = [mlps[k][0]] + mlps[k] # [16, 16] -> [16, 16, 16]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps]) # mlps: [[19, 16, 16], [19, 16, 16]] c_in: 32

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k] # [16, 16] -> [1, 16, 16]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0] # 计算点云位于x轴上的第几个grid [2, 2048]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride # 计算点云位于bev特征图的位置 [2, 2048]
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k] # [2048]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C) [200, 176, 256]
            # 由于计算出来的cur_x_idxs、cur_y_idxs是带小数部分的虚拟像素坐标,所以要通过双线性插值获得虚拟像素坐标的特征
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs) # [2048, 256]
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0)) # [1, 2048, 256]

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0) [B, 2048, 256]
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4] # batch_dict['points']: [B, x, y, z, r]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C) # 对于raw,使用的是点特征(一般为反射率r);对于bev,使用的是bev双线性插值后的特征;对于3D_convi,使用的的是voxel特征
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict) # 采集一个batch在源空间的关键点 [B, 2048, 3] 

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            ) # [2, 2048, 256]
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3) # [4096, 3] 此时new_xyz是一个batch的数据
        '''
            tensor.fill_(value)
            其中 tensor 是要填充的张量，而 value 是要用来填充的标量值

            torch.new_zeros(size, dtype=None, device=None, requires_grad=False)
            size: 一个表示新张量形状的元组或列表
            dtype: 可选参数，指定新张量的数据类型
            device: 可选参数，指定新张量所在的设备(例如，"cuda" 表示 GPU)
            requires_grad: 可选参数，指定是否需要计算梯度
        '''
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints) # shape: [2] data: [2048, 2048]

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int() # shape: [2] data: [0, 0]
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum() # 统计batch中每帧数据点云的数量
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None # 将batch_dict['points'] xyz之后的特征作为point_features

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt, # shape: [2] data: [num_frame1, num_frame2]
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            ) # pooled_features: [4096, 32]
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1)) # point_features_list[0]: [2, 2048, 32]

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices # [32000, 4] VoxelBackBone8x的forward里面indices=voxel_coords
            xyz = common_utils.get_voxel_centers( # 获取在当前尺度下voxel_centers的坐标
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name], # 下采样倍率
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            ) # [Ni, 3] N0 = 32000 Batch_size x 16000
            xyz_batch_cnt = xyz.new_zeros(batch_size).int() # data: tensor([0, 0]
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum() # data: tensor([16000, 16000]

            pooled_points, pooled_features = self.SA_layers[k]( 
                xyz=xyz.contiguous(), # 使用voxel_centers表示了这个voxel,ball里面有多少个xyz点,就代表了有多少个voxel,实现了VSA的目的
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(), # k = 0 [32000, 16] k = 2 [N1, 32] voxel的特征
            ) # k = 0 [4096, 32] k = 1 [4096, 64]
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1)) # point_features_list[2]: [2, 2048, 32] 前面还有bev以及raw_points的特征
 
        point_features = torch.cat(point_features_list, dim=2) # [2, 2048, 640] 
        """
            torch.arange(start=0, end, step=1, dtype=None, layout=torch.strided, device=None, requires_grad=False)

            start: 数列的起始值，默认为 0
            end: 数列的结束值（不包括），必须指定
            step: 数列中每两个值之间的差，步长，默认为 1
            dtype: 可选参数，指定新张量的数据类型
            layout: 可选参数，指定张量的布局，默认为 torch.strided
            device: 可选参数，指定新张量所在的设备(例如，"cuda" 表示 GPU)
            requires_grad: 可选参数，指定是否需要计算梯度

        """
        # a = batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1) # shape: [2, 1]
        # b = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]) # data: tensor([[0, 0, 0,  ..., 0, 0, 0], [1, 1, 1,  ..., 1, 1, 1]], device='cuda:0') shape: [2, 2048]
        # view(-1) 将张量重塑为一个只有一维的张量，而该张量的长度（元素数量）由原始张量的形状来确定
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1) # data: tensor([0, 0, 0,  ..., 1, 1, 1], device='cuda:0') shape: [4096]
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1) # [4096, 4] 前2048个关键点作为第一帧的点,后2048个点作为第二帧的点

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1]) # [4096, 640]
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1])) # Linear,BN,Relu [4096, 128]

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict
