import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS # [[64, 64], [64, 64]]
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k] # mlps: [[128, 64, 64], [128, 64, 64]]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE # 6
        c_out = sum([x[-1] for x in mlps]) # 128
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out # 27648

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)  [B x 2048]
                    point_head_simple里面得到的前背景点/voxel(对于raw与bev层是点,对于convi层用voxel_coord表示voxel)分类分数
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] # [B, 128, 7]
        point_coords = batch_dict['point_coords'] # [B x 2048, 4] 4: batch_im, x, y, z keypoints与batch_idx拼接得到
        point_features = batch_dict['point_features'] # [B x 2048, 128]

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1) # [B x 2048, 128] 将point_features乘于对应权重

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi( # 对每个roi采样216个grid_points
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # [B x 128, 216, 3]
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3) # [B, 128 x 216, 3]

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum() # xyz_batch_cnt: tensor([2048, 2048]

        new_xyz = global_roi_grid_points.view(-1, 3) # [B x 128 x 216, 3]
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1]) # tensor([128 x 216, 128 x 216]
        pooled_points, pooled_features = self.roi_grid_pool_layer( # 注意这里的xyz数量是少于new_xyz的,但是仍是在new_xyz的ball_query领域内从xyz采点
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),  # [B x 2048, 128]
        )  # pooled_features： [B x 128 x 216, 128] 128：两个尺度的MLP输出特征进行拼接

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        ) # [B x 128, 216, 128]
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) # [B x 128, 7]
        batch_size_rcnn = rois.shape[0]

        # 使用归一化坐标乘于roi尺寸得到的grid_point位置,并以roi中心为原点,因此这是规范化后的坐标
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B x 128, 6x6x6, 3) 
        """
            new_tensor = tensor.squeeze(dim=1)

            tensor: 原始的 PyTorch 张量
            dim: 指定要压缩的维度。如果给定的维度大小不是 1,则不会对该维度产生影响
        """
        # transfer local coords to LiDAR coords
        # 生成的grid_points是基于roi局部坐标系下的坐标,因此需要对xyz进行旋转平移,得到真正的全局坐标
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)  # # (B x 128, 6x6x6, 3)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)

        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        """
            new_ones_tensor = tensor.new_ones(size, dtype=None, device=None, requires_grad=False)

            tensor: 给定的张量，将用于指定新张量的设备和数据类型。
            size: 新张量的形状，可以是一个元组或与给定张量的形状相同的张量。
            dtype: 新张量的数据类型，如果不指定，则默认与给定张量的数据类型相同。
            device: 新张量所在的设备，如果不指定，则默认与给定张量相同的设备。
            requires_grad: 是否需要计算梯度，默认为 False
        """
        faked_features = rois.new_ones((grid_size, grid_size, grid_size)) # [6, 6, 6]
        """
            nonzero() 函数用于获取数组或张量中非零元素的索引。
            它返回一个元组，其中包含非零元素所在的位置索引。
            如果是多维数组或张量，返回的是一个元组，每个元素对应一个维度的非零元素索引数组
        """
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx] [216, 3] 
        """
            tensor([[[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        ...,

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]]], device='cuda:0')
        """
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3) [B x 128, 216, 3]
 
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6] # [B x 128, 3]
        # a = (dense_idx + 0.5) / grid_size # 得到归一化坐标
        """
        a: tensor([[[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        ...,

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]],

        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 2.],
         ...,
         [5., 5., 3.],
         [5., 5., 4.],
         [5., 5., 5.]]], device='cuda:0')
        """
        # (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) 将roi分成6x6x6的网格形状 ---- 使用归一化坐标乘于每个roi的尺寸,就可以获取每个grid point点的位置 
        # - (local_roi_size.unsqueeze(dim=1) / 2 将得到的grid point点转移到roi的中心为原点 ---- local coords
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B x 128, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer( # 对pre_boxes(box_preds与anchors进行decode所得)进行NMS处理生成proposals  
            # train: num_anchors -> 512 test: num_anchors -> 100
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        ) 
        if self.training:
            targets_dict = self.assign_targets(batch_dict) # 对proposal进行上采样 512 -> 128, 并进行正则变换、坐标转换、角度反转
            batch_dict['rois'] = targets_dict['rois'] # [B, 128, 7]
            batch_dict['roi_labels'] = targets_dict['roi_labels'] # [B, 128]

        # RoI aware pooling 对每个roi采样216个点,利用VSA特征以及keypoints进行SA操作
        pooled_features = self.roi_grid_pool(batch_dict)  # train: [B x 128, 216, Cmlp1 + Cmlp2] test: [B x 100, 216, Cmlp1 + Cmlp2]

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # [B x 128, Cmlp1 + Cmlp2, 6, 6, 6]

        # a = pooled_features.view(batch_size_rcnn, -1, 1) # [B x 128, 216 x (Cmlp1 + Cmlp2), 1]
        """
            Conv1d() 期望的输入形状是 (N, C_in, L)，其中：

            N 是批次大小(batch size);
            C_in 是输入通道数(input channels)或输入的深度;
            L 是输入的长度(input length)。

            经过 Conv1d() 一维卷积操作后，输出的张量形状为 (N, C_out, L_out)，其中：

            N 是批次大小；
            C_out 是输出通道数(output channels)或输出的深度；
            L_out 是输出的长度(output length),可能与输入的长度 L 不同，取决于卷积操作的参数（如步长、填充等）

            torch.nn.Conv2d() 用于实现二维卷积操作，其参数中的 out_channels 决定了卷积层输出的通道数，每个输出通道对应一个卷积核
        """
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1)) # train: [B x 128, 256, 1] test： [B x 100, 256, 1]
        # 如果 A 是一个三维张量 (N, C, H)，那么 A.transpose(1, 2) 将会将 C 和 H 维度交换，形成新的张量 (N, H, C)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # [B x 128/100, 1] 先执行tranpose再执行contiguous
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # [B x 128/100, 7] train: 128 test: 100

        if not self.training:
            # test模式下 proposal_layer 里面：
            # rois.shape: (batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            ) # num_rois: train: [B, 128, 7] test: [B, 100, 7] 
            batch_dict['batch_cls_preds'] = batch_cls_preds # 将第二阶段的batch_cls_preds对字典中的batch_dict['batch_cls_preds']进行更新(原来为第一阶段的输出结果)
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
