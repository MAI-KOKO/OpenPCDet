import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs) # (num_voxel_nonzero, 32, 64)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x) # (num_voxel_nonzero, 32, 64)
        x_max = torch.max(x, dim=1, keepdim=True)[0] # (num_voxel_nonzero, 1, 64)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # [10, 64]

        pfn_layers = []
        """
            range(start, stop, step)
                start: 序列起始值,默认为0
                stop: 序列结束值(不包含在序列内，即序列最后一个值为 stop - 1)
                step: 递增(或递减)的步长,默认为1
        """
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)) # nn.Linear + nn.BatchNorm1d
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] 
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1) # (num_voxel_nonzero) -> (num_voxel_nonzero, 1)
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
        max_num_shape[axis + 1] = -1 # [1, -1]
        """
            numpy.arange([start, ]stop, [step, ], dtype=None)
                start: 可选参数,序列的起始值,默认为0
                stop: 序列的结束值（不包含在序列内）
                step: 可选参数,序列中的步长,即两个相邻数字之间的差值,默认为1
                dtype: 可选参数,返回数组的数据类型。如果未提供,则会自动推断

            range() 是Python内置函数,返回一个迭代器对象,用于生成整数序列;
            numpy.arange() 是NumPy库中的函数,返回一个NumPy数组,用于创建数值序列的数组
        """
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape) # (1, 32)
        paddings_indicator = actual_num.int() > max_num # 广播机制 (num_voxel_nonzero, 32) 将每个voxel的num_point赋值32次与[0,1...31]作比较
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        # voxel_features: (num_voxel_nonzero, 32, 4) voxel_num_points: (num_voxel_nonzero) coords: (num_voxel_nonzero, 4) 其中4是在坐标的基础上加上了Batch_size
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # [Ncg, 1, 3]
        f_cluster = voxel_features[:, :, :3] - points_mean # [num_voxel_nonzero, 32, 3]

        """
            torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) 
                用于创建一个与输入张量(input)具有相同形状(size)和设备(device)的全零张量

            input: 输入的张量，新创建的全零张量将与此张量具有相同的形状。
            dtype(可选): 新创建的张量的数据类型。如果不指定，将采用输入张量的数据类型。
            layout(可选): 新创建的张量的布局。如果不指定，将采用输入张量的布局。
            device(可选): 新创建的张量将存储在的设备。如果不指定，将采用输入张量的设备。
            requires_grad(可选): 是否需要计算梯度，默认为 False
        """
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        # (1) (x, y, z) - (z, y, x)  (2) 一个voxel所有的点的坐标 - voxel的中心坐标  (3) + self.x_offset: 将coord移到voxel中间
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # (num_voxel_nonzero, 32, 10)

        voxel_count = features.shape[1] # 32
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0) # (num_voxel_nonzero, 32)
        """
            mask:
                tensor([[ True,  True,  True,  ..., False, False, False],
                        [ True,  True,  True,  ..., False, False, False],
                        [ True,  True,  True,  ..., False, False, False],
                        ...,
                        [ True, False, False,  ..., False, False, False],
                        [ True, False, False,  ..., False, False, False],
                        [ True, False, False,  ..., False, False, False]], device='cuda:0')
        """
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) # (num_voxel_nonzero, 32, 1)
        broadcasted_mask = mask.expand_as(features)
        # (1) 由于f_cluster与f_center的参与,本来没有特征的点也带有了特征，所以需要这些点的feature设为0
        # 处理过后features特征为0的点和voxel_features特征为0是对应的
        # (2) 扩展操作,mask的形状会被“扩展”为与features的形状相匹配，然后进行逐元素的相乘
        features *= mask # (num_voxel_nonzero, 32, 10)  
        for pfn in self.pfn_layers:
            features = pfn(features) # (num_voxel_nonzero, 1, 64)
        features = features.squeeze() # squeeze() 是一个张量操作，用于去除张量中尺寸为 1 的维度  (num_voxel_nonzero, 64)
        batch_dict['pillar_features'] = features
        return batch_dict
