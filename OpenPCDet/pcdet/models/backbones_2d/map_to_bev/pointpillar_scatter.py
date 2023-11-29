import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        """
            transform_points_to_voxels():
                grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
        """
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
            transform_points_to_voxels():
                data_dict['voxel_coords'] = coordinates 
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # pillar_features: (num_voxel_nonzero, 64) coords: (num_voxel_nonzero, 4)
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1 # .item() 方法可以得到标量值
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device) # (num_bev_features, num_voxel_withzero)

            batch_mask = coords[:, 0] == batch_idx # (num_voxel_nonzero)
            this_coords = coords[batch_mask, :] # (num_voxel_nonzero_onebatch, 4) ---- (B, z, x, y) 获取第i个batch的voxel的坐标
            """
                由PillarVFE可得coords: (B, z, y, x) 因为在points_to_voxel中采用了reverse_index
                f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            """
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # 根据坐标获取对应的voxel索引
            indices = indices.type(torch.long) # (num_voxel_nonzero_onebatch)
            """
                pillars = pillar_features[batch_mask, :]: 使用布尔掩码 batch_mask 对 pillar_features 进行索引
                这将选择 pillar_features 张量中所有满足条件的行（对应 batch_mask 中值为 True 的位置）
            """
            pillars = pillar_features[batch_mask, :] # (num_voxel_nonzero_onebatch, 64) 获取第i个batch的voxel的特征
            pillars = pillars.t() # 将矩阵 pillars 进行转置操作 (64, num_voxel_nonzero_onebatch)
            spatial_feature[:, indices] = pillars # 将对应特征放回到稀疏空间对应位置
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # (B, num_bev_features, num_voxel_withzero)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # PseudoImage
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
