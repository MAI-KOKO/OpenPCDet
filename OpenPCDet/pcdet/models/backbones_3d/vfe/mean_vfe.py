import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points'] # (32000, 5, 4) (32000) batch * 16000
        """
            dim=1: 表示沿着指定的维度进行求和。在这个例子中,dim=1 意味着沿着第二个维度(从0开始计数)进行求和操作
            keepdim=False: 这是一个布尔参数，表示在计算求和后是否保持结果张量的维度
            如果设置为 False,则结果张量的维度会缩减。如果设置为 True,结果张量会保持和原始张量相同的维度
        """
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) # (32000, 4)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features) # (32000, 1)
        points_mean = points_mean / normalizer # (32000, 4)
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
