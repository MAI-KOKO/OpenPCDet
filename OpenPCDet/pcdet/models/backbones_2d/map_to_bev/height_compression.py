import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # model_cfg: {'NAME': 'HeightCompression', 'NUM_BEV_FEATURES': 256}
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'] # feature: (N4, 128) spatial_shape: [200, 176, 2]
        """
            dense() 用于将稀疏卷积张量转换为密集张
                在稀疏卷积中，数据通常以稀疏张量的形式存储，这意味着只有非零元素和它们的索引被存储
                而在某些情况下，需要将稀疏数据转换为密集形式，这样所有元素的数值都被存储，即使是零值元素
        """
        spatial_features = encoded_spconv_tensor.dense() # (B, 128, 2, 200, 176)
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W) # (2, 256, 200, 176) .view() 用于改变张量的形状，但并不改变张量的元素数量
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
