import numpy as np
import torch
import torch.nn as nn



class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]] # 非关键字参数(位置参数)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [ # 下采样层
                nn.ZeroPad2d(1), # 二维零填充，表示在图像的四个边缘(上、下、左、右)各填充一个像素的零值
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ), # padding=0 不在输入的周围添加额外的像素值
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]): # 特征融合层
                # extend() 方法接受一个可迭代对象（如列表、元组、集合等）作为参数，将该可迭代对象中的元素逐个添加到列表的末尾
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01), # 在输入特征图的边缘周围添加 1 个像素的填充值
                    nn.ReLU()
                ])
            # append()：用于向 Python 的列表中添加元素，它不创建新的列表，而是直接将新元素添加到列表的末尾
            # nn.Sequential()：是 PyTorch 神经网络模块，用于将多个神经网络层以序列的方式连接在一起
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0: # 上采样层
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) # 512 将每个下采样尺度的特征上采样到一个尺度，然后进行拼接
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict: # HeightCompression输出的batch_dict
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features'] # (B, num_bev_features*nz, ny, nx)  nx、ny、nz = POINT_CLOUD_RANGE / VOXEL_SIZE
        ups = []
        ret_dict = {}
        x = spatial_features # (B, num_bev_features*nz, ny, nx)
        for i in range(len(self.blocks)): # 2
            # 尽管stride相等,但是第二次下采样前的尺度和上采样后的尺度不一样,这是因为下采样与上采样的kernel_size与padding不一样 
            x = self.blocks[i](x) #  (B, num_bev_features*nz, ny_i, nx_i)    

            stride = int(spatial_features.shape[2] / x.shape[2]) 
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0: # 2
                ups.append(self.deblocks[i](x)) # ups[i]: (B, num_upsample_filters[i], ny_up, nx_up) 每个layer层上采样输出的特征图大小是一样的
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1) # (B, sum_num_upsample_filters[i], ny_up, nx_up)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x # (B, sum_num_upsample_filters[i], ny_up, nx_up) 

        return data_dict
