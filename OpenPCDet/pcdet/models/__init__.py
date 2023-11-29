from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # dataloader_iter = iter(train_loader) 
        # batch_dict = batch = next(dataloader_iter)
        ret_dict, tb_dict, disp_dict = model(batch_dict) # kitti_dataset.py中的data_dict少一个batch_dict['batch_size'] model里面包括损失计算部分

        """
            torch.mean(input, dim=None, keepdim=False, dtype=None)
            input 是输入的张量
            dim 是可选参数，表示在哪个维度上进行平均值计算。如果不指定，将对整个张量进行平均
            keepdim 是可选参数，表示是否保持输出张量的维度和输入张量相同。默认为 False
            dtype 是可选参数，表示输出张量的数据类型。如果不指定，将保持和输入张量相同的数据类型
        """
        loss = ret_dict['loss'].mean()
        # 用于检查一个对象（在这里是 model）是否有一个指定的属性或方法（在这里是 'update_global_step'）如果存在，返回 True，否则返回 False
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
