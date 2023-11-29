import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        # super().__init__() 是调用父类（或超类）的构造函数，初始化从父类继承的属性
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location) # 6 每个类别有两个anchor(两个朝向),一共有三个类别

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, # 18 3个种类、2个朝向的acnhor预测三个类别 3 * 2 * 3
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, # code_size: 使用sin()编码角度时为8,否则为7
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, # NUM_DIR_BINS 分类朝向的个数
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict): 
        spatial_features_2d = data_dict['spatial_features_2d'] # (B, sum_num_upsample_filters[i], ny_up, nx_up)

        cls_preds = self.conv_cls(spatial_features_2d) # (B, 18, ny_up, nx_up)
        box_preds = self.conv_box(spatial_features_2d) # (B, 6 x code_size, ny_up, nx_up)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] (B, ny_up, nx_up, 18)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] (B, ny_up, nx_up6, 6 x code_size)

        self.forward_ret_dict['cls_preds'] = cls_preds # 在基类AnchorHeadTemplate的构造函数里self.forward_ret_dict = {}
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) # (B, 12, ny_up, nx_up)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() # (B, ny_up, nx_up, 12)
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            # 返回一个batch所有类别的target信息
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds # (B, num_anchor_allcls, 3)
            data_dict['batch_box_preds'] = batch_box_preds # (B, num_anchor_allcls, 7) 注意此处不是code_size,已经decode过了
            data_dict['cls_preds_normalized'] = False

        return data_dict
