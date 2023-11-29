import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner



class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg # DenseHead的字典
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        """
            getattr(object, 'attribute_name', default)
                * object 是要获取属性的对象
                * 'attribute_name' 是要获取的属性或方法的名称
                * default 是可选参数，如果指定的属性或方法不存在，则返回默认值
        """
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors( # 生成anchors
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size  
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg) # 初始化target_assigner

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        # pointpillar经过2d_backbone后,feature图由(B, num_bev_features*nz, ny, nx) -> (B, sum_num_upsample_filters[i], ny_up, nx_up) 
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg] # [array([nx_up, ny_up]),array([nx_up, ny_up]), array([nx_up, ny_up])]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7: # anchor_ndim=self.box_coder.code_size
            for idx, anchors in enumerate(anchors_list):
                """
                    anchors.new_zeros() 用于创建与输入张量相同 device 和 dtype 的全零张量

                    [*anchors.shape[0:-1], anchor_ndim - 7]: 创建新张量形状的列表
                        ---- [0:-1] 表示从索引 0 开始,直到但不包括索引 -1 处的元素  
                        ---- *: 扩展语法,用于解包元组或列表(将切片的结果展开为一个新的列表)
                """
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])  # (nz, ny_up, nx_up, num_anchor_size, num_anchor_rotation, 1)
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1) # (nz, ny_up, nx_up, num_anchor_size, num_anchor_rotation, 8)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        """
            self.add_module() 是 PyTorch 中 Module 类的方法，用于向模型添加子模块
                * 'cls_loss_func' 是要添加的模块的名称
                * loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0) 是要添加的模块对象
        """
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']) # getattr(): 获取对象的属性值或方法
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8) 7 + cls
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self): # 该处注释为Pointpillar
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # assign_targets里面得到的每个anchor的标签,正样本为gt类别,负样本为0,其余为-1
        batch_size = int(cls_preds.shape[0])
        """
            box_cls_labels >= 0 是一个逐元素的比较操作，返回一个布尔型张量
            其中元素为 True 表示对应位置的类别标签大于等于零
            而元素为 False 表示对应位置的类别标签小于零
        """
        cared = box_cls_labels >= 0  # [N, num_anchors] True的位置代表了前景和背景anchor，False位置代表iou阈值在前景和背景之间的anchor
        positives = box_cls_labels > 0 # [N, num_anchors] 正样本为True,其余为False
        negatives = box_cls_labels == 0 # 背景(负样本)为True,其余为False
        negative_cls_weights = negatives * 1.0
        # tensor = torch.tensor([True, False, False]) result = tensor * 2 输出结果为: tensor([2, 0, 0]) 
        cls_weights = (negative_cls_weights + 1.0 * positives).float() # 将正样本以及背景anchor的权重设为1.0,其余设为0
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        """ 
            sum(1, keepdim=True) 是 PyTorch 中对张量进行求和的操作，其中：

            * 1 是指定沿着第二个维度(维度索引从0开始)进行求和。
            * keepdim=True 表示保持原始张量的维度。
            
            假设有一个张量 tensor,它的形状为 (m, n)，那么 sum(1, keepdim=True) 的结果将是一个形状为 (m, 1) 的张量，其中每个元素是原始张量沿着第二个维度的和
        """
        pos_normalizer = positives.sum(1, keepdim=True).float() # [B, 1] 计算每个batch正样本数量
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels) # [N, num_anchors] 将dont'care(-1 x 0)以及负样本(0 x 1)设为0，正样本(gt_label x 1)保持gt_label

        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        ) # [N, num_anchors, 4] 4 = don't care and 负样本 + Car + Pedestrian + Cyclist
        """
            scatter_() 是 PyTorch 张量的一个原地操作,用于按照给定的索引和值将元素散布(scatter)到张量中,这个操作可以用于在指定位置更新张量的值

            scatter_(dim, index, src)           
                im: 表示在哪个维度上进行散布操作
                index: 包含用于确定散布位置的索引的张量
                src: 包含要散布的值的张量
        """
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0) # 根据cls_targets的分类标签在one_hot_targets最后一个维度对应类别位置处赋1
        cls_preds = cls_preds.view(batch_size, -1, self.num_class) # (B, ny_up, nx_up, 18) -> [B, num_anchor_allcls, 3] 
        one_hot_targets = one_hot_targets[..., 1:] # [N, num_anchors, 3] N为batch_size
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, num_anchors, 3]
        """
            如果 cls_loss_src 是一个形状为 (batch_size, num_classes) 的张量，每行表示一个样本，每列表示一个类别
            那么 cls_loss_src.sum() 就是对整个 batch 中所有样本、所有类别的分类损失求和的结果
        """
        cls_loss = cls_loss_src.sum() / batch_size 

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        # rts = [rt_cos, rt_sin] rt_cos = torch.cos(rg) - torch.cos(ra) 
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1]) # [B, num_anchor, 1] 
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1) # [B, num_anchor, 8]
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod # 将box_reg_targets与anchor进行decode得到rot_gt,并与pi比较得到gt朝向(1: rot_gt > pi; 0: rot_gt < pi),最后返回one_hot编码
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1]) # [B, num_anchor_allcls, code_size]
        rot_gt = reg_targets[..., 6] + anchors[..., 6] # [B, num_anchor_allcls]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi) 
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long() # 判断offset_rot是否大于pi,大于返回1,否则返回0
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1) # [B, num_anchor_allcls]

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device) # [B, num_anchor_allcls, 2] 
            # 根据dir_cls_targets将每个anchor的direction对应位置赋1
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0) # [B, num_anchor_allcls, 2] 2：正向、反向 
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self): # 此处注释为Pointpillar
        box_preds = self.forward_ret_dict['box_preds'] # (B, ny_up, nx_up6, 6 x code_size)
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # [B, num_anchor_allcls] 前景anchor为对应gt类别(1,2,3),背景anchor为0,其他anchor为-1
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0 # [B, num_anchor_allcls] tensor([[False, False, False,  ..., False, False, False]], device='cuda:0')
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float() # [B, 1]
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3) # [nz, ny_up, nx_up, num_size * 3, num_dir, code_size]
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1) # [B, num_anchor_allcls, code_size]
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1]) # [B, num_anchor_allcls, code_size]
        # sin(a - b) = sinacosb-cosasinb
        # 在self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)里生成bbox_targets，即box_reg_targets reg(regression): 回归
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets) # 对角度残差进行编码
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [B, num_anchor_allcls, code_size]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS) # [B, ny_up, nx_up, 12] -> [B, num_anchor_allcls, 2]
            weights = positives.type_as(dir_logits) # [B, num_anchor_allcls] 在正样本处权值为1,其他地方为0
            """
                torch.clamp(input, min, max, out=None) 将张量 input 中的每个元素限制在 [min, max] 范围内
                input 是输入的张量
                min 是要限制的下界
                max 是要限制的上界
                out 是可选参数，用于指定输出张量的位置

                torch.sum(input, dim=None, keepdim=False, dtype=None)
                input: 输入的张量
                dim: 指定在哪个维度上进行求和。如果不指定，则对整个张量进行求和
                keepdim: 是否保持输出张量的维度和输入张量相同，默认为 False
                dtype: 指定输出张量的数据类型，默认为 None,表示保持和输入张量相同的数据类型
            """
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0) # [B, num_anchor_allcls] 将权重归一化
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3) #  [nz, ny_up, nx_up, num_anchor_size * num_cls, num_anchor_dir, code_size]
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0] # num_anchor_allcls
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1) # (B, num_anchor_allcls, code_size) 将生成的anchor复制batch_size份
        # a = isinstance(cls_preds, list) # False
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds  # (B, ny_up, nx_up, 18) -> (B, num_anchor_allcls, 3)
        
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1) # (B, ny_up, nx_up, 48) -> (B, num_anchor_allcls, code_size)
        
        # Pointpilalr batch_box_preds: (B, num_anchor_allcls, 8) -> (B, num_anchor_allcls, 7) 因为回归参数使用sin、cos两个量,计算得到的角度只有一个量
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors) # 根据anchor和预测box的回归参数计算预测box的位置参数

        # 根据方向分类结果对计算后的预测角度进一步处理
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET # 0.78539(45度)
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET # 0.0
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1) # (B, num_anchor_allcls, num_anchor_dir)
            
            # torch.max(dir_cls_preds, dim=-1) 返回一个元组，其中包含两个张量：
            # 第一个张量包含了沿着指定维度（这里是最后一个维度）的最大值，第二个张量包含了最大值所在的索引
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1] # (B, num_anchor_allcls) 取出所有方向预测结果：正向或反向

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS) # pi
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            ) # (B, num_anchor_allcls) 当 batch_box_preds[..., 6] - dir_offset 在周期内时,dir_rot = batch_box_preds[..., 6] - dir_offset
            
            # 如果dir_labels==1,说明反向,需要加回180度
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds # (B, num_anchors, 3) (B, num_anchors, 7)

    def forward(self, **kwargs):
        raise NotImplementedError
