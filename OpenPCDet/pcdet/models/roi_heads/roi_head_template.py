import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer

# 此处代码无特殊注释为PVRCNN

class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        """
            value = dictionary.get(key, default)
            dictionary: 字典对象
            key: 要检索的键
            default: 可选参数，如果键不存在时返回的默认值
        """
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None # pvrcnn里面 self.forward_ret_dict = targets_dict

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds'] # [B, num_boxes, 7] 在anchor_head_singer的generate_predicted_boxes(predict_boxes_when_training)里传入,这里的num_boxes等于num_anchors
        batch_cls_preds = batch_dict['batch_cls_preds'] # [B, num_boxes, 3] num_anchors == 2(anchor朝向) x \
        # 200 x 176 (feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]) x 3(classes)
        # new_zeros_tensor = torch.new_zeros(size, dtype=None, device=None, requires_grad=False)
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1])) # [B, 512, 7]
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE)) # [B, 512]
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long) # [B, 512]

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask] # [num_anchors, 7] 取出第一帧的位置参数(注意不是回归残差,已经经过decode处理了)
            cls_preds = batch_cls_preds[batch_mask] # [num_anchors, 3]

            # pre_box的分类分数最大值以及对应的类别,这里的pre_box就是经过decode之后的anchor
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1) # [num_anchors]  cur_roi_labels: tensor([0, 1, 2,  ..., 0, 0, 1], device='cuda:0')

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms( # 相当于执行Second的后处理中的NMS操作
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config # 传入最大类别预测分数,位置参数
                ) # 返回NMS过后的pre_box索引

            # 将NMS后的pre_box作为proposals
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois # [B, 512, 7]
        batch_dict['roi_scores'] = roi_scores # [B, 512]
        batch_dict['roi_labels'] = roi_labels + 1 # [B, 512]
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        """
            value = batch_dict.pop('batch_index', None)

            * batch_dict: 是一个字典对象
            * 'batch_index': 是要删除的键的名称
            * None: 是可选参数,表示如果键不存在,pop 方法会返回 None
        """
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict) # 对proposal进行上采样 512 -> 128

        rois = targets_dict['rois']  # (B, N, 7 + C) [B, 128, 7]
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1) [B, 128, 8] proposal对应的gt
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach() # detach() 返回一个新的张量,它具有与原始张量相同的数值，但不再追踪梯度


        # canonical transformation 正则变换
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)

        # 1、平移: 原点位于box proposal的中心
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center 
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry 

        # 2、旋转：X'指向proposal的头部方向，Z'垂直于X' 
        gt_of_rois = common_utils.rotate_points_along_z( 
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])


        # flip orientation if rois have opposite orientation 将大于90度小于270度的角度值调整到-90度到+90度
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi [B, 128]
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi # [B, 128]
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1) # ProposalTargetLayer得到,大于REG_FG_THRESH的roi掩码
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size] # rois经过canonical transformation所得
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0  # 由于gt_boxes3d_ct是规范化坐标,以rois_anchor的中心为原点 roi_center = rois[:, :, 0:3]
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch( # 计算真值与roi的定位残差
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            ) 

            rcnn_loss_reg = self.reg_loss_func( 
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            
            # 计算前景roi的回归损失,当fg_sum为0时,rcnn_loss_reg也为0
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()

                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)

                batch_anchors[:, :, 0:3] = 0 # 将rcnn_boxes3d的xyz规范化
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size) 

                # transfer local coords to LiDAR coords 以下的旋转和平移只对xzy有影响
                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)

                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls'] 
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1) # ProposalTargetLayer里面得到的fg、bg、interval的分类标签
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            # cls_layers用于预测roi的confidence,即判断roi是不是前景roi,而rcnn_cls_labels是roi与其匹配gt的iou同CLS_THRESH比较得到的真实roi类别(fg,bg,interval)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none') 
            cls_valid_mask = (rcnn_cls_labels >= 0).float() 
            # 计算fg以及interval的损失
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0) 
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7) [B, 100, 7]
            cls_preds: (BN, num_class) [B x 100, 1]
            box_preds: (BN, code_size) [B x 100, 7]

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1]) # [B, 100, 1]
        batch_box_preds = box_preds.view(batch_size, -1, code_size) # [B, 100, 7]

        roi_ry = rois[:, :, 6].view(-1) # [B x 100]
        roi_xyz = rois[:, :, 0:3].view(-1, 3) # [B x 100, 3]
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0 
        """
            tensor([[[0.0000, 0.0000, 0.0000,  ..., 1.6040, 1.5322, 5.9416],
                [0.0000, 0.0000, 0.0000,  ..., 1.6119, 1.5381, 5.9438],
                [0.0000, 0.0000, 0.0000,  ..., 1.6038, 1.5317, 5.8921],
                ...,
                [0.0000, 0.0000, 0.0000,  ..., 0.4999, 1.7195, 3.1942],
                [0.0000, 0.0000, 0.0000,  ..., 1.6545, 1.5191, 1.0755],
                [0.0000, 0.0000, 0.0000,  ..., 1.6627, 1.5394, 4.9616]],

                [[0.0000, 0.0000, 0.0000,  ..., 0.5753, 1.7427, 3.2033],
                [0.0000, 0.0000, 0.0000,  ..., 1.6103, 1.5353, 5.9536],
                [0.0000, 0.0000, 0.0000,  ..., 1.6165, 1.5405, 5.8682],
                ...,
                [0.0000, 0.0000, 0.0000,  ..., 1.5775, 1.4986, 3.0630],
                [0.0000, 0.0000, 0.0000,  ..., 0.5147, 1.7136, 3.1857],
                [0.0000, 0.0000, 0.0000,  ..., 1.6539, 1.5193, 1.0560]]],
            device='cuda:0')
        """
        
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size) # [B x 100, 7]

        # 微调后的box_preds的参数x,y,z是在roi坐标系下的,因此需要通过角度旋转到全局坐标系下
        batch_box_preds = common_utils.rotate_points_along_z( 
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1) # [B x 100, 7]

        # local_rois[:, :, 0:3] = 0 
        batch_box_preds[:, 0:3] += roi_xyz

        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
