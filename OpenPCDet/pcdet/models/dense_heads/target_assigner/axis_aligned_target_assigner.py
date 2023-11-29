import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names) # array(['Car', 'Pedestrian', 'Cyclist']
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None # None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...] (1, ny_up, nx_up, 1, 2, 7) 第一个1: z轴坐标只有一个 第二个1: num_anchor_size
            gt_boxes: (B, M, 8) M: num_gt_max 由于在初始化一个batch的帧数据时,将每帧数据的gt个数设为拥有最多gt帧的个数,因此每帧的gt个数并非真实个数,含有补0的gt
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1] # (B, M)
        gt_boxes = gt_boxes_with_classes[:, :, :-1] # (B, M, 7)
        for k in range(batch_size):
            cur_gt = gt_boxes[k] # (M, 7)
            cnt = cur_gt.__len__() - 1 # M - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0: # 删除空的gt
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int() # (M) 

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    # 获取gt类别(1, 2, 3)对应的名称('Car', 'Pedestrian', 'Cyclist'),将gt实类别为car的mask设为True 
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name) # (M) 
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3] # (nz, ny_up, nx_up)
                    anchors = anchors.view(-1, anchors.shape[-1]) # (nz, ny_up, nx_up, num_anchor_size, num_anchor_rotation, 8) -> (num_anchor, 8)
                    selected_classes = cur_gt_classes[mask] # (M_select) 获取mask为True的类别

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list], # view: [num_achor] => [nz, ny_up, nx_up, num_anchor_size * num_anchor_rotation]
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list], # (num_anchor, 8) => (nz, ny_up, nx_up, num_anchor_size * num_anchor_rotation, code_size)
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list] # view: [num_achor] => [nz, ny_up, nx_up, num_anchor_size * num_anchor_rotation]
                }
                # 合并三个类别的target信息
                # cat之后: (nz, ny_up, nx_up, num_anchor_size * num_anchor_rotation * 3, code_size) 
                # view之后: (nz * ny_up * nx_up * num_anchor_size * num_anchor_rotation * 3, code_size) 即(num_anchor_allcls, codesize)
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size) # 在box_coder_utils.py里面 if self.encode_angle_by_sincos: self.code_size += 1

                # a = torch.cat(target_dict['box_cls_labels'], dim=-1)
                # cat之后：(nz * ny_up * nx_up * num_anchor_size * num_anchor_rotation * 6)
                # view之后：(nz * ny_up * nx_up * num_anchor_size * num_anchor_rotation * 3) 即(num_anchor_allcls)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1) # (num_anchor_allcls)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        # 将一个batch的target合并
        bbox_targets = torch.stack(bbox_targets, dim=0) # [B, num_anchor_allcls, 8]

        cls_labels = torch.stack(cls_labels, dim=0) # [B, num_anchor_allcls]
        reg_weights = torch.stack(reg_weights, dim=0) # [B, num_anchor_allcls]
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):

        num_anchors = anchors.shape[0] 
        num_gt = gt_boxes.shape[0] # gt_boxes = cur_gt[mask]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 # 初始化为-1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7]) # (num_anchors, M_select)

            # argmax(axis=1)：沿着指定轴（axis）找出最大值"索引" # 找到iou最大的gt索引
            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda() # (num_anchors)
            # [torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]:     
            # 索引操作,第一个索引用于选择 anchor_by_gt_overlap 张量的行，第二个索引用于选择列
            anchor_to_gt_max = anchor_by_gt_overlap[ 
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ] # (num_anchors)

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda() # (M_select)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] # (M_select)
            empty_gt_mask = gt_to_anchor_max == 0 # 标志没有匹配到anchor的gt,并将这些gt的iou赋值为-1
            gt_to_anchor_max[empty_gt_mask] = -1 
            """
                * (anchor_by_gt_overlap == gt_to_anchor_max):
                        比较两个张量，创建一个布尔掩码，判断两个张量中相应位置上的值是否相等,满足条件的位置为 True
                * .nonzero(): 用于找出满足条件(即为 True)的位置
                        返回的索引张量是一个二维张量，其中每行表示一个满足条件的位置，包含两列：第一列是行索引，第二列是列索引
            """
            # 返回gt匹配到的anchor在anchor_by_gt_overlap中的索引 
            # a = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero() # (M_select, 2)
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] # (M_select) 选取返回位置的行索引(anchor索引)
            # gt匹配到的achor(一共有M_select个)不用进行阈值判断,直接保存
            # ground-truth indices force" 指的是强制性地选择（或指定）一组地面真实数据（ground truth data）的索引或位置
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # (M_select) 根据anchor的索引值获取anchor匹配到的gt索引
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 获取匹配到的gt类别,将其作为anchor的类别
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() # 储存anchor匹配到的gt索引

            pos_inds = anchor_to_gt_max >= matched_threshold # [num_anchors] 判断正样本
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] 
            labels[pos_inds] = gt_classes[gt_inds_over_thresh] # 将正样本保存 [num_anchors]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            """
                对于一维张量,nonzero() 返回的结果是一个二维张量，形状为 (N, 1)，其中 N 表示非零元素的数量。

                第一个维度 (N) 表示非零元素的数量，即返回的索引数量
                第二个维度 (1) 是因为在一维张量中每个索引只需要一个位置来表示
            """
            # a = (anchor_to_gt_max < unmatched_threshold).nonzero() # [num_nonzero_bg, 1] num_nonzero判断结果为True的数量
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] # [num_nonzero_bg] # 判断背景
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0] # 判断前景 [num_nonzero_fg]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0 # 将背景索引处的anchor的labels设为0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        # 创建一个形状为 (num_anchors, self.box_coder.code_size) 的新张量，该张量中的所有元素值均为零 
        # anchors 是一个张量，这里使用了它的属性 new_zeros()
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size)) # (num_anchors, 7) code_size：7
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :] # 前景anchor索引 => gt索引 => gt
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors) # (num_anchors, 8) 8：rts = [rt_cos, rt_sin]

        reg_weights = anchors.new_zeros((num_anchors,)) # (num_anchors)

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
