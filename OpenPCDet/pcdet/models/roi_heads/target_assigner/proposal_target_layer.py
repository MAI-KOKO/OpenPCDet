import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils

# 此处注释无特殊说明为PVRCNN

class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C) pre_box进行NMS处理得到rois
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        # 计算每个proposal对应最大iou的gt(max_overlaps),并根据前背景(fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH))、
        # 易难区分背景(hard_bg_thresh = (max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) & (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)\
        # easy_bg_thresh = (max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO))进行上采样  512 -> 128
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        ) # [B, 128, 7]proposal位置参数 [B, 128, 8]proposal对应gt的位置参数 [B, 128]proposal与对应gt的iou [B, 128]proposal对应类别的分数 [B, 128]proposal的类别
        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long() # [B, 128]

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh # [B, 128]
            bg_mask = batch_roi_ious < iou_bg_thresh # [B, 128]
            interval_mask = (fg_mask == 0) & (bg_mask == 0) # [B, 128]

            batch_cls_labels = (fg_mask > 0).float() # [B, 128] 前景区域为1 背景区域为0
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh) # 既不是前景也不是背景的区域通过公式计算为一个带小数的值
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels} # batch_cls_labels：前景、背景、间隔区域标志

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] # [B, 512, 7]
        roi_scores = batch_dict['roi_scores'] # [B, 512]
        roi_labels = batch_dict['roi_labels'] # [B, 512]
        gt_boxes = batch_dict['gt_boxes'] # [B, num_gt, 8] 8 = 7 + class

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size) # [B, 128, 7]
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1) # [B, 128, 8]
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) # [B, 128]
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt 

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class( # 计算每个类别的proposals对应gt的最大值iou以及gt索引
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                ) # [512]
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps) # 分类并上采样前背景区域 [ROI_PER_IMAGE]

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE)) # 每张图片前景iou数量
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1) # [num_max_overlaps >= fg_thresh] exmaple: [11] 返回大于前景阈值的proposal索引
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1) # 返回易分背景的proposal索引
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1) # 返回难分背景的proposal索引

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0: # 前背景数量大于0
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois) # 将前景数量与设定指标的最小值作为该张图片的前景数量

            # numpy.random.permutation(x) x: 要进行随机排列的数组或整数。如果是整数，将返回一个从 0 到 x-1 的随机排列
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]] # 随机取出fg_rois_per_this_image个前景区域

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0: # exam: hard_bg_inds: tensor([222, 356, 385, 420, 434, 461, 490, 493]]
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds)) 
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            """
                torch.randint(low, high, size, dtype=None, layout=torch.strided, device=None, requires_grad=False) 用于生成整数张量
                low: 整数，指定生成随机整数的下界（包含在内）
                high: 整数，指定生成随机整数的上界（不包含在内）
                size: 表示生成张量的形状，可以是整数或元组
            """
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long() # [hard_bg_rois_num] examp: tensor([2, 1, 4, 2, 3, 0, 6, 7])
            hard_bg_inds = hard_bg_inds[rand_idx] # exam: hard_bg_inds: tensor([385, 356, 434, 385, 420, 222, 490, 493] 可能会重复取得同一个区域

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7) [512, 7]
            roi_labels: (N) [512]
            gt_boxes: (N, ) [num_gt, 7]
            gt_labels: [num_gt]

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0]) # [512]
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0]) # [512]

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1): # 遍历每一个gt类别(1, 2, 3)
            roi_mask = (roi_labels == k) # [512] 取出对应类别的proposals
            gt_mask = (gt_labels == k) # [num_gt] 取出对应类别的gt
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask] # example: [numpro_classi, 7] i = 1, 2, 3
                cur_gt = gt_boxes[gt_mask] # example: [numgt_classi, 7]
                original_gt_assignment = gt_mask.nonzero().view(-1) # [numgt_classi]

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N) [numpro_classi, numgt_classi]
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1) # 返回每个anchor对应gt最大值iou以及对应gt索引 [numpro_classi]
                max_overlaps[roi_mask] = cur_max_overlaps # [512] 将值存储到对应类别proposals位置处
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
