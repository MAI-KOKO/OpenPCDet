import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None): # 输入box_preds：[num_anchor_allcls, 7] 由于在model的generate_predicted_boxes里面进行了decoder
    src_box_scores = box_scores  # [num_anchor_allcls] 在后处理中经过torch.max(cls_preds, dim=-1)
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh) 
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        """
            torch.topk(input, k, dim=None, largest=True, sorted=True)

            * input 是输入的张量
            * k 表示要找到的最大或最小的 k 个值
            * dim 是沿着哪个维度进行计算。如果为 None,则在整个张量中计算
            * largest 是一个布尔值,指示是要找到最大值(largest=True)还是最小值(largest=False)
            * sorted 表示返回的结果是否按排序顺序。如果设置为 False,则返回结果是未排序的
        """
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0])) # 前k个分类分数
        boxes_for_nms = box_preds[indices] # 根据分类分数的排序索引取出对应box的位置参数
        """
            getattr() 是 Python 中的一个内建函数，用于获取对象的属性值或方法 Get Attributes

            getattr(object, name[, default])
                object 是要获取属性或方法的对象
                name 是属性或方法的名称
                default 是可选参数，表示如果指定的属性或方法不存在，返回的默认值

            具体来说,getattr() 返回对象 object 的属性值或方法
            如果指定的属性或方法不存在，会引发 AttributeError
            但如果提供了 default 参数，则返回 default 的值
        """
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        # keep_idx是selected_scores在box_scores_nms中的索引 
        # indices是box_scores_nms在box_scores中的索引
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        # original_idxs是box_scores在src_box_scores中的索引
        original_idxs = scores_mask.nonzero().view(-1) # nonzero() 函数返回一个张量，其中包含了输入张量中所有非零元素的索引
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
