import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        """
            torch.split() 函数是PyTorch中的函数,用于在张量沿着指定的维度进行分割。

            * anchors 是要进行分割的张量
            * 1 是指定分割的大小，表示每个小块（分割后的张量块）的大小为 1
            * dim=-1 是指定进行分割的维度，这里 -1 代表最后一个维度

            这行代码的作用是将张量 anchors 沿着最后一个维度（即 dim=-1,通常是张量的列维度）拆分成多个大小为 1 的张量块。它返回一个包含这些张量块的元组
        """
        # a = torch.split(anchors, 1, dim=-1) # (tenror_xa,tenror_ya,...,[tensor_cas1,tensor_cas2,...]) 注意cas经过解包后是个列表
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1) # anchor[:, 6] == 0 or 1.57 anchor[:, 7] == 0
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2) # 对角线
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)] # cgs:[] cas:[tensor([[0],[0],....[0]],device='cuda:0'])] => cts:[] 
        # *args 通常用于表示可变数量的位置参数。这意味着函数可以接受任意数量的位置参数，并将它们打包成一个元组
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        """
            torch.split(tensor, split_size_or_sections, dim=0)

            tensor: 要被分割的张量
            split_size_or_sections: 如果是一个整数 split_size,则表示每个子张量的尺寸；如果是一个列表 split_sections,则表示分割点的索引
            dim(可选): 沿着哪个维度进行分割，默认是 0
        """
        # 在 AnchorHeadTemplate 里 new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1) 
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            # torch.atan2(input1, input2) 的作用是返回一个新张量，其元素是通过将输入张量 input1 的元素作为y坐标，input2 的元素作为x坐标，计算反正切值得到的
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
