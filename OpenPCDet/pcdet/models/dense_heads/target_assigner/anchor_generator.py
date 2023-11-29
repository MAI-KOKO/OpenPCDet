import torch

# 以下注释无特殊说明为Second网络

class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range # [0., -40., -3., 70.4, 40., 1.]
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes): # grid_sizes = feature_map_size
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        # 对每个类别分别进行anchor生成
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip( # 这里的grid_size是feature_map_size
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height)) # 2
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else: # 计算特征图上的一个单位在源空间的尺寸 [0., -40., -3., 70.4, 40., 1.] [xmin, ymin, zmin, xmax, ymax, zmax]
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0
            """
                self.anchor_range[0] + x_offset 是起始值
                self.anchor_range[3] + 1e-5 是结束值。注意，加上 1e-5 是为了避免最后一个值超出结束范围
                step=x_stride 是两个连续值之间的差，表示步长
                dtype=torch.float32 表示生成的张量数据类型为 32 位浮点数

                这个函数的目的是创建一个从起始值到结束值的等差数列。生成的数列中的值从起始值开始，依照给定的步长逐渐递增，直到结束值
            """
            # 生成anchor在源空间处的中心点坐标值
            x_shifts = torch.arange( # x_shifts: tensor([ 0.0000,  0.4023,  0.8046,  1.2069,  ....], device='cuda:0')
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            # x_shifts.new_tensor(anchor_height) 创建了一个新的张量，内容与 anchor_height 类似，但与 x_shifts 具有相同的设备和数据类型
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__() 
            anchor_rotation = x_shifts.new_tensor(anchor_rotation) # (2)
            anchor_size = x_shifts.new_tensor(anchor_size) # (1, 3)
            """
                meshgrid(*xi, copy=True, sparse=False, indexing='xy')
                
                *xi: 一维数组列表,表示不同轴上的坐标值。这些数组将用于生成坐标网格。这些数组可以代表 x 轴坐标、y 轴坐标、z 轴坐标等。
                copy: 默认为 True,表示是否复制输入数组,设置为 False 会提高运算效率,但可能会对输入数组产生影响。
                sparse: 默认为 False,如果设置为 True,则返回稀疏网格；如果设置为 False,则返回密集网格。
                indexing: 默认为 'xy'，表示返回的坐标顺序。
                          'xy' 表示 ('X', 'Y')，即第一个参数对应 X 轴，第二个参数对应 Y 轴；
                          'ij' 表示 ('I', 'J')，即第一个参数对应行，第二个参数对应列。
            """
            # x_shifts: (nx_up) -> (nx_up, ny_up, nz) y_shifts: (ny_up) -> (nx_up, ny_up, nz) z_shifts: (nz) -> (nx_up, ny_up, nz)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([ 
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3] (nx_up, ny_up, nz, 3)
            # .repeat() 会按照指定的参数重复张量数据 参数 (1, 1, 1, anchor_size.shape[0], 1) 代表了各个维度上的重复次数
            # None 或者 np.newaxis 表示插入一个新的维度
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1) # (nx_up, ny_up, nz, num_anchor_size, 3)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1]) # (num_anchor_size, 3) -> (nx_up, ny_up, nz, num_anchor_size, 3)
            anchors = torch.cat((anchors, anchor_size), dim=-1) # (nx_up, ny_up, 1, num_anchor_size, 6)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1) # (nx_up, ny_up, nz, num_anchor_size, num_anchor_rotation, 6)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1]) # (num_anchor_rotation) -> (nx_up, ny_up, nz, num_anchor_size, num_anchor_rotation, 6)
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7] (nx_up, ny_up, nz, num_anchor_size, num_anchor_rotation, 7)

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous() # (nz, ny_up, nx_up, num_anchor_size, num_anchor_rotation, 7)
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location # num_anchors_per_location: [2, 2, 2]


if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
