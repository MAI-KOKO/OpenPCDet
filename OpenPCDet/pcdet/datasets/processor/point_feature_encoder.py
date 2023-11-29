import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list # 输入特征列表 ['x', 'y', 'z', 'intensity']
        self.src_feature_list = self.point_encoding_config.src_feature_list # 输出特征列表 ['x', 'y', 'z', 'intensity']
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        """
            getattr(object, name[, default])
                object: 表示一个对象，可以是一个实例、模块、类等
                name: 是一个字符串，表示对象中要获取的属性名称
                default(可选参数): 如果属性不存在，则返回默认值。如果不提供默认值，则在属性不存在时会引发 AttributeError 异常
        """
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x) # index() 方法会返回列表中第一个匹配元素 x 的索引。如果列表中不存在该元素，则会抛出 ValueError 异常
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True
