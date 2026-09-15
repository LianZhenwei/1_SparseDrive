# 类型提示工具
from typing import List, Tuple, Union, Dict
# 数值计算库
import numpy as np
# 导入Shapely线几何对象
from shapely.geometry import LineString
from numpy.typing import NDArray
# MMCV数据容器
from mmcv.parallel import DataContainer as DC
# 流水线注册器
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(force=True)
class VectorizeMap(object):
    """
    地图矢量化处理器
    核心作用：将Shapely几何对象转换为采样点数组格式的矢量地图真值
    支持三种采样模式：固定点数采样、固定距离采样、道格拉斯普克简化
    可选归一化、排列增强，适配DETR式的稀疏建模范式
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 normalize: bool,
                 coords_dim: int=2,
                 simplify: bool=False, 
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
                 permute: bool=False
        ):
        """
        初始化矢量化处理器
        Args:
            roi_size (tuple/list): BEV感知范围，(x总长度, y总长度)
            normalize (bool): 是否将坐标归一化到(0,1)范围
            coords_dim (int): 坐标点的维度，默认2维平面坐标
            simplify (bool): 是否启用道格拉斯-普克算法简化折线
            sample_num (int): 固定点数采样的点数量，-1表示不启用
            sample_dist (float): 固定间隔采样的距离，-1表示不启用
            permute (bool): 是否启用折线起点排列增强，用于数据增广
        """
        # 1. 基础参数保存
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.roi_size = np.array(roi_size)
        self.normalize = normalize
        self.simplify = simplify
        self.permute = permute

        # 2. 根据配置选择采样函数，三选一
        if sample_dist > 0:
            # 固定距离采样模式
            assert sample_num < 0 and not simplify
            self.sample_fn = self.interp_fixed_dist
        elif sample_num > 0:
            # 固定点数采样模式
            assert sample_dist < 0 and not simplify
            self.sample_fn = self.interp_fixed_num
        else:
            # 折线简化模式
            assert simplify

    def interp_fixed_num(self, line: LineString) -> NDArray:
        '''
        固定点数采样：将一条折线插值为固定数量的点
        Args:
            line (LineString): 输入折线几何
        Returns:
            points (NDArray): 采样点坐标，形状 (N, 2)
        '''
        # 1. 生成均匀分布的距离序列，从0到线总长，共sample_num个点
        distances = np.linspace(0, line.length, self.sample_num)
        # 2. 按距离插值采样每个点的坐标
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        return sampled_points

    def interp_fixed_dist(self, line: LineString) -> NDArray:
        '''
        固定间隔采样：按固定物理距离沿折线采样点
        Args:
            line (LineString): 输入折线几何
        Returns:
            points (NDArray): 采样点坐标，形状 (N, 2)
        '''
        # 1. 生成等间隔的距离序列，从sample_dist开始，步长为sample_dist
        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # 2. 首尾补充起点和终点，保证至少两个点
        distances = [0,] + distances + [line.length,] 
        
        # 3. 按距离插值采样
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, map_geoms: Dict) -> Dict:
        '''
        批量矢量化：对所有类别的地图元素执行矢量化处理
        Args:
            map_geoms (Dict): 分类别的地图几何字典，值为Shapely几何列表
        Returns:
            vectors (Dict): 分类别的矢量化结果字典，值为点数组列表
        '''
        vectors = {}
        # 1. 遍历每个地图类别
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            # 2. 遍历该类别每个几何实例
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    # (1) 简化模式：道格拉斯普克算法抽稀折线
                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    else:
                        # (2) 采样模式：调用对应采样函数
                        line = self.sample_fn(geom)
                    
                    # (3) 截取前coords_dim维坐标
                    line = line[:, :self.coords_dim]

                    # (4) 坐标归一化到(0,1)
                    if self.normalize:
                        line = self.normalize_line(line)
                    
                    # (5) 排列增强：生成多起点排列
                    if self.permute:
                        line = self.permute_line(line)
                    
                    # (6) 加入结果列表
                    vectors[label].append(line)

                elif geom.geom_type == 'Polygon':
                    # 多边形不做矢量化，直接跳过
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors
    
    def normalize_line(self, line: NDArray) -> NDArray:
        '''
        坐标归一化：将物理坐标转换为0~1的归一化坐标
        Args:
            line (NDArray): 原始物理坐标点
        Returns:
            normalized (NDArray): 归一化后的点坐标
        '''
        # 1. 计算ROI左下角原点坐标
        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])
        # 2. 平移到以ROI左下角为原点
        line[:, :2] = line[:, :2] - origin
        # 3. 除以ROI总尺寸，归一化到0~1，加eps避免除零
        eps = 1e-5
        line[:, :2] = line[:, :2] / (self.roi_size + eps)
        return line
    
    def permute_line(self, line: np.ndarray, padding=1e5):
        '''
        折线排列增强：生成不同起点的折线排列，解决DETR匹配时起点不一致的问题
        闭合线生成2*(N-1)种排列，非闭合线生成正反两种
        Args:
            line (np.ndarray): 输入折线点，形状 (num_pts, 2)
            padding (float): 填充值，用于对齐长度
        Returns:
            permute_lines_array: 排列后的折线集合，形状 (num_permute, num_pts, 2)
        '''
        # 1. 判断是否为闭合折线（首尾点重合）
        is_closed = np.allclose(line[0], line[-1], atol=1e-3)
        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []

        # 2. 闭合线处理：循环移位+翻转，生成多种起点排列
        if is_closed:
            # (1) 去掉重复的首尾点
            pts_to_permute = line[:-1, :]
            # (2) 正向循环移位，生成permute_num种起点
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(pts_to_permute, shift_i, axis=0))
            # (3) 翻转后再循环移位，生成反向排列
            flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(flip_pts_to_permute, shift_i, axis=0))
        
        # 3. 非闭合线处理：仅正向和反向两种排列
        else:
            permute_lines_list.append(line)
            permute_lines_list.append(np.flip(line, axis=0))

        # 4. 堆叠为数组
        permute_lines_array = np.stack(permute_lines_list, axis=0)

        # 5. 闭合线补回首尾重复点
        if is_closed:
            tmp = np.zeros((permute_num * 2, num_points, self.coords_dim))
            tmp[:, :-1, :] = permute_lines_array
            tmp[:, -1, :] = permute_lines_array[:, 0, :] # 首尾闭合
            permute_lines_array = tmp
        else:
            # 非闭合线用填充值对齐到相同数量的排列
            padding = np.full([permute_num * 2 - 2, num_points, self.coords_dim], padding)
            permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array
    
    def __call__(self, input_dict):
        """
        流水线执行入口：生成矢量化地图真值并写入数据字典
        Args:
            input_dict (dict): 数据字典，需包含map_geoms字段
        Returns:
            dict: 更新后的数据字典
        """
        # 1. 无地图几何则直接返回
        if "map_geoms" not in input_dict:
            return input_dict
        
        # 2. 取出原始地图几何
        map_geoms = input_dict['map_geoms']
        # 3. 执行矢量化
        vectors = self.get_vectorized_lines(map_geoms)

        # 4. 排列增强模式：整理为标签+点的训练格式
        if self.permute:
            gt_map_labels, gt_map_pts = [], []
            for label, vecs in vectors.items():
                for vec in vecs:
                    gt_map_labels.append(label)
                    gt_map_pts.append(vec)
            # 转换为数组格式
            input_dict['gt_map_labels'] = np.array(gt_map_labels, dtype=np.int64)
            input_dict['gt_map_pts'] = np.array(gt_map_pts, dtype=np.float32).reshape(-1, 2 * (self.sample_num - 1), self.sample_num, self.coords_dim)
        
        # 5. 普通模式：直接封装矢量结果
        else:
            input_dict['vectors'] = DC(vectors, stack=False, cpu_only=True)
        
        return input_dict

    def __repr__(self):
        """返回类的字符串表示"""
        repr_str = self.__class__.__name__
        repr_str += f'(simplify={self.simplify}, '
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), ' 
        repr_str += f'roi_size={self.roi_size})'
        repr_str += f'normalize={self.normalize})'
        repr_str += f'coords_dim={self.coords_dim})'
        return repr_str
