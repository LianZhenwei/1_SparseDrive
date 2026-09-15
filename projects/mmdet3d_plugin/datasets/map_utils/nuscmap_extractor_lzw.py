'''
    nuscmap_extractor.py 是 nuScenes 数据集矢量地图真值提取器，是 SparseDrive 地图分支训练的数据核心。
    它基于 nuScenes 官方地图 API，根据每一帧自车的位姿，裁剪出 ROI 范围内的车道分隔线、人行横道、道路边界、可行驶区域四类矢量地图元素，作为地图任务的监督真值。

    nuscmap_extractor.py 是 nuScenes 专属的业务逻辑，对接官方地图 API，调用 utils.py 完成几何后处理，输出训练用的矢量地图真值：
                    nuscmap_extractor.py（上层业务：按类别提取地图真值）
                        ↓ 调用
                    utils.py（底层工具：通用几何计算、轮廓提取、集合拆分）

    nuscmap_extractor.py 属于数据预处理模块，运行在训练 / 推理的数据加载阶段：
        1. 数据加载时，根据每一帧的自车位姿，调用 get_map_geom 得到 ROI 内的矢量地图真值；
        2. 真值会进一步被采样、编码，转为和地图分支输出格式一致的监督标签；
        3. 训练时地图分支的 target.py 用这些真值做匈牙利匹配、计算损失。

    nuscmap_extractor.py 核心设计要点
        1. 四类地图要素：对应 SparseDrive 地图分支的预测类别，覆盖自动驾驶最核心的矢量地图元素：分隔线、人行横道、道路边界、可行驶区域；
        2. ROI 对齐：所有地图元素都裁剪到自车坐标系下的感知范围内，和检测、规划的空间范围保持一致；
        3. 实例级真值：通过拆分、合并操作，保证每个地图元素是独立完整的实例，适配 DETR 式的稀疏实例建模范式；
        4. 方向约定：边界线统一顺时针 / 逆时针方向，隐含了「哪一侧是可行驶区域」的语义，提升模型学习的稳定性；
        5. 空间索引加速：人行横道合并使用 STRtree 空间索引，避免 O (n²) 暴力匹配，提升数据加载速度。
'''

# 导入Shapely几何对象：线、矩形框、多边形
from shapely.geometry import LineString, box, Polygon
# 导入Shapely几何运算工具与空间索引树
from shapely import ops, strtree
# 数值计算库
import numpy as np
# 导入nuScenes官方地图API与地图探索器
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
# 导入四元数转航向角工具
from nuscenes.eval.common.utils import quaternion_yaw
# 导入四元数库
from pyquaternion import Quaternion

# 导入同目录下的工具函数
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
# 类型提示
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union


class NuscMapExtractor(object):
    """
    nuScenes数据集地图真值提取器
    核心功能：根据自车位姿，从全局矢量地图中裁剪出当前帧ROI范围内的各类地图元素，作为训练真值
    支持提取4类地图要素：车道分隔线、人行横道、道路边界、可行驶区域
    """

    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        '''
        初始化地图提取器，加载四个城区的全局地图
        Args:
            data_root (str): nuScenes数据集根目录路径
            roi_size (Union[List, Tuple]): BEV感知范围，格式为(x方向总长度, y方向总长度)
        '''
        # 保存ROI尺寸
        self.roi_size = roi_size
        # nuScenes包含的4个地图区域
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        # 存放每个区域的地图对象
        self.nusc_maps = {}
        # 存放每个区域的地图探索器（提供图层查询能力）
        self.map_explorer = {}
        # 遍历4个区域，逐个加载地图
        for loc in self.MAPS:
            # 加载该区域的全局矢量地图
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            # 创建对应地图探索器
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        # 构造自车坐标系下的ROI矩形框（nuScenes格式）
        # 以自车为中心，x前后、y左右各延伸一半范围
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)

    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        '''
        合并空间邻近且方向一致的人行横道碎块
        背景：nuScenes原始地图中，同一条人行横道可能被拆成多个小多边形，需要合并为完整实例
        Args:
            ped_geoms (List[Polygon]): 原始人行横道多边形列表
        Returns:
            List[Polygon]: 合并后的人行横道多边形列表
        '''
        def get_rec_direction(geom):
            '''内部工具：计算多边形最小外接矩形的长边方向向量与长度'''
            # 获取多边形的最小外接矩形
            rect = geom.minimum_rotated_rect
            # 取矩形前三个顶点坐标
            rect_v_p = np.array(rect.exterior.coords)[:3]
            # 计算两条邻边向量
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            # 计算两条边的长度
            v_len = np.linalg.norm(rect_v, axis=-1)
            # 找到长边的索引
            longest_v_i = v_len.argmax()
            # 返回长边方向向量和长边长度
            return rect_v[longest_v_i], v_len[longest_v_i]
        
        # 构建空间索引树，加速邻近多边形查询
        tree = strtree.STRtree(ped_geoms)
        # 建立「几何对象内存地址 → 列表索引」的映射
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))
        
        # 存放合并后的最终人行横道
        final_pgeom = []
        # 待处理的索引列表
        remain_idx = [i for i in range(len(ped_geoms))]
        
        # 遍历每个人行横道多边形，做合并
        for i, pgeom in enumerate(ped_geoms):
            # 已处理过则跳过
            if i not in remain_idx:
                continue
            
            # 从待处理列表移除当前索引
            remain_idx.pop(remain_idx.index(i))
            # 计算当前人行横道的长边方向
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            # 加入结果列表
            final_pgeom.append(pgeom)
            
            # 查询空间上与当前多边形相交的所有人行横道
            for o in tree.query(pgeom):
                # 获取相交多边形的索引
                o_idx = index_by_id[id(o)]
                # 已处理则跳过
                if o_idx not in remain_idx:
                    continue
                
                # 计算待合并人行横道的长边方向
                o_v, o_v_norm = get_rec_direction(o)
                # 计算两个方向向量的余弦值
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                # 方向夹角小于8度（余弦接近1），认为是同一条横道，合并
                if 1 - np.abs(cos) < 0.01:
                    # 多边形取并集，合并到结果中最后一个元素
                    final_pgeom[-1] = final_pgeom[-1].union(o)
                    # 从待处理列表移除
                    remain_idx.pop(remain_idx.index(o_idx))
        
        # 合并后可能产生MultiPolygon，拆分为单个多边形
        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results
        
    def get_map_geom(self, 
                     location: str, 
                     translation: Union[List, NDArray],
                     rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        '''
        核心方法：根据当前帧的位置与朝向，提取ROI范围内的所有地图几何元素
        Args:
            location (str): 所属城区名称，对应nuScenes的4个地图
            translation (Union[List, NDArray]): 自车全局平移坐标，形状(3,)，单位米
            rotation (Union[List, NDArray]): 自车全局旋转四元数，形状(4,)
        Returns:
            Dict[str, List[Union[LineString, Polygon]]]: 分类别的地图几何结果，包含4类要素
                - divider: 车道分隔线列表，LineString类型
                - ped_crossing: 人行横道线列表，LineString类型
                - boundary: 道路边界线列表，LineString类型
                - drivable_area: 可行驶区域列表，Polygon类型
        '''
        # nuScenes的patch_box格式：(中心x, 中心y, y方向长度, x方向长度)
        patch_box = (translation[0], translation[1], 
                self.roi_size[1], self.roi_size[0])
        # 四元数转航向角（弧度），再转为角度制
        rotation = Quaternion(rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # ========== 1. 提取车道分隔线 ==========
        # 提取车道分隔线图层
        lane_dividers = self.map_explorer[location]._get_layer_line(patch_box, yaw, 'lane_divider')
        # 提取道路分隔线图层
        road_dividers = self.map_explorer[location]._get_layer_line(patch_box, yaw, 'road_divider')
        
        # 合并两类分隔线，拆分复合几何为单条线
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # ========== 2. 提取人行横道 ==========
        ped_crossings = []
        # 提取人行横道路层（多边形格式）
        ped = self.map_explorer[location]._get_layer_polygon(patch_box, yaw, 'ped_crossing')
        
        # 拆分复合多边形
        for p in ped:
            ped_crossings += split_collections(p)
        
        # 合并邻近的人行横道碎块
        ped_crossings = self._union_ped(ped_crossings)
        
        # 提取人行横道的闭合轮廓线
        ped_crossing_lines = []
        for p in ped_crossings:
            # 裁剪ROI范围内的轮廓
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # ========== 3. 提取可行驶区域与道路边界 ==========
        # 取路段+车道的并集作为可行驶区域（官方drivable_area定义模糊，自行组合更准确）
        road_segments = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
        lanes = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
        
        # 所有路段取并集
        union_roads = ops.unary_union(road_segments)
        # 所有车道取并集
        union_lanes = ops.unary_union(lanes)
        # 路段与车道取并集，得到最终可行驶区域
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        
        # 拆分复合多边形
        drivable_areas = split_collections(drivable_areas)
        
        # 从可行驶区域提取边界线
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # 返回分类别的地图结果
        return dict(
            divider=all_dividers,          # 车道分隔线
            ped_crossing=ped_crossing_lines,# 人行横道线
            boundary=boundaries,           # 道路边界线
            drivable_area=drivable_areas,  # 可行驶区域多边形
        )
