# 导入Shapely核心几何对象：线、矩形框、多边形、线性环
from shapely.geometry import LineString, box, Polygon, LinearRing
# 导入Shapely几何基类，用于类型校验
from shapely.geometry.base import BaseGeometry
# 导入Shapely几何运算工具集
from shapely import ops
# 数值计算库
import numpy as np
# 导入SciPy距离计算工具
from scipy.spatial import distance
# 类型提示工具
from typing import List, Optional, Tuple
from numpy.typing import NDArray


'''
    utils.py 是纯几何工具，不依赖 nuScenes API，可复用到其他数据集的地图处理。
    utils.py 是地图真值提取的基础几何工具库，封装了 Shapely 几何对象的通用操作，包括几何集合拆分、可行驶区域轮廓提取、人行横道轮廓提取，为上层地图提取器提供底层几何计算能力。
'''

def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    '''
    将Multi类型的复合几何对象拆分为单个几何对象的列表，同时过滤掉无效、空的几何
    作用：统一处理Shapely运算后可能产生的MultiLineString/MultiPolygon，输出规整的单元素列表
    Args:
        geom (BaseGeometry): 待拆分的几何对象，支持线、多边形及其Multi复合类型
    Returns:
        List[Optional[BaseGeometry]]: 拆分后的单个几何对象列表，无效几何会被过滤
    '''
    # 1. 输入类型校验：仅支持线、多边形及其复合类型
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    
    # 2. Multi复合几何拆分
    if 'Multi' in geom.geom_type:
        outs = []
        # (1) 遍历复合对象中的所有子几何，逐个校验有效性
        for g in geom.geoms:
            # 只保留有效且非空的几何对象
            if g.is_valid and not g.is_empty:
                outs.append(g)
        # (2) 返回拆分后的单元素列表
        return outs
    
    # 3. 单几何对象处理
    else:
        # (1) 有效且非空则包装为单元素列表返回
        if geom.is_valid and not g.is_empty:
            return [geom,]
        # (2) 无效/空几何返回空列表
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon], 
                              roi_size: Tuple) -> List[LineString]:
    '''
    从可行驶区域多边形中提取边界轮廓线
    设计约定：外边界统一为顺时针方向（保证右手边是可行驶区域、左手边是非行驶区），内孔洞边界为逆时针
    Args:
        drivable_areas (List[Polygon]): 可行驶区域多边形列表
        roi_size (Tuple): BEV感知范围，格式为(x方向范围, y方向范围)
    Returns:
        List[LineString]: 裁剪后的边界线列表
    '''
    # 1. 构造ROI裁剪区域
    # (1) 计算ROI的x、y方向半长
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2
    # (2) 构造局部ROI矩形框，向内收缩0.2m，避免边缘产生异常边界
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    # 2. 提取所有边界环
    exteriors = []  # 存放所有外边界环
    interiors = []  # 存放所有内孔洞边界环
    
    # (1) 遍历每个可行驶区域多边形，分离外边界与内孔洞
    for poly in drivable_areas:
        # 提取多边形的外边界
        exteriors.append(poly.exterior)
        # 提取多边形的所有内孔洞边界
        for inter in poly.interiors:
            interiors.append(inter)
    
    # 3. 处理外边界：统一方向+ROI裁剪+合并
    results = []
    for ext in exteriors:
        # (1) 保证外边界为顺时针方向：右手边为可行驶区域
        if ext.is_ccw:
            # 如果是逆时针，反转坐标点顺序转为顺时针
            ext = LinearRing(list(ext.coords)[::-1])
        # (2) 用ROI框裁剪外边界，得到ROI范围内的线段
        lines = ext.intersection(local_patch)
        # (3) 若裁剪后是多段线，尝试合并为连续线
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        # (4) 校验结果类型合法性
        assert lines.geom_type in ['MultiLineString', 'LineString']
        # (5) 拆分后加入结果列表
        results.extend(split_collections(lines))
    
    # 4. 处理内孔洞边界：统一方向+ROI裁剪+合并
    for inter in interiors:
        # (1) 保证内边界为逆时针方向
        if not inter.is_ccw:
            # 如果是顺时针，反转转为逆时针
            inter = LinearRing(list(inter.coords)[::-1])
        # (2) 用ROI框裁剪内边界
        lines = inter.intersection(local_patch)
        # (3) 多段线合并
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        # (4) 类型校验
        assert lines.geom_type in ['MultiLineString', 'LineString']
        # (5) 拆分后加入结果
        results.extend(split_collections(lines))
    
    # 5. 返回所有边界线列表
    return results


def get_ped_crossing_contour(polygon: Polygon, 
                             local_patch: box) -> Optional[LineString]:
    '''
    从人行横多边形中提取闭合的轮廓线
    与可行驶区域轮廓不同：人行横道要求输出闭合折线，用于矢量地图建模
    Args:
        polygon (Polygon): 人行横道多边形
        local_patch (box): 局部ROI矩形框，用于裁剪
    Returns:
        Optional[LineString]: 裁剪后的闭合轮廓线，为空则返回None
    '''
    # 1. 提取外边界并统一方向
    ext = polygon.exterior
    # 保证边界环为逆时针方向
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    
    # 2. ROI裁剪边界环
    lines = ext.intersection(local_patch)
    
    # 3. 多段线后处理：非单条线则合并/拼接
    if lines.type != 'LineString':
        # (1) 过滤掉点类型的几何，只保留线段
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        # (2) 尝试自动合并多段线
        lines = ops.linemerge(lines)
        
        # (3) 合并后仍不连通则手动拼接坐标
        if lines.type != 'LineString':
            ls = []
            # 提取每段线的坐标数组
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            # 所有线段坐标拼接为一个连续数组
            lines = np.concatenate(ls, axis=0)
            # 转为LineString几何对象
            lines = LineString(lines)
    
    # 4. 有效性校验与返回
    if not lines.is_empty:
        return lines
    # 空结果返回None
    return None
