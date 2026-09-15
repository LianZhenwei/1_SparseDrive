
import os                             # 导入 os 模块，用于创建目录、获取当前工作路径、判断路径等
import math                           # 导入 math 模块，用于 atan2、asin 等数学函数
import copy                           # 导入 copy 模块，当前文件里 copy 实际没有使用，属于冗余导入
import argparse                       # 导入 argparse，用于解析命令行参数，例如 --root-path、--version 等
from os import path as osp            # 从 os 中导入 path 并命名为 osp，常用于 osp.join 拼接路径
from collections import OrderedDict   # 导入 OrderedDict。当前文件中 OrderedDict 实际没有使用，属于冗余导入
from typing import List, Tuple, Union # 导入类型标注。当前文件中 List、Tuple、Union 实际基本没有使用，属于冗余导入
import numpy as np                    # 导入 numpy，用于矩阵运算、数组拼接、坐标变换、轨迹处理等

# 导入 Quaternion
# nuScenes 中位姿和传感器标定大量使用四元数
from pyquaternion import Quaternion

# 从 shapely 中导入几何对象
# 当前文件中 MultiPoint、box 实际没有直接使用，可能是历史代码残留
from shapely.geometry import MultiPoint, box

# 导入 mmcv
# 用于 dump pkl、检查文件是否存在、进度条等
import mmcv

# 导入 nuScenes 官方 Python API
from nuscenes.nuscenes import NuScenes

# 导入 nuScenes CAN bus API
# 用于读取自车状态，例如速度、加速度、转角等
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# 导入 transform_matrix
# 用于根据平移和四元数构造 4x4 齐次变换矩阵
from nuscenes.utils.geometry_utils import transform_matrix

# 导入 Box 数据结构
# 当前文件中 Box 没有直接使用，boxes 是从 nusc.get_sample_data 返回的
from nuscenes.utils.data_classes import Box

# 导入 view_points
# 当前文件中 view_points 实际没有使用
from nuscenes.utils.geometry_utils import view_points

# 导入 nuScenes prediction 工具
# PredictHelper 用于读取 agent 未来轨迹
# convert_local_coords_to_global 用于把 agent 局部坐标轨迹转到全局坐标
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global

# 导入 SparseDrive 自定义地图提取器
# 用于从 nuScenes map 中提取局部 ROI 内的地图元素
from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor


# 将 nuScenes 原始细分类名映射到 SparseDrive 训练类别名的映射字典【nuScenes 原始类别更细，例如 vehicle.bus.bendy 和 vehicle.bus.rigid，这里统一映射到 10 类 detection 类别】
NameMapping = {
    "movable_object.barrier": "barrier",                  # 可移动障碍物 barrier 映射成 barrier
    "vehicle.bicycle": "bicycle",                         # 自行车
    "vehicle.bus.bendy": "bus",                           # 弯节公交车映射成 bus
    "vehicle.bus.rigid": "bus",                           # 普通公交车映射成 bus
    "vehicle.car": "car",                                 # 小汽车
    "vehicle.construction": "construction_vehicle",       # 工程车
    "vehicle.motorcycle": "motorcycle",                   # 摩托车
    "human.pedestrian.adult": "pedestrian",               # 成年行人
    "human.pedestrian.child": "pedestrian",               # 儿童行人
    "human.pedestrian.construction_worker": "pedestrian", # 施工人员
    "human.pedestrian.police_officer": "pedestrian",      # 警察
    "movable_object.trafficcone": "traffic_cone",         # 交通锥
    "vehicle.trailer": "trailer",                         # 拖车
    "vehicle.truck": "truck",                             # 卡车
}


# 将四元数转换成 roll、pitch、yaw：
def quart_to_rpy(qua): # 注意函数名 quart_to_rpy 里 quart 可能是 quaternion 的简写，该函数在当前文件中没有被调用，属于工具函数或历史残留
    x, y, z, w = qua                                                # 解包四元数，这里假设输入顺序是 x, y, z, w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) # 根据四元数计算 roll
    pitch = math.asin(2 * (w * y - x * z))                          # 根据四元数计算 pitch
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))  # 根据四元数计算 yaw
    return roll, pitch, yaw                                         # 返回欧拉角


# 根据目标时间戳 utime，在一组 CAN bus 消息时间戳 utimes 中找最近的一条【用于把 sample timestamp 和 CAN bus pose/steer 消息对齐】：
def locate_message(utimes, utime):
    # (1) np.searchsorted 返回 utime 应该插入到 utimes 的位置
    i = np.searchsorted(utimes, utime)

    # (2) 如果 i 到了末尾，或者前一个时间戳比后一个时间戳更接近 utime，就选择前一个消息
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1 # 就选择前一个消息

    # (3) 返回最近消息索引
    return i


# 将 NuscMapExtractor 提取出来的 shapely 几何对象转换成训练用 annotation【输出格式是 vectors[label] = [line1, line2, ...]，每条 line 是若干二维点坐标组成的 numpy array】：
def geom2anno(map_geoms):
    # (1) 定义 SparseDrive 使用的地图类别【虽然 Map Expansion 有很多语义图层，但 SparseDrive 最终在线地图任务只预测以下三类】
    MAP_CLASSES = (
        'ped_crossing', # 人行横道
        'divider',      # 分隔线
        'boundary',     # 边界线
    )

    # (2) 构建地图 annotation 字典
    # (2.1) 初始化地图 annotation 字典
    vectors = {}

    # (2.2) 遍历地图几何对象，填充地图 annotation 字典
    for cls, geom_list in map_geoms.items():
        '''
            map_geoms 形式大致是：
                    {
                        'divider': [LineString, LineString, ...],      # 每个元素 LineString 是一条二维折线的人行横道，每条人行横道是由若干 (x,y) 的点组成
                        'ped_crossing': [LineString, LineString, ...], # 每个元素 LineString 是一条二维折线的分隔线，每条分隔线是由若干 (x,y) 的点组成
                        'boundary': [LineString, LineString, ...]      # 每个元素 LineString 是一条二维折线的边界线，每条边界线是由若干 (x,y) 的点组成
                    }
        '''
        if cls in MAP_CLASSES: # 只处理 MAP_CLASSES 中定义的类别
            # 获取类别 id【即 “人行横道 ped_crossing -> 0”，“车道分隔线 divider -> 1”，“道路边界 boundary -> 2”】
            label = MAP_CLASSES.index(cls)

            # 初始化该类别的 vector list
            vectors[label] = []

            # 遍历该类别下每一个几何对象，将其加入 vectors 字典里
            for geom in geom_list:
                line = np.array(geom.coords) # geom.coords 是线上的坐标点，现转成 numpy array
                vectors[label].append(line)  # 加入 vectors
        '''
            填充后的 vectors 形式是: 
                        {
                            1: [array([num_points, 2]), array([num_points, 2]), ...], # 每个元素 array([num_points, 2]) 是一条二维折线的人行横道，每条人行横道是由若干 (x,y) 的点组成
                            0: [array([num_points, 2]), array([num_points, 2]), ...], # 每个元素 array([num_points, 2]) 是一条二维折线的分隔线，每条分隔线是由若干 (x,y) 的点组成
                            2: [array([num_points, 2]), array([num_points, 2]), ...]  # 每个元素 array([num_points, 2]) 是一条二维折线的边界线，每条边界线是由若干 (x,y) 的点组成
                        }
        '''

    # (2.3) 返回地图 annotation 字典
    return vectors # 输出格式是 vectors[label] = [line1, line2, ...]，每条 line 是若干二维点坐标组成的 numpy array


# 【总入口】创建 NuScenes API、地图提取器、CAN bus API，并保存 pkl：
def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          roi_size=(30, 60),):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    # 打印当前转换的数据版本和数据路径
    print(version, root_path)

    # 1. 创建 nuScenes 数据集对象，创建地图提取器，创建 CAN bus 读取对象
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True) # 创建 nuScenes 数据集对象：version 可以是 v1.0-trainval、v1.0-test、v1.0-mini
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)         # 创建地图提取器：用于根据当前帧位置提取 ROI 范围内的地图元素
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)          # 创建 CAN bus 读取对象：用于读取自车速度、加速度、转角等信息
    from nuscenes.utils import splits # 导入 nuScenes 官方划分

    # 2. 根据数据版本初始化 train scene 名称列表 train_scenes 和 val scene 名称列表 val_scenes：
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # 支持的数据版本为 'v1.0-trainval', 'v1.0-test', 'v1.0-mini'
    assert version in available_vers                             # 检查传入的数据版本是否合法
    if version == 'v1.0-trainval':            # (1) 如果是完整 trainval 版本
        train_scenes = splits.train           # 官方 train scene 名称列表
        val_scenes = splits.val               # 官方 val scene 名称列表
    elif version == 'v1.0-test':              # (2) 如果是 test 版本
        train_scenes = splits.test            # test scene 作为 train_scenes 变量使用，后面会统一写入 test info
        val_scenes = []                       # test 版本没有 val
    elif version == 'v1.0-mini':              # (3) 如果是 mini 版本
        train_scenes = splits.mini_train      # mini train scene
        val_scenes = splits.mini_val          # mini val scene
        out_path = osp.join(out_path, 'mini') # mini 数据输出到 out_path/mini 目录
    else:                                     # (4) 其他情况直接报错
        raise ValueError('unknown')

    # 创建输出目录
    os.makedirs(out_path, exist_ok=True)

    # 3. 过滤出本地有效 scene 文件的数据集，并转化为 scene token 集合
    # (1) 过滤出本地有效 scene 文件的数据集
    available_scenes = get_available_scenes(nusc)                                   # 过滤本地实际存在的 scene，因为有时候数据集没下载完整
    available_scene_names = [s['name'] for s in available_scenes]                   # 取出实际存在的 scene name
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes)) # 只保留 train_scenes 中本地存在的场景
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))     # 只保留 val_scenes 中本地存在的场景

    # (2) 将 train scene name 转换成 train scene token 集合
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])

    # (3) 将 val scene name 转换成 val scene token 集合
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes ])

    # (4) 打印 scene 数量
    test = 'test' in version                                                               # 判断是否是 test 版本
    if test:                                                                               # 如果是 test，
        print('test scene: {}'.format(len(train_scenes)))                                  # 则打印 test scene 数量；
    else:                                                                                  # 如果不是 test，
        print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes))) # 则打印 train 和 val scene 数量。

    # 4. 生成 train/val/test 的 info list
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc,                  # nuScenes API 对象
        nusc_map_extractor,    # 地图提取器
        nusc_can_bus,          # CAN bus API 对象
        train_scenes,          # train scene token 集合
        val_scenes,            # val scene token 集合
        test,                  # 是否 test 模式
        max_sweeps=max_sweeps  # sweeps 数量
    )

    # 5. 保存为 .pkl：
    # (1) metadata 保存数据版本信息
    metadata = dict(version=version)

    # (2) 保存 .pkl：
    # (2.a) 如果是 test 版本
    if test:
        print('test sample: {}'.format(len(train_nusc_infos))) # 打印 test sample 数量 
    
        # 保存 pkl
        data = dict(infos=train_nusc_infos, metadata=metadata)                  # 构造保存数据
        info_path = osp.join(out_path, '{}_infos_test.pkl'.format(info_prefix)) # 构建 test info 输出路径
        mmcv.dump(data, info_path)                                              # 保存 pkl
    
    # (2.b) 如果是 trainval 或 mini
    else:   
        print('train sample: {}, val sample: {}'.format(len(train_nusc_infos), len(val_nusc_infos))) # 打印 train/val sample 数量
        
        # 保存 train pkl
        data = dict(infos=train_nusc_infos, metadata=metadata)                   # 构造 train 保存数据
        info_path = osp.join(out_path, '{}_infos_train.pkl'.format(info_prefix)) # 构建 train info 输出路径
        mmcv.dump(data, info_path)                                               # 保存 train pkl

        # 保存 val pkl
        data['infos'] = val_nusc_infos                                             # 替换 infos 为 val
        info_val_path = osp.join(out_path, '{}_infos_val.pkl'.format(info_prefix)) # 构建 val info 输出路径
        mmcv.dump(data, info_val_path)                                             # 保存 val pkl
    '''
        最终得到的 .pkl 的内容为：
            {
                "infos": [info_0, info_1, info_2, ...], # 每个 info 就是一个训练帧的完整信息，即一个 info = 一个 nuScenes keyframe = SparseDrive 训练时的一帧样本
                "metadata": {"version": "v1.0-mini"}
            }
    
    '''


# 检查本地哪些 scene 文件真实存在：
def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """

    # 保存本地存在的 scene
    available_scenes = []

    # 打印 nuScenes 标注中总 scene 数
    print('total scene num: {}'.format(len(nusc.scene)))

    # 遍历每个 scene
    for scene in nusc.scene:
        # 获取 scene token
        scene_token = scene['token']

        # 根据 token 获取 scene 记录
        scene_rec = nusc.get('scene', scene_token)

        # 获取该 scene 的第一个 sample
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        # 获取第一个 sample 的 LIDAR_TOP sample_data
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

        # 是否还有更多帧
        has_more_frames = True

        # 标记该 scene 是否不存在
        scene_not_exist = False

        # 循环检查该 scene 至少第一帧 lidar 文件是否存在
        while has_more_frames:

            # 获取 lidar 文件路径、box 等
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])

            # 转成字符串路径
            lidar_path = str(lidar_path)

            # 如果当前工作目录出现在 lidar_path 中
            if os.getcwd() in lidar_path:

                # 有些数据集 API 返回绝对路径，这里转成相对路径
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]

                # relative path

            # 如果 lidar 文件路径不存在
            if not mmcv.is_filepath(lidar_path):

                # 标记该 scene 不存在
                scene_not_exist = True

                # 跳出 while
                break

            # 如果文件存在
            else:

                # 只要第一帧存在，就认为该 scene 可用
                break

        # 如果 scene 不存在
        if scene_not_exist:

            # 跳过该 scene
            continue

        # 加入可用 scene 列表
        available_scenes.append(scene)

    # 打印实际存在 scene 数量
    print('exist scene num: {}'.format(len(available_scenes)))

    # 返回可用 scene
    return available_scenes


# 【核心函数】逐 sample 生成 info 字典，包括相机、lidar、map、GT、轨迹、ego 状态：
def _fill_trainval_infos(nusc,
                         nusc_map_extractor,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=12,
                         ego_fut_ts=6):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    # 初始化
    train_nusc_infos = [] # 保存训练集 info
    val_nusc_infos = []   # 保存验证集 info
    cat2idx = {}          # 类别名到类别索引的映射，当前文件后续没有使用 cat2idx，属于冗余变量

    # 一、遍历 nuScenes 全部 category，建立 category name 到 index 的映射
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx # 建立 category name 到 index 的映射

    # 创建 prediction helper，用于提取 agent 未来轨迹
    predict_helper = PredictHelper(nusc)


    # 二、遍历 nuScenes 中每个 sample 帧【mmcv.track_iter_progress() 会显示进度条】
    for sample in mmcv.track_iter_progress(nusc.sample):
        # ================================ 1. 将当前帧的基础信息和 LIDAR_TOP frame → ego vehicle frame → global frame 的坐标变换（平移和旋转四元数）存入 info 字典 ================================ 
        # (1) 提取 lidar 相关信息
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location'] # 获取当前 sample 所在地图位置，例如 singapore-onenorth、boston-seaport 等
        lidar_token = sample['data']['LIDAR_TOP']                                                         # 获取当前 sample 的顶层激光雷达 token
        sd_rec = nusc.get('sample_data', lidar_token)                                                     # 获取 lidar sample_data 记录
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])                      # 获取 lidar 标定参数
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])                                      # 获取 ego pose
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)                                          # 获取 lidar 文件路径和 3D boxes
        mmcv.check_file_exist(lidar_path)                                                                 # 检查 lidar 文件是否存在

        # (2) 构造当前 sample 的 info 字典
        info = {
            'lidar_path': lidar_path,                             # lidar 文件路径【当前关键帧的顶置激光雷达文件 .pcd.bin 路径】
            'token': sample['token'],                             # 当前 sample token【nuScenes 中当前 sample 的唯一 ID】
            'sweeps': [],                                         # 历史 lidar sweeps 信息，后面会填充
            'cams': dict(),                                       # 多相机信息，后面会填充
            'scene_token': sample['scene_token'],                 # 所属 scene token【当前帧属于哪一个 scene】
            'lidar2ego_translation': cs_record['translation'],    # lidar 到 ego 的平移
            'lidar2ego_rotation': cs_record['rotation'],          # lidar 到 ego 的旋转四元数
            'ego2global_translation': pose_record['translation'], # ego 到 global 的平移
            'ego2global_rotation': pose_record['rotation'],       # ego 到 global 的旋转四元数
            'timestamp': sample['timestamp'],                     # 当前 sample 时间戳
            'map_location': map_location,                         # 当前地图位置
        }
        '''
            重点：LIDAR_TOP frame → ego vehicle frame → global frame 的坐标变换（平移和旋转四元数）：
                SparseDrive 处理 nuScenes 时，经常要在几个坐标系之间转换：
                        camera frame：原始图片投影用
                        lidar frame：3D框、BEV、预测轨迹常用
                        ego frame：自车状态、局部运动
                        global frame：nuScenes 地图、长时序轨迹、pose
                nuscenes_converter.py 里就是先拿当前 LiDAR 的 calibrated_sensor 和 ego_pose，再构造 lidar2global = ego2global @ lidar2ego，之后用它去取当前局部地图。
        '''

        # (3) 提取 info 字典中的旋转四元数和平移，构建坐标变换矩阵
        l2e_r = info['lidar2ego_rotation']            # 取 lidar 到 ego 的旋转四元数
        l2e_t = info['lidar2ego_translation']         # 取 lidar 到 ego 的平移
        e2g_r = info['ego2global_rotation']           # 取 ego 到 global 的旋转四元数
        e2g_t = info['ego2global_translation']        # 取 ego 到 global 的平移
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix # lidar 到 ego 的四元数转旋转矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix # ego 到 global 的旋转矩阵


        # ================================ 2. 将当前帧 ego 附近 ROI 局部地图的几何元素 GT 标签 annotation 存入 info['map_annos']【这里地图元素的 GT 标签其实就是二维点 (x,y)】================================ 
        # (1) 获取 lidar 到 ego 的 4x4 矩阵 lidar2ego
        lidar2ego = np.eye(4)                                                      # 初始化 lidar 到 ego 的 4x4 矩阵
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix # 填入旋转矩阵
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])                 # 填入平移

        # (2) 获取 ego 到 global 的 4x4 矩阵 ego2global
        ego2global = np.eye(4)                                                       # 初始化 ego 到 global 的 4x4 矩阵
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix # 填入旋转矩阵
        ego2global[:3, 3] = np.array(info["ego2global_translation"])                 # 填入平移

        # (3) 计算 lidar 到 global 的 4x4 矩阵 lidar2global
        lidar2global = ego2global @ lidar2ego              # lidar 到 global 的变换
        translation = list(lidar2global[:3, 3])            # 当前 lidar 在 global 下的平移
        rotation = list(Quaternion(matrix=lidar2global).q) # 当前 lidar 在 global 下的旋转四元数

        # (4) 将当前 ROI 内的局部地图几何元素保存为 info['map_annos']
        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation) # 从 nuScenes map 中提取当前 ROI 内的局部地图几何元素
        map_annos = geom2anno(map_geoms) # 将地图几何对象转换成训练 annotation
        info['map_annos'] = map_annos    # 保存地图 annotation


        # ================================ 3. 将当前帧 6 个相机的内外参信息存入 info['cams'] ================================ 
        # (1) nuScenes 6 个相机类型
        camera_types = [
            'CAM_FRONT',       # 前视相机
            'CAM_FRONT_RIGHT', # 前右相机
            'CAM_FRONT_LEFT',  # 前左相机
            'CAM_BACK',        # 后视相机
            'CAM_BACK_LEFT',   # 后左相机
            'CAM_BACK_RIGHT',  # 后右相机
        ]

        # (2) 遍历每个相机，提取每个相机的内外参，存到 info['cams'] 中
        for cam in camera_types:
            # (a) 获取该相机的 sample_data token
            cam_token = sample['data'][cam]

            # (b) 获取相机图片路径、box、相机内参
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)

            # (c) 获取该相机到 top lidar 的外参变换信息
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)

            # (d) 加入相机内参
            cam_info.update(cam_intrinsic=cam_intrinsic)

            # (e) 保存到 info['cams']
            info['cams'].update({cam: cam_info})
        '''
            最终得到的 info['cams'] = {"CAM_FRONT":{...}, 'CAM_FRONT_RIGHT':{...}, 'CAM_FRONT_LEFT':{...}, 'CAM_BACK':{...}, 'CAM_BACK_LEFT':{...}, 'CAM_BACK_RIGHT':{...}}
            即 info['cams'] 有6个元素，分别是当前帧的6个相机，而每个相机的内容为：
                    {
                        "data_path": 图片路径,
                        "type": 相机名,
                        "sample_data_token": 当前相机 sample_data token,
                        "sensor2ego_translation": 相机到自车的平移,
                        "sensor2ego_rotation": 相机到自车的旋转,
                        "ego2global_translation": 当前相机时刻 ego 到 global 的平移,
                        "ego2global_rotation": 当前相机时刻 ego 到 global 的旋转,
                        "timestamp": 相机时间戳,
                        "sensor2lidar_rotation": 相机坐标系到当前 LIDAR_TOP 坐标系的旋转,
                        "sensor2lidar_translation": 相机坐标系到当前 LIDAR_TOP 坐标系的平移,
                        "cam_intrinsic": 相机内参矩阵
                    }
        '''


        # ================================ 4. 将当前关键帧往前的历史 lidar sweep 帧信息存入 info['sweeps']，最多存 max_sweeps=10 ================================ 
        # (1) 重新获取当前 LIDAR_TOP sample_data
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        # (2) 初始化 sweeps 列表
        sweeps = []

        # (3) 最多收集 max_sweeps=10 个历史 sweep
        while len(sweeps) < max_sweeps:
            # 如果当前 lidar sample_data 有上一帧
            if not sd_rec['prev'] == '':
                # 获取上一帧 lidar 到当前 top lidar 的变换
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')

                # 加入 sweeps
                sweeps.append(sweep)

                # 继续往前找
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:     # 如果没有上一帧
                break # 则结束循环

        # (4) 保存 sweeps
        info['sweeps'] = sweeps
        '''
            当前帧点云：lidar_path
            历史帧点云：sweeps[0], sweeps[1], ..., sweeps[9]
        '''


        # ================================ 5. 如果不是 test 模式，就可以读取 GT annotation ================================ 
        if not test:
            # ---------------------------- (1) 根据 sample['anns'] 获取每个 sample_annotation ---------------------------- 
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]


            # ---------------------------- (2) 获取目标检测 annotation：box、速度、类别名、有效标记等 ----------------------------
            # (a) 提取 boxes 的中心坐标、尺寸、yaw角：
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)                        # 从 boxes 中取中心坐标【shape: [num_box(当前帧GT框数), 3(x,y,z)]】
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)                           # 从 boxes 中取尺寸 wlh【shape: [num_box(当前帧GT框数), 3(w,l,h)]，因为 nuScenes Box 默认是 width, length, height】
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) # 取 yaw 角【shape: [num_box(当前帧GT框数), 1(即yaw)]，因为 yaw_pitch_roll[0] 是 yaw】

            # (b) 将 box 速度从 global 坐标系转换到 lidar 坐标系
            # (b.1) 获取每个 annotation 的速度【nusc.box_velocity 返回 global 坐标系下速度，这里只取 vx, vy】
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            
            # (b.2) 遍历每个 box，将速度从 global 坐标系转换到 lidar 坐标系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])                                  # 把二维速度扩展成三维，z 速度设为 0
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T # global -> ego -> lidar 旋转
                velocity[i] = velo[:2]                                                # 保存 lidar 坐标系下 vx, vy

            # (c) 将类别名映射成 SparseDrive 使用的 10 类名称
            # (c.1) 获取原始类别名【shape: [num_box(当前帧GT框数), 1(NuScenes原始类别名)]】
            names = [b.name for b in boxes]

            # (c.2) 遍历类别名，将类别名映射成 SparseDrive 使用的 10 类名称【shape: [num_box(当前帧GT框数)(映射到SparseDrive的十个类别名之一)]】
            for i in range(len(names)):
                if names[i] in NameMapping:          # 如果类别名在映射表中
                    names[i] = NameMapping[names[i]] # 映射成 SparseDrive 使用的 10 类名称
            names = np.array(names) # 转成 numpy array

            # (d) 判断每个 GT box 是否有效，用于训练时过滤 GT【当前规则：lidar 点数 + radar 点数 > 0 才有效，即如果一个标注框里没有任何 LiDAR 点、也没有任何 Radar 点，就认为它不太可靠或者不可见】【shape: [num_box(当前帧GT框数)(大于0为True否则为False)]】
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)  ## TODO update valid flag for tracking

            # (e.1) 构造 GT boxes【shape: [num_box(当前帧GT框数), 7(x,y,z,l,w,h,yaw)]】：
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            '''
                locs: x, y, z
                dims[:, [1, 0, 2]]: 将 nuScenes w,l,h 转成 l,w,h
                rots: yaw
                最终 shape: [num_box, 7]
            '''

            # (e.2) 检查 box 数量和 annotations 数量一致
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
            

            # ---------------------------- (3) 获取目标跟踪 annotation：instance id ---------------------------- 
            # 将 instance_token 转成 nuScenes 内部整数 instance index【它是目标实例 ID，同一个实例（如同一辆车、同一个行人）在连续帧里会有同一个 instance_token，下行代码已转成整数】
            instance_inds = [nusc.getind('instance', anno['instance_token']) for anno in annotations]
            '''
                1. 先从 nuScenes 数据集的 sample_annotation.json 的当前 3D 框标注中取得 instance_token【在 nuScenes 数据集中类似于 "abc..."】；
                2. 再查找这个 token 在 nuScenes instance 表中的整数下标；
                3. 将字符串 token 转换为整数。
            '''

            # ---------------------------- (4) 获取运动预测 annotation：其他 agent 的未来轨迹 offset 和有效 mask ---------------------------- 
            # (a) 获取 box 数量
            num_box = len(boxes)

            # (b) 初始化 agent 未来轨迹偏移 offset，初始化 agent 未来轨迹有效 mask
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2)) # 初始化 agent 未来轨迹偏移 offset【shape: [num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒), 2(x,y)(第0步是相对当前目标位置的偏移，后面每一步是相邻未来点之间的增量)]，其中 fut_ts 默认 12，因为 nuScenes 2Hz，所以对应未来 6 秒】
            gt_fut_masks = np.zeros((num_box, fut_ts))    # 初始化 agent 未来轨迹有效 mask【shape: [num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒)(值为1或0，如果某个目标未来没那么长，比如scene结束了，后面的mask就是0)]】

            # (c) 遍历每个 annotation，填充 agent 未来轨迹偏移 offset、填充 agent 未来轨迹有效 mask：
            for i, anno in enumerate(annotations):
                # (c.1) 获取该目标的 instance token
                instance_token = anno['instance_token']

                # (c.2) 获取该目标未来轨迹
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token, 
                    sample['token'], 
                    seconds=fut_ts/2,   # seconds=fut_ts/2，因为 nuScenes prediction 通常 2Hz，12 步对应 6 秒，也就是未来 6 秒 的 agent 轨迹
                    in_agent_frame=True # in_agent_frame=True 表示先返回 agent 自身局部坐标系下的未来轨迹
                )

                # (c.3) 如果未来轨迹非空，则填充 agent 未来轨迹、填充 agent 未来轨迹有效 mask
                if fut_traj_local.shape[0] > 0:
                    # 当前目标 box、当前目标中心、当前目标旋转
                    box = boxes[i]                               # 当前目标 box
                    trans = box.center                           # 当前目标中心
                    rot = Quaternion(matrix=box.rotation_matrix) # 当前目标旋转

                    # 将 agent 局部坐标系未来轨迹转回全局/场景坐标
                    fut_traj_scene = convert_local_coords_to_global(fut_traj_local, trans, rot)

                    # 有效未来步数
                    valid_step = fut_traj_scene.shape[0]

                    # 计算未来点与当前中心的偏移 offset【shape: [num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒), 2(x,y)(第0步是相对当前目标位置的偏移，后面每一步是相邻未来点之间的增量)]】
                    gt_fut_trajs[i, 0] = fut_traj_scene[0] - box.center[:2]                  # 第 0 步 offset = 第一未来点 - 当前 box 中心
                    gt_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1] # 后续 offset = 当前未来点 - 上一个未来点【也就是轨迹增量，而不是绝对坐标】

                    # 标记有效未来步【shape: [num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒)(有未来轨迹的位置为1，如果某个目标未来没那么长，比如scene结束了，后面的mask就是0)]】
                    gt_fut_masks[i, :valid_step] = 1


            # ---------------------------- (5) 获取自车规划 annotation ---------------------------- 
            # ---------------------------- (5.1) 获取当前帧自车状态、自车未来轨迹有效 mask ---------------------------- 
            # (a) 初始化自车当前时刻及未来轨迹，初始化自车当前时刻及未来轨迹有效 mask
            ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3)) # 初始化自车当前时刻及未来轨迹【shape: [ego_fut_ts+1=7, 3]，其中 ego_fut_ts + 1 是因为要包含当前点，然后再做差得到 ego_fut_ts 个 offset】【其中 ego_fut_ts 默认 6，因为 nuScenes 2Hz，所以对应自车未来 3 秒】
            ego_fut_masks = np.zeros((ego_fut_ts + 1))    # 初始化自车当前时刻及未来轨迹有效 mask【shape: [ego_fut_ts+1=7]】

            # (b) 当前 sample
            sample_cur = sample

            # (c) 读取当前帧自车状态，shape: [10]【自车加速度(ax,ay,az)、角速度(wx,wy,wz)、速度(vx,vy,vz)、方向盘转角，共10维，如果 CAN bus 读取失败，代码会用 10 个 0 代替】
            ego_status = get_ego_status(nusc, nusc_can_bus, sample_cur)

            # (d) 读取当前帧及未来 ego_fut_ts 帧的 ego pose
            for i in range(ego_fut_ts + 1):
                # (d.1) 获取当前 sample 的 LIDAR_TOP 在 global 下的位姿
                pose_mat = get_global_sensor_pose(sample_cur, nusc)

                # (d.2) 保存平移部分
                ego_fut_trajs[i] = pose_mat[:3, 3]

                # (d.3) 标记该步有效
                ego_fut_masks[i] = 1

                # (d.4) 如果没有下一帧，后面未来位置全部填成当前最后位置
                if sample_cur['next'] == '':
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i] # 后面未来位置全部填成当前最后位置
                    break                                  # 跳出循环

                # (d.5) 如果有下一帧，则移动到下一个 sample
                else:
                    sample_cur = nusc.get('sample', sample_cur['next']) # 移动到下一个 sample

            # ---------------------------- (5.2) 将 ego_fut_trajs 从 global 坐标系转到当前 ego 坐标系 ----------------------------
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])  # (a) 先减去当前 ego 在 global 下的平移
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix # (b) 当前 ego rotation 的逆
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T                    # (c) 旋转到当前 ego 坐标系

            # ---------------------------- (5.3) 再从当前 ego 坐标系转到当前 lidar 坐标系 ----------------------------
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])  # (a) 减去 lidar 到 ego 的平移
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix # (b) lidar 到 ego rotation 的逆
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T                  # (c) 转到 lidar 坐标系

            # ---------------------------- (5.4) 根据未来最终位置偏移判断高层驾驶命令 ----------------------------
            # 注意 command 不是人工标的高层导航命令，而是 SparseDrive 根据未来自车轨迹自动构造出来的粗粒度高层驾驶命令
            if ego_fut_trajs[-1][0] >= 2:     # (a) 如果未来最终终点 x >= 2
                command = np.array([1, 0, 0]) # 认为是右转，command 是一个 one-hot 向量 # Turn Right 
            elif ego_fut_trajs[-1][0] <= -2:  # (b) 如果未来最终终点 x <= -2
                command = np.array([0, 1, 0]) # 认为是左转，command 是一个 one-hot 向量 # Turn Left 
            else:                             # (c) 否则（即 -2 < 未来最终终点 x < 2）
                command = np.array([0, 0, 1]) # 认为是直行，command 是一个 one-hot 向量 # Go Straight 

            # ---------------------------- (5.5) 计算自车未来轨迹偏移 offset：通过将绝对未来位置转换成相邻帧 offset 得到----------------------------
            # [p1-p0, p2-p1, ..., p6-p5]
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1] # 原本 ego_fut_trajs 是当前时刻及未来轨迹，现在将其变成相邻位移，即未来相邻时间步的位移，称为偏移 offset


            # ---------------------------- (6) 保存以上 annotation 到 info 字典【在 /home/lzw/SparseDrive/data/infos/nuscenes_infos_train_from_lzw.json 可看到以下各量的具体 shape】 ----------------------------
            info['gt_boxes'] = gt_boxes                                                 # 【用于检测任务】保存 3D 检测 GT boxes，[num_box(当前帧GT框数), 7(x,y,z,l,w,h,yaw)]
            info['gt_names'] = names                                                    # 【用于检测任务】保存 3D 检测 GT boxes 的类别名，[num_box(当前帧GT框数)(映射到SparseDrive的十个类别名之一)]
            info['gt_velocity'] = velocity.reshape(-1, 2)                               # 【用于检测任务】保存 3D 检测 GT boxes 的速度，[num_box(当前帧GT框数), 2(lidar坐标系下的GT框的vx,vy)]【注意速度已从 global 坐标系转换到 lidar 坐标系下】
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations]) # 【用于检测任务】保存每个 annotation 的 lidar 点数，[num_box(当前帧GT框数)]
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations]) # 【用于检测任务】保存每个 annotation 的 radar 点数，[num_box(当前帧GT框数)]
            info['valid_flag'] = valid_flag                                             # 【用于检测任务】保存每个 GT box 是否有效的有效标记【规则为：lidar 点数 + radar 点数 > 0 才有效】，[num_box(当前帧GT框数)(大于0为True否则为False)]
            info['instance_inds'] = instance_inds                                       # 【用于跟踪任务】保存 instance index，用于 tracking
            info['gt_agent_fut_trajs'] = gt_fut_trajs.astype(np.float32)                # 【用于预测任务】保存 agent 未来轨迹偏移 offset，[num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒), 2(x,y)(第0步是相对当前目标位置的偏移，后面每一步是相邻未来点之间的增量)]
            info['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)                # 【用于预测任务】保存 agent 未来轨迹有效 mask，[num_box(当前帧GT框数), fut_ts=12(他车未来时间步数,默认6秒)(有未来轨迹的位置为1，如果某个目标未来没那么长，比如scene结束了，后面的mask就是0)]
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)          # 【用于规划任务】保存 ego 未来轨迹偏移 offset【即未来相邻时间步的位移】，只取 x,y 两维，shape: [ego_fut_ts=6(自车未来时间步数,默认3秒), 2(x,y)]
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)             # 【用于规划任务】保存 ego 未来轨迹有效 mask：去掉第 0 个当前点，只保留未来 ego_fut_ts 步，shape: [ego_fut_ts=6(自车未来时间步数,默认3秒)(有效未来步为1)]
            info['gt_ego_fut_cmd'] = command.astype(np.float32)                         # 【用于规划任务】保存高层驾驶命令，shape: [3(注意是一个one-hot向量)]【右转、左转、直行的 one-hot 向量】
            info['ego_status'] = ego_status                                             # 【用于规划任务】保存自车状态，shape: [10]【自车加速度(ax,ay,az)、角速度(wx,wy,wz)、速度(vx,vy,vz)、方向盘转角，共10维】


        # ================================ 6. 将 info 追加进训练列表 train_nusc_infos 或验证列表 val_nusc_infos ================================ 
        # (1) 如果当前 sample 属于 train scene，则将 info 字典加入训练列表
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info) # 加入训练列表

        # (2) 否则加入 val 列表，则将 info 字典加入训验证列表
        else:
            # 注意 test 模式下 val_scenes 为空，所以 test sample 也会走这里吗
            # 但前面 test 时 train_scenes 实际是 test scene token
            # 所以 test sample 应该会进入 train_nusc_infos
            val_nusc_infos.append(info) # 加入验证列表

    # 三、返回 train info 和 val info
    return train_nusc_infos, val_nusc_infos # 重点：train_nusc_infos 是 info 的列表，每个 info 就是一个训练帧的完整信息，即一个 info = 一个 nuScenes keyframe = SparseDrive 训练时的一帧样本。val_nusc_infos 同理


# 从 CAN bus 中读取自车加速度(ax,ay,az)、角速度(wx,wy,wz)、速度(vx,vy,vz)、方向盘转角，共10维：
def get_ego_status(nusc, nusc_can_bus, sample):
    '''
        获取自车状态信息
        包括：
            acceleration 3 维
            rotation_rate 3 维
            velocity 3 维
            steering angle 1 维
        总共 10 维    
    '''

    # 初始化自车状态列表
    ego_status = []

    # 获取当前 sample 所属 scene
    ref_scene = nusc.get("scene", sample['scene_token'])

    # 尝试读取 CAN bus 信息
    try:
        # 获取该 scene 下 pose 消息
        pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')

        # 获取该 scene 下转角反馈消息
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')

        # 取 pose 消息时间戳列表
        pose_uts = [msg['utime'] for msg in pose_msgs]

        # 取 steer 消息时间戳列表
        steer_uts = [msg['utime'] for msg in steer_msgs]

        # 当前 sample 时间戳
        ref_utime = sample['timestamp']

        # 找到最接近当前 sample 时间戳的 pose 消息
        pose_index = locate_message(pose_uts, ref_utime)

        # 取对应 pose 数据
        pose_data = pose_msgs[pose_index]

        # 找到最接近当前 sample 时间戳的 steer 消息
        steer_index = locate_message(steer_uts, ref_utime)

        # 取对应 steer 数据
        steer_data = steer_msgs[steer_index]

        # 加入加速度
        ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s

        # 加入角速度
        ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s

        # 加入速度
        ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s

        # 加入方向盘转角
        ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn

    # 如果 CAN bus 读取失败
    except:
        ego_status = [0] * 10 # 如果 CAN bus 读取失败，代码会用 10 个 0 代替
    
    # 转成 float32 numpy array
    return np.array(ego_status).astype(np.float32)


# 获取当前 sample 的 LIDAR_TOP 传感器在 global 坐标系下的位姿：
def get_global_sensor_pose(rec, nusc):
    # 获取当前 sample 的 LIDAR_TOP sample_data
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])


    # 获取 ego pose
    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])

    # 获取 lidar calibrated sensor
    cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])


    # 构造 ego 到 global 的 4x4 变换
    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)

    # 构造 sensor 到 ego 的 4x4 变换
    sensor2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)

    # sensor 到 global 的变换
    pose = ego2global.dot(sensor2ego)


    # 返回 4x4 pose
    return pose


# 计算任意传感器到当前关键帧 Top LiDAR 坐标系的外参：
def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """

    # 根据 sensor token 获取 sample_data
    sd_rec = nusc.get('sample_data', sensor_token)

    # 获取该传感器标定参数
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

    # 获取该传感器对应时刻的 ego pose
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    # 获取该传感器数据路径
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))

    # 如果路径中包含当前工作目录
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        # 转成相对路径
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path

    # 构造 sweep 信息
    sweep = {
        'data_path': data_path,                               # 数据路径
        'type': sensor_type,                                  # 传感器类型，例如 CAM_FRONT 或 lidar
        'sample_data_token': sd_rec['token'],                 # sample_data token
        'sensor2ego_translation': cs_record['translation'],   # sensor 到 ego 的平移
        'sensor2ego_rotation': cs_record['rotation'],         # sensor 到 ego 的旋转
        'ego2global_translation': pose_record['translation'], # ego 到 global 的平移
        'ego2global_rotation': pose_record['rotation'],       # ego 到 global 的旋转
        'timestamp': sd_rec['timestamp']                      # 该传感器数据时间戳
    }

    # 当前 sensor 到 ego 的旋转四元数
    l2e_r_s = sweep['sensor2ego_rotation']

    # 当前 sensor 到 ego 的平移
    l2e_t_s = sweep['sensor2ego_translation']

    # 当前 sensor 对应 ego 到 global 的旋转四元数
    e2g_r_s = sweep['ego2global_rotation']

    # 当前 sensor 对应 ego 到 global 的平移
    e2g_t_s = sweep['ego2global_translation']


    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    # 求当前 sensor 坐标系到当前关键帧 Top LiDAR 坐标系的变换

    # 当前 sensor 到 ego 的旋转矩阵
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix

    # 当前 sensor 对应 ego 到 global 的旋转矩阵
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    # 计算 sensor 到当前 top lidar 的旋转
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)

    # 计算 sensor 到当前 top lidar 的平移
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)

    # 减去当前关键帧 top lidar 的平移项
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    # 保存旋转【注释中说明使用方式是 points @ R.T + T】
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T

    # 保存平移
    sweep['sensor2lidar_translation'] = T

    # 返回 sweep/camera 信息
    return sweep


# 入口：
def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """

    # 调用 create_nuscenes_infos 生成 info pkl，其中 dataset_name 参数在当前函数中没有实际使用
    create_nuscenes_infos(root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)





# 一、解析命令行参数
# (1) 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Data converter arg parser')

# (2) 添加参数
# 添加位置参数 dataset，这里实际期望传入 nuscenes
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
# 添加 root-path 参数
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
# 添加 canbus 参数
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
# 添加 version 参数
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
# 添加 max-sweeps 参数
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
# 添加 out-dir 参数
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
# 添加 extra-tag 参数，用于控制输出文件名前缀
parser.add_argument('--extra-tag', type=str, default='kitti')
# 添加 workers 参数，当前文件中没有实际使用 workers
parser.add_argument('--workers', type=int, default=4, help='number of threads to be used')

# (3) 解析命令行参数
args = parser.parse_args()


# 二、Python 脚本入口
if __name__ == '__main__':
    # 1. 如果数据集是 nuscenes，并且 version 不是 v1.0-mini
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # (1) 构造 trainval 版本名：如果命令行传 --version v1.0，这里得到 v1.0-trainval
        train_version = f'{args.version}-trainval'

        # (2) 准备 trainval 数据
        nuscenes_data_prep(
            root_path=args.root_path,       # nuScenes 根目录
            can_bus_root_path=args.canbus,  # CAN bus 根目录
            info_prefix=args.extra_tag,     # 输出文件名前缀
            version=train_version,          # 数据版本
            dataset_name='NuScenesDataset', # 数据集名，当前没有实际使用
            out_dir=args.out_dir,           # 输出目录
            max_sweeps=args.max_sweeps)     # sweeps 数

        # (3) 构造 test 版本名，例如 v1.0-test
        test_version = f'{args.version}-test'

        # (4) 准备 test 数据
        nuscenes_data_prep(
            root_path=args.root_path,       # nuScenes 根目录
            can_bus_root_path=args.canbus,  # CAN bus 根目录
            info_prefix=args.extra_tag,     # 输出文件名前缀
            version=test_version,           # 数据版本
            dataset_name='NuScenesDataset', # 数据集名
            out_dir=args.out_dir,           # 输出目录
            max_sweeps=args.max_sweeps)     # sweeps 数

    # 2. 如果数据集是 nuscenes，并且使用 mini 版本
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        # (1) 构造 trainval 版本名：mini 版本直接使用 v1.0-mini
        train_version = f'{args.version}'

        # (2) 准备 mini train/val 数据
        nuscenes_data_prep(
            root_path=args.root_path,       # nuScenes mini 根目录
            can_bus_root_path=args.canbus,  # CAN bus 根目录
            info_prefix=args.extra_tag,     # 输出文件名前缀
            version=train_version,          # 数据版本 v1.0-mini
            dataset_name='NuScenesDataset', # 数据集名
            out_dir=args.out_dir,           # 输出目录
            max_sweeps=args.max_sweeps)     # sweeps 数
