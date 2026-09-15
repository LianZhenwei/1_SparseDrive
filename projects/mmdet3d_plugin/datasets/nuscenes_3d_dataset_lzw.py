import random # 随机数工具，用于数据增强随机采样
import math
import os
from os import path as osp # 路径操作工具
import cv2                 # OpenCV库，用于图像可视化、视频生成
import tempfile            # 临时文件工具，用于评估时临时存储结果
import copy                # 深拷贝工具，用于变换矩阵、配置的安全复制
import prettytable         # 表格打印工具，用于运动评估指标的美观输出
import numpy as np
import torch
from torch.utils.data import Dataset    # PyTorch数据集基类
import pyquaternion                     # 四元数运算库，用于3D空间旋转变换
from shapely.geometry import LineString # Shapely线几何对象，用于地图矢量标注
from nuscenes.utils.data_classes import Box as NuScenesBox               # nuScenes官方Box类，用于结果格式转换
from nuscenes.eval.detection.config import config_factory as det_configs # nuScenes检测评估配置工厂
from nuscenes.eval.common.config import config_factory as track_configs  # nuScenes跟踪评估配置工厂
import mmcv                                  # MMCV核心库
from mmcv.utils import print_log             # MMCV日志打印工具
from mmdet.datasets import DATASETS          # MMDetection数据集注册器
from mmdet.datasets.pipelines import Compose # MMDetection数据流水线组合器

# 导入同目录下的可视化工具函数
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


# 一、nuScenes 3D多任务数据集类
@DATASETS.register_module() # 注册到MMDetection数据集体系
class NuScenes3DDataset(Dataset):
    """
        nuScenes 3D多任务数据集类
        支持任务：3D目标检测、多目标跟踪、在线矢量建图、运动预测、轨迹规划
        坐标系约定：激光雷达坐标系为输入基准，支持lidar→ego→global三级坐标变换
    """

    # ========== 类常量定义 ==========
    # 每个类别的默认属性，用于检测结果提交时的属性补全
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    # 检测误差项到官方指标名称的映射
    ErrNameMapping = {
        "trans_err": "mATE",   # 平均平移误差
        "scale_err": "mASE",   # 平均尺寸误差
        "orient_err": "mAOE",  # 平均朝向误差
        "vel_err": "mAVE",     # 平均速度误差
        "attr_err": "mAAE",    # 平均属性误差
    }

    # 检测类别列表，共10类，与nuScenes官方一致
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    # 地图类别列表，共3类矢量元素
    MAP_CLASSES = (
        'ped_crossing',  # 人行横道
        'divider',       # 车道分隔线
        'boundary',      # 道路边界
    )

    # 可视化颜色映射，每个类别对应一个BGR颜色
    ID_COLOR_MAP = [
        (59, 59, 238),    # car 蓝色
        (0, 255, 0),      # truck 绿色
        (0, 0, 255),      # trailer 深蓝色
        (255, 255, 0),    # bus 青色
        (0, 255, 255),    # construction_vehicle 黄色
        (255, 0, 255),    # bicycle 品红
        (255, 255, 255),  # motorcycle 白色
        (0, 127, 255),    # pedestrian 橙色
        (71, 130, 255),   # traffic_cone 橙红色
        (127, 127, 0),    # barrier 橄榄色
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
    ):
        """
        初始化数据集
        Args:
            ann_file (str): 标注文件路径，pkl格式
            pipeline (list): 数据流水线配置列表
            data_root (str): 数据集根目录
            classes (list): 自定义检测类别，默认使用类内CLASSES
            map_classes (list): 自定义地图类别
            load_interval (int): 帧加载间隔，1为全量加载，越大数据量越少
            with_velocity (bool): 3D框是否包含速度维度
            modality (dict): 模态配置，指定是否使用相机、激光雷达、地图等
            test_mode (bool): 是否为测试模式
            det3d_eval_version (str): 检测评估版本
            track3d_eval_version (str): 跟踪评估版本
            version (str): nuScenes数据集版本
            use_valid_flag (bool): 是否使用官方有效标记过滤真值
            vis_score_threshold (float): 可视化置信度阈值
            data_aug_conf (dict): 数据增强配置
            sequences_split_num (int): 序列拆分数目，用于分组采样
            with_seq_flag (bool): 是否启用序列分组标记
            keep_consistent_seq_aug (bool): 同序列是否保持增强一致
            work_dir (str): 工作目录，用于保存结果、日志
            eval_config (dict): 评估配置
        """
        # 1. 基础参数保存
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        # 调用父类初始化
        super().__init__()

        # 2. 路径与模式配置
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0  # 3D框模式，0为激光雷达坐标系

        # 3. 类别配置
        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None: 
            self.MAP_CLASSES = map_classes
        # 建立类别名到索引的映射字典
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # 4. 加载标注数据
        self.data_infos = self.load_annotations(self.ann_file)

        # 5. 初始化数据流水线
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # 6. 检测与评估配置
        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        # 加载检测评估配置
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        # 设置评估类别名
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys())
        
        self.track3d_eval_version = track3d_eval_version
        # 加载跟踪评估配置
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys())

        # 7. 默认模态配置：默认只用激光雷达
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        # 8. 可视化与增强配置
        self.vis_score_threshold = vis_score_threshold
        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        # 9. 序列分组标记：用于Dataloader分组采样，保证同序列样本同batch
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        # 10. 工作目录与评估配置
        self.work_dir = work_dir
        self.eval_config = eval_config

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        为每个样本设置序列分组标记
        作用：让Dataloader按组采样，保证同一段连续序列的样本在同一个batch
        支持将长序列拆分为多个子序列，适配不同的batch策略
        """
        # 拆分数为-1时，每个样本单独一组
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return
        
        res = []
        curr_sequence = 0  # 当前序列ID

        # 遍历所有样本，根据sweeps判断序列边界
        for idx in range(len(self.data_infos)):
            # 不是第一帧且sweeps为空，说明是新序列的开头
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                curr_sequence += 1
            res.append(curr_sequence)
        
        self.flag = np.array(res, dtype=np.int64)

        # 子序列拆分逻辑
        if self.sequences_split_num != 1:
            # 拆分为all时，每个样本单独一组
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                # 统计每个原始序列的长度
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0

                # 每个原始序列拆分为指定数目的子序列
                for curr_flag in range(len(bin_counts)):
                    # 生成分割点
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )
                    # 为每个子序列分配新的组ID
                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1
                
                # 校验长度一致
                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        """
        生成单帧的数据增强配置
        训练模式：随机采样增强参数；测试模式：固定参数，无随机增强
        Returns:
            aug_config (dict): 包含resize、crop、flip、rotate、rotate_3d的增强参数
        """
        # 无增强配置则返回None
        if self.data_aug_conf is None:
            return None
        
        # 取出原始图像高宽和最终输出高宽
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]

        # 训练模式：随机采样增强参数
        if not self.test_mode:
            # 随机resize缩放比例
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            # 随机裁剪：从底部向上裁剪，保留天空少的区域
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # 随机水平翻转
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True

            # 随机2D图像旋转
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            # 随机3D点云旋转
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        
        # 测试模式：固定增强参数，无随机性
        else:
            # 按比例缩放，保证覆盖最终尺寸
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            # 居中裁剪
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # 无翻转、无旋转
            flip = False
            rotate = 0
            rotate_3d = 0

        # 组装增强配置字典
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        """
        PyTorch Dataset标准入口：获取单个样本
        Args:
            idx (int/dict): 样本索引，或包含索引和增强配置的字典
        Returns:
            data: 经过流水线处理后的样本数据
        """
        # 输入为字典时，取出增强配置和索引
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            # 普通索引，自行生成增强配置
            aug_config = self.get_augmentation()

        # 获取单帧原始数据信息
        data = self.get_data_info(idx)
        # 注入增强配置
        data["aug_config"] = aug_config
        # 经过数据流水线处理，得到最终模型输入
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        """
        获取指定样本包含的所有类别ID
        用于类别平衡采样，避免某类样本过多
        Args:
            idx (int): 样本索引
        Returns:
            cat_ids (list): 该样本包含的类别索引列表
        """
        info = self.data_infos[idx]

        # 根据有效标记过滤真值
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        # 类别名转索引
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """
        加载pkl格式的标注文件
        Args:
            ann_file (str): 标注文件路径
        Returns:
            data_infos (list): 按时间排序的样本信息列表
        """
        # 加载pkl文件
        data = mmcv.load(ann_file, file_format="pkl")
        # 按时间戳升序排序所有样本
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        # 按间隔采样，减少数据量
        data_infos = data_infos[:: self.load_interval]
        # 保存数据集元数据
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos
    
    def anno2geom(self, annos):
        """
        将地图标注的点数组转换为Shapely LineString几何对象
        Args:
            annos (dict): 分类别的地图标注，值为点数组列表
        Returns:
            map_geoms (dict): 分类别的Shapely几何对象列表
        """
        map_geoms = {}
        # 遍历每个地图类别
        for label, anno_list in annos.items():
            map_geoms[label] = []
            # 每个标注转成LineString
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms

    # 【重点】组装单帧的所有原始输入信息：
    def get_data_info(self, index):
        """
        核心方法：组装单帧的所有原始输入信息
        包含：元数据、点云信息、位姿变换、相机参数、地图几何、真值标注
        Args:
            index (int): 样本索引
        Returns:
            input_dict (dict): 单帧完整数据字典，输入到后续流水线
        """
        info = self.data_infos[index]

        # 1. 基础元数据与传感器路径，并注入 input_dict 字典
        input_dict = dict(
            token=info["token"],               # 样本唯一标识
            map_location=info["map_location"], # 所属城区地图
            pts_filename=info["lidar_path"],   # 点云文件路径
            sweeps=info["sweeps"],             # 历史扫帧列表
            timestamp=info["timestamp"] / 1e6, # 时间戳，转秒
            lidar2ego_translation=info["lidar2ego_translation"],   # 激光雷达到自车的平移
            lidar2ego_rotation=info["lidar2ego_rotation"],         # 激光雷达到自车的旋转
            ego2global_translation=info["ego2global_translation"], # 自车到全局的平移
            ego2global_rotation=info["ego2global_rotation"],       # 自车到全局的旋转
            ego_status=info['ego_status'].astype(np.float32),      # 自车状态（速度、加速度等）
            map_infos=info["map_annos"],                           # 地图标注原始信息
        )


        # 2. 计算激光雷达→全局的变换矩阵，并注入 input_dict 字典
        # (1) 构建 lidar → ego 的 4x4 变换矩阵
        lidar2ego = np.eye(4)                                                                   # 激光雷达到自车的4x4变换矩阵，初始化全 1
        lidar2ego[:3, :3] = pyquaternion.Quaternion(info["lidar2ego_rotation"]).rotation_matrix # 旋转部分：四元数转旋转矩阵
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])                              # 平移部分

        # (2) 构建 ego → global 的 4x4 变换矩阵
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])

        # (3) 计算激光雷达→全局的变换矩阵（矩阵乘法：先lidar→ego，再ego→global），并注入 input_dict 字典
        input_dict["lidar2global"] = ego2global @ lidar2ego


        # 3. 地图标注转Shapely几何对象，并注入 input_dict 字典
        map_geoms = self.anno2geom(info["map_annos"])
        input_dict["map_geoms"] = map_geoms

        # 4. 【重点】相机模态信息：多视角图像、投影矩阵、内参
        if self.modality["use_camera"]:
            image_paths = []   # 各视角图像路径
            lidar2img_rts = [] # 激光雷达到图像的投影矩阵
            lidar2cam_rts = [] # 激光雷达到相机坐标系的变换矩阵（即激光到相机的外参）
            cam_intrinsic = [] # 相机内参矩阵

            # 遍历每个相机
            for cam_type, cam_info in info["cams"].items():
                # (1) 获取各视角图像路径
                image_paths.append(cam_info["data_path"])

                # (2) 计算激光雷达到相机坐标系的变换矩阵（即激光到相机的外参）
                # 计算激光雷达到相机的旋转：相机到lidar旋转的逆
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                # 计算激光雷达到相机的平移
                lidar2cam_t = (cam_info["sensor2lidar_translation"] @ lidar2cam_r.T)
                # 组装4x4变换矩阵
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                # (3) 保存相机内参
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)

                # (4) 计算激光雷达到图像的投影矩阵：内参矩阵扩展为4x4，与外参相乘得到lidar→图像的投影矩阵
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T

                # (5) 保存激光雷达到相机坐标系的变换矩阵、保存激光雷达到图像的投影矩阵
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)

            # 所有相机信息注入字典
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        # 5. 加载真值标注，并注入 input_dict 字典
        annos = self.get_ann_info(index)
        input_dict.update(annos)

        # 6. 返回 input_dict 字典
        return input_dict

    # 【重点】获取单帧的各任务的 GT 标注信息
    def get_ann_info(self, index):
        """
        获取单帧的真值标注信息，支持多任务真值
        Args:
            index (int): 样本索引
        Returns:
            anns_results (dict): 包含检测、跟踪、运动、规划各类真值
        """
        info = self.data_infos[index]

        # 1. 生成有效目标掩码
        if self.use_valid_flag:       # 若使用官方有效标记
            mask = info["valid_flag"] # 则使用官方有效标记
        else:
            mask = info["num_lidar_pts"] > 0 # 否则，默认激光雷达点数大于0为有效

        # 2. 3D框与类别真值
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []

        # 类别名转索引，不在类别内的标为-1
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # 3. 速度真值处理
        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            # NaN速度置为0
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            # 速度维度拼接到3D框末尾，框维度从7变为9
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # 4. 基础检测真值组装
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        # 5. 跟踪实例ID真值
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds
            
        # 6. 运动预测真值：周围目标的未来轨迹与掩码
        if 'gt_agent_fut_trajs' in info:
            anns_results['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][mask]
            anns_results['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][mask]

        # 7. 规划相关真值：自车未来轨迹、掩码、驾驶命令
        if 'gt_ego_fut_trajs' in info:
            anns_results['gt_ego_fut_trajs'] = info['gt_ego_fut_trajs']
            anns_results['gt_ego_fut_masks'] = info['gt_ego_fut_masks']
            anns_results['gt_ego_fut_cmd'] = info['gt_ego_fut_cmd']
        
            # 生成规划评估用的未来帧真值框：对齐到当前帧坐标系
            fut_ts = int(info['gt_ego_fut_masks'].sum())  # 未来时间步数量
            fut_boxes = []
            cur_scene_token = info["scene_token"]
            cur_T_global = get_T_global(info)  # 当前帧lidar→global变换

            # 遍历每个未来时间步
            for i in range(1, fut_ts + 1):
                fut_info = self.data_infos[index + i]
                fut_scene_token = fut_info["scene_token"]
                # 跨场景则终止
                if cur_scene_token != fut_scene_token:
                    break

                # 未来帧真值过滤
                if self.use_valid_flag:
                    mask = fut_info["valid_flag"]
                else:
                    mask = fut_info["num_lidar_pts"] > 0
                fut_gt_bboxes_3d = fut_info["gt_boxes"][mask]
                
                # 计算未来帧→当前帧的坐标变换矩阵
                fut_T_global = get_T_global(fut_info)
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

                # 中心点坐标变换到当前帧
                center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
                # 航向角变换到当前帧：用旋转矩阵旋转航向向量再反正切
                yaw = np.stack([np.cos(fut_gt_bboxes_3d[:, 6]), np.sin(fut_gt_bboxes_3d[:, 6])], axis=-1)
                yaw = yaw @ T_fut2cur[:2, :2].T
                yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

                # 更新变换后的框
                fut_gt_bboxes_3d[:, :3] = center
                fut_gt_bboxes_3d[:, 6] = yaw
                fut_boxes.append(fut_gt_bboxes_3d)

            anns_results['fut_boxes'] = fut_boxes
        
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        """
        将检测/跟踪结果转换为nuScenes官方提交格式
        Args:
            results (list): 所有样本的模型预测结果
            jsonfile_prefix (str): 输出json文件的前缀路径
            tracking (bool): 是否为跟踪模式
        Returns:
            res_path (str): 生成的结果json文件路径
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")

        # 遍历每个样本的预测结果
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            # 模型输出转nuScenes Box对象
            boxes = output_to_nusc_box(det, threshold=self.tracking_threshold if tracking else None)
            sample_token = self.data_infos[sample_id]["token"]

            # 框坐标从激光雷达坐标系转到全局坐标系
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )

            # 遍历每个框，组装提交格式
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]

                # 跟踪模式下，跳过无跟踪意义的类别
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue

                # 根据速度分配属性
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                # 基础提交字段
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )

                # 检测模式追加检测字段
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                # 跟踪模式追加跟踪字段
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                annos.append(nusc_anno)

            nusc_annos[sample_token] = annos

        # 组装完整提交结构
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        # 创建目录并保存json
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        """
        调用nuScenes官方工具执行单任务评估
        Args:
            result_path (str): 结果json文件路径
            logger: 日志器
            result_name (str): 结果名称，用于指标前缀
            tracking (bool): 是否为跟踪评估
        Returns:
            detail (dict): 评估指标字典
        """
        from nuscenes import NuScenes
        # 输出目录为结果文件所在目录
        output_dir = osp.join(*osp.split(result_path)[:-1])

        # 初始化nuScenes工具
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)

        # 评估集映射
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }

        # 检测评估
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval
            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            # 执行评估
            nusc_eval.main(render_curves=False)

            # 读取评估结果
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"

            # 逐类别解析AP和误差指标
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val

                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            # 核心指标NDS和mAP
            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]

        # 跟踪评估
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval
            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # 读取跟踪指标
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"

            # 所有跟踪指标存入字典
            keys = [
                "amota", "amotp", "recall", "motar", "gt",
                "mota", "motp", "mt", "ml", "faf",
                "tp", "fp", "fn", "ids", "frag",
                "tid", "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        """
        统一的结果格式化入口，支持多模态结果
        Args:
            results (list): 预测结果列表
            jsonfile_prefix (str): 输出前缀
            tracking (bool): 是否为跟踪模式
        Returns:
            result_files: 结果文件路径
            tmp_dir: 临时目录对象
        """
        assert isinstance(results, list), "results must be a list"

        # 无指定前缀则创建临时目录
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # 单模态结果直接格式化
        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        # 多模态结果逐模态格式化
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        """
        格式化矢量地图预测结果，生成提交文件
        Args:
            results (list): 地图预测结果
            prefix (str): 输出目录前缀
        Returns:
            out_path (str): 生成的提交文件路径
        """
        submissions = {'results': {},}
        
        # 遍历每个样本的预测
        for j, pred in enumerate(results):
            # 空预测跳过
            if pred is None:
                continue
            pred = pred['img_bbox']
            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']

            # 遍历每个预测实例
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                # 至少2个点才是有效线
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        # 保存提交文件
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        """
        格式化运动预测结果，将轨迹附加到检测结果中
        Args:
            results (list): 运动预测结果
            jsonfile_prefix (str): 输出前缀
            tracking (bool): 是否为跟踪模式
            thresh (float): 置信度阈值
        Returns:
            nusc_submissions (dict): 格式化后的提交字典
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            # 检测结果转nuScenes Box
            boxes = output_to_nusc_box(det['img_bbox'], threshold=None)
            sample_token = self.data_infos[sample_id]["token"]

            # 坐标转全局
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )

            # 遍历每个框，附加轨迹
            for i, box in enumerate(boxes):
                # 阈值过滤
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]

                # 跟踪模式跳过无关类别
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue

                # 属性分配（同检测逻辑）
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                # 基础字段
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )

                # 检测/跟踪字段
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                # 附加预测轨迹
                nusc_anno.update(
                    dict(
                        trajs=det['img_bbox']['trajs_3d'][i].numpy(),
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        # 组装完整提交结构
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }
        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """
        运动预测单任务评估
        Args:
            results: 格式化后的运动预测结果
            result_path: 结果路径
            logger: 日志器
            metric: 指标名称
            result_name: 结果名称前缀
        Returns:
            metrics: 运动评估指标字典
        """
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)

        # 评估集映射
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        # 初始化运动评估器
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)  # 预测6秒未来轨迹

        # 执行评估
        metrics = nusc_eval.main(render_curves=False)
        
        # 核心运动指标
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        # 表格打印指标
        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)

        return metrics

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """
        总评估入口：根据配置执行多任务评估，汇总所有指标
        Args:
            results (list): 所有样本的预测结果
            eval_mode (dict): 评估模式配置，指定各任务是否开启
            metric: 指标名称
            logger: 日志器
            jsonfile_prefix: json输出前缀
            result_names: 结果名称列表
            show (bool): 是否可视化
            out_dir: 可视化输出目录
            pipeline: 可视化流水线
        Returns:
            results_dict (dict): 所有任务的评估指标汇总
        """
        # 保存完整结果pkl
        res_path = "results.pkl" if "trainval" in self.version else "results_mini.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()

        # 1. 检测与跟踪评估
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]

            # 分别执行检测和跟踪评估
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                # 未开启跟踪则跳过
                if tracking and not self.tracking:
                    continue

                # 格式化结果
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                # 多模态结果逐模态评估
                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                # 单模态结果直接评估
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                # 清理临时目录
                if tmp_dir is not None:
                    tmp_dir.cleanup()

        # 2. 地图任务评估
        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            # 格式化地图结果
            result_path = self.format_map_results(results, prefix=self.work_dir)
            # 执行地图评估
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        # 3. 运动预测评估
        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            # 格式化运动结果
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            # 执行运动评估
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)
        
        # 4. 规划任务评估
        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            # 执行规划评估
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        # 5. 可视化输出
        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        
        # 6. 打印核心指标摘要
        metric_str = '\n'
        # 检测核心指标
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'
        
        # 跟踪核心指标
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 
        
        # 地图核心指标
        if "mAP_normal" in results_dict:
            metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n' 
            metric_str += f'divider= {results_dict["divider"]:.4f}\n' 
            metric_str += f'boundary= {results_dict["boundary"]:.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        # 运动核心指标
        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n' 
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n' 

        # 规划核心指标
        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'
        
        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        """
        可视化预测结果：生成多视角图像+BEV的拼接图，以及视频
        Args:
            results (list): 预测结果
            save_dir (str): 保存目录
            show (bool): 是否即时显示
            pipeline: 数据流水线，用于加载原始图像
        """
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 视频编码器
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        # 遍历每个样本可视化
        for i, result in enumerate(results):
            # 取出检测结果
            if "img_bbox" in result.keys():
                result = result["img_bbox"]

            # 加载原始数据
            data_info = pipeline(self.get_data_info(i))
            imgs = []
            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]

            # 按置信度阈值过滤预测框
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]

            # 确定颜色：跟踪模式按ID上色，否则按类别上色
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== 多视角图像绘制3D框 =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== BEV图绘制3D框 =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== 添加视角名称标签并拼接 =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                # 白色背景条
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                # 计算文字位置，居中
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)
                # 绘制文字
                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 6视角拼接成2行3列
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            # 左侧拼接BEV图
            image = np.concatenate([image, bev], axis=1)

            # ===== 保存图片与视频 =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)

        # 释放视频写入器
        videoWriter.release()


# ========== 全局工具函数 ==========
# 1. 将模型输出的检测结果转换为 nuScenes Box 对象列表
def output_to_nusc_box(detection, threshold=None):
    """
    将模型输出的检测结果转换为nuScenes Box对象列表
    Args:
        detection (dict): 模型检测结果，含boxes_3d、scores_3d、labels_3d
        threshold (float): 置信度阈值，None则不过滤
    Returns:
        box_list (list): NuScenesBox对象列表
    """

    # (1) 提取模型输出的检测结果
    box3d = detection["boxes_3d"]           # 从检测结果字典中获取 3D 边界框张量（可能为 torch.Tensor 或 numpy 数组）
    scores = detection["scores_3d"].numpy() # 获取每个检测框的置信度分数，并转换为 numpy 数组
    labels = detection["labels_3d"].numpy() # 获取每个检测框的类别标签，并转换为 numpy 数组
    if "instance_ids" in detection:     # 若检测结果中包含实例跟踪ID信息
        ids = detection["instance_ids"] # 则获取实例跟踪 ID 数组（用于后续关联同一目标）

    # (2) 阈值过滤
    if threshold is not None: # 若指定了置信度阈值
        # (a) 构建 box 的类别得分 ≥ 阈值的 mask
        if "cls_scores" in detection:                           # 若存在类别得分
            mask = detection["cls_scores"].numpy() >= threshold # 则用类别得分生成布尔掩码，保留得分≥阈值的框
        else:                                                   # 若不存在类别得分
            mask = scores >= threshold                          # 则使用常规的scores分数，用总体置信度生成布尔掩码
        # (b) 根据 mask 过滤
        box3d = box3d[mask]   # 根据掩码过滤3D框
        scores = scores[mask] # 根据掩码过滤置信度分数
        labels = labels[mask] # 根据掩码过滤类别标签
        ids = ids[mask]       # 根据掩码过滤跟踪 ID

    # (3) 解析框参数：中心点、尺寸、航向
    # (a) 若 box3d 具有官方 Box 类的属性，则解析官方 box 类格式的框参数
    if hasattr(box3d, "gravity_center"): 
        # 官方Box类格式
        box_gravity_center = box3d.gravity_center.numpy() # 获取所有框的重心（中心点）坐标，并转为numpy数组
        box_dims = box3d.dims.numpy()                     # 获取所有框的尺寸（长宽高），并转为numpy数组
        nus_box_dims = box_dims[:, [1, 0, 2]]             # 长宽顺序转换：尺寸排列从 [长, 宽, 高] 转换为nuScenes要求的 [宽, 长, 高]（即交换前两维）
        box_yaw = box3d.yaw.numpy()                       # 获取所有框的航向角（绕Z轴旋转角度），转为numpy数组
    # (b) 否则 box3d 是原始数组格式 [x,y,z, l,w,h, yaw, vx, vy, vz]
    else:
        # 数组格式
        box3d = box3d.numpy() # 将张量转换为numpy数组
        box_gravity_center = box3d[..., :3].copy() # 提取前3个元素作为中心坐标 (x, y, z)
        box_dims = box3d[..., 3:6].copy()          # 提取第4-6个元素作为尺寸 (l, w, h)
        nus_box_dims = box_dims[..., [1, 0, 2]]    # 同样将尺寸排列转换为 [宽, 长, 高]
        box_yaw = box3d[..., 6].copy()             # 提取第7个元素作为航向角 (yaw)

    # (4) 逐个构造Box对象
    box_list = []               # 初始化空列表，用于存放生成的NuScenesBox对象
    for i in range(len(box3d)): # 遍历每一个检测框
        # (a) 航向角转四元数，绕 z 轴旋转
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i]) # 根据航向角（弧度）创建绕Z轴旋转的单位四元数

        # (b) 构建速度向量，其中 vz 设为 0
        if hasattr(box3d, "gravity_center"):        # 若为官方 Box 类格式，
            velocity = (*box3d.tensor[i, 7:9], 0.0) # 则从 Box 张量中提取第 8-9 个元素作为 vx,vy，而 vz 设为0，构成速度三元组；
        else:                                       # 若为原始数组格式 [x,y,z, l,w,h, yaw, vx, vy, vz]，
            velocity = (*box3d[i, 7:9], 0.0)        # 则从数组的第 8-9 个元素提取 vx,vy ，而 vz 设为0。

        # (c) 构造nuScenes Box对象
        box = NuScenesBox(
            box_gravity_center[i], # 当前框的中心点坐标 (x, y, z)
            nus_box_dims[i],       # 当前框的尺寸 (宽, 长, 高)
            quat,                  # 当前框的四元数表示
            label=labels[i],       # 当前框的类别标签
            score=scores[i],       # 当前框的置信度分数
            velocity=velocity,     # 当前框的速度向量 (vx, vy, 0)
        )

        # (d) 附加跟踪ID
        if "instance_ids" in detection:
            box.token = ids[i] # 将当前框的实例ID存入Box对象的token属性

        # (e) 将构造好的Box对象添加到结果列表中
        box_list.append(box)

    # (5) 返回包含所有转换后Box对象的列表
    return box_list

# 2. 将激光雷达坐标系下的框转换到全局坐标系，并按类别范围过滤
def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    filter_with_cls_range=True,
):
    """
    将激光雷达坐标系下的框转换到全局坐标系，并按类别范围过滤
    Args:
        info (dict): 样本信息，含位姿变换
        boxes (list): 激光雷达坐标系下的Box列表
        classes (list): 类别列表
        eval_configs: 评估配置，含类别有效范围
        eval_version (str): 评估版本
        filter_with_cls_range (bool): 是否按类别范围过滤
    Returns:
        box_list (list): 全局坐标系下的Box列表
    """

    # 创建全局坐标系下的 Box 列表
    box_list = []

    # 遍历激光雷达坐标系下的 Box 列表，将其进行
    for i, box in enumerate(boxes):
        # (1) 第一步：lidar → ego 坐标系
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))

        # 按类别感知范围过滤，超出范围的目标不参与评估
        if filter_with_cls_range:                         # 检查是否启用类别感知范围过滤
            cls_range_map = eval_configs.class_range      # 获取类别到最大感知距离的映射字典（如 {'car': 50, 'pedestrian': 30}）
            radius = np.linalg.norm(box.center[:2], 2)    # 计算 box 目标中心在 XY 平面上的欧几里得距离，忽略 Z 轴
            det_range = cls_range_map[classes[box.label]] # 根据目标类别查询该类别允许的最大径向距离阈值
            if radius > det_range: # 如果当前目标的径向距离超过该类别设定的阈值
                continue           # 则跳过该目标，不参与后续的评估（如匹配、指标计算等）

        # (2) 第二步：ego → global 坐标系
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

# 3. 计算激光雷达到全局坐标系的4x4变换矩阵：lidar  → ego → global 
def get_T_global(info):
    """
    计算激光雷达到全局坐标系的4x4变换矩阵
    Args:
        info (dict): 样本信息，含lidar2ego、ego2global变换
    Returns:
        4x4变换矩阵
    """
    # (1) 构造 lidar → ego 变换矩阵
    lidar2ego = np.eye(4)                                                                   # 创建一个 4×4 的单位矩阵，作为 lidar 坐标系到 ego 车辆坐标系的变换矩阵的初始值
    lidar2ego[:3, :3] = pyquaternion.Quaternion(info["lidar2ego_rotation"]).rotation_matrix # 从 info 字典中读取 lidar 到 ego 的旋转四元数，转换为旋转矩阵，并赋值给变换矩阵的左上 3×3 区域
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])                              # 从 info 中读取 lidar 到 ego 的平移向量（3维），赋值给变换矩阵的右上 3×1 区域（最后一列的前三行）

    # (2) 构造 ego → global 变换矩阵
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])

    # (3) 矩阵相乘得到 lidar → global 的总变换
    return ego2global @ lidar2ego
