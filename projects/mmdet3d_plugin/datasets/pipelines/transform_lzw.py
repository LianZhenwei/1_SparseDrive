# 数值计算库
import numpy as np
# mmcv基础库
import mmcv
# 导入MMCV数据容器，用于封装不同类型的真值数据，适配batch拼接逻辑
from mmcv.parallel import DataContainer as DC
# 数据集流水线注册器
from mmdet.datasets.builder import PIPELINES
# 导入数组转tensor工具
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    """
    多尺度深度图生成器
    核心作用：将激光雷达点云投影到多视角图像上，生成不同下采样倍率的深度真值图，用于深度监督训练
    设计：支持多尺度输出，远到近排序赋值保证近处点遮挡远处点，符合视觉成像逻辑
    """

    def __init__(self, downsample=1, max_depth=60):
        """
        初始化深度图生成器
        Args:
            downsample (int/list/tuple): 深度图下采样倍率，支持多尺度输出
            max_depth (float): 最大有效深度值，超过该值的点会被截断
        """
        # 1. 下采样倍率统一为列表格式，支持多尺度
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        # 2. 保存最大有效深度
        self.max_depth = max_depth

    def __call__(self, input_dict):
        """
        流水线执行入口：生成多尺度深度图并写入input_dict
        Args:
            input_dict (dict): 数据字典，需包含points、lidar2img、img_shape等字段
        Returns:
            dict: 更新后的数据字典，新增gt_depth字段
        """
        # 1. 取出点云的xyz坐标，增加维度适配矩阵乘法，形状 [N, 3, 1]
        points = input_dict["points"][..., :3, None]
        # 初始化深度图结果列表
        gt_depth = []

        # 2. 遍历每个相机，逐张生成深度图
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            # (1) 获取当前图像的高宽
            H, W = input_dict["img_shape"][i][:2]

            # (2) 激光雷达点投影到图像平面：旋转+平移
            # 先乘旋转部分，形状 [N, 3]
            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            # (3) 除以深度，得到像素坐标uv
            pts_2d[:, :2] /= pts_2d[:, 2:3]

            # (4) 像素坐标取整，提取深度值
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]

            # (5) 过滤有效点：在图像范围内、深度大于0.1m
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,  # 深度上限在后续截断处理
                ]
            )
            # 按掩码筛选有效点
            V, U, depths = V[mask], U[mask], depths[mask]

            # (6) 深度从远到近排序：保证后续赋值时近处点覆盖远处点，符合遮挡逻辑
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            # 深度值截断到有效范围
            depths = np.clip(depths, 0.1, self.max_depth)

            # (7) 生成不同下采样倍率的深度图
            for j, downsample in enumerate(self.downsample):
                # 初始化对应尺度的结果列表
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                # 计算下采样后的图像高宽
                h, w = (int(H / downsample), int(W / downsample))
                # 像素坐标同步下采样取整
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)

                # 初始化深度图，无效值填充-1
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                # 将深度值赋值到对应像素位置
                depth_map[v, u] = depths
                # 加入对应尺度的结果列表
                gt_depth[j].append(depth_map)

        # 3. 每个尺度的多视角深度图堆叠为tensor，写入数据字典
        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    """
    nuScenes数据格式适配器
    核心作用：将nuScenes原始数据格式转换为SparseDrive模型所需的标准输入格式
    是数据流水线的核心适配层，统一坐标变换、图像格式、真值封装规范
    """

    def __init(self):
        """初始化，无额外参数"""
        pass

    def __call__(self, input_dict):
        """
        流水线执行入口：完成所有格式转换与真值封装
        Args:
            input_dict (dict): nuScenes原始加载数据字典
        Returns:
            dict: 适配后的标准数据字典
        """
        # 1. 坐标变换与图像元数据整理
        # (1) 激光雷达到图像的投影矩阵，float32格式
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        # (2) 图像宽高，格式为[W, H]，连续内存排布
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        # (3) 全局坐标系到激光雷达的逆变换矩阵
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        # (4) 激光雷达到全局坐标系的变换矩阵
        input_dict["T_global"] = input_dict["lidar2global"]

        # 2. 相机内参处理
        if "cam_intrinsic" in input_dict:
            # 内参矩阵转float32
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            # 提取焦距（x方向焦距，近似认为fx=fy）
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]

        # 3. 跟踪ID重命名
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        # 4. 3D检测真值处理
        if "gt_bboxes_3d" in input_dict:
            # (1) 航向角做周期限制，约束到[-pi, pi]范围，避免角度周期性导致损失异常
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            # (2) 封装为DataContainer，转float tensor
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )

        # 5. 3D类别标签处理
        if "gt_labels_3d" in input_dict:
            # 封装为DataContainer，转long tensor
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        # 6. 图像格式转换
        # (1) 通道变换：HWC -> CHW，适配PyTorch卷积输入格式
        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        # (2) 多视角堆叠为 [N_cam, C, H, W]，保证内存连续
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        # (3) 封装为DataContainer，batch时自动堆叠
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        # 7. 地图与运动预测真值封装（不堆叠，每个样本数量不同）
        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            # stack=False表示batch时不堆叠，保留列表格式
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        # 8. 规划类真值封装（每个样本一个，可堆叠）
        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            # stack=True表示batch时自动堆叠
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        """
        角度周期限制工具：将角度值约束到单个周期范围内
        Args:
            val (np.ndarray): 原始角度数组
            offset (float): 偏移量，控制角度范围，0.5对应[-period/2, period/2]
            period (float): 角度周期，通常为pi或2pi
        Returns:
            np.ndarray: 限制周期后的角度数组
        """
        # 公式：val - floor(val/period + offset) * period
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """
    按类别名称过滤真值目标
    作用：只保留训练指定的类别，剔除不需要的类别真值，保证类别体系与模型一致
    """

    def __init__(self, classes):
        """
        初始化类别过滤器
        Args:
            classes (list[str]): 需要保留的类别名称列表
        """
        # 保留的类别名
        self.classes = classes
        # 对应类别索引
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """
        执行过滤：按类别掩码筛选所有关联真值
        Args:
            input_dict (dict): 数据字典
        Returns:
            dict: 过滤后的数据字典
        """
        # 1. 取出真值标签
        gt_labels_3d = input_dict["gt_labels_3d"]
        # 2. 生成类别掩码：在保留类别内的为True
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )

        # 3. 按掩码过滤3D框与标签
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]

        # 4. 同步过滤实例跟踪ID
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]

        # 5. 同步过滤运动预测真值
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """返回类的字符串表示，用于日志打印"""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    """
    圆形距离范围过滤器
    作用：按目标到自车的距离过滤真值，不同类别设置不同的有效距离阈值
    只保留感知范围内的目标，避免远处低质量真值干扰训练
    """

    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        """
        初始化距离过滤器
        Args:
            class_dist_thred (list[float]): 每个类别对应的最大有效距离，索引与类别ID一一对应
        """
        # 每个类别的距离阈值
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        """
        执行距离过滤
        Args:
            input_dict (dict): 数据字典
        Returns:
            dict: 过滤后的数据字典
        """
        # 1. 取出3D框与标签
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]

        # 2. 计算每个目标到自车(原点)的水平距离
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )

        # 3. 生成距离掩码：每个类别按自身阈值判断是否有效
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            # 同类且距离小于阈值则为有效
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        # 4. 按掩码过滤3D框与标签
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d

        # 5. 同步过滤实例ID
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]

        # 6. 同步过滤运动预测真值
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]

        return input_dict

    def __repr__(self):
        """返回类的字符串表示"""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """
    多视角图像归一化
    作用：对每张视角图像执行减均值、除标准差，可选BGR转RGB，统一图像数值分布
    与标准图像归一化逻辑一致，专门适配多视角输入格式
    """

    def __init__(self, mean, std, to_rgb=True):
        """
        初始化归一化器
        Args:
            mean (sequence): 三通道均值
            std (sequence): 三通道标准差
            to_rgb (bool): 是否将BGR转为RGB，默认True
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """
        执行归一化
        Args:
            results (dict): 数据字典
        Returns:
            dict: 归一化后的数据字典，新增img_norm_cfg
        """
        # 1. 逐张对多视角图像执行归一化
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        # 2. 保存归一化配置，用于反归一化可视化
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        """返回类的字符串表示"""
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
