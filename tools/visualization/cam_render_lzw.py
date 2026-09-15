"""
SparseDrive 的六相机（Camera View）预测结果可视化器。

本文件由 tools/visualization/visualize.py 调用：
(1) 读取当前帧 nuScenes 的六张原始相机图；
(2) 将 SparseDrive 在当前自车/LiDAR 坐标系输出的预测 3D boxes、agent motion 和 final planning 投影到图像平面；
(3) 输出到 <out_dir>/cam_pred/，随后由 visualize.py 与两张 BEV 图横向拼接。

注意：
(1) 本文件只展示 Prediction，不单独生成 Camera GT 图；
(2) 它不会影响模型推理、训练 loss 或 nuScenes 评测指标；
(3) 这里的投影采用 SparseDrive 数据管线中“行向量点坐标”的 lidar2cam 矩阵存储约定。
"""

# (1) os：拼接输出目录、创建 cam_pred 文件夹。
import os

# (2) numpy：读取图像、拼接轨迹和进行投影前的数组运算。
import numpy as np

# (3) cv2：绘制相机名文字；文字颜色为黑色，因此 BGR/RGB 通道顺序不会影响颜色。
import cv2

# (4) PIL：读取原始 RGB 相机图像。
from PIL import Image

# (5) 原始文件导入 matplotlib，但本文件未直接使用 matplotlib 模块名；保留以不改变源码依赖结构。
import matplotlib

# (6) matplotlib.pyplot：创建 2×3 相机图排布、显示图片并保存图像。
import matplotlib.pyplot as plt

# (7) Quaternion：处理 3D box 的 yaw 方向，并与 NuScenesBox 的旋转接口对接。
from pyquaternion import Quaternion

# (8) NuScenesBox：nuScenes 官方 Box 数据结构，内置 3D box 在相机图上的 render() 方法。
from nuscenes.utils.data_classes import Box as NuScenesBox

# (9) view_points：将 camera-frame 3D 点投影到像素平面；
#     box_in_image：判断一个 3D box 是否可见于当前相机；
#     BoxVisibility / transform_matrix：原始文件导入但当前实现未使用，保留原样。
from nuscenes.utils.geometry_utils import (
    view_points,
    box_in_image,
    BoxVisibility,
    transform_matrix,
)

# (10) 复用 BEV renderer 的共享显示配置，保证 BEV 与 Camera 中的目标颜色和分数阈值一致。
from tools.visualization.bev_render import (
    color_mapping,    # instance_id -> RGB 的循环颜色表。
    SCORE_THRESH,     # Detection / Motion 预测框的最低可视化分数。
    MAP_SCORE_THRESH, # 原始文件导入但本文件未绘制 map；保留原样。
    CMD_LIST,         # 仅在已注释的旧 planning 绘制逻辑中出现；当前可执行逻辑未使用。
)


# -----------------------------------------------------------------------------
# 1. 六相机顺序定义
# -----------------------------------------------------------------------------

# (1) 当前 2×3 拼图中希望展示的相机顺序。
#     display index 0~5 依次对应：左前、正前、右前、右后、正后、左后。
CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]

# (2) SparseDrive / nuScenes converter 写入 data['img_filename']、cam_intrinsic、lidar2cam 时的相机顺序。
#     因而不能直接用 display index 访问 data 中的相机字段，需要先通过 .index(cam) 映射到 converter index。
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


# -----------------------------------------------------------------------------
# 2. CamRender：六相机预测叠加渲染类
# -----------------------------------------------------------------------------
class CamRender:
    """
    作用：把模型预测的 3D Detection、Motion 和 final Planning 重投影到六张相机图上。

    核心坐标链：
    (1) SparseDrive box / trajectory：当前自车或 LiDAR 坐标系；
    (2) lidar2cam extrinsic：将三维点转换到对应 camera 坐标系；
    (3) cam_intrinsic：将 camera-frame 3D 点投影为像素坐标；
    (4) Matplotlib axes：在 2×3 原始相机图中叠加 box 边线或 trajectory 散点。
    """

    # -------------------------------------------------------------------------
    # 2.1 初始化：保存显示开关并创建输出目录
    # -------------------------------------------------------------------------
    def __init__(
        self,          # 当前 CamRender 实例。
        plot_choices,  # dict：控制 draw_pred / det / motion / planning 等子任务是否可视化。
        out_dir,       # str：可视化总输出目录。
    ):
        """
        作用：初始化相机可视化器并确保 cam_pred 输出目录存在。

        参数：
        (1) plot_choices：与 BEVRender 共享的任务开关字典。
        (2) out_dir：可视化根目录；本类会在其中创建 cam_pred 子目录。
        """
        self.plot_choices = plot_choices  # 保存显示开关，供各 draw_* 函数判断。

        # 所有六相机预测拼图都保存为 <out_dir>/cam_pred/xxxx.jpg。
        self.pred_dir = os.path.join(out_dir, "cam_pred")

        # 若目录不存在则递归创建，若已存在则不报错。
        os.makedirs(self.pred_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2.2 reset_canvas：创建一张干净的 2×3 六相机画布
    # -------------------------------------------------------------------------
    def reset_canvas(self):
        """
        作用：关闭前一帧画布，并建立 2 行 3 列的六相机拼图画布。

        返回：
        无。self.fig 是 figure，self.axes 的 shape 为 [2, 3]。
        """
        # (1) 关闭当前 figure，避免长序列渲染时图对象累积。
        plt.close()

        # (2) 原实现尝试关闭当前默认坐标轴；真正用于绘制的是下一行新建的 self.axes。
        #     该两行具有一定冗余，但保留原始行为。
        plt.gca().set_axis_off()
        plt.axis('off')

        # (3) 新建 2×3 子图：总宽约 53.3 英寸、高 20 英寸，以尽量维持 six-camera 的宽屏比例。
        self.fig, self.axes = plt.subplots(2, 3, figsize=(160 / 3, 20))

        # (4) 在最终 save_fig() 清边距前，先由 tight_layout 尝试处理子图间的默认布局。
        plt.tight_layout()

    # -------------------------------------------------------------------------
    # 2.3 render：渲染当前样本的完整六相机 Prediction 图
    # -------------------------------------------------------------------------
    def render(
        self,       # 当前 CamRender 实例。
        data,       # dict：dataset.get_data_info(index) 返回的相机路径、内外参和 GT command 等。
        result,     # dict：当前样本 SparseDrive 的预测输出。
        index,      # int：样本序号，用于生成四位补零文件名。
    ):
        """
        作用：完成“底图 -> Detection -> Motion -> Planning -> 保存”的六相机可视化流程。

        绘制顺序：
        (1) 先显示原始相机图片；
        (2) 再叠加预测 3D boxes；
        (3) 再叠加预测 agent motion；
        (4) 最后只在正前相机叠加 final planning trajectory；
        (5) 保存为 cam_pred/xxxx.jpg。

        返回：
        save_path：已保存的六相机拼图路径。
        """
        # (1) 为当前样本创建新的 2×3 空白画布。
        self.reset_canvas()

        # (2) 将六张原始相机图按照 CAM_NAMES_NUSC 的显示顺序放到子图中。
        self.render_image_data(data, index)

        # (3) 将高置信度 Detection 3D box 投影并叠加到每一张相机图。
        self.draw_detection_pred(data, result)

        # (4) 将每个可见目标的 Top-1 Motion trajectory 投影并叠加到每一张相机图。
        self.draw_motion_pred(data, result)

        # (5) 将 result['final_planning'] 投影到正前相机图。
        self.draw_planning_pred(data, result)

        # (6) 与 BEV 图片保持一致，使用四位补零的文件名，便于后续排序与视频生成。
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')

        # (7) 去边距并保存当前 figure。
        self.save_fig(save_path)

        # (8) 返回路径，供 Visualizer.combine() 读取并与 BEV 图片拼接。
        return save_path

    # -------------------------------------------------------------------------
    # 2.4 load_image：读取单张原始图片并写上相机名称
    # -------------------------------------------------------------------------
    def load_image(
        self,       # 当前 CamRender 实例。
        data_path,  # str：某个相机原始图片的绝对/相对路径。
        cam,        # str：相机名，例如 'CAM_FRONT'；作为文字写入图片左上角。
    ):
        """
        作用：读取一张原始相机图片，并直接在图像像素上绘制相机名称。

        参数：
        (1) data_path：图片文件路径。
        (2) cam：要显示在图片左上角的相机名字。

        返回：
        image：numpy RGB 图像数组，已包含相机名称文字。
        """
        # (1) PIL 读取图片通常得到 RGB；np.array 后 shape 通常为 [H, W, 3]。
        image = np.array(Image.open(data_path))

        # (2) 选择 OpenCV 的简单无衬线字体。
        font = cv2.FONT_HERSHEY_SIMPLEX

        # (3) 相机名称文字左下角基准点坐标；(50, 60) 接近图片左上区域。
        org = (50, 60)

        # (4) 字体缩放倍数。
        fontScale = 2

        # (5) 文字颜色。黑色在 BGR 与 RGB 下均为 (0, 0, 0)，因此无通道顺序歧义。
        color = (0, 0, 0)

        # (6) 文字线宽，较粗以适应大分辨率 nuScenes 图片。
        thickness = 4

        # (7) 在原图数组上原地写字，并返回写字后的 image。
        return cv2.putText(
            image,
            cam,
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # 2.5 update_image：把单张图片放到 2×3 拼图的指定子图
    # -------------------------------------------------------------------------
    def update_image(
        self,    # 当前 CamRender 实例。
        image,   # ndarray：[H, W, 3] 的 RGB 图片。
        index,   # int：显示顺序索引 0~5。
        cam,     # str：相机名；原实现传入但该函数内部未使用，保留接口一致性。
    ):
        """
        作用：将单张已标注的相机图片显示到第 index 个子图，并关闭子图的坐标轴与网格。

        参数：
        (1) image：待显示的 RGB 图片。
        (2) index：0~5；通过 get_axis() 映射到 2×3 网格。
        (3) cam：当前未使用，仅保留调用接口。
        """
        # (1) 根据线性 index 找到对应的 Matplotlib Axes。
        ax = self.get_axis(index)

        # (2) 先显示原始图片，它将成为后续 box/trajectory 叠加的底图。
        ax.imshow(image)

        # (3) 关闭当前 pyplot 轴；在多 axes 场景下，下两行 ax 级别设置才是关键。
        plt.axis('off')

        # (4) 关闭该子图的坐标轴刻度与边框。
        ax.axis('off')

        # (5) 关闭网格线，防止默认图表风格覆盖在相机图上。
        ax.grid(False)

    # -------------------------------------------------------------------------
    # 2.6 get_axis：将 0~5 的显示索引映射为 [row, col]
    # -------------------------------------------------------------------------
    def get_axis(
        self,    # 当前 CamRender 实例。
        index,   # int：线性显示索引，预期范围为 0~5。
    ):
        """
        作用：将一维相机序号转换成 2×3 subplot 的二维索引。

        映射：
        index=0,1,2 -> 第 0 行第 0,1,2 列；
        index=3,4,5 -> 第 1 行第 0,1,2 列。
        """
        # 使用整除得到行号、取模得到列号。
        return self.axes[index // 3, index % 3]

    # -------------------------------------------------------------------------
    # 2.7 save_fig：无边距保存六相机拼图
    # -------------------------------------------------------------------------
    def save_fig(
        self,       # 当前 CamRender 实例。
        filename,   # str：输出图片的完整路径。
    ):
        """
        作用：去除 figure 外边距和子图间隔，并保存当前六相机拼图。

        参数：
        filename：目标 jpg 文件路径。
        """
        # (1) 让 2×3 子图尽可能铺满 figure，减少拼图中无用的白边。
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0,
            hspace=0, wspace=0,
        )

        # (2) 取消 Matplotlib 默认边距。
        plt.margins(0, 0)

        # (3) 保存当前 figure；原实现不显式指定 dpi，使用 Matplotlib 默认设置。
        plt.savefig(filename)

    # -------------------------------------------------------------------------
    # 2.8 render_image_data：按 display order 读取并放置六张相机底图
    # -------------------------------------------------------------------------
    def render_image_data(
        self,   # 当前 CamRender 实例。
        data,   # dict：至少包含 img_filename；其第 0 维遵循 CAM_NAMES_NUSC_converter。
        index,  # int：原接口传入的样本序号；当前函数未使用，保留与 render() 调用一致。
    ):
        """
        作用：从 data['img_filename'] 读取六张相机图，再按照 CAM_NAMES_NUSC 排列为 2×3 拼图。

        关键点：
        data 的存储顺序和最终展示顺序不同，因此每次都通过
        CAM_NAMES_NUSC_converter.index(cam) 求 converter index。
        """
        # (1) i 是显示位置 0~5；cam 是该位置要展示的相机名。
        for i, cam in enumerate(CAM_NAMES_NUSC):
            # (1.1) 找到该相机名在 converter 数据顺序中的索引。
            idx = CAM_NAMES_NUSC_converter.index(cam)

            # (1.2) 按 converter index 读取当前相机原始图像路径。
            img_path = data['img_filename'][idx]

            # (1.3) 读取图像并在像素上写入相机名。
            image = self.load_image(img_path, cam)

            # (1.4) 将图像放到 i 对应的 display subplot。
            self.update_image(image, i, cam)

    # -------------------------------------------------------------------------
    # 2.9 draw_detection_pred：将预测 3D boxes 投影到六张相机图
    # -------------------------------------------------------------------------
    def draw_detection_pred(
        self,     # 当前 CamRender 实例。
        data,     # dict：至少包含 cam_intrinsic 与 lidar2cam。
        result,   # dict：至少包含 boxes_3d / labels_3d / scores_3d / instance_ids。
    ):
        """
        作用：把高置信度预测 3D Detection boxes 投影并渲染到每个 camera view。

        坐标与格式：
        (1) result['boxes_3d']：通常为 [N_pred, 7]，格式 [x, y, z, w, l, h, yaw]；
        (2) data['lidar2cam'][idx]：4×4 外参，按“行向量点”约定使用；
        (3) data['cam_intrinsic'][idx]：3×3 相机内参；
        (4) NuScenesBox.render()：接收 camera-frame 的 box，再用内参投影到像素平面。
        """
        # (1) 仅在要求画预测、开启 Detection 且 result 有 boxes_3d 时继续。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['det']
            and "boxes_3d" in result
        ):
            return

        # (2) 原实现假设 boxes_3d 是 CPU tensor；.numpy() 转为 numpy，供 NuScenesBox 使用。
        bboxes = result['boxes_3d'].numpy()

        # (3) 对六个“显示顺序”的 camera 分别进行 box 投影。
        for j, cam in enumerate(CAM_NAMES_NUSC):
            # --------------------- (a) 读取该 camera 的内外参 ---------------------
            # display camera name -> converter index。
            idx = CAM_NAMES_NUSC_converter.index(cam)

            # K，shape 通常为 [3, 3]。
            cam_intrinsic = data['cam_intrinsic'][idx]

            # 所有 camera 的 LiDAR-to-camera 外参，shape 通常为 [6, 4, 4]。
            lidar2cam = data['lidar2cam']

            # 取当前 camera 的 4×4 外参。
            extrinsic = lidar2cam[idx]

            # SparseDrive 这里按“行向量”存点：p_cam = p_lidar @ R + t，因此平移向量在第 4 行。
            trans = extrinsic[3, :3]

            # NuScenesBox.rotate() 内部按列向量运算；为匹配上面的行向量 R，需要使用 R 的逆（旋转矩阵时即 R^T）。
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse

            # nuScenes 原图分辨率，格式为 (width, height)，用于可见性判断和图像坐标范围。
            imsize = (1600, 900)

            # ----------------------- (b) 遍历当前帧预测 boxes -----------------------
            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]  # 当前 box 的置信度。

                # (b.1) 低分 box 不绘制。
                if score < SCORE_THRESH:
                    continue

                # (b.2) 颜色按 instance_id 取，确保同一个 track 在 BEV / Camera / 时间上具有相同颜色。
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

                # ---------------- (c) SparseDrive box -> NuScenesBox ----------------
                # 取当前 box 的三维中心 [x, y, z]，仍在当前自车/LiDAR 坐标系。
                center = bboxes[i, 0:3]

                # 取 [w, l, h] 三个尺寸分量。
                box_dims = bboxes[i, 3:6]

                # SparseDrive 与 NuScenesBox 的局部轴/尺寸约定不同，因此将前两维交换后再构造 NuScenesBox。
                nusc_dims = box_dims[..., [1, 0, 2]]

                # SparseDrive yaw 绕 z 轴旋转；创建表示该 yaw 的四元数。
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])

                # 在 LiDAR/ego 坐标系中构造 NuScenes 3D box。
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat,
                )

                # (c.1) 旋转到 camera 坐标系。这里使用的是上面解释的 inverse quaternion。
                box.rotate(rot)

                # (c.2) 平移到 camera 坐标系。
                box.translate(trans)

                # --------------------- (d) 可见性检查与真正绘制 --------------------
                # 若 box 不落在该 camera 图像可见范围内，则不绘制，避免无意义的屏外投影。
                if box_in_image(box, cam_intrinsic, imsize):
                    # NuScenesBox.render() 会投影 8 个 3D corners 并绘制 box 边线。
                    box.render(
                        self.axes[j // 3, j % 3],
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(color, color, color),  # 三组颜色分别供不同 box 边使用；这里统一为同一种颜色。
                        linewidth=4,
                    )

            # ---------------------- (e) 将图像坐标系恢复为像素方向 -------------------
            # 横轴像素范围为 [0, width]。
            self.axes[j // 3, j % 3].set_xlim(0, imsize[0])

            # Matplotlib 的 y 轴默认向上；图像像素 y 向下，因此用 (height, 0) 翻转 y 轴。
            self.axes[j // 3, j % 3].set_ylim(imsize[1], 0)

    # -------------------------------------------------------------------------
    # 2.10 draw_motion_pred：将每个目标的 Top-1 motion trajectory 投影到六相机
    # -------------------------------------------------------------------------
    def draw_motion_pred(
        self,                 # 当前 CamRender 实例。
        data,                 # dict：至少包含 cam_intrinsic 与 lidar2cam。
        result,               # dict：至少包含 boxes_3d / scores_3d / trajs_3d / trajs_score / instance_ids。
        points_per_step=10,   # int：原接口参数；当前函数未直接使用，实际密度由 _render_traj 默认值控制。
    ):
        """
        作用：把每个高置信度检测目标的最高分 motion mode 投影到六张相机图。

        说明：
        (1) 与 BEV renderer 显示 Top-K 不同，本函数只绘制 trajs_score.argmax() 的 Top-1 mode；
        (2) 轨迹点被放到目标 3D box 的底部高度 z = center_z - h/2，视觉上更贴近路面；
        (3) 若该目标的 3D box 不可见，则其轨迹也不绘制，即使部分轨迹本身可能进入画面。
        """
        # (1) 必须同时开启 Prediction 和 Motion，且 result 有 trajs_3d。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['motion']
            and "trajs_3d" in result
        ):
            return

        # (2) boxes_3d 假设为 CPU tensor，转为 numpy 后用于轨迹起点和高度计算。
        bboxes = result['boxes_3d'].numpy()

        # (3) 对每一个显示相机重复进行投影。
        for j, cam in enumerate(CAM_NAMES_NUSC):
            # --------------------- (a) 读取当前相机内外参 ---------------------
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            # --------------------- (b) 对每个预测 agent 绘制轨迹 ---------------------
            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]

                # (b.1) Detection 分数过低时不绘制其 motion trajectory。
                if score < SCORE_THRESH:
                    continue

                # (b.2) 与 3D box 相同，轨迹颜色由 instance_id 决定。
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

                # (b.3) trajectory mode score，shape 通常为 [num_modes]。
                traj_score = result['trajs_score'][i].numpy()

                # (b.4) 所有候选轨迹，shape 通常为 [num_modes, T_future, 2]。
                traj = result['trajs_3d'][i].numpy()

                # (b.5) Camera view 只显示最高分 mode，而非 BEV 中的 Top-K。
                mode_idx = traj_score.argmax()
                traj = traj[mode_idx]

                # (b.6) 将当前 detection box 的 x-y center 作为轨迹起点，shape [1, 2]。
                origin = bboxes[i, :2][None]

                # (b.7) 在 future trajectory 前补入当前起点，形成 [T_future+1, 2]。
                #       这里不做 cumsum，表明可视化器假定 trajs_3d 已是当前坐标系下的绝对 future x-y。
                traj = np.concatenate([origin, traj], axis=0)

                # (b.8) 新建 z 列，使二维轨迹成为可投影的三维点集 [T_future+1, 3]。
                traj_expand = np.ones((traj.shape[0], 1))

                # (b.9) 将轨迹放在 box 底面高度，以近似车辆/行人接触地面的位置。
                traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2

                # (b.10) 拼接为 [x, y, z] 三维轨迹点。
                traj = np.concatenate([traj, traj_expand], axis=1)

                # ------------------ (c) 构造 box，仅用于相机可见性过滤 ----------------
                center = bboxes[i, 0:3]
                box_dims = bboxes[i, 3:6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat,
                )

                # 将检测 box 从 LiDAR/ego 坐标变换到 camera 坐标。
                box.rotate(rot)
                box.translate(trans)

                # (c.1) 目标 box 不可见时，不继续投影这条 agent trajectory。
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue

                # --------------------- (d) 轨迹坐标变换和投影绘制 --------------------
                # 按行向量约定：trajectory_cam = trajectory_lidar @ R + t。
                traj_points = traj @ extrinsic[:3, :3] + trans

                # 调用辅助函数：插值、过滤 camera 前方点、内参投影并在第 j 个子图 scatter。
                # s=15 比 planning 默认的散点小，避免大量 agent trajectory 覆盖原始相机图。
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

    # -------------------------------------------------------------------------
    # 2.11 draw_planning_pred：将 final_planning 投影到正前相机
    # -------------------------------------------------------------------------
    def draw_planning_pred(
        self,     # 当前 CamRender 实例。
        data,     # dict：至少包含 cam_intrinsic 与 lidar2cam；当前可执行路径不读取 GT command。
        result,   # dict：至少包含 final_planning；前置条件仍检查是否有 planning 字段。
    ):
        """
        作用：将最终规划轨迹 result['final_planning'] 叠加到正前相机图。

        重要区别：
        (1) BEVRender.draw_planning_pred() 展示的是与 GT command 对齐的 Top-K planning candidate modes；
        (2) 本函数展示的是模型经过 planning selection / collision-aware rescore 后的 final_planning；
        (3) 当前实现只投影 CAM_FRONT，不在其它五个相机显示自车规划轨迹。
        """
        # (1) 保持原始 gating：要求开启 Prediction、Planning，并且 result 含 planning。
        #     但下方实际使用的是 final_planning，因此调用者还需要保证 result 同时含 final_planning。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['planning']
            and "planning" in result
        ):
            return

        # ---------------------------------------------------------------------
        # (a) 原始源码中保留的“显示多 command / 多 mode planning 候选”的旧实现。
        #     以下代码全部被注释，不会执行；当前版本改为直接显示 result['final_planning']。
        #
        #     额外注意：原代码写作 enumerate(CAM_NAMES_NUSC[1])。
        #     CAM_NAMES_NUSC[1] 是字符串 'CAM_FRONT'，若取消注释会逐字符遍历而不是遍历一个相机列表。
        #     若要恢复该逻辑，应改为 enumerate([CAM_NAMES_NUSC[1]]) 或 enumerate(CAM_NAMES_NUSC[1:2])。
        # ---------------------------------------------------------------------
        # for j, cam in enumerate(CAM_NAMES_NUSC[1]):
        #     # 通过 camera name 找到该 camera 在 converter fields 中的索引。
        #     idx = CAM_NAMES_NUSC_converter.index(cam)
        #     # 读取正前相机内参。
        #     cam_intrinsic = data['cam_intrinsic'][idx]
        #     # 读取六相机外参数组。
        #     lidar2cam = data['lidar2cam']
        #     # 读取当前相机的 4×4 外参。
        #     extrinsic = lidar2cam[idx]
        #     # 行向量约定下的平移项。
        #     trans = extrinsic[3, :3]
        #     # 为 NuScenesBox 的列向量旋转接口构造 inverse quaternion。
        #     rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        #     # 固定的 nuScenes 原图尺寸。
        #     imsize = (1600, 900)
        #
        #     # 原计划从 result['planning'][0] 获取 planning candidates。
        #     plan_trajs = result['planning'][0].cpu().numpy()
        #     # 原计划将 command 和 mode 维度重排为 [3, num_mode, 6, 2]。
        #     plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        #     # 三种离散 command 的数量。
        #     num_cmd = len(CMD_LIST)
        #     # 每种 command 下的 mode 数。
        #     num_mode = plan_trajs.shape[1]
        #     # 在每条候选轨迹前补 ego 原点。
        #     plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        #     # 原计划将相对位移累积为绝对 future points。
        #     plan_trajs = plan_trajs.cumsum(axis=-2)
        #     # 读取相应的 candidate scores。
        #     plan_score = result['planning_score'][0].cpu().numpy()
        #     # 重排为 [3, num_mode]。
        #     plan_score = plan_score.reshape(3, -1)
        #
        #     # 根据 GT command 选择一个 command branch。
        #     cmd = data['gt_ego_fut_cmd'].argmax()
        #     # 仅保留该 branch 的 trajectories。
        #     plan_trajs = plan_trajs[cmd]
        #     # 仅保留该 branch 的 scores。
        #     plan_score = plan_score[cmd]
        #
        #     # 选择最大 score 的 planning candidate。
        #     mode_idx = plan_score.argmax()
        #     # 获取 Top-1 trajectory。
        #     plan_traj = plan_trajs[mode_idx]
        #     # 生成 z 列，旧实现使用 -2 米作为自车轨迹的显示高度。
        #     traj_expand = np.ones((plan_traj.shape[0], 1)) * -2
        #     # 以下一行是旧尝试，原本想用检测 box 的 bottom z；planning 并没有 bboxes[i] 上下文，因此保持注释。
        #     # traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
        #     # 拼成 3D trajectory。
        #     plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)
        #
        #     # 通过 lidar2cam 变换到相机坐标。
        #     traj_points = plan_traj @ extrinsic[:3, :3] + trans
        #     # 投影并绘制到当前相机子图。
        #     self._render_traj(traj_points, cam_intrinsic, j)

        # ---------------------------------------------------------------------
        # (b) 当前实际执行逻辑：只在 CAM_FRONT 显示 final_planning
        # ---------------------------------------------------------------------
        # data 的 converter order 中 idx=0 对应 CAM_FRONT。
        idx = 0  # front camera。

        # 读取 CAM_FRONT 的内参 K。
        cam_intrinsic = data['cam_intrinsic'][idx]

        # 读取全部 lidar2cam 外参矩阵。
        lidar2cam = data['lidar2cam']

        # 取 CAM_FRONT 的 4×4 外参。
        extrinsic = lidar2cam[idx]

        # 行向量表示下，平移量位于最后一行的前三列。
        trans = extrinsic[3, :3]

        # 原代码计算了 rot，但当前 final_planning 逻辑没有构造 NuScenesBox，因此该局部变量未使用；保留原始行为。
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse

        # ---------------------------------------------------------------------
        # (c) 原始源码中第二段被注释的旧 candidate planning 选择逻辑。
        #     与上方思想相同，只是改成固定使用正前相机；当前也不会执行。
        # ---------------------------------------------------------------------
        # plan_trajs = result['planning'][0].cpu().numpy()
        # plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_trajs = plan_trajs.cumsum(axis=-2)
        # plan_score = result['planning_score'][0].cpu().numpy()
        # plan_score = plan_score.reshape(3, -1)
        #
        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]
        #
        # mode_idx = plan_score.argmax()
        # plan_traj = plan_trajs[mode_idx]

        # ---------------------------------------------------------------------
        # (d) 读取 final planning 并把它抬升/下沉到可投影的 3D 轨迹
        # ---------------------------------------------------------------------
        # final_planning 预期为 numpy 数组，shape 通常为 [T_future, 2]，坐标位于当前 ego/LiDAR 平面。
        plan_traj = result["final_planning"]

        # 在轨迹开头补自车当前位置 [0, 0]，使可视化从当前车辆位置开始。
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)

        # 为每个 x-y 点创建统一的 z 值列，shape [T_future+1, 1]。
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8

        # -1.8 m 是硬编码的显示高度，用于近似自车 LiDAR/ego 原点下方的地面位置。
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        # 使用行向量外参将 [x, y, z] ego/LiDAR points 变换到 CAM_FRONT 坐标系。
        traj_points = plan_traj @ extrinsic[:3, :3] + trans

        # display index j=1 对应 CAM_NAMES_NUSC[1] == CAM_FRONT。
        # 未传 color / s，因此使用 _render_traj 默认橙色与较大散点，突出 ego final planning。
        self._render_traj(traj_points, cam_intrinsic, j=1)

    # -------------------------------------------------------------------------
    # 2.12 _render_traj：插值、过滤相机前方点、投影成像素散点
    # -------------------------------------------------------------------------
    def _render_traj(
        self,                 # 当前 CamRender 实例。
        traj_points,          # ndarray，shape [T, 3]；已在当前 camera 坐标系的三维轨迹点。
        cam_intrinsic,        # ndarray，shape [3, 3]；当前相机内参 K。
        j,                    # int：显示相机的线性索引 0~5。
        color=(1, 0.5, 0),    # tuple：Matplotlib RGB 颜色；默认橙色用于 ego planning。
        s=150,                # float/int：scatter 点面积；planning 默认更大，motion 调用时会传 15。
        points_per_step=10,   # int：相邻轨迹点间的线性插值密度。
    ):
        """
        作用：将 camera-frame 离散 3D trajectory 插值后投影到像素平面，并画为散点。

        参数：
        (1) traj_points：已完成 lidar2cam 变换的 [T, 3] 点序列；第三维是 camera depth。
        (2) cam_intrinsic：投影矩阵 K。
        (3) j：目标 subplot 的线性相机序号。
        (4) color / s：散点视觉样式。
        (5) points_per_step：线性插值密度。

        重要保留说明：
        原始实现没有写 total_xy[-1] = traj_points[-1]。
        因而预分配数组的最后一行保持 [0, 0, 0]，随后会被 z > 0.1 过滤掉；
        这意味着真实“终点”并未被画出。此注释版为保持原始运行行为，没有修改它。
        """
        # (1) 对 T 个离散点形成 (T-1) 段，每段插入 points_per_step 个采样点，最后预留一个数组位置。
        total_steps = (len(traj_points) - 1) * points_per_step + 1

        # (2) 预分配插值后的 camera-frame 3D 点，shape [total_steps, 3]。
        total_xy = np.zeros((total_steps, 3))

        # (3) 对每一段执行线性插值；循环只到 total_steps-2，因此最后一行保持零值（原始行为）。
        for k in range(total_steps - 1):
            # 当前密集点落在第 k // points_per_step 段；下面保留原始下标表达式。
            # 该段三维向量，已在 camera 坐标系中。
            unit_vec = traj_points[k // points_per_step + 1] - traj_points[k // points_per_step]

            # 当前点在这一段中的线性比例为 (k / points_per_step - k // points_per_step)，取值 [0, 1)。
            # p = p_start + ratio * (p_end - p_start)。
            total_xy[k] = (k / points_per_step - k // points_per_step) * \
                unit_vec + traj_points[k // points_per_step]

        # (4) 只保留相机前方且具有正深度的点；z <= 0.1 的点不能进行稳定透视投影。
        #     注意：由于原代码未写入 total_xy[-1]，其默认 [0,0,0] 也会在这里被过滤。
        in_range_mask = total_xy[:, 2] > 0.1

        # (5) view_points 输入为 [3, N]，并使用 normalize=True 执行 K 投影及除深度操作。
        #     返回前两行像素坐标 [u, v]，shape 为 [2, total_steps]。
        traj_points = view_points(
            total_xy.T,
            cam_intrinsic,
            normalize=True,
        )[:2, :]

        # (6) 仅保留相机前方有效点；前面的 [:2] 在这里是冗余安全切片，保持原始代码形式。
        traj_points = traj_points[:2, in_range_mask]

        # (7) 在第 j 个 camera subplot 中绘制投影后的像素散点。
        self.axes[j // 3, j % 3].scatter(
            traj_points[0],
            traj_points[1],
            color=color,
            s=s,
        )
