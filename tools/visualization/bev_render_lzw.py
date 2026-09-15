"""
    SparseDrive 的 BEV（Bird's-Eye View，鸟瞰图）可视化器。

    本文件由 tools/visualization/visualize.py 调用：
        (1) 对每一个 nuScenes 样本分别绘制 GT 图和预测图；
        (2) 将 Detection / Tracking / Motion / Map / Planning 统一投影到当前自车坐标系的 x-y 平面；
        (3) 输出到 <out_dir>/bev_gt/ 与 <out_dir>/bev_pred/，供 visualize.py 再与六相机图拼接。

    注意：本文件只做“结果展示”，不会参与 SparseDrive 的前向推理、损失计算或评测。
"""


import os          # os：拼接输出目录与创建文件夹。
import numpy as np # numpy：处理轨迹、角点、颜色与数组形状。
import cv2         # cv2：读取并将 OpenCV 默认 BGR 图片转换为 Matplotlib 使用的 RGB 图片。
import matplotlib  # matplotlib：访问 colormap，并创建/保存 BEV 图像。
import matplotlib.pyplot as plt
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners # 将 SparseDrive 的 3D box 参数 [x, y, z, w, l, h, yaw] 转换为 8 个 3D 角点。


# -----------------------------------------------------------------------------
# 1. 全局显示配置
# -----------------------------------------------------------------------------
# (1) 自车未来驾驶指令的类别顺序【data['gt_ego_fut_cmd'] 是与该顺序对应的 one-hot / multi-hot 向量；argmax() 得到命令索引】
CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']

# (2) 在线地图三类 vector 的显示颜色【label 是 map head 输出的类别索引；“索引 -> 具体地图语义”的定义来自配置文件，渲染器本身只负责按索引取色】
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']

# (3) 3D Detection / Tracking / Motion 预测的可视化置信度阈值【低于该值的预测框与其轨迹不绘制，避免大量低置信度 query 干扰观察】
SCORE_THRESH = 0.3

# (4) Online Map 预测的可视化置信度阈值。
MAP_SCORE_THRESH = 0.3

# (5) Tracking 可视化颜色表。
#     每一行是一个 [R, G, B] 颜色，数值范围先为 [0, 255]；末尾除以 255 后变为 Matplotlib 接受的 [0, 1] 浮点 RGB。
#     预测目标使用 instance_id % len(color_mapping) 取色，因此同一 track 在不同帧、BEV 和 Camera 图中通常保持同色。
color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255  # 将 uint8 风格 RGB 归一化为 Matplotlib 所需的浮点 RGB。


# -----------------------------------------------------------------------------
# 2. BEVRender：BEV GT / Prediction 渲染类
# -----------------------------------------------------------------------------
class BEVRender:
    """
        作用：将当前帧的 GT 或 SparseDrive 预测结果绘制到同一张 BEV 平面图。

        坐标约定：
            (1) 所有 box、agent trajectory、map vector 与 ego planning 都已位于“当前帧自车/LiDAR 坐标系”；
            (2) 本类只取平面 x-y 坐标绘图，不再做时序坐标变换或相机投影；
            (3) 自车图片固定放在原点附近，因此图中其他元素的位置都应相对于当前自车原点理解。
    """

    # -------------------------------------------------------------------------
    # 2.1 初始化：记录开关、BEV 范围与输出目录
    # -------------------------------------------------------------------------
    def __init__(
        self,         # 当前 BEVRender 实例。
        plot_choices, # dict：各子任务是否绘制，例如 det / track / motion / map / planning / draw_pred。
        out_dir,      # str：可视化总输出目录，例如 'vis'。
        xlim=40,      # float：BEV 横轴范围，最终显示 [-xlim, xlim] 米。
        ylim=40,      # float：BEV 纵轴范围，最终显示 [-ylim, ylim] 米。
    ):
        """
            作用：初始化绘制配置，并创建 GT 和 Prediction 的 BEV 输出目录。

            参数：
                (1) plot_choices：控制各任务是否显示的字典；该字典由 visualize.py 的全局配置传入。
                (2) out_dir：本次可视化的根目录。
                (3) xlim / ylim：以自车为中心的 BEV 可视化半径，不会改变模型本身的 ROI 或数据。
        """
        self.plot_choices = plot_choices  # 保存任务开关，供后续每个 draw_* 函数判断。
        self.xlim = xlim                  # 保存横轴显示范围。
        self.ylim = ylim                  # 保存纵轴显示范围。

        # (1) GT 图输出到 <out_dir>/bev_gt。
        self.gt_dir = os.path.join(out_dir, "bev_gt")

        # (2) Prediction 图输出到 <out_dir>/bev_pred。
        self.pred_dir = os.path.join(out_dir, "bev_pred")

        # (3) 若目录已经存在不报错；若不存在则递归创建。
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2.2 reset_canvas：为一张新的 BEV 图创建干净画布
    # -------------------------------------------------------------------------
    def reset_canvas(self):
        """
            作用：关闭上一张 Matplotlib 图，并新建一个固定范围、无坐标轴的单子图画布。

            返回：
                无。新图对象保存到 self.fig，坐标轴对象保存到 self.axes。
        """
        plt.close()  # 关闭当前 figure，避免连续渲染多帧时不断累积图窗口和内存。

        # 创建 1 行 1 列子图；figsize=(20, 20) 使 BEV 输出近似正方形。
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))

        # 设置以自车原点为中心的横轴范围。
        self.axes.set_xlim(-self.xlim, self.xlim)

        # 设置以自车原点为中心的纵轴范围。
        self.axes.set_ylim(-self.ylim, self.ylim)

        # 关闭坐标轴刻度、边框与标签，使结果更像视频可视化而不是坐标图。
        self.axes.axis('off')

    # -------------------------------------------------------------------------
    # 2.3 render：完整绘制当前样本的 GT BEV 图与 Prediction BEV 图
    # -------------------------------------------------------------------------
    def render(
        self,       # 当前 BEVRender 实例。
        data,       # dict：dataset.get_data_info(index) 返回的当前样本数据与 GT 标注。
        result,     # dict：results[index]['img_bbox'] 中当前样本的预测结果。
        index,      # int：当前样本序号；用于生成四位补零的图片文件名。
    ):
        """
        作用：按固定顺序输出两张 BEV 图片：一张 GT 图、一张预测图。

        绘制顺序：
        (1) GT 图：Detection GT -> Motion GT -> Map GT -> Planning GT -> 自车图标/指令/图例；
        (2) Prediction 图：Detection -> Track history -> Motion -> Map -> Planning -> 自车图标/指令/图例；
        (3) 返回两张已保存图片的路径，供 visualize.py 拼接使用。

        参数：
        data：当前帧的输入元信息和 GT。
        result：模型对当前帧的输出字典。
        index：当前帧在可视化区间内的索引。

        返回：
        (save_path_gt, save_path_pred)：GT 图片路径与 Prediction 图片路径。
        """
        # ========================= 1. 绘制 GT BEV 图 =========================
        self.reset_canvas()  # 每张图都从一个空白画布开始。

        # (1) 绘制 3D detection GT 的俯视框。
        self.draw_detection_gt(data)

        # (2) 绘制周围 agent 的未来运动 GT。
        self.draw_motion_gt(data)

        # (3) 绘制 vectorized online map GT。
        self.draw_map_gt(data)

        # (4) 绘制自车未来 planning GT。
        self.draw_planning_gt(data)

        # (5) 将自车 PNG 覆盖在原点处，强调自车当前位置与朝向。
        self._render_sdc_car()

        # (6) 在左下角显示当前帧对应的 GT command。
        self._render_command(data)

        # (7) 在右下角覆盖图例 PNG。
        self._render_legend()

        # (8) 以 0000.jpg、0001.jpg 这类固定长度文件名保存，便于后续按字典序合成视频。
        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)

        # ====================== 2. 绘制 Prediction BEV 图 ====================
        self.reset_canvas()  # GT 图保存后，创建新的空白画布，防止 GT 与 Prediction 混在同一张图中。

        # (1) 绘制当前帧通过阈值的预测 3D boxes。
        self.draw_detection_pred(result)

        # (2) 若 result 中保存了 instance memory queue，则绘制历史 tracking boxes 与中心轨迹。
        self.draw_track_pred(result)

        # (3) 绘制每个检测目标概率最高的若干 future motion modes。
        self.draw_motion_pred(result)
#     
        # (4) 绘制通过地图置信度阈值的预测 vectors。
        self.draw_map_pred(result)

        # (5) 绘制与当前 GT command 对应的若干 planning candidate modes。
        self.draw_planning_pred(data, result)

        # (6) Prediction 图与 GT 图使用相同的自车、command 和 legend，便于肉眼对齐比较。
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()

        # (7) 保存 Prediction BEV 图。
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_pred)

        # (8) 返回路径给 Visualizer.combine()，而不是返回图片数组，减少调用方的接口复杂度。
        return save_path_gt, save_path_pred

    # -------------------------------------------------------------------------
    # 2.4 save_fig：无边距保存当前 Matplotlib 画布
    # -------------------------------------------------------------------------
    def save_fig(
        self,       # 当前 BEVRender 实例。
        filename,   # str：目标 jpg 文件的完整保存路径。
    ):
        """
        作用：去除 Matplotlib 默认边距，并将当前 figure 保存为图片。

        参数：
        filename：待保存图片的完整路径。
        """
        # 将子图撑满整个 figure；若不设置，Matplotlib 会保留白色边缘，影响后续拼接。
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0,
            hspace=0, wspace=0,
        )

        # 取消 Matplotlib 默认坐标轴外边距。
        plt.margins(0, 0)

        # 保存当前 figure；原实现不显式指定 dpi，因此使用 Matplotlib 的默认 dpi。
        plt.savefig(filename)

    # -------------------------------------------------------------------------
    # 2.5 draw_detection_gt：绘制 GT 3D boxes 的俯视轮廓与方向线
    # -------------------------------------------------------------------------
    def draw_detection_gt(
        self,   # 当前 BEVRender 实例。
        data,   # dict：至少包含 gt_labels_3d 与 gt_bboxes_3d。
    ):
        """
        作用：在 BEV 平面绘制当前帧的 3D Detection GT。

        输入字段：
        (1) data['gt_labels_3d']：shape 通常为 [N_gt]；-1 表示 padding / ignore 的无效框。
        (2) data['gt_bboxes_3d']：shape 通常为 [N_gt, 7]，格式为 [x, y, z, w, l, h, yaw]。

        说明：
        GT 没有在此处使用 track instance_id，因此颜色按当前帧目标下标 i 分配，只保证“同一张 GT 图内”易于区分。
        """
        # (1) 若 visualization 配置不显示 Detection，则整个函数直接退出。
        if not self.plot_choices['det']:
            return

        # (2) 逐个遍历当前帧的 GT 3D box。
        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]  # 当前 GT 的类别标签。

            # (2.1) -1 是无效/忽略标签，不应绘制。
            if label == -1:
                continue

            # (2.2) GT 没有使用 tracking id，这里用 i 在调色板中循环取色。
            color = color_mapping[i % len(color_mapping)]

            # --------------------- (a) 绘制 3D box 的地面矩形 ---------------------
            # box3d_to_corners(...) 输出 [N_gt, 8, 3]；索引 [0, 3, 7, 4, 0] 取 z 较低平面的四个角并闭合。
            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]

            # 仅在 BEV 中使用前两维 x、y；z 维在鸟瞰视角下不显示。
            x = corners[:, 0]
            y = corners[:, 1]

            # 将四边形画为实线框。
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # ---------------------- (b) 绘制用于识别 yaw 的方向线 -------------------
            # selected corners 的第 2、3 个点构成代码定义的“前侧边”；其均值代表该边中心。
            forward_center = np.mean(corners[2:4], axis=0)

            # 前四个不重复角点的均值即 box 的平面中心。
            center = np.mean(corners[0:4], axis=0)

            # 用“前侧边中心 -> box 中心”的短线直观显示 box yaw；这不是模型额外预测的向量。
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    # -------------------------------------------------------------------------
    # 2.6 draw_detection_pred：绘制预测 3D boxes 的俯视轮廓与方向线
    # -------------------------------------------------------------------------
    def draw_detection_pred(
        self,   # 当前 BEVRender 实例。
        result, # dict：至少包含 boxes_3d / labels_3d / scores_3d / instance_ids。
    ):
        """
        作用：在 BEV 平面绘制置信度达到 SCORE_THRESH 的预测 3D boxes。

        输入字段：
        (1) result['boxes_3d']：shape 通常为 [N_pred, 7]，格式为 [x, y, z, w, l, h, yaw]。
        (2) result['labels_3d']：shape [N_pred]，本函数仅用于确定遍历数量。
        (3) result['scores_3d']：shape [N_pred]，用于过滤低置信度目标。
        (4) result['instance_ids']：shape [N_pred]，用于使同一 tracking instance 保持稳定颜色。
        """
        # (1) 必须同时满足：绘制预测、开启 Detection、结果包含 boxes_3d。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['det']
            and "boxes_3d" in result
        ):
            return

        # (2) 缓存 box 张量/数组，避免在循环内反复从字典读取。
        bboxes = result['boxes_3d']

        # (3) 逐一绘制 prediction slots 中保留下来的目标。
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]  # 当前预测框的分类/质量分数。

            # (3.1) 低分预测直接跳过。
            if score < SCORE_THRESH:
                continue

            # (3.2) 预测颜色依赖 instance_id，而非当前列表下标 i；这是 tracking 结果的视觉标识。
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

            # --------------------- (a) 绘制当前帧预测 box 的地面矩形 ----------------
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # ---------------------- (b) 绘制当前预测 box 的方向线 ---------------------
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    # -------------------------------------------------------------------------
    # 2.7 draw_track_pred：绘制 InstanceBank 缓存的历史 box 与中心轨迹
    # -------------------------------------------------------------------------
    def draw_track_pred(
        self,   # 当前 BEVRender 实例。
        result, # dict：至少包含 anchor_queue / period，以及当前帧 boxes_3d 等字段。
    ):
        """
        作用：可视化 SparseDrive Tracking 的 instance memory queue。

        输入字段：
        (1) result['anchor_queue']：通常可理解为 [N_instance, T_cache, box_dim]；
            其中每一个历史 anchor 已在上游时间对齐到当前帧坐标系，因而此处可直接画到当前 BEV。
        (2) result['period']：shape 通常为 [N_instance]；每个当前目标可用的历史缓存帧数。
        (3) result['boxes_3d']：当前帧的预测 boxes，用作历史中心折线的起点。

        说明：
        该函数不再做 pose transform；时间对齐是 InstanceBank / temporal 模块在前面完成的。
        """
        # (1) 只有 draw_pred、track 开关均开启，且 result 保存了 anchor_queue 时才绘制。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['track']
            and "anchor_queue" in result
        ):
            return

        # (2) 读取历史 anchor 序列、每个目标的有效历史长度和当前预测 boxes。
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']

        # (3) 遍历当前帧预测目标。
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]

            # (3.1) 与 Detection / Motion 保持一致：低分目标不显示 tracking history。
            if score < SCORE_THRESH:
                continue

            # (3.2) 追踪历史与当前 box 共享同一个 instance_id 颜色。
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

            # (3.3) 先将当前 box 的 [x, y, z] 作为中心轨迹的第一个点。
            center = bboxes[i, :3]
            centers = [center]

            # --------------------- (a) 按“最近 -> 更早”绘制历史缓存 ----------------
            # period[i] 表示该 instance 真正有效的历史槽位数量；-1-j 从 queue 尾部倒序读取。
            for j in range(period[i]):
                # temp_bboxes[:, -1-j] 取出第 j 个历史时刻的全部 instances，随后索引 i 取当前 instance。
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]

                # (a.1) 画历史 box 的地面矩形；线宽 2 小于当前 box 的 3，以区分“历史”与“当前”。
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # (a.2) 画历史 box 的方向线。
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # (a.3) 记录历史 box 中心，后面将所有中心连接成折线。
                centers.append(center)

            # --------------------------- (b) 绘制中心连线 ---------------------------
            # 将 [当前中心, 最近历史中心, ..., 更早历史中心] 堆叠成 [T_valid+1, 3]。
            centers = np.stack(centers)

            # BEV 只需要中心的 x、y。
            xs = centers[:, 0]
            ys = centers[:, 1]

            # 连线显示该 instance 在缓存窗口内的空间位置变化；其方向是“当前 -> 过去”。
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    # -------------------------------------------------------------------------
    # 2.8 draw_motion_gt：绘制周围 agent 的 GT future trajectory
    # -------------------------------------------------------------------------
    def draw_motion_gt(
        self,   # 当前 BEVRender 实例。
        data,   # dict：至少包含 detection GT、gt_agent_fut_masks、gt_agent_fut_trajs。
    ):
        """
        作用：绘制每个 GT 目标的未来运动轨迹。

        输入字段：
        (1) gt_agent_fut_masks：shape 通常为 [N_gt, T_future]，指示每个未来时刻是否有有效 GT；
        (2) gt_agent_fut_trajs：shape 通常为 [N_gt, T_future, 2]，这里按“相邻时刻位移增量”解释；
        (3) gt_bboxes_3d[:, :2]：当前时刻每个 agent 的平面起点。

        核心计算：
        future absolute position = current center + cumsum(relative displacement)。
        """
        # (1) 若 Motion 开关关闭，不绘制任何 agent future GT。
        if not self.plot_choices['motion']:
            return

        # (2) 遍历 GT instances；其索引与 gt_agent_fut_* 的第 0 维一一对应。
        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]

            # (2.1) 无效 GT 目标直接跳过。
            if label == -1:
                continue

            # (2.2) GT 颜色按当前帧目标下标 i 循环取色。
            color = color_mapping[i % len(color_mapping)]

            # (2.3) 这些硬编码标签被绘制得更大，以强化车辆/大交通参与者的轨迹可见性。
            #       精确类别含义依赖当前数据集配置；本函数只按标签数字决定点大小。
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            # (2.4) 当前 3D box 中心的平面坐标，是未来轨迹的积分起点。
            center = data['gt_bboxes_3d'][i, :2]

            # (2.5) 将未来有效位 mask 转成 bool，便于 numpy 高级索引。
            masks = data['gt_agent_fut_masks'][i].astype(bool)

            # (2.6) 若第一个未来时刻就无 GT，则该 agent 没有可用 future trajectory，直接跳过。
            if masks[0] == 0:
                continue

            # (2.7) 仅取有效 future steps；这里 trajs 表示相邻 future step 的二维位移增量。
            trajs = data['gt_agent_fut_trajs'][i][masks]

            # (2.8) 对位移累加后加上当前中心，得到当前自车坐标系中的未来绝对位置。
            trajs = trajs.cumsum(axis=0) + center

            # (2.9) 在序列首部补入当前中心，使轨迹从“当前时刻”自然开始。
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            # (2.10) winter colormap 的轨迹点从一种颜色渐变到另一种颜色；GT score 固定为 1，不做淡化。
            self._render_traj(
                trajs,
                traj_score=1.0,
                colormap='winter',
                dot_size=dot_size,
            )

    # -------------------------------------------------------------------------
    # 2.9 draw_motion_pred：绘制每个预测目标的 Top-K motion modes
    # -------------------------------------------------------------------------
    def draw_motion_pred(
        self,       # 当前 BEVRender 实例。
        result,     # dict：至少包含 boxes_3d / labels_3d / scores_3d / trajs_3d / trajs_score。
        top_k=3,    # int：每个目标显示的轨迹模式数；按 trajectory score 从高到低选择。
    ):
        """
        作用：绘制每个高置信度检测目标的 Top-K 预测未来轨迹。

        输入字段：
        (1) result['trajs_3d'][i]：shape 通常为 [num_modes, T_future, 2]；
            从本函数的拼接方式可知，渲染器将其视为“当前帧坐标系中的未来绝对 x-y 位置”。
        (2) result['trajs_score'][i]：shape [num_modes]，通常是各 trajectory mode 的未归一化分数/logit。

        可视化策略：
        (a) 根据 score 降序选择 Top-K；
        (b) score 越低，颜色越接近白色；
        (c) 按“低分先画、高分后画”的顺序叠加，避免高分轨迹被低分轨迹遮住。
        """
        # (1) 同时要求：绘制预测、开启 Motion、结果中确实存在 trajs_3d。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['motion']
            and "trajs_3d" in result
        ):
            return

        # (2) 获取当前预测 boxes 与类别标签；后者仅用于调节点大小。
        bboxes = result['boxes_3d']
        labels = result['labels_3d']

        # (3) 遍历每个当前帧预测目标。
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]

            # (3.1) 低置信度检测框对应的 motion prediction 不绘制。
            if score < SCORE_THRESH:
                continue

            # (3.2) 按类别使用较大或较小的轨迹散点。
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            # (3.3) 将 PyTorch CPU tensor 转为 numpy，后续使用 numpy 排序与拼接。
            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()

            # (3.4) num_modes 是当前 agent 的多模态轨迹候选数量。
            num_modes = len(traj_score)

            # (3.5) 为每个 mode 复制一份当前目标 x-y 中心，shape 为 [num_modes, 1, 2]。
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()

            # (3.6) 在每个 mode 的轨迹开头拼入当前中心，形成“当前点 + future points”。
            traj = np.concatenate([center, traj], axis=1)

            # (3.7) 按轨迹分数从高到低排序。
            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]

            # (3.8) 最高分 mode 的 exp(score) 用作归一化基准，使最高分的显示透明度/饱和度为 1。
            norm_score = np.exp(sorted_score[0])

            # (3.9) 倒序画 Top-K：低分轨迹先画，高分轨迹后画，视觉层级更清楚。
            #       前提是 num_modes >= top_k；SparseDrive 默认候选模式数通常满足这一条件。
            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]  # 当前要绘制的第 j 个高分轨迹。

                # exp(score_j) / exp(score_top1) ∈ (0, 1]，用于将低分轨迹混白、淡化。
                traj_score = np.exp(sorted_score[j]) / norm_score

                # 使用 winter colormap 绘制 agent motion；点大小由类别决定。
                self._render_traj(
                    viz_traj,
                    traj_score=traj_score,
                    colormap='winter',
                    dot_size=dot_size,
                )

    # -------------------------------------------------------------------------
    # 2.10 draw_map_gt：绘制 vectorized online map GT
    # -------------------------------------------------------------------------
    def draw_map_gt(
        self,   # 当前 BEVRender 实例。
        data,   # dict：至少包含 map_infos。
    ):
        """
        作用：绘制当前帧的 vector map Ground Truth。

        输入字段：
        data['map_infos']：字典，通常为 {label: [vector_0, vector_1, ...]}；
        每条 vector 的 shape 通常为 [num_points, D]，前两维为当前自车坐标系中的 x、y。
        """
        # (1) Map 开关关闭时直接返回。
        if not self.plot_choices['map']:
            return

        # (2) 读取按 map 类别组织的 GT vector 字典。
        vectors = data['map_infos']

        # (3) 遍历每一个 map label 和该 label 下的 polyline 列表。
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]  # 标签索引决定颜色。

            # (3.1) 每一条 vector 是一段 polyline，例如车道线/边界线等。
            for vector in vector_list:
                pts = vector[:, :2]  # 只取 x、y；可能存在的额外维度不参与 BEV 显示。

                # (3.2) 将点序列拆成 Matplotlib plot 所需的 x 数组与 y 数组。
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])

                # (3.3) 用实线连接相邻采样点，同时绘制圆形 marker，体现 vector 的离散采样结构。
                self.axes.plot(
                    x, y,
                    color=color,
                    linewidth=3,
                    marker='o',
                    linestyle='-',
                    markersize=7,
                )

    # -------------------------------------------------------------------------
    # 2.11 draw_map_pred：绘制预测 vector map
    # -------------------------------------------------------------------------
    def draw_map_pred(
        self,     # 当前 BEVRender 实例。
        result,   # dict：至少包含 scores / labels / vectors。
    ):
        """
        作用：绘制置信度超过 MAP_SCORE_THRESH 的预测 map vectors。

        输入字段：
        (1) result['scores']：shape [N_vector]，各预测 vector 的置信度；
        (2) result['labels']：shape [N_vector]，vector 类别索引；
        (3) result['vectors']：shape 通常为 [N_vector, num_points, 2]。
        """
        # (1) 必须开启 Prediction 和 Map，且输出中包含 vectors。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['map']
            and "vectors" in result
        ):
            return

        # (2) 逐条遍历预测 vector。
        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]

            # (2.1) 过滤低置信度 vector。
            if score < MAP_SCORE_THRESH:
                continue

            # (2.2) 根据 map label 选择显示颜色。
            color = COLOR_VECTORS[result['labels'][i]]

            # (2.3) vectors[i] 通常已经是 [num_points, 2] 的绝对 BEV 坐标点序列。
            pts = result['vectors'][i]
            x = pts[:, 0]
            y = pts[:, 1]

            # (2.4) 原代码使用 plt.plot 而不是 self.axes.plot；在当前图只有一个 axes 时二者等价。
            plt.plot(
                x, y,
                color=color,
                linewidth=3,
                marker='o',
                linestyle='-',
                markersize=7,
            )

    # -------------------------------------------------------------------------
    # 2.12 draw_planning_gt：绘制自车未来轨迹 GT
    # -------------------------------------------------------------------------
    def draw_planning_gt(
        self,   # 当前 BEVRender 实例。
        data,   # dict：至少包含 gt_ego_fut_masks / gt_ego_fut_trajs / gt_ego_fut_cmd。
    ):
        """
        作用：绘制自车未来 Planning Ground Truth。

        输入字段：
        (1) gt_ego_fut_masks：shape 通常为 [T_future]，表示每个未来 step 是否有效；
        (2) gt_ego_fut_trajs：shape 通常为 [T_future, 2]，在这里按相邻时刻的二维位移增量解释；
        (3) gt_ego_fut_cmd：当前样本的离散驾驶指令，用于其它函数显示文字；本函数中读出后未继续使用。
        """
        # (1) Planning 开关关闭时直接返回。
        if not self.plot_choices['planning']:
            return

        # --------------------------- (a) 读取有效 future GT --------------------------
        masks = data['gt_ego_fut_masks'].astype(bool)  # 转 bool 以便按有效时间步索引。

        # (a.1) 首个未来时刻有效，才认为该帧存在可绘制的自车 future trajectory。
        if masks[0] != 0:
            # (a.2) 取所有有效的相对位移增量。
            plan_traj = data['gt_ego_fut_trajs'][masks]

            # (a.3) 原代码读取 command，但该局部变量后续未使用；保留以维持原始源码行为。
            cmd = data['gt_ego_fut_cmd']

            # (a.4) 将绝对值极小的数值置零，避免浮点噪声在图中形成极细小偏移。
            plan_traj[abs(plan_traj) < 0.01] = 0.0

            # (a.5) 对逐步位移进行累积，得到以当前 ego 原点为起点的 future absolute positions。
            plan_traj = plan_traj.cumsum(axis=0)

            # (a.6) 在序列开头补 [0, 0]，显式表示自车当前时刻位于 BEV 原点。
            plan_traj = np.concatenate(
                (np.zeros((1, plan_traj.shape[1])), plan_traj),
                axis=0,
            )

            # (a.7) autumn colormap 专用于 ego planning，使其与 agent motion 的 winter 色系可区分。
            self._render_traj(
                plan_traj,
                traj_score=1.0,
                colormap='autumn',
                dot_size=50,
            )

    # -------------------------------------------------------------------------
    # 2.13 draw_planning_pred：绘制 tracking ego history 与 Top-K planning modes
    # -------------------------------------------------------------------------
    def draw_planning_pred(
        self,       # 当前 BEVRender 实例。
        data,       # dict：用于读取当前样本的 gt_ego_fut_cmd，选择要展示的 planning command branch。
        result,     # dict：至少包含 planning / planning_score；可选包含 ego_anchor_queue / ego_period。
        top_k=3,    # int：在选定 command branch 内显示的 planning modes 数量。
    ):
        """
        作用：绘制自车的历史 tracking boxes（若存在）和预测 Planning Top-K 轨迹。

        输入字段：
        (1) result['planning']：通常为 [num_cmd, num_mode, T_future, 2]；
        (2) result['planning_score']：通常为 [num_cmd, num_mode]；
        (3) data['gt_ego_fut_cmd']：用于选择对应 GT command 的预测分支。

        关键说明：
        本函数选择的是“GT command 对应的 planning branch”，而不是在三种 command 中再次做 argmax 选择。
        因此它更适合检查某一 GT 驾驶意图下候选轨迹的质量；它不等价于 final_planning 的最终推理展示。
        """
        # (1) 绘制 prediction planning 的必要条件。
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['planning']
            and "planning" in result
        ):
            return

        # ===================== (a) 可选：绘制自车历史 ego anchors ====================
        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            # ego_anchor_queue 的第 0 个 instance 是自车对应的 ego anchor。
            ego_temp_bboxes = result["ego_anchor_queue"]

            # ego_period[0] 是自车可用的历史缓存长度。
            ego_period = result["ego_period"]

            # 逐个绘制从最近到更早的自车历史 box。
            for j in range(ego_period[0]):
                # (a.1) 取当前 ego 在第 j 个历史时刻的 bottom-face corners。
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]

                # (a.2) 历史 ego box 使用 mediumseagreen；细线强调其是历史信息。
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                # (a.3) 继续画方向线，显示历史 ego yaw。
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

        # 原仓库遗留的断点代码；保持注释状态，不会执行。
        # import ipdb; ipdb.set_trace()

        # ======================= (b) 读取 multi-command planning ======================
        # (b.1) 将 CPU tensor 转为 numpy；预期 shape 为 [num_cmd, num_mode, T_future, 2]。
        plan_trajs = result['planning'].cpu().numpy()

        # (b.2) num_cmd 应为 3，与 CMD_LIST 的三个离散 command 对应。
        num_cmd = len(CMD_LIST)

        # (b.3) 第 1 维为每个 command 分支下的多模态 trajectory 候选数。
        num_mode = plan_trajs.shape[1]

        # (b.4) 为每个 command / mode 在序列开头补 [0, 0]，即当前 ego 原点。
        #       此处没有 cumsum，意味着该渲染器将 result['planning'] 视为已经在 ego 坐标系表达的 future positions。
        plan_trajs = np.concatenate(
            (np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs),
            axis=2,
        )

        # (b.5) 每个 command 分支内各 mode 的 prediction score。
        plan_score = result['planning_score'].cpu().numpy()

        # ===================== (c) 按当前样本 GT command 选择分支 ====================
        # cmd 是 0/1/2，对应 Turn Right / Turn Left / Go Straight。
        cmd = data['gt_ego_fut_cmd'].argmax()

        # 只保留对应 command 的 [num_mode, T_future+1, 2] 轨迹候选。
        plan_trajs = plan_trajs[cmd]

        # 只保留该 command 下的 [num_mode] 分数。
        plan_score = plan_score[cmd]

        # ========================= (d) 排序并绘制 Top-K =============================
        # (d.1) 根据 planning score 从高到低排序。
        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]

        # (d.2) 以 Top-1 的 exp(score) 为基准，将其它轨迹分数压到 (0, 1]。
        norm_score = np.exp(sorted_score[0])

        # (d.3) 仍按低分先画、高分后画，保证最高分 planning mode 最醒目。
        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score

            # autumn colormap 表示 ego planning。
            self._render_traj(
                viz_traj,
                traj_score=traj_score,
                colormap='autumn',
                dot_size=50,
            )

    # -------------------------------------------------------------------------
    # 2.14 _render_traj：对折线轨迹做线性插值，并以渐变散点绘制
    # -------------------------------------------------------------------------
    def _render_traj(
        self,                 # 当前 BEVRender 实例。
        future_traj,          # ndarray，shape [T, 2]；第 0 点通常是当前时刻，后续点是 future positions。
        traj_score=1,         # float：(0, 1]；越小，轨迹颜色越向白色淡化。
        colormap='winter',    # str：Matplotlib colormap 名称；winter 用于 motion，autumn 用于 planning。
        points_per_step=20,   # int：相邻离散轨迹点之间插入的线性采样密度。
        dot_size=25,          # float/int：Matplotlib scatter 的点面积。
    ):
        """
        作用：把离散 future trajectory 插值为密集散点，并用颜色渐变表达时间与模式置信度。

        参数：
        (1) future_traj：轨迹点序列；相邻点之间按直线插值，不做样条或运动学平滑。
        (2) traj_score：用于将颜色与白色混合；不是重新计算的真实概率。
        (3) colormap：决定沿时间方向的颜色变化。
        (4) points_per_step：插值密度，越大则轨迹越平滑但绘制开销越高。
        (5) dot_size：每个密集散点的面积。
        """
        # (1) T 个离散点共有 T-1 段；每段采样 points_per_step 个点，最后额外补真实终点。
        total_steps = (len(future_traj) - 1) * points_per_step + 1

        # (2) 在 [0, 1] 上均匀采样 colormap，取前三个 RGB 通道，忽略可能存在的 alpha 通道。
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps)
        )[:, :3]

        # (3) 低置信度轨迹向白色混合：score=1 时保留原色，score 越小颜色越浅。
        dot_colors = (
            dot_colors * traj_score
            + (1 - traj_score) * np.ones_like(dot_colors)
        )

        # (4) 预分配插值后的平面坐标数组，shape 为 [total_steps, 2]。
        total_xy = np.zeros((total_steps, 2))

        # (5) 对每一段 future_traj[k] -> future_traj[k+1] 做线性插值。
        for i in range(total_steps - 1):
            # 当前密集采样点属于第 i // points_per_step 段；下面保留原始下标表达式。
            # 该段从起点到终点的二维向量。
            unit_vec = future_traj[i // points_per_step + 1] - future_traj[i // points_per_step]

            # 段内比例为 [0, 1)；例如 points_per_step=20 时依次为 0, 0.05, ..., 0.95。
            # 线性插值：p = p_start + ratio * (p_end - p_start)。
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]

        # (6) 显式写入真实终点，避免最后一点只停在最后一段的 0.95 位置。
        total_xy[-1] = future_traj[-1]

        # (7) 用渐变散点展示密集轨迹；不连线，轨迹视觉上呈现连续点带。
        self.axes.scatter(
            total_xy[:, 0],
            total_xy[:, 1],
            c=dot_colors,
            s=dot_size,
        )

    # -------------------------------------------------------------------------
    # 2.15 _render_sdc_car：在原点覆盖自车 PNG
    # -------------------------------------------------------------------------
    def _render_sdc_car(self):
        """
        作用：读取 resources/sdc_car.png，并固定贴到 BEV 原点附近作为自车图标。

        注意：
        图片路径是相对路径，因此通常应从 SparseDrive 项目根目录启动 visualize.py；
        否则 cv2.imread() 可能读不到资源文件并导致 cvtColor() 报错。
        """
        # (1) OpenCV 读取的三通道图片默认是 BGR 排列。
        sdc_car_png = cv2.imread('resources/sdc_car.png')

        # (2) 转为 RGB，避免 Matplotlib 显示成红蓝颠倒的颜色。
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)

        # (3) 将 PNG 放到 x∈[-1,1]、y∈[-2,2] 的固定区域，中心对应当前自车原点。
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

        # (4) 设置较高 z-order，确保车图标覆盖在轨迹和 box 之上。
        im.set_zorder(2)

    # -------------------------------------------------------------------------
    # 2.16 _render_legend：在右下角覆盖图例 PNG
    # -------------------------------------------------------------------------
    def _render_legend(self):
        """
        作用：读取 resources/legend.png，并放到 BEV 图的右下区域。

        注意：
        legend 位置写死为 x∈[15,40]、y∈[-40,-30]，因此只有 xlim/ylim 足够大时图例才完整可见。
        """
        # (1) 读取 BGR 格式图例图片
        legend = cv2.imread('resources/legend.png')

        # (2) 转为 RGB，供 Matplotlib 正确显示
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

        # (3) 将图例放在默认 40 m BEV 范围的右下角
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    # -------------------------------------------------------------------------
    # 2.17 _render_command：在左下角显示 GT 离散驾驶指令
    # -------------------------------------------------------------------------
    def _render_command(
        self, # 当前 BEVRender 实例。
        data, # dict：至少包含 gt_ego_fut_cmd。
    ):
        """
        作用：读取当前样本 GT command，并以大字体写在 BEV 左下角。

        输入字段：
        data['gt_ego_fut_cmd']：长度为 3 的 one-hot / multi-hot 命令向量；argmax() 选择最大元素索引。
        """
        # (1) 得到 command 索引，通常 0/1/2 分别对应右转/左转/直行。
        cmd = data['gt_ego_fut_cmd'].argmax()

        # (2) 在固定 BEV 坐标 (-38, -38) 标注命令文字；默认 xlim/ylim=40 时位于左下角。
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)
