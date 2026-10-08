import os                          # 导入 os，当前代码中没有使用，属于冗余导入
import pickle                      # 导入 pickle，当前代码中没有使用，属于冗余导入
from tqdm import tqdm              # 导入 tqdm，用于显示进度条
import numpy as np                 # 导入 numpy，用于轨迹、矩阵、数组操作
import matplotlib.pyplot as plt    # 导入 matplotlib，用于可视化聚类轨迹
from sklearn.cluster import KMeans # 导入 KMeans，用于对未来轨迹进行聚类
import mmcv                        # 导入 mmcv，用于读取 pkl 文件


# nuScenes 3D 检测类别，共 10 个目标类别
CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


# 将 lidar 坐标系下的未来轨迹 offset 转换到 agent 自身坐标系下
def lidar2agent(trajs_offset, boxes):
    '''
        将 lidar 坐标系下的未来轨迹 offset 转换到 agent 自身坐标系下
        输入：
            trajs_offset: [num_agent, 12, 2]，每个 agent 的未来轨迹增量
            boxes: [num_agent, box_dim]，当前 agent 的 3D box，其中 boxes[:, 6] 是 yaw
            
        输出：
            trajs_new: [num_agent, 12, 2]，agent 坐标系下的未来轨迹   
    '''
    # (1) 计算他车 agent 未来轨迹相对于当前时刻的绝对坐标
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32) # 构造原点 offset，shape: [num_agent, 1, 2]
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)      # 在未来 offset 前面拼接当前原点【trajs_offset shape: [N, 12, 2]，拼接后 shape: [N, 13, 2]】
    trajs = trajs_offset.cumsum(axis=1)                                # 对 offset 做累加，得到相对于当前时刻的绝对轨迹点，trajs shape: [N, 13, 2]

    # (2) 构建旋转矩阵
    yaws = - boxes[:, 6]   # 取 box yaw 的相反数：用于从 lidar 坐标系旋转到 agent 局部坐标系
    rot_sin = np.sin(yaws) # 计算 sin
    rot_cos = np.cos(yaws) # 计算 cos
    rot_mat_T = np.stack(  # 构造旋转矩阵的转置形式，rot_mat_T shape 大致是 [2, 2, N]
        [
            np.stack([rot_cos, rot_sin]),  # 第一行：[cos, sin]
            np.stack([-rot_sin, rot_cos]), # 第二行：[-sin, cos]
        ]
    )

    # (3) 使用 einsum 对每个 agent 的轨迹做旋转，得到他车 agent 自身坐标系下的未来轨迹点坐标
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    '''    
        trajs: [N, 13, 2]
        rot_mat_T: [2, 2, N]
        输出 trajs_new: [N, 13, 2]
    '''

    # (4) 去掉第 0 个当前原点，只保留未来 12 步
    trajs_new = trajs_new[:, 1:]

    # (5) 返回未来 12 步的 agent 坐标系下的 agent 未来轨迹
    return trajs_new


# (1) 参数设置
K = 6           # 每类 motion anchor 聚类数量：fut_mode=6，所以每个类别聚类 6 个运动意图
DIS_THRESH = 55 # 距离阈值，只使用距离自车 55m 内的 agent

# (2) 加载 pkl 数据，并按 timestamp 排序
fp = 'data/infos/nuscenes_infos_train.pkl'                             # 训练集 info 文件
data = mmcv.load(fp)                                                   # 加载训练集信息
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"])) # 按时间戳 timestamp 排序


# (3) 获取 “自车 55m 范围内的所有 agent” 的 “有效未来 12 步” 的 “agent 坐标系下” 的 agent 未来轨迹
# (3.1) intention 用于按类别保存未来轨迹，intention[i] 是第 i 类目标的未来轨迹列表
intention = dict()

# (3.2) 初始化每个类别的列表
for i in range(len(CLASSES)):
    intention[i] = []

# (3.3) 遍历所有 sample，获取 “自车 55m 范围内的所有 agent” 的 “有效未来 12 步” 的 “agent 坐标系下” 的 agent 未来轨迹
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]                 # 当前 sample info
    boxes = info['gt_boxes']               # 当前帧 GT boxes
    names = info['gt_names']               # 当前帧 GT 类别名
    fut_masks = info['gt_agent_fut_masks'] # agent 未来轨迹 mask
    trajs = info['gt_agent_fut_trajs']     # agent 未来轨迹 offset
    velos = info['gt_velocity']            # agent 当前速度

    # (a) 保存类别 id
    labels = []

    # (b) 遍历每个类别名，如果类别在 CLASSES 中，则转成类别 id，否则标记为 -1
    for cat in names:
        # 如果类别在 CLASSES 中，则转成类别 id，否则标记为 -1
        if cat in CLASSES:
            labels.append(CLASSES.index(cat)) # 转成类别 id
        else:
            labels.append(-1) # 否则标记为 -1
    labels = np.array(labels) # 转成 numpy array

    # (c) 如果当前帧没有 box，则跳过
    if len(boxes) == 0:
        continue # 跳过 

    # (d) 遍历每个类别
    for i in range(len(CLASSES)):
        cls_mask = (labels == i)            # 找到当前类别的目标 mask
        box_cls = boxes[cls_mask]           # 当前类别的 boxes
        fut_masks_cls = fut_masks[cls_mask] # 当前类别的未来 mask
        trajs_cls = trajs[cls_mask]         # 当前类别的未来轨迹
        velos_cls = velos[cls_mask]         # 当前类别的速度

        # (d.1) 计算当前类别目标距离自车的 xy 平面距离
        distance = np.linalg.norm(box_cls[:, :2], axis=1)

        # (d.2) 筛选出未来 12 步全部有效、且位于自车 55m 范围内的所有 agent
        mask = np.logical_and(
            fut_masks_cls.sum(axis=1) == 12, # 必须未来 12 步全部有效
            distance < DIS_THRESH,           # 必须在 55m 范围内
        )                           # 构造筛选 mask
        trajs_cls = trajs_cls[mask] # 筛选轨迹
        box_cls = box_cls[mask]     # 筛选 box
        velos_cls = velos_cls[mask] # 筛选速度

        # (d.3) 将筛选出的 agent 的未来轨迹从 lidar 坐标系转换到 agent 坐标系
        trajs_agent = lidar2agent(trajs_cls, box_cls)

        # (d.4) 如果当前类别没有有效轨迹，则跳过，否则加入该类别的轨迹列表
        if trajs_agent.shape[0] == 0:
            continue # 跳过
        intention[i].append(trajs_agent) # 加入该类别的轨迹列表


# (4) 对所有 agent 未来轨迹点做 KMeans 聚类，并可视化
# (4.1) 保存所有类别的聚类结果
clusters = []

# (4.2) 遍历每个类别，对所有 agent 未来轨迹做 KMeans 聚类，并可视化
for i in range(len(CLASSES)):
    # (a) 将该类别所有轨迹拼接【原始每条轨迹 shape [12, 2]，reshape(-1, 24) 表示每条轨迹拉平成 24 维】
    intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 24)

    # (b) 如果该类别样本数量小于 K，则无法做 KMeans，跳过
    if intention_cls.shape[0] < K:
        continue # 无法做 KMeans，跳过

    # (c) 对该类别轨迹聚类成 K 个运动模式【cluster shape: [6, 24]】
    cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_

    # (d) reshape 回轨迹形式【cluster shape: [6, 12, 2]】
    cluster = cluster.reshape(-1, 12, 2)

    # (e) 加入总 clusters
    clusters.append(cluster)

    # (f) 可视化该类别的 6 条 motion anchor
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1]) # 画第 j 条轨迹的 12 个点

    # (g) 保存该类别可视化图
    plt.savefig(f'vis/kmeans/motion_intention_{CLASSES[i]}_{K}', bbox_inches='tight')
    plt.close() # 关闭当前图，避免不同类别画在一张图上


# (5) 保存聚类结果 motion anchors 为 .npy 文件
clusters = np.stack(clusters, axis=0)                   # 将所有类别的 motion anchors 堆叠【理想 shape: [num_classes=10, 6, 12, 2]，即10 个目标类别，每个类别 6 条典型未来轨迹，每条轨迹 12 个未来点，每个点是 [x, y]；注意，如果某些类别有效轨迹数量不足 6，代码会 continue，这可能导致最终类别数小于 10】
np.save(f'data/kmeans/kmeans_motion_{K}.npy', clusters) # 保存 motion anchors，输出：data/kmeans/kmeans_motion_6.npy
