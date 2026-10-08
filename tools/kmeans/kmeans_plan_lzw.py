import os                          # 导入 os，当前代码中没有使用，属于冗余导入
import pickle                      # 导入 pickle，当前代码中没有使用，属于冗余导入
from tqdm import tqdm              # 导入 tqdm，用于显示进度条
import numpy as np                 # 导入 numpy，用于轨迹处理
import matplotlib.pyplot as plt    # 导入 matplotlib，用于可视化规划 anchor
from sklearn.cluster import KMeans # 导入 KMeans，用于对 ego 未来轨迹聚类
import mmcv                        # 导入 mmcv，用于读取 pkl 文件


# (1) 每个导航命令下的规划 anchor 数
K = 6 # ego_fut_mode=6

# (2) 加载 pkl 数据，并按 timestamp 排序
fp = 'data/infos/nuscenes_infos_train.pkl'                             # 训练集 info 文件
data = mmcv.load(fp)                                                   # 加载训练集信息
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"])) # 按时间戳排序

# (3) 按导航命令保存 ego 未来轨迹
# 这里有 3 类命令：
# 0: Turn Right
# 1: Turn Left
# 2: Go Straight
navi_trajs = [[], [], []]


# (4) 获取按高层命令的自车未来轨迹的绝对坐标
# 遍历所有 sample
for idx in tqdm(range(len(data_infos))):
    # (a) 读取
    info = data_infos[idx]                               # 当前 sample info
    plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2) # 读取 ego 未来轨迹 offset，并累加成相对于当前帧的绝对轨迹。gt_ego_fut_trajs shape: [6, 2]，cumsum(axis=-2) 后还是 [6, 2]
    plan_mask = info['gt_ego_fut_masks']                 # 读取 ego 未来轨迹有效 mask
    cmd = info['gt_ego_fut_cmd'].astype(np.int32)        # 读取 ego 高层命令 one-hot：例如 [1,0,0] 右转，[0,1,0] 左转，[0,0,1] 直行

    # (b) one-hot 转类别 id：cmd 取值为 0、1、2
    cmd = cmd.argmax(axis=-1)

    # (c) 如果未来 6 步不是全部有效，则跳过，否则按命令类别保存轨迹
    if not plan_mask.sum() == 6:
        continue # 跳过
    navi_trajs[cmd].append(plan_traj) # 按命令类别保存轨迹


# (5) 对3个高层命令下的自车未来轨迹点做 KMeans 聚类，并可视化
# (5.1) 保存三个命令下的聚类结果
clusters = []

# (5.2) 遍历三类导航命令的轨迹列表
for trajs in navi_trajs:
    # (a) 拼接该命令下所有轨迹
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12) # 每条轨迹 shape: [6, 2]，reshape(-1, 12) 表示每条轨迹拉平成 12 维

    # (b) KMeans 聚类成 6 条规划 anchor
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_ # KMeans 聚类成 6 条规划 anchor，cluster shape: [6, 12]
    cluster = cluster.reshape(-1, 6, 2)                        # reshape 回轨迹形式，cluster shape: [6, 6, 2]

    # (c) 保存当前命令下的 6 条规划 anchor
    clusters.append(cluster)

    # (d) 视化当前命令下的 6 条轨迹，每条轨迹话 6 个点
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1]) # 画第 j 条轨迹的 6 个点
 
# (e) 保存所有导航命令的规划 anchor 可视化
plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
plt.close() # 关闭图像
 

# (6) 保存聚类结果 motion anchors 为 .npy 文件
clusters = np.stack(clusters, axis=0)                 # 堆叠三个导航命令的规划 anchor，clusters shape: [3, 6, 6, 2]【3 类导航命令，每类命令 6 条典型自车轨迹，每条轨迹 6 个未来点，每个点是 [x, y]】
np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters) # 保存 planning anchors，输出：data/kmeans/kmeans_plan_6.npy
