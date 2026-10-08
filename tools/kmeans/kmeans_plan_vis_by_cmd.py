'''
见《2.6.2_可视化高层驾驶指令各自6种anchor的代码1.md》
'''
import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 6

# SparseDrive 的 gt_ego_fut_cmd 顺序：
# [1, 0, 0] -> Turn Right
# [0, 1, 0] -> Turn Left
# [0, 0, 1] -> Go Straight
CMD_NAMES = ["turn_right", "turn_left", "go_straight"]
CMD_TITLES = ["Turn Right", "Turn Left", "Go Straight"]

os.makedirs("data/kmeans", exist_ok=True)
os.makedirs("vis/kmeans", exist_ok=True)

fp = "data/infos/nuscenes_infos_train.pkl"
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

# navi_trajs[0]：右转样本
# navi_trajs[1]：左转样本
# navi_trajs[2]：直行样本
navi_trajs = [[], [], []]

for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]

    # gt_ego_fut_trajs 是 6 个未来时刻的位移增量；
    # cumsum 后变成从当前时刻出发的累计未来轨迹点，shape = (6, 2)
    plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
    plan_mask = info["gt_ego_fut_masks"]

    # gt_ego_fut_cmd 是 one-hot: right / left / straight
    cmd = info["gt_ego_fut_cmd"].astype(np.int32)
    cmd = cmd.argmax(axis=-1)

    # 只使用未来 6 帧都有效的样本
    if not plan_mask.sum() == 6:
        continue

    navi_trajs[cmd].append(plan_traj)

clusters = []

# 分别对 3 个 high-level command 各自聚类、各自可视化
for cmd_id, trajs in enumerate(navi_trajs):
    if len(trajs) < K:
        raise ValueError(
            f"{CMD_NAMES[cmd_id]} only has {len(trajs)} valid trajectories, "
            f"which is fewer than K={K}."
        )

    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)

    # 1. 单独保存该 command 下的 6 条 planning anchors
    plt.figure()
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :, 1], label=f"mode_{j}")
        plt.plot(cluster[j, :, 0], cluster[j, :, 1], linewidth=1)

    plt.title(f"Planning anchors: {CMD_TITLES[cmd_id]} ({K} modes)")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.savefig(
        f"vis/kmeans/plan_{K}_{cmd_id}_{CMD_NAMES[cmd_id]}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

# 2. 额外保存一张总览图：3 个 command 各 6 条，共 18 条
plt.figure()
for cmd_id, cluster in enumerate(clusters):
    for j in range(K):
        plt.scatter(
            cluster[j, :, 0],
            cluster[j, :, 1],
            label=f"{CMD_NAMES[cmd_id]}_mode_{j}",
        )
        plt.plot(cluster[j, :, 0], cluster[j, :, 1], linewidth=1)

plt.title(f"Planning anchors: all commands (3 x {K} modes)")
plt.xlabel("x / m")
plt.ylabel("y / m")
plt.axis("equal")
plt.grid(True)
plt.legend(loc="best", fontsize=6, ncol=2)
plt.savefig(f"vis/kmeans/plan_{K}_all_commands.png", bbox_inches="tight", dpi=200)
plt.close()

clusters = np.stack(clusters, axis=0)
print("planning anchor shape:", clusters.shape)  # expected: (3, 6, 6, 2)
np.save(f"data/kmeans/kmeans_plan_{K}.npy", clusters)
