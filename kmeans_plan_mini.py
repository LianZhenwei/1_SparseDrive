import os
import mmcv
import numpy as np
from sklearn.cluster import KMeans


def build_fallback_anchors():
    """
    如果 mini 数据中完全提取不到可用 ego future trajectory，
    就生成一组简单的兜底 planning anchors。
    这只用于跑通流程，不用于论文复现。
    输出形状: [3, 6, 6, 2]
    3: left / right / straight
    6: 每个 command 下 6 条候选轨迹
    6: 未来 6 个时间步
    2: x, y
    """
    anchors = np.zeros((3, 6, 6, 2), dtype=np.float32)

    t = np.arange(1, 7, dtype=np.float32)

    # 6 种不同前进距离
    forward_scales = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)

    for cmd in range(3):
        for k, scale in enumerate(forward_scales):
            y = t * scale * 0.5

            if cmd == 0:      # left
                x = -0.08 * t * t
            elif cmd == 1:    # right
                x = 0.08 * t * t
            else:             # straight
                x = np.zeros_like(t)

            anchors[cmd, k, :, 0] = x
            anchors[cmd, k, :, 1] = y

    return anchors


def main():
    os.makedirs("data/kmeans", exist_ok=True)

    K = 6
    files = [
        "data/infos/nuscenes_infos_train.pkl",
        "data/infos/nuscenes_infos_val.pkl",
    ]

    navi_trajs = [[], [], []]
    all_trajs = []

    for fp in files:
        if not os.path.exists(fp):
            print(f"[WARN] pkl not found, skip: {fp}")
            continue

        print(f"[INFO] loading: {fp}")
        data = mmcv.load(fp)
        data_infos = sorted(data["infos"], key=lambda e: e["timestamp"])

        for info in data_infos:
            if "gt_ego_fut_trajs" not in info:
                continue

            traj = np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32)
            mask = np.asarray(info.get("gt_ego_fut_masks", np.ones((6, 2))), dtype=np.float32)
            cmd_raw = np.asarray(info.get("gt_ego_fut_cmd", [0, 0, 1]))

            # navigation command: 0 left, 1 right, 2 straight
            if cmd_raw.ndim == 0:
                cmd = int(cmd_raw)
            else:
                cmd = int(cmd_raw.argmax(axis=-1))
            cmd = max(0, min(2, cmd))

            # traj 通常是 [6, 2]，表示每一步的相对增量
            traj = traj.reshape(-1, 2)

            if traj.shape[0] == 0:
                continue

            # mask 兼容 [6] 或 [6, 2]
            if mask.ndim == 2:
                valid = mask.all(axis=-1).astype(bool)
            else:
                valid = mask.astype(bool)

            valid = valid[:traj.shape[0]]

            if valid.sum() == 0:
                continue

            # 只取有效部分
            valid_traj = traj[:len(valid)][valid]

            # 如果不足 6 步，则用最后一步补齐
            if valid_traj.shape[0] < 6:
                pad_num = 6 - valid_traj.shape[0]
                pad = np.repeat(valid_traj[-1:], pad_num, axis=0)
                valid_traj = np.concatenate([valid_traj, pad], axis=0)
            else:
                valid_traj = valid_traj[:6]

            # SparseDrive 原始 kmeans_plan.py 里会做 cumsum
            plan_traj = valid_traj.cumsum(axis=0).reshape(1, 6, 2)

            navi_trajs[cmd].append(plan_traj)
            all_trajs.append(plan_traj)

    print("[INFO] valid traj count by command:", [len(x) for x in navi_trajs])
    print("[INFO] valid traj count total:", len(all_trajs))

    # 如果 mini 完全没有可用轨迹，生成兜底 anchors
    if len(all_trajs) == 0:
        print("[WARN] No valid ego planning trajectories found.")
        print("[WARN] Use handcrafted fallback anchors for mini debug only.")
        anchors = build_fallback_anchors()
        np.save("data/kmeans/kmeans_plan_6.npy", anchors)
        print("[DONE] saved data/kmeans/kmeans_plan_6.npy", anchors.shape)
        return

    all_trajs_arr = np.concatenate(all_trajs, axis=0)
    clusters = []

    for i in range(3):
        # 如果某个 command 没有轨迹，就用所有轨迹兜底
        source = navi_trajs[i] if len(navi_trajs[i]) > 0 else [all_trajs_arr]
        arr = np.concatenate(source, axis=0).reshape(-1, 12)

        n_clusters = min(K, len(arr))

        if n_clusters == 1:
            centers = arr[:1]
        else:
            centers = KMeans(
                n_clusters=n_clusters,
                random_state=0,
                n_init=10,
            ).fit(arr).cluster_centers_

        # 如果样本数不足 6 类，复制最后一个 center 补齐
        if n_clusters < K:
            centers = np.concatenate(
                [centers, np.repeat(centers[-1:], K - n_clusters, axis=0)],
                axis=0,
            )

        clusters.append(centers.reshape(K, 6, 2))

    clusters = np.stack(clusters, axis=0).astype(np.float32)

    np.save("data/kmeans/kmeans_plan_6.npy", clusters)

    print("[DONE] saved data/kmeans/kmeans_plan_6.npy", clusters.shape)


if __name__ == "__main__":
    main()