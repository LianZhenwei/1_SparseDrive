'''
见《2.6.3_可视化高层驾驶指令各自6种anchor的代码2.md》
'''

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


K = 6

# SparseDrive 的 gt_ego_fut_cmd 顺序：
# [1, 0, 0] -> Turn Right
# [0, 1, 0] -> Turn Left
# [0, 0, 1] -> Go Straight
CMD_NAMES = ["turn_right", "turn_left", "go_straight"]
CMD_TITLES = ["Turn Right", "Turn Left", "Go Straight"]


def plot_one_command(cluster, cmd_id, out_dir, k):
    """可视化单个 high-level command 下的 K 条 planning anchors.

    Args:
        cluster: shape = (K, T, 2)，当前 command 下的 K 条轨迹 anchor。
        cmd_id: 0/1/2，分别对应 right/left/straight。
        out_dir: 图片保存目录。
        k: 每个 command 下的 anchor 数量。
    """
    plt.figure(figsize=(7, 6))

    for mode_id in range(k):
        x = cluster[mode_id, :, 0]
        y = cluster[mode_id, :, 1]

        # scatter 看每个未来时刻的采样点
        plt.scatter(x, y, label=f"mode_{mode_id}")

        # plot 把采样点连成轨迹，更容易看出形状
        plt.plot(x, y, linewidth=1)

        # 起点附近加一个淡淡的标记，表示都是从 ego 当前位姿附近出发
        plt.scatter([x[0]], [y[0]], marker="x", s=50)

    plt.title(f"Planning anchors: {CMD_TITLES[cmd_id]} ({k} modes)")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)

    save_path = os.path.join(out_dir, f"plan_{k}_{cmd_id}_{CMD_NAMES[cmd_id]}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved: {save_path}")


def plot_all_commands(clusters, out_dir, k):
    """把 3 个 command 的 anchor 画到同一张总览图中."""
    plt.figure(figsize=(8, 7))

    for cmd_id in range(3):
        cluster = clusters[cmd_id]
        for mode_id in range(k):
            x = cluster[mode_id, :, 0]
            y = cluster[mode_id, :, 1]
            plt.scatter(x, y, label=f"{CMD_NAMES[cmd_id]}_mode_{mode_id}")
            plt.plot(x, y, linewidth=1)

    plt.title(f"Planning anchors: all commands (3 x {k} modes)")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best", fontsize=6, ncol=2)

    save_path = os.path.join(out_dir, f"plan_{k}_all_commands.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved: {save_path}")


def plot_all_commands_subplots(clusters, out_dir, k):
    """把 3 个 command 分成 3 个子图，便于横向对比."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

    for cmd_id, ax in enumerate(axes):
        cluster = clusters[cmd_id]
        for mode_id in range(k):
            x = cluster[mode_id, :, 0]
            y = cluster[mode_id, :, 1]
            ax.scatter(x, y, label=f"mode_{mode_id}")
            ax.plot(x, y, linewidth=1)

        ax.set_title(f"{CMD_TITLES[cmd_id]} ({k} modes)")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="best", fontsize=7)

    save_path = os.path.join(out_dir, f"plan_{k}_three_commands_subplots.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SparseDrive planning anchors by high-level command."
    )
    parser.add_argument(
        "--anchor",
        default=f"data/kmeans/kmeans_plan_{K}.npy",
        help="Path to kmeans_plan_6.npy. Expected shape: (3, 6, 6, 2).",
    )
    parser.add_argument(
        "--out-dir",
        default="vis/kmeans/vis_plan_anchor_by_cmd",
        help="Directory to save visualization images.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.anchor):
        raise FileNotFoundError(
            f"Cannot find {args.anchor}. "
            "这个脚本只负责可视化已有 planning anchors。"
            "如果你还没有 kmeans_plan_6.npy，请用完整 nuScenes train 数据重新聚类，"
            "mini 数据可能缺少某些 command，例如 turn_right。"
        )

    clusters = np.load(args.anchor)
    print("loaded:", args.anchor)
    print("planning anchor shape:", clusters.shape)

    expected_shape = (3, K, 6, 2)
    if clusters.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, but got {clusters.shape}. "
            "请确认加载的是 SparseDrive 的 kmeans_plan_6.npy。"
        )

    # 分别保存 3 张图：每个 high-level command 各 6 条 anchor
    for cmd_id in range(3):
        plot_one_command(clusters[cmd_id], cmd_id, args.out_dir, K)

    # 保存 1 张总览图：18 条全部放一起
    plot_all_commands(clusters, args.out_dir, K)

    # 保存 1 张三子图总览：更适合对比 right/left/straight
    plot_all_commands_subplots(clusters, args.out_dir, K)


if __name__ == "__main__":
    main()
