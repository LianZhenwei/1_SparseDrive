import os                          # 导入 os 模块，用于创建输出目录
import pickle                      # 导入 pickle，当前代码中没有直接使用，属于冗余导入
from tqdm import tqdm              # 导入 tqdm，用于显示循环进度条
import numpy as np                 # 导入 numpy，用于数组处理、拼接、保存 .npy 文件等
import matplotlib.pyplot as plt    # 导入 matplotlib，用于可视化聚类中心
from sklearn.cluster import KMeans # 从 sklearn 中导入 KMeans 聚类算法，用于对 3D box 中心点进行聚类
import mmcv                        # 导入 mmcv，用于加载 pkl 文件

# 创建保存文件夹
os.makedirs('data/kmeans', exist_ok=True) # 创建 data/kmeans 目录，exist_ok=True 表示目录已经存在也不报错
os.makedirs('vis/kmeans', exist_ok=True)  # 创建 vis/kmeans 目录，用于保存聚类结果可视化图片


# (1) 参数设置
K = 900         # KMeans 聚类中心数量：detection head 中使用 900 个 3D anchors
DIS_THRESH = 55 # 距离阈值：只使用距离自车 55m 内的 GT box 中心点做聚类

# (2) 加载 pkl 数据，并按 timestamp 排序
fp = 'data/infos/nuscenes_infos_train.pkl'                             # 训练集 info 文件路径：这个文件由 nuscenes_converter.py 生成
data = mmcv.load(fp)                                                   # 加载 pkl 数据【data 通常是一个 dict，包含 data["infos"] 和 data["metadata"]】
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"])) # 按 timestamp 对所有 sample info 排序，这样后续处理顺序和时间顺序一致


# (3) 获取自车 55m 范围内的的所有 GT box 中心
# (3.1) 用于保存所有 GT box 中心和部分 box 信息
center = []

# (3.2) 遍历所有 sample，获取自车 55m 范围内的的所有 GT box 中心
for idx in tqdm(range(len(data_infos))):
    # (a) 取出当前 sample 的 GT box 的前三维【gt_boxes shape 是 [num_box, 7(x, y, z, l, w, h, yaw)]，[:, :3] 表示只取中心点 x, y, z】
    boxes = data_infos[idx]['gt_boxes'][:, :3]

    # (b) 如果当前帧没有 GT box，则跳过
    if len(boxes) == 0:
        continue # 跳过

    # (c) 计算每个 box 在 xy 平面上距离自车的距离【distance shape: [num_box]】
    distance = np.linalg.norm(boxes[:, :2], axis=1)

    # (d) 只保留自车 55m 范围内的 box 中心
    center.append(boxes[distance < DIS_THRESH])

# (3.3) 将所有帧收集到的 box 中心拼成一个大数组【center shape: [total_num_boxes, 3]】
center = np.concatenate(center, axis=0)


# (4) 对所有 box 中心点做 KMeans 聚类，并可视化
print("start clustering, may take a few minutes.")             # 打印提示
cluster = KMeans(n_clusters=K).fit(center).cluster_centers_    # 对所有 box 中心点做 KMeans 聚类，得到 900 个聚类中心【cluster shape: [900, 3]】
plt.scatter(cluster[:, 0], cluster[:, 1])                      # 可视化聚类中心的 x,y 分布
plt.savefig(f'vis/kmeans/det_anchor_{K}', bbox_inches='tight') # 保存可视化图片，文件路径类似：vis/kmeans/det_anchor_900.png


# (5) 保存聚类结果 det anchors 为 .npy 文件
# (5.1) 构造 box 其他维度的默认值
# cluster 当前只有 [x, y, z] 三维
# detection anchor 需要补成 11 维左右的状态，这里 others 是 [1,1,1,1,0,0,0,0]，通常可以理解成：默认尺寸、yaw、速度等初始化值
others = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].repeat(K, axis=0)

# (5.2) 将聚类中心和默认其他状态拼接：cluster shape 从 [900, 3] 变成 [900, 11]【即 900 个最典型的目标中心位置的 x,y,z, log(w)=1,log(l)=1,log(h)=1, sin(yaw)=1,cos(yaw)=0, vx=0,vy=0,vz=0】
cluster = np.concatenate([cluster, others], axis=1)

# (5.3) 保存 detection anchor，输出：data/kmeans/kmeans_det_900.npy
np.save(f'data/kmeans/kmeans_det_{K}.npy', cluster)

