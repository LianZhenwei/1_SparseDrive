import os                          # 导入 os，当前代码里没有使用 os，属于冗余导入
import pickle                      # 导入 pickle，当前代码中没有使用，属于冗余导入
from tqdm import tqdm              # 导入 tqdm，用于显示进度条
import numpy as np                 # 导入 numpy，用于数组处理
import matplotlib.pyplot as plt    # 导入 matplotlib，用于可视化地图 anchor
from sklearn.cluster import KMeans # 导入 KMeans，用于聚类地图元素中心
import mmcv                        # 导入 mmcv，用于读取 pkl 文件


# (1) 参数设置：
K = 100         # map anchor 数量：map head 中使用 100 个 map anchors
num_sample = 20 # 每条地图线采样点数量：和配置文件中的 num_sample=20 对应

# (2) 加载 pkl 数据，并按 timestamp 排序
fp = 'data/infos/nuscenes_infos_train.pkl'                             # 训练集 info 文件路径
data = mmcv.load(fp)                                                   # 加载训练集 info
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"])) # 按 timestamp 排序


# (3) 获取所有地图线的中心点
# (3.1) 保存地图元素中心点
center = []

# (3.2) 遍历所有 sample，将所有地图线的中心点取出并存放进 center
for idx in tqdm(range(len(data_infos))):
    # (a) 遍历当前 sample 的地图 annotation
    for cls, geoms in data_infos[idx]["map_annos"].items():
        '''
            map_annos 是 nuscenes_converter.py 里生成的，其内容是：
                {
                    1: [array([num_points, 2]), array([num_points, 2]), ...], # 每个元素 array([num_points, 2]) 是一条二维折线的人行横道，每条人行横道是由若干 (x,y) 的点组成
                    0: [array([num_points, 2]), array([num_points, 2]), ...], # 每个元素 array([num_points, 2]) 是一条二维折线的分隔线，每条分隔线是由若干 (x,y) 的点组成
                    2: [array([num_points, 2]), array([num_points, 2]), ...]  # 每个元素 array([num_points, 2]) 是一条二维折线的边界线，每条边界线是由若干 (x,y) 的点组成
                }   
        '''
        # (b) 遍历该类别下的每条地图线，通过 .mean(axis=0) 计算这条线的中心点 [x, y]，并加入 center
        for geom in geoms:
            center.append(geom.mean(axis=0)) # geom shape 通常是 [num_points, 2]，geom.mean(axis=0) 得到这条线的中心点 [x, y]

# (3.3) 将所有地图线中心点堆叠，center shape: [total_num_lines, 2]
center = np.stack(center, axis=0)


# (4) 对所有地图线中心点做 KMeans 聚类，并在每个聚类中心加上默认竖直线，以此作为最终 map anchor，并可视化
# (4.1) 对地图线中心点做 KMeans 聚类：输出 100 个中心点，center shape: [100, 2]
center = KMeans(n_clusters=K).fit(center).cluster_centers_ 

# (4.2) 构建竖直线
delta_y = np.linspace(-4, 4, num_sample)      # 构造一个默认的竖直线形状：y 从 -4 到 4，共 20 个采样点
delta_x = np.zeros([num_sample])              # x 全部为 0
delta = np.stack([delta_x, delta_y], axis=-1) # delta shape: [20, 2]，表示一条以原点为中心、沿 y 方向展开的默认线

# (4.3) 将每个聚类中心加上这条默认竖直线
vecs = center[:, np.newaxis] + delta[np.newaxis] # [100, 20, 2]，即 100 条地图线 anchor（地图线聚类中心点+竖直线）、每条 anchor 竖直线有 20 个点、每个点是 [x, y]
'''
    center shape: [100, 2]
    center[:, np.newaxis] shape: [100, 1, 2]
    delta[np.newaxis] shape: [1, 20, 2]
    vecs shape: [100, 20, 2]，即 100 条地图线 anchor（地图线聚类中心点+竖直线）、每条 anchor 有 20 个点、每个点是 [x, y]
'''

# (4.4) 可视化每条 map anchor
for i in range(K):
    x = vecs[i, :, 0]                                                    # 第 i 条 anchor 的 x 坐标
    y = vecs[i, :, 1]                                                    # 第 i 条 anchor 的 y 坐标
    plt.plot(x, y, linewidth=1, marker='o', linestyle='-', markersize=2) # 画出这条线
plt.savefig(f'vis/kmeans/map_anchor_{K}', bbox_inches='tight')           # 保存可视化图片


# (5) 保存聚类结果 map anchors 为 .npy 文件
np.save(f'data/kmeans/kmeans_map_{K}.npy', vecs) # 输出：data/kmeans/kmeans_map_100.npy

