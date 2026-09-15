: '
    这个脚本的作用就是一次性生成 SparseDrive 配置文件里需要的 4 类 anchor：
        data/kmeans/
        ├── kmeans_det_900.npy       # 3D 检测 anchor
        ├── kmeans_map_100.npy       # 地图线 anchor
        ├── kmeans_motion_6.npy      # agent 运动预测轨迹 anchor
        └── kmeans_plan_6.npy        # ego 自车规划轨迹 anchor

    这些 .npy 文件后面会在配置文件里被加载。例如：
        anchor="data/kmeans/kmeans_det_900.npy"
        anchor="data/kmeans/kmeans_map_100.npy"
        motion_anchor=f"data/kmeans/kmeans_motion_{fut_mode}.npy"
        plan_anchor=f"data/kmeans/kmeans_plan_{ego_fut_mode}.npy"
'



# 运行检测任务的 KMeans 聚类脚本
# 作用：根据训练集中所有 3D box 的中心点，聚类生成 900 个 detection anchors
# 输出：data/kmeans/kmeans_det_900.npy
# Shape：[900, 11]【即 900 个最典型的目标中心位置的 x,y,z, log(w)=1,log(l)=1,log(h)=1, sin(yaw)=1,cos(yaw)=0, vx=0,vy=0,vz=0】【只聚类自车 55m 范围内的 box 中心】
python tools/kmeans/kmeans_det.py


# 运行地图任务的 KMeans 聚类脚本
# 作用：根据训练集中局部地图元素的中心点，聚类生成 100 个 map anchors
# 输出：data/kmeans/kmeans_map_100.npy
# Shape：[100, 20, 2]，即 100 条地图线 anchor（地图线聚类中心点+竖直线）、每条 anchor 竖直线有 20 个点、每个点是 [x, y]
python tools/kmeans/kmeans_map.py


# 运行运动预测任务的 KMeans 聚类脚本
# 作用：根据训练集中其他交通参与者的未来轨迹，聚类生成 motion trajectory anchors
# 输出：data/kmeans/kmeans_motion_6.npy
# Shape：理想为 [num_classes=10, 6, 12, 2]，即 10 个目标类别，每个类别 6 条典型未来轨迹，每条轨迹 12 个未来点，每个点是 [x, y]
# 补充：只聚类未来 12 步都有效、且距离自车 55m 内的目标；另外，已把轨迹点坐标从 lidar 坐标系转到 agent 自身坐标系下；注意，如果某些类别有效轨迹数量不足 6，代码会 continue，这可能导致最终类别数小于 10
python tools/kmeans/kmeans_motion.py


# 运行自车规划任务的 KMeans 聚类脚本
# 作用：根据训练集中 ego 自车未来轨迹，按导航命令聚类生成 planning anchors
# 输出：data/kmeans/kmeans_plan_6.npy
# Shape：[3, 6, 6, 2]【3 类导航命令（右转、左转、直行），每类命令 6 条典型自车未来轨迹，每条轨迹 6 个未来点，每个点是 [x, y]】
python tools/kmeans/kmeans_plan.py
