# 深拷贝工具，用于投影矩阵等数据的安全复制
import copy
# OpenCV库，用于图像绘制
import cv2
# 数值计算库
import numpy as np
# PyTorch核心库
import torch
# 导入3D框各维度的索引常量（W/L/H/YAW等）
from projects.mmdet3d_plugin.core.box3d import *


def box3d_to_corners(box3d):
    """
    将3D框参数转换为8个角点的全局坐标
    3D框格式：[x, y, z, w, l, h, yaw]，航向角绕z轴旋转
    Args:
        box3d (Tensor/ndarray): 3D框参数，形状 [N, 7]
    Returns:
        corners (ndarray): 8个角点坐标，形状 [N, 8, 3]，角点顺序为底面4点+顶面4点
    """
    # 1. 输入格式兼容：tensor转numpy，且断开梯度、移到CPU
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    
    # 2. 生成归一化8角点坐标：用unravel_index生成2x2x2立方体的8个顶点索引，值为0或1
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    # 调整角点顺序，保证底面、顶面按顺时针排列，方便后续绘制
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # 归一化坐标平移到以框中心为原点（范围从0~1变为-0.5~0.5）
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    
    # 3. 乘以框的长宽高，得到实际尺寸的角点（相对中心）
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])
    
    # 4. 构造绕z轴的旋转矩阵，对每个框做航向旋转
    # 计算航向角的余弦值
    rot_cos = np.cos(box3d[:, YAW])
    # 计算航向角的正弦值
    rot_sin = np.sin(box3d[:, YAW])
    # 初始化单位旋转矩阵，批量复制N份
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    # 填充旋转矩阵第一行
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    # 填充旋转矩阵第二行
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    # z轴保持不变，旋转矩阵第三行为[0,0,1]
    
    # 5. 批量矩阵乘法：旋转每个角点
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    
    # 6. 加上框的中心坐标，得到全局坐标系下的最终角点
    corners += box3d[:, None, :3]
    
    return corners


def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """
    在2D图像上绘制3D框的12条边界棱线
    Args:
        img (ndarray): 输入图像，形状 [H, W, 3]
        num_rects (int): 3D框的数量
        rect_corners (ndarray): 3D框投影后的像素角点，形状 [num_rect, 8, 2]
        color (tuple): 绘制颜色，BGR格式；也可以是列表，每个框对应一个颜色
        thickness (int): 线条粗细
    Returns:
        img (ndarray): 绘制后的图像
    """
    # 1. 定义立方体12条棱的端点索引，对应8个角点的连接关系
    line_indices = (
        (0, 1), (0, 3), (0, 4),  # 底面顶点连接的三条棱
        (1, 2), (1, 5),          # 底面与侧面棱
        (3, 2), (3, 7),          # 底面与侧面棱
        (4, 5), (4, 7),          # 顶面棱
        (2, 6), (5, 6), (6, 7),  # 剩余侧面棱
    )
    
    # 2. 获取图像高宽，用于边界判断
    h, w = img.shape[:2]
    
    # 3. 遍历每个3D框逐一绘制
    for i in range(num_rects):
        # 取出当前框的8个角点像素坐标，裁剪数值范围避免溢出，转整型
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        
        # 遍历每条棱，绘制线段
        for start, end in line_indices:
            # 判断：如果两个端点都在图像外，则跳过绘制，减少无效操作
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            
            # 统一颜色：单颜色直接用，多颜色取当前框对应的颜色
            if isinstance(color[0], int):
                # 绘制抗锯齿线段
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )
    
    # 返回绘制后的图像，确保为uint8格式
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """
    将激光雷达坐标系下的3D框投影到2D图像上并绘制
    核心流程：3D框转8角点 → 齐次坐标投影 → 像素坐标转换 → 绘制
    Args:
        bboxes3d (Tensor/ndarray): 激光雷达坐标系下的3D框，形状 [N, 7]
        raw_img (ndarray): 原始图像
        lidar2img_rt (ndarray/Tensor): 激光雷达到图像的投影矩阵，形状 [4, 4]
        img_metas (dict): 图像元数据，预留参数
        color (tuple): 绘制颜色
        thickness (int): 线条粗细
    Returns:
        ndarray: 绘制后的图像
    """
    # 1. 拷贝原始图像，避免修改原图
    img = raw_img.copy()
    
    # 2. 将3D框转换为8个3D角点坐标
    corners_3d = box3d_to_corners(bboxes3d)
    # 获取框的数量
    num_bbox = corners_3d.shape[0]
    
    # 3. 构造齐次坐标：3D点补充1维，形状变为 [N*8, 4]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    
    # 4. 投影矩阵深拷贝，reshape为4x4
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    # tensor转numpy
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    
    # 5. 投影计算：齐次点 @ 投影矩阵转置，得到图像平面齐次坐标
    pts_2d = pts_4d @ lidar2img_rt.T
    
    # 6. 深度值裁剪，避免除零和异常值
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    # 除以深度，得到像素x坐标
    pts_2d[:, 0] /= pts_2d[:, 2]
    # 除以深度，得到像素y坐标
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    # 7. 重塑为 [N, 8, 2] 的角点像素格式
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    
    # 8. 调用绘制函数，在图像上画出3D框
    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    """
    将激光雷达点云投影到2D图像上，绘制为圆点
    Args:
        points (Tensor): 点云坐标，形状 [N, 3] 或 [N, 4]
        img (ndarray): 原始图像
        lidar2img_rt (ndarray/Tensor): 激光雷达到图像的投影矩阵，4x4
        color (tuple): 点的颜色，支持单颜色和每个点不同颜色
        circle (int): 圆点半径
    Returns:
        ndarray: 绘制后的图像
    """
    # 1. 拷贝原图
    img = img.copy()
    # 获取点的数量
    N = points.shape[0]
    
    # 2. 点云转numpy数组
    points = points.cpu().numpy()
    
    # 3. 投影矩阵深拷贝并格式化为4x4
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    # tensor转numpy
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    
    # 4. 点云投影：旋转 + 平移
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    
    # 5. 深度裁剪，避免除零
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    # 除以深度得到像素坐标
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    
    # 6. 坐标范围裁剪，转整型
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)
    
    # 7. 逐点绘制圆点
    for i in range(N):
        for point in pts_2d[i]:
            # 颜色处理：单颜色直接用，多颜色取当前点对应颜色
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            # 绘制实心圆（thickness=-1表示填充）
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
    """
    在BEV鸟瞰图上绘制3D框的底面轮廓，同时绘制刻度圈和坐标轴
    BEV坐标系：图像中心为自车原点，x向右对应lidar x轴，y向下对应lidar -y轴
    Args:
        bboxes_3d (Tensor/ndarray): 3D框参数，形状 [N, 7]
        bev_size (int/tuple): BEV图像的尺寸，单值表示正方形，tuple为(高, 宽)
        bev_range (float): BEV对应的物理范围，单位米，默认前后左右各115/2米
        color (tuple): 框的颜色，支持单颜色和每个框不同颜色
        thickness (int): 线条粗细
    Returns:
        bev (ndarray): 绘制后的BEV图像
    """
    # 1. 处理BEV图像尺寸
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h = bev_w = bev_size
    
    # 初始化黑色BEV背景图
    bev = np.zeros([bev_h, bev_w, 3])
    
    # 2. 刻度与辅助线颜色（灰色）
    marking_color = (127, 127, 127)
    # 计算BEV分辨率：每个像素对应多少米
    bev_resolution = bev_range / bev_h
    
    # 3. 绘制距离刻度圈，每10米一圈
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),  # 圆心在图像中心
            int((cir + 1) * 10 / bev_resolution), # 半径像素值
            marking_color,
            thickness=thickness,
        )
    
    # 4. 绘制水平坐标轴（x轴）
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    # 绘制垂直坐标轴（y轴）
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    
    # 5. 绘制3D框的底面
    if len(bboxes_3d) != 0:
        # (1) 3D框转8角点，取出底面4个角点(0,3,4,7)，只取xy坐标
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][..., [0, 1]]
        
        # (2) 物理坐标转BEV像素坐标
        # x坐标：缩放 + 中心偏移
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        # y坐标：取反（因为图像y向下，lidar y向前）+ 中心偏移
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        
        # (3) 遍历每个目标，绘制底面四边形的四条边
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                # 颜色处理
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                # 绘制边
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    
    # 返回绘制完成的BEV图
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    """
    可视化总入口：生成多视角图像+BEV的拼接可视化结果
    自动处理多视角图像拼接，左侧放BEV图，右侧放多视角图像
    Args:
        bboxes_3d (Tensor/ndarray): 3D检测框
        imgs (list[ndarray]): 多视角图像列表
        lidar2imgs (list[ndarray]): 每个视角对应的投影矩阵列表
        color (tuple): 绘制颜色
    Returns:
        vis_imgs (ndarray): 拼接后的完整可视化图像
    """
    # 1. 逐个视角绘制3D框
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )
    
    # 2. 多视角图像拼接
    num_imgs = len(vis_imgs)
    # 视角少于4个或数量为奇数，横向拼接成一行
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    # 视角为偶数且≥4，拼接成两行两列
    else:
        vis_imgs = np.concatenate([
            # 上半部分：前半段视角横向拼接
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),
            # 下半部分：后半段视角横向拼接
            np.concatenate(vis_imgs[num_imgs//2:], axis=1)
        ], axis=0)
    
    # 3. 生成BEV图，高度与多视图拼接后的图像高度一致
    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    
    # 4. BEV图在左，多视角图像在右，横向拼接
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    
    # 返回最终可视化结果
    return vis_imgs

