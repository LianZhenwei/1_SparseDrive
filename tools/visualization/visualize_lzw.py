import os             # 用于路径拼接、目录创建等文件系统操作。
import glob           # 用于按照通配符批量查找 combine 目录下的 jpg 图片。
import argparse       # 用于解析命令行参数，例如 config、result-path、out-dir。
from tqdm import tqdm # 用于给循环显示进度条，方便观察可视化处理进度。
import cv2            # OpenCV：用于读取图片、横向拼接图片、写图片、合成视频。
import numpy as np    # NumPy：原文件中导入但当前 visualize.py 内没有直接使用；保留以兼容原始代码风格。
from PIL import Image # PIL Image：原文件中导入但当前 visualize.py 内没有直接使用；保留以兼容原始代码风格。

import mmcv                              # 用于 mmcv.load() 读取推理结果 pkl。
from mmcv import Config                  # 用于 Config.fromfile() 读取 .py 配置文件。
from mmdet.datasets import build_dataset # 用于根据 cfg.data.val 构建验证集 dataset。

from tools.visualization.bev_render import BEVRender # BEV 视角渲染器：画 GT 和预测结果。
from tools.visualization.cam_render import CamRender # Camera 视角渲染器：画六相机预测结果。


# 一、全局可视化开关
plot_choices = dict(
    # (1) 是否绘制预测结果。
    #     (a) True：BEV 会画 GT + Pred，Camera 会画 Pred。
    #     (b) False：通常只保留 GT 相关绘制，具体还取决于 bev_render.py / cam_render.py 内部判断。
    draw_pred=True,

    # (2) 是否绘制 3D detection。
    #     (a) BEV 中对应 3D box 鸟瞰框。
    #     (b) Camera 中对应投影到相机图像上的 3D box。
    det=True,

    # (3) 是否绘制 tracking 历史轨迹框。
    #     (a) BEV 中会使用 result["anchor_queue"] / result["period"] 等历史缓存结果。
    #     (b) Camera 文件中当前主要没有单独绘制 tracking 历史框。
    track=True,

    # (4) 是否绘制 motion prediction。
    #     (a) BEV 中绘制每个 agent 的未来轨迹。
    #     (b) Camera 中绘制投影到图像上的 agent 未来轨迹。
    motion=True,

    # (5) 是否绘制 map prediction / map GT。
    #     (a) BEV 中绘制道路元素 polyline / vector。
    #     (b) Camera 文件当前没有绘制 map。
    map=True,

    # (6) 是否绘制 ego planning。
    #     (a) BEV 中绘制自车未来规划轨迹。
    #     (b) Camera 中通常只在前视相机上绘制 final_planning。
    planning=True,
)


# 二、可视化帧范围控制
START = 0    # 起始帧 index；默认从第 0 帧开始。
END = 81     # 结束帧上界；range(START, END, INTERVAL) 不包含 END。
INTERVAL = 1 # 帧间隔；1 表示每一帧都可视化，2 表示隔一帧取一帧。


# 三、Visualizer 类：核心调度类
class Visualizer:
    # 1. 初始化：创建 vis/combine 目录、读取配置文件并构建验证集 dataset、读取模型预测结果、初始化 BEVRender 和 CamRender
    def __init__(
        self,         # 当前 Visualizer 对象自身。
        args,         # argparse.Namespace，包含 config / result_path / out_dir 等命令行参数。
        plot_choices, # dict，可视化开关，控制是否画 det / track / motion / map / planning 等。
    ):
        # (1) 创建 vis/combine 目录
        self.out_dir = args.out_dir                              # 保存总输出目录，默认是 vis
        self.combine_dir = os.path.join(self.out_dir, 'combine') # 构造 combine 子目录路径，用来保存横向拼接后的图片，这里设置为 vis/combine
        os.makedirs(self.combine_dir, exist_ok=True)             # 创建 combine 目录【即创建 vis/combine 目录】，exist_ok=True 表示目录已存在时不报错

        # (2) 从配置文件中读取完整 config
        #     (a) args.config 通常类似 projects/configs/sparsedrive_small_stage2.py。
        #     (b) Config.fromfile 会执行并解析该 .py 配置文件。
        cfg = Config.fromfile(args.config)

        # (3) 根据 cfg.data.val 构建验证集 dataset
        #     (a) dataset 提供 get_data_info(index)，可以取出某一帧的图片路径、标定参数、GT 等信息。
        #     (b) 这里不是训练，只是利用 dataset 的数据读取与标定信息组织能力。
        self.dataset = build_dataset(cfg.data.val)

        # (4) 读取模型预测结果。
        #     (a) args.result_path 通常是 tools/test.py 或 dist_test.sh 保存出的 .pkl 文件。
        #     (b) self.results[index] 对应该 index 样本的推理输出。
        self.results = mmcv.load(args.result_path)

        # (5) 初始化 BEV 视角渲染器和 Camera 视角渲染器
        self.bev_render = BEVRender(plot_choices, self.out_dir) # BEVRender 内部会创建目录 vis/bev_gt 和 vis/bev_pred
        self.cam_render = CamRender(plot_choices, self.out_dir) # CamRender 内部会创建目录 vis/cam_pred

    # 2. 对某一帧 index 样本执行完整可视化，流程是：取数据 → 取预测结果 → 画 BEV GT 和 BEV Pred → 画 Camera → 拼接这三张图拼成一张 combine 图
    def add_vis(
        self,  # 当前 Visualizer 对象自身
        index, # int，当前要可视化的数据帧下标【即当前样本在 dataset / results 中的下标，注意它需要同时能索引 self.dataset 和 self.results】
    ):
        """
            数据流说明
                (1) data：
                    (a) 来自 self.dataset.get_data_info(index)。
                    (b) 包含 img_filename、lidar2cam、cam_intrinsic、GT box、GT motion、GT map、GT planning 等字段。
                (2) result：
                    (a) 来自 self.results[index]['img_bbox']。
                    (b) 包含 boxes_3d、scores_3d、labels_3d、instance_ids、trajs_3d、planning 等预测字段。
        """
        # (1) 从 dataset 中取出第 index 帧的数据与标定信息【对可视化来说，data 不只是 GT，还包括相机图片路径和相机内外参】
        data = self.dataset.get_data_info(index)

        # (2) 从预测结果列表中取出第 index 帧的 3D 预测输出【SparseDrive / mmdet3d 的结果通常包在 ['img_bbox'] 里】
        result = self.results[index]['img_bbox']

        # (3) 调用 BEVRender 生成 BEV GT 图和 BEV Pred 图，调用 CamRender 生成六相机预测可视化图：
        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        '''
            bev_gt_path  : BEV GT     ，保存到 vis/bev_gt/xxxx.jpg
            bev_pred_path: BEV Pred   ，保存到 vis/cam_pred/xxxx.jpg
            cam_pred_path: Camera Pred，保存到 vis/bev_pred/xxxx.jpg
        '''

        # (4) 把三张图横向拼接（cam_image 在左侧、bev_image 在中间、bev_gt 在右侧），保存到 vis/combine/xxxx.jpg
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)

    # 3. 把 Camera Pred、BEV Pred、BEV GT 三张图片横向拼接成一张总览图，输出到 vis/combine 目录中，后续将由 image2video() 会读取这些 vis/combine 下的图片合成视频
    def combine(
        self,          # 当前 Visualizer 对象自身。
        bev_gt_path,   # str，BEV GT 图片路径。
        bev_pred_path, # str，BEV Pred 图片路径。
        cam_pred_path, # str，Camera Pred 图片路径。
        index,         # int，当前帧下标，用于生成保存文件名。
    ):
        # (1) 读取 BEV GT 图片、BEV Pred 图片和 Camera Pred 图片
        bev_gt = cv2.imread(bev_gt_path)      # cv2.imread 返回 BGR 格式的 numpy array
        bev_image = cv2.imread(bev_pred_path) # 原变量名 bev_image 实际表示 BEV 预测图
        cam_image = cv2.imread(cam_pred_path) # 这张图通常包含 6 个相机视角的拼图

        # (2) 横向拼接三张图，拼接顺序是 Camera Pred → BEV Pred → BEV GT
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt]) # cv2.hconcat 要求三张图片高度一致，当前 BEVRender / CamRender 的 figsize 设置就是为了让输出高度匹配

        # (3) 构造 combine 输出路径
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg') # zfill(4) 让编号固定为 4 位，方便按字符串排序时保持时间顺序

        # (4) 将拼接后的图片写入磁盘
        cv2.imwrite(save_path, merge_image)

    # 4. 将 vis/combine 目录下的 jpg 图片序列合成为 video.mp4【支持通过 downsample 降采样，减小视频分辨率和文件大小】
    def image2video(
        self,         # 当前 Visualizer 对象自身
        fps=12,       # int，输出视频帧率，每秒播放多少张图片。默认 12 fps，表示每秒播放 12 帧
        downsample=4, # int，输出视频降采样倍数。默认 4 表示宽高都缩小到原来的 1/4【原始拼接图通常比较大，降采样可以显著减小视频体积】
    ):
        """
            1. 函数作用
                (1) 扫描 self.combine_dir 目录下的所有 jpg 图片。
                (2) 按文件名排序，形成时间顺序。
                (3) 逐张读取、缩放、加入 img_array。
                (4) 使用 OpenCV VideoWriter 写成 video.mp4。

            2. 注意事项：当前实现假设 combine 目录至少有一张 jpg【如果没有图片，size 变量不会被定义，VideoWriter 会报错】
        """
        # (1) 查找 vis/combine 目录下所有 jpg 图片路径【glob 返回的顺序不保证天然有序】
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))

        # (2) 按文件名排序【因为文件名是 0000.jpg、0001.jpg，所以字符串排序就是时间顺序】
        imgs_path = sorted(imgs_path)

        # (3) 初始化图片数组列表
        img_array = [] # 后续每一张 resize 后的图都会 append 进来

        # (4) 遍历所有 vis/combine 下的图片
        for img_path in tqdm(imgs_path): # tqdm 用于显示图片读取和缩放进度
            # (4.1) 读取当前图片【img 是 H x W x C 的 BGR 数组】
            img = cv2.imread(img_path)

            # (4.2) 获取原始图片尺寸
            height, width, channel = img.shape # 图像高度、图像宽度、通道数（通常为 3）

            # (4.3) 对图片进行降采样
            img = cv2.resize(
                img, 
                (width // downsample, height // downsample), # width // downsample：缩放后的宽度，height // downsample：缩放后的高度
                interpolation=cv2.INTER_AREA                 # INTER_AREA 通常适合图像缩小
            )

            # (4.4) 重新获取缩放后的尺寸【VideoWriter 需要使用缩放后的 width 和 height】
            height, width, channel = img.shape

            # (4.5) 保存视频帧尺寸【OpenCV VideoWriter 的 size 格式是 (width, height)，不是 (height, width)，该变量会在循环结束后用于创建 VideoWriter】
            size = (width, height)

            # (4.6) 把缩放后的图加入视频帧列表
            img_array.append(img)

        # (5) 构造视频输出路径，默认输出到 vis/video.mp4
        out_path = os.path.join(self.out_dir, 'video.mp4')

        # (6) 创建 VideoWriter
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'), # mp4v 是 mp4 常用编码器之一
            fps,                             # fps 控制播放速度
            size                             # size 控制视频分辨率
        )

        # (7) 将图片数组逐帧写入视频
        for i in range(len(img_array)):
            out.write(img_array[i]) # 写入第 i 帧

        # (8) 释放 VideoWriter【必须 release，否则视频文件可能没有正确写入尾部信息】
        out.release()


# 解析命令行参数，支持用户指定 config、result-path、out-dir。
def parse_args():
    """
    1. 函数作用
    (1) 定义并解析 visualize.py 的命令行参数。
    (2) 返回 args，供 main() 创建 Visualizer 使用。

    2. 输入来源
    (1) 该函数没有显式形参。
    (2) argparse 会自动从命令行读取参数，例如：
        (a) python tools/visualization/visualize.py CONFIG --result-path RESULT.pkl --out-dir vis

    3. 返回值
    (1) args：
        (a) argparse.Namespace 对象。
        (b) args.config：配置文件路径。
        (c) args.result_path：预测结果路径。
        (d) args.out_dir：可视化输出目录。
    """
    # (1) 创建命令行参数解析器。
    #     (a) description 会显示在 --help 帮助信息中。
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results'
    )

    # (2) 添加位置参数 config。
    #     (a) 位置参数是必须提供的。
    #     (b) 用于告诉程序按哪个 SparseDrive config 构建验证集。
    parser.add_argument(
        'config',
        help='config file path'
    )

    # (3) 添加可选参数 --result-path。
    #     (a) 指向模型预测结果文件。
    #     (b) default=None 表示如果用户不传，则默认为 None。
    #     (c) 但注意：当前 Visualizer.__init__ 中会直接 mmcv.load(args.result_path)，所以实际运行通常必须提供。
    parser.add_argument(
        '--result-path',
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized'
    )

    # (4) 添加可选参数 --out-dir。
    #     (a) 指定所有可视化结果保存到哪个目录。
    #     (b) 默认保存到 vis。
    parser.add_argument(
        '--out-dir',
        default='vis',
        help='directory where visualize results will be saved'
    )

    # (5) 解析命令行参数。
    #     (a) 例如 args.config / args.result_path / args.out_dir。
    args = parser.parse_args()

    # (6) 返回解析结果。
    return args


# 脚本主入口：解析命令行参数、创建 Visualizer、逐帧可视化、合成视频。
def main():
    # (1) 解析命令行参数
    args = parse_args()

    # (2) 创建可视化总调度器【内部会读取 config、构建 dataset、读取 results、创建 BEVRender 和 CamRender】
    visualizer = Visualizer(args, plot_choices)

    # (3) 遍历要可视化的帧 index【tqdm 用于显示整体可视化进度】
    for idx in tqdm(range(START, END, INTERVAL)): 
        # (a) 如果 idx 超过结果列表长度，则停止【原代码使用 >，若希望避免 idx == len(results) 越界，通常可改成 >=】
        if idx > len(visualizer.results):
            break

        # (b) 对当前帧执行完整可视化：生成 BEV GT 和 BEV Pred → 生成 Camera Pred → 拼接成 combine 图
        visualizer.add_vis(idx)

    # (4) 将 combine 图片序列合成为 video.mp4
    visualizer.image2video()


if __name__ == '__main__':
    main()
