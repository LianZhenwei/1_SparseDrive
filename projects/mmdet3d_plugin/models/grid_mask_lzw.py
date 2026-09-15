'''
    1. grid_mask.py 文件实现的是 GridMask 图像增强：随机在图像上生成网格状遮挡区域，让模型不要过度依赖局部纹理，从而提升泛化能力。

    2. GridMask 是一种图像数据增强方法。它不是随机遮挡一个矩形，而是遮挡网格状区域。直观效果见《grid_mask.md》。它的目的不是让图像变好看，而是强迫模型不要只依赖某些局部纹理或局部像素，提高鲁棒性。

    3. 类Grid 和 类GridMask 的区别：
        (1) 文件 grid_mask.py 里有两个类：
            class Grid(object)
            class GridMask(nn.Module)
        (2) 区别是：
                (i) Grid：
                        普通 Python 数据增强类
                        输入 img, label
                        输出 img, label
                        更适合数据 pipeline 里使用
                (ii) GridMask：
                        PyTorch nn.Module
                        输入 x
                        输出 x
                        更适合直接放在模型 forward 里使用
        (3) SparseDrive 用的是第二个类即GridMask: self.grid_mask = GridMask(...)
'''

import torch          # 导入 PyTorch：用于张量运算，例如 torch.from_numpy、expand_as、view 等
import torch.nn as nn # 导入 PyTorch 神经网络模块：GridMask 继承自 nn.Module，可以作为模型中的一层使用
import numpy as np    # 导入 numpy：用于随机数生成、mask 数组构造等
from PIL import Image # 从 PIL 导入 Image：用于把 numpy mask 转成图像，并进行旋转操作


'''
    Grid类 是一个普通 Python callable 类，它不是 nn.Module，通常用于数据 pipeline 级别的数据增强
    Grid类的 __call__(img, label)函数 会同时返回增强后的 img 和 label
'''
class Grid(object):
    # 一、__init__()函数：初始化 GridMask 的参数
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        # 1. use_h：是否生成横向遮挡条纹
        self.use_h = use_h

        # 2. use_w：是否生成纵向遮挡条纹
        self.use_w = use_w

        # 3. rotate：mask 随机旋转角度的上限
        # 后面 r = np.random.randint(self.rotate)
        # 若 rotate=1，则 r 只能是 0，也就是不旋转
        self.rotate = rotate

        # 4. offset：是否用随机噪声填充被遮挡区域
        # False 时，遮挡区域直接置 0
        self.offset = offset

        # 5. ratio：遮挡条纹宽度占网格间距 d 的比例
        # 例如 ratio=0.5 表示每个周期内大约遮挡一半
        self.ratio = ratio

        # 6. mode：为 mask 模式
        # mode=0：遮挡网格线区域，即 mask 中 0 的地方被遮挡
        # mode=1：反转 mask（mode=1时会执行 mask = 1 - mask，相当于反过来遮挡），即 mask 中 1 的地方被遮挡、mask 中 0 的地方被保留，即反过来遮挡
        self.mode = mode

        # 7. st_prob：初始设定的最大概率
        # 后续的 set_prob()函数 会基于它做线性增长
        self.st_prob = prob

        # 8. prob：当前实际使用概率
        self.prob = prob


    # 二、set_prob()函数：根据训练 epoch 动态调整 GridMask 的使用概率
    def set_prob(self, epoch, max_epoch):
        # 根据当前 epoch 动态调整 GridMask 使用概率
        # 训练初期概率小，训练后期逐渐增大（训练越往后，GridMask 概率越接近 st_prob）
        self.prob = self.st_prob * epoch / max_epoch


    # 三、__call__()函数：对输入图像进行 GridMask 增强
    def __call__(self, img, label):
        '''
            __call__()函数：让 Grid 对象可以像函数一样调用
            __call__()传参：
                img 通常是单张图像 Tensor，形状大致为 [C, H, W]
                label 是对应标签，这里不改变 label，只原样返回        
        '''

        # 1. 生成一个 [0,1) 的随机数
        # 若随机数大于当前概率 self.prob，则不使用 GridMask，直接返回原图和原标签：
        if np.random.rand() > self.prob:
            return img, label # 直接返回原图和原标签

        # 2.1 获取图像高度h：
        h = img.size(1) # img.size(1) 对应 H，因为 img 形状通常为 [C, H, W]

        # 2.2 获取图像宽度q：
        w = img.size(2) # img.size(2) 对应 W

        # 2.3 设置网格周期 d 的最小值d1：
        self.d1 = 2 # 网格周期 d 的最小值设置为 2，表示至少每隔 2 个像素就有一个遮挡条纹

        # 2.4 设置网格周期 d 的最大值d2：
        self.d2 = min(h, w) # 取 h 和 w 的较小值。网格周期 d 的最大值设置为图像较小边长，确保至少有一个完整的周期

        # 3.1 构造一个比原图更大的 mask 高度 hh，便于旋转后中心裁剪：这里 mask 的高度设置为原图的 1.5 倍
        # 用 1.5 倍大小是为了旋转后再中心裁剪，避免边缘空缺
        hh = int(1.5 * h)

        # 3.2 构造一个比原图更大的 mask 宽度 ww，便于旋转后中心裁剪：这里 mask 的宽度设置为原图的 1.5 倍
        ww = int(1.5 * w)

        # 3.3 随机采样网格周期 d：从 [d1, d2) 范围内随机选择一个整数作为网格周期
        # d 越大，遮挡越稀（即网格越稀疏）
        # d 越小，遮挡越密（即网格越密集）
        d = np.random.randint(self.d1, self.d2)

        # 4. 根据 ratio 计算遮挡长度 l：
        # (1) 若 ratio 等于 1
        if self.ratio == 1:
            # 遮挡长度 l 在 [1, d) 之间随机
            self.l = np.random.randint(1, d)

        # (2) 若 ratio 不等于 1
        else:
            # 根据 ratio 计算遮挡长度 l
            # int(d * ratio + 0.5) 相当于四舍五入
            # min/max 确保 l 至少为 1，最多为 d-1
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # 5. 初始化 mask:
        # 1 表示保留，0 表示遮挡
        # shape 为 [hh, ww]
        # 初始全 1，表示默认区域保留
        mask = np.ones((hh, ww), np.float32)

        # 6.1 横向条纹遮挡的随机起始偏移
        st_h = np.random.randint(d)

        # 6.2 纵向条纹遮挡的随机起始偏移
        st_w = np.random.randint(d)

        # 7.1 若启用横向遮挡，则每隔 d 个像素生成一条横向遮挡带：
        if self.use_h:
            # 每隔 d 个像素生成一条横向遮挡带
            for i in range(hh // d):
                # 当前遮挡带起点
                s = d * i + st_h

                # 当前遮挡带终点
                # 最大不超过 hh
                t = min(s + self.l, hh)

                # 横向区域置 0：将 [s:t, :] 区域置 0
                # 即横向整行区域被遮挡
                mask[s:t, :] *= 0

        # 7.2 若启用纵向遮挡，则每隔 d 个像素生成一条纵向遮挡带：
        if self.use_w:
            # 每隔 d 个像素生成一条纵向遮挡带
            for i in range(ww // d):
                # 当前遮挡带起点
                s = d * i + st_w

                # 当前遮挡带终点
                # 最大不超过 ww
                t = min(s + self.l, ww)

                # 纵向区域置 0：将 [:, s:t] 区域置 0
                # 即纵向整列区域被遮挡
                mask[:, s:t] *= 0

        # 8. 获取随机旋转角度r，范围是 [0, rotate)：
        # rotate=1 时不旋转：若 rotate=1，则 np.random.randint(1) 只能得到 0
        # rotate=360 时随机 0 到 359 度：若 rotate=360，则随机旋转 0 到 359 度
        r = np.random.randint(self.rotate)

        # 9.1 将 numpy mask 转成 PIL Image
        # np.uint8(mask) 会把 0/1 转成 8-bit 图像
        mask = Image.fromarray(np.uint8(mask))

        # 9.2 对 mask 旋转 r 度
        mask = mask.rotate(r)

        # 9.3 再从 PIL Image 转回 numpy array
        mask = np.asarray(mask)

        # 9.4 中心裁剪回原图大小 [h, w]：从较大的 mask 中心裁剪出原图大小 [h, w]：
        mask = mask[
            # 高度裁剪：高度方向裁剪起点到终点
            (hh - h) // 2 : (hh - h) // 2 + h,

            # 宽度裁剪：宽度方向裁剪起点到终点
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        # 9.5 将 numpy mask 转成 torch Tensor：
        # shape 为 [H, W]
        mask = torch.from_numpy(mask).float()

        # 10 若 mode=1 则反转 mask：
        if self.mode == 1:
            # 反转 mask：原来保留的地方变遮挡，原来遮挡的地方变保留
            mask = 1 - mask

        # 11. 将 mask 扩展成和 img 一样的形状：
        # img 通常是 [C, H, W]
        # mask 原来是 [H, W]，expand_as 后变成 [C, H, W]
        mask = mask.expand_as(img)

        # 12.1 若 offset=True 则用随机噪声填充遮挡区域，否则直接置 0：
        if self.offset:
            # 生成随机 offset 噪声
            # np.random.rand(h, w) 范围是 [0,1)
            # 2 * (... - 0.5) 范围约为 [-1,1)
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()

            # 只在被遮挡区域保留 offset
            # mask=1 的区域 offset 为 0
            # mask=0 的区域 offset 为随机噪声
            offset = (1 - mask) * offset

            # 原图保留区域乘 mask
            # 遮挡区域填充 offset
            img = img * mask + offset

        # 12.2 若 offset=False，则直接用 mask 乘图像，即遮挡区域直接置0、保留区域保持原值：
        else:
            # 直接用 mask 乘图像
            # mask=0 的地方变成 0
            # mask=1 的地方保持原值
            img = img * mask

        # 13. 返回增强后的图像和原标签
        return img, label


'''
    GridMask类 是 Grid类的 nn.Module 版本，可以直接放在模型 forward() 里使用
    因此 SparseDrive 的 sparsedrive.py 里使用的就是这个GridMask类：
        self.grid_mask = GridMask(...)
        img = self.grid_mask(img)
'''
class GridMask(nn.Module):
    # 一、__init__()函数：初始化 GridMask 的参数
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__() # 调用 nn.Module 的初始化函数

        # 1. use_h：是否生成横向遮挡条纹
        self.use_h = use_h

        # 2. use_w：是否生成纵向遮挡条纹
        self.use_w = use_w

        # 3. rotate：mask 随机旋转角度的上限
        # 后面 r = np.random.randint(self.rotate)
        # 若 rotate=1，则 r 只能是 0，也就是不旋转
        self.rotate = rotate

        # 4. offset：是否用随机噪声填充被遮挡区域
        # False 时，遮挡区域直接置 0
        self.offset = offset

        # 5. ratio：遮挡条纹宽度占网格间距 d 的比例
        # 例如 ratio=0.5 表示每个周期内大约遮挡一半
        self.ratio = ratio

        # 6. mode：为 mask 模式
        # mode=0：遮挡网格线区域，即 mask 中 0 的地方被遮挡
        # mode=1：反转 mask（mode=1时会执行 mask = 1 - mask，相当于反过来遮挡），即 mask 中 1 的地方被遮挡、mask 中 0 的地方被保留，即反过来遮挡
        self.mode = mode

        # 7. st_prob：初始设定的最大概率
        # 后续的 set_prob()函数 会基于它做线性增长
        self.st_prob = prob

        # 8. prob：当前实际使用概率
        self.prob = prob


    # 二、set_prob()函数：根据训练 epoch 动态调整 GridMask 的使用概率
    def set_prob(self, epoch, max_epoch):
        # 根据训练 epoch 动态调整 GridMask 使用概率
        # 训练初期概率小，训练后期逐渐增大（训练越往后，GridMask 概率越接近 st_prob）
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5


    # 三、forward()函数：对输入图像x进行 GridMask 增强
    def forward(self, x):
        '''
        forward()函数：GridMask 前向传播
        forward()传参：
            x 通常是 batch 图像张量
            在 SparseDrive 中，多相机图像会先 flatten 成 [B*num_cams, C, H, W]
            所以这里 x 的形状通常为 [N, C, H, W]
        '''

        # 1. 生成一个 [0,1) 的随机数，若随机数大于 prob 或当前不是训练模式，则不使用 GridMask，直接返回原输入 x：
        if np.random.rand() > self.prob or not self.training:
            # 不做 GridMask，直接返回原输入
            # 这意味着 eval/test 模式不会使用 GridMask
            return x

        # 2.1 获取输入张量形状：
        n, c, h, w = x.size()
        '''
            n：batch 数量，可能是 B*num_cams
            c：通道数，通常为 3
            h：图像高度
            w：图像宽度     
        '''

        # 2.2 将 [n, c, h, w] reshape 成 [n*c, h, w]
        # 这样每个通道都单独应用同一个二维 mask 的 expand 逻辑
        x = x.view(-1, h, w)

        # 3.1 构造一个比原图更大的 mask 高度 hh，便于旋转后中心裁剪：这里 mask 的高度设置为原图的 1.5 倍
        # 用 1.5 倍大小是为了旋转后再中心裁剪，避免边缘空缺
        hh = int(1.5 * h)

        # 3.2 构造一个比原图更大的 mask 宽度 ww，便于旋转后中心裁剪：这里 mask 的宽度设置为原图的 1.5 倍
        ww = int(1.5 * w)

        # 3.3 随机采样网格周期 d：从 [2, h) 范围内随机选择一个整数作为网格周期
        # d 越大，遮挡越稀（即网格越稀疏）
        # d 越小，遮挡越密（即网格越密集）
        d = np.random.randint(2, h)

        # 4. 根据 ratio 计算遮挡长度 l：l 至少为 1，最多为 d-1
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        '''
            int(d * ratio + 0.5) 相当于四舍五入
            min/max 确保 l 至少为 1，最多为 d-1
        '''

        # 5. 初始化大 mask
        # 1 表示保留，0 表示遮挡
        # shape 为 [hh, ww]
        # 初始全 1，表示默认区域保留
        mask = np.ones((hh, ww), np.float32)

        # 6.1 横向条纹遮挡的随机起始偏移
        st_h = np.random.randint(d)

        # 6.2 纵向条纹遮挡的随机起始偏移
        st_w = np.random.randint(d)

        # 7.1 若启用横向遮挡，则每隔 d 个像素生成一条横向遮挡带：
        if self.use_h:
            # 每隔 d 个像素生成一条横向遮挡带
            for i in range(hh // d):
                # 当前遮挡带起点
                s = d * i + st_h

                # 当前遮挡带终点
                # 最大不超过 hh
                t = min(s + self.l, hh)

                # 横向区域置0：将 [s:t, :] 区域置 0
                # 即横向整行区域被遮挡
                mask[s:t, :] *= 0

        # 7.2 若启用纵向遮挡，则每隔 d 个像素生成一条纵向遮挡带：
        if self.use_w:
            # 每隔 d 个像素生成一条纵向遮挡带
            for i in range(ww // d):
                # 当前遮挡带起点
                s = d * i + st_w

                # 当前遮挡带终点
                # 最大不超过 ww
                t = min(s + self.l, ww)

                # 纵向区域置 0：将 [:, s:t] 区域置 0
                # 即纵向整列区域被遮挡
                mask[:, s:t] *= 0

        # 8. 获取随机旋转角度r，范围是 [0, rotate)：
        # rotate=1 时不旋转：若 rotate=1，则 np.random.randint(1) 只能得到 0
        # rotate=360 时随机 0 到 359 度：若 rotate=360，则随机旋转 0 到 359 度
        r = np.random.randint(self.rotate)

        # 9.1 将 numpy mask 转成 PIL Image
        # np.uint8(mask) 会把 0/1 转成 8-bit 图像
        mask = Image.fromarray(np.uint8(mask))

        # 9.2 对 mask 旋转 r 度
        mask = mask.rotate(r)

        # 9.3 再从 PIL Image 转回 numpy array
        mask = np.asarray(mask)

        # 9.4 中心裁剪回原图大小 [h, w]：从较大的 mask 中心裁剪出原图大小 [h, w]：
        mask = mask[
            # 高度裁剪：高度方向裁剪起点到终点
            (hh - h) // 2 : (hh - h) // 2 + h,

            # 宽度裁剪：宽度方向裁剪起点到终点
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        # 9.5 将 numpy mask 转成 torch Tensor，并放到 GPU，shape 为 [H, W]：
        # mask.copy() 是为了避免 numpy array 由于内存 stride 问题导致 from_numpy 报错
        # 注意：这里直接 .cuda()，默认要求输入 x 也在 CUDA 上
        mask = torch.from_numpy(mask.copy()).float().cuda()

        # 10. 若 mode=1 则反转 mask：
        if self.mode == 1:
            # 反转 mask：原来保留的地方变遮挡，原来遮挡的地方变保留
            mask = 1 - mask

        # 11. 将二维 mask 扩展成和 x 一样的形状
        # x 当前形状是 [n*c, H, W]
        # mask 原来是 [H, W]，expand_as 后扩展成 [n*c, H, W]
        mask = mask.expand_as(x)

        # 12.1 若 offset=True 则用随机噪声填充遮挡区域，否则直接置 0：
        if self.offset:
            # 生成随机 offset 噪声
            # shape 是 [h, w]
            # 数值范围约为 [-1, 1)：先 np.random.rand(h, w) 范围是 [0,1)；再 2 * (... - 0.5) 范围约为 [-1,1)
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .cuda()
            )

            # 只在被遮挡区域保留 offset
            # 保留区域：x * mask
            # 遮挡区域：offset * (1 - mask)
            x = x * mask + offset * (1 - mask)
            '''
                相当于2步：
                    步1：
                        # 只在被遮挡区域保留 offset
                        # mask=1 的区域 offset 为 0
                        # mask=0 的区域 offset 为随机噪声
                        offset = (1 - mask) * offset

                    步2：
                        # 原图保留区域乘 mask
                        # 遮挡区域填充 offset
                        img = img * mask + offset            
            '''

        # 12.2 若 offset=False，则直接用 mask 乘图像张量x，即遮挡区域直接置0、保留区域保持原值：
        else:
            # 直接把遮挡区域置 0
            # 直接用 mask 乘图像
            # mask=0 的地方变成 0
            # mask=1 的地方保持原值
            x = x * mask

        # 13. 将 x 从 [n*c, h, w] reshape 回原来的 [n, c, h, w]，现在图像张量 x 已经经过 GridMask 增强了：
        return x.view(n, c, h, w)
