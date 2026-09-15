# 从 Python 标准库 abc 中导入 ABC 和 abstractmethod
# ABC = Abstract Base Class，抽象基类
# abstractmethod 用于定义必须由子类实现的方法
from abc import ABC, abstractmethod


# __all__ 用于控制 from base_target import * 时暴露哪些对象
# 这里只暴露 BaseTargetWithDenoising
__all__ = ["BaseTargetWithDenoising"]


# 定义一个带 denoising 能力的 target 基类
# 它继承 ABC，说明这是一个抽象基类，不建议直接实例化
class BaseTargetWithDenoising(ABC):

    # 初始化函数
    # num_dn_groups：当前帧 denoising query 的组数
    # num_temp_dn_groups：时序 denoising query 的组数
    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        # 调用父类 ABC 的初始化函数
        super(BaseTargetWithDenoising, self).__init__()

        # 保存当前帧 denoising group 数量
        # 如果 num_dn_groups=0，通常表示不使用普通 DN training
        self.num_dn_groups = num_dn_groups

        # 保存 temporal denoising group 数量
        # 用于跨帧缓存 noisy anchors
        self.num_temp_dn_groups = num_temp_dn_groups

        # 初始化 denoising metadata 缓存
        # 后面 cache_dn() 会把上一帧的 DN 信息存在这里
        # Sparse4DHead.forward() 中会把 self.sampler.dn_metas 传给 InstanceBank.get()
        self.dn_metas = None


    # 抽象方法
    # 子类必须实现 sample()
    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        Perform Hungarian matching between predictions and ground truth,
        returning the matched ground truth corresponding to the predictions
        along with the corresponding regression weights.
        """
        # 这个函数在基类里只定义接口，不写具体实现
        # 子类需要实现具体的 Hungarian matching 逻辑
        # 例如：
        # SparseBox3DTarget.sample()
        # SparsePoint3DTarget.sample()


    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """
        Generate noisy instances for the current frame, with a total of
        'self.num_dn_groups' groups.
        """
        # 默认不生成 denoising anchors
        # 如果子类支持 DN training，需要重写这个函数
        return None


    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """
        Insert the previously saved 'self.dn_metas' into the noisy instances
        of the current frame.
        """
        # 默认不做 temporal DN 更新
        # 注意：这个函数没有显式 return
        # 所以默认返回 None
        # 如果子类支持 temporal denoising，需要重写它


    def cache_dn(
        self,
        dn_instance_feature, # denoising instance feature
        dn_anchor,           # denoising anchor
        dn_cls_target,       # denoising 分类 target
        valid_mask,          # 有效 mask
        dn_id_target,        # denoising instance id target
    ):
        """
        Randomly save information for 'self.num_temp_dn_groups' groups of
        temporal noisy instances to 'self.dn_metas'.
        """

        # 如果 temporal DN group 数量小于 0
        # 说明不启用 temporal DN 缓存
        if self.num_temp_dn_groups < 0:

            # 直接返回，不缓存任何信息
            return

        # 缓存 temporal DN 信息
        # 当前基类只缓存 dn_anchor 的前 num_temp_dn_groups 组
        # 子类如果需要缓存更多内容，例如 feature、cls_target、valid_mask、id_target，
        # 通常会重写这个函数
        self.dn_metas = dict(
            # dn_anchor shape 通常类似：
            # [B, num_dn_groups, num_dn, box_dim]
            # 这里只取前 num_temp_dn_groups 组作为 temporal DN 缓存
            dn_anchor=dn_anchor[:, : self.num_temp_dn_groups]
        )
