: '
    这个脚本的作用一句话概括：
    把 nuScenes 原始数据转换成 SparseDrive 训练、验证、测试要用的 infos pkl 文件。

    如果你运行当前未注释版本：
    bash scripts/create_data.sh

    最终主要生成：
    data/infos/mini/nuscenes_infos_train.pkl
    data/infos/mini/nuscenes_infos_val.pkl

    如果你改成完整版本 --version v1.0，则会生成：
    data/infos/nuscenes_infos_train.pkl
    data/infos/nuscenes_infos_val.pkl
    data/infos/nuscenes_infos_test.pkl
'


# 将 SparseDrive 项目根目录加入 PYTHONPATH
# dirname $0 表示当前脚本所在目录，也就是 scripts
# $(dirname $0)/.. 表示 scripts 的上一级目录，也就是 SparseDrive 项目根目录
# 加入 PYTHONPATH 后，Python 才能正常 import projects.mmdet3d_plugin 里的自定义模块
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


# 调用 nuScenes 数据转换脚本
# 作用是把原始 nuScenes 数据转换成 SparseDrive 训练需要的 .pkl 信息文件
python tools/data_converter/nuscenes_converter.py nuscenes \
    # nuScenes 原始数据根目录
    # 里面一般包含 samples、sweeps、maps、v1.0-mini 等目录
    --root-path ./data/nuscenes \

    # CAN bus 数据目录
    # SparseDrive 会用 CAN bus 读取 ego acceleration、velocity、steering angle 等自车状态
    --canbus ./data/nuscenes \

    # 输出 info pkl 文件的目录
    # v1.0-mini 时，源码内部会额外拼接 mini，所以实际输出到 ./data/infos/mini/
    --out-dir ./data/infos/ \

    # 输出文件名前缀
    # 最终会生成类似：
    # nuscenes_infos_train.pkl
    # nuscenes_infos_val.pkl
    --extra-tag nuscenes \

    # 使用 nuScenes mini 数据集
    # 适合快速调试
    --version v1.0-mini


# 下面这段被注释掉了
# 如果你取消注释，就是转换完整 nuScenes v1.0 数据集
# 它会生成 trainval 和 test 两套 info 文件

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     # 完整 nuScenes 数据根目录
#     --root-path ./data/nuscenes \
#
#     # CAN bus 数据目录
#     --canbus ./data/nuscenes \
#
#     # 输出目录
#     --out-dir ./data/infos/ \
#
#     # 输出文件名前缀
#     --extra-tag nuscenes \
#
#     # 完整版本 v1.0
#     # 脚本内部会自动转换成：
#     # v1.0-trainval
#     # v1.0-test
#     --version v1.0