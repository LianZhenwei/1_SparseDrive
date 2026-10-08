import os
# 导入 os 标准库。
# 这里主要用于：
# 1. os.getenv 读取 FORCE_CUDA 环境变量；
# 2. os.path.join 拼接源文件路径。

import torch
# 导入 PyTorch。
# 这里主要用于 torch.cuda.is_available() 判断当前环境是否能看到 CUDA。

from setuptools import setup
# 从 setuptools 导入 setup。
# setup 是 Python 包构建/安装的标准入口。
# 这里用它来编译 PyTorch C++/CUDA 扩展。

from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)
# BuildExtension：
#   PyTorch 提供的 setuptools build_ext 命令扩展，
#   用于正确编译 PyTorch C++/CUDA extension。
#
# CppExtension：
#   用于构建纯 C++ 扩展，不包含 CUDA。
#
# CUDAExtension：
#   用于构建 CUDA 扩展，可以同时编译 .cpp 和 .cu 文件。


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):
    """
    创建一个 PyTorch C++/CUDA 扩展模块的构建配置。

    作用：
        根据当前环境是否支持 CUDA，决定使用 CUDAExtension 还是 CppExtension，
        并把源码路径、编译宏、编译参数、include 路径等打包成 extension 对象，
        供 setup(ext_modules=[...]) 使用。

    参数：
        name:
            扩展模块名称。
            例如 "deformable_aggregation_ext"。
        module:
            扩展模块所在的 Python 包路径。
            这里传 module="."，表示当前目录。
        sources:
            C++ 源文件列表。
            本文件中传入：
                ["src/deformable_aggregation.cpp",
                 "src/deformable_aggregation_cuda.cu"]
        sources_cuda:
            额外 CUDA 源文件列表。
            默认空列表。
            如果 CUDA 可用，会追加到 sources。
        extra_args:
            额外编译参数，会传给 cxx 和 nvcc。
        extra_include_path:
            额外头文件搜索路径。

    返回：
        CppExtension 或 CUDAExtension 对象。
        这个对象会被 setuptools.setup 的 ext_modules 使用。

    常见编译命令：
        cd ~/SparseDrive/projects/mmdet3d_plugin/ops
        python setup.py build_ext --inplace
    """

    define_macros = []
    # 初始化编译宏列表。
    # 如果启用 CUDA，后面会加入 ("WITH_CUDA", None)。

    extra_compile_args = {"cxx": [] + extra_args}
    # 初始化额外编译参数字典。
    # "cxx" 表示传给 C++ 编译器的参数。
    # [] + extra_args 是为了创建新 list，避免直接引用传入对象。

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        # 如果当前 PyTorch 能看到 CUDA，或者用户显式设置 FORCE_CUDA=1，
        # 就按照 CUDA 扩展来编译。
        #
        # FORCE_CUDA=1 的常见用途：
        #   有些环境 build 时 torch.cuda.is_available() 可能是 False，
        #   但用户仍然希望强制编译 CUDA 版本。

        define_macros += [("WITH_CUDA", None)]
        # 添加 WITH_CUDA 编译宏。
        # C++ 源码里可以通过 #ifdef WITH_CUDA 判断是否编译 CUDA 分支。

        extension = CUDAExtension
        # 选择 CUDAExtension。
        # 这意味着 setuptools 会用 nvcc 编译 .cu 文件，并链接 CUDA 相关库。

        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        # 设置传给 nvcc 的编译参数。
        # 这三个宏用于禁用 CUDA half / half2 的某些默认操作符和转换。
        # 目的通常是避免和 PyTorch/ATen 的 half 类型定义发生冲突。
        #
        # 简单理解：
        #   自定义 CUDA 扩展中经常加这些宏，减少半精度类型相关的编译冲突。

        sources += sources_cuda
        # 如果传入了额外 CUDA 源文件 sources_cuda，则追加到 sources。
        # 注意：
        #   当前 setup.py 调用 make_cuda_ext 时，.cu 文件已经写在 sources 里了，
        #   所以 sources_cuda 默认空，不影响编译。

    else:
        print("Compiling {} without CUDA".format(name))
        # 如果 CUDA 不可用且没有 FORCE_CUDA=1，则打印提示：不带 CUDA 编译。

        extension = CppExtension
        # 选择 CppExtension。
        # 但是对于本项目这个算子来说，如果 sources 中包含 .cu，
        # 在无 CUDA 环境下通常仍可能无法真正成功编译。
        # 因为 deformable aggregation 主要就是 CUDA 算子。

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    # 返回扩展模块配置对象。
    #
    # name="{}.{}".format(module, name)：
    #   生成 Python import 名称。
    #   当前 module="."，name="deformable_aggregation_ext"。
    #
    # sources=[os.path.join(*module.split("."), p) for p in sources]：
    #   把 module 的点号路径转成文件路径，再和源码文件名拼起来。
    #
    # include_dirs：
    #   额外头文件目录。
    #
    # define_macros：
    #   编译宏，例如 WITH_CUDA。
    #
    # extra_compile_args：
    #   C++/NVCC 编译参数。


if __name__ == "__main__":
    # 只有直接运行这个文件时才会执行 setup。
    # 如果它被 import，则不会进入这里。

    setup(
        name="deformable_aggregation_ext",
        # 当前扩展包/构建目标名称。
        # 这个名字主要用于 setuptools 构建过程记录。

        ext_modules=[
            make_cuda_ext(
                "deformable_aggregation_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation.cpp",
                    f"src/deformable_aggregation_cuda.cu",
                ],
            ),
        ],
        # ext_modules 指定要编译的扩展模块列表。
        #
        # 这里调用 make_cuda_ext 创建一个扩展：
        #   名称：deformable_aggregation_ext
        #   module：当前目录 "."
        #   源文件：
        #       src/deformable_aggregation.cpp
        #       src/deformable_aggregation_cuda.cu
        #
        # deformable_aggregation.cpp：
        #   通常负责 Python/C++ 绑定、参数检查、声明 CUDA 函数等。
        #
        # deformable_aggregation_cuda.cu：
        #   通常负责真正的 CUDA kernel 实现。

        cmdclass={"build_ext": BuildExtension},
        # 指定 build_ext 命令使用 PyTorch 的 BuildExtension。
        # 这样才能正确处理 PyTorch include path、ABI、CUDA 编译、链接等细节。
    )