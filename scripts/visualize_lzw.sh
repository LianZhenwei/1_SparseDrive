# 将 SparseDrive 项目根目录加入 PYTHONPATH
# dirname $0 表示当前脚本所在目录，也就是 scripts
# $(dirname $0)/.. 表示 scripts 的上一级目录，也就是项目根目录
# 这样 Python 才能 import projects/mmdet3d_plugin 里的自定义模块
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


# 调用 SparseDrive 的可视化脚本
python tools/visualization/visualize.py \
	# 配置文件
	# 可视化时需要根据配置文件构建 dataset、读取类别、坐标系、pipeline 等信息
	projects/configs/sparsedrive_small_stage2.py \
	
	# 测试阶段保存的结果文件
	# 这个文件一般由 test.sh 中的 --result_file 参数生成
	--result-path work_dirs/sparsedrive_small_stage2/results.pkl