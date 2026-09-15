## stage1 第一阶段训练
# 这一阶段主要训练 SparseDrive 的感知部分：detection、tracking、map prediction，不训练 motion prediction 和 planning
bash ./tools/dist_train.sh \
   # 配置文件：SparseDrive small stage1 配置
   projects/configs/sparsedrive_small_stage1.py \
   # 使用 8 张 GPU 进行分布式训练
   8 \
   # 开启确定性训练模式，作用是尽量保证每次运行结果可复现
   --deterministic


## stage2 第二阶段训练
# 这一阶段在 stage1 权重基础上继续训练，启用 motion prediction 和 ego planning
bash ./tools/dist_train.sh \
   # 配置文件：SparseDrive small stage2 配置
   projects/configs/sparsedrive_small_stage2.py \
   # 使用 8 张 GPU 分布式训练
   8 \
   # 开启确定性训练模式，作用是尽量保证每次运行结果可复现
   --deterministic