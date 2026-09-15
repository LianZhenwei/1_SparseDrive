# 使用分布式测试脚本测试 SparseDrive stage2 模型
bash ./tools/dist_test.sh \
    # 测试配置文件
    # 一般使用 stage2 配置，因为最终模型包含 detection、map、motion、planning
    projects/configs/sparsedrive_small_stage2.py \
    
    # 需要测试的模型权重
    # 这里使用已经训练好的 stage2 checkpoint
    ckpt/sparsedrive_stage2.pth \
    
    # 使用 8 张 GPU 进行分布式测试
    8 \
    
    # 开启确定性模式
    # 尽量保证测试结果可复现
    --deterministic \
    
    # 指定评估类型
    # bbox 是 MMDetection/MMDetection3D 中常见的评估入口
    # 在 SparseDrive 里，具体评估哪些任务还要看配置文件里的 eval_mode
    --eval bbox

    # 可选参数：保存模型预测结果
    # 如果取消注释，会把预测结果保存为 results.pkl
    # 后续 visualize.py 可读取这个 pkl 做可视化
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl