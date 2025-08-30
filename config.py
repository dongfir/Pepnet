import torch
import random
import numpy as np
import os

class Config:
    # === 模型结构 ===
    in_dim = 256
    hidden_dim = 256
    num_layers = 4
    dropout = 0.2
    use_mask = False
    use_global = False

    # === 数据与路径 ===
    data_path = os.getenv("PEPNET_DATA_PATH", "./data/biolip1.pt")
    model_save_path = os.getenv("PEPNET_MODEL_PATH", "./results/models/pepnet_model4_seed42.pt")
    val_ratio = 0.10   # 9:1 划分

    # 脚本路径
    train_script_path   = os.getenv("PEPNET_TRAIN_SCRIPT_PATH", "./train/train.py")
    eval_script_path    = os.getenv("PEPNET_EVAL_SCRIPT_PATH", "./train/eval.py")
    predict_script_path = os.getenv("PEPNET_PREDICT_SCRIPT_PATH", "./predict.py")  # 预测脚本在主目录

    # 预测结果保存目录
    prediction_save_path = os.getenv("PEPNET_PREDICTION_PATH", "./results/predictions")

    # === 训练超参 ===
    batch_size = 16
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 80

    # === 优化与稳定化 ===
    use_amp = True
    warmup_ratio = 0.10
    grad_clip = 0.5
    bad_limit_per_epoch = 10

    # === 损失函数===
    loss_name = "bce"
    gamma_pos = 0.25
    gamma_neg = 2.0
    asym_clip = 0.0

    # === 采样===
    use_sampler = False
    sampler_alpha = 1.0

    # === 评估===
    use_calibration = True
    eval_threshold = 0.35

    # === 早停 ===
    early_stop_patience = 10
    use_swa = False
    swa_start_epoch = 40

    # === 随机种子 ===
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 评估：强校准+MCC 搜阈值
    use_isotonic = True
    use_temp_bias = False
    th_search_min = 0.28
    th_search_max = 0.55
    th_search_step = 0.002

    # 训练：EMA 稳定验证曲线
    use_ema = True
    ema_decay = 0.999

    head_type = "cosine"
    head_s = 32.0
    head_m = 0.25
    gate_fused_base = 0.60
    gate_fused_gain = 0.40
    gate_aux_gain = 0.40
    weight_gamma = 1.20

    pos_weight_scale = 1.05
    consistency_weight = 0.03

    # === 硬件 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
