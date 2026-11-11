# config.py
import torch

CFG = {
    # ===== 路径 =====
    "data_root":         "/path/to/deepfashion2",
    "crops_dir":         "crops",
    "save_dir":          "checkpoints",
    "log_dir":           "logs",

    # ===== 模型 =====
    "backbone":          "convnext_small",    # 50M参数，比b3强且不过拟合
    "num_classes":       13,
    "dropout":           0.4,               # 参数量更大，略微加大
    "img_size":          224,

    # ===== 训练 =====
    "batch_size":        64,
    "num_workers":       8,
    "epochs":            40,
    "lr":                3e-4,
    "backbone_lr_ratio": 0.1,                 # backbone lr = lr * 0.1
    "weight_decay":      1e-4,
    "warmup_epochs":     3,
    "label_smoothing":   0.1,
    "mixup_alpha":       0.2,
    "grad_clip":         1.0,
    "bbox_padding":      0.1,

    # ===== 硬件 =====
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp":           torch.cuda.is_available(),
}

CATEGORY_NAMES = [
    "short_sleeve_top",       # 0
    "long_sleeve_top",        # 1
    "short_sleeve_outwear",   # 2
    "long_sleeve_outwear",    # 3
    "vest",                   # 4
    "sling",                  # 5
    "shorts",                 # 6
    "trousers",               # 7
    "skirt",                  # 8
    "short_sleeve_dress",     # 9
    "long_sleeve_dress",      # 10
    "vest_dress",             # 11
    "sling_dress",            # 12
]

CATEGORY_NAMES_ZH = [
    "短袖上衣", "长袖上衣", "短袖外套", "长袖外套",
    "马甲", "吊带上衣", "短裤", "长裤",
    "半身裙", "短袖连衣裙", "长袖连衣裙", "背心裙", "吊带裙"
]

ID_TO_LABEL = {i + 1: i for i in range(13)}
ID_TO_NAME  = {i + 1: CATEGORY_NAMES[i] for i in range(13)}