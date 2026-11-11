# ablation.py
# 消融实验专用脚本
# 用法：python ablation.py --exp 2
#   exp=2: +WeightedSampler
#   exp=3: +UserAug
#   exp=4: +RareAug (Full Aug)
#   exp=5: +FocalLoss
#
# 结果自动保存到 ablation_results.json

import os
import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, CATEGORY_NAMES
from model import FashionClassifier

# 稀少类
RARE_CLASS_LABELS = {2, 5, 10, 12}
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
IMG_SIZE = CFG["img_size"]
NUM_CLASSES = CFG["num_classes"]


# ============================================================
# 增强策略
# ============================================================

def get_standard_aug():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0),
        ], p=0.85),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([A.GaussNoise(var_limit=(10,60),p=1.0), A.GaussianBlur(blur_limit=(3,7),p=1.0)], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=128, p=0.35),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])

def get_user_aug():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.45, 1.0), ratio=(0.6, 1.6)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=50, val_shift_limit=35, p=1.0),
        ], p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.CoarseDropout(max_holes=12, max_height=48, max_width=48, min_holes=2, min_height=16, min_width=16, fill_value=128, p=0.5),
        A.OneOf([A.GaussNoise(var_limit=(20,80),p=1.0), A.GaussianBlur(blur_limit=(3,9),p=1.0)], p=0.45),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])

def get_rare_aug():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.4, 1.0), ratio=(0.5, 2.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.25, p=1.0),
            A.HueSaturationValue(hue_shift_limit=45, sat_shift_limit=60, val_shift_limit=40, p=1.0),
        ], p=0.9),
        A.ToGray(p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.Perspective(scale=(0.05, 0.2), p=0.4),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.ElasticTransform(alpha=60, sigma=6, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.CoarseDropout(max_holes=15, max_height=56, max_width=56, min_holes=2, min_height=16, min_width=16, fill_value=128, p=0.5),
        A.OneOf([A.GaussNoise(var_limit=(20,100),p=1.0), A.GaussianBlur(blur_limit=(3,11),p=1.0)], p=0.45),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])

def get_val_aug():
    return A.Compose([
        A.Resize(int(IMG_SIZE*1.14), int(IMG_SIZE*1.14)),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])


# ============================================================
# Dataset
# ============================================================

class AblationDataset(Dataset):
    """根据实验编号选择不同的增强和采样策略"""

    def __init__(self, root, exp_id, is_train=True, source_filter=None):
        self.exp_id   = exp_id
        self.is_train = is_train
        self.samples  = []
        self.sources  = []

        root = Path(root)
        for label, cat_name in enumerate(CATEGORY_NAMES):
            cat_dir = root / cat_name
            if not cat_dir.exists():
                continue
            for img_path in sorted(cat_dir.glob("*.jpg")):
                source = img_path.name.split("_")[0]
                if source_filter and source != source_filter:
                    continue
                self.samples.append((str(img_path), label, source))
                self.sources.append(source)

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

        if is_train:
            # exp >= 3：使用 user 专项增强
            # exp >= 4：使用 rare 专项增强
            self.tfm_standard = get_standard_aug()
            self.tfm_user     = get_user_aug() if exp_id >= 3 else get_standard_aug()
            self.tfm_rare     = get_rare_aug() if exp_id >= 4 else get_standard_aug()
        else:
            self.tfm_val = get_val_aug()

        labels   = [s[1] for s in self.samples]
        counts   = np.bincount(labels, minlength=NUM_CLASSES)
        shop_cnt = self.sources.count("shop")
        user_cnt = self.sources.count("user")
        split    = root.name
        print(f"[AblationDataset] exp={exp_id} {split} total={len(self.samples)} "
              f"shop={shop_cnt} user={user_cnt}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            if label in RARE_CLASS_LABELS:
                tfm = self.tfm_rare
            elif source == "user":
                tfm = self.tfm_user
            else:
                tfm = self.tfm_standard
        else:
            tfm = self.tfm_val

        return tfm(image=img)["image"], label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_sample_weights(self, exp_id):
        labels        = self.get_labels()
        counts        = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        class_weights = 1.0 / np.where(counts > 0, counts, 1.0)
        weights = []
        for (_, label, source) in self.samples:
            w = class_weights[label]
            # exp >= 4：加 rare_boost 和 user_boost
            if exp_id >= 4:
                if label in RARE_CLASS_LABELS:
                    w *= 3.0
                if source == "user":
                    w *= 1.5
            weights.append(w)
        return torch.FloatTensor(weights)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal_w = (1.0 - pt) ** self.gamma
        focal_w = focal_w / (focal_w.mean().detach() + 1e-8)
        return (focal_w * ce_loss).mean()


# ============================================================
# 训练 + 验证
# ============================================================

def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def mixup_data(x, y, alpha, device):
    lam     = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index   = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_amp):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, 0.2, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp, source_filter="all"):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"  Eval[{source_filter}]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    rare_f1  = f1_score(all_labels, all_preds, average=None, zero_division=0)
    rare_avg = float(np.mean([rare_f1[i] for i in RARE_CLASS_LABELS]))
    return acc, macro_f1, rare_avg, all_preds, all_labels


# ============================================================
# 主流程
# ============================================================

def run_ablation(exp_id):
    device  = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]

    # exp=1,2,3,4,5 都用 efficientnet_b3
    backbone = "efficientnet_b3"
    dropout  = 0.3
    epochs   = CFG["epochs"]   # 40

    print(f"\n{'='*60}")
    print(f"消融实验 {exp_id}")
    print(f"  backbone  : {backbone}")
    print(f"  sampling  : {'Weighted' if exp_id >= 2 else 'Uniform'}")
    print(f"  user_aug  : {exp_id >= 3}")
    print(f"  rare_aug  : {exp_id >= 4}")
    print(f"  loss      : {'FocalLoss' if exp_id == 5 else 'CrossEntropy'}")
    print(f"{'='*60}")

    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = AblationDataset(train_root, exp_id, is_train=True)
    val_ds   = AblationDataset(val_root,   exp_id, is_train=False)

    if exp_id >= 2:
        sample_weights = train_ds.get_sample_weights(exp_id)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        sampler=sampler, shuffle=shuffle,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2,
        shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
    )

    model = FashionClassifier(
        backbone=backbone, pretrained=True,
        num_classes=CFG["num_classes"], dropout=dropout,
    ).to(device)

    if exp_id == 5:
        criterion = FocalLoss(gamma=1.5, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    backbone_lr = CFG["lr"] * CFG["backbone_lr_ratio"]
    head_lr     = CFG["lr"]
    optimizer   = optim.AdamW(
        model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
        weight_decay=CFG["weight_decay"],
    )
    total_steps  = epochs * len(train_loader)
    warmup_steps = CFG["warmup_epochs"] * len(train_loader)
    scheduler    = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc   = 0.0
    best_ckpt_path = f"checkpoints/ablation_exp{exp_id}_best.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, use_amp)

        # 快速验证（只用 all）
        val_acc_all, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp, "all")

        print(f"  Epoch [{epoch:02d}/{epochs}] train_loss={train_loss:.4f} "
              f"val_acc={val_acc_all:.4f} macro_f1={macro_f1:.4f} rare_f1={rare_f1:.4f}")

        if val_acc_all > best_val_acc:
            best_val_acc = val_acc_all
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": val_acc_all}, best_ckpt_path)
            print(f"  ✓ 保存最佳模型 val_acc={val_acc_all:.4f}")

    # 加载最佳模型，做完整评估
    print(f"\n加载最佳模型做最终评估...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # ALL 评估
    acc_all, macro_f1_all, rare_f1_all, preds_all, labels_all = evaluate(model, val_loader, device, use_amp, "all")

    # SHOP 评估
    shop_ds     = AblationDataset(val_root, exp_id, is_train=False, source_filter="shop")
    shop_loader = DataLoader(shop_ds, batch_size=CFG["batch_size"]*2, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
    acc_shop, _, _, _, _ = evaluate(model, shop_loader, device, use_amp, "shop")

    # USER 评估
    user_ds     = AblationDataset(val_root, exp_id, is_train=False, source_filter="user")
    user_loader = DataLoader(user_ds, batch_size=CFG["batch_size"]*2, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
    acc_user, _, _, _, _ = evaluate(model, user_loader, device, use_amp, "user")

    # 每个类的 F1
    per_class_f1 = f1_score(labels_all, preds_all, average=None, zero_division=0)

    result = {
        "exp_id":    exp_id,
        "backbone":  backbone,
        "sampling":  "Weighted" if exp_id >= 2 else "Uniform",
        "user_aug":  exp_id >= 3,
        "rare_aug":  exp_id >= 4,
        "loss":      "FocalLoss" if exp_id == 5 else "CrossEntropy",
        "acc_all":   round(float(acc_all),  4),
        "acc_shop":  round(float(acc_shop), 4),
        "acc_user":  round(float(acc_user), 4),
        "macro_f1":  round(float(macro_f1_all), 4),
        "rare_f1":   round(float(rare_f1_all),  4),
        "per_class_f1": {CATEGORY_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
    }

    print(f"\n{'='*60}")
    print(f"实验 {exp_id} 最终结果：")
    print(f"  ALL  acc = {acc_all:.4f}")
    print(f"  SHOP acc = {acc_shop:.4f}")
    print(f"  USER acc = {acc_user:.4f}")
    print(f"  Macro F1 = {macro_f1_all:.4f}")
    print(f"  Rare  F1 = {rare_f1_all:.4f}")
    print(f"{'='*60}")

    # 保存到 JSON
    result_file = "ablation_results.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[str(exp_id)] = result
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {result_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[2, 3, 4, 5],
                        help="实验编号：2=+WeightedSampler, 3=+UserAug, 4=+RareAug, 5=+FocalLoss")
    args = parser.parse_args()
    run_ablation(args.exp)
