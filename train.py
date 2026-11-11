# train.py
# 运行：python train.py

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CFG, CATEGORY_NAMES
from dataset import FashionCropDataset, FashionMixedDataset, get_val_transforms
from model import FashionClassifier


# ============================================================
# Mixup
# ============================================================

def mixup_data(x, y, alpha, device):
    lam     = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index   = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 学习率调度：线性 warmup + cosine decay（按 batch step）
# 用 LambdaLR 手写，避免 OneCycleLR 的初始化 warning
# ============================================================

def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================
# 训练一个 epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, scheduler, device, use_amp):
    model.train()
    total_loss = 0.0
    total      = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs, y_a, y_b, lam = mixup_data(imgs, labels, CFG["mixup_alpha"], device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
        )

    return total_loss / total


# ============================================================
# 验证
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, \
           np.array(all_preds), np.array(all_labels)


# ============================================================
# 训练结束后：分 shop / user 评估
# ============================================================

@torch.no_grad()
def evaluate_by_source(model, val_root, criterion, device, use_amp):
    val_tfm = get_val_transforms(CFG["img_size"])
    results = {}

    for source_filter in [None, "shop", "user"]:
        tag = source_filter or "all"
        try:
            ds = FashionCropDataset(
                root=val_root,
                transforms=val_tfm,
                source_filter=source_filter,
            )
        except RuntimeError:
            continue

        loader = DataLoader(
            ds, batch_size=CFG["batch_size"] * 2,
            shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
        )

        all_preds, all_labels = [], []
        model.eval()
        for imgs, labels in tqdm(loader, desc=f"  Eval [{tag}]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        results[tag] = acc
        print(f"\n{'='*60}")
        print(f"[{tag.upper()}]  n={len(all_labels)}  Accuracy={acc:.4f}")
        print("=" * 60)
        print(classification_report(
            all_labels, all_preds,
            target_names=CATEGORY_NAMES, digits=4,
        ))

    return results


# ============================================================
# 绘图
# ============================================================

def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_acc"], color="green", label="Val Acc")
    axes[1].set_title("Val Accuracy"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history["lr"], color="orange", label="LR")
    axes[2].set_title("Learning Rate"); axes[2].legend(); axes[2].grid(True)
    axes[2].set_yscale("log")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  训练曲线: {path}")


def plot_confusion_matrix(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks  = np.arange(len(CATEGORY_NAMES))
    short_names = [n.replace("_", "\n") for n in CATEGORY_NAMES]
    ax.set_xticks(tick_marks); ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(short_names, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  混淆矩阵: {path}")


# ============================================================
# 主训练流程
# ============================================================

def main():
    device  = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]
    os.makedirs(CFG["save_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"],  exist_ok=True)

    print("=" * 60)
    print(f"backbone : {CFG['backbone']}")
    print(f"img_size : {CFG['img_size']}")
    print(f"device   : {device}  amp={use_amp}")
    print(f"epochs   : {CFG['epochs']}")
    print("=" * 60)

    # ------ 数据 ------
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    # ------ 模型 ------
    model = FashionClassifier(
        backbone=CFG["backbone"], pretrained=True,
        num_classes=CFG["num_classes"], dropout=CFG["dropout"],
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ------ 损失 + 优化器 ------
    criterion   = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    backbone_lr = CFG["lr"] * CFG["backbone_lr_ratio"]
    head_lr     = CFG["lr"]
    optimizer   = optim.AdamW(
        model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
        weight_decay=CFG["weight_decay"],
    )

    # ------ 学习率：线性 warmup + cosine decay，按 batch step ------
    total_steps  = CFG["epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_epochs"] * len(train_loader)
    scheduler    = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------ 训练循环 ------
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(CFG["save_dir"], "best_model.pth")
    history        = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\nEpoch [{epoch:02d}/{CFG['epochs']}]  "
              f"backbone_lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"head_lr={optimizer.param_groups[-1]['lr']:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, use_amp,
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, use_amp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[-1]["lr"])

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_acc": val_acc, "cfg": CFG,
            }, best_ckpt_path)
            print(f"  ✓ 保存最佳模型  val_acc={val_acc:.4f}")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_acc": val_acc, "cfg": CFG,
            }, os.path.join(CFG["save_dir"], f"epoch_{epoch:02d}.pth"))

    # ------ 最终评估 ------
    print(f"\n{'='*60}")
    print(f"训练完成！最佳 val_acc = {best_val_acc:.4f}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    results = evaluate_by_source(model, val_root, criterion, device, use_amp)
    print(f"\n{'='*60}\n汇总：")
    for tag, acc in results.items():
        print(f"  {tag.upper():6s} acc = {acc:.4f}")

    plot_history(history, CFG["save_dir"])
    _, _, preds, labels = validate(model, val_loader, criterion, device, use_amp)
    plot_confusion_matrix(labels, preds, CFG["save_dir"])
    print(f"\n所有文件已保存到: {CFG['save_dir']}/")


if __name__ == "__main__":
    main()