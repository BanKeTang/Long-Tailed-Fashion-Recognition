# repeat_experiments.py
# 重复实验脚本：对最优配置（ConvNeXt-Large full）跑3次取均值±标准差
# 用法：
#   python repeat_experiments.py --seed 42
#   python repeat_experiments.py --seed 123
#   python repeat_experiments.py --seed 2024
#   python repeat_experiments.py --summarize   # 汇总3次结果

import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from config import CFG, CATEGORY_NAMES
from dataset import FashionCropDataset, FashionMixedDataset, get_val_transforms
from model import FashionClassifier

RARE_CLASS_LABELS = {2, 5, 10, 12}
BACKBONE = "convnext_large"
DROPOUT  = 0.4
LR       = 2e-4


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def mixup_data(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_amp):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, CFG["mixup_alpha"], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)
    rare_f1  = float(np.mean([per_cls[i] for i in RARE_CLASS_LABELS]))
    return acc, macro_f1, rare_f1


def run(seed: int):
    set_seed(seed)
    device  = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]
    epochs  = CFG["epochs"]

    print(f"\n{'='*60}")
    print(f"重复实验: ConvNeXt-Large Full  seed={seed}")
    print(f"{'='*60}")

    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                              num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    model = FashionClassifier(backbone=BACKBONE, pretrained=True,
                              num_classes=CFG["num_classes"], dropout=DROPOUT).to(device)
    criterion   = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    backbone_lr = LR * CFG["backbone_lr_ratio"]
    optimizer   = optim.AdamW(
        model.get_param_groups(backbone_lr=backbone_lr, head_lr=LR),
        weight_decay=CFG["weight_decay"]
    )
    total_steps  = epochs * len(train_loader)
    warmup_steps = CFG["warmup_epochs"] * len(train_loader)
    scheduler    = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = f"checkpoints/repeat_seed{seed}_best.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     scaler, scheduler, device, use_amp)
        val_acc, macro_f1, rare_f1 = evaluate(model, val_loader, device, use_amp)
        print(f"  Epoch [{epoch:02d}/{epochs}] loss={train_loss:.4f} "
              f"acc={val_acc:.4f} macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": val_acc}, best_ckpt)
            print(f"  ✓ 保存最佳 val_acc={val_acc:.4f}")

    # 最终评估（分 shop/user）
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    final = {}
    for source_filter in [None, "shop", "user"]:
        tag = source_filter or "all"
        ds  = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]),
                                 source_filter=source_filter)
        loader = DataLoader(ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)
        acc, macro_f1, rare_f1 = evaluate(model, loader, device, use_amp)
        final[tag] = {"acc": round(float(acc),4),
                      "macro_f1": round(float(macro_f1),4),
                      "rare_f1": round(float(rare_f1),4)}
        print(f"  [{tag.upper()}] acc={acc:.4f} macro={macro_f1:.4f} rare={rare_f1:.4f}")

    # 保存
    out_file = "repeat_results.json"
    data = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            data = json.load(f)
    data[str(seed)] = final
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n结果已保存到 {out_file}")
    return final


def summarize():
    """汇总3次实验结果，计算均值±标准差"""
    out_file = "repeat_results.json"
    if not os.path.exists(out_file):
        print("还没有结果，请先跑实验")
        return

    with open(out_file) as f:
        data = json.load(f)

    print(f"\n已有 {len(data)} 次实验结果: seeds = {list(data.keys())}")
    print("="*60)

    for source in ["all", "shop", "user"]:
        metrics = ["acc", "macro_f1", "rare_f1"]
        print(f"\n[{source.upper()}]")
        for m in metrics:
            vals = [data[seed][source][m] for seed in data if source in data[seed]]
            if not vals:
                continue
            mean = np.mean(vals)
            std  = np.std(vals)
            print(f"  {m:10s}: {mean:.4f} ± {std:.4f}  (runs: {vals})")

    if len(data) >= 3:
        print("\n" + "="*60)
        print("论文用格式（均值±标准差）：")
        for source in ["all", "shop", "user"]:
            acc_vals = [data[s][source]["acc"] for s in data if source in data[s]]
            mf1_vals = [data[s][source]["macro_f1"] for s in data if source in data[s]]
            rf1_vals = [data[s][source]["rare_f1"] for s in data if source in data[s]]
            print(f"  {source.upper():6s}: "
                  f"Acc={np.mean(acc_vals):.4f}±{np.std(acc_vals):.4f}  "
                  f"MacroF1={np.mean(mf1_vals):.4f}±{np.std(mf1_vals):.4f}  "
                  f"RareF1={np.mean(rf1_vals):.4f}±{np.std(rf1_vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    elif args.seed is not None:
        run(args.seed)
    else:
        print("请指定 --seed 或 --summarize")
