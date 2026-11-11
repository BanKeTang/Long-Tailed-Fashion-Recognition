# run_combined.py
# CB-Loss + 3D-Aug + 3D-Sampling + ConvNeXt-Large
# 这是论文的最终最强组合
#
# 运行：python run_combined.py --seed 42

import os, math, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from config import CFG, CATEGORY_NAMES
from dataset import FashionCropDataset, FashionMixedDataset, get_val_transforms
from model import FashionClassifier

RARE_CLASS_LABELS = {2, 5, 10, 12}
NUM_CLASSES = CFG["num_classes"]


class CBLoss(nn.Module):
    """Class-Balanced Loss (Cui et al., CVPR 2019)"""
    def __init__(self, samples_per_class, beta=0.9999, label_smoothing=0.1):
        super().__init__()
        effective_num = [1.0 - beta**n for n in samples_per_class]
        weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]
        total = sum(weights)
        weights = [w / total * NUM_CLASSES for w in weights]
        self.register_buffer('weights', torch.FloatTensor(weights))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets,
                               weight=self.weights,
                               label_smoothing=self.label_smoothing)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_scheduler(optimizer, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return float(step+1)/float(warmup_steps)
        p = float(step-warmup_steps)/float(max(1,total_steps-warmup_steps))
        return 0.5*(1.0+math.cos(math.pi*p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)

def mixup_data(x, y, alpha, device):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_amp):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, CFG["mixup_alpha"], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = lam*criterion(logits,y_a)+(1-lam)*criterion(logits,y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item()*imgs.size(0); total += imgs.size(0)
    return total_loss/total

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
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls = f1_score(all_labels, all_preds, average=None, zero_division=0)
    rare_f1 = float(np.mean([per_cls[i] for i in RARE_CLASS_LABELS]))
    return acc, macro_f1, rare_f1, all_preds, all_labels

def run(seed=42):
    set_seed(seed)
    device = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]

    print(f"\n{'='*60}")
    print(f"CB-Loss + 3D-Aug + 3D-Sampling + ConvNeXt-Large  seed={seed}")
    print(f"{'='*60}")

    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))

    # 获取训练集各类样本数（CB-Loss 需要）
    labels_list = train_ds.get_labels()
    class_counts = np.bincount(labels_list, minlength=NUM_CLASSES).tolist()
    print(f"各类样本数: {class_counts}")

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                              num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    model = FashionClassifier(backbone="convnext_large", pretrained=True,
                              num_classes=NUM_CLASSES, dropout=0.4).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion   = CBLoss(class_counts, beta=0.9999, label_smoothing=0.1).to(device)
    backbone_lr = 2e-5
    head_lr     = 2e-4
    optimizer   = optim.AdamW(
        model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
        weight_decay=CFG["weight_decay"]
    )
    total_steps  = CFG["epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_epochs"] * len(train_loader)
    scheduler    = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = f"checkpoints/combined_seed{seed}_best.pth"

    for epoch in range(1, CFG["epochs"]+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     scaler, scheduler, device, use_amp)
        val_acc, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"  Epoch [{epoch:02d}/{CFG['epochs']}] loss={train_loss:.4f} "
              f"acc={val_acc:.4f} macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":val_acc}, best_ckpt)
            print(f"  ✓ 保存最佳 val_acc={val_acc:.4f}")

    # 最终评估
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    results = {}
    for source_filter in [None, "shop", "user"]:
        tag = source_filter or "all"
        ds  = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]),
                                 source_filter=source_filter)
        loader = DataLoader(ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)
        acc, macro_f1, rare_f1, preds, labels = evaluate(model, loader, device, use_amp)
        per_cls = f1_score(labels, preds, average=None, zero_division=0)
        results[tag] = {
            "acc": round(float(acc),4), "macro_f1": round(float(macro_f1),4),
            "rare_f1": round(float(rare_f1),4),
            "per_class_f1": {CATEGORY_NAMES[i]: round(float(per_cls[i]),4) for i in range(NUM_CLASSES)},
        }
        print(f"  [{tag.upper()}] acc={acc:.4f} macro={macro_f1:.4f} rare={rare_f1:.4f}")

    print(f"\nClassification Report (ALL):")
    _, _, _, preds_all, labels_all = evaluate(model, val_loader, device, use_amp)
    print(classification_report(labels_all, preds_all, target_names=CATEGORY_NAMES, digits=4))

    # 保存
    out_file = "combined_results.json"
    data = {}
    if os.path.exists(out_file):
        with open(out_file) as f: data = json.load(f)
    data[f"seed{seed}"] = results
    with open(out_file,"w") as f: json.dump(data, f, indent=2)
    print(f"结果已保存到 {out_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.seed)
