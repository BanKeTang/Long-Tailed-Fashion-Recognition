# comparison_methods.py
# 横向对比方法实现
# 用法：
#   python comparison_methods.py --method cb_loss  --seed 42
#   python comparison_methods.py --method ldam     --seed 42
#   python comparison_methods.py --method bbn      --seed 42
#   python comparison_methods.py --method mislas   --seed 42
#
# 建议每个方法跑 3 个 seed (42, 123, 456) 取均值±std
# 结果自动保存到 comparison_results.json

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

RARE = {2, 5, 10, 12}
K    = CFG["num_classes"]


# ============================================================
# Loss Functions
# ============================================================

class ClassBalancedLoss(nn.Module):
    """Cui et al., CVPR 2019"""
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        eff = 1.0 - np.power(beta, samples_per_class)
        w   = (1.0 - beta) / np.array(eff)
        w   = w / w.sum() * K
        self.register_buffer('w', torch.FloatTensor(w))

    def forward(self, logits, targets):
        return (F.cross_entropy(logits, targets, reduction='none') * self.w[targets]).mean()


class LDAMLoss(nn.Module):
    """Cao et al., NeurIPS 2019"""
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m = m * (max_m / m.max())
        self.register_buffer('m_list', torch.FloatTensor(m))
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        idx = torch.zeros_like(x, dtype=torch.bool)
        idx.scatter_(1, target.view(-1,1), 1)
        batch_m = torch.matmul(self.m_list[None,:], idx.float().t()).view(-1,1)
        output  = torch.where(idx, x - batch_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# ============================================================
# Shared utilities
# ============================================================

def make_scheduler(optimizer, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return float(step+1) / warmup_steps
        p = float(step-warmup_steps) / max(1, total_steps-warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def get_cls_num_list(dataset):
    labels = dataset.get_labels()
    return np.bincount(labels, minlength=K).tolist()


def build_model():
    return FashionClassifier(
        backbone="convnext_large", pretrained=True,
        num_classes=K, dropout=0.4
    ).to(torch.device(CFG["device"]))


@torch.no_grad()
def evaluate(model, loader, device, use_amp, source="all"):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"  Eval[{source}]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)
    rare_f1  = float(np.mean([per_cls[i] for i in RARE]))
    return acc, macro_f1, rare_f1, all_preds, all_labels


def full_eval(model, ckpt_path, val_root, device, use_amp, method, seed):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    val_tfm = get_val_transforms(CFG["img_size"])
    results = {}
    for sf in [None, "shop", "user"]:
        tag = sf or "all"
        ds  = FashionCropDataset(val_root, transforms=val_tfm, source_filter=sf)
        ld  = DataLoader(ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                         num_workers=CFG["num_workers"], pin_memory=True)
        acc, macro_f1, rare_f1, preds, labels = evaluate(model, ld, device, use_amp, tag)
        per_cls = f1_score(labels, preds, average=None, zero_division=0)
        results[tag] = {
            "acc": round(float(acc),4),
            "macro_f1": round(float(macro_f1),4),
            "rare_f1": round(float(rare_f1),4),
            "per_class_f1": {CATEGORY_NAMES[i]: round(float(per_cls[i]),4) for i in range(K)},
        }
        print(f"  [{tag.upper()}] acc={acc:.4f}  macro={macro_f1:.4f}  rare={rare_f1:.4f}")

    # Append to results JSON
    out_file = "comparison_results.json"
    all_res  = json.load(open(out_file)) if os.path.exists(out_file) else {}
    key = f"{method}_seed{seed}"
    all_res[key] = results
    json.dump(all_res, open(out_file,"w"), indent=2)
    print(f"\n结果已保存到 {out_file} (key={key})")
    return results


# ============================================================
# Method: CB-Loss
# ============================================================

def run_cb_loss(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device(CFG["device"]); use_amp = CFG["use_amp"]
    epochs  = CFG["epochs"]
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds    = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds      = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    cls_num     = get_cls_num_list(train_ds)
    criterion   = ClassBalancedLoss(cls_num).to(device)

    # CB-Loss 原始论文用 uniform sampling
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    model     = build_model()
    optimizer = optim.AdamW(model.get_param_groups(backbone_lr=2e-5, head_lr=2e-4),
                            weight_decay=CFG["weight_decay"])
    scheduler = make_scheduler(optimizer, epochs*len(train_loader),
                               CFG["warmup_epochs"]*len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = "checkpoints_comparison"; os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"cb_loss_seed{seed}_best.pth")
    best_acc  = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        tl, tot = 0.0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Ep{epoch:02d}[CB]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam = mixup(imgs, labels, CFG["mixup_alpha"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss   = lam*criterion(logits,ya) + (1-lam)*criterion(logits,yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item()*imgs.size(0); tot += imgs.size(0)
        val_acc, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"  Ep[{epoch:02d}/{epochs}] loss={tl/tot:.4f} acc={val_acc:.4f} "
              f"macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":val_acc}, ckpt_path)
            print(f"  ✓ Best {val_acc:.4f}")

    return full_eval(model, ckpt_path, val_root, device, use_amp, "cb_loss", seed)


# ============================================================
# Method: LDAM + DRW
# ============================================================

def run_ldam_drw(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device(CFG["device"]); use_amp = CFG["use_amp"]
    epochs  = CFG["epochs"]; drw_start = epochs // 2
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    cls_num  = get_cls_num_list(train_ds)

    loader_uniform = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    sampler_bal    = WeightedRandomSampler(train_ds.get_sample_weights(),
                                           len(train_ds), replacement=True)
    loader_bal     = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler_bal,
                                num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader     = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

    model         = build_model()
    ce_loss       = nn.CrossEntropyLoss(label_smoothing=0.1)
    cb_w          = torch.FloatTensor([(1-0.9999)/(1-0.9999**n) for n in cls_num]).to(device)
    cb_w          = cb_w / cb_w.sum() * K
    ldam_loss     = LDAMLoss(cls_num, max_m=0.5, s=30, weight=cb_w).to(device)

    optimizer = optim.AdamW(model.get_param_groups(backbone_lr=2e-5, head_lr=2e-4),
                            weight_decay=CFG["weight_decay"])
    scheduler = make_scheduler(optimizer, epochs*len(loader_uniform),
                               CFG["warmup_epochs"]*len(loader_uniform))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = "checkpoints_comparison"; os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"ldam_drw_seed{seed}_best.pth")
    best_acc  = 0.0

    for epoch in range(1, epochs+1):
        if epoch <= drw_start:
            loader = loader_uniform; crit = ce_loss; tag = "CE"
        else:
            loader = loader_bal;     crit = ldam_loss; tag = "LDAM"
        model.train()
        tl, tot = 0.0, 0
        for imgs, labels in tqdm(loader, desc=f"  Ep{epoch:02d}[{tag}]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam = mixup(imgs, labels, CFG["mixup_alpha"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss   = lam*crit(logits,ya) + (1-lam)*crit(logits,yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item()*imgs.size(0); tot += imgs.size(0)
        val_acc, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"  Ep[{epoch:02d}/{epochs}][{tag}] loss={tl/tot:.4f} acc={val_acc:.4f} "
              f"macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":val_acc}, ckpt_path)
            print(f"  ✓ Best {val_acc:.4f}")

    return full_eval(model, ckpt_path, val_root, device, use_amp, "ldam_drw", seed)


# ============================================================
# Method: BBN
# ============================================================

def run_bbn(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device(CFG["device"]); use_amp = CFG["use_amp"]
    epochs  = CFG["epochs"]
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    bs_half  = CFG["batch_size"] // 2

    loader1 = DataLoader(train_ds, batch_size=bs_half, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    sampler2 = WeightedRandomSampler(train_ds.get_sample_weights(),
                                      len(train_ds), replacement=True)
    loader2  = DataLoader(train_ds, batch_size=bs_half, sampler=sampler2,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.get_param_groups(backbone_lr=2e-5, head_lr=2e-4),
                            weight_decay=CFG["weight_decay"])
    scheduler = make_scheduler(optimizer, epochs*len(loader1),
                               CFG["warmup_epochs"]*len(loader1))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = "checkpoints_comparison"; os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"bbn_seed{seed}_best.pth")
    best_acc  = 0.0
    iter2     = iter(loader2)

    for epoch in range(1, epochs+1):
        alpha = 1.0 - epoch / epochs  # BBN cumulative learning
        model.train()
        tl, tot = 0.0, 0
        for imgs1, labels1 in tqdm(loader1, desc=f"  Ep{epoch:02d}[BBN α={alpha:.2f}]", leave=False):
            try:
                imgs2, labels2 = next(iter2)
            except StopIteration:
                iter2 = iter(loader2)
                imgs2, labels2 = next(iter2)
            imgs1, labels1 = imgs1.to(device), labels1.to(device)
            imgs2, labels2 = imgs2.to(device), labels2.to(device)
            mixed = alpha * imgs1 + (1-alpha) * imgs2
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(mixed)
                loss   = alpha*criterion(logits,labels1) + (1-alpha)*criterion(logits,labels2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item()*mixed.size(0); tot += mixed.size(0)
        val_acc, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"  Ep[{epoch:02d}/{epochs}] α={alpha:.2f} acc={val_acc:.4f} "
              f"macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":val_acc}, ckpt_path)
            print(f"  ✓ Best {val_acc:.4f}")

    return full_eval(model, ckpt_path, val_root, device, use_amp, "bbn", seed)


# ============================================================
# Method: MiSLAS
# ============================================================

def run_mislas(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device(CFG["device"]); use_amp = CFG["use_amp"]
    epochs  = CFG["epochs"]
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root   = os.path.join(CFG["crops_dir"], "validation")

    train_ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)
    val_ds   = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    cls_num  = get_cls_num_list(train_ds)
    max_cnt  = max(cls_num)

    sampler      = WeightedRandomSampler(train_ds.get_sample_weights(),
                                          len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                              num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    # Class-aware label smoothing: rare classes get smaller smoothing
    smooth_per_cls = torch.FloatTensor(
        [0.15 * (n / max_cnt) ** 0.5 for n in cls_num]
    ).to(device)

    model     = build_model()
    optimizer = optim.AdamW(model.get_param_groups(backbone_lr=2e-5, head_lr=2e-4),
                            weight_decay=CFG["weight_decay"])
    scheduler = make_scheduler(optimizer, epochs*len(train_loader),
                               CFG["warmup_epochs"]*len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = "checkpoints_comparison"; os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"mislas_seed{seed}_best.pth")
    best_acc  = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        tl, tot = 0.0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Ep{epoch:02d}[MiSLAS]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam = mixup(imgs, labels, CFG["mixup_alpha"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                # Class-aware smoothing
                smooth_a = smooth_per_cls[ya]
                smooth_b = smooth_per_cls[yb]
                loss_a = F.cross_entropy(logits, ya, label_smoothing=float(smooth_a.mean()))
                loss_b = F.cross_entropy(logits, yb, label_smoothing=float(smooth_b.mean()))
                loss   = lam * loss_a + (1-lam) * loss_b
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item()*imgs.size(0); tot += imgs.size(0)
        val_acc, macro_f1, rare_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"  Ep[{epoch:02d}/{epochs}] loss={tl/tot:.4f} acc={val_acc:.4f} "
              f"macro={macro_f1:.4f} rare={rare_f1:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":val_acc}, ckpt_path)
            print(f"  ✓ Best {val_acc:.4f}")

    return full_eval(model, ckpt_path, val_root, device, use_amp, "mislas", seed)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["cb_loss", "ldam", "bbn", "mislas"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"对比实验: {args.method}  seed={args.seed}")
    print(f"backbone: convnext_large  epochs: {CFG['epochs']}")
    print(f"{'='*60}\n")

    if   args.method == "cb_loss": run_cb_loss(args.seed)
    elif args.method == "ldam":    run_ldam_drw(args.seed)
    elif args.method == "bbn":     run_bbn(args.seed)
    elif args.method == "mislas":  run_mislas(args.seed)
