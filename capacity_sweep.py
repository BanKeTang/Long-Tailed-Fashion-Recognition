# capacity_sweep.py
# 同族容量扫描：ConvNeXt-T / S / B / L 与 EfficientNet-B3 / B5  ×  {不加稀少类增强, 加稀少类增强}
# 只改变 backbone 尺寸和"稀少类增强开关"，其余设置与论文完全一致：
#   3D-W 采样 (rare_boost=3.0, user_boost=1.5)、user 增强、CE + label smoothing 0.1、
#   Mixup 0.2、AdamW、40 epochs、224x224、batch 64、warmup 3 + cosine
#
# 用法（在 garment_type 目录下）：
#   1) 冒烟测试，几分钟：      python capacity_sweep.py --smoke
#   2) 正式运行，挂后台：      nohup python capacity_sweep.py > log_capacity.txt 2>&1 &
#   3) 只跑指定组：            python capacity_sweep.py --runs tiny_norare tiny_rare
#   4) 补其他种子：            python capacity_sweep.py --seed 123 --runs tiny_norare tiny_rare
#   5) 写到别的结果文件：      python capacity_sweep.py --out capacity_seeds.json ...
# 每组跑完立即写入 capacity_results.json；已完成的组自动跳过，中断后重新运行即可续跑。
#
# 第三轮扩展（组名后面可加可选字段，不加时与之前完全相同 = CE-LS + 3D-W）：
#   组名 = <backbone>_<增强>[_<损失>][_<采样>]
#   增强: norare | rare(=strong) | mild | medium | geom | photo   （见 rare_policies.py）
#   损失: ce（默认，CE + LS 0.1） | cb（CB-Loss β=0.9999 + LS 0.1，与 run_combined.py 相同）
#   采样: 3dw（默认） | uni（均匀 shuffle，与 interaction_runs.py 相同）
#   例：tiny_norare_cb_uni、tiny_rare_ce_uni、tiny_geom

import os, gc, json, time, random, argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CFG, CATEGORY_NAMES
from dataset import (FashionCropDataset, FashionMixedDataset,
                     get_val_transforms, RARE_CLASS_LABELS, IMG_SIZE)
from model import FashionClassifier
from train import mixup_data, mixup_criterion, make_scheduler

# backbone 名称（timm）, head lr（backbone lr = head lr × 0.1）, dropout —— 均与论文 4.1 / Table 2 一致
BACKBONES = {
    "tiny":  ("convnext_tiny",   3e-4, 0.4),
    "small": ("convnext_small",  3e-4, 0.4),
    "base":  ("convnext_base",   3e-4, 0.4),
    "large": ("convnext_large",  2e-4, 0.4),   # 与论文 Large 实验一致
    "effb3": ("efficientnet_b3", 3e-4, 0.3),
    "effb5": ("efficientnet_b5", 3e-4, 0.3),
    # Transformer 家族（第二轮补充）
    "swint": ("swin_tiny_patch4_window7_224", 3e-4, 0.4),
    "swinb": ("swin_base_patch4_window7_224", 3e-4, 0.4),
}

# 默认 11 组，按优先级排序：先 ConvNeXt 同族（编辑最关心的），再 EfficientNet 同族。
# Large + 稀少类增强 论文里已有（3 seeds），不重复跑。
# 两个架构族内部各自做"有/无稀少类增强"对照，采样（3D-W）在两臂中完全相同。
DEFAULT_RUNS = [
    "tiny_norare",  "tiny_rare",
    "small_norare", "small_rare",
    "base_norare",  "base_rare",
    "large_norare",
    "effb3_norare", "effb3_rare",
    "effb5_norare", "effb5_rare",
]

RARE = sorted(RARE_CLASS_LABELS)
SAVE_ROOT = "checkpoints_capacity"


# ============================================================
# 数据集：和 FashionMixedDataset 完全一样，只多一个稀少类增强开关
#   use_rare_aug=True  → 稀少类用 rare 增强（论文 3D-Aug）
#   use_rare_aug=False → 稀少类按来源用 shop / user 增强（采样权重不变）
# ============================================================

class SweepDataset(FashionMixedDataset):
    def __init__(self, root, use_rare_aug=True, rare_tfm=None, **kw):
        super().__init__(root, **kw)
        self.use_rare_aug = use_rare_aug
        if rare_tfm is not None:          # 剂量实验：替换稀少类增强的强度 / 组成
            self.tfm_rare = rare_tfm

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_rare_aug and label in RARE_CLASS_LABELS:
            tfm = self.tfm_rare
        elif source == "user":
            tfm = self.tfm_user
        else:
            tfm = self.tfm_shop
        return tfm(image=img)["image"], label


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, use_amp, max_batches=0):
    model.eval()
    preds, labels = [], []
    for i, (imgs, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds.append(model(imgs).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def compute_metrics(preds, labels, sources):
    preds, labels, src = np.asarray(preds), np.asarray(labels), np.asarray(sources)
    f1 = f1_score(labels, preds, labels=list(range(len(CATEGORY_NAMES))),
                  average=None, zero_division=0)

    def acc(mask):
        return float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else None

    return {
        "acc_all":  float((preds == labels).mean()),
        "acc_shop": acc(src == "shop"),
        "acc_user": acc(src == "user"),
        "macro_f1": float(f1.mean()),
        "rare_f1":  float(f1[RARE].mean()),
        "per_class_f1": {CATEGORY_NAMES[i]: float(f1[i]) for i in range(len(f1))},
    }


def fmt(x):
    return "  -   " if x is None else f"{x:.4f}"

def save_result(path, key, value):
    """读-改-写并加文件锁：多个进程写同一个结果文件也不会互相覆盖"""
    import fcntl
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.load(open(path)) if os.path.exists(path) else {}
        data[key] = value
        tmp = path + ".tmp"
        json.dump(data, open(tmp, "w"), indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return data



# ============================================================
# 跑一组实验
# ============================================================

class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted cross-entropy: CE(logits + τ·log π, y)；推理时不加偏置。"""
    def __init__(self, prior, tau=1.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("log_prior", torch.tensor(np.log(prior), dtype=torch.float32))
        self.tau, self.ls = tau, label_smoothing

    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits + self.tau * self.log_prior, targets, label_smoothing=self.ls)


AUGS = ("norare", "rare", "mild", "medium", "geom", "photo")


def parse_run(run_name):
    parts = run_name.split("_")
    size, aug, opts = parts[0], parts[1] if len(parts) > 1 else "", parts[2:]
    loss = "cb" if "cb" in opts else ("la" if "la" in opts else "ce")
    sampler = "uni" if "uni" in opts else ("cbs" if "cbs" in opts else "3dw")
    bad = [o for o in opts if o not in ("ce", "cb", "la", "3dw", "uni", "cbs")]
    if size not in BACKBONES or aug not in AUGS or bad:
        raise ValueError(f"无效的组名: {run_name}（格式: <{'/'.join(BACKBONES)}>_<{'/'.join(AUGS)}>[_ce|_cb|_la][_3dw|_uni|_cbs]）")
    return size, aug, loss, sampler


def run_one(run_name, args, device, use_amp, train_root, val_ds, val_loader):
    size, aug, loss_name, sampler_name = parse_run(run_name)
    backbone, lr, dropout = BACKBONES[size]
    use_rare = (aug != "norare")
    set_seed(args.seed)

    print("\n" + "=" * 70)
    print(f"[{run_name}]  backbone={backbone}  rare_aug={aug}  loss={loss_name}  sampler={sampler_name}  "
          f"lr={lr}  dropout={dropout}  seed={args.seed}")
    print("=" * 70)

    rare_tfm = None
    if aug in ("mild", "medium", "geom", "photo"):
        from rare_policies import policy
        rare_tfm = policy(aug, IMG_SIZE)
    train_ds = SweepDataset(train_root, use_rare_aug=use_rare, rare_tfm=rare_tfm,
                            user_boost=1.5, rare_boost=3.0)
    if sampler_name == "3dw":
        sampler = WeightedRandomSampler(train_ds.get_sample_weights(),
                                        num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    elif sampler_name == "cbs":       # 标准类别均衡采样：权重 = 1/n_c，没有 rare / user 额外倍数
        lab = np.array(train_ds.get_labels())
        cnt = np.bincount(lab, minlength=CFG["num_classes"]).astype(float)
        sampler = WeightedRandomSampler(torch.tensor(1.0 / cnt[lab], dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

    model = FashionClassifier(backbone=backbone, pretrained=True,
                              num_classes=CFG["num_classes"], dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n_params:.1f}M")

    if loss_name == "cb":
        from run_combined import CBLoss      # 与 run_combined.py / interaction_runs.py 同一实现
        counts = np.bincount(train_ds.get_labels(), minlength=CFG["num_classes"]).tolist()
        criterion = CBLoss(counts, beta=0.9999, label_smoothing=CFG["label_smoothing"]).to(device)
    elif loss_name == "la":             # logit-adjusted loss (Menon et al., 2021)，τ = 1，推理时用原始 logits
        counts = np.bincount(train_ds.get_labels(), minlength=CFG["num_classes"]).astype(float)
        criterion = LogitAdjustedLoss(counts / counts.sum(), tau=1.0,
                                      label_smoothing=CFG["label_smoothing"]).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    optimizer = optim.AdamW(
        model.get_param_groups(backbone_lr=lr * CFG["backbone_lr_ratio"], head_lr=lr),
        weight_decay=CFG["weight_decay"],
    )
    steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
    warmup_epochs = min(CFG["warmup_epochs"], args.epochs)
    scheduler = make_scheduler(optimizer, args.epochs * steps, warmup_epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = os.path.join(SAVE_ROOT, f"{run_name}_s{args.seed}")   # 每个种子单独保存（第四轮起）
    os.makedirs(save_dir, exist_ok=True)

    best, history, t0 = None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  [{run_name}] Ep{epoch:02d}", leave=False)
        for i, (imgs, labels) in enumerate(pbar):
            if args.max_batches and i >= args.max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs, ya, yb, lam = mixup_data(imgs, labels, CFG["mixup_alpha"], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = mixup_criterion(criterion, model(imgs), ya, yb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        preds, labels_all = predict(model, val_loader, device, use_amp, args.max_batches)
        m = compute_metrics(preds, labels_all, val_ds.sources[:len(preds)])
        history.append({"epoch": epoch, "train_loss": total_loss / max(n, 1),
                        **{k: v for k, v in m.items() if k != "per_class_f1"}})
        print(f"  Ep{epoch:02d}  loss={total_loss / max(n, 1):.4f}  "
              f"ALL={fmt(m['acc_all'])}  SHOP={fmt(m['acc_shop'])}  USER={fmt(m['acc_user'])}  "
              f"MacroF1={fmt(m['macro_f1'])}  RareF1={fmt(m['rare_f1'])}")

        # 与论文一致：按整体 val acc 选最佳 epoch
        if best is None or m["acc_all"] > best["acc_all"]:
            best = dict(m, best_epoch=epoch)
            if not args.smoke:
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "metrics": m, "backbone": backbone, "rare_aug": use_rare},
                           os.path.join(save_dir, "best_model.pth"))

    minutes = (time.time() - t0) / 60
    print(f"[{run_name}] 完成，用时 {minutes:.1f} 分钟，最佳 epoch {best['best_epoch']}  "
          f"ALL={fmt(best['acc_all'])}  RareF1={fmt(best['rare_f1'])}")

    result = {
        "run": run_name, "backbone": backbone, "params_M": round(n_params, 1),
        "rare_aug": use_rare, "aug_policy": aug, "loss": loss_name, "sampler": sampler_name,
        "head_lr": lr, "dropout": dropout, "seed": args.seed,
        "epochs": args.epochs, "minutes": round(minutes, 1),
        **best, "final": dict(m, epoch=args.epochs), "history": history,
    }

    del model, optimizer, scheduler, scaler, train_loader, train_ds
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：LA 损失、类别均衡采样、Large+CB × 1 epoch × 30 batch")
    ap.add_argument("--out", default="capacity_results.json", help="结果文件")
    args = ap.parse_args()

    args.max_batches = 0
    result_file = args.out
    if args.smoke:
        args.runs, args.epochs, args.max_batches = ["tiny_norare_la_uni", "tiny_norare_ce_cbs", "large_norare_cb_uni"], 1, 30
        result_file = "capacity_smoke.json"
        if os.path.exists(result_file):
            os.remove(result_file)

    for r in args.runs:
        parse_run(r)

    device = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root = os.path.join(CFG["crops_dir"], "validation")

    val_ds = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    results = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)

    for r in args.runs:
        key = r if args.seed == 42 else f"{r}_s{args.seed}"   # seed 42 沿用原键名，其他种子加后缀
        if key in results:
            print(f"[{key}] 已有结果，跳过")
            continue
        results = save_result(result_file, key,
                              run_one(r, args, device, use_amp, train_root, val_ds, val_loader))
        print(f"  → 已写入 {result_file}")

    # 汇总表
    print("\n" + "=" * 86)
    print(f"{'Run':26s}{'Params':>8s}{'ALL':>9s}{'SHOP':>9s}{'USER':>9s}"
          f"{'MacroF1':>9s}{'RareF1':>9s}{'BestEp':>8s}{'Min':>8s}")
    print("-" * 86)
    for r in args.runs:
        key = r if args.seed == 42 else f"{r}_s{args.seed}"
        if key not in results:
            continue
        x = results[key]
        print(f"{key:26s}{x['params_M']:>7.1f}M{fmt(x['acc_all']):>9s}{fmt(x['acc_shop']):>9s}"
              f"{fmt(x['acc_user']):>9s}{fmt(x['macro_f1']):>9s}{fmt(x['rare_f1']):>9s}"
              f"{x['best_epoch']:>8d}{x['minutes']:>8.1f}")
    print("=" * 86)
    if args.smoke:
        print("冒烟测试通过（数字没有意义，只确认能跑）。现在可以正式运行。")


if __name__ == "__main__":
    main()
