# interaction_runs.py
# 把论文里两个"不兼容"在 DeepFashion2 上补成完整的 2×2 析因设计。
#
# ── 冲突 1：CB-Loss × 3D-W 采样（ConvNeXt-Large）──────────────────────────
# 读 comparison_methods.py 发现：论文里的 CB-Loss 用的是 FashionMixedDataset，
# 本身就带 3D-Aug 增强（含稀少类增强）和 Mixup，只是采样是均匀的。
# 所以 "CB-Loss" 与 "CB-Loss + 3D-Aug" 两行的增强其实相同，真正变化的只有采样。
# 另外 run_combined.py（CB + 3D-W）的 CB 带 label smoothing 0.1，而 comparison_methods 的 CB 不带，
# 所以 CB 那一行两格还差一个 label smoothing。为让 2×2 每一格只差一个因素，补两格：
#                     均匀采样                      3D-W 采样
#   CB-Loss (LS 0.1)  ★ large_cbls_uniform          已有 (run_combined.py, 3 seeds)
#   CE-LS             ★ large_ce_uniform            已有 (Ours, 3 seeds)
# （comparison_methods 的 CB 不带 LS，作为参考保留）
# 两格的数据集、增强、Mixup、lr 2e-4、dropout 0.4 都与 run_combined.py 完全一致，只把采样换成均匀。
#
# ── 冲突 2：Focal（NFL）× Mixup（EfficientNet-B3）────────────────────────
# 论文 Table 4 的 Exp4（CE-LS）和 Exp5（NFL）都开了 Mixup，没有不开 Mixup 的对照。
# 用与 ablation.py --exp 4/5 完全相同的数据集、增强、3D-W 采样，补齐 2×2：
#                     有 Mixup                 无 Mixup
#   CE-LS             b3_ce_mix  (= Exp4)      ★ b3_ce_nomix
#   NFL               b3_nfl_mix (= Exp5)      ★ b3_nfl_nomix
# 有 Mixup 的两格也在本脚本里重跑一遍，保证四格代码路径和随机种子完全一致。
#
# 用法（在 garment_type 目录下）：
#   冒烟测试：  python interaction_runs.py --smoke
#   正式运行：  nohup python interaction_runs.py > log_interaction.txt 2>&1 &
#   补种子：    python interaction_runs.py --seed 123 --runs large_ce_uniform
# 结果写入 interaction_results.json，可断点续跑。

import os, gc, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CFG
from dataset import FashionCropDataset, FashionMixedDataset, get_val_transforms
from model import FashionClassifier
from train import mixup_data, mixup_criterion, make_scheduler
from ablation import AblationDataset, FocalLoss
from capacity_sweep import set_seed, predict, compute_metrics, fmt, save_result
from run_combined import CBLoss          # 与 run_combined.py 同一个实现（beta 0.9999, LS 0.1）

RUNS = {
    # 冲突 2（便宜的先跑）
    "b3_ce_nomix":      dict(backbone="efficientnet_b3", dropout=0.3, lr=3e-4,
                             data="ablation4", sampler="3dw", loss="ce", mixup=False),
    "b3_nfl_nomix":     dict(backbone="efficientnet_b3", dropout=0.3, lr=3e-4,
                             data="ablation4", sampler="3dw", loss="nfl", mixup=False),
    # 冲突 1
    "large_ce_uniform":   dict(backbone="convnext_large", dropout=0.4, lr=2e-4,
                               data="mixed", sampler="uniform", loss="ce", mixup=True),
    "large_cbls_uniform": dict(backbone="convnext_large", dropout=0.4, lr=2e-4,
                               data="mixed", sampler="uniform", loss="cb_ls", mixup=True),
    # 冲突 2 的有 Mixup 两格（与 Exp4 / Exp5 同配置重跑）
    "b3_ce_mix":        dict(backbone="efficientnet_b3", dropout=0.3, lr=3e-4,
                             data="ablation4", sampler="3dw", loss="ce", mixup=True),
    "b3_nfl_mix":       dict(backbone="efficientnet_b3", dropout=0.3, lr=3e-4,
                             data="ablation4", sampler="3dw", loss="nfl", mixup=True),
}
SAVE_ROOT = "checkpoints_interaction"


def build_train(cfg, train_root):
    if cfg["data"] == "ablation4":
        ds = AblationDataset(train_root, 4, is_train=True)          # = ablation.py --exp 4/5
        weights = ds.get_sample_weights(4)                          # 3D-W
    else:
        ds = FashionMixedDataset(train_root, user_boost=1.5, rare_boost=3.0)   # = run_cb_loss
        weights = ds.get_sample_weights()

    if cfg["sampler"] == "3dw":
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=CFG["batch_size"], sampler=sampler,
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True,
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    return ds, loader


def run_one(name, args, device, use_amp, train_root, val_ds, val_loader):
    cfg = RUNS[name]
    set_seed(args.seed)
    print("\n" + "=" * 70)
    print(f"[{name}]  {cfg}  seed={args.seed}")
    print("=" * 70)

    train_ds, train_loader = build_train(cfg, train_root)
    model = FashionClassifier(backbone=cfg["backbone"], pretrained=True,
                              num_classes=CFG["num_classes"], dropout=cfg["dropout"]).to(device)
    if cfg["loss"] == "nfl":
        criterion = FocalLoss(gamma=1.5, label_smoothing=0.1)
    elif cfg["loss"] == "cb_ls":
        counts = np.bincount(train_ds.get_labels(), minlength=CFG["num_classes"]).tolist()
        criterion = CBLoss(counts, beta=0.9999, label_smoothing=0.1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.get_param_groups(backbone_lr=cfg["lr"] * CFG["backbone_lr_ratio"], head_lr=cfg["lr"]),
        weight_decay=CFG["weight_decay"],
    )
    steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
    scheduler = make_scheduler(optimizer, args.epochs * steps,
                               min(CFG["warmup_epochs"], args.epochs) * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_dir = os.path.join(SAVE_ROOT, name if args.seed == 42 else f"{name}_s{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    best, history, t0 = None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"  [{name}] Ep{epoch:02d}", leave=False)):
            if args.max_batches and i >= args.max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if cfg["mixup"]:
                    imgs, ya, yb, lam = mixup_data(imgs, labels, CFG["mixup_alpha"], device)
                    loss = mixup_criterion(criterion, model(imgs), ya, yb, lam)
                else:
                    loss = criterion(model(imgs), labels)
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
        print(f"  Ep{epoch:02d}  loss={total_loss / max(n, 1):.4f}  ALL={fmt(m['acc_all'])}  "
              f"SHOP={fmt(m['acc_shop'])}  USER={fmt(m['acc_user'])}  "
              f"MacroF1={fmt(m['macro_f1'])}  RareF1={fmt(m['rare_f1'])}")
        if best is None or m["acc_all"] > best["acc_all"]:
            best = dict(m, best_epoch=epoch)
            if not args.smoke:
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": m, "cfg": cfg},
                           os.path.join(save_dir, "best_model.pth"))

    minutes = (time.time() - t0) / 60
    print(f"[{name}] 完成，用时 {minutes:.1f} 分钟，最佳 epoch {best['best_epoch']}  "
          f"ALL={fmt(best['acc_all'])}  RareF1={fmt(best['rare_f1'])}")
    result = {"run": name, **cfg, "seed": args.seed, "epochs": args.epochs,
              "minutes": round(minutes, 1), **best, "final": dict(m, epoch=args.epochs),
              "history": history}

    del model, optimizer, scheduler, scaler, train_loader, train_ds
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS.keys()))
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.max_batches = 0
    result_file = "interaction_results.json"
    if args.smoke:
        args.runs, args.epochs, args.max_batches = ["b3_nfl_nomix", "large_cbls_uniform"], 1, 20
        result_file = "interaction_smoke.json"
        if os.path.exists(result_file):
            os.remove(result_file)
    for r in args.runs:
        if r not in RUNS:
            raise ValueError(f"无效的组名: {r}（可选: {', '.join(RUNS)}）")

    device = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]
    train_root = os.path.join(CFG["crops_dir"], "train")
    val_root = os.path.join(CFG["crops_dir"], "validation")
    # 验证集与 ablation.py / comparison_methods.py 相同（Resize 1.14 + CenterCrop）
    val_ds = FashionCropDataset(val_root, transforms=get_val_transforms(CFG["img_size"]))
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    results = json.load(open(result_file)) if os.path.exists(result_file) else {}
    for r in args.runs:
        key = r if args.seed == 42 else f"{r}_s{args.seed}"
        if key in results:
            print(f"[{key}] 已有结果，跳过")
            continue
        results = save_result(result_file, key,
                              run_one(r, args, device, use_amp, train_root, val_ds, val_loader))
        print(f"  → 已写入 {result_file}")

    print("\n" + "=" * 80)
    print(f"{'Run':22s}{'ALL':>9s}{'SHOP':>9s}{'USER':>9s}{'MacroF1':>9s}{'RareF1':>9s}{'Min':>8s}")
    print("-" * 80)
    for key, x in results.items():
        print(f"{key:22s}{fmt(x['acc_all']):>9s}{fmt(x['acc_shop']):>9s}{fmt(x['acc_user']):>9s}"
              f"{fmt(x['macro_f1']):>9s}{fmt(x['rare_f1']):>9s}{x['minutes']:>8.1f}")
    print("=" * 80)
    if args.smoke:
        print("冒烟测试通过（数字没有意义，只确认能跑）。")


if __name__ == "__main__":
    main()
