# df1_sweep.py
# 第二个真实长尾数据集：DeepFashion（第一代）类别分类，复现"容量 × 稀少类增强"交互。
# 先运行 prepare_df1.py 生成 crops_df1/{train,val,test}/。
#
# 协议（尽量与 DeepFashion2 一致，并修正"用同一个集合挑模型又报告"的问题）：
#   - 用官方 val 挑最佳 epoch，在官方 test 上报告；同时报告最后一个 epoch 在 test 上的结果
#   - 稀少类 = 训练样本数最少的 30% 类别（与 DF2 的 4/13、CIFAR 的 30/100 比例相近）
#   - 采样 = 逆类别频率 × 稀少类 boost 3.0（对应 3D-W，DF1 没有 shop/user 之分）
#   - 稀少类增强 = 与 DF2 完全相同的 rare 专项增强；其余类用 DF2 的标准（shop）增强
#   - CE + label smoothing 0.1、Mixup 0.2、AdamW、warmup 3 + cosine、224×224、batch 64
#   - 两臂（有/无稀少类增强）采样完全相同，只差增强
#
# 用法（在 garment_type 目录下）：
#   冒烟测试：  python df1_sweep.py --smoke
#   正式运行：  nohup python df1_sweep.py --seeds 42 123 2024 > log_df1.txt 2>&1 &
# 结果写入 df1_results.json，可断点续跑。

import os, gc, time, json, argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CFG
from dataset import get_train_transforms, get_train_transforms_rare, get_val_transforms, IMG_SIZE
from model import FashionClassifier
from train import mixup_data, mixup_criterion, make_scheduler
from capacity_sweep import BACKBONES, set_seed, fmt, save_result

DEFAULT_SIZES = ["tiny", "small", "base"]
RARE_FRACTION = 0.30
RARE_BOOST = 3.0
SAVE_ROOT = "checkpoints_df1"


class DF1Dataset(Dataset):
    def __init__(self, root, class_ids, train=False, rare=None, use_rare_aug=False):
        self.idx_of = {c: i for i, c in enumerate(class_ids)}
        self.samples = []
        for c in class_ids:
            d = Path(root) / c
            if d.exists():
                self.samples += [(str(p), self.idx_of[c]) for p in sorted(d.glob("*.jpg"))]
        self.train, self.rare, self.use_rare_aug = train, rare or set(), use_rare_aug
        self.tf_std = get_train_transforms(IMG_SIZE) if train else get_val_transforms(IMG_SIZE)
        self.tf_rare = get_train_transforms_rare(IMG_SIZE) if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = cv2.imread(path)
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8) if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf = self.tf_rare if (self.train and self.use_rare_aug and y in self.rare) else self.tf_std
        return tf(image=img)["image"], y

    def labels(self):
        return np.array([y for _, y in self.samples])


@torch.no_grad()
def evaluate(model, loader, device, use_amp, K, rare, max_batches=0):
    model.eval()
    P, Y = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.amp.autocast("cuda", enabled=use_amp):
            P.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    present = sorted(set(Y.tolist()))
    f1 = f1_score(Y, P, labels=present, average=None, zero_division=0)
    f1d = dict(zip(present, f1))
    rare_present = [c for c in present if c in rare]
    return {
        "acc": float((P == Y).mean()),
        "macro_f1": float(np.mean(f1)),
        "rare_f1": float(np.mean([f1d[c] for c in rare_present])) if rare_present else None,
        "common_f1": float(np.mean([f1d[c] for c in present if c not in rare])),
        "per_class_f1": {int(c): float(v) for c, v in f1d.items()},
    }


def run_one(size, use_rare, seed, args, data, device, use_amp):
    class_ids, rare, train_root, val_loader, test_loader = data
    backbone, lr, dropout = BACKBONES[size]
    K = len(class_ids)
    set_seed(seed)
    name = f"{size}_{'rare' if use_rare else 'norare'}{'_uni' if args.sampler == 'uniform' else ''}_s{seed}"
    print("\n" + "=" * 70 + f"\n[DF1 {name}] backbone={backbone} lr={lr} dropout={dropout}\n" + "=" * 70)

    ds = DF1Dataset(train_root, class_ids, train=True, rare=rare, use_rare_aug=use_rare)
    y = ds.labels()
    cnt = np.bincount(y, minlength=K).astype(float)
    if args.sampler == "uniform":
        loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True,
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    else:
        w = (1.0 / np.maximum(cnt, 1))[y] * np.where(np.isin(y, list(rare)), RARE_BOOST, 1.0)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=CFG["batch_size"], sampler=sampler,
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

    model = FashionClassifier(backbone=backbone, pretrained=True, num_classes=K, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    opt = optim.AdamW(model.get_param_groups(backbone_lr=lr * CFG["backbone_lr_ratio"], head_lr=lr),
                      weight_decay=CFG["weight_decay"])
    steps = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    sched = make_scheduler(opt, args.epochs * steps, min(CFG["warmup_epochs"], args.epochs) * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, "best_model.pth")

    best_acc, best_ep, history, t0 = -1, 0, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for i, (x, t) in enumerate(tqdm(loader, desc=f"  [{name}] Ep{ep:02d}", leave=False)):
            if args.max_batches and i >= args.max_batches:
                break
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            x, ya, yb, lam = mixup_data(x, t, CFG["mixup_alpha"], device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = mixup_criterion(crit, model(x), ya, yb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
        v = evaluate(model, val_loader, device, use_amp, K, rare, args.max_batches)
        history.append({"epoch": ep, **{k: v[k] for k in ("acc", "macro_f1", "rare_f1", "common_f1")}})
        print(f"  Ep{ep:02d} val: acc={fmt(v['acc'])} macro={fmt(v['macro_f1'])} rare={fmt(v['rare_f1'])}")
        if v["acc"] > best_acc:
            best_acc, best_ep = v["acc"], ep
            torch.save(model.state_dict(), ckpt)

    test_final = evaluate(model, test_loader, device, use_amp, K, rare, args.max_batches)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_best = evaluate(model, test_loader, device, use_amp, K, rare, args.max_batches)
    minutes = (time.time() - t0) / 60
    print(f"[{name}] {minutes:.1f} 分钟 | test(best ep{best_ep}): acc={fmt(test_best['acc'])} "
          f"rare={fmt(test_best['rare_f1'])} | test(final): acc={fmt(test_final['acc'])} rare={fmt(test_final['rare_f1'])}")
    if args.smoke:
        os.remove(ckpt)
    del model, opt, sched, scaler, loader, ds
    gc.collect(); torch.cuda.empty_cache()
    return {"size": size, "sampler": args.sampler, "backbone": backbone, "params_M": round(n_params, 1), "rare_aug": use_rare,
            "seed": seed, "epochs": args.epochs, "minutes": round(minutes, 1), "best_epoch": best_ep,
            "test_best": test_best, "test_final": test_final, "val_history": history}


def summarize(results, sizes, seeds):
    print("\n" + "=" * 96)
    print("DeepFashion1 test set: rare aug − no rare aug (paired by seed), mean ± std")
    for tag in ("test_best", "test_final"):
        print(f"  [{tag}]")
        for s in sizes:
            for metric in ("acc", "macro_f1", "rare_f1"):
                d = [results[f"{s}_rare_s{x}"][tag][metric] - results[f"{s}_norare_s{x}"][tag][metric]
                     for x in seeds if f"{s}_rare_s{x}" in results and f"{s}_norare_s{x}" in results]
                base = [results[f"{s}_norare_s{x}"][tag][metric] for x in seeds if f"{s}_norare_s{x}" in results]
                if d:
                    print(f"    {s:6s} {metric:9s} no-aug {np.mean(base):.4f}   Δ {np.mean(d):+.4f} ± {np.std(d):.4f}  (n={len(d)})")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="crops_df1")
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="df1_results.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sampler", choices=["resample", "uniform"], default="resample",
                    help="resample = 逆频率 × 稀少类 3 倍（论文 3D-W 对应）；uniform = 均匀 shuffle")
    ap.add_argument("--arms", choices=["both", "norare", "rare"], default="both")
    args = ap.parse_args()
    args.max_batches = 0
    if args.smoke:
        args.sizes, args.seeds, args.epochs, args.max_batches, args.out = ["tiny"], [42], 1, 20, "df1_smoke.json"
        if os.path.exists(args.out):
            os.remove(args.out)

    device, use_amp = torch.device(CFG["device"]), CFG["use_amp"]
    train_root = os.path.join(args.root, "train")
    class_ids = sorted(d.name for d in Path(train_root).iterdir() if d.is_dir())
    counts = {c: len(list((Path(train_root) / c).glob("*.jpg"))) for c in class_ids}
    order = sorted(range(len(class_ids)), key=lambda i: counts[class_ids[i]])
    rare = set(order[:max(1, round(RARE_FRACTION * len(class_ids)))])
    print(f"DF1: {len(class_ids)} 类，训练集 {sum(counts.values())} 张，"
          f"最多 {max(counts.values())} / 最少 {min(counts.values())}，稀少类 {len(rare)} 个："
          f"{[class_ids[i] for i in sorted(rare)]}")

    mk = lambda split: DataLoader(DF1Dataset(os.path.join(args.root, split), class_ids),
                                  batch_size=CFG["batch_size"] * 2, shuffle=False,
                                  num_workers=CFG["num_workers"], pin_memory=True)
    data = (class_ids, rare, train_root, mk("val"), mk("test"))

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for seed in args.seeds:
        for size in args.sizes:
            arms = {"both": (False, True), "norare": (False,), "rare": (True,)}[args.arms]
            for use_rare in arms:
                key = f"{size}_{'rare' if use_rare else 'norare'}{'_uni' if args.sampler == 'uniform' else ''}_s{seed}"
                if key in results:
                    print(f"[{key}] 已有结果，跳过"); continue
                results = save_result(args.out, key, run_one(size, use_rare, seed, args, data, device, use_amp))
    summarize(results, args.sizes, args.seeds)
    if args.smoke:
        print("冒烟测试通过（数字没有意义，只确认能跑）。")


if __name__ == "__main__":
    main()
