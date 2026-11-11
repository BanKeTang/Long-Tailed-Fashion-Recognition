# cifar_lt_conflict.py
# 在 CIFAR-100-LT（imbalance ratio = 100）上验证论文的两个"不兼容"发现：
#   冲突 1：CB Loss 与 类别重采样（re-sampling）叠加
#   冲突 2：Focal Loss 与 Mixup 叠加
# 每个冲突都跑 2×2 析因设计：基线 / 只加 A / 只加 B / A+B，3 个随机种子，报告 mean ± std。
#
# 完全独立，不依赖 garment_type 里的任何代码。首次运行会自动下载 CIFAR-100 到 ./data_cifar
# 设置沿用 LDAM-DRW (Cao et al., 2019) 的标准 CIFAR-LT 协议：
#   ResNet-32、SGD lr 0.1、momentum 0.9、wd 2e-4、batch 128、200 epochs、
#   前 5 epoch 线性 warmup，第 160 / 180 epoch 学习率各降 100 倍
# Mixup alpha = 0.2、Focal gamma = 1.5、CB beta = 0.9999，与 DeepFashion2 实验一致。
#
# 用法：
#   冒烟测试：  python cifar_lt_conflict.py --smoke
#   正式运行：  nohup python cifar_lt_conflict.py > log_cifar.txt 2>&1 &
# 每次跑完立即写入 cifar_lt_results.json；已完成的自动跳过，可断点续跑。

import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

NUM_CLASSES = 100

# 7 个不同配置（CE 基线两个冲突共用）
EXPS = {
    "CE":          dict(loss="ce",    resample=False, mixup=False),
    "CE+RS":       dict(loss="ce",    resample=True,  mixup=False),
    "CB":          dict(loss="cb",    resample=False, mixup=False),
    "CB+RS":       dict(loss="cb",    resample=True,  mixup=False),
    "CE+Mixup":    dict(loss="ce",    resample=False, mixup=True),
    "Focal":       dict(loss="focal", resample=False, mixup=False),
    "Focal+Mixup": dict(loss="focal", resample=False, mixup=True),
}
GROUPS = {
    "Conflict 1: CB Loss x Re-sampling": ["CE", "CE+RS", "CB", "CB+RS"],
    "Conflict 2: Focal Loss x Mixup":    ["CE", "CE+Mixup", "Focal", "Focal+Mixup"],
}


# ============================================================
# ResNet-32（CIFAR 版）
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(out)
        self.sh = nn.Sequential()
        if stride != 1 or inp != out:
            self.sh = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sh(x))


class ResNet32(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers, inp = [], 16
        for out, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(5):
                layers.append(BasicBlock(inp, out, stride if i == 0 else 1))
                inp = out
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ============================================================
# 数据：CIFAR-100-LT（指数衰减，n_i = 500 × (1/ratio)^(i/99)）
# ============================================================

def build_data(root, ratio):
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_full = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tf_test)

    targets = np.array(train_full.targets)
    counts = [int(500 * (1.0 / ratio) ** (i / (NUM_CLASSES - 1))) for i in range(NUM_CLASSES)]
    idx = []
    for c in range(NUM_CLASSES):
        idx.extend(np.where(targets == c)[0][:counts[c]].tolist())   # 确定性选取，所有实验相同
    train_lt = Subset(train_full, idx)
    train_labels = targets[idx]
    return train_lt, train_labels, test, np.array(counts)


# ============================================================
# 损失函数
# ============================================================

def make_loss(kind, counts, device, beta=0.9999, gamma=1.5):
    if kind == "ce":
        return lambda logits, y: F.cross_entropy(logits, y)
    if kind == "cb":
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff
        w = w / w.sum() * len(counts)
        w = torch.tensor(w, dtype=torch.float32, device=device)
        return lambda logits, y: F.cross_entropy(logits, y, weight=w)
    if kind == "focal":
        def focal(logits, y):
            ce = F.cross_entropy(logits.float(), y, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()
        return focal
    raise ValueError(kind)


# ============================================================
# 训练 + 评估
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lr_at(epoch, base_lr, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    if epoch >= 180:
        return base_lr * 1e-4
    if epoch >= 160:
        return base_lr * 1e-2
    return base_lr


@torch.no_grad()
def evaluate(model, loader, counts, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    correct = preds == labels
    cls_count = counts[labels]
    return {
        "acc":    float(correct.mean()),
        "many":   float(correct[cls_count > 100].mean()),
        "medium": float(correct[(cls_count <= 100) & (cls_count >= 20)].mean()),
        "few":    float(correct[cls_count < 20].mean()),
    }


def run_one(name, cfg, seed, args, data, device):
    train_lt, train_labels, test, counts = data
    set_seed(seed)

    if cfg["resample"]:
        w = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(train_labels), replacement=True)
        train_loader = DataLoader(train_lt, batch_size=128, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_lt, batch_size=128, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet32().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    criterion = make_loss(cfg["loss"], counts, device, gamma=args.focal_gamma)

    t0 = time.time()
    for epoch in range(args.epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch, 0.1)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cfg["mixup"]:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                logits = model(x)
                loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[perm])
            else:
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            m = evaluate(model, test_loader, counts, device)
            print(f"  [{name} s{seed}] ep{epoch + 1:3d}  acc={m['acc']:.4f}  "
                  f"many={m['many']:.4f}  med={m['medium']:.4f}  few={m['few']:.4f}")

    m = evaluate(model, test_loader, counts, device)   # 按惯例报告最后一个 epoch
    m.update(exp=name, seed=seed, minutes=round((time.time() - t0) / 60, 1), **cfg)
    return m


def summarize(results, seeds):
    print("\n" + "=" * 78)
    for gname, exps in GROUPS.items():
        print(gname)
        print(f"  {'Method':14s}{'Overall':>16s}{'Many':>16s}{'Medium':>16s}{'Few':>16s}")
        for e in exps:
            rows = [results[f"{e}_s{s}"] for s in seeds if f"{e}_s{s}" in results]
            if not rows:
                continue
            cells = []
            for k in ["acc", "many", "medium", "few"]:
                v = np.array([r[k] for r in rows]) * 100
                cells.append(f"{v.mean():6.2f}±{v.std():4.2f}")
            print(f"  {e:14s}" + "".join(f"{c:>16s}" for c in cells) + f"   (n={len(rows)})")
        print()
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--exps", nargs="+", default=list(EXPS.keys()))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=100)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    ap.add_argument("--data_root", default="./data_cifar")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.max_batches = 0
    result_file = "cifar_lt_results.json"
    if args.smoke:
        args.seeds, args.epochs, args.max_batches = [42], 2, 10
        result_file = "cifar_lt_smoke.json"
        if os.path.exists(result_file):
            os.remove(result_file)

    device = torch.device("cuda")
    data = build_data(args.data_root, args.ratio)
    print(f"CIFAR-100-LT: {len(data[1])} 张训练图，最多 {data[3].max()} / 最少 {data[3].min()}，"
          f"imbalance ratio = {args.ratio:g}")

    results = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)

    for seed in args.seeds:
        for e in args.exps:
            key = f"{e}_s{seed}"
            if key in results:
                print(f"[{key}] 已有结果，跳过")
                continue
            print(f"\n>>> {key}  {EXPS[e]}")
            results[key] = run_one(e, EXPS[e], seed, args, data, device)
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → 完成，用时 {results[key]['minutes']} 分钟，已写入 {result_file}")

    summarize(results, args.seeds)
    if args.smoke:
        print("冒烟测试通过（数字没有意义，只确认能跑）。现在可以正式运行。")


if __name__ == "__main__":
    main()
