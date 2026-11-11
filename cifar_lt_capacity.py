# cifar_lt_capacity.py
# 在第二个数据集（CIFAR-100-LT, imbalance ratio = 100）上复现"容量 × 稀少类增强"交互。
#
# 设计：只改变网络宽度，深度与结构完全不变 —— 用纯容量变量回应"容量与架构混淆"的质疑。
#   ResNet-32 宽度倍数 k ∈ {1, 2, 4, 8}  →  约 0.47M / 1.9M / 7.4M / 29.6M 参数
#   × {不加尾部增强, 加尾部增强} × 3 个种子 = 24 次
# 其余设置照搬 DeepFashion2 实验的逻辑：
#   尾部类 = 训练样本 < 20 的 30 个类（few-shot），约占 30% 类别，与 DeepFashion2 的 4/13 接近
#   采样 = 逆频率 × 尾部 boost 3.0（对应 3D-W，两臂完全相同）
#   CE + label smoothing 0.1、Mixup 0.2
#   尾部增强 = 对应 rare 专项增强：大范围 RandomResizedCrop、强颜色抖动、随机灰度、
#              弹性形变、透视、随机擦除（只作用于尾部类样本）
# 训练协议同 cifar_lt_conflict.py：SGD 0.1、200 epochs、warmup 5、160/180 降学习率
#
# 用法（等 cifar_lt_conflict.py 跑完再跑，或同时跑也可以）：
#   冒烟测试：  python cifar_lt_capacity.py --smoke
#   正式运行：  nohup python cifar_lt_capacity.py > log_cifar_capacity.txt 2>&1 &
# 结果写入 cifar_lt_capacity_results.json，可断点续跑。

import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from cifar_lt_conflict import BasicBlock, set_seed, lr_at, evaluate, NUM_CLASSES

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
TAIL_THRESHOLD = 20      # 训练样本 < 20 的类为尾部类
RARE_BOOST = 3.0


# ============================================================
# 宽度可调的 ResNet-32（深度固定，只改通道数）
# ============================================================

class WideResNet32(nn.Module):
    def __init__(self, k=1, num_classes=NUM_CLASSES):
        super().__init__()
        w = [16 * k, 32 * k, 64 * k]
        self.conv = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(w[0])
        layers, inp = [], w[0]
        for out, stride in [(w[0], 1), (w[1], 2), (w[2], 2)]:
            for i in range(5):
                layers.append(BasicBlock(inp, out, stride if i == 0 else 1))
                inp = out
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(w[2], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ============================================================
# 数据
# ============================================================

def std_transform():
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def tail_transform():
    ops = [
        T.RandomResizedCrop(32, scale=(0.4, 1.0), ratio=(0.5, 2.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.6, 0.6, 0.6, 0.25)], p=0.9),
        T.RandomGrayscale(p=0.15),
        T.RandomPerspective(distortion_scale=0.2, p=0.4),
    ]
    if hasattr(T, "ElasticTransform"):   # torchvision >= 0.13
        ops.append(T.RandomApply([T.ElasticTransform(alpha=30.0, sigma=4.0)], p=0.3))
    ops += [T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2))]
    return T.Compose(ops)


class LTDataset(Dataset):
    """CIFAR-100-LT 训练集；use_tail_aug=True 时尾部类样本用尾部增强，其余用标准增强"""
    def __init__(self, base, idx, labels, tail_classes, use_tail_aug):
        self.base, self.idx, self.labels = base, idx, labels
        self.tail = set(tail_classes)
        self.use_tail_aug = use_tail_aug
        self.tf_std, self.tf_tail = std_transform(), tail_transform()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img, y = self.base[self.idx[i]]          # base 不带 transform，返回 PIL
        tf = self.tf_tail if (self.use_tail_aug and y in self.tail) else self.tf_std
        return tf(img), y


def build(root, ratio):
    base = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=None)
    test = torchvision.datasets.CIFAR100(root, train=False, download=True,
                                         transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    targets = np.array(base.targets)
    counts = np.array([int(500 * (1.0 / ratio) ** (i / (NUM_CLASSES - 1))) for i in range(NUM_CLASSES)])
    idx = []
    for c in range(NUM_CLASSES):
        idx.extend(np.where(targets == c)[0][:counts[c]].tolist())   # 与 cifar_lt_conflict.py 同一子集
    labels = targets[idx]
    tail = [c for c in range(NUM_CLASSES) if counts[c] < TAIL_THRESHOLD]
    return base, idx, labels, test, counts, tail


# ============================================================
# 训练
# ============================================================

def run_one(k, use_tail_aug, seed, args, data, device):
    base, idx, labels, test, counts, tail = data
    set_seed(seed)

    ds = LTDataset(base, idx, labels, tail, use_tail_aug)
    w = 1.0 / counts[labels]
    w = w * np.where(np.isin(labels, tail), RARE_BOOST, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                    num_samples=len(labels), replacement=True)
    train_loader = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=6,
                              pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = WideResNet32(k).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    t0 = time.time()
    for epoch in range(args.epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch, 0.1)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            perm = torch.randperm(x.size(0), device=device)
            x = lam * x + (1 - lam) * x[perm]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = lam * F.cross_entropy(logits, y, label_smoothing=0.1) + \
                   (1 - lam) * F.cross_entropy(logits, y[perm], label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            m = evaluate(model, test_loader, counts, device)
            print(f"  [k={k} tail_aug={use_tail_aug} s{seed}] ep{epoch + 1:3d}  acc={m['acc']:.4f}  "
                  f"many={m['many']:.4f}  med={m['medium']:.4f}  few={m['few']:.4f}")

    m = evaluate(model, test_loader, counts, device)
    m.update(k=k, params_M=round(n_params, 2), tail_aug=use_tail_aug, seed=seed,
             minutes=round((time.time() - t0) / 60, 1))
    return m


def summarize(results, widths, seeds):
    print("\n" + "=" * 92)
    print("Capacity x tail augmentation on CIFAR-100-LT (mean ± std over seeds, %)")
    print(f"  {'Width':6s}{'Params':>8s}{'Aug':>6s}{'Overall':>15s}{'Few':>15s}"
          f"{'Δ Overall':>15s}{'Δ Few':>15s}")
    for k in widths:
        rows = {a: [results[f"k{k}_{a}_s{s}"] for s in seeds if f"k{k}_{a}_s{s}" in results]
                for a in ("noaug", "aug")}
        for a in ("noaug", "aug"):
            if not rows[a]:
                continue
            acc = np.array([r["acc"] for r in rows[a]]) * 100
            few = np.array([r["few"] for r in rows[a]]) * 100
            line = (f"  k={k:<4d}{rows[a][0]['params_M']:>7.2f}M{('yes' if a == 'aug' else 'no'):>6s}"
                    f"{acc.mean():>9.2f}±{acc.std():4.2f}{few.mean():>9.2f}±{few.std():4.2f}")
            if a == "aug" and rows["noaug"]:
                # 按种子配对求差（同一种子两臂初始化相同）
                pairs = [(results[f"k{k}_aug_s{s}"], results[f"k{k}_noaug_s{s}"]) for s in seeds
                         if f"k{k}_aug_s{s}" in results and f"k{k}_noaug_s{s}" in results]
                if pairs:
                    da = np.array([p[0]["acc"] - p[1]["acc"] for p in pairs]) * 100
                    df = np.array([p[0]["few"] - p[1]["few"] for p in pairs]) * 100
                    line += f"{da.mean():>+9.2f}±{da.std():4.2f}{df.mean():>+9.2f}±{df.std():4.2f}"
            print(line)
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=100)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--data_root", default="./data_cifar")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.max_batches = 0
    result_file = "cifar_lt_capacity_results.json"
    if args.smoke:
        args.widths, args.seeds, args.epochs, args.max_batches = [1, 8], [42], 1, 10
        result_file = "cifar_lt_capacity_smoke.json"
        if os.path.exists(result_file):
            os.remove(result_file)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    data = build(args.data_root, args.ratio)
    print(f"CIFAR-100-LT: {len(data[2])} 张训练图，尾部类 {len(data[5])} 个（样本 < {TAIL_THRESHOLD}）")

    results = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)

    for seed in args.seeds:
        for k in args.widths:
            for a in ("noaug", "aug"):
                key = f"k{k}_{a}_s{seed}"
                if key in results:
                    print(f"[{key}] 已有结果，跳过")
                    continue
                print(f"\n>>> {key}")
                results[key] = run_one(k, a == "aug", seed, args, data, device)
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → 完成，{results[key]['params_M']}M 参数，用时 {results[key]['minutes']} 分钟")

    summarize(results, args.widths, args.seeds)
    if args.smoke:
        print("冒烟测试通过（数字没有意义，只确认能跑）。")


if __name__ == "__main__":
    main()
