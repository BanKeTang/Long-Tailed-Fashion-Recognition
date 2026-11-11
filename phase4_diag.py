# phase4_diag.py —— 多种子机制诊断 + cRT（只做推理和线性层重训，不训练 backbone）
# 对 checkpoints_capacity/<run>_s<seed>/best_model.pth（第四轮起每个种子单独保存）计算：
#   1. 训练好的分类器：Rare F1 / Macro F1 / 准确率
#   2. 事后 logit 校正（偶数下标选 τ，奇数下标报告）
#   3. NCM（每类至多 500 张训练图的特征中心，余弦最近邻）
#   4. cRT（Kang et al., 2020）：冻结 backbone 和 head 前几层，重新初始化最后一层线性层，
#      用类别均衡采样在缓存的训练集特征上训练 10 个 epoch
#   5. 分类器权重范数比、稀少类预测数/真实数
# 用法（在 garment_type 目录下，第四轮训练完成后）：
#   nohup python phase4_diag.py > log_p4diag.txt 2>&1 &
# 结果写入 phase4_diag.json，可断点续跑。约每个 checkpoint 15～25 分钟（主要是提取 31 万张训练图特征）。

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support

from config import CFG, CATEGORY_NAMES
from dataset import FashionCropDataset, get_val_transforms, RARE_CLASS_LABELS
from model import FashionClassifier
from capacity_sweep import parse_run, BACKBONES
from sklearn.metrics import f1_score, recall_score


def scores(pred, y):
    f1 = f1_score(y, pred, labels=list(range(CFG["num_classes"])), average=None, zero_division=0)
    rec = recall_score(y, pred, labels=list(range(CFG["num_classes"])), average=None, zero_division=0)
    rare = sorted(RARE_CLASS_LABELS)
    return {"acc_all": float((pred == y).mean()), "macro_f1": float(f1.mean()), "rare_f1": float(f1[rare].mean()),
            "balanced_acc": float(rec.mean()), "rare_recall": float(rec[rare].mean())}

RUNS = ["tiny_norare_ce_uni", "tiny_rare_ce_uni", "tiny_norare_la_uni", "tiny_norare_ce_cbs"]
SEEDS = [42, 123, 2024]
RARE = sorted(RARE_CLASS_LABELS)
K = CFG["num_classes"]


@torch.no_grad()
def extract(model, loader, device):
    """返回：logits、L2 归一化的 backbone 特征、head 倒数第二层（最后一个线性层的输入）特征、标签"""
    model.eval()
    L, F, H, Y = [], [], [], []
    pre = model.head[:-1]            # LayerNorm, Dropout, Linear, GELU, Dropout
    last = model.head[-1]
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=CFG["use_amp"]):
            f = model.backbone(x)
            h = pre(f)
            l = last(h)
        F.append(nn.functional.normalize(f.float(), dim=1).cpu())
        H.append(h.float().cpu()); L.append(l.float().cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(F).numpy(), torch.cat(H), torch.cat(Y).numpy()


def crt(Htr, ytr, Hval, yval, device, epochs=10, seed=0):
    torch.manual_seed(seed)
    clf = nn.Linear(Htr.shape[1], K).to(device)
    cnt = np.bincount(ytr, minlength=K).astype(float)
    w = torch.tensor(1.0 / cnt[ytr], dtype=torch.double)
    opt = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    n = len(ytr); bs = min(512, n); steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    Ht, yt = Htr.to(device), torch.tensor(ytr, device=device)
    for _ in range(epochs):
        idx = torch.multinomial(w, steps * bs, replacement=True).to(device)
        for i in range(steps):
            b = idx[i * bs:(i + 1) * bs]
            loss = nn.functional.cross_entropy(clf(Ht[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = clf(Hval.to(device)).argmax(1).cpu().numpy()
    return scores(pred, yval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default="phase4_diag.json")
    args = ap.parse_args()
    device = torch.device(CFG["device"])
    tf = get_val_transforms(CFG["img_size"])
    val_ds = FashionCropDataset(os.path.join(CFG["crops_dir"], "validation"), transforms=tf)
    train_ds = FashionCropDataset(os.path.join(CFG["crops_dir"], "train"), transforms=tf)
    mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    val_loader, train_loader = mk(val_ds), mk(train_ds)
    ytr_all = np.array(train_ds.get_labels())
    prior = np.bincount(ytr_all, minlength=K) / len(ytr_all)
    rng = np.random.RandomState(0)
    proto_idx = np.concatenate([rng.choice(np.where(ytr_all == c)[0], min(500, (ytr_all == c).sum()), replace=False) for c in range(K)])

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for run in args.runs:
        for seed in args.seeds:
            key = f"{run}_s{seed}"
            path = os.path.join("checkpoints_capacity", key, "best_model.pth")
            if key in out:
                print(f"[{key}] 已有，跳过"); continue
            if not os.path.exists(path):
                print(f"[{key}] 没有 checkpoint，跳过"); continue
            ck = torch.load(path, map_location="cpu")
            backbone, _, dropout = BACKBONES[parse_run(run)[0]]
            model = FashionClassifier(backbone=backbone, pretrained=False, num_classes=K, dropout=dropout)
            model.load_state_dict(ck["state_dict"]); model.to(device)

            logits, feats, Hval, y = extract(model, val_loader, device)
            _, Ftr, Htr, ytr = extract(model, train_loader, device)
            pred = logits.argmax(1)
            p, r, _, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
            pc, tc = np.bincount(pred, minlength=K), np.bincount(y, minlength=K)
            wn = np.linalg.norm(model.head[-1].weight.detach().float().cpu().numpy(), axis=1)

            tune, rep = np.arange(len(y)) % 2 == 0, np.arange(len(y)) % 2 == 1
            adj = -np.log(prior)
            taus = np.round(np.arange(-2.0, 2.01, 0.1), 2)   # 负值：对已做过先验校正的模型（LA 损失）往回调
            bt = max(taus, key=lambda t: scores((logits[tune] + t * adj).argmax(1), y[tune])["macro_f1"])

            cent = np.stack([Ftr[proto_idx][ytr[proto_idx] == c].mean(0) for c in range(K)])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)

            res = {"run": run, "seed": seed, "argmax": scores(pred, y),
                   "rare_pred_over_true": float(np.mean(pc[RARE] / tc[RARE])),
                   "rare_recall": float(np.mean(r[RARE])), "rare_precision": float(np.mean(p[RARE])),
                   "weight_norm_rare_over_common": float(wn[RARE].mean() / np.delete(wn, RARE).mean()),
                   "logit_adjust": {"tau": float(bt), "report_half_tau0": scores(logits[rep].argmax(1), y[rep]),
                                    "report_half_tau_best": scores((logits[rep] + bt * adj).argmax(1), y[rep])},
                   "ncm": scores((feats @ cent.T).argmax(1), y),
                   "crt": crt(Htr, ytr, Hval, y, device, seed=seed)}
            out[key] = res
            json.dump(out, open(args.out, "w"), indent=2)
            print(f"[{key}] trained {res['argmax']['rare_f1']:.4f} | LA {res['logit_adjust']['report_half_tau0']['rare_f1']:.4f}"
                  f"→{res['logit_adjust']['report_half_tau_best']['rare_f1']:.4f} | NCM {res['ncm']['rare_f1']:.4f} | cRT {res['crt']['rare_f1']:.4f}")
            del model; torch.cuda.empty_cache()
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
