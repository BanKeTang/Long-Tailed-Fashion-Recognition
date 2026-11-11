# diag_checkpoints.py —— 机制诊断（只做推理，不训练；可以和正在跑的训练同时运行）
# 回答：稀少类强增强之所以伤害 Rare F1，是因为 (a) 分类器对稀少类"不敢预测"（先验 / 偏置问题，
# 可以事后校正），还是 (b) 特征本身变差（表示问题，校正不了）？
#
# 对每个 checkpoint 计算：
#   1. 稀少类的 recall / precision / 预测数量与真实数量之比（< 1 表示"少报"）
#   2. 最后一层分类器每类权重的范数（CUDA, Ahn et al. 2023 的解释：强增强 → 该类权重范数变小）
#   3. 事后 logit 校正：logits + τ·log(1/π_c)，τ 在验证集偶数下标一半上选（最大化 Macro F1），
#      在奇数下标另一半上报告 —— 选和报告不在同一批图片上
#   4. 最近类均值（NCM）分类器：每类从训练集取至多 500 张（验证集变换），用 backbone 特征算类中心，
#      在验证集上做余弦最近邻 —— 不依赖分类器和类别先验，只看特征质量
#
# 用法（在 garment_type 目录下）：
#   python diag_checkpoints.py            → 写入 diag_results.json，约 1 小时（和训练共用 GPU）
# 注意：同一组的 3 个种子共用一个 checkpoint 目录，目录里只剩最后保存的那个种子；
#       脚本会根据 checkpoint 里存的指标自动认出是哪个种子。

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import CFG, CATEGORY_NAMES
from dataset import FashionCropDataset, get_val_transforms, RARE_CLASS_LABELS
from model import FashionClassifier
from capacity_sweep import parse_run, BACKBONES

RUNS = ["tiny_norare_ce_uni", "tiny_norare", "tiny_norare_cb_uni", "tiny_norare_cb",
        "tiny_rare_ce_uni", "tiny_rare", "tiny_rare_cb_uni", "tiny_rare_cb",
        "tiny_geom", "tiny_photo", "tiny_mild", "tiny_medium"]
RESULT_FILES = ["phase3_results.json", "capacity_results.json", "capacity_seeds.json"]
RARE = sorted(RARE_CLASS_LABELS)
K = CFG["num_classes"]


def identify_seed(run, ckpt_metrics):
    for f in RESULT_FILES:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for key in (run, f"{run}_s123", f"{run}_s2024"):
            x = d.get(key)
            if x and abs(x["acc_all"] - ckpt_metrics["acc_all"]) < 1e-6 and abs(x["rare_f1"] - ckpt_metrics["rare_f1"]) < 1e-6:
                return 42 if key == run else int(key.rsplit("_s", 1)[1])
    return None


@torch.no_grad()
def forward_all(model, loader, device):
    model.eval()
    L, F, Y = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=CFG["use_amp"]):
            f = model.backbone(x)
            l = model.head(f)
        F.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
        L.append(l.float().cpu())
        Y.append(y)
    return torch.cat(L).numpy(), torch.cat(F).numpy(), torch.cat(Y).numpy()


def scores(pred, y):
    f1 = f1_score(y, pred, labels=list(range(K)), average=None, zero_division=0)
    return {"acc_all": float((pred == y).mean()), "macro_f1": float(f1.mean()), "rare_f1": float(f1[RARE].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--out", default="diag_results.json")
    args = ap.parse_args()
    device = torch.device(CFG["device"])

    tf = get_val_transforms(CFG["img_size"])
    val_ds = FashionCropDataset(os.path.join(CFG["crops_dir"], "validation"), transforms=tf)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    train_ds = FashionCropDataset(os.path.join(CFG["crops_dir"], "train"), transforms=tf)
    ytr = np.array(train_ds.get_labels())
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], min(args.per_class, (ytr == c).sum()), replace=False)
                          for c in range(K)])
    proto_loader = DataLoader(Subset(train_ds, idx), batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    yproto = ytr[idx]

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for run in args.runs:
        path = os.path.join("checkpoints_capacity", run, "best_model.pth")
        if not os.path.exists(path):
            print(f"[{run}] 没有 checkpoint，跳过"); continue
        ck = torch.load(path, map_location="cpu")
        seed = identify_seed(run, ck["metrics"])
        key = f"{run}_s{seed}"
        if key in out:
            print(f"[{key}] 已有，跳过"); continue
        if seed is None:
            print(f"[{run}] checkpoint 与已有结果对不上（可能还在训练），跳过"); continue
        size = parse_run(run)[0]
        backbone, _, dropout = BACKBONES[size]
        model = FashionClassifier(backbone=backbone, pretrained=False, num_classes=K, dropout=dropout)
        model.load_state_dict(ck["state_dict"]); model.to(device)

        logits, feats, y = forward_all(model, val_loader, device)
        pred = logits.argmax(1)
        p, r, _, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
        pred_cnt, true_cnt = np.bincount(pred, minlength=K), np.bincount(y, minlength=K)
        W = model.head[-1].weight.detach().float().cpu().numpy()
        wn = np.linalg.norm(W, axis=1)

        # 事后 logit 校正：偶数下标选 τ，奇数下标报告
        tune, rep = np.arange(len(y)) % 2 == 0, np.arange(len(y)) % 2 == 1
        adj = -np.log(prior)
        taus = np.round(np.arange(0, 2.01, 0.1), 2)
        best_tau = max(taus, key=lambda t: scores((logits[tune] + t * adj).argmax(1), y[tune])["macro_f1"])
        la = {"tau": float(best_tau),
              "report_half_tau0": scores(logits[rep].argmax(1), y[rep]),
              "report_half_tau_best": scores((logits[rep] + best_tau * adj).argmax(1), y[rep])}

        # NCM：训练子集特征的类中心，余弦最近邻
        _, fp, _ = forward_all(model, proto_loader, device)
        cent = np.stack([fp[yproto == c].mean(0) for c in range(K)])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        ncm = scores((feats @ cent.T).argmax(1), y)

        res = {"run": run, "seed": seed, "argmax": scores(pred, y),
               "rare_recall": {CATEGORY_NAMES[c]: float(r[c]) for c in RARE},
               "rare_precision": {CATEGORY_NAMES[c]: float(p[c]) for c in RARE},
               "rare_pred_over_true": {CATEGORY_NAMES[c]: float(pred_cnt[c] / true_cnt[c]) for c in RARE},
               "weight_norm": {CATEGORY_NAMES[c]: float(wn[c]) for c in range(K)},
               "weight_norm_rare_over_common": float(wn[RARE].mean() / np.delete(wn, RARE).mean()),
               "logit_adjust": la, "ncm": ncm}
        out[key] = res
        json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[{key}] rare F1 {res['argmax']['rare_f1']:.4f} | 预测/真实(稀少类均值) "
              f"{np.mean(list(res['rare_pred_over_true'].values())):.2f} | 权重范数比 {res['weight_norm_rare_over_common']:.3f} | "
              f"校正 τ={best_tau}: rare {la['report_half_tau0']['rare_f1']:.4f}→{la['report_half_tau_best']['rare_f1']:.4f} | "
              f"NCM rare {ncm['rare_f1']:.4f}")
        del model; torch.cuda.empty_cache()
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
