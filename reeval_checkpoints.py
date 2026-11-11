# reeval_checkpoints.py
# 用同一套评估代码，把现有的所有 best checkpoint 重新评估一遍，
# 生成一份可追溯的结果文件 reeval_results.json，作为论文所有已有数字的唯一来源。
#
# - 自动扫描当前目录下所有 checkpoints* 文件夹里的 *.pth（跳过每 5 epoch 存的 epoch_XX.pth，
#   以及本次新实验的 checkpoints_capacity / checkpoints_interaction）
# - 自动识别 backbone（不需要你告诉它每个文件是什么模型）
# - 评估协议与所有训练脚本一致：验证集 52,490 张，Resize 1.14 + CenterCrop 224
# - 记录文件修改时间、checkpoint 里保存的 epoch 和 val_acc，方便核对是哪次运行
#
# 用法（实验跑着的时候也可以跑，只占少量显存，每个 checkpoint 约 1–3 分钟）：
#   python reeval_checkpoints.py
#   python reeval_checkpoints.py --only checkpoints_comparison   # 只评估某个文件夹
# 已评估过的文件会跳过（文件改动过会重新评估），可断点续跑。

import os, re, glob, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import FashionCropDataset, get_val_transforms
from model import FashionClassifier
from capacity_sweep import predict, compute_metrics, fmt

SKIP_DIRS = {"checkpoints_capacity", "checkpoints_interaction"}
# 按分类头输入维度缩小候选范围，再逐个尝试严格加载
CANDIDATES = {
    768:  ["convnext_tiny", "convnext_small"],
    1024: ["convnext_base"],
    1536: ["convnext_large", "efficientnet_b3"],
    2048: ["efficientnet_b5"],
    1280: ["efficientnet_b0", "efficientnet_b1"],
    1408: ["efficientnet_b2"],
    1792: ["efficientnet_b4"],
}


def find_checkpoints(only):
    dirs = sorted(d for d in glob.glob("checkpoints*") if os.path.isdir(d) and d not in SKIP_DIRS)
    if only:
        dirs = [d for d in dirs if d in only]
    files = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.pth"), recursive=True)):
            if re.fullmatch(r"epoch_\d+\.pth", os.path.basename(f)):
                continue
            files.append(f)
    return files


def load_model(state_dict, device):
    in_feat = state_dict["head.0.weight"].shape[0]          # head 第一层是 LayerNorm(in_features)
    for name in CANDIDATES.get(in_feat, []):
        model = FashionClassifier(backbone=name, pretrained=False, num_classes=CFG["num_classes"])
        try:
            model.load_state_dict(state_dict, strict=True)
            return name, model.to(device)
        except RuntimeError:
            del model
    raise RuntimeError(f"无法识别 backbone（head 输入维度 {in_feat}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="只评估这些文件夹")
    ap.add_argument("--out", default="reeval_results.json")
    args = ap.parse_args()

    device = torch.device(CFG["device"])
    use_amp = CFG["use_amp"]
    files = find_checkpoints(args.only)
    print(f"找到 {len(files)} 个 checkpoint：")
    for f in files:
        print("  ", f)

    val_ds = FashionCropDataset(os.path.join(CFG["crops_dir"], "validation"),
                                transforms=get_val_transforms(CFG["img_size"]))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for f in files:
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))
        if f in results and results[f].get("file_mtime") == mtime:
            print(f"[跳过] {f}（已评估）")
            continue
        print(f"\n[评估] {f}")
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            backbone, model = load_model(sd, device)
        except Exception as e:
            print(f"  ✗ 读取失败：{e}")
            results[f] = {"error": str(e), "file_mtime": mtime}
            json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
            continue

        preds, labels = predict(model, val_loader, device, use_amp)
        m = compute_metrics(preds, labels, val_ds.sources)
        results[f] = {
            "backbone": backbone,
            "file_mtime": mtime,
            "saved_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "saved_val_acc": ckpt.get("val_acc") if isinstance(ckpt, dict) else None,
            **m,
        }
        print(f"  {backbone}  epoch={results[f]['saved_epoch']}  ALL={fmt(m['acc_all'])}  "
              f"SHOP={fmt(m['acc_shop'])}  USER={fmt(m['acc_user'])}  "
              f"MacroF1={fmt(m['macro_f1'])}  RareF1={fmt(m['rare_f1'])}")
        json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 118)
    print(f"{'Checkpoint':52s}{'Backbone':17s}{'Modified':>17s}{'ALL':>8s}{'SHOP':>8s}"
          f"{'USER':>8s}{'MacroF1':>8s}{'RareF1':>8s}")
    print("-" * 118)
    for f, x in results.items():
        if "error" in x:
            print(f"{f[:51]:52s}  读取失败：{x['error'][:50]}")
            continue
        print(f"{f[:51]:52s}{x['backbone']:17s}{x['file_mtime']:>17s}{fmt(x['acc_all']):>8s}"
              f"{fmt(x['acc_shop']):>8s}{fmt(x['acc_user']):>8s}{fmt(x['macro_f1']):>8s}{fmt(x['rare_f1']):>8s}")
    print("=" * 118)
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
