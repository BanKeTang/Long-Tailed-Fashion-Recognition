# prepare_df1.py
# 把 DeepFashion（第一代）Category and Attribute Prediction Benchmark 按 bbox 裁剪成
#   crops_df1/{train,val,test}/<类别号两位数>/<序号>.jpg
# 与 DeepFashion2 的处理一致：bbox 四周加 10% padding。
#
# 需要的原始文件（官方 Google Drive 下载，不需要密码），放在同一个根目录下：
#   <root>/Img/img/...                        （img.zip 解压后的 img 文件夹）
#   <root>/Anno/list_category_img.txt
#   <root>/Anno/list_category_cloth.txt
#   <root>/Anno/list_bbox.txt
#   <root>/Eval/list_eval_partition.txt
# 也兼容这些 txt 直接放在 <root>/ 下、或 Anno_coarse/ 等目录下（自动查找）。
#
# 用法：python prepare_df1.py --root /path/to/DeepFashion1 --out crops_df1

import os, argparse, glob, json
from collections import Counter
import cv2
from tqdm import tqdm


def find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"在 {root} 下找不到 {name}")
    return sorted(hits, key=len)[0]


def read_list(path):
    """官方格式：第 1 行条目数，第 2 行表头，之后每行一条，空格分隔"""
    with open(path) as f:
        lines = f.read().strip().splitlines()
    return [ln.split() for ln in lines[2:] if ln.strip()]


def find_img_root(root, sample_rel):
    """sample_rel 形如 img/Sheer_Pleated-Front_Blouse/img_00000001.jpg，找出它相对的根目录"""
    for cand in [root, os.path.join(root, "Img"), os.path.join(root, "img")]:
        if os.path.exists(os.path.join(cand, sample_rel)):
            return cand
    hits = glob.glob(os.path.join(root, "**", os.path.basename(os.path.dirname(sample_rel))), recursive=True)
    for h in hits:
        cand = os.path.dirname(os.path.dirname(h))
        if os.path.exists(os.path.join(cand, sample_rel)):
            return cand
    raise FileNotFoundError(f"找不到图片 {sample_rel}，请确认 img.zip 已解压")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="crops_df1")
    ap.add_argument("--padding", type=float, default=0.1)
    args = ap.parse_args()

    cat = {r[0]: int(r[1]) for r in read_list(find(args.root, "list_category_img.txt"))}
    part = {r[0]: r[1] for r in read_list(find(args.root, "list_eval_partition.txt"))}
    bbox = {r[0]: tuple(map(int, r[1:5])) for r in read_list(find(args.root, "list_bbox.txt"))}
    names = [r[0] for r in read_list(find(args.root, "list_category_cloth.txt"))]   # 第 k 行 = 类别 k

    img_root = find_img_root(args.root, next(iter(cat)))
    print(f"图片根目录: {img_root}")
    print(f"标注：类别 {len(cat)}，划分 {len(part)}，bbox {len(bbox)}，类别名 {len(names)}")

    counts = {s: Counter() for s in ("train", "val", "test")}
    failed = 0
    for i, (rel, label) in enumerate(tqdm(sorted(cat.items()), desc="裁剪")):
        split = part.get(rel)
        if split not in counts or rel not in bbox:
            failed += 1
            continue
        img = cv2.imread(os.path.join(img_root, rel))
        if img is None:
            failed += 1
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox[rel]
        px, py = int((x2 - x1) * args.padding), int((y2 - y1) * args.padding)
        x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            failed += 1
            continue
        d = os.path.join(args.out, split, f"{label:02d}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{i:06d}.jpg"), crop)
        counts[split][label] += 1

    meta = {"names": {k + 1: n for k, n in enumerate(names)},
            "counts": {s: dict(sorted(c.items())) for s, c in counts.items()}, "failed": failed}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\n完成。失败/跳过 {failed} 张")
    for s, c in counts.items():
        print(f"  {s:5s}: {sum(c.values()):7d} 张，{len(c)} 类")
    tr = counts["train"]
    print("训练集各类数量（从多到少）：")
    for k, v in tr.most_common():
        print(f"  {k:02d} {names[k - 1] if k <= len(names) else '?':20s} {v:6d}")


if __name__ == "__main__":
    main()
