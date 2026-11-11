# prepare_dataset.py
# 功能：解析 DeepFashion2 标注，crop bbox 区域，保存到 crops/ 目录
# 运行：python prepare_dataset.py
# python prepare_dataset.py --data_root "xxx"

import os
import json
import cv2
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import CFG, CATEGORY_NAMES, ID_TO_LABEL, ID_TO_NAME


def parse_deepfashion2(root_dir: str, split: str) -> pd.DataFrame:
    """
    解析 DeepFashion2 标注 JSON，返回 DataFrame。
    columns: image_path, label, category_name, source,
             x1, y1, x2, y2, scale, occlusion, unique_id
    """
    image_dir = Path(root_dir) / split / "image"
    anno_dir  = Path(root_dir) / split / "annos"

    if not anno_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {anno_dir}")

    records = []
    anno_files = sorted(anno_dir.glob("*.json"))

    for anno_file in tqdm(anno_files, desc=f"Parsing {split}"):
        with open(anno_file, "r") as f:
            try:
                anno = json.load(f)
            except json.JSONDecodeError:
                print(f"  [WARN] 跳过损坏文件: {anno_file}")
                continue

        img_name = anno_file.stem + ".jpg"
        img_path = str(image_dir / img_name)
        source   = anno.get("source", "unknown")  # "shop" or "user"

        for key, item in anno.items():
            if not key.startswith("item"):
                continue

            cat_id = item.get("category_id")
            bbox   = item.get("bounding_box")

            if cat_id is None or bbox is None:
                continue
            if cat_id not in ID_TO_LABEL:
                continue
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            # 过滤无效 bbox
            if x2 <= x1 or y2 <= y1:
                continue

            # 用 hash 生成唯一文件名，避免任何重复
            uid = hashlib.md5(
                f"{split}_{anno_file.stem}_{key}".encode()
            ).hexdigest()[:12]

            records.append({
                "image_path":    img_path,
                "label":         ID_TO_LABEL[cat_id],   # 0-12
                "category_id":   cat_id,
                "category_name": ID_TO_NAME[cat_id],
                "source":        source,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "scale":         item.get("scale", 3),
                "occlusion":     item.get("occlusion", 1),
                "unique_id":     uid,
            })

    df = pd.DataFrame(records)
    print(f"  [{split}] 共 {len(df)} 条记录")
    print(df["category_name"].value_counts().to_string())
    print(f"  source 分布: {df['source'].value_counts().to_dict()}")
    return df


def crop_and_save(df: pd.DataFrame, output_dir: str, padding_ratio: float = 0.1):
    """
    crop bbox 区域并保存。
    文件名格式：{source}_{unique_id}.jpg  ← 全局唯一，不会覆盖
    目录结构：output_dir/category_name/{source}_{uid}.jpg
    """
    output_dir = Path(output_dir)
    failed, saved = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping"):
        img_path = row["image_path"]

        if not os.path.exists(img_path):
            failed += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        h, w = img.shape[:2]
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])

        # 加 padding
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * padding_ratio), int(bh * padding_ratio)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            failed += 1
            continue

        cat_dir = output_dir / row["category_name"]
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 唯一文件名：source_uid.jpg
        save_name = f"{row['source']}_{row['unique_id']}.jpg"
        cv2.imwrite(str(cat_dir / save_name), crop)
        saved += 1

    print(f"  保存: {saved}  失败: {failed}")


def main():
    root    = CFG["data_root"]
    pad     = CFG["bbox_padding"]

    for split in ["train", "validation"]:
        print(f"\n{'='*50}")
        print(f"处理 {split} ...")

        df = parse_deepfashion2(root, split)
        df.to_csv(f"df2_{split}.csv", index=False)

        out_dir = Path(CFG["crops_dir"]) / split
        crop_and_save(df, str(out_dir), padding_ratio=pad)

    print("\n✓ 数据准备完成！")
    print(f"  crops 目录: {CFG['crops_dir']}/")
    print("  下一步运行: python train.py")


if __name__ == "__main__":
    main()
