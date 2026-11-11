# dataset.py

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CATEGORY_NAMES, CFG

NUM_CLASSES = CFG["num_classes"]
IMG_SIZE    = CFG["img_size"]
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)

# 稀少类（训练集样本数 < 2000），用专项增强
# 2=short_sleeve_outwear, 5=sling, 10=long_sleeve_dress, 12=sling_dress
RARE_CLASS_LABELS = {2, 5, 10, 12}


# ============================================================
# 数据增强
# ============================================================

def get_train_transforms(img_size=IMG_SIZE):
    """shop 图标准增强"""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.85),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 60), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8,  min_width=8,
            fill_value=128, p=0.35,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4,
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_train_transforms_user(img_size=IMG_SIZE):
    """user 街拍图专用增强（更激进）"""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.45, 1.0), ratio=(0.6, 1.6)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=50, val_shift_limit=35, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
        ], p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101, p=0.6,
        ),
        A.CoarseDropout(
            max_holes=12, max_height=48, max_width=48,
            min_holes=2,  min_height=16, min_width=16,
            fill_value=128, p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(20, 80), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.2, 0.7), p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
        ], p=0.45),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_train_transforms_rare(img_size=IMG_SIZE):
    """稀少类专项增强：最大多样性弥补样本不足"""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.4, 1.0), ratio=(0.5, 2.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.25, p=1.0),
            A.HueSaturationValue(hue_shift_limit=45, sat_shift_limit=60, val_shift_limit=40, p=1.0),
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1.0),
        ], p=0.9),
        A.ToGray(p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.Perspective(scale=(0.05, 0.2), p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.15, scale_limit=0.2, rotate_limit=25,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
        A.ElasticTransform(
            alpha=60, sigma=6,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.CoarseDropout(
            max_holes=15, max_height=56, max_width=56,
            min_holes=2,  min_height=16, min_width=16,
            fill_value=128, p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(20, 100), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.1), intensity=(0.2, 0.8), p=1.0),
            A.MotionBlur(blur_limit=11, p=1.0),
            A.GaussianBlur(blur_limit=(3, 11), p=1.0),
        ], p=0.45),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ============================================================
# FashionCropDataset：验证集用
# ============================================================

class FashionCropDataset(Dataset):
    def __init__(self, root: str, transforms=None, source_filter: str = None):
        self.transforms    = transforms
        self.source_filter = source_filter
        self.samples       = []   # (img_path, label)
        self.sources       = []   # "shop" or "user"

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"目录不存在: {root}，请先运行 prepare_dataset.py")

        for label, cat_name in enumerate(CATEGORY_NAMES):
            cat_dir = root / cat_name
            if not cat_dir.exists():
                continue
            for img_path in sorted(cat_dir.glob("*.jpg")):
                source = img_path.name.split("_")[0]
                if source_filter and source != source_filter:
                    continue
                self.samples.append((str(img_path), label))
                self.sources.append(source)

        if not self.samples:
            raise RuntimeError(f"未找到任何图片: {root}")

        shop_cnt = self.sources.count("shop")
        user_cnt = self.sources.count("user")
        print(f"[FashionCropDataset] {root.name} | "
              f"total={len(self.samples)} shop={shop_cnt} user={user_cnt} "
              f"filter={source_filter or 'all'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# ============================================================
# FashionMixedDataset：训练集用
# 三维度策略：
#   稀少类           → rare 增强（最激进）
#   user 普通类      → user 增强（激进）
#   shop 普通类      → 标准增强
#   采样权重：类别权重 × rare_boost × user_boost
# ============================================================

class FashionMixedDataset(Dataset):
    def __init__(self, root: str, user_boost: float = 1.5, rare_boost: float = 3.0):
        self.user_boost = user_boost
        self.rare_boost = rare_boost
        self.samples    = []   # (img_path, label, source)
        self.sources    = []

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"目录不存在: {root}，请先运行 prepare_dataset.py")

        self.tfm_shop = get_train_transforms(IMG_SIZE)
        self.tfm_user = get_train_transforms_user(IMG_SIZE)
        self.tfm_rare = get_train_transforms_rare(IMG_SIZE)

        for label, cat_name in enumerate(CATEGORY_NAMES):
            cat_dir = root / cat_name
            if not cat_dir.exists():
                continue
            for img_path in sorted(cat_dir.glob("*.jpg")):
                source = img_path.name.split("_")[0]
                self.samples.append((str(img_path), label, source))
                self.sources.append(source)

        if not self.samples:
            raise RuntimeError(f"未找到任何图片: {root}")

        labels   = [s[1] for s in self.samples]
        counts   = np.bincount(labels, minlength=NUM_CLASSES)
        shop_cnt = self.sources.count("shop")
        user_cnt = self.sources.count("user")
        rare_cnt = sum(1 for s in self.samples if s[1] in RARE_CLASS_LABELS)

        print(f"[FashionMixedDataset] {root.name} | "
              f"total={len(self.samples)} shop={shop_cnt} user={user_cnt} "
              f"rare={rare_cnt} user_boost={user_boost} rare_boost={rare_boost}")
        print("  各类样本数：")
        for i, cnt in enumerate(counts):
            flag = " ← 稀少类" if i in RARE_CLASS_LABELS else ""
            print(f"    {CATEGORY_NAMES[i]:25s}: {cnt:6d}{flag}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if label in RARE_CLASS_LABELS:
            tfm = self.tfm_rare
        elif source == "user":
            tfm = self.tfm_user
        else:
            tfm = self.tfm_shop

        return tfm(image=img)["image"], label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_sample_weights(self) -> torch.FloatTensor:
        labels        = self.get_labels()
        counts        = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        class_weights = 1.0 / np.where(counts > 0, counts, 1.0)
        weights = []
        for (_, label, source) in self.samples:
            w = class_weights[label]
            if label in RARE_CLASS_LABELS:
                w *= self.rare_boost
            if source == "user":
                w *= self.user_boost
            weights.append(w)
        return torch.FloatTensor(weights)