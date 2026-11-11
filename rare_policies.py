# rare_policies.py
# Parametric rare-class augmentation for the dose-response study (phase 3).
#
# make_rare_policy(mg, mp) interpolates every parameter linearly between the
# standard shop policy (m = 0) and the aggressive rare policy used in the paper
# (m = 1).  mg scales the geometric / spatial group, mp the photometric group:
#   geometric : RandomResizedCrop, Perspective, ShiftScaleRotate, ElasticTransform,
#               GridDistortion, CoarseDropout (occlusion)
#   photometric: colour jitter / HSV / RGB shift, ToGray, brightness-contrast,
#               noise and blur
# HorizontalFlip (p = 0.5) is common to both policies and always kept.
# make_rare_policy(1, 1) reproduces dataset.get_train_transforms_rare exactly
# (checked by test_equivalence below).  Operators absent from the shop policy
# (Perspective, Elastic, Grid, ToGray) start from probability 0 / magnitude 0.
#
# Named levels used in the paper:
#   mild   = (1/3, 1/3)   medium = (2/3, 2/3)   strong = (1, 1)
#   geom   = (1, 0)       strong geometry, standard photometry
#   photo  = (0, 1)       standard geometry, strong photometry

import math
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

LEVELS = {
    "mild": (1 / 3, 1 / 3),
    "medium": (2 / 3, 2 / 3),
    "strong": (1.0, 1.0),
    "geom": (1.0, 0.0),
    "photo": (0.0, 1.0),
}


def _lerp(a, b, m):
    return a + (b - a) * m


def _odd(x):
    """nearest odd integer >= 3 (blur kernels)"""
    k = int(round(x))
    if k % 2 == 0:
        k += 1
    return max(k, 3)


def _r(x, nd=4):
    return round(x, nd)


def make_rare_policy(mg, mp, img_size=224):
    L = _lerp
    ops = []
    # geometric: crop
    ops.append(A.RandomResizedCrop(
        img_size, img_size,
        scale=(_r(L(0.65, 0.4, mg)), 1.0),
        ratio=(_r(math.exp(L(math.log(0.75), math.log(0.5), mg))),
               _r(math.exp(L(math.log(1.33), math.log(2.0), mg))))))
    ops.append(A.HorizontalFlip(p=0.5))
    # photometric: colour
    ops.append(A.OneOf([
        A.ColorJitter(brightness=_r(L(0.4, 0.6, mp)), contrast=_r(L(0.4, 0.6, mp)),
                      saturation=_r(L(0.4, 0.6, mp)), hue=_r(L(0.15, 0.25, mp)), p=1.0),
        A.HueSaturationValue(hue_shift_limit=round(L(25, 45, mp)),
                             sat_shift_limit=round(L(35, 60, mp)),
                             val_shift_limit=round(L(25, 40, mp)), p=1.0),
        A.RGBShift(r_shift_limit=round(L(20, 40, mp)), g_shift_limit=round(L(20, 40, mp)),
                   b_shift_limit=round(L(20, 40, mp)), p=1.0),
    ], p=_r(L(0.85, 0.9, mp))))
    if mp > 0:
        ops.append(A.ToGray(p=_r(L(0.0, 0.15, mp))))
    ops.append(A.RandomBrightnessContrast(brightness_limit=_r(L(0.2, 0.4, mp)),
                                          contrast_limit=_r(L(0.2, 0.4, mp)),
                                          p=_r(L(0.5, 0.6, mp))))
    # geometric: warps
    if mg > 0:
        ops.append(A.Perspective(scale=(_r(0.05 * mg), _r(0.2 * mg)), p=_r(0.4 * mg)))
    ops.append(A.ShiftScaleRotate(
        shift_limit=_r(L(0.05, 0.15, mg)), scale_limit=_r(L(0.1, 0.2, mg)),
        rotate_limit=round(L(10, 25, mg)),
        border_mode=cv2.BORDER_REFLECT_101, p=_r(L(0.4, 0.7, mg))))
    if mg > 0:
        ops.append(A.ElasticTransform(alpha=_r(60 * mg), sigma=6,
                                      border_mode=cv2.BORDER_REFLECT_101, p=_r(0.3 * mg)))
        ops.append(A.GridDistortion(num_steps=5, distort_limit=_r(0.2 * mg), p=_r(0.2 * mg)))
    # geometric: occlusion
    ops.append(A.CoarseDropout(
        max_holes=round(L(8, 15, mg)), max_height=round(L(32, 56, mg)), max_width=round(L(32, 56, mg)),
        min_holes=round(L(1, 2, mg)), min_height=round(L(8, 16, mg)), min_width=round(L(8, 16, mg)),
        fill_value=128, p=_r(L(0.35, 0.5, mg))))
    # photometric: noise / blur
    ops.append(A.OneOf([
        A.GaussNoise(var_limit=(_r(L(10, 20, mp)), _r(L(60, 100, mp))), p=1.0),
        A.ISONoise(color_shift=(_r(L(0.01, 0.02, mp)), _r(L(0.05, 0.1, mp))),
                   intensity=(_r(L(0.1, 0.2, mp)), _r(L(0.5, 0.8, mp))), p=1.0),
        A.MotionBlur(blur_limit=_odd(L(7, 11, mp)), p=1.0),
        A.GaussianBlur(blur_limit=(3, _odd(L(7, 11, mp))), p=1.0),
    ], p=_r(L(0.3, 0.45, mp))))
    ops += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(ops)


def policy(name, img_size=224):
    mg, mp = LEVELS[name]
    return make_rare_policy(mg, mp, img_size)


def test_equivalence():
    from dataset import get_train_transforms_rare
    a = A.to_dict(make_rare_policy(1.0, 1.0))
    b = A.to_dict(get_train_transforms_rare(224))
    assert a == b, "strong level differs from get_train_transforms_rare"
    print("OK: make_rare_policy(1, 1) == get_train_transforms_rare")


if __name__ == "__main__":
    test_equivalence()
    for k in LEVELS:
        policy(k)
    print("all levels build:", ", ".join(LEVELS))
