# model.py
# 统一用 FashionClassifier，通过 timm num_classes=0 + 自建 head 方式
# 支持任意 timm backbone，不依赖各架构的头名字

import torch
import torch.nn as nn
import timm

from config import CFG, CATEGORY_NAMES

NUM_CLASSES = CFG["num_classes"]


class FashionClassifier(nn.Module):
    """
    通用服装分类器。

    backbone 用 timm.create_model(num_classes=0) 提取特征，
    自建 head 做分类，不依赖 .classifier/.head/.fc 等架构差异。

    Args:
        backbone:   timm 模型名，如 "efficientnet_b3", "convnext_tiny", "resnet50"
        pretrained: 是否加载 ImageNet 预训练权重
        num_classes: 分类数，默认 13
        dropout:    分类头 dropout
    """

    def __init__(
        self,
        backbone:    str   = "efficientnet_b3",
        pretrained:  bool  = True,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # num_classes=0：去掉 timm 自带头，只保留特征提取部分
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # 初始化分类头权重
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)   # [B, in_features]
        return self.head(feat)    # [B, num_classes]

    def get_param_groups(self, backbone_lr: float, head_lr: float):
        """
        返回分层学习率的参数组：
        backbone 用小学习率（微调），head 用大学习率（从头训练）
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": self.head.parameters(),     "lr": head_lr,     "name": "head"},
        ]


if __name__ == "__main__":
    print("测试模型...")
    model = FashionClassifier("efficientnet_b3", pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"输出形状: {out.shape}")   # [4, 13]
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {total:.1f}M")

    print("\n测试 ConvNeXt...")
    model2 = FashionClassifier("convnext_tiny", pretrained=False)
    out2   = model2(dummy)
    print(f"输出形状: {out2.shape}")  # [4, 13]
