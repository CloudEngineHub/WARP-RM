"""DINOv3 ViT backbones for feature extraction. Multiple size ladders."""

import numpy as np
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    """ViT-B/16 — 86M params, 768-dim features. Current production default."""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    DIM = 768
    HF_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.HF_ID)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output.float()


class DINOv3LargeBackbone(DINOv3Backbone):
    """ViT-L/16 — 300M params, 1024-dim features. ~4x precompute cost."""
    DIM = 1024
    HF_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class DINOv3SmallBackbone(DINOv3Backbone):
    """ViT-S/16 — 22M params, 384-dim features. Faster, cheaper features."""
    DIM = 384
    HF_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOv3GiantBackbone(DINOv3Backbone):
    """ViT-G/16 — 1.1B params, 1536-dim features. Heavy; only for comparison."""
    DIM = 1536
    HF_ID = "facebook/dinov3-vitg16-pretrain-lvd1689m"


class DINOv3PatchesBackbone(DINOv3Backbone):
    """ViT-B/16 patch tokens reshaped to 14x14x768 spatial grid.

    Returns the per-patch tokens (last_hidden_state minus the CLS token),
    reshaped to (B, 14, 14, 768). For 224x224 inputs at patch_size=16, the
    grid is exactly 14x14. Per-frame storage is 196x larger than CLS-only
    features (~150 KB/frame vs 1.5 KB/frame).
    """
    DIM = 768  # per-patch dim; spatial grid is (14, 14, DIM)

    def forward(self, x):
        # last_hidden_state: (B, 1 + n_register + N_patches, D)
        # DINOv3 has register tokens between CLS and patches; we want only patches.
        out = self.model(pixel_values=x)
        h = out.last_hidden_state.float()
        B, S, D = h.shape
        # Identify the patch tokens: trailing block of size n_patches = (H/p)^2
        # For ViT-B/16 at 224x224 that's 14*14 = 196.
        side = int(round((S - 1) ** 0.5))
        # Some DINOv3 builds include register tokens; trim to last side*side.
        n_patches = side * side
        patches = h[:, -n_patches:, :]                       # (B, 196, 768)
        return patches.reshape(B, side, side, D).contiguous()  # (B, 14, 14, 768)
