from .dinov3 import (
    DINOv3Backbone,
    DINOv3SmallBackbone,
    DINOv3LargeBackbone,
    DINOv3GiantBackbone,
    DINOv3PatchesBackbone,
)

BACKBONE_REGISTRY = {
    "dinov3": DINOv3Backbone,
    "dinov3-small": DINOv3SmallBackbone,
    "dinov3-large": DINOv3LargeBackbone,
    "dinov3-giant": DINOv3GiantBackbone,
    "dinov3-patches": DINOv3PatchesBackbone,
}


def build_backbone(name: str):
    """Build a backbone by name. Returns (backbone_module, feature_dim)."""
    cls = BACKBONE_REGISTRY[name]
    return cls(), cls.DIM
