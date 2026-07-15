"""
Shared hyperparameters for WARP-RM training/scoring scripts.

Edit this file to change defaults across all entry points. The time-warp
sampling constants (AR(1) alpha, reversal rate, path budget, etc.) live in
`ARSampler.__init__` (`warp_rm/data/samplers.py`).
"""

import os
from pathlib import Path


def warp_rm_s3_bucket() -> str:
    """S3 bucket for cloud artifacts (features, checkpoints, sidecars).

    Required for the optional cloud-training path: set
    `WARP_RM_S3_BUCKET=your-bucket`. Returns "" if unset — local
    training/scoring never calls this.
    """
    return os.environ.get("WARP_RM_S3_BUCKET", "")


def default_feature_cache_dir(backbone: str, feature_stride: int) -> str:
    """Where backbone features land by default.

    Persistent (~/.cache, not /tmp) so reboots don't blow them away. The
    per-dataset subdir is added downstream by `_ep_cache_path` (in
    `warp_rm/utils/caching.py`), giving the final layout:

        <root>/<backbone>_fs<stride>/<dataset>/<key>.npy

    Override with --cache-dir on any training/scoring/render script, or
    set `WARP_RM_FEATURE_CACHE` to a different root for the whole shell.
    """
    root = os.environ.get(
        "WARP_RM_FEATURE_CACHE", str(Path.home() / ".cache" / "warp_rm" / "features")
    )
    return f"{root}/{backbone}_fs{feature_stride}"


# ── Data ─────────────────────────────────────────────────────────────────────
N_VAL = 20
WINDOW_SIZE = 32
IMAGE_SIZE = 224
SEED = 42

# ── Feature extraction ───────────────────────────────────────────────────────
FEATURE_STRIDE = 3
BACKBONE = "dinov3"

# ── Stride / span ────────────────────────────────────────────────────────────
SOURCE_STANDARD_STRIDE = 45
STANDARD_FEAT_STEPS = SOURCE_STANDARD_STRIDE // FEATURE_STRIDE
STRIDE_OPTIONS_SRC = [3, 6, 15, 30, 45, 90, 135, 180]

# ── Model ────────────────────────────────────────────────────────────────────
N_HEADS = 8
N_LAYERS = 12
DROPOUT = 0.15
MAX_SEQ_LEN = 32

# ── Training ─────────────────────────────────────────────────────────────────
# bs=1024 needs a >=40GB GPU (A100-40GB / L40S); smaller cards (24GB) OOM in
# fp32 — pass BOTH `--batch-size 256 --lr 1e-4` to fall back. NOTE: LR is NOT
# auto-scaled when you override --batch-size (train.py resolves bs and lr
# independently), so you must set --lr yourself. Linear reference:
# 1e-4 @ bs256 -> 4e-4 @ bs1024 (= 1e-4 * 1024/256). Sampler/attention defaults
# (--sampler ar, --attention bidirectional) are set in scripts/train.py Args.
BATCH_SIZE = 1024
LR = 4e-4
WEIGHT_DECAY = 1e-3
MAX_STEPS = 15000
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
EVAL_EVERY = 200
TOTAL_BUDGET = 99999  # no time limit; use MAX_STEPS instead

# ── Dataloader (CachedVideoLoader / online mode) ────────────────────────────
DECODE_THREADS = 16
LOOKAHEAD_BATCHES = 4

# ── Precache ─────────────────────────────────────────────────────────────────
PRECACHE_BATCH_SIZE = 4096 # 16384 OOMs on 40-48GB cloud GPUs (fp32 + fragmentation); 4096 peak alloc ~2.5GB fits
PRECACHE_DECODE_WORKERS = 8
PRECACHE_MEGA_BATCH = 8
