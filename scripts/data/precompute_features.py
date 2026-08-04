#!/usr/bin/env python3
"""
Standalone feature pre-computation script.

Pre-extracts backbone features for all episodes and caches to disk.
Run once before multiple training experiments to avoid redundant work.

Supports multi-GPU sharding for ~linear speedup:
    # Single GPU (default):
    python scripts/precompute_features.py --data-dir /path/to/episodes

    # 3 GPUs in parallel (run each in a separate terminal):
    python scripts/precompute_features.py --data-dir /path --gpu 3 --shard 0 --num-shards 3
    python scripts/precompute_features.py --data-dir /path --gpu 5 --shard 1 --num-shards 3
    python scripts/precompute_features.py --data-dir /path --gpu 0 --shard 2 --num-shards 3

    # Or use the --multi-gpu shorthand to auto-launch all shards:
    python scripts/precompute_features.py --data-dir /path --multi-gpu 3,5,0
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import default_feature_cache_dir

from warp_rm.data.dataset import discover_episodes
from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.models.backbones import build_backbone
from warp_rm.utils.caching import precompute_features
from warp_rm.utils.environment import seed_everything

import tyro


def _filter_shortest(episodes, frac: float):
    """Keep the shortest ``frac`` of episodes by n_frames.

    Selection is identical to scripts/train.py:filter_shortest
    (sorted-by-n_frames, take the shortest ``round(frac*N)``), so the
    precomputed cache covers exactly the episode set the RM trains on under
    the same --shortest-frac. Deterministic, no seed.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"shortest_frac must be in (0, 1], got {frac}")
    n_total = len(episodes)
    n_keep = max(1, int(round(frac * n_total)))
    kept = sorted(episodes, key=lambda e: e.n_frames)[:n_keep]
    thr = kept[-1].n_frames if kept else None
    print(f"[filter_shortest] frac={frac} -> kept {len(kept)}/{n_total} "
          f"(threshold n_frames<={thr})")
    return kept


@dataclasses.dataclass
class Args:
    """Pre-compute backbone features."""

    data_dir: str
    """Path to episode data."""

    backbone: Literal["dinov3", "clip"] = "dinov3"
    """Backbone model to use."""

    feature_stride: int = 3
    """Feature stride."""

    image_size: int = 224
    """Image size."""

    crop_mode: str = "squash"
    """Frame geometry: 'squash' (resize ignoring aspect ratio, the default and
    a no-op for already-square videos) or 'center' (center-crop to the largest
    centered square first). MUST match what training uses — the mode is part of
    the feature-cache key, so a mismatch silently produces a second cache rather
    than reusing the first."""

    batch_size: int = 512
    """GPU batch size (default: 512, A100 can handle 1024)."""

    decode_workers: int = 8
    """Parallel CPU decode threads."""

    mega_batch: int = 8
    """Episodes to decode in parallel per GPU batch."""

    cache_dir: Optional[str] = None
    """Cache directory (default: /tmp/feat_rm3_{backbone}_fs{stride})."""

    prefer_resized: Optional[int] = None
    """Prefer pre-resized MP4s at this resolution (e.g. 224)."""

    gpu: Optional[int] = None
    """GPU ID to use (sets CUDA_VISIBLE_DEVICES)."""

    shard: int = 0
    """Shard index for multi-GPU (0-indexed)."""

    num_shards: int = 1
    """Total number of shards."""

    multi_gpu: Optional[str] = None
    """Comma-separated GPU IDs to auto-launch parallel shards (e.g. '3,5,0')."""

    camera_key: str = "top_camera-images-rgb"
    """LeRobot dataset camera key (used only for v2.1/v3.0 datasets).

    Single-camera convenience flag; superseded by --cameras when that lists
    more than one camera. Either flag lets you precompute a SPECIFIC camera
    (e.g. a wrist camera) so multi-cam RMs have their feature caches ready."""

    cameras: Optional[str] = None
    """Comma-separated camera keys for multi-camera precompute (LeRobot only).

    None → single camera = --camera-key (legacy behavior, identical cache
    layout). When set, each camera is extracted + cached separately under its
    own dataset/camera namespace (see warp_rm/utils/caching._ep_cache_path).
    Example: --cameras top_camera-images-rgb,left_camera-images-rgb"""

    shortest_frac: float = 1.0
    """Keep only the shortest N-frac of discovered episodes (by n_frames).

    1.0 = keep all (default, legacy behavior). 0.25 mirrors the RM train's
    filter_shortest(0.25) so the precomputed cache covers exactly the
    shortest-25% episode set the reward model trains on (the RM trains then
    use --require-cached to consume exactly this set)."""


def main():
    args = tyro.cli(Args)

    # Auto-launch mode: spawn one process per GPU
    if args.multi_gpu:
        gpu_ids = [int(g) for g in args.multi_gpu.split(",")]
        num_shards = len(gpu_ids)
        print(f"Launching {num_shards} shards across GPUs {gpu_ids}")

        procs = []
        for shard_id, gpu_id in enumerate(gpu_ids):
            cmd = [
                sys.executable, __file__,
                "--data-dir", args.data_dir,
                "--backbone", args.backbone,
                "--feature-stride", str(args.feature_stride),
                "--image-size", str(args.image_size),
                "--crop-mode", args.crop_mode,
                "--batch-size", str(args.batch_size),
                "--decode-workers", str(args.decode_workers),
                "--mega-batch", str(args.mega_batch),
                "--gpu", str(gpu_id),
                "--shard", str(shard_id),
                "--num-shards", str(num_shards),
            ]
            if args.cache_dir:
                cmd.extend(["--cache-dir", args.cache_dir])
            if args.prefer_resized is not None:
                cmd.extend(["--prefer-resized", str(args.prefer_resized)])
            cmd.extend(["--camera-key", args.camera_key])
            if args.cameras:
                cmd.extend(["--cameras", args.cameras])
            cmd.extend(["--shortest-frac", str(args.shortest_frac)])

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, p))
            print(f"  Shard {shard_id} on GPU {gpu_id} (pid {p.pid})")

        # Wait for all
        for gpu_id, p in procs:
            rc = p.wait()
            print(f"  GPU {gpu_id} finished (exit code {rc})")

        print("All shards complete.")
        return

    # Single-shard mode
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building {args.backbone} backbone...")
    backbone, _ = build_backbone(args.backbone)
    backbone = backbone.to(device).eval()

    # Parse the camera list (multi-cam precompute). None / empty → single
    # camera = --camera-key (legacy behavior, identical cache layout).
    cameras_list = (
        [c.strip() for c in args.cameras.split(",") if c.strip()]
        if args.cameras else [args.camera_key]
    )

    # Detect LeRobot v2.1/v3.0 datasets (have meta/info.json) vs raw flat layout
    if (Path(args.data_dir) / "meta" / "info.json").exists():
        print(f"Detected LeRobot dataset, using discover_lerobot_episodes (cameras={cameras_list})")
        episodes = discover_lerobot_episodes(
            args.data_dir, camera_key=cameras_list[0], cameras=cameras_list,
        )
    else:
        episodes = discover_episodes(args.data_dir, prefer_resized=args.prefer_resized)
    print(f"Found {len(episodes)} episodes")

    # Scope to the shortest-N-frac BEFORE sharding so the selection is global
    # (matches the RM train's filter_shortest, which runs on the full
    # discovered set). Each shard then extracts its slice of the kept set.
    if args.shortest_frac < 1.0:
        episodes = _filter_shortest(episodes, args.shortest_frac)

    # Shard if needed
    if args.num_shards > 1:
        episodes = [ep for i, ep in enumerate(episodes) if i % args.num_shards == args.shard]
        print(f"Shard {args.shard}/{args.num_shards}: {len(episodes)} episodes")

    cache_dir = args.cache_dir or default_feature_cache_dir(args.backbone, args.feature_stride)

    # Single camera passes cameras=None → camera=None cache key (byte-identical
    # to legacy). Multi-cam caches each camera under its own namespace.
    precompute_cameras = cameras_list if len(cameras_list) > 1 else None
    precompute_features(
        episodes, backbone, device,
        args.feature_stride, args.image_size, cache_dir=cache_dir,
        backbone=args.backbone, batch_size=args.batch_size,
        mean=backbone.MEAN, std=backbone.STD,
        decode_workers=args.decode_workers,
        mega_batch_episodes=args.mega_batch,
        crop_mode=args.crop_mode,
        cameras=precompute_cameras,
    )
    print("Done.")


if __name__ == "__main__":
    main()
