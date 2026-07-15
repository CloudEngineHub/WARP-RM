#!/usr/bin/env python3
"""
Write WARP-RM annotations for a LeRobot dataset.

Two steps, each selectable via --mode:

  score:  dense inference per episode → sidecar at
          assets/warp_rm_annotations/{dataset}/versions/{YYYY-MM-DD}_{tag}/
          (metadata.json, episode_stats.tsv, dense_predictions.parquet).
          Can run locally or on cloud (same script, same output layout).

  inject: read an existing sidecar → write canonical fields into the
          LeRobot dataset parquets + register features in
          meta/info.json + trace provenance in meta/warp_rm_meta.json.
          Always local (modifies on-disk LeRobot).

  both:   score then inject in one pass (default; convenient for local
          all-in-one runs).

Consumers / invocation patterns:

    # All-local one-shot:
    write_warp_rm_annotations.py \
        --checkpoint ckpts/<tag>.pt --lerobot-repo ~/.cache/huggingface/lerobot/<org>/<repo>

    # Cloud-score + local-inject (decoupled GPU work): run --mode score on a
    # GPU worker with --upload-sidecar-to s3://<bucket>/.../sidecar/ ...
    # ...then locally once the sidecar lands on S3:
    aws s3 sync s3://.../sidecar/ assets/warp_rm_annotations/<dataset>/versions/<ver>/
    write_warp_rm_annotations.py --mode inject \
        --sidecar-dir assets/warp_rm_annotations/<dataset>/versions/<ver>/ \
        --lerobot-repo ~/.cache/huggingface/lerobot/<org>/<repo>

Shared scoring logic lives in `warp_rm/eval/scoring.py` — this file is
just the orchestrator + sidecar/manifest management + LeRobot
injection.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import default_feature_cache_dir

from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.eval.scoring import (
    EpisodeScore,
    score_dataset,
    write_sidecar,
    load_sidecar_rows,
)
from warp_rm.models.backbones import build_backbone
from warp_rm.visualization.inference import has_abs_progress_head
from warp_rm.visualization.renderer import extract_features_on_the_fly
from scripts.eval.render import load_checkpoint, try_load_cached_features


@dataclasses.dataclass
class Args:
    """Write WARP-RM annotations (dense per-frame preds) for a LeRobot dataset."""

    lerobot_repo: str
    """Path to the LeRobot dataset root (contains meta/, data/, videos/).
    Auto-detects v2.1 (one parquet per episode) vs v3.0 (sharded parquets,
    multiple episodes per file) from meta/info.json's codebase_version."""

    mode: str = "both"
    """One of 'score' | 'inject' | 'both'. See module docstring."""

    # ── Used in 'score' and 'both' ─────────────────────────────────
    checkpoint: Optional[str] = None
    """Trained WARP-RM checkpoint (.pt). Required for --mode score|both."""

    tag: Optional[str] = None
    """Annotation version tag; defaults to derivation from checkpoint stem.
    Ignored in --mode inject (taken from sidecar metadata)."""

    backbone: str = "dinov3"
    """Must match the checkpoint's backbone."""

    feature_stride: Optional[int] = None
    """Frames between sampled features. None → read from checkpoint; must match
    training (the value the RM was TRAINED with)."""
    image_size: int = 224
    window_size: Optional[int] = None
    """RM attention window (feature steps). None → read from checkpoint; must
    match training (the value the RM was TRAINED with)."""
    cache_dir: Optional[str] = None
    """Default /tmp/feat_rm3_<backbone>_fs<stride>/."""

    camera_key: str = "top_camera-images-rgb"

    cameras: Optional[str] = None
    """Comma-separated camera keys for scoring a MULTI-camera RM. None →
    use the checkpoint's stamped ``cameras`` (the recommended path — the RM
    must be scored with the exact cameras + fusion it was trained on). Set
    explicitly only to override. Single-cam ckpts ignore this."""

    fusion: Optional[str] = None
    """Multi-camera fusion mode ('concat'|'tokens'). None → use the
    checkpoint's stamped ``fusion``. Must match training."""

    require_video: bool = True
    """Drop episodes whose source MP4 is missing. Default True (safe local).
    Set False on cloud jobs where only meta+features are mounted."""

    no_extract_fallback: bool = False
    """If True, never build an encoder and never extract features on the fly
    from video. Episodes without cached features are silently SKIPPED (absent
    from the output sidecar). Set True on cloud scoring jobs where videos
    aren't mounted — on-the-fly extraction fails there anyway."""

    n_episodes: Optional[int] = None
    """Cap on number of episodes to score. None = all. Smoke-test usage:
    --n-episodes 5 against a v3 dataset to validate the inject path quickly.
    Combine with --allow-partial when injecting a small slice into a real
    dataset (since most episodes will be unannotated)."""

    # ── Used in 'inject' and 'both' ────────────────────────────────
    sidecar_dir: Optional[str] = None
    """Path to a pre-computed sidecar (metadata.json + episode_stats.tsv
    + dense_predictions.parquet). For --mode inject; also used as the
    output dir for --mode score if set, overriding the default layout."""

    inject_lerobot: bool = True
    """If True (default), --mode both/inject writes the canonical
    warp_rm_* fields into LeRobot episode parquets + updates
    meta/info.json + meta/warp_rm_meta.json."""

    allow_partial: bool = False
    """v3 only. If True, shards containing unannotated episodes get those
    rows zero-filled (warp_rm_signed_magnitude=0 → downstream RABC sample_weight=0,
    effectively dropped). If False (default), refuse to inject partial
    coverage to avoid silent NaN/zero corruption downstream."""

    # ── Outputs ────────────────────────────────────────────────────
    annotations_root: str = "assets/warp_rm_annotations"
    """Where sidecar versions are stored."""

    upload_sidecar_to: Optional[str] = None
    """If set, after writing the sidecar, `aws s3 cp --recursive <dir>
    <upload-sidecar-to>`. Used by cloud-scoring jobs to deliver the
    sidecar back to S3 for a later local inject."""

    dry_run: bool = False


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _derive_tag(checkpoint_path: str, explicit_tag: Optional[str]) -> str:
    if explicit_tag:
        return explicit_tag
    stem = Path(checkpoint_path).stem
    if stem.startswith("best_model_"):
        stem = stem[len("best_model_"):]
    return stem


def _version_dir(annotations_root: Path, dataset_name: str, tag: str) -> Path:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return annotations_root / dataset_name / "versions" / f"{date}_{tag}"


def _update_canonical_symlink(dataset_ann_dir: Path, version_name: str) -> None:
    canonical = dataset_ann_dir / "canonical"
    target = dataset_ann_dir / "versions" / version_name
    if canonical.exists() or canonical.is_symlink():
        canonical.unlink()
    canonical.symlink_to(target.relative_to(dataset_ann_dir))
    print(f"  [canonical] {canonical} → versions/{version_name}")


def _update_global_manifest(annotations_root: Path, dataset_name: str,
                            version_name: str) -> None:
    manifest_path = annotations_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"datasets": {}}
    ds = manifest["datasets"].setdefault(
        dataset_name, {"canonical": None, "versions": []}
    )
    if version_name not in ds["versions"]:
        ds["versions"].append(version_name)
    ds["canonical"] = version_name
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _upload_sidecar(sidecar_dir: Path, dest: str) -> None:
    cmd = ["aws", "s3", "cp", "--recursive", str(sidecar_dir), dest]
    print(f"  [upload] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ────────────────────────────────────────────────────────────────────
# LeRobot dataset injection
# ────────────────────────────────────────────────────────────────────

def _update_info_schema(info_path: Path) -> None:
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    for name in ("warp_rm_progress", "warp_rm_signed_magnitude"):
        if name not in features:
            # shape=[1] (not []) — lerobot's get_hf_features_from_features
            # rejects empty-shape scalar features as invalid. Match the
            # convention used by frame_index/episode_index.
            features[name] = {"dtype": "float32", "shape": [1], "names": None}
        elif features[name].get("shape") == []:
            features[name]["shape"] = [1]
    info_path.write_text(json.dumps(info, indent=2))


def _inject_into_parquet(
    ep_file: Path,
    arrays: dict[str, np.ndarray],
    strict_len: bool = False,
) -> None:
    df = pd.read_parquet(ep_file)
    n_frames = len(df)

    def _fit(x: np.ndarray, name: str) -> np.ndarray:
        if len(x) == n_frames:
            return x
        # Length mismatch between TSV-derived arrays and the parquet means the
        # scoring run saw a different episode length than the parquet has —
        # downstream consumers will silently see wrong labels on trailing
        # frames (pad_val = x[-1] is not a real progress/velocity value).
        delta = len(x) - n_frames
        msg = (
            f"[inject] {ep_file.name}: {name} length mismatch "
            f"(got {len(x)}, parquet has {n_frames}, delta={delta:+d})"
        )
        if strict_len or abs(delta) > 1:
            raise ValueError(msg + " — refusing to silently truncate/pad")
        print("  WARN " + msg + " — clipping/padding by 1 frame")
        if len(x) > n_frames:
            return x[:n_frames]
        pad_val = x[-1] if len(x) else 0.0
        return np.concatenate([x, np.full(n_frames - len(x), pad_val, dtype=x.dtype)])

    df["warp_rm_progress"] = _fit(arrays["warp_rm_progress"], "warp_rm_progress")
    df["warp_rm_signed_magnitude"] = _fit(arrays["warp_rm_signed_magnitude"], "warp_rm_signed_magnitude")
    df.to_parquet(ep_file, index=False)


def _write_lerobot_meta(
    dataset_root: Path,
    metadata: dict,
) -> None:
    meta = {
        "current_model": metadata.get("tag"),
        "checkpoint_path": metadata.get("checkpoint"),
        "composite": metadata.get("composite"),
        "git_sha": metadata.get("git_sha"),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sidecar_version": metadata.get("version_name"),
        "fields": ["warp_rm_progress", "warp_rm_signed_magnitude"],
    }
    (dataset_root / "meta" / "warp_rm_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )


def _is_v3_dataset(dataset_root: Path) -> bool:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return info.get("codebase_version", "v2.0") >= "v3.0"


def _load_v3_episode_index(dataset_root: Path) -> pd.DataFrame:
    """Read all v3 episode-meta parquets into one DataFrame keyed by episode_index.

    Each row maps an episode_index to the shard file containing its rows
    (data/chunk_index, data/file_index) and its global row range
    (dataset_from_index, dataset_to_index, length).
    """
    episodes_dir = dataset_root / "meta" / "episodes"
    parts = []
    for pf in sorted(episodes_dir.rglob("*.parquet")):
        if pf.name.endswith(".bak"):
            continue
        parts.append(pd.read_parquet(pf, columns=[
            "episode_index", "length",
            "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index",
        ]))
    if not parts:
        raise FileNotFoundError(f"No episode-meta parquets under {episodes_dir}")
    return pd.concat(parts, ignore_index=True).set_index("episode_index", drop=False)


def _inject_into_v3_shard(
    shard_path: Path,
    eps_in_shard: list[dict],     # list of {ep_idx, length, arrays}
    allow_partial: bool,
) -> tuple[int, int, bool]:
    """Inject annotations into a single v3 data shard. Atomic write via tmp + rename.

    Returns (n_eps_injected, n_eps_unannotated_in_shard, columns_were_new).
    """
    df = pd.read_parquet(shard_path)
    n_rows = len(df)
    shard_eps = set(df["episode_index"].unique().tolist())
    annotated_eps = {ep["ep_idx"] for ep in eps_in_shard}
    unannotated = sorted(shard_eps - annotated_eps)

    if unannotated and not allow_partial:
        raise ValueError(
            f"[inject] {shard_path.name}: shard has {len(unannotated)} unannotated "
            f"episodes ({unannotated[:5]}{'...' if len(unannotated) > 5 else ''}). "
            f"Pass --allow-partial to zero-fill them (downstream RABC will get "
            f"sample_weight=0 → effectively dropped)."
        )

    # Initialize columns: zero-fill if not already present, else preserve prior values.
    # Zero-fill (not NaN): downstream ComputeRABCWeights does np.sum(velocity)
    # and (q - q_min)/(q_max - q_min) — NaN propagates, 0 is a benign weight=0.
    columns_were_new = "warp_rm_progress" not in df.columns
    for col in ("warp_rm_progress", "warp_rm_signed_magnitude"):
        if col not in df.columns:
            df[col] = np.zeros(n_rows, dtype=np.float32)
        else:
            df[col] = df[col].astype(np.float32)

    n_injected = 0
    for ep in eps_in_shard:
        ep_idx = ep["ep_idx"]
        mask = (df["episode_index"] == ep_idx).values
        n_match = int(mask.sum())
        if n_match == 0:
            print(f"  WARN [{shard_path.name}] ep {ep_idx}: 0 matching rows in shard, skipping")
            continue
        # Length sanity vs annotation arrays
        for name in ("warp_rm_progress", "warp_rm_signed_magnitude"):
            arr = ep["arrays"][name]
            if len(arr) != n_match:
                # Match v2.1 _fit semantics: tolerate ±1 frame, otherwise raise
                delta = len(arr) - n_match
                msg = (f"[{shard_path.name}] ep {ep_idx}: {name} length mismatch "
                       f"(got {len(arr)}, shard rows={n_match}, delta={delta:+d})")
                if abs(delta) > 1:
                    raise ValueError(msg + " — refusing to silently truncate/pad")
                print("  WARN " + msg + " — clipping/padding by 1 frame")
                if len(arr) > n_match:
                    arr = arr[:n_match]
                else:
                    pad_val = arr[-1] if len(arr) else 0.0
                    arr = np.concatenate(
                        [arr, np.full(n_match - len(arr), pad_val, dtype=arr.dtype)]
                    )
                ep["arrays"][name] = arr
        df.loc[mask, "warp_rm_progress"] = ep["arrays"]["warp_rm_progress"].astype(np.float32)
        df.loc[mask, "warp_rm_signed_magnitude"] = ep["arrays"]["warp_rm_signed_magnitude"].astype(np.float32)
        n_injected += 1

    tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, shard_path)
    return n_injected, len(unannotated), columns_were_new


def _inject_lerobot_v3(
    dataset_root: Path,
    ep_annotations: dict,
    allow_partial: bool,
) -> None:
    """v3.0 path: episodes packed into shard parquets; map each annotated episode
    to its (chunk, file) shard via meta/episodes, then one read+write per shard."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    data_template: str = info["data_path"]  # data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet

    ep_index = _load_v3_episode_index(dataset_root)

    missing = sorted(set(ep_annotations) - set(ep_index["episode_index"].tolist()))
    if missing:
        print(f"  WARN: {len(missing)} annotated episodes not found in dataset meta "
              f"(first 5: {missing[:5]}) — these will be skipped")

    # Group annotated eps by (chunk, file).
    by_shard: dict[tuple[int, int], list[dict]] = {}
    for ep_idx, ann in ep_annotations.items():
        if ep_idx not in ep_index.index:
            continue
        meta_row = ep_index.loc[ep_idx]
        key = (int(meta_row["data/chunk_index"]), int(meta_row["data/file_index"]))
        by_shard.setdefault(key, []).append({
            "ep_idx": ep_idx,
            "length": int(meta_row["length"]),
            "arrays": {
                "warp_rm_progress": ann["warp_rm_progress"],
                "warp_rm_signed_magnitude": ann["warp_rm_signed_magnitude"],
            },
        })

    # Touch EVERY shard, not only those with annotated episodes. With partial
    # scoring (e.g. --no-extract-fallback over a shortest-frac subset) some
    # shards contain ONLY unannotated episodes; if skipped they keep no velocity
    # column → an INCONSISTENT dataset schema that crashes a unified parquet read
    # downstream (openpi rabc_precompute does
    # pq.read_table(all_data_files, columns=[...,warp_rm_signed_magnitude])).
    # With allow_partial such shards get an all-zero column (RABC weight=0);
    # without it _inject_into_v3_shard raises (full coverage required).
    all_shards = {
        (int(r["data/chunk_index"]), int(r["data/file_index"]))
        for _, r in ep_index.iterrows()
    }
    print(f"[inject] v3: {len(all_shards)} shards in dataset "
          f"({len(by_shard)} with annotated episodes), "
          f"{sum(len(v) for v in by_shard.values())} episodes total")

    n_injected = 0
    n_unannotated_zero_filled = 0
    n_unannotated_preserved = 0
    for (chunk_idx, file_idx) in sorted(all_shards):
        eps = by_shard.get((chunk_idx, file_idx), [])
        rel = data_template.format(chunk_index=chunk_idx, file_index=file_idx)
        shard_path = dataset_root / rel
        if not shard_path.exists():
            print(f"  WARN: shard missing: {shard_path}, skipping {len(eps)} eps")
            continue
        n_inj, n_unann, cols_new = _inject_into_v3_shard(shard_path, eps, allow_partial)
        n_injected += n_inj
        if n_unann:
            if cols_new:
                n_unannotated_zero_filled += n_unann
                tag = f"{n_unann} unannotated zero-filled"
            else:
                n_unannotated_preserved += n_unann
                tag = f"{n_unann} unannotated kept prior values"
        else:
            tag = ""
        print(f"  [{chunk_idx:03d}/{file_idx:03d}] {n_inj}/{len(eps)} eps injected"
              + (f" ({tag})" if tag else ""))

    extra = []
    if n_unannotated_zero_filled:
        extra.append(f"zero-filled {n_unannotated_zero_filled} unannotated (new columns)")
    if n_unannotated_preserved:
        extra.append(f"preserved prior values for {n_unannotated_preserved} unannotated")
    suffix = f" ({'; '.join(extra)})" if extra else ""
    print(f"[inject] v3: updated {n_injected} episodes across {len(all_shards)} shards{suffix}")


def _inject_lerobot(
    dataset_root: Path,
    sidecar_dir: Path,
    allow_partial: bool = False,
) -> None:
    print(f"[inject] reading sidecar: {sidecar_dir}")
    metadata = json.loads((sidecar_dir / "metadata.json").read_text())
    ep_annotations = load_sidecar_rows(sidecar_dir)

    print(f"[inject] updating LeRobot schema: {dataset_root}/meta/info.json")
    _update_info_schema(dataset_root / "meta" / "info.json")

    if _is_v3_dataset(dataset_root):
        print(f"[inject] detected LeRobot v3.0 layout (sharded shards)")
        _inject_lerobot_v3(dataset_root, ep_annotations, allow_partial=allow_partial)
    else:
        print(f"[inject] detected LeRobot v2.1 layout (one parquet per episode)")
        data_root = dataset_root / "data"
        n_injected = 0
        for chunk_dir in sorted(data_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for ep_file in sorted(chunk_dir.iterdir()):
                if ep_file.suffix != ".parquet":
                    continue
                stem = ep_file.stem
                try:
                    ep_idx = int(stem.split("_")[-1])
                except ValueError:
                    continue
                ann = ep_annotations.get(ep_idx)
                if ann is None:
                    continue
                arrays = {
                    "warp_rm_progress": ann["warp_rm_progress"],
                    "warp_rm_signed_magnitude": ann["warp_rm_signed_magnitude"],
                }
                _inject_into_parquet(ep_file, arrays)
                n_injected += 1
        print(f"[inject] v2.1: updated {n_injected} episode parquets")

    _write_lerobot_meta(dataset_root, metadata)
    print(f"[inject] wrote {dataset_root}/meta/warp_rm_meta.json")


# ────────────────────────────────────────────────────────────────────
# Scoring step
# ────────────────────────────────────────────────────────────────────

def _resolve_ckpt_hparam(name, cli_val, ckpt, hint):
    """Resolve a scoring hyperparam that MUST match training. Prefer the value
    stamped in the checkpoint; allow a CLI override only if it agrees. NEVER
    fall back to a silent global default — a wrong window_size/feature_stride
    produces silently-wrong magnitudes (the load-bearing footgun this fixes)."""
    stored = ckpt.get(name)
    flag = f"--{name.replace('_', '-')}"
    if cli_val is not None:
        if stored is not None and stored != cli_val:
            raise SystemExit(
                f"{flag} {cli_val} conflicts with the checkpoint's stored {name}={stored}. "
                f"Scoring with a mismatched {name} yields silently-wrong magnitudes. "
                f"Omit {flag} to use the checkpoint's value.")
        return cli_val
    if stored is not None:
        return stored
    raise SystemExit(
        f"Checkpoint has no stored {name} (legacy ckpt) and {flag} was not passed. "
        f"Pass {flag} explicitly — the value the RM was TRAINED with (e.g. {hint}).")


def _run_scoring(args: Args, sidecar_dir: Path) -> dict:
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --mode score|both")
    if args.dry_run:
        print(f"[dry_run] would score {args.lerobot_repo} with "
              f"{args.checkpoint} → {sidecar_dir}")
        return {"tag": _derive_tag(args.checkpoint, args.tag),
                "version_name": sidecar_dir.name,
                "dataset": Path(args.lerobot_repo).resolve().name}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[score] device={device}")

    model, ckpt = load_checkpoint(args.checkpoint, device)
    source_standard_stride = ckpt.get("standard_stride_src", 45)
    feature_stride = _resolve_ckpt_hparam("feature_stride", args.feature_stride, ckpt, hint=1)
    window_size = _resolve_ckpt_hparam("window_size", args.window_size, ckpt, hint=32)
    standard_feat_steps = source_standard_stride // feature_stride
    print(f"[score] feature_stride={feature_stride} window_size={window_size} (checkpoint-resolved)")
    enhanced = has_abs_progress_head(model)
    # Delta-mode ckpts stamp label_mode='delta' + label_anchor_frames in extra_ckpt_meta.
    # Legacy ckpts have no such key → default to 'relative' for back-compat.
    label_mode = ckpt.get("label_mode", "relative")
    label_anchor_frames = ckpt.get("label_anchor_frames") or 15

    # Multi-camera: the RM must be scored with the SAME cameras + fusion it was
    # trained on. Take them from the checkpoint unless overridden on the CLI.
    # Legacy single-cam ckpts → [camera_key], fusion concat (== before).
    fusion = args.fusion or ckpt.get("fusion", "concat")
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        cameras = ckpt.get("cameras") or [args.camera_key]
    n_cameras = len(cameras)
    print(f"[score] standard_feat_steps={standard_feat_steps} enhanced={enhanced} "
          f"label_mode={label_mode}" + (f" anchor={label_anchor_frames}" if label_mode == "delta" else "")
          + f" cameras={cameras} fusion={fusion}")

    dataset_root = Path(args.lerobot_repo).resolve()
    episodes = discover_lerobot_episodes(
        str(dataset_root), camera_key=cameras[0], cameras=cameras,
        n=args.n_episodes,
        require_video=args.require_video,
    )
    if not episodes:
        raise SystemExit(f"[score] no episodes discovered under {dataset_root}")
    print(f"[score] {len(episodes)} episodes")

    cache_dir = args.cache_dir or default_feature_cache_dir(args.backbone, feature_stride)
    encoder = None
    backbone_obj = None

    def _load_one_camera(ep, camera):
        """Cached-or-on-the-fly features for ONE camera of an episode.

        ``camera`` is None for single-cam (legacy cache key) or a camera key
        for multi-cam (per-camera cache namespace via _ep_cache_path)."""
        from warp_rm.utils.caching import _ep_cache_path, _ep_camera_video
        cp = _ep_cache_path(cache_dir, ep, args.backbone, feature_stride, camera)
        if cp.exists():
            return np.load(str(cp))
        if encoder is None:
            return None
        # On-the-fly: build a throwaway Episode pointed at this camera's video so
        # the existing extractor uses the right path + per-camera frame_offset.
        vp, off = _ep_camera_video(ep, camera)
        cam_ep = dataclasses.replace(ep, video_path=vp, frame_offset=off)
        return extract_features_on_the_fly(
            encoder, cam_ep, device, feature_stride, args.image_size,
            backbone_obj.MEAN, backbone_obj.STD,
        )

    # Per-camera cache key: single-cam uses None (legacy); multi-cam uses the
    # camera name so each camera resolves to its own .npy.
    cam_keys = cameras if n_cameras > 1 else [None]

    if args.no_extract_fallback:
        n_cached = sum(
            1 for ep in episodes
            if all(_load_one_camera(ep, c) is not None for c in cam_keys)
        )
        print(f"[score] no_extract_fallback=True — {n_cached}/{len(episodes)} eps "
              f"have cached features (all cameras); the rest will be skipped")
    else:
        from warp_rm.utils.caching import _ep_cache_path
        need_encoder = any(
            any(
                not _ep_cache_path(cache_dir, ep, args.backbone, feature_stride, c).exists()
                for c in cam_keys
            )
            for ep in episodes[:50]
        )
        if need_encoder:
            print(f"[score] building {args.backbone} encoder (some features uncached)")
            backbone_obj, _ = build_backbone(args.backbone)
            backbone_obj = backbone_obj.to(device).eval()
            encoder = backbone_obj

    def _fuse(arrs):
        """Fuse per-camera arrays exactly like data.dataset.load_fused_features."""
        if len(arrs) == 1:
            return arrs[0]
        n_feat = min(a.shape[0] for a in arrs)
        arrs = [a[:n_feat] for a in arrs]
        if fusion == "tokens":
            return np.stack(arrs, axis=1)        # (n_feat, N, 768)
        return np.concatenate(arrs, axis=1)      # (n_feat, 768*N)

    def feature_loader(ep):
        arrs = []
        for c in cam_keys:
            a = _load_one_camera(ep, c)
            if a is None:
                # Any camera missing (and no encoder) → skip this episode.
                return None
            arrs.append(a)
        return _fuse(arrs)

    # Iterate scoring; accumulate into a list (sidecar writer takes a list).
    print(f"[score] writing sidecar to {sidecar_dir}")
    scores: list[EpisodeScore] = []
    for s in score_dataset(
        model=model, episodes=episodes, device=device,
        window_size=window_size, standard_feat_steps=standard_feat_steps,
        feature_stride=feature_stride, feature_loader=feature_loader,
        label_mode=label_mode, label_anchor_frames=label_anchor_frames,
    ):
        scores.append(s)
        if (s.ep_idx + 1) % 100 == 0:
            print(f"  [{s.ep_idx + 1}/{len(episodes)}] mean_vel={s.mean_velocity:.3f} {s.episode_name}")

    composite = float(ckpt.get("composite_score", ckpt.get("val_spearman", 0.0) or 0.0))
    tag = _derive_tag(args.checkpoint, args.tag)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    version_name = sidecar_dir.name  # e.g. "2026-04-22_<tag>"
    metadata = {
        "tag": tag,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": dataset_root.name,
        "composite": composite,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(Path(__file__).resolve().parent.parent.parent),
        "backbone": args.backbone,
        "feature_stride": feature_stride,
        "window_size": window_size,
        "standard_feat_steps": standard_feat_steps,
        "enhanced_model": enhanced,
        "n_episodes": len(episodes),
        "camera_key": cameras[0],
        "cameras": cameras,
        "fusion": fusion,
        "version_name": version_name,
    }

    if args.dry_run:
        print(f"[dry_run] would write sidecar at {sidecar_dir}")
        return metadata

    write_sidecar(sidecar_dir, metadata, scores, feature_stride=feature_stride)
    print(f"[score] sidecar: {sidecar_dir}")
    return metadata


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    if args.mode not in ("score", "inject", "both"):
        raise SystemExit(f"--mode must be one of score|inject|both, got {args.mode!r}")

    dataset_root = Path(args.lerobot_repo).resolve()
    dataset_name = dataset_root.name
    annotations_root = Path(args.annotations_root).resolve()

    # Resolve sidecar dir.
    if args.mode in ("score", "both"):
        if args.checkpoint is None:
            raise SystemExit("--checkpoint required for --mode score|both")
        tag = _derive_tag(args.checkpoint, args.tag)
        if args.sidecar_dir:
            sidecar_dir = Path(args.sidecar_dir).resolve()
        else:
            sidecar_dir = _version_dir(annotations_root, dataset_name, tag)
        version_name = sidecar_dir.name
    else:  # inject
        if args.sidecar_dir is None:
            raise SystemExit("--sidecar-dir required for --mode inject")
        sidecar_dir = Path(args.sidecar_dir).resolve()
        version_name = sidecar_dir.name
        tag = None  # derived from sidecar metadata when needed

    print(f"[annotations] mode={args.mode} dataset={dataset_name}")
    print(f"[annotations] sidecar={sidecar_dir}")
    print(f"[annotations] inject_lerobot={args.inject_lerobot} dry_run={args.dry_run}")

    # 1. Score (if requested).
    if args.mode in ("score", "both"):
        metadata = _run_scoring(args, sidecar_dir)
        if args.dry_run:
            return
        # Manifest + canonical symlink live in the annotations tree. Skip if
        # user pointed --sidecar-dir outside the standard tree (they can
        # manage it manually in that case).
        try:
            sidecar_dir.relative_to(annotations_root)
            _update_canonical_symlink(annotations_root / dataset_name, version_name)
            _update_global_manifest(annotations_root, dataset_name, version_name)
        except ValueError:
            print(f"  [skip] sidecar outside {annotations_root}; "
                  f"not updating canonical / manifest")
        if args.upload_sidecar_to:
            _upload_sidecar(sidecar_dir, args.upload_sidecar_to)

    # 2. Inject (if requested).
    if args.mode in ("inject", "both") and args.inject_lerobot:
        if args.dry_run:
            print("[dry_run] would inject LeRobot fields")
        else:
            _inject_lerobot(dataset_root, sidecar_dir, allow_partial=args.allow_partial)

    print("\n[annotations] DONE.")
    print(f"  sidecar: {sidecar_dir}")
    if args.mode in ("inject", "both") and args.inject_lerobot and not args.dry_run:
        print(f"  lerobot meta: {dataset_root}/meta/warp_rm_meta.json")


if __name__ == "__main__":
    main()
