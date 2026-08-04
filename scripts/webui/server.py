#!/usr/bin/env python3
"""
WARP-RM inspector — interactive web UI for per-frame reward predictions.

Run:
    uv run --extra webui python scripts/webui/server.py --host 0.0.0.0 --port 8000

Opens a viewer for WARP-RM predictions on LeRobot episodes:
  - Checkpoint dropdown (discovered from checkpoints/*.pt).
  - LeRobot dataset multi-select (discovered under --dataset-root).
  - Episode table (union of selected repos, sortable by length or mean velocity).
  - Video scrubber on the left, live Plotly panels on the right mirroring
    scripts/eval/render.py (absolute progress / per-frame velocity / zoomed
    velocity), all shaded by the per-frame WARP-BC weight.

Two signal sources, switchable in the header:
  - ``ckpt``    — live dense inference from the loaded checkpoint.
  - ``sidecar`` — the ``warp_rm_progress`` / ``warp_rm_signed_magnitude``
                  columns already injected into the dataset's parquets by
                  scripts/data/write_warp_rm_annotations.py. No GPU needed.

Caches (all under ~/.cache/warp_rm, override with WARP_RM_FEATURE_CACHE):
  - DINOv3 features: shared with training/scoring via `warp_rm.utils.caching`,
    so anything the trainer already extracted is reused here for free.
  - Inference + per-episode velocity summaries: ``webui_infer/``. The summary
    cache is the same file scripts/eval/score_episodes.py writes, so an
    offline scoring run populates the episode list without re-inference.
  - Sliced per-episode clips (LeRobot v3.0 concat shards): ``webui_clips/``.
"""

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import default_feature_cache_dir  # noqa: E402
from warp_rm.data.dataset import Episode  # noqa: E402
from warp_rm.data.lerobot_dataset import (  # noqa: E402
    discover_lerobot_episodes, get_splits,
)
from warp_rm.models.backbones import build_backbone  # noqa: E402
from warp_rm.utils.caching import _ep_cache_path  # noqa: E402
from warp_rm.visualization.inference import (  # noqa: E402
    dense_inference_relative, dense_inference_absolute, has_abs_progress_head,
)
from warp_rm.visualization.renderer import extract_features_on_the_fly  # noqa: E402
from scripts.eval.render import load_checkpoint  # noqa: E402

CKPT_DIR = REPO_ROOT / "checkpoints"
CACHE_ROOT = Path(
    os.environ.get("WARP_RM_WEBUI_CACHE", str(Path.home() / ".cache" / "warp_rm"))
).expanduser()
# Must match scripts/eval/score_episodes.py --webui-cache-dir so an offline
# scoring run and this server share one velocity-summary cache.
INFER_CACHE = CACHE_ROOT / "webui_infer"
CLIP_CACHE_DIR = CACHE_ROOT / "webui_clips"
# Sidecars written by scripts/data/write_warp_rm_annotations.py.
SIDECAR_ROOT = REPO_ROOT / "assets" / "warp_rm_annotations"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Dataset scan roots. Overridden by --dataset-root (repeatable) or the
# WARP_RM_LEROBOT_ROOTS env var (os.pathsep-separated).
DEFAULT_DATASET_ROOTS = [
    Path("~/.cache/huggingface/lerobot"),
    Path("~/data/lerobot"),
    Path("~/datasets"),
]
DATASET_ROOTS: list[Path] = []

# Canonical per-frame columns written by write_warp_rm_annotations.py --mode inject.
SIDECAR_PROGRESS_COL = "warp_rm_progress"
SIDECAR_VELOCITY_COL = "warp_rm_signed_magnitude"

# Default feature stride; the loaded ckpt's stamped value overrides this and
# is the source of truth (see _load_checkpoint_into_state). Anything that
# touches the feature cache or runs inference must read STATE.feature_stride
# instead of this constant — otherwise an FS=2 ckpt silently scores against
# the FS=3 cache (different content under the same key).
DEFAULT_FEATURE_STRIDE = 3
IMAGE_SIZE = 224
# Inference window. 32 matches the training recipe (scripts/config.WINDOW_SIZE)
# and scripts/eval/score_episodes.py; the aggregator caps at 32 and smaller
# windows change the velocity magnitude.
WINDOW_SIZE = 32
BACKBONE_NAME = "dinov3"


@dataclasses.dataclass
class State:
    device: torch.device
    backbone: torch.nn.Module
    backbone_mean: np.ndarray
    backbone_std: np.ndarray
    model: Optional[torch.nn.Module] = None
    ckpt_path: Optional[Path] = None
    ckpt_meta: Optional[dict] = None
    std_stride_src: int = 45
    # Read from the loaded ckpt's stamped `feature_stride`. Falls back to
    # DEFAULT_FEATURE_STRIDE for legacy ckpts that predate the stamp.
    feature_stride: int = 3
    # Frame geometry the ckpt was trained with ("squash" | "center"). Read from
    # the ckpt stamp; pre-crop-mode ckpts were all trained squashed.
    crop_mode: str = "squash"
    ep_cache: dict[str, list[Episode]] = dataclasses.field(default_factory=dict)
    gpu_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    # Per-checkpoint velocity-summary cache, loaded from disk JSON on ckpt
    # load. Keyed by str(episode path). Values: {mean_velocity, delta_v_std,
    # n_feat}.
    summary_cache: dict[str, dict] = dataclasses.field(default_factory=dict)
    summary_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    # mtime_ns of the disk cache at last load. Used to auto-reload if an
    # offline scorer (scripts/eval/score_episodes.py) writes under us.
    summary_cache_mtime_ns: int = 0


STATE: State | None = None  # populated in main()


# ── Caches ───────────────────────────────────────────────────────────────────

def _feat_cache_path(ep: Episode) -> Path:
    """Canonical training feature-cache path for this episode.

    Delegates to `warp_rm.utils.caching._ep_cache_path` so the inspector reads
    (and writes) exactly the files scripts/train.py and the precompute script
    use — a dataset already precached for training costs zero GPU here.
    """
    cache_dir = default_feature_cache_dir(BACKBONE_NAME, STATE.feature_stride)
    return _ep_cache_path(cache_dir, ep, BACKBONE_NAME, STATE.feature_stride,
                          None, STATE.crop_mode)


def _inference_cache_path(ckpt_path: Path, ep: Episode) -> Path:
    ckpt_stat = ckpt_path.stat()
    # v3.0 concat shards: include frame_offset + n_frames so episodes sharing
    # a shard mp4 don't collide. Without this, the first episode's cached
    # inference is reused for every other episode in the shard.
    suffix = f"@{ep.frame_offset}+{ep.n_frames}" if ep.frame_offset else f"+{ep.n_frames}"
    raw = (
        f"{ckpt_path.resolve()}:{ckpt_stat.st_mtime_ns}:"
        f"{ep.video_path}{suffix}:{STATE.feature_stride}:{WINDOW_SIZE}:{STATE.std_stride_src}"
    )
    key = hashlib.md5(raw.encode()).hexdigest()[:16]
    return INFER_CACHE / f"{key}.npz"


def _abs_head_cache_path(ckpt_path: Path, ep: Episode) -> Path:
    return _inference_cache_path(ckpt_path, ep).with_suffix(".abs.npz")


def _ckpt_hash(ckpt_path: Path) -> str:
    stat = ckpt_path.stat()
    raw = f"{ckpt_path.resolve()}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _summary_json_path(ckpt_path: Path) -> Path:
    # Filename matches scripts/eval/score_episodes.py:_webui_cache_path.
    return INFER_CACHE / f"quality_{_ckpt_hash(ckpt_path)}.json"


def _json_safe(obj):
    """NaN/inf → None so browser JSON.parse doesn't choke and JSON round-trips."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _load_summary_cache(ckpt_path: Path) -> tuple[dict[str, dict], int]:
    """Read the on-disk velocity-summary cache + its mtime_ns.

    Returns (scores, mtime_ns). mtime_ns is 0 when the file is absent or
    unreadable; otherwise it's the file's current mtime, used by the
    in-request auto-reload path to detect offline writers.
    """
    p = _summary_json_path(ckpt_path)
    if not p.exists():
        return {}, 0
    try:
        mtime_ns = p.stat().st_mtime_ns
        obj = json.loads(p.read_text())
        scores = obj.get("scores", {})
        return (scores if isinstance(scores, dict) else {}), mtime_ns
    except Exception:
        traceback.print_exc()
        return {}, 0


def _maybe_reload_summary_cache() -> None:
    """Reload the summary cache from disk if another writer has updated it
    since we last loaded. Called at the top of summary-serving paths so an
    offline scorer writing the same file becomes visible without an explicit
    checkpoint reload."""
    if STATE.ckpt_path is None:
        return
    p = _summary_json_path(STATE.ckpt_path)
    try:
        mtime_ns = p.stat().st_mtime_ns
    except FileNotFoundError:
        return
    if mtime_ns <= STATE.summary_cache_mtime_ns:
        return
    with STATE.summary_lock:
        # Double-checked under lock: another request may have reloaded already.
        if mtime_ns <= STATE.summary_cache_mtime_ns:
            return
        scores, loaded_mtime = _load_summary_cache(STATE.ckpt_path)
        STATE.summary_cache = scores
        STATE.summary_cache_mtime_ns = loaded_mtime
    print(f"  summary cache reloaded: {len(scores)} entries ({p})")


def _persist_summary_cache():
    """Atomically write the in-memory summary cache to disk for the current
    ckpt. Refreshes ``STATE.summary_cache_mtime_ns`` so our own write is not
    later mistaken for an external update that needs reloading.
    """
    if STATE.ckpt_path is None:
        return
    p = _summary_json_path(STATE.ckpt_path)
    tmp = p.with_suffix(".json.tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt": str(STATE.ckpt_path),
            "mtime_ns": STATE.ckpt_path.stat().st_mtime_ns,
            "scores": STATE.summary_cache,
        }
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)
        try:
            STATE.summary_cache_mtime_ns = p.stat().st_mtime_ns
        except FileNotFoundError:
            pass
    except Exception:
        traceback.print_exc()


# ── Discovery ────────────────────────────────────────────────────────────────

# Datasets are very often symlinked into a scan root (e.g. a dataset kept on a
# big disk, linked into ~/data/lerobot). `Path.rglob("meta/info.json")` cannot
# be used for this: on Python >= 3.13 `**` no longer recurses into symlinked
# directories (glob(recurse_symlinks=False) became the default), so a symlinked
# dataset silently fails to appear — and it DID appear on <= 3.12, making the
# behavior version-dependent. os.walk(followlinks=True) is explicit and behaves
# the same on every supported version.
_MAX_SCAN_DEPTH = 4


def _find_dataset_infos(scan_root: Path) -> list[Path]:
    """Find every `meta/info.json` under `scan_root`, following symlinks.

    Depth-capped (datasets live at `<root>/<dataset>/` or `<root>/<org>/<dataset>/`)
    and guarded against symlink cycles by tracking resolved directories, which
    `followlinks=True` would otherwise loop on.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    root_depth = len(scan_root.parts)
    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=True):
        here = Path(dirpath)
        try:
            real = here.resolve()
        except OSError:
            dirnames[:] = []
            continue
        if real in seen:            # symlink cycle / diamond — do not re-descend
            dirnames[:] = []
            continue
        seen.add(real)
        if len(here.parts) - root_depth >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
            continue
        if here.name == "meta" and "info.json" in filenames:
            out.append(here / "info.json")
            dirnames[:] = []        # nothing of interest below a dataset's meta/
    return sorted(out)


def _discover_datasets() -> list[dict]:
    seen: set[Path] = set()
    out: list[dict] = []

    for scan_root in DATASET_ROOTS:
        if not scan_root.exists():
            continue
        for info_path in _find_dataset_infos(scan_root):
            root = info_path.parent.parent.resolve()
            if root in seen:
                continue
            seen.add(root)
            try:
                meta = json.loads(info_path.read_text())
            except Exception:
                continue
            video_keys = [k for k, v in meta.get("features", {}).items()
                          if v.get("dtype") == "video"]
            splits = get_splits(str(root))
            try:
                name = str(root.relative_to(scan_root))
            except ValueError:
                name = root.name
            out.append({
                "path": str(root),
                "name": name,
                "root": str(scan_root),
                "n_episodes": int(meta.get("total_episodes", 0)),
                "n_frames": int(meta.get("total_frames", 0)),
                "fps": int(meta.get("fps", 30)),
                "video_keys": video_keys,
                "splits": splits,
            })
    out.sort(key=lambda d: d["name"])
    return out


def _discover_checkpoints() -> list[dict]:
    if not CKPT_DIR.exists():
        return []
    out = []
    for p in sorted(CKPT_DIR.rglob("*.pt")):
        stat = p.stat()
        rel = p.relative_to(CKPT_DIR)
        out.append({
            "path": str(p),
            "name": str(rel),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "mtime": int(stat.st_mtime),
        })
    return out


def _get_episodes(repo_path: str, camera_key: str,
                  split: Optional[str] = None) -> list[Episode]:
    key = f"{repo_path}::{camera_key}::{split or 'all'}"
    if key not in STATE.ep_cache:
        STATE.ep_cache[key] = discover_lerobot_episodes(
            repo_path, camera_key=camera_key, split=split,
        )
    return STATE.ep_cache[key]


def _load_checkpoint_into_state(ckpt_path: Path):
    model, ckpt = load_checkpoint(str(ckpt_path), STATE.device)
    STATE.model = model
    STATE.ckpt_path = ckpt_path
    STATE.std_stride_src = int(ckpt.get("standard_stride_src", 45))
    ckpt_fs = ckpt.get("feature_stride")
    if ckpt_fs is None:
        STATE.feature_stride = DEFAULT_FEATURE_STRIDE
        print(f"  feature_stride={STATE.feature_stride} "
              f"(no ckpt stamp — assuming legacy default)")
    else:
        STATE.feature_stride = int(ckpt_fs)
        print(f"  feature_stride={STATE.feature_stride} (from checkpoint)")
    STATE.crop_mode = ckpt.get("crop_mode", "squash")
    print(f"  crop_mode={STATE.crop_mode} "
          f"({'from checkpoint' if ckpt.get('crop_mode') else 'legacy default'})")
    STATE.ckpt_meta = {
        "step": ckpt.get("step"),
        "val_loss": float(ckpt["val_loss"]) if "val_loss" in ckpt else None,
        "val_spearman": float(ckpt["val_spearman"]) if "val_spearman" in ckpt else None,
        "standard_stride_src": STATE.std_stride_src,
        "feature_stride": STATE.feature_stride,
        "crop_mode": STATE.crop_mode,
        "window_size": WINDOW_SIZE,
    }
    with STATE.summary_lock:
        STATE.summary_cache, STATE.summary_cache_mtime_ns = _load_summary_cache(ckpt_path)
    print(f"  summary cache: {len(STATE.summary_cache)} entries "
          f"({_summary_json_path(ckpt_path)})")


# ── Inference ────────────────────────────────────────────────────────────────

def _features_for(ep: Episode) -> np.ndarray:
    """Load this episode's DINOv3 features, extracting + caching on a miss.

    Caller must already hold STATE.gpu_lock.
    """
    feat_path = _feat_cache_path(ep)
    if feat_path.exists():
        return np.load(feat_path)
    feat = extract_features_on_the_fly(
        STATE.backbone, ep, STATE.device,
        STATE.feature_stride, IMAGE_SIZE,
        STATE.backbone_mean, STATE.backbone_std,
        crop_mode=STATE.crop_mode,
    )
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, feat)
    return feat


def _run_inference(ep: Episode) -> dict:
    """Compute (or load) the per-frame relative head outputs for an episode.

    Cache format: source-frame-interpolated `abs_progress`, `velocity`,
    `weights`, plus `n_frames`. Abs-head arrays live in a separate sibling npz
    so they can be added without invalidating this cache.
    """
    if STATE.model is None or STATE.ckpt_path is None:
        raise HTTPException(400, "no checkpoint loaded — POST /api/checkpoint first")

    infer_path = _inference_cache_path(STATE.ckpt_path, ep)
    if infer_path.exists():
        z = np.load(infer_path)
        return {
            "abs_progress": z["abs_progress"].tolist(),
            "velocity": z["velocity"].tolist(),
            "weights": z["weights"].tolist(),
            "n_frames": int(z["n_frames"]),
            "cached": True,
        }

    with STATE.gpu_lock:
        feat = _features_for(ep)
        std_feat_steps = STATE.std_stride_src // STATE.feature_stride
        abs_prog, vel = dense_inference_relative(
            STATE.model, feat, STATE.device,
            window_size=WINDOW_SIZE,
            standard_feat_steps=std_feat_steps,
        )

    n_frames = ep.n_frames
    t_src = np.linspace(0, 1, n_frames, dtype=np.float32)
    t_feat = np.linspace(0, 1, len(abs_prog), dtype=np.float32)
    abs_frames = np.interp(t_src, t_feat, abs_prog).astype(np.float32)
    vel_frames = np.interp(t_src, t_feat, vel).astype(np.float32)
    # WARP-BC per-frame weight: the positive part of the signed velocity.
    weights = np.clip(vel_frames, 0.0, None).astype(np.float32)

    INFER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        infer_path,
        abs_progress=abs_frames, velocity=vel_frames, weights=weights,
        n_frames=np.int32(n_frames),
    )
    return {
        "abs_progress": abs_frames.tolist(),
        "velocity": vel_frames.tolist(),
        "weights": weights.tolist(),
        "n_frames": int(n_frames),
        "cached": False,
    }


def _compute_abs_head(ep: Episode) -> Optional[dict[str, np.ndarray]]:
    """Compute (or load) the absolute-progress head outputs for an episode.

    Caches to a sibling `.abs.npz` next to the main inference cache so the main
    cache is untouched. Returns None if the model has no abs head (e.g. a
    `no_abs` ablation checkpoint).
    """
    if STATE.model is None or STATE.ckpt_path is None:
        raise HTTPException(400, "no checkpoint loaded — POST /api/checkpoint first")
    if not has_abs_progress_head(STATE.model):
        return None

    abs_path = _abs_head_cache_path(STATE.ckpt_path, ep)
    if abs_path.exists():
        z = np.load(abs_path)
        return {k: z[k] for k in z.files}

    with STATE.gpu_lock:
        feat = _features_for(ep)
        std_feat_steps = STATE.std_stride_src // STATE.feature_stride
        unc = dense_inference_absolute(
            STATE.model, feat, STATE.device,
            window_size=WINDOW_SIZE,
            standard_feat_steps=std_feat_steps,
        )

    INFER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(abs_path, **{k: v.astype(np.float32) for k, v in unc.items()})
    return {k: v.astype(np.float32) for k, v in unc.items()}


# ── Per-episode velocity summary ─────────────────────────────────────────────

def _dataset_name_for_ep(ep: Episode) -> Optional[str]:
    """LeRobot Episode paths look like <root>/data/chunk-NNN/episode_MMMMMM.
    Return the repo root's name (the 'dataset name' in the sidecar layout)
    or None if the path doesn't fit the LeRobot shape."""
    for a in ep.path.parents:
        if a.name == "data":
            return a.parent.name
    return None


def _sidecar_match_for_ckpt(dataset_name: str, ckpt_path: Path) -> Optional[Path]:
    """Find a sidecar (under SIDECAR_ROOT/<dataset>/) whose metadata.json
    references the currently-loaded checkpoint.

    Search order:
      1. <dataset>/canonical (fast path — common case)
      2. any <dataset>/versions/*/metadata.json

    Step 2 covers the case where the canonical symlink points at a different
    scorer than the one currently loaded — we still find the matching versioned
    sidecar and serve its stats from there. Returns the first match, or None.
    """
    ds_root = SIDECAR_ROOT / dataset_name
    try:
        target = ckpt_path.resolve()
    except Exception:
        return None

    candidates: list[Path] = []
    canonical = ds_root / "canonical"
    if canonical.exists():
        candidates.append(canonical)
    versions_dir = ds_root / "versions"
    if versions_dir.exists():
        # Sort so traversal is deterministic; newest-date-first (version dir
        # names start with YYYY-MM-DD_).
        candidates.extend(sorted(versions_dir.iterdir(), reverse=True))

    for side in candidates:
        meta_path = side / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        side_ckpt = meta.get("checkpoint")
        if not side_ckpt:
            continue
        try:
            if Path(side_ckpt).resolve() == target:
                return side
        except Exception:
            continue
    return None


# Parsed sidecar TSVs, keyed by sidecar-dir str → {episode_name: summary dict}.
# Populated lazily on first hit for a given sidecar.
_SIDECAR_ROWS: dict[str, dict[str, dict]] = {}


def _sidecar_rows(sidecar_dir: Path) -> dict[str, dict]:
    """Parse the sidecar's per-episode stats TSV into {episode_name: summary}.

    `warp_rm.eval.scoring.write_sidecar` writes `episode_quality.tsv`; older /
    doc-referenced sidecars use `episode_stats.tsv`. Both carry the same
    columns, so accept either. Memoized per sidecar dir.
    """
    key = str(sidecar_dir.resolve())
    cached = _SIDECAR_ROWS.get(key)
    if cached is not None:
        return cached
    tsv = next(
        (p for p in (sidecar_dir / "episode_quality.tsv",
                     sidecar_dir / "episode_stats.tsv") if p.exists()),
        None,
    )
    if tsv is None:
        _SIDECAR_ROWS[key] = {}
        return {}
    rows: dict[str, dict] = {}
    import csv as _csv
    with open(tsv, "r") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            rows[r["episode"]] = {
                "mean_velocity": _f(r.get("mean_velocity")),
                "delta_v_std": _f(r.get("delta_v_std")),
                "n_feat": int(float(r.get("n_feat") or 0)),
            }
    _SIDECAR_ROWS[key] = rows
    return rows


def _try_sidecar_hit(ep: Episode) -> Optional[dict]:
    """Look for a precomputed velocity summary from the sidecar matching the
    current checkpoint + this episode's dataset. Returns the summary or None."""
    if STATE.ckpt_path is None:
        return None
    dataset_name = _dataset_name_for_ep(ep)
    if dataset_name is None:
        return None
    side = _sidecar_match_for_ckpt(dataset_name, STATE.ckpt_path)
    if side is None:
        return None
    return _sidecar_rows(side).get(ep.path.name)


def _summary_for_episode(ep: Episode) -> dict:
    """Per-episode velocity summary. Reuses cached inference arrays; only runs
    what's missing.

    Mirrors `scripts/eval/score_episodes.py:episode_velocity_summary` —
    ``mean_velocity`` is the mean per-frame progress velocity (pace) and
    ``delta_v_std`` the std of its first difference (smoothness).

    Results are memoized per-checkpoint in an on-disk JSON so repeat calls
    (including across server restarts) are ~instant. If an external writer
    (e.g. score_episodes.py) has updated the on-disk cache since we last
    loaded, we pull those entries in on-demand. Before running inference we
    check for a sidecar under ``assets/warp_rm_annotations/<dataset>/`` pinned
    to the same checkpoint and reuse its precomputed stats.
    """
    if STATE.model is None or STATE.ckpt_path is None:
        raise HTTPException(400, "no checkpoint loaded — POST /api/checkpoint first")

    _maybe_reload_summary_cache()

    key = str(ep.path.resolve())
    with STATE.summary_lock:
        hit = STATE.summary_cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}

    # Sidecar fast-path: reuse the offline stats from a matching scoring run.
    sidecar_hit = _try_sidecar_hit(ep)
    if sidecar_hit is not None:
        safe_hit = _json_safe(sidecar_hit)
        with STATE.summary_lock:
            STATE.summary_cache[key] = safe_hit
            _persist_summary_cache()
        return {**safe_hit, "cached": True, "source": "sidecar"}

    infer_path = _inference_cache_path(STATE.ckpt_path, ep)
    if not infer_path.exists():
        _run_inference(ep)
    z = np.load(infer_path)
    vel = np.asarray(z["velocity"], dtype=np.float32)

    result = {
        "mean_velocity": float(np.mean(vel)) if vel.size else float("nan"),
        "delta_v_std": float(np.std(np.diff(vel))) if vel.size > 1 else float("nan"),
        "n_feat": int(vel.size),
    }

    safe_result = _json_safe(result)
    with STATE.summary_lock:
        STATE.summary_cache[key] = safe_result
        _persist_summary_cache()

    return {**safe_result, "cached": False}


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="WARP-RM Inspector", docs_url="/api/docs")


class LoadCheckpointBody(BaseModel):
    path: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/checkpoints")
def api_checkpoints():
    current = str(STATE.ckpt_path) if STATE.ckpt_path else None
    return {
        "checkpoints": _discover_checkpoints(),
        "current": current,
        "current_meta": STATE.ckpt_meta,
    }


@app.post("/api/checkpoint")
def api_load_checkpoint(body: LoadCheckpointBody):
    p = Path(body.path).resolve()
    if not p.exists():
        raise HTTPException(404, f"checkpoint not found: {p}")
    try:
        with STATE.gpu_lock:
            _load_checkpoint_into_state(p)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"checkpoint load failed: {e}")
    return {"current": str(p), "current_meta": STATE.ckpt_meta}


@app.get("/api/datasets")
def api_datasets():
    return {"datasets": _discover_datasets()}


@app.get("/api/episodes")
def api_episodes(repo: str, camera_key: str = "top_camera-images-rgb",
                 split: Optional[str] = None):
    root = Path(repo).resolve()
    if not (root / "meta" / "info.json").exists():
        raise HTTPException(404, f"not a LeRobot dataset: {repo}")
    try:
        episodes = _get_episodes(str(root), camera_key, split=split)
    except Exception as e:
        raise HTTPException(500, f"discover failed: {e}")
    return {
        "repo": str(root),
        "camera_key": camera_key,
        "split": split,
        "episodes": [
            {
                "idx": int(ep.path.name.split("_")[-1]),
                "name": ep.path.name,
                "n_frames": ep.n_frames,
            }
            for ep in episodes
        ],
    }


def _episode_or_404(repo: str, camera_key: str, ep_idx: int,
                    split: Optional[str] = None) -> Episode:
    root = Path(repo).resolve()
    episodes = _get_episodes(str(root), camera_key, split=split)
    target = f"episode_{ep_idx:06d}"
    for ep in episodes:
        if ep.path.name == target:
            return ep
    # Fallback: try without split filter in case the episode exists but is
    # outside the requested split (e.g. video serving for a cross-split ref).
    if split is not None:
        for ep in _get_episodes(str(root), camera_key, split=None):
            if ep.path.name == target:
                return ep
    raise HTTPException(404, f"episode {ep_idx} not in {repo}/{camera_key}")


@app.get("/api/inference")
def api_inference(repo: str, ep_idx: int, camera_key: str = "top_camera-images-rgb"):
    ep = _episode_or_404(repo, camera_key, ep_idx)
    try:
        data = _run_inference(ep)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"inference failed: {e}")
    # Optional abs-head progress overlay. Only present when the loaded ckpt has
    # an absolute-progress head; otherwise None. The integrated rel-head
    # abs_progress already lives in `abs_progress` (above) — this adds the
    # direct head's view, useful as a second-opinion line on the same plot.
    try:
        unc = _compute_abs_head(ep)
        if unc is not None and "abs_progress" in unc:
            n = int(data.get("n_frames", ep.n_frames))
            ah = unc["abs_progress"]
            # Interpolate to the same source-frame grid as data["abs_progress"].
            src_t = np.linspace(0, 1, n, dtype=np.float32)
            feat_t = np.linspace(0, 1, len(ah), dtype=np.float32)
            data["abs_head_progress"] = np.interp(src_t, feat_t, ah).astype(np.float32).tolist()
        else:
            data["abs_head_progress"] = None
    except Exception:
        data["abs_head_progress"] = None
    data.update({
        "repo": str(Path(repo).resolve()),
        "ep_idx": ep_idx,
        "camera_key": camera_key,
        "video_path": str(ep.video_path),
    })
    return data


@app.get("/api/dataset_signals")
def api_dataset_signals(repo: str, ep_idx: int,
                        camera_key: str = "top_camera-images-rgb"):
    """Read the injected reward columns straight from the dataset's parquets.

    No checkpoint required — this returns whatever the dataset *currently has
    written* under the canonical column names::

        warp_rm_progress         → abs_progress  (per-frame, [0,1])
        warp_rm_signed_magnitude → velocity      (per-frame, signed)

    Output shape matches /api/inference so client plotting code is drop-in,
    plus a `meta` block with the last-injected ckpt name (read from
    ``meta/warp_rm_meta.json``) so the UI can display *which* ckpt's signal is
    sitting in the parquets.

    Raises 404 if the dataset has no warp_rm_* columns yet (inject hasn't run).
    """
    import pandas as pd

    ep = _episode_or_404(repo, camera_key, ep_idx)
    root = Path(repo).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise HTTPException(404, f"missing meta/info.json: {root}")
    info = json.loads(info_path.read_text())
    is_v3 = info.get("codebase_version", "v2.0") >= "v3.0"

    needed = [SIDECAR_PROGRESS_COL, SIDECAR_VELOCITY_COL]
    inject_hint = ("run `scripts/data/write_warp_rm_annotations.py --mode inject` "
                   "to populate them")

    # ── Locate the parquet shard + row range for this episode ─────────────
    if is_v3:
        episodes_dir = root / "meta" / "episodes"
        meta_files = sorted(p for p in episodes_dir.rglob("*.parquet")
                            if not p.name.endswith(".bak"))
        if not meta_files:
            raise HTTPException(500, f"no episode-meta parquets under {episodes_dir}")
        ep_row = None
        for pf in meta_files:
            df = pd.read_parquet(pf, columns=[
                "episode_index", "length",
                "data/chunk_index", "data/file_index",
                "dataset_from_index", "dataset_to_index",
            ])
            hit = df[df["episode_index"] == int(ep_idx)]
            if len(hit) > 0:
                ep_row = hit.iloc[0]
                break
        if ep_row is None:
            raise HTTPException(404, f"episode {ep_idx} not in meta/episodes/")
        chunk_index = int(ep_row["data/chunk_index"])
        file_index = int(ep_row["data/file_index"])
        shard_path = (root / "data" / f"chunk-{chunk_index:03d}"
                      / f"file-{file_index:03d}.parquet")
        if not shard_path.is_file():
            raise HTTPException(404, f"data shard not found: {shard_path}")
        # Check the schema before reading: pyarrow raises a noisy FieldRef
        # error on a missing column, and "you haven't injected yet" deserves a
        # 404 with a fix in it, not a 500 with a schema dump.
        import pyarrow.parquet as pq
        try:
            schema_names = set(pq.read_schema(shard_path).names)
        except Exception as e:
            raise HTTPException(500, f"failed to read schema of {shard_path}: {e}")
        missing = [c for c in needed if c not in schema_names]
        if missing:
            raise HTTPException(404, f"dataset parquets lack {missing} — {inject_hint}")
        # Read just the columns we need, then mask to the episode_index match
        # (cheaper than computing a row-group offset from the global range).
        try:
            shard_df = pd.read_parquet(shard_path, columns=["episode_index"] + needed)
        except Exception as e:
            raise HTTPException(500, f"failed to read {shard_path}: {e}")
        ep_rows = shard_df[shard_df["episode_index"] == int(ep_idx)]
    else:
        # v2.1: one parquet per episode
        chunks_size = int(info.get("chunks_size", 1000))
        chunk = ep_idx // chunks_size
        ep_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not ep_path.is_file():
            raise HTTPException(404, f"v2.1 episode parquet not found: {ep_path}")
        try:
            ep_rows = pd.read_parquet(ep_path, columns=needed)
        except Exception as e:
            # Most likely the columns don't exist.
            raise HTTPException(
                404, f"dataset parquets lack warp_rm_* columns — {inject_hint} ({e})")

    if len(ep_rows) == 0:
        raise HTTPException(404, f"episode {ep_idx} has no rows in shard")

    n_frames = int(len(ep_rows))
    abs_progress = ep_rows[SIDECAR_PROGRESS_COL].astype("float32").to_numpy()
    velocity = ep_rows[SIDECAR_VELOCITY_COL].astype("float32").to_numpy()
    # Match ckpt-mode `weights` semantics — per-frame max(0, velocity) drives
    # the red-to-green heatmap gradient.
    weights = np.clip(velocity, 0.0, None).astype("float32")

    # ── Read the inject provenance ───────────────────────────────────────
    meta_block: dict = {"injected": False}
    rmeta_path = root / "meta" / "warp_rm_meta.json"
    if rmeta_path.is_file():
        try:
            rmeta = json.loads(rmeta_path.read_text())
            meta_block = {
                "injected": True,
                "current_model": rmeta.get("current_model"),
                "checkpoint_path": rmeta.get("checkpoint_path"),
                "sidecar_version": rmeta.get("sidecar_version"),
                "updated": rmeta.get("updated"),
                "composite": rmeta.get("composite"),
            }
        except Exception:
            pass

    return {
        "source": "dataset_sidecar",
        "n_frames": n_frames,
        "abs_progress": abs_progress.tolist(),
        "velocity": velocity.tolist(),
        "weights": weights.tolist(),
        "abs_head_progress": None,  # not stored — only ckpt inference produces this
        "cached": True,             # signal that this didn't run inference
        "repo": str(root),
        "ep_idx": int(ep_idx),
        "camera_key": camera_key,
        "video_path": str(ep.video_path),
        "meta": meta_block,
    }


@app.get("/api/summary")
def api_summary(repo: str, ep_idx: int, camera_key: str = "top_camera-images-rgb"):
    """Per-episode velocity summary for one episode (runs inference on a miss)."""
    ep = _episode_or_404(repo, camera_key, ep_idx)
    try:
        s = _summary_for_episode(ep)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"summary failed: {e}")
    s.update({
        "repo": str(Path(repo).resolve()),
        "ep_idx": ep_idx,
        "camera_key": camera_key,
        "ckpt": str(STATE.ckpt_path) if STATE.ckpt_path else None,
    })
    return s


@app.get("/api/summary/all")
def api_summary_all(repo: str, camera_key: str = "top_camera-images-rgb",
                    split: Optional[str] = None):
    """Return every *already-computed* velocity summary for the dataset.

    Serves entries from (a) the on-disk JSON cache — shared with
    scripts/eval/score_episodes.py — and (b) a sidecar under
    ``assets/warp_rm_annotations/<dataset>/`` when its metadata.json pins the
    current checkpoint. Does NOT run inference; clients fall back to
    ``/api/summary?ep_idx=N`` for anything missing.
    """
    episodes = _get_episodes(repo, camera_key, split=split)

    if STATE.model is None or STATE.ckpt_path is None:
        raise HTTPException(400, "no checkpoint loaded — POST /api/checkpoint first")
    _maybe_reload_summary_cache()

    # Look up sidecar once per request (cheap — parsed TSVs are memoized).
    sidecar_rows: dict[str, dict] = {}
    if episodes:
        dataset_name = _dataset_name_for_ep(episodes[0])
        if dataset_name is not None:
            side = _sidecar_match_for_ckpt(dataset_name, STATE.ckpt_path)
            if side is not None:
                sidecar_rows = _sidecar_rows(side)

    scores: dict[int, dict] = {}
    source_counts = {"json_cache": 0, "sidecar": 0}
    to_persist: list[tuple[str, dict]] = []
    with STATE.summary_lock:
        snap = STATE.summary_cache
        for ep in episodes:
            try:
                ep_idx = int(ep.path.name.split("_")[-1])
            except (ValueError, IndexError):
                continue
            key = str(ep.path.resolve())
            hit = snap.get(key)
            if hit is not None:
                scores[ep_idx] = hit
                source_counts["json_cache"] += 1
                continue
            side_hit = sidecar_rows.get(ep.path.name)
            if side_hit is not None:
                safe = _json_safe(side_hit)
                scores[ep_idx] = safe
                to_persist.append((key, safe))
                source_counts["sidecar"] += 1

        # Fold sidecar hits into the JSON cache so subsequent single-ep queries
        # + restarts see them without re-parsing.
        if to_persist:
            for k, v in to_persist:
                STATE.summary_cache[k] = v
            _persist_summary_cache()

    return {
        "repo": str(Path(repo).resolve()),
        "camera_key": camera_key,
        "ckpt": str(STATE.ckpt_path),
        "n_episodes": len(episodes),
        "n_cached": len(scores),
        "sources": source_counts,
        "scores": scores,
    }


# ── Video serving ────────────────────────────────────────────────────────────

async def _range_stream(path: Path, start: int, length: int,
                        chunk_size: int = 64 * 1024):
    """Yield the requested byte Range in chunks so the event loop can handle
    client-disconnect cleanly. Chunked streaming prevents the failure mode
    where a browser-aborted video seek leaves a multi-MB chunk queued in the
    kernel send buffer while uvicorn holds a threadpool worker — after enough
    aborted seeks the pool saturates and the UI feels stuck.
    """
    remaining = length
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            buf = f.read(min(chunk_size, remaining))
            if not buf:
                break
            yield buf
            remaining -= len(buf)


_CLIP_CACHE_LOCK = threading.Lock()


def _episode_clip_path(ep: Episode, fps: int) -> Path:
    """Return a video file containing ONLY this episode's frames.

    LeRobot v3.0 packs many episodes into a single shard mp4; serving the raw
    `ep.video_path` would load the whole shard (often hours long) and in-episode
    `<video>.currentTime` seeks would land in the wrong episode. This
    materializes a per-episode clip on first request, caches it under
    ``webui_clips/``, and returns the cached path on subsequent requests.

    LeRobot v2.1 (one episode per video) is the no-op fast path:
    `frame_offset == 0` AND the file's frame count matches `n_frames`, so we
    just hand back `ep.video_path`.
    """
    src = ep.video_path

    needs_slice = ep.frame_offset > 0
    if not needs_slice:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", str(src)],
                capture_output=True, text=True, timeout=15, check=True,
            )
            video_frames = int(r.stdout.strip() or 0)
            # Allow ±5 frames slack for keyframe edge cases / muxer rounding.
            needs_slice = video_frames > ep.n_frames + 5
        except Exception:
            # ffprobe missing or failed → assume per-episode (v2.1 path).
            needs_slice = False
    if not needs_slice:
        return src

    # Lazy-build cached clip. Key: (src, frame_offset, n_frames) — these
    # uniquely identify the slice within a given dataset version.
    src_id = hashlib.md5(
        f"{src}:{ep.frame_offset}:{ep.n_frames}".encode()
    ).hexdigest()[:16]
    clip_path = CLIP_CACHE_DIR / f"{src_id}.mp4"
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return clip_path

    with _CLIP_CACHE_LOCK:
        # Double-checked: another request may have built it while we waited.
        if clip_path.exists() and clip_path.stat().st_size > 0:
            return clip_path
        CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        start_s = ep.frame_offset / fps
        # Fast-seek (`-ss` before `-i`) tells ffmpeg to jump to the nearest
        # keyframe in the demuxer, avoiding decode of all preceding frames.
        # Without it, seeking must decode every frame 0..N — 30-60s+ on large
        # v3 shards. Frames between the keyframe and the seek target are
        # decoded but discarded, so the output starts cleanly with no bleed-in
        # from the previous episode. `-frames:v` caps output length so we never
        # overrun into the next episode. ±1-2 frame imprecision at the start is
        # acceptable; the JS clip windowing handles fine positioning.
        tmp = clip_path.with_name(clip_path.stem + ".tmp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.6f}",
            "-i", str(src),
            "-frames:v", str(int(ep.n_frames)),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart",
            str(tmp),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            os.replace(tmp, clip_path)
        except subprocess.CalledProcessError as e:
            tmp.unlink(missing_ok=True)
            err_tail = (e.stderr.decode()[-300:] if e.stderr else "unknown")
            raise HTTPException(500, f"ffmpeg slice failed: {err_tail}")
        except subprocess.TimeoutExpired:
            tmp.unlink(missing_ok=True)
            raise HTTPException(500, "ffmpeg slice timed out (>120s)")
        except FileNotFoundError:
            raise HTTPException(
                500, "ffmpeg not found — required to slice LeRobot v3.0 "
                     "concatenated video shards into per-episode clips")
    return clip_path


@app.get("/api/video")
async def api_video(repo: str, ep_idx: int,
                    camera_key: str = "top_camera-images-rgb",
                    fps: int = 30, request: Request = None):
    ep = _episode_or_404(repo, camera_key, ep_idx)
    if not ep.video_path.exists():
        raise HTTPException(404, f"missing video: {ep.video_path}")

    served = _episode_clip_path(ep, fps)
    file_size = served.stat().st_size
    range_header = request.headers.get("range") if request else None
    if range_header and range_header.startswith("bytes="):
        start_str, _, end_str = range_header[6:].partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(
            _range_stream(served, start, length),
            status_code=206, headers=headers, media_type="video/mp4",
        )

    return FileResponse(served, media_type="video/mp4",
                        headers={"Accept-Ranges": "bytes"})


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@dataclasses.dataclass
class Args:
    """Serve the WARP-RM inspector web UI."""

    host: str = "127.0.0.1"
    """Bind address. Use 0.0.0.0 to expose on the network."""

    port: int = 8000

    checkpoint: Optional[str] = None
    """Initial checkpoint to load at startup (optional — the UI can switch)."""

    dataset_root: list[str] = dataclasses.field(default_factory=list)
    """Directories to scan (recursively) for LeRobot datasets.

    Pass several SPACE-SEPARATED after one flag — `--dataset-root A B` — not by
    repeating the flag (`--dataset-root A --dataset-root B` keeps only B).
    Defaults to ~/.cache/huggingface/lerobot, ~/data/lerobot and ~/datasets,
    or the os.pathsep-separated WARP_RM_LEROBOT_ROOTS env var when set."""

    gpu: Optional[int] = None
    """GPU ID to use for inference."""


def main(args: Args):
    global STATE, DATASET_ROOTS
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.dataset_root:
        roots = [Path(p) for p in args.dataset_root]
    elif os.environ.get("WARP_RM_LEROBOT_ROOTS"):
        roots = [Path(p) for p in
                 os.environ["WARP_RM_LEROBOT_ROOTS"].split(os.pathsep) if p]
    else:
        roots = list(DEFAULT_DATASET_ROOTS)
    DATASET_ROOTS = [p.expanduser().resolve() for p in roots]
    for p in DATASET_ROOTS:
        print(f"Dataset root: {p}{'' if p.exists() else '  (missing)'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Building {BACKBONE_NAME} backbone...")
    backbone_obj, _ = build_backbone(BACKBONE_NAME)
    backbone_obj = backbone_obj.to(device).eval()

    STATE = State(
        device=device,
        backbone=backbone_obj,
        backbone_mean=backbone_obj.MEAN,
        backbone_std=backbone_obj.STD,
    )

    if args.checkpoint:
        p = Path(args.checkpoint).resolve()
        print(f"Loading checkpoint: {p}")
        _load_checkpoint_into_state(p)

    INFER_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Discovered {len(_discover_checkpoints())} checkpoints, "
          f"{len(_discover_datasets())} LeRobot datasets.")
    print(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(tyro.cli(Args))
