"""Per-episode progress curves for piecewise-linear reward shaping.

Each curve is a 1-D array of length ``n_frames`` with values in [0, 1] giving
"task progress at frame f" for one episode. Three shapes:

  - ``uniform``: progress(f) = f / n_frames. Reproduces the legacy labeler.
  - ``piecewise_equal``: each canonical segment contributes 1/N to progress
    (chronological order, equal weight).
  - ``piecewise_weighted``: each canonical segment contributes its
    dataset-average duration share (renormalized to sum to 1.0) — long
    canonical stages like "flatten" get bigger progress jumps than brief
    ones like "reset arms".

Both piecewise variants use **chronological frame order** (segments sorted
by ``from_frame``), so out-of-order canonical stage IDs don't produce
non-monotone curves. Uncovered frames (mistake gaps, pre-first, post-last)
hold flat at the previous boundary value — the model learns "no canonical
task progress during these frames."

Sidecar location: ``meta/episodes_segments.jsonl`` (per-episode rows) and
``meta/canonical_stage_map.json`` (dataset-level stage weights), produced
by ``scripts/data/build_canonical_segments.py``. Datasets without these
files only support ``--progress-shape uniform``; piecewise shapes silently
fall back to uniform per-episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# Shape names used by the --progress-shape CLI flag and downstream logging.
SHAPE_UNIFORM = "uniform"
SHAPE_PIECEWISE_EQUAL = "piecewise_equal"
SHAPE_PIECEWISE_WEIGHTED = "piecewise_weighted"
VALID_SHAPES = (SHAPE_UNIFORM, SHAPE_PIECEWISE_EQUAL, SHAPE_PIECEWISE_WEIGHTED)


def uniform_progress(n_frames: int) -> np.ndarray:
    """progress(f) = f / n_frames. Final element = (n-1)/n, NOT 1.0.

    This off-by-one matches the legacy ``RelativeCumulativeLabeler``
    normalization exactly: under uniform fallback, the new labeler with
    ``progress_per_frame * scale`` produces bit-exact identical labels
    to the old labeler. Use this curve for unannotated episodes when
    the run is in piecewise mode but the episode has no sidecar entry.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    return (np.arange(n_frames, dtype=np.float32) / float(n_frames))


def piecewise_progress(
    n_frames: int,
    canonical_segments: list[dict],
    stage_weights: Optional[dict[int, float]] = None,
    stage_field: str = "stage_14",
) -> np.ndarray:
    """Piecewise progress curve for one episode.

    ``canonical_segments`` is the per-episode list from the sidecar; the
    function defensively sorts by ``from_frame`` (matches the
    ``build_canonical_segments.py`` invariant since 2026-04-29). Within
    each segment, progress ramps linearly from the running cumulative
    boundary by either ``1/N`` (when ``stage_weights`` is None) or
    ``stage_weights[segment.stage_field]`` (duration-weighted). Frames
    not covered by any canonical segment hold flat at the last boundary.
    """
    progress = np.zeros(n_frames, dtype=np.float32)
    if n_frames <= 0 or not canonical_segments:
        return progress
    segs = sorted(
        canonical_segments,
        key=lambda s: (int(s["from_frame"]), int(s["to_frame"])),
    )
    n_canonical = len(segs)
    last_boundary = 0.0
    cursor = 0
    for seg in segs:
        if stage_weights is not None:
            stage = seg.get(stage_field)
            w = float(stage_weights[int(stage)]) if stage is not None and int(stage) in stage_weights else (1.0 / n_canonical)
        else:
            w = 1.0 / n_canonical
        f0 = max(0, int(seg["from_frame"]))
        f1 = min(n_frames, int(seg["to_frame"]))
        # Fill gap before this segment with last_boundary (mistake gaps).
        progress[cursor:f0] = last_boundary
        denom = max(1, f1 - f0)
        start_p = last_boundary
        end_p = last_boundary + w
        # Linear ramp from start_p (exclusive) → end_p (inclusive at f1-1).
        ramp = start_p + (np.arange(1, f1 - f0 + 1, dtype=np.float32) / denom) * (end_p - start_p)
        progress[f0:f1] = ramp
        last_boundary = end_p
        cursor = max(cursor, f1)
    progress[cursor:] = last_boundary
    return progress


def load_segments_sidecar(dataset_root: Path) -> dict[int, dict]:
    """Load ``meta/episodes_segments.jsonl`` keyed by ep_idx.

    Returns an empty dict when the sidecar isn't present — caller falls
    back to uniform per-episode.
    """
    p = dataset_root / "meta" / "episodes_segments.jsonl"
    if not p.is_file():
        return {}
    out: dict[int, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["ep_idx"])] = row
    return out


def load_stage_weights(dataset_root: Path,
                       stage_field: str = "weights_14") -> Optional[dict[int, float]]:
    """Load per-stage average-duration weights from
    ``meta/canonical_stage_map.json``. Returns None when the map or the
    requested ``weights_*`` key is missing.
    """
    p = dataset_root / "meta" / "canonical_stage_map.json"
    if not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    raw = obj.get(stage_field)
    if not isinstance(raw, dict):
        return None
    return {int(k): float(v) for k, v in raw.items()}


def build_progress_curve(
    episode_n_frames: int,
    shape: str,
    sidecar_row: Optional[dict],
    stage_weights: Optional[dict[int, float]],
) -> np.ndarray:
    """Pick a progress curve for one episode under the chosen shape.

    Falls back to uniform when piecewise was requested but no sidecar
    row is available for this episode.
    """
    if shape not in VALID_SHAPES:
        raise ValueError(f"unknown progress_shape={shape!r}; "
                         f"expected one of {VALID_SHAPES}")
    if shape == SHAPE_UNIFORM or sidecar_row is None:
        return uniform_progress(episode_n_frames)
    canonical = sidecar_row.get("canonical_segments", []) or []
    if not canonical:
        return uniform_progress(episode_n_frames)
    if shape == SHAPE_PIECEWISE_EQUAL:
        return piecewise_progress(episode_n_frames, canonical, stage_weights=None)
    # SHAPE_PIECEWISE_WEIGHTED
    return piecewise_progress(
        episode_n_frames, canonical, stage_weights=stage_weights, stage_field="stage_14",
    )


def attach_progress_curves(
    episodes: list,
    shape: str,
    dataset_roots: list[Path],
) -> tuple[int, int]:
    """Attach ``episode.progress_per_frame`` to every episode in the list.

    Tries each entry in ``dataset_roots`` (typically the configured
    --lerobot-repo paths) and matches episodes by their ``episode_index``
    against the sidecar of any root that has one. Episodes that don't match
    any sidecar fall back to uniform (which is fine — same behavior as the
    legacy labeler for those episodes).

    Returns ``(n_with_sidecar, n_with_uniform_fallback)`` for logging.
    """
    sidecars: list[tuple[Path, dict[int, dict], Optional[dict[int, float]]]] = []
    for root in dataset_roots:
        if root is None:
            continue
        rows = load_segments_sidecar(Path(root))
        if not rows:
            continue
        weights = load_stage_weights(Path(root), "weights_14")
        sidecars.append((Path(root), rows, weights))

    n_hit, n_miss = 0, 0
    for ep in episodes:
        sidecar_row: Optional[dict] = None
        weights: Optional[dict[int, float]] = None
        ep_idx = getattr(ep, "episode_index", None)
        if ep_idx is not None:
            for root, rows, w in sidecars:
                # Match if this episode's path is under `root` and ep_idx is in rows.
                ep_path = str(getattr(ep, "path", ""))
                if str(root) in ep_path and int(ep_idx) in rows:
                    sidecar_row = rows[int(ep_idx)]
                    weights = w
                    break
        ep.progress_per_frame = build_progress_curve(
            ep.n_frames, shape, sidecar_row, weights,
        )
        if sidecar_row is not None and shape != SHAPE_UNIFORM:
            n_hit += 1
        else:
            n_miss += 1
    return n_hit, n_miss
