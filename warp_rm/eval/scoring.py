"""
Shared scoring pipeline — the single source of truth for running WARP-RM
dense inference + Q aggregation over a LeRobot (or raw) dataset.

Consumers (all delegate into this module):
  - `scripts/eval/score_episodes.py` — TSV-only CLI.
  - `scripts/data/write_warp_rm_annotations.py` — full sidecar + LeRobot
    field injection.
  - future cloud scoring jobs.

Design:
  - `score_dataset()` is an iterator, yielding one `EpisodeScore` per
    episode. Callers decide what to do with each (write TSV row, add
    to parquet, update webUI cache, etc.).
  - `write_sidecar()` is the standard persistence — writes
    `metadata.json`, `episode_quality.tsv`, `dense_predictions.parquet`
    in the layout `docs/dataset_schema.md` describes.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from ..data.dataset import Episode
from ..visualization.inference import (
    dense_inference_relative,
    dense_inference_delta,
    dense_inference_absolute,
    has_abs_progress_head,
)


__all__ = [
    "EpisodeScore",
    "score_dataset",
    "write_sidecar",
    "load_sidecar_rows",
]


@dataclass
class EpisodeScore:
    """One episode's full scoring output (feature-frequency arrays + raw stats).

    Carries the dense per-frame outputs (``abs_progress`` / ``velocity``) plus
    two raw per-episode summary stats (``mean_velocity`` = mean(velocity),
    ``delta_v_std`` = mean(|Δvelocity|)). No Q/P/S aggregation — that
    human-curation scorecard was removed from the public interface.
    """

    episode: Episode
    ep_idx: int                # index into the episodes list (0..N-1)
    abs_progress: np.ndarray   # (n_feat,) velocity-integrated, [0, 1]
    velocity: np.ndarray       # (n_feat,) per-step velocity
    n_feat: int                # number of feature-stride frames
    mean_velocity: float       # mean(velocity) over feature-stride frames
    delta_v_std: float         # mean(|Δvelocity|) — smoothness proxy
    abs_direct: Optional[np.ndarray] = None   # direct abs-head preds, if trained
    abs_entropy: Optional[np.ndarray] = None

    @property
    def episode_name(self) -> str:
        return Path(self.episode.path).name

    @property
    def episode_index_int(self) -> int:
        """LeRobot-style integer episode index parsed from the folder name
        (e.g. 'episode_000042' → 42). Falls back to the iterator index."""
        name = self.episode_name
        if name.startswith("episode_"):
            try:
                return int(name.split("_")[-1])
            except ValueError:
                pass
        return self.ep_idx

    def to_source_frame_arrays(self, feature_stride: int) -> dict[str, np.ndarray]:
        """Interpolate feature-frequency signals (warp_rm_progress,
        warp_rm_signed_magnitude) to source-frame frequency so they
        align 1:1 with LeRobot's per-frame index."""
        n_feat = len(self.velocity)
        n_source = self.episode.n_frames
        feat_idx = np.arange(n_feat) * feature_stride
        src_idx = np.arange(n_source)
        progress = np.interp(src_idx, feat_idx, self.abs_progress).astype(np.float32)
        signed_magnitude = np.interp(src_idx, feat_idx, self.velocity).astype(np.float32)
        return {
            "warp_rm_progress": progress,
            "warp_rm_signed_magnitude": signed_magnitude,
        }


def score_dataset(
    model,
    episodes: list[Episode],
    device: torch.device,
    window_size: int,
    standard_feat_steps: int,
    feature_stride: int,
    feature_loader,  # callable(Episode) -> np.ndarray or None
    label_mode: str = "relative",
    label_anchor_frames: int = 15,
) -> Iterator[EpisodeScore]:
    """Iterate episodes and yield their full scoring output.

    `feature_loader` is an injectable: given an Episode, return its
    (n_feat, D) feature array or None if it can't be obtained. Typical
    implementations:

        # cache-only
        def loader(ep):
            return try_load_cached_features(ep, cache_dir, backbone, stride)

        # cache + on-the-fly fallback
        def loader(ep):
            feat = try_load_cached_features(...)
            if feat is None and encoder is not None:
                feat = extract_features_on_the_fly(...)
            return feat

    Separating feature loading lets us plug GPU-batch fast-paths
    without touching the scoring loop.
    """
    enhanced = has_abs_progress_head(model)
    model.eval()
    for i, ep in enumerate(episodes):
        feat = feature_loader(ep)
        if feat is None:
            continue

        if label_mode == "delta":
            abs_prog, vel = dense_inference_delta(
                model, feat, device,
                window_size=window_size,
                standard_feat_steps=standard_feat_steps,
                label_anchor_frames=label_anchor_frames,
                feature_stride=feature_stride,
            )
        else:
            abs_prog, vel = dense_inference_relative(
                model, feat, device,
                window_size=window_size,
                standard_feat_steps=standard_feat_steps,
            )
        abs_direct = None
        abs_entropy = None
        if enhanced:
            unc = dense_inference_absolute(
                model, feat, device,
                window_size=window_size,
                standard_feat_steps=standard_feat_steps,
            )
            if unc:
                abs_direct = unc.get("abs_progress")
                abs_entropy = unc.get("abs_entropy")

        # Raw per-episode summary stats (no Q/P/S aggregation):
        #   mean_velocity = mean(velocity), delta_v_std = mean(|Δvelocity|).
        v = np.asarray(vel, dtype=np.float64)
        n_feat = int(len(v))
        mean_velocity = float(v.mean()) if n_feat else 0.0
        delta_v_std = float(np.abs(np.diff(v)).mean()) if n_feat >= 2 else 0.0

        yield EpisodeScore(
            episode=ep,
            ep_idx=i,
            abs_progress=abs_prog,
            velocity=vel,
            n_feat=n_feat,
            mean_velocity=mean_velocity,
            delta_v_std=delta_v_std,
            abs_direct=abs_direct,
            abs_entropy=abs_entropy,
        )


# ────────────────────────────────────────────────────────────────────
# Sidecar I/O
# ────────────────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def write_sidecar(
    sidecar_dir: Path,
    metadata: dict,
    scores: list[EpisodeScore],
    feature_stride: int,
) -> None:
    """Write metadata.json + episode_quality.tsv + dense_predictions.parquet
    to `sidecar_dir`. Overwrites existing files in the directory; does not
    touch sibling version directories."""
    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    (sidecar_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    # Episode-stats TSV. Emits the two raw per-episode summary stats
    # (mean_velocity, delta_v_std) plus shape info. The Q/P/S curation
    # scorecard was removed from the public interface.
    with open(sidecar_dir / "episode_quality.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "episode", "n_frames", "n_feat",
            "mean_velocity", "delta_v_std",
        ])
        for s in scores:
            w.writerow([
                s.episode_name, s.episode.n_frames, s.n_feat,
                _fmt(s.mean_velocity), _fmt(s.delta_v_std),
            ])

    # Dense predictions parquet (frame-level).
    dense_rows = []
    for s in scores:
        ep_idx_int = s.episode_index_int
        arrays = s.to_source_frame_arrays(feature_stride)
        n_frames = s.episode.n_frames
        for fi in range(n_frames):
            dense_rows.append({
                "episode_index": ep_idx_int,
                "frame_index": fi,
                "warp_rm_progress": float(arrays["warp_rm_progress"][fi]),
                "warp_rm_signed_magnitude": float(arrays["warp_rm_signed_magnitude"][fi]),
            })
    pd.DataFrame(dense_rows).to_parquet(
        sidecar_dir / "dense_predictions.parquet", index=False,
    )

    # mean-velocity distribution report (q_distribution.html + .png).
    # Best-effort: a rendering failure should never block the sidecar write.
    try:
        _render_q_report(sidecar_dir, metadata, scores)
    except Exception as e:
        print(f"  [render] distribution report failed (non-fatal): {e}")


def load_sidecar_rows(sidecar_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """Inverse of write_sidecar: read dense_predictions.parquet and return
    a dict `{episode_index: {field: (n_frames,) ndarray}}`. Used by the
    injection step to re-apply values without rerunning inference.

    The public format requires the canonical `warp_rm_*` columns."""
    path = Path(sidecar_dir) / "dense_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no dense_predictions.parquet at {sidecar_dir}")
    df = pd.read_parquet(path)
    required = {"warp_rm_progress", "warp_rm_signed_magnitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is not a WARP-RM sidecar; missing canonical columns {sorted(missing)}"
        )
    progress_col = "warp_rm_progress"
    signed_col = "warp_rm_signed_magnitude"
    out: dict[int, dict[str, np.ndarray]] = {}
    for ep_idx, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(ep_idx)] = {
            "warp_rm_progress": g[progress_col].to_numpy().astype(np.float32),
            "warp_rm_signed_magnitude": g[signed_col].to_numpy().astype(np.float32),
        }
    return out


# ────────────────────────────────────────────────────────────────────
# Q-distribution report (HTML + PNG, auto-generated per sidecar)
# ────────────────────────────────────────────────────────────────────

_SUBSCORE_COLORS = {
    "P": "#d87c36",  # forward pace
    "E": "#3a7fbd",  # forward-dominance
    "M": "#3f8c59",  # monotonicity (underwater area)
    "F": "#7b5fc2",  # clean fraction (v >= 0)
    "S": "#c7525a",  # smoothness (std(Δv))
    "Q": "#1a1a2e",  # aggregate
}


def _quantiles(values: np.ndarray, qs: list[float]) -> list[float]:
    if len(values) == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(values, q)) for q in qs]


def _svg_histogram(values: np.ndarray, color: str, n_bins: int = 40,
                   width: int = 640, height: int = 220,
                   show_quantiles: bool = True) -> str:
    """Inline SVG histogram with p05/p50/p95 rules."""
    if len(values) == 0:
        return f'<svg viewBox="0 0 {width} {height}"></svg>'

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    counts, edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    max_count = int(counts.max()) if counts.max() > 0 else 1

    pad_l, pad_r, pad_t, pad_b = 38, 12, 12, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    bar_w = plot_w / n_bins

    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}" '
                        'xmlns="http://www.w3.org/2000/svg">']
    # Axes
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" '
        f'x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" '
        'stroke="#bbb" stroke-width="0.8"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{pad_t + plot_h}" '
        'stroke="#bbb" stroke-width="0.8"/>'
    )

    # Bars
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        h = (c / max_count) * plot_h
        x = pad_l + i * bar_w
        y = pad_t + plot_h - h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 0.6:.2f}" '
            f'height="{h:.2f}" fill="{color}" opacity="0.82"/>'
        )

    def xpos(v: float) -> float:
        return pad_l + (v - vmin) / (vmax - vmin) * plot_w

    if show_quantiles:
        qs = _quantiles(values, [0.05, 0.5, 0.95])
        labels = ["p05", "p50", "p95"]
        colors = ["#c7525a", "#1a1a2e", "#c7525a"]
        # Stagger labels vertically when rules would overlap (tight
        # distributions). Label rows at y = pad_t - 2, pad_t - 12, pad_t + plot_h + 22.
        y_rows = [pad_t - 2, pad_t + plot_h + 34, pad_t - 2]
        xs = [xpos(q) for q in qs]
        # If p05 and p50 fall within 24px, shove p50 to the bottom row.
        if abs(xs[1] - xs[0]) < 24:
            y_rows[1] = pad_t + plot_h + 34
        if abs(xs[2] - xs[1]) < 24:
            y_rows[2] = pad_t + plot_h + 44 if y_rows[1] != pad_t + plot_h + 34 else pad_t - 12
        for q, lbl, qc, yy, x in zip(qs, labels, colors, y_rows, xs):
            parts.append(
                f'<line x1="{x:.2f}" y1="{pad_t}" '
                f'x2="{x:.2f}" y2="{pad_t + plot_h}" '
                f'stroke="{qc}" stroke-width="1" stroke-dasharray="3,2" opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{yy}" font-size="9" fill="{qc}" '
                f'font-family="JetBrains Mono" text-anchor="middle">'
                f'{lbl} {q:.3f}</text>'
            )

    # X-axis ticks
    for frac in (0.0, 0.5, 1.0):
        v = vmin + frac * (vmax - vmin)
        x = pad_l + frac * plot_w
        parts.append(
            f'<text x="{x:.2f}" y="{pad_t + plot_h + 14}" '
            'font-size="10" fill="#666" font-family="JetBrains Mono" '
            f'text-anchor="middle">{v:.3f}</text>'
        )

    # Y-axis label (count)
    parts.append(
        f'<text x="{pad_l - 6}" y="{pad_t + 8}" font-size="9" fill="#888" '
        'font-family="JetBrains Mono" text-anchor="end">count</text>'
    )
    parts.append(
        f'<text x="{pad_l - 6}" y="{pad_t + plot_h + 4}" font-size="9" '
        f'fill="#888" font-family="JetBrains Mono" text-anchor="end">0</text>'
    )
    parts.append(
        f'<text x="{pad_l - 6}" y="{pad_t + 16}" font-size="9" fill="#888" '
        f'font-family="JetBrains Mono" text-anchor="end">{max_count}</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def _build_q_report_html(metadata: dict, scores: list[EpisodeScore]) -> str:
    """Render a standalone HTML page summarizing the mean-velocity distribution.

    The Q/P/S curation scorecard was removed from the public interface; this
    report now ranks episodes by the raw ``mean_velocity`` stat.
    """
    mv_vals = np.array([s.mean_velocity for s in scores], dtype=np.float64)
    # Histogram panel 1 ranks by mean_velocity (formerly Q).
    q_vals = mv_vals

    qs = _quantiles(q_vals, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    qmin, q05, q10, q25, q50, q75, q90, q95, qmax = qs
    qmean = float(q_vals.mean()) if len(q_vals) else float("nan")
    qstd = float(q_vals.std()) if len(q_vals) else float("nan")

    # Top / bottom 5 episodes
    order = np.argsort(q_vals)
    bottom = [scores[i] for i in order[:5]]
    top = [scores[i] for i in order[-5:][::-1]]

    def _ep_row(s: EpisodeScore) -> str:
        return (
            '<tr>'
            f'<td class="mono">{s.episode_name}</td>'
            f'<td class="num">{s.mean_velocity:.4f}</td>'
            f'<td class="num">{s.delta_v_std:.4f}</td>'
            f'<td class="num">{s.episode.n_frames}</td>'
            '</tr>'
        )

    mv_mean = float(mv_vals.mean()) if len(mv_vals) else float("nan")
    mv_std = float(mv_vals.std()) if len(mv_vals) else float("nan")
    mv_min = float(mv_vals.min()) if len(mv_vals) else float("nan")
    mv_max = float(mv_vals.max()) if len(mv_vals) else float("nan")
    v_ref_auto = float(metadata.get("v_ref_auto", float("nan")))

    dataset = metadata.get("dataset", "?")
    tag = metadata.get("tag", "?")
    composite = metadata.get("composite", float("nan"))
    n_episodes = metadata.get("n_episodes", len(scores))
    date_utc = metadata.get("date_utc", "?")
    git_sha = (metadata.get("git_sha") or "?")[:12]

    # Threshold suggestion: drop bottom 5%
    drop_thresh = q05
    kept = int((q_vals >= drop_thresh).sum())

    parts: list[str] = []
    parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WARP-RM — Q distribution · {dataset}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@400;500&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ overflow: hidden; scrollbar-width: none; }}
  body {{
    font-family: 'DM Sans', system-ui, sans-serif;
    background: #f8f7f4;
    color: #1a1a2e;
    padding: 32px 40px;
    min-width: 1300px;
  }}
  h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.3px; color: #111; }}
  h2 {{ font-size: 16px; font-weight: 700; color: #2c3e50; margin-bottom: 12px; }}
  h3 {{ font-size: 13px; font-weight: 600; color: #444; margin-bottom: 6px; }}
  .subtitle {{ font-size: 13px; color: #666; margin-top: 4px; max-width: 900px; line-height: 1.55; }}
  .mono {{ font-family: 'JetBrains Mono', monospace; }}
  .muted {{ color: #6f6f7a; font-size: 12px; line-height: 1.55; }}
  .num {{ font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }}
  .eyebrow {{
    display: inline-block; font-size: 10px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    color: #8d7f6a; padding: 2px 8px; border-radius: 3px;
    background: #efece4; margin-bottom: 8px;
  }}
  .page-header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    border-bottom: 1px solid #ddd; padding-bottom: 16px; margin-bottom: 24px;
  }}
  .meta-row {{ font-size: 11px; color: #888; font-family: 'JetBrains Mono', monospace; }}
  .section {{ margin-bottom: 28px; }}
  .card {{
    background: #fff; border: 1.5px solid #ddd; border-radius: 10px;
    padding: 16px 20px;
  }}
  .card svg {{ width: 100%; display: block; }}
  .row {{ display: grid; grid-template-columns: 2.2fr 1fr; gap: 18px; align-items: start; }}
  .sub-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
  .sub-card {{
    background: #fff; border: 1.5px solid #ddd; border-radius: 10px;
    padding: 12px 14px;
  }}
  .sub-head {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2px; }}
  .sub-letter {{ font-size: 18px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .sub-name {{ font-size: 11px; color: #555; }}
  .sub-stats {{ font-size: 10px; color: #888; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }}
  .breakdown {{ display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 12px; }}
  .bd-row {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed #ece9e3; }}
  .bd-label {{ font-family: 'JetBrains Mono', monospace; color: #555; }}
  .bd-val {{ font-family: 'JetBrains Mono', monospace; color: #1a1a2e; font-variant-numeric: tabular-nums; }}
  .bd-row.final .bd-label {{ font-weight: 700; color: #1a1a2e; }}
  .bd-row.final .bd-val {{ font-weight: 700; color: #3a7fbd; }}
  table.ep {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.ep th, table.ep td {{
    padding: 6px 10px; text-align: left;
    border-bottom: 1px solid #ece9e3; font-family: 'JetBrains Mono', monospace;
  }}
  table.ep th {{ font-weight: 600; color: #2c3e50; font-size: 10px; text-transform: uppercase; letter-spacing: 0.4px; border-bottom: 1.5px solid #ddd; }}
  table.ep td.num, table.ep th.num {{ text-align: right; }}
  .top-bot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .top-card {{ border-top: 3px solid #3f8c59; }}
  .bot-card {{ border-top: 3px solid #c7525a; }}
</style>
</head>
<body>

<div class="page-header">
  <div>
    <div class="eyebrow">Data Curation</div>
    <h1>Q Distribution <span class="mono" style="font-size:18px; color:#888;">{dataset}</span></h1>
    <div class="subtitle">
      Episode Quality Index <span class="mono">Q</span> &#8712; [0, 1]: scalar blend of P/E/M/S (plus C/A on enhanced models).
      Higher = cleaner demo. See <a href="https://github.com/" style="color:#5a6b8d;">docs/quality_scoring.html</a>
      for the derivation.
    </div>
  </div>
  <div style="text-align:right;">
    <div class="meta-row">scored by: <span style="color:#1a1a2e; font-weight:600;">{tag}</span></div>
    <div class="meta-row">composite (val): <span style="color:#1a1a2e; font-weight:600;">{composite:.4f}</span></div>
    <div class="meta-row">{date_utc} · git {git_sha}</div>
  </div>
</div>

<div class="section">
  <h2>1 &nbsp; Episode Q histogram <span class="muted">(n = {n_episodes})</span></h2>
  <div class="row">
    <div class="card">
      {_svg_histogram(q_vals, _SUBSCORE_COLORS["Q"], n_bins=50, width=840, height=260)}
      <div class="muted" style="margin-top:8px;">
        Dashed red rules at p05/p95; dashed black at median.
      </div>
    </div>
    <div class="card">
      <h3>Percentiles</h3>
      <div class="breakdown">
        <div class="bd-row"><span class="bd-label">min</span><span class="bd-val">{qmin:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p05</span><span class="bd-val">{q05:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p10</span><span class="bd-val">{q10:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p25</span><span class="bd-val">{q25:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">median</span><span class="bd-val">{q50:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p75</span><span class="bd-val">{q75:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p90</span><span class="bd-val">{q90:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">p95</span><span class="bd-val">{q95:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">max</span><span class="bd-val">{qmax:.4f}</span></div>
        <div class="bd-row"><span class="bd-label">mean &middot; std</span><span class="bd-val">{qmean:.4f} &middot; {qstd:.4f}</span></div>
        <div class="bd-row final"><span class="bd-label">drop-bottom-5% thresh</span><span class="bd-val">Q &ge; {drop_thresh:.4f}</span></div>
        <div class="bd-row final"><span class="bd-label">kept</span><span class="bd-val">{kept} / {n_episodes}</span></div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <h2>2 &nbsp; mean_velocity distribution <span class="muted">(v_ref_auto = {v_ref_auto:.4f})</span></h2>
  <div class="card">
    {_svg_histogram(mv_vals, _SUBSCORE_COLORS.get("P", "#3b82f6"), n_bins=50, width=840, height=200)}
    <div class="muted" style="margin-top:8px;">
      mean={mv_mean:.4f} · std={mv_std:.4f} · [{mv_min:.4f}, {mv_max:.4f}]. Each
      episode's Q = clip(mean_velocity, 0, v_ref_auto) / v_ref_auto, where
      v_ref_auto is the dataset-wide mean — so the histogram and Q distribution
      are scaled images of each other.
    </div>
  </div>
</div>

<div class="section">
  <h2>3 &nbsp; Extremes — cleanest &amp; scrappiest episodes</h2>
  <div class="top-bot-grid">
    <div class="card top-card">
      <h3>Top 5 mean_velocity</h3>
      <table class="ep">
        <thead><tr>
          <th>episode</th><th class="num">mean_velocity</th><th class="num">delta_v_std</th>
          <th class="num">n_frames</th>
        </tr></thead>
        <tbody>''')
    for s in top:
        parts.append(_ep_row(s))
    parts.append('''</tbody>
      </table>
    </div>

    <div class="card bot-card">
      <h3>Bottom 5 mean_velocity</h3>
      <table class="ep">
        <thead><tr>
          <th>episode</th><th class="num">mean_velocity</th><th class="num">delta_v_std</th>
          <th class="num">n_frames</th>
        </tr></thead>
        <tbody>''')
    for s in bottom:
        parts.append(_ep_row(s))
    parts.append('''</tbody>
      </table>
    </div>
  </div>
</div>

</body>
</html>
''')
    return "".join(parts)


def _render_png_from_html(html_path: Path, png_path: Path,
                          width: int = 1400, height: int = 1500) -> None:
    """Best-effort HTML → PNG via headless Chrome. Silent no-op if
    chrome/chromium is missing (HTML still usable)."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium") \
        or shutil.which("chromium-browser")
    if not chrome:
        print("  [render] no chrome/chromium on PATH — skipping PNG")
        return
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", f"--window-size={width},{height}",
        f"--screenshot={png_path}",
        f"file://{html_path.resolve()}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _render_q_report(sidecar_dir: Path, metadata: dict,
                     scores: list[EpisodeScore]) -> None:
    """Write q_distribution.html (+ .png if chrome is available) into
    the sidecar directory. Safe to call on empty score lists."""
    if not scores:
        return
    html_path = sidecar_dir / "q_distribution.html"
    png_path = sidecar_dir / "q_distribution.png"
    html = _build_q_report_html(metadata, scores)
    html_path.write_text(html)
    print(f"  [render] q_distribution.html → {html_path}")
    _render_png_from_html(html_path, png_path)
    if png_path.exists():
        print(f"  [render] q_distribution.png → {png_path}")
