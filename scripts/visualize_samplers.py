#!/usr/bin/env python3
"""
Side-by-side comparison of the three window samplers:
  1. UnifiedSampler      — legacy 3-axis (direction × speed × magnitude), DEPRECATED
  2. ContinuousWarpSampler — Beta-CDF warp + optional single mid-flip (current default)
  3. ARSampler            — AR(1) log-speed + Poisson reversals + full-flip (new)

UnifiedSampler is no longer in warp_rm/data/samplers.py (removed in 1f2470c); we
inline a read-only copy here purely for visual comparison.

Outputs a 2-row figure:
  Row 1: source-time vs window-position trajectories (one line per sample)
  Row 2: per-step stride distribution + per-window reversal-count distribution
          (all three samplers overlaid)

Usage:
    uv run python scripts/visualize_samplers.py
    uv run python scripts/visualize_samplers.py --duration 90 --n-samples 25 --n-stats 2000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.config import (
    FEATURE_STRIDE,
    SOURCE_STANDARD_STRIDE,
    STRIDE_OPTIONS_SRC,
    STANDARD_FEAT_STEPS,
    WINDOW_SIZE,
)
from warp_rm.data.samplers import (
    ARSampler,
    ContinuousWarpSampler,
    SamplingMode,
    TrajectorySampler,
    TrueARSampler,
)


# ── DEPRECATED UnifiedSampler (local read-only copy for visualization) ─────
# Original lived in warp_rm/data/samplers.py up through commit 1f2470c. Kept here
# only so we can visually diff it against the current samplers.
class UnifiedSampler(TrajectorySampler):
    """Legacy 3-axis sampler: direction × speed-profile × magnitude."""

    def __init__(
        self,
        window_size: int = 20,
        feature_stride: int = 3,
        stride_options_src: list[int] | None = None,
        p_backward: float = 0.50,
        p_mid_flip: float = 0.20,
        p_speed_constant: float = 0.30,
        p_speed_curve: float = 0.35,
        p_standard_speed: float = 0.35,
        standard_feat_steps: int = 15,
        p_edge_anchor: float = 0.10,
    ):
        if stride_options_src is None:
            stride_options_src = [6, 15, 30, 45, 90, 135, 180]
        self.N = window_size
        self.p_backward = p_backward
        self.p_mid_flip = p_mid_flip
        self.p_speed_constant = p_speed_constant
        self.p_speed_curve = p_speed_curve
        self.p_standard_speed = p_standard_speed
        self.standard_feat_steps = standard_feat_steps
        self.p_edge_anchor = p_edge_anchor

        self.min_fs = 1
        self.max_fs = max(s // feature_stride for s in stride_options_src)
        self.log_min = math.log(self.min_fs)
        self.log_max = math.log(self.max_fs)

    def sample_indices(self, n_feat: int, mode: Optional[SamplingMode] = None) -> list[int]:
        N = self.N
        if random.random() < self.p_mid_flip:
            k = random.randint(1, N - 2)
            if random.random() < 0.5:
                directions = [1] * k + [-1] * (N - 1 - k)
            else:
                directions = [-1] * k + [1] * (N - 1 - k)
        else:
            d = -1 if random.random() < self.p_backward else 1
            directions = [d] * (N - 1)

        r = random.random()
        if r < self.p_speed_constant:
            if random.random() < self.p_standard_speed:
                base = self.standard_feat_steps
            else:
                base = max(1, int(math.exp(random.uniform(self.log_min, self.log_max))))
            magnitudes = [base] * (N - 1)
        elif r < self.p_speed_constant + self.p_speed_curve:
            log_start = random.uniform(self.log_min, self.log_max)
            log_end = random.uniform(self.log_min, self.log_max)
            magnitudes = [
                max(1, int(math.exp(log_start + (i / max(1, N - 2)) * (log_end - log_start))))
                for i in range(N - 1)
            ]
        else:
            magnitudes = [
                max(1, int(math.exp(random.uniform(self.log_min, self.log_max))))
                for _ in range(N - 1)
            ]

        steps = [d * m for d, m in zip(directions, magnitudes)]
        traj = [0]
        for s in steps:
            traj.append(traj[-1] + s)

        span = max(traj) - min(traj)
        if span >= n_feat:
            scale = (n_feat - 2) / max(span, 1)
            magnitudes = [max(1, int(m * scale)) for m in magnitudes]
            steps = [d * m for d, m in zip(directions, magnitudes)]
            traj = [0]
            for s in steps:
                traj.append(traj[-1] + s)

        lo = -min(traj)
        hi = (n_feat - 1) - max(traj)
        if hi > lo and random.random() < self.p_edge_anchor:
            offset = lo if random.random() < 0.5 else hi
        else:
            offset = random.randint(lo, max(lo, hi))
        return [max(0, min(n_feat - 1, t + offset)) for t in traj]


# ── Trajectory plotting ────────────────────────────────────────────────────
def draw_trajectories(ax, sampler, n_feat, n_samples, title, sec_per_feat):
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_samples - 1, 1)) for i in range(n_samples)]
    window_size = None
    for i in range(n_samples):
        indices = sampler.sample_indices(n_feat)
        window_size = len(indices)
        times = [idx * sec_per_feat for idx in indices]
        ys = list(range(1, window_size + 1))
        ax.plot(times, ys, color=colors[i], alpha=0.55, linewidth=1.6)
        ax.scatter(times, ys, color=colors[i], alpha=0.7, s=12, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Source time (s)")
    ax.set_ylabel("Window position (1..N)")
    ax.set_xlim(0, (n_feat - 1) * sec_per_feat)
    ax.set_ylim(0.5, window_size + 0.5)
    ax.grid(True, alpha=0.3)


# ── Stats collection across many samples ──────────────────────────────────
def collect_stats(sampler, n_feat, n_windows, sec_per_feat):
    """Returns (per_step_seconds, reversal_counts_per_window)."""
    stride_secs = []
    reversals = []
    for _ in range(n_windows):
        idx = np.asarray(sampler.sample_indices(n_feat), dtype=np.int64)
        diffs = np.diff(idx)
        stride_secs.extend(np.abs(diffs) * sec_per_feat)
        # Count reversals: sign changes (ignoring zero-stride steps)
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        n_rev = int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0
        reversals.append(n_rev)
    return np.asarray(stride_secs), np.asarray(reversals)


AR_PRESETS = [
    # (label, kwargs-overrides-on-top-of-defaults, color)
    ("default (α=0.5, σ=ln2, λ=1, 1.5±1.0s)",
     dict(), "tab:orange"),
    ("more reversals (λ=3)",
     dict(lambda_reversals=3.0), "tab:red"),
    ("jittery (α=0.1, σ=ln3)",
     dict(alpha=0.1, sigma_inf=math.log(3)), "tab:green"),
    ("smooth+wild (α=0.9, σ=ln3)",
     dict(alpha=0.9, sigma_inf=math.log(3)), "tab:purple"),
    ("wide path (3.0±2.5s, λ=2)",
     dict(center_stride_sec=3.0, half_range_sec=2.5, lambda_reversals=2.0), "tab:brown"),
    ("all-aggressive (α=0.9, σ=ln4, λ=3, 3±2.5s)",
     dict(alpha=0.9, sigma_inf=math.log(4), lambda_reversals=3.0,
          center_stride_sec=3.0, half_range_sec=2.5), "tab:pink"),
]


def build_ar(overrides):
    return ARSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
        **overrides,
    )


TRUEAR_PRESETS = [
    ("default (α=0.5, σ=ln2, λ=1)", dict(), "tab:orange"),
    ("more reversals (λ=3)",
     dict(lambda_reversals=3.0), "tab:red"),
    ("jittery (α=0.1, σ=ln3)",
     dict(alpha=0.1, sigma_inf=math.log(3)), "tab:green"),
    ("smooth (α=0.9, σ=ln2)",
     dict(alpha=0.9, sigma_inf=math.log(2)), "tab:purple"),
    ("wide (σ=ln4)",
     dict(sigma_inf=math.log(4)), "tab:brown"),
    ("aggressive (α=0.9, σ=ln4, λ=3)",
     dict(alpha=0.9, sigma_inf=math.log(4), lambda_reversals=3.0), "tab:pink"),
]


def build_truear(overrides):
    return TrueARSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
        source_standard_stride=SOURCE_STANDARD_STRIDE,
        **overrides,
    )


def render_ar_sweep(args, fps, sec_per_feat, n_feat):
    presets = [(name, build_ar(kw), color) for name, kw, color in AR_PRESETS]
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.0], hspace=0.45, wspace=0.25)

    traj_axes = []
    for i, (name, sampler, _color) in enumerate(presets):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c], sharey=traj_axes[0] if traj_axes else None)
        draw_trajectories(ax, sampler, n_feat, args.n_samples, name, sec_per_feat)
        traj_axes.append(ax)

    ax_stride = fig.add_subplot(gs[2, 0:2])
    ax_rev = fig.add_subplot(gs[2, 2])

    stride_all, rev_all = [], []
    for name, sampler, color in presets:
        strides, revs = collect_stats(sampler, n_feat, args.n_stats, sec_per_feat)
        stride_all.append((name, strides, color))
        rev_all.append((name, revs, color))

    max_stride = max(s.max() for _, s, _ in stride_all if s.size) if stride_all else 5.0
    bins = np.linspace(0, min(max_stride, 20.0), 60)
    for name, strides, color in stride_all:
        ax_stride.hist(strides, bins=bins, alpha=0.35, label=name, color=color, density=True)
    ax_stride.set_xlabel("Per-step stride (seconds)")
    ax_stride.set_ylabel("Density")
    ax_stride.set_title("Per-step stride distribution")
    ax_stride.legend(loc="upper right", fontsize=8)
    ax_stride.grid(True, alpha=0.3)

    max_rev = max(r.max() for _, r, _ in rev_all if r.size) if rev_all else 3
    rev_bins = np.arange(-0.5, max_rev + 1.5, 1)
    for name, revs, color in rev_all:
        ax_rev.hist(revs, bins=rev_bins, alpha=0.35, label=name, color=color, density=True)
    ax_rev.set_xlabel("Reversals per window")
    ax_rev.set_ylabel("Density")
    ax_rev.set_title("Direction-reversals per window")
    ax_rev.set_xticks(np.arange(0, max_rev + 1))
    ax_rev.legend(loc="upper right", fontsize=8)
    ax_rev.grid(True, alpha=0.3)

    fig.suptitle(
        f"ARSampler parameter sweep  |  episode={args.duration:.0f}s  "
        f"|  window_size={WINDOW_SIZE}  |  trajectories/n={args.n_samples}  |  stats/n={args.n_stats}",
        fontsize=13,
    )
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")

    print("\nAR preset summary (stride p05/p50/p95 seconds, reversals mean/max, P(≥1 rev)):")
    for (name, strides, _), (_, revs, _) in zip(stride_all, rev_all):
        p5, p50, p95 = np.percentile(strides, [5, 50, 95])
        print(f"  {name:45s}  stride {p5:.2f}/{p50:.2f}/{p95:.2f}  "
              f"|  rev mean={revs.mean():.2f} max={revs.max()}  "
              f"|  P(≥1)={(revs >= 1).mean():.2f}")


def render_truear_sweep(args, fps, sec_per_feat, n_feat):
    presets = [(name, build_truear(kw), color) for name, kw, color in TRUEAR_PRESETS]
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.0], hspace=0.45, wspace=0.25)

    traj_axes = []
    for i, (name, sampler, _color) in enumerate(presets):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c], sharey=traj_axes[0] if traj_axes else None)
        draw_trajectories(ax, sampler, n_feat, args.n_samples, name, sec_per_feat)
        traj_axes.append(ax)

    ax_stride = fig.add_subplot(gs[2, 0:2])
    ax_rev = fig.add_subplot(gs[2, 2])

    stride_all, rev_all = [], []
    for name, sampler, color in presets:
        strides, revs = collect_stats(sampler, n_feat, args.n_stats, sec_per_feat)
        stride_all.append((name, strides, color))
        rev_all.append((name, revs, color))

    max_stride = max(s.max() for _, s, _ in stride_all if s.size) if stride_all else 5.0
    bins = np.linspace(0, min(max_stride, 20.0), 60)
    for name, strides, color in stride_all:
        ax_stride.hist(strides, bins=bins, alpha=0.35, label=name, color=color, density=True)
    ax_stride.set_xlabel("Per-step stride (seconds)")
    ax_stride.set_ylabel("Density")
    ax_stride.set_title("Per-step stride distribution")
    ax_stride.legend(loc="upper right", fontsize=8)
    ax_stride.grid(True, alpha=0.3)

    max_rev = max(r.max() for _, r, _ in rev_all if r.size) if rev_all else 3
    rev_bins = np.arange(-0.5, max_rev + 1.5, 1)
    for name, revs, color in rev_all:
        ax_rev.hist(revs, bins=rev_bins, alpha=0.35, label=name, color=color, density=True)
    ax_rev.set_xlabel("Reversals per window")
    ax_rev.set_ylabel("Density")
    ax_rev.set_title("Direction-reversals per window")
    ax_rev.set_xticks(np.arange(0, max_rev + 1))
    ax_rev.legend(loc="upper right", fontsize=8)
    ax_rev.grid(True, alpha=0.3)

    fig.suptitle(
        f"TrueARSampler parameter sweep (no path budget)  |  episode={args.duration:.0f}s  "
        f"|  window_size={WINDOW_SIZE}  |  trajectories/n={args.n_samples}  |  stats/n={args.n_stats}",
        fontsize=13,
    )
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")

    print("\nTrueAR preset summary (stride p05/p50/p95 seconds, reversals mean/max, P(≥1 rev)):")
    for (name, strides, _), (_, revs, _) in zip(stride_all, rev_all):
        p5, p50, p95 = np.percentile(strides, [5, 50, 95])
        print(f"  {name:45s}  stride {p5:.2f}/{p50:.2f}/{p95:.2f}  "
              f"|  rev mean={revs.mean():.2f} max={revs.max()}  "
              f"|  P(≥1)={(revs >= 1).mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["compare", "ar-sweep", "truear-sweep"], default="compare",
                        help="'compare' = Unified vs CW vs AR-default vs TrueAR (4 panels). "
                             "'ar-sweep' = 6 AR parameter presets side-by-side. "
                             "'truear-sweep' = 6 TrueAR parameter presets side-by-side.")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulated episode duration in seconds (default: 60)")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Trajectories drawn per panel (default: 15)")
    parser.add_argument("--n-stats", type=int, default=1500,
                        help="Windows drawn per panel for histograms (default: 1500)")
    parser.add_argument("--save", type=str, default="sampler_comparison.png",
                        help="Output path (default: sampler_comparison.png)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fps = 30
    sec_per_feat = FEATURE_STRIDE / fps
    n_feat = int(args.duration / sec_per_feat)

    if args.mode == "ar-sweep":
        render_ar_sweep(args, fps, sec_per_feat, n_feat)
        return
    if args.mode == "truear-sweep":
        render_truear_sweep(args, fps, sec_per_feat, n_feat)
        return

    unified = UnifiedSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
        stride_options_src=STRIDE_OPTIONS_SRC,
        standard_feat_steps=STANDARD_FEAT_STEPS,
    )
    cw = ContinuousWarpSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
        stride_options_src=STRIDE_OPTIONS_SRC,
        standard_feat_steps=STANDARD_FEAT_STEPS,
    )
    ar = ARSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
    )
    truear = TrueARSampler(
        window_size=WINDOW_SIZE,
        feature_stride=FEATURE_STRIDE,
        source_standard_stride=SOURCE_STANDARD_STRIDE,
    )
    samplers = [
        ("UnifiedSampler (deprecated)", unified, "tab:gray"),
        ("ContinuousWarpSampler (current)", cw, "tab:blue"),
        ("ARSampler", ar, "tab:orange"),
        ("TrueARSampler (no budget)", truear, "tab:green"),
    ]

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    # Row 1: trajectories
    traj_axes = []
    for col, (name, sampler, _color) in enumerate(samplers):
        ax = fig.add_subplot(gs[0, col], sharey=traj_axes[0] if traj_axes else None)
        draw_trajectories(ax, sampler, n_feat, args.n_samples, name, sec_per_feat)
        traj_axes.append(ax)

    # Row 2: stats overlaid
    ax_stride = fig.add_subplot(gs[1, 0:3])
    ax_rev = fig.add_subplot(gs[1, 3])

    stride_all = []
    rev_all = []
    for name, sampler, color in samplers:
        strides, revs = collect_stats(sampler, n_feat, args.n_stats, sec_per_feat)
        stride_all.append((name, strides, color))
        rev_all.append((name, revs, color))

    max_stride = max(s.max() for _, s, _ in stride_all if s.size) if stride_all else 5.0
    bins = np.linspace(0, min(max_stride, 15.0), 60)
    for name, strides, color in stride_all:
        ax_stride.hist(strides, bins=bins, alpha=0.45, label=name, color=color, density=True)
    ax_stride.set_xlabel("Per-step stride (seconds of source time between consecutive sampled frames)")
    ax_stride.set_ylabel("Density")
    ax_stride.set_title("Per-step stride distribution (|Δt| between consecutive window positions)")
    ax_stride.legend(loc="upper right", fontsize=9)
    ax_stride.grid(True, alpha=0.3)

    max_rev = max(r.max() for _, r, _ in rev_all if r.size) if rev_all else 3
    rev_bins = np.arange(-0.5, max_rev + 1.5, 1)
    for name, revs, color in rev_all:
        ax_rev.hist(revs, bins=rev_bins, alpha=0.45, label=name, color=color, density=True)
    ax_rev.set_xlabel("Reversals per window")
    ax_rev.set_ylabel("Density")
    ax_rev.set_title("Direction-reversals per window")
    ax_rev.set_xticks(np.arange(0, max_rev + 1))
    ax_rev.legend(loc="upper right", fontsize=9)
    ax_rev.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sampler comparison  |  episode={args.duration:.0f}s  "
        f"|  window_size={WINDOW_SIZE}  |  feature_stride={FEATURE_STRIDE}  "
        f"|  trajectories/n={args.n_samples}  |  stats/n={args.n_stats}",
        fontsize=12,
    )

    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")

    # Also print terse summary stats to stdout
    print("\nSummary (median / p05 / p95 per-step stride in seconds, mean reversals):")
    for (name, strides, _), (_, revs, _) in zip(stride_all, rev_all):
        if strides.size:
            p5, p50, p95 = np.percentile(strides, [5, 50, 95])
            print(f"  {name:38s}  stride_sec p05/50/95 = {p5:.2f} / {p50:.2f} / {p95:.2f}  "
                  f"|  reversals mean={revs.mean():.2f}  max={revs.max()}  "
                  f"|  P(≥1 rev)={(revs >= 1).mean():.2f}")


if __name__ == "__main__":
    main()
