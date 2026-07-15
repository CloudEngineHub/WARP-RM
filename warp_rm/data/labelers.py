"""
Label generation: decides WHAT the ground truth target is for a given trajectory.

Decoupled from sampling and data loading so you can swap label strategies
(relative cumulative, absolute, delta velocity) without touching the dataset.
"""

from typing import Optional

import numpy as np


class LabelGenerator:
    """Base class for label generation strategies."""

    def generate(
        self,
        feat_indices: list[int],
        progress_per_frame: Optional[np.ndarray] = None,
        n_frames: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class RelativeCumulativeLabeler(LabelGenerator):
    """
    Cumulative relative progress labels.

    Two equivalent formulations:

    LEGACY (no progress_per_frame, baked-in uniform progress):
      label[0]  = 0.0
      label[j]  = sum_{i<j} (feat_indices[i+1] - feat_indices[i]) * feature_stride
                           / ((N-1) * source_standard_stride)
      Standard forward → labels [0, 1/(N-1), …, 1.0]; 2x → [0, …, 2.0];
      reverse → [0, …, -1.0]; mid-rewind → non-monotone.

    PIECEWISE (with episode's progress_per_frame array):
      label[j] = (P[src_idx_j] - P[src_idx_0]) * scale
      where P = progress_per_frame, src_idx = feat_idx * feature_stride,
      scale = n_frames / ((N-1) * source_standard_stride).
      Per-episode scale makes a "standard forward window in a uniform-shape
      episode" produce labels [0, 1/(N-1), …, 1.0] — bit-exactly identical
      to LEGACY when P[k] = k/n_frames. With a piecewise P, per-step
      velocity reflects "fraction of canonical task progressed per
      source-frame-stride", giving the model a richer signal where stages
      with bigger dataset-average duration share contribute more progress
      per step than brief ones.
    """

    def __init__(self, feature_stride: int = 3, source_standard_stride: int = 45):
        self.feature_stride = feature_stride
        self.source_standard_stride = source_standard_stride

    def generate(
        self,
        feat_indices: list[int],
        progress_per_frame: Optional[np.ndarray] = None,
        n_frames: Optional[int] = None,
    ) -> np.ndarray:
        N = len(feat_indices)
        if progress_per_frame is None or n_frames is None:
            # Legacy fast path: cumulative delta_src / ((N-1) * sss).
            norm = float((N - 1) * self.source_standard_stride)
            labels = [0.0]
            for i in range(N - 1):
                delta_src = (feat_indices[i + 1] - feat_indices[i]) * self.feature_stride
                labels.append(labels[-1] + delta_src / norm)
            return np.array(labels, dtype=np.float32)
        # Piecewise-aware path. progress_per_frame is shape (n_frames,);
        # we lookup the episode's progress at each sampled source frame
        # and diff to the window's anchor (index 0). Per-episode scale
        # preserves "1.0 per standard forward window" semantics.
        src_idx = np.asarray(
            [int(fi) * self.feature_stride for fi in feat_indices],
            dtype=np.int64,
        )
        # Clamp out-of-bounds (sampler can produce indices == n_feat-1
        # whose source mapping equals n_frames - 1 in the worst case).
        src_idx = np.clip(src_idx, 0, n_frames - 1)
        progress_diffs = progress_per_frame[src_idx] - progress_per_frame[src_idx[0]]
        scale = n_frames / ((N - 1) * self.source_standard_stride)
        return (progress_diffs * scale).astype(np.float32)


class DeltaVelocityLabeler(LabelGenerator):
    """
    Per-step delta (velocity) labels — no cumulative integration across the window.

    label[0]  = 0.0                    (first frame has no prior; pad)
    label[j]  = (feat_indices[j] - feat_indices[j-1]) * feature_stride
                / label_anchor_frames    for j >= 1

    With `label_anchor_frames = 15` (≈ 0.5 s @ 30 fps):
      forward at stride_src = 15 (1.0× pace): label = 1.0 per step
      forward at stride_src = 30 (2× pace):   label = 2.0 per step
      forward at stride_src = 45:             label = 3.0 per step
      forward at stride_src = 90 (the cap):   label = 6.0 per step
      reversed:                               negative with same magnitude
      mid-rewind:                             sign flips mid-window

    Semantic: label units are "fraction of anchor-interval traversed this feat-step".
    Compared to RelativeCumulativeLabeler, this target:
      - is NOT anchored to the window's first frame (window-position invariant)
      - directly encodes local motion (the model can't "cheat" with a trend line)
      - stays anchor-free for dense inference — just integrate the averaged deltas once

    At the paired dense_inference_delta side, we take the average predicted
    delta at each source frame (across overlapping windows) and cumsum once to
    recover abs_progress. No per-step-diff trick needed on the aggregation side.
    """

    def __init__(self, feature_stride: int = 3, label_anchor_frames: int = 15):
        self.feature_stride = feature_stride
        self.label_anchor_frames = int(label_anchor_frames)

    def generate(
        self,
        feat_indices: list[int],
        progress_per_frame: Optional[np.ndarray] = None,
        n_frames: Optional[int] = None,
    ) -> np.ndarray:
        # Delta mode is uniform-only for now; piecewise arguments are
        # accepted for signature parity with RelativeCumulativeLabeler
        # but ignored. Future work: wire piecewise into delta-mode if
        # it becomes useful (extra hyperparam: scale factor).
        del progress_per_frame, n_frames
        N = len(feat_indices)
        labels = np.zeros(N, dtype=np.float32)
        # label[0] = 0.0 by construction. label[j] is the per-step delta,
        # not the cumulative. Downstream loss masks index 0 (no supervision
        # on the "initial state" delta).
        for j in range(1, N):
            delta_src = (feat_indices[j] - feat_indices[j - 1]) * self.feature_stride
            labels[j] = delta_src / self.label_anchor_frames
        return labels


