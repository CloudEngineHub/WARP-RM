"""
Trajectory sampling: decides WHICH frame indices to select from an episode.

Decoupled from label generation and data loading.
"""

import math
import random
from enum import Enum
from typing import Optional

import numpy as np
from scipy.special import betainc


class SamplingMode(Enum):
    STANDARD_FORWARD = "standard_forward"
    VARIABLE_UNIFORM = "variable_uniform"
    NON_UNIFORM = "non_uniform"
    FULL_REWIND = "full_rewind"
    FULL_REWIND_VARIABLE = "full_rewind_variable"
    FULL_REWIND_NONUNIFORM = "full_rewind_nonuniform"
    MID_REWIND = "mid_rewind"
    DOWN_THEN_UP_MID_REWIND = "down_then_up_mid_rewind"


class TrajectorySampler:
    """Base class for trajectory sampling strategies."""

    def sample_indices(self, n_feat: int, mode: Optional[SamplingMode] = None) -> list[int]:
        raise NotImplementedError


class ContinuousWarpSampler(TrajectorySampler):
    """
    Continuous parametric time-warping sampler (Beta-CDF-based).

    Paper-clean formulation: every window's trajectory is an affine scaling
    of a warp function g(tau): [0, 1] -> [0, 1] drawn from the Beta CDF
    family. No discrete speed-mode branches, no integer arithmetic until
    the final index rounding.

        idx(tau) = offset + span * s(tau)
        s(tau)   = direction-applied version of BetaCDF(tau; alpha, beta),
                   smoothly blended through a cosine velocity ramp for
                   mid-flip windows.

    Span + shape priors tuned so the marginal matches UnifiedSampler
    ----------------------------------------------------------------
    Window span (max(idx) - min(idx)) and per-step velocity distribution
    are drawn so the *marginal* matches UnifiedSampler:

      - p_linear_shape of the time (default 0.30): (alpha, beta) = (1, 1)
        which gives g(tau) = tau exactly. Matches UnifiedSampler's
        constant-speed mode marginally.
      - Otherwise: (alpha, beta) are log-uniform in
        [exp(alpha_log_min), exp(alpha_log_max)]; smooth Beta-CDF shapes
        covering convex / concave / S-curve / inverse-S families.

      - With p_standard_speed: span = standard_feat_steps * (N-1)
        (preserves canonical label=1.0 windows); otherwise
        per_step log-uniform on [min_fs, max_fs].

    Shape is always drawn from the Beta-CDF family — no discrete
    constant / log-curve / jitter branches. The Dirac at (alpha, beta)
    = (1, 1) is still Beta (it's the degenerate case g(tau) = tau), so
    the full shape prior lives in one parametric family, which keeps the
    formulation clean for paper framing while tracking UnifiedSampler's
    empirical marginal.

    Mid-flip dynamics
    -----------------
    With probability p_mid_flip, the window crosses zero velocity once.
    The default smooth_flip=True uses a cosine velocity ramp of width
    `flip_blend` steps so g'(tau_peak) = 0 — no instantaneous jerk.
    Set smooth_flip=False for the legacy hard triangular flip.

    Key differences from UnifiedSampler at the marginal level are limited
    to: (a) smooth mid-flip dynamics, (b) float accumulation throughout,
    (c) a smooth parametric shape family instead of 3 discrete modes.
    Span and anchor distributions match UnifiedSampler by construction.
    """

    def __init__(
        self,
        window_size: int = 20,
        feature_stride: int = 3,
        stride_options_src: list[int] | None = None,
        standard_feat_steps: int = 15,
        # Direction
        p_backward: float = 0.50,
        p_mid_flip: float = 0.20,
        # Span: probability the window uses the canonical label=1.0 span
        # (standard_feat_steps × (N-1)). SOTA default 0.35.
        p_standard_speed: float = 0.35,
        # Shape family (Beta CDF over tau). p_linear_shape puts a Dirac at
        # (alpha, beta) = (1, 1) which yields g(tau) = tau exactly.
        p_linear_shape: float = 0.30,
        alpha_log_min: float = -0.7,
        alpha_log_max: float = 1.1,
        # Window anchoring
        p_edge_anchor: float = 0.10,
        # Continuous-warp contributions:
        smooth_flip: bool = True,
        flip_blend: int = 3,
    ):
        if stride_options_src is None:
            stride_options_src = [6, 15, 30, 45, 90, 135, 180]

        self.N = window_size
        self.std_step = standard_feat_steps
        self.p_backward = p_backward
        self.p_mid_flip = p_mid_flip
        self.p_standard_speed = p_standard_speed
        self.p_linear_shape = p_linear_shape
        self.alpha_log_min = alpha_log_min
        self.alpha_log_max = alpha_log_max
        self.p_edge_anchor = p_edge_anchor
        self.smooth_flip = smooth_flip
        self.flip_blend = flip_blend

        self.min_fs = 1
        self.max_fs = max(s // feature_stride for s in stride_options_src)
        self.log_fs_min = math.log(self.min_fs)
        self.log_fs_max = math.log(self.max_fs)

    def _sample_shape(self) -> np.ndarray:
        """Monotone warp shape s in [0, 1] of length N, drawn from Beta CDF.

        With probability p_linear_shape, (alpha, beta) = (1, 1) giving
        g(tau) = tau exactly (the Dirac in the Beta family that tracks
        UnifiedSampler's constant-speed marginal). Otherwise alpha, beta
        log-uniform in [exp(alpha_log_min), exp(alpha_log_max)]: alpha > beta
        accelerates; alpha < beta decelerates; both < 1 produces S-curves.
        """
        tau = np.linspace(0.0, 1.0, self.N)
        if random.random() < self.p_linear_shape:
            return tau.copy()
        alpha = math.exp(random.uniform(self.alpha_log_min, self.alpha_log_max))
        beta = math.exp(random.uniform(self.alpha_log_min, self.alpha_log_max))
        return betainc(alpha, beta, tau).astype(np.float64)

    def _sample_span(self, n_feat: int) -> float:
        """Target span drawn from the same marginal as UnifiedSampler."""
        if random.random() < self.p_standard_speed:
            per_step = float(self.std_step)
        else:
            per_step = math.exp(random.uniform(self.log_fs_min, self.log_fs_max))
        return min(per_step * (self.N - 1), float(n_feat - 1))

    def _apply_direction(self, shape: np.ndarray) -> np.ndarray:
        """Return signed path values in [-max(shape), max(shape)] after
        applying the direction profile. For monotone windows that's just
        ±shape; for mid-flip windows we rebuild the path from a cosine
        velocity envelope to keep dynamics smooth through the inflection.
        """
        N = self.N
        if random.random() < self.p_mid_flip:
            return self._apply_mid_flip(shape)
        d = -1.0 if random.random() < self.p_backward else 1.0
        return d * shape

    def _apply_mid_flip(self, shape: np.ndarray) -> np.ndarray:
        """Mid-flip: velocity crosses zero once. Rebuild path from a signed
        velocity envelope so the inflection has a smooth (optionally
        zero-derivative) sign change.
        """
        N = self.N
        # Use the shape's per-step deltas as raw magnitude, then apply a
        # direction envelope (cosine-blended or hard triangle).
        magnitudes = np.diff(shape)  # length N-1
        k = random.randint(1, N - 2)
        up_first = random.random() < 0.5

        directions = np.empty(N - 1, dtype=np.float64)
        if self.smooth_flip:
            blend = max(1, self.flip_blend)
            for i in range(N - 1):
                x = i - k + 0.5
                if x < -blend:
                    directions[i] = 1.0
                elif x > blend:
                    directions[i] = -1.0
                else:
                    t = (x + blend) / (2.0 * blend)
                    directions[i] = math.cos(math.pi * t)
        else:
            directions[:k] = 1.0
            directions[k:] = -1.0
        if not up_first:
            directions = -directions

        steps = directions * magnitudes
        path = np.concatenate([[0.0], np.cumsum(steps)])
        return path

    def sample_indices(self, n_feat: int, mode: Optional[SamplingMode] = None) -> list[int]:
        # 1. Monotone shape from Beta CDF (always smooth, parametric)
        shape = self._sample_shape()
        # 2. Apply direction (monotone ±1 or smooth mid-flip envelope)
        signed = self._apply_direction(shape)
        # 3. Scale to target span (matched to Unified's per_step marginal)
        span = self._sample_span(n_feat)
        shape_min = float(signed.min())
        shape_max = float(signed.max())
        shape_range = max(shape_max - shape_min, 1e-6)
        # Normalize into [0, span] relative trajectory (anchor at min=0)
        traj = (signed - shape_min) * (span / shape_range)

        # 4. Anchor within episode bounds
        max_offset = max(0.0, (n_feat - 1) - float(traj.max()))
        min_offset = 0.0
        if max_offset > min_offset and random.random() < self.p_edge_anchor:
            offset = min_offset if random.random() < 0.5 else max_offset
        else:
            offset = random.uniform(min_offset, max_offset)

        return [
            max(0, min(n_feat - 1, int(round(offset + t))))
            for t in traj.tolist()
        ]


class EvalSampler:
    """
    Deterministic sampler for evaluation: generates 8 window types x 2 windows per episode.

    Uses a fixed seed for reproducible, low-variance evaluation windows.
    """

    def __init__(
        self,
        window_size: int = 20,
        feature_stride: int = 3,
        stride_options_src: list[int] | None = None,
        standard_feat_steps: int = 15,
        seed: int = 42,
    ):
        if stride_options_src is None:
            stride_options_src = [6, 15, 30, 45, 90, 135, 180]

        self.N = window_size
        self.std_step = standard_feat_steps
        self.min_fs = 1
        self.max_fs = max(s // feature_stride for s in stride_options_src)
        self.log_min = math.log(self.min_fs)
        self.log_max = math.log(self.max_fs)
        self.seed = seed

    def generate_eval_windows(self, n_feat: int) -> list[tuple[list[int], bool]]:
        """
        Returns list of (feat_indices, is_mid_rewind) for all 16 eval windows.

        8 types x 2 windows each:
          std_fwd, var_fwd, nonuniform_fwd,
          std_rew, var_rew, nonuniform_rew,
          mid_up_down, mid_down_up
        """
        rng = random.Random(self.seed)
        N = self.N
        std_step = self.std_step

        def _fwd_std():
            max_s = max(0, n_feat - 1 - (N - 1) * std_step)
            sf = rng.randint(0, max_s)
            return [min(sf + i * std_step, n_feat - 1) for i in range(N)]

        def _fwd_var():
            step = max(1, int(math.exp(rng.uniform(self.log_min, self.log_max))))
            if (N - 1) * step >= n_feat:
                step = max(1, (n_feat - 1) // (N - 1))
            max_s = max(0, n_feat - 1 - (N - 1) * step)
            sf = rng.randint(0, max_s)
            return [min(sf + i * step, n_feat - 1) for i in range(N)]

        def _fwd_nonuniform():
            steps = [max(1, int(math.exp(rng.uniform(self.log_min, self.log_max)))) for _ in range(N - 1)]
            total = sum(steps)
            if total >= n_feat:
                scale = (n_feat - 2) / max(total, 1)
                steps = [max(1, int(s * scale)) for s in steps]
            max_s = max(0, n_feat - 1 - sum(steps))
            sf = rng.randint(0, max_s)
            indices = [sf]
            for s in steps:
                indices.append(min(indices[-1] + s, n_feat - 1))
            return indices

        def _mid_up_down():
            k = rng.randint(1, N - 2)
            step = std_step
            if k * step >= n_feat:
                k = max(1, (n_feat - 1) // step - 1)
            max_start = max(0, n_feat - 1 - k * step)
            start = rng.randint(0, max_start)
            indices = [min(start + i * step, n_feat - 1) for i in range(k + 1)]
            peak = indices[k]
            for i in range(1, N - k):
                indices.append(max(0, peak - i * step))
            return indices

        def _mid_down_up():
            k = rng.randint(1, N - 2)
            step = std_step
            max_trough = max(0, n_feat - 1 - (N - 1 - k) * step)
            trough = rng.randint(0, max_trough)
            indices = [trough + (k - i) * step for i in range(k + 1)]
            for i in range(1, N - k):
                indices.append(trough + i * step)
            return [min(max(0, idx), n_feat - 1) for idx in indices]

        windows = []
        # 2 windows each of 8 types = 16 total
        for _ in range(2):
            windows.append((_fwd_std(), False))
        for _ in range(2):
            windows.append((_fwd_var(), False))
        for _ in range(2):
            windows.append((_fwd_nonuniform(), False))
        for _ in range(2):
            windows.append((_fwd_std()[::-1], False))
        for _ in range(2):
            windows.append((_fwd_var()[::-1], False))
        for _ in range(2):
            windows.append((_fwd_nonuniform()[::-1], False))
        for _ in range(2):
            windows.append((_mid_up_down(), True))
        for _ in range(2):
            windows.append((_mid_down_up(), True))

        return windows


class ARSampler(TrajectorySampler):
    """
    Sampler backed by the AR(1) curve parameterization.

    Path length budget is sampled uniformly in [path_min_feat, path_max_feat]
    (absolute feature frames), then clipped to the actual episode length.

    The default path range is derived from ``center_stride_sec`` (the desired
    centre of the per-step stride distribution, in seconds) and
    ``half_range_sec`` (half-width of that uniform band).  With
    ``fps=30, feature_stride=3`` one feature step = 0.1 s, so:

        centre path = centre_stride_sec / (feature_stride / fps) * (N - 1)

    You can still override ``path_min_feat`` / ``path_max_feat`` directly;
    the seconds-based parameters are ignored when explicit bounds are given.

    Speed variation is an AR(1) process in log-space. Reversals are applied
    to the direction signs before placement, so the bounding box naturally
    shrinks with more reversals. A 50% full-sequence flip is applied on top,
    giving both forward and backward overall trajectories.

    Returns integer feature indices clamped to [0, n_feat - 1].
    """

    def __init__(
        self,
        window_size: int = 20,
        alpha: float = 0.5,
        sigma_inf: float = math.log(2),
        path_min_feat: int | None = None,
        path_max_feat: int | None = None,
        center_stride_sec: float = 1.5,
        half_range_sec: float = 1.0,
        fps: int = 30,
        feature_stride: int = 3,
        lambda_reversals: float = 1.0,
        p_full_flip: float = 0.5,
        iid_speed: bool = False,
    ):
        self.N = window_size
        self.alpha = alpha
        self.sigma_inf = sigma_inf
        self.lambda_reversals = lambda_reversals
        self.p_full_flip = p_full_flip
        # Ablation: when True, the speed process is sampled i.i.d. log-normal
        # with the same marginal (sigma_inf) instead of AR(1). Reversal signs
        # and path-length budget are unchanged. Used for the "AR(1) vs IID"
        # reward-model training ablation.
        self.iid_speed = iid_speed

        if path_min_feat is not None and path_max_feat is not None:
            self.path_min_feat = path_min_feat
            self.path_max_feat = path_max_feat
        else:
            feat_per_sec = fps / feature_stride
            n_steps = window_size - 1
            center_path = center_stride_sec * feat_per_sec * n_steps
            half_path = half_range_sec * feat_per_sec * n_steps
            self.path_min_feat = max(n_steps, int(round(center_path - half_path)))
            self.path_max_feat = max(n_steps, int(round(center_path + half_path)))

    def sample_indices(self, n_feat: int, mode: Optional[SamplingMode] = None) -> list[int]:
        from .curve_parameterizations import (
            sample_speed_process,
            sample_speed_process_iid,
            sample_reversal_signs,
        )
        rng = np.random.default_rng()

        path_length = float(rng.integers(self.path_min_feat, self.path_max_feat + 1))
        path_length = min(path_length, float(n_feat - 1))

        if self.iid_speed:
            speeds = sample_speed_process_iid(self.N, self.sigma_inf, rng)
        else:
            speeds = sample_speed_process(self.N, self.alpha, self.sigma_inf, rng)
        signs = sample_reversal_signs(self.N, self.lambda_reversals, rng)

        delta = path_length / speeds.sum()
        steps = signs * delta * speeds

        rel = np.concatenate([[0.0], np.cumsum(steps)])
        rel_min, rel_max = rel.min(), rel.max()
        bbox = rel_max - rel_min

        max_p0 = max((n_feat - 1) - bbox, 0.0)
        p0 = rng.uniform(0.0, max_p0) - rel_min
        positions = p0 + rel

        if rng.random() < self.p_full_flip:
            positions = positions[::-1]

        return [int(np.clip(round(p), 0, n_feat - 1)) for p in positions]


class TrueARSampler(TrajectorySampler):
    """
    Pure AR(1) sampler — no path-length budget, no Σ|steps| rescaling.

    The AR(1) speed `s` is interpreted directly as a playback-rate multiple
    of one *standard* stride (``standard_feat_steps`` features). With the
    project defaults (fps=30, feature_stride=3, source_standard_stride=45)
    one standard stride = 15 features = 1.5 s, so s=1.0 means 1× playback
    and s=2.0 means 2× (frames delivered twice as fast).

        steps = signs * standard_feat_steps * speeds

    Reversal signs from a Poisson process are applied to the steps; the
    cumulative trajectory's bounding box determines a uniform starting
    frame, then with probability ``p_full_flip`` the whole window is
    reversed in time.

    The bbox is scaled to fit the episode only when an unusually wild
    draw exceeds n_feat-1 (rare for default settings, but a safeguard
    for short episodes / heavy-tailed sigma_inf).
    """

    def __init__(
        self,
        window_size: int = 20,
        feature_stride: int = 3,
        source_standard_stride: int = 45,
        alpha: float = 0.5,
        sigma_inf: float = math.log(2),
        lambda_reversals: float = 1.0,
        p_full_flip: float = 0.5,
    ):
        self.N = window_size
        self.standard_feat_steps = max(1, source_standard_stride // feature_stride)
        self.alpha = alpha
        self.sigma_inf = sigma_inf
        self.lambda_reversals = lambda_reversals
        self.p_full_flip = p_full_flip

    def sample_indices(self, n_feat: int, mode: Optional[SamplingMode] = None) -> list[int]:
        from .curve_parameterizations import (
            sample_speed_process,
            sample_reversal_signs,
        )
        rng = np.random.default_rng()

        speeds = sample_speed_process(self.N, self.alpha, self.sigma_inf, rng)
        signs = sample_reversal_signs(self.N, self.lambda_reversals, rng)
        steps = signs * float(self.standard_feat_steps) * speeds

        rel = np.concatenate([[0.0], np.cumsum(steps)])
        rel_min, rel_max = rel.min(), rel.max()
        bbox = rel_max - rel_min

        if bbox > (n_feat - 1) and bbox > 0:
            scale = (n_feat - 1) / bbox
            rel = rel * scale
            rel_min, rel_max = rel.min(), rel.max()
            bbox = rel_max - rel_min

        max_p0 = max((n_feat - 1) - bbox, 0.0)
        p0 = rng.uniform(0.0, max_p0) - rel_min
        positions = p0 + rel

        if rng.random() < self.p_full_flip:
            positions = positions[::-1]

        return [int(np.clip(round(p), 0, n_feat - 1)) for p in positions]
