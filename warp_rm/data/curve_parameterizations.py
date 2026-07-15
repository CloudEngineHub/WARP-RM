"""
Trajectory augmentation sampling via composable parameterizations.

Three independent components compose to produce augmented trajectory windows:
  1. Path length budget (γ): log-uniform fraction of trajectory as total distance traveled
  2. Speed process: AR(1) in log-space (discrete Ornstein-Uhlenbeck)
  3. Reversal structure: multi-reversal via Poisson-sampled changepoints

γ controls total path length (Σ|steps| = γ*T), not net displacement. With reversals
the trajectory folds back on itself, so net displacement < γ*T. The starting position
is chosen so the bounding box of the resulting path fits within [0, T].
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class AugmentedTrajectory:
    positions: np.ndarray          # shape (N,), float positions in [0, T]
    per_state_weights: np.ndarray  # shape (N,), importance weights for uniform coverage
    window_weight: float           # scalar harmonic-mean window weight
    gamma: float                   # path length budget as fraction of T
    path_length: float             # total distance traveled = gamma * T
    bbox_span: float               # net extent of bounding box (max - min positions)


def sample_window_scale(
    gamma_min: float = 0.05,
    gamma_max: float = 0.5,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample γ log-uniformly from [gamma_min, gamma_max]."""
    if rng is None:
        rng = np.random.default_rng()
    return math.exp(rng.uniform(math.log(gamma_min), math.log(gamma_max)))


def sample_speed_process(
    N: int,
    alpha: float = 0.9,
    sigma_inf: float = math.log(2),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample N-1 inter-point speeds via AR(1) in log-space (discrete OU process).

    Returns exp(z) where z is the stationary AR(1) sequence centered at 0.
    The returned speeds are unnormalized positive scalars; caller scales them
    to match the desired path length budget.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_gaps = N - 1
    z = np.empty(n_gaps)
    z[0] = rng.normal(0.0, sigma_inf)
    noise_scale = math.sqrt(1 - alpha**2) * sigma_inf
    for k in range(1, n_gaps):
        z[k] = alpha * z[k - 1] + noise_scale * rng.normal()
    return np.exp(z)


def sample_speed_process_iid(
    N: int,
    sigma_inf: float = math.log(2),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """IID baseline for ``sample_speed_process``.

    Samples z_k ~ N(0, sigma_inf) i.i.d. and returns exp(z). The MARGINAL
    stationary distribution matches the AR(1) version exactly (log-normal,
    geometric mean 1.0); only the temporal autocorrelation is gone — at the
    AR(1) defaults of alpha=0.5 the lag-1 correlation in log-speed drops
    from 0.5 to 0.0, leaving the noise envelope per step identical.

    Used as the ablation sampler for "AR(1) vs IID" comparisons (the speed
    process is the ONLY thing that differs from the AR(1) sampler — reversal
    signs and path-length budget are reused unchanged downstream).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_gaps = N - 1
    z = rng.normal(0.0, sigma_inf, size=n_gaps)
    return np.exp(z)


def sample_reversal_signs(
    N: int,
    lambda_reversals: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample N-1 direction signs (+1 / -1) encoding reversal structure.

    Samples R ~ Poisson(lambda_reversals) changepoints, divides the N-1 gaps
    into R+1 segments, and alternates direction between segments. The first
    segment is always +1 (forward).

    Returns an array of shape (N-1,) with values in {+1, -1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_gaps = N - 1
    signs = np.ones(n_gaps, dtype=float)
    R = rng.poisson(lambda_reversals)
    if R == 0:
        return signs

    cp_pool = np.arange(1, n_gaps)  # changepoints live between gaps
    R = min(R, len(cp_pool))
    changepoints = np.sort(rng.choice(cp_pool, size=R, replace=False))

    boundaries = np.concatenate([[0], changepoints, [n_gaps]])
    for seg_idx in range(1, len(boundaries) - 1):
        lo, hi = boundaries[seg_idx], boundaries[seg_idx + 1]
        if seg_idx % 2 == 1:
            signs[lo:hi] = -1.0
    return signs


def compute_importance_weights(
    positions: np.ndarray,
    bbox_span: float,
    T: float,
) -> tuple[np.ndarray, float]:
    """
    Compute per-state importance weights for uniform trajectory coverage.

    Uses the bounding box span as the effective window size S for coverage
    calculations. coverage(x, S) = min(x, S, T-x) / (T-S).
    """
    S = bbox_span
    denom = max(T - S, 1e-6)
    coverage = np.minimum(np.minimum(positions, S), T - positions) / denom
    coverage = np.clip(coverage, 1e-9, None)
    per_state_weights = 1.0 / coverage
    window_weight = float(len(positions)) / float(coverage.sum())
    return per_state_weights, window_weight


def sample_augmented_trajectory(
    T: float,
    N: int,
    alpha: float = 0.9,
    sigma_inf: float = math.log(2),
    gamma_min: float = 0.05,
    gamma_max: float = 0.5,
    lambda_reversals: float = 1.0,
    rng: np.random.Generator | None = None,
) -> AugmentedTrajectory:
    """
    Full augmentation pipeline: path budget → speed process → reversals
    → placement → importance weights.

    Args:
        T: episode length (positions live in [0, T])
        N: number of sampled points
        alpha: AR(1) autocorrelation coefficient
        sigma_inf: stationary std of log-speed process (ln2 ≈ 0.693 makes 2× a 1-sigma event)
        gamma_min: minimum path length budget as fraction of T
        gamma_max: maximum path length budget as fraction of T
        lambda_reversals: Poisson rate for reversal changepoints (0 = always forward)

    Returns:
        AugmentedTrajectory with positions in [0, T], importance weights, and metadata.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Path length budget
    gamma = sample_window_scale(gamma_min, gamma_max, rng)
    path_length = gamma * T

    # 2. Speed process — unnormalized positive step sizes
    speeds = sample_speed_process(N, alpha, sigma_inf, rng)

    # 3. Reversal signs — determined before placement
    signs = sample_reversal_signs(N, lambda_reversals, rng)

    # 4. Scale speeds so total path length = γ*T, then apply signs
    delta = path_length / speeds.sum()
    steps = signs * delta * speeds  # signed displacements

    # 5. Build relative path from origin, then shift to fit in [0, T]
    rel = np.concatenate([[0.0], np.cumsum(steps)])
    rel_min, rel_max = rel.min(), rel.max()
    bbox_span = rel_max - rel_min

    # Sample starting position so bounding box fits within [0, T]
    max_p0 = max(T - bbox_span, 0.0)
    p0 = rng.uniform(0.0, max_p0) - rel_min  # shift so min of path lands at p0_base
    positions = p0 + rel

    # 6. Importance weights based on bounding box coverage
    per_state_weights, window_weight = compute_importance_weights(positions, bbox_span, T)

    return AugmentedTrajectory(
        positions=positions,
        per_state_weights=per_state_weights,
        window_weight=window_weight,
        gamma=gamma,
        path_length=path_length,
        bbox_span=bbox_span,
    )
