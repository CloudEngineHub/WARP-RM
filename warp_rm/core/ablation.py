"""
Ablation configuration for systematic feature evaluation.

Each flag controls one innovation from the WARP-RM ablations.
The full model has all flags True; the WARP-RM baseline has all flags False.

Active innovations ( — NEW BEST composite=4.057):
  - C51 distributional head (30 bins, [-3, 3])
  - Temporal diffs (concat [features, features_t - features_{t-1}])
  - Stochastic depth (p=0.1)
  - Absolute progress auxiliary head (50 bins, [0, 1], weight=0.5)

SSL (mask prediction, next-frame prediction) was disabled in 
(ALPHA_MASK=0, BETA_NEXT_FRAME=0) and is NOT included.
"""

from dataclasses import dataclass


@dataclass
class AblationConfig:
    """Feature flags for ablation experiments."""

    # C51 distributional head (vs original Huber regression)
    use_c51: bool = True

    # Temporal diffs: concat [f_t, f_t - f_{t-1}] → 2x input dim
    use_temporal_diffs: bool = True

    # Stochastic depth
    use_stochastic_depth: bool = True
    stochastic_depth_p: float = 0.1

    # Absolute progress auxiliary head
    use_abs_progress: bool = True
    abs_progress_weight: float = 0.5

    @property
    def name(self) -> str:
        """Generate a descriptive name for this ablation config."""
        if self.is_baseline:
            return "baseline"
        if self.is_full:
            return "full"

        disabled = []
        if not self.use_c51:
            disabled.append("c51")
        if not self.use_temporal_diffs:
            disabled.append("temporal_diffs")
        if not self.use_stochastic_depth:
            disabled.append("stoch_depth")
        if not self.use_abs_progress:
            disabled.append("abs")

        if disabled:
            return "no_" + "_".join(disabled)
        return "full"

    @property
    def is_baseline(self) -> bool:
        """True if this is the plain baseline (no innovations)."""
        return not any([
            self.use_c51,
            self.use_temporal_diffs,
            self.use_stochastic_depth,
            self.use_abs_progress,
        ])

    @property
    def is_full(self) -> bool:
        """True if all innovations are enabled."""
        return all([
            self.use_c51,
            self.use_temporal_diffs,
            self.use_stochastic_depth,
            self.use_abs_progress,
        ])


def get_ablation_configs() -> dict[str, AblationConfig]:
    """Return all ablation experiment configurations.

    Keys are experiment names, values are configs. Includes:
      - baseline: original WARP-RM (no innovations)
      - full: all innovations enabled (matches )
      - no_X: full model with feature X disabled
    """
    configs = {}

    # Original WARP-RM baseline
    configs["baseline"] = AblationConfig(
        use_c51=False,
        use_temporal_diffs=False,
        use_stochastic_depth=False,
        use_abs_progress=False,
    )

    # Full model — matches  config
    configs["full"] = AblationConfig()

    # Single-feature ablations (disable one at a time)
    configs["no_c51"] = AblationConfig(use_c51=False)
    configs["no_temporal_diffs"] = AblationConfig(use_temporal_diffs=False)
    configs["no_stoch_depth"] = AblationConfig(use_stochastic_depth=False)
    configs["no_abs"] = AblationConfig(use_abs_progress=False)

    # Minimal stack: only the dominant component (C51)
    configs["c51_only"] = AblationConfig(
        use_c51=True, use_temporal_diffs=False,
        use_stochastic_depth=False, use_abs_progress=False,
    )
    # C51 + the next-biggest contributor (temporal_diffs)
    configs["c51_td_only"] = AblationConfig(
        use_c51=True, use_temporal_diffs=True,
        use_stochastic_depth=False, use_abs_progress=False,
    )
    # C51 + trained abs head (no td, no stoch). Enables online Q tracking
    # with direct abs predictions (for M, C, A sub-scores) at minimal cost
    # vs pure c51_only.
    configs["c51_abs_only"] = AblationConfig(
        use_c51=True, use_temporal_diffs=False,
        use_stochastic_depth=False, use_abs_progress=True,
    )
    # Drop only the marginal abs_progress — candidate production-lite
    configs["no_abs_no_stoch"] = AblationConfig(
        use_c51=True, use_temporal_diffs=True,
        use_stochastic_depth=False, use_abs_progress=False,
    )
    # Stack the no_abs winner with no_temporal_diffs (next-best ablation
    # on the singlefold grid; both individually scored ~3.78-3.83).
    configs["no_abs_no_td"] = AblationConfig(
        use_c51=True, use_temporal_diffs=False,
        use_stochastic_depth=True, use_abs_progress=False,
    )

    return configs
