"""
Evaluation metrics for reward model training.

Collects step-wise predictions across eval windows and computes:
  - Velocity Spearman (primary): rank-order correlation of pred vs label diffs
  - Velocity sign accuracy (primary): directional correctness
  - Cumulative sign accuracy, forward/rewind splits, dense scan Spearman
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import spearmanr


@dataclass
class EvalMetrics:
    loss: float
    vel_spearman: float
    vel_sign_acc_all: float
    mid_rewind_sign: float
    fwd_sign: float
    rew_sign: float
    cum_sign: float
    cum_pos_sign: float
    cum_neg_sign: float
    vel_corr: float
    spearman: float
    # Speed-invariance: dense scan at std_step vs 2×std_step, pearsonr over
    # intersection of covered frames. 1.0 = predictions depend only on video
    # content (tempo-invariant); <1.0 = predictions shift with stride. This
    # is a structural deploy property — models should output the same value
    # for the same content regardless of how fast the video is fed in.
    # Default 0.0 for backward-compat with old checkpoints; not in composite.
    speed_invariance: float = 0.0

    # Magnitude-calibration scorecard (separate from composite so we don't
    # silently retroactively change the leaderboard). `composite_score`
    # remains sign/rank-only; these are reported alongside for model
    # selection. Default 0.0 = unknown (old ckpts).
    vel_magnitude_ratio: float = 0.0   # mean(|pred_diff|) / mean(|label_diff|); 1.0 = calibrated
    pred_magnitude_mean: float = 0.0   # raw mean(|pred_diff|), for diagnosis
    label_magnitude_mean: float = 0.0  # raw mean(|label_diff|), for diagnosis

    # Net-displacement calibration: mean(|pred[-1]-pred[0]|) / mean(|label[-1]-label[0]|),
    # one term per window. Per-step jitter rectifies into vel_magnitude_ratio
    # (Jensen on |·|), biasing it upward even for an unbiased model. Net
    # displacement sums diffs first so zero-mean noise cancels — a noise-
    # unbiased calibration check. Windows whose net label displacement is
    # ~0 (e.g. mid-rewinds returning to start) are excluded from both
    # numerator and denominator. Default 0.0 = unknown (old ckpts).
    vel_net_magnitude_ratio: float = 0.0
    pred_net_magnitude_mean: float = 0.0
    label_net_magnitude_mean: float = 0.0

    @property
    def composite_score(self) -> float:
        # Sign + rank aggregate. Magnitude-agnostic by design —
        # magnitude calibration surfaces via vel_magnitude_ratio / vel_corr.
        return self.vel_spearman + self.vel_sign_acc_all + self.cum_sign + self.spearman

    @property
    def calibration_score(self) -> float:
        """Magnitude-calibration summary in [0, 1]. 1.0 = perfectly calibrated.

        Combines Pearson correlation (shape of the regression) with a
        magnitude-ratio penalty that bottoms out at ratio=0 or ratio→∞.
        Not used for SOTA selection; reported as a peer metric.
        """
        import math
        r = max(0.0, self.vel_corr)  # clip negative pearson at 0
        if self.vel_magnitude_ratio <= 0.0:
            mag = 0.0
        else:
            # Penalty for being too small OR too large. log-symmetric.
            mag = math.exp(-abs(math.log(self.vel_magnitude_ratio)))
        return r * mag

    def to_dict(self) -> dict:
        return {
            "val_loss": self.loss,
            "val_vel_spearman": self.vel_spearman,
            "val_vel_sign_acc_all": self.vel_sign_acc_all,
            "val_mid_rewind_sign": self.mid_rewind_sign,
            "val_fwd_sign": self.fwd_sign,
            "val_rew_sign": self.rew_sign,
            "val_cum_sign": self.cum_sign,
            "val_cum_pos_sign": self.cum_pos_sign,
            "val_cum_neg_sign": self.cum_neg_sign,
            "val_vel_corr": self.vel_corr,
            "val_spearman": self.spearman,
            "val_speed_invariance": self.speed_invariance,
            # Magnitude scorecard (peer metrics; not in composite):
            "val_vel_magnitude_ratio": self.vel_magnitude_ratio,
            "val_pred_magnitude_mean": self.pred_magnitude_mean,
            "val_label_magnitude_mean": self.label_magnitude_mean,
            "val_vel_net_magnitude_ratio": self.vel_net_magnitude_ratio,
            "val_pred_net_magnitude_mean": self.pred_net_magnitude_mean,
            "val_label_net_magnitude_mean": self.label_net_magnitude_mean,
            "val_calibration_score": self.calibration_score,
        }

    def print_summary(self):
        print(f"  val_loss: {self.loss:.6f} | vel_spearman: {self.vel_spearman:.4f} | "
              f"d[all:{self.vel_sign_acc_all:.3f} fwd:{self.fwd_sign:.3f} "
              f"rew:{self.rew_sign:.3f} mid:{self.mid_rewind_sign:.3f}] | "
              f"c[all:{self.cum_sign:.3f} +:{self.cum_pos_sign:.3f} "
              f"-:{self.cum_neg_sign:.3f}] | "
              f"corr:{self.vel_corr:.4f} | cum_sp:{self.spearman:.4f} | "
              f"mag_ratio:{self.vel_magnitude_ratio:.2f} "
              f"(p={self.pred_magnitude_mean:.3f}/l={self.label_magnitude_mean:.3f}) "
              f"| net_mag_ratio:{self.vel_net_magnitude_ratio:.2f} "
              f"(p={self.pred_net_magnitude_mean:.3f}/l={self.label_net_magnitude_mean:.3f}) "
              f"| calib:{self.calibration_score:.3f}")


class MetricTracker:
    """Accumulates per-window predictions and computes final evaluation metrics."""

    def __init__(self):
        self.total_loss = 0.0
        self.n_batches = 0
        self.all_pred_diffs: list[float] = []
        self.all_label_diffs: list[float] = []
        self.mid_pred_diffs: list[float] = []
        self.mid_label_diffs: list[float] = []
        self.all_cum_preds: list[float] = []
        self.all_cum_labels: list[float] = []
        self.net_pred_disp: list[float] = []
        self.net_label_disp: list[float] = []

    def update(self, preds_1d: np.ndarray, labels_1d: np.ndarray,
               is_mid_rewind: bool, loss_val: float):
        """
        Update with a single eval window.

        Args:
            preds_1d: (T,) model predictions for one window
            labels_1d: (T,) ground truth labels for one window
            is_mid_rewind: whether this window is a mid-rewind type
            loss_val: scalar loss for this window
        """
        self.total_loss += loss_val
        self.n_batches += 1
        T = len(preds_1d)

        for j in range(T - 1):
            pd = float(preds_1d[j + 1] - preds_1d[j])
            ld = float(labels_1d[j + 1] - labels_1d[j])
            self.all_pred_diffs.append(pd)
            self.all_label_diffs.append(ld)
            if is_mid_rewind:
                self.mid_pred_diffs.append(pd)
                self.mid_label_diffs.append(ld)

        for j in range(1, T):
            pv = float(preds_1d[j])
            lv = float(labels_1d[j])
            if abs(lv) > 1e-6:
                self.all_cum_preds.append(pv)
                self.all_cum_labels.append(lv)

        if T >= 2:
            self.net_pred_disp.append(float(preds_1d[T - 1] - preds_1d[0]))
            self.net_label_disp.append(float(labels_1d[T - 1] - labels_1d[0]))

    def compute(self) -> EvalMetrics:
        """Compute all metrics from accumulated data."""
        val_loss = self.total_loss / max(self.n_batches, 1)

        # Velocity Spearman
        pd_arr = np.array(self.all_pred_diffs, dtype=np.float64)
        ld_arr = np.array(self.all_label_diffs, dtype=np.float64)
        sp_v, _ = spearmanr(pd_arr, ld_arr)
        val_vel_spearman = float(sp_v) if not np.isnan(sp_v) else 0.0

        # Velocity sign accuracy (all types)
        nonzero = [i for i, ld in enumerate(self.all_label_diffs) if ld != 0.0]
        if nonzero:
            correct = sum(1 for i in nonzero if self.all_pred_diffs[i] * self.all_label_diffs[i] > 0)
            val_vel_sign_acc_all = correct / len(nonzero)
        else:
            val_vel_sign_acc_all = 0.0

        # Mid-rewind sign accuracy
        mid_nonzero = [i for i, ld in enumerate(self.mid_label_diffs) if ld != 0.0]
        if mid_nonzero:
            mid_correct = sum(1 for i in mid_nonzero if self.mid_pred_diffs[i] * self.mid_label_diffs[i] > 0)
            val_mid_rewind_sign = mid_correct / len(mid_nonzero)
        else:
            val_mid_rewind_sign = 0.0

        # Forward/Rewind sign accuracy
        fwd_pds = [pd for pd, ld in zip(self.all_pred_diffs, self.all_label_diffs) if ld > 1e-6]
        rew_pds = [pd for pd, ld in zip(self.all_pred_diffs, self.all_label_diffs) if ld < -1e-6]
        val_fwd_sign = sum(1 for pd in fwd_pds if pd > 0) / max(len(fwd_pds), 1)
        val_rew_sign = sum(1 for pd in rew_pds if pd < 0) / max(len(rew_pds), 1)

        # Cumulative sign accuracy
        cum_correct = sum(1 for pv, lv in zip(self.all_cum_preds, self.all_cum_labels) if pv * lv > 0)
        val_cum_sign = cum_correct / max(len(self.all_cum_preds), 1)
        cum_pos = [(pv, lv) for pv, lv in zip(self.all_cum_preds, self.all_cum_labels) if lv > 0]
        cum_neg = [(pv, lv) for pv, lv in zip(self.all_cum_preds, self.all_cum_labels) if lv < 0]
        val_cum_pos_sign = sum(1 for pv, _ in cum_pos if pv > 0) / max(len(cum_pos), 1)
        val_cum_neg_sign = sum(1 for pv, _ in cum_neg if pv < 0) / max(len(cum_neg), 1)

        # Velocity Pearson correlation (magnitude-sensitive; not in composite)
        if len(pd_arr) > 1 and pd_arr.std() > 0 and ld_arr.std() > 0:
            val_vel_corr = float(np.corrcoef(pd_arr, ld_arr)[0, 1])
            if np.isnan(val_vel_corr):
                val_vel_corr = 0.0
        else:
            val_vel_corr = 0.0

        # Magnitude scorecard (peer metrics; not in composite)
        pred_mag = float(np.abs(pd_arr).mean()) if len(pd_arr) else 0.0
        label_mag = float(np.abs(ld_arr).mean()) if len(ld_arr) else 0.0
        mag_ratio = pred_mag / label_mag if label_mag > 1e-8 else 0.0

        # Net-displacement magnitude ratio. Restrict to windows with
        # meaningfully nonzero net label displacement so that mid-rewinds
        # returning to start (|net_label| ~ 0) don't dominate.
        npd = np.array(self.net_pred_disp, dtype=np.float64)
        nld = np.array(self.net_label_disp, dtype=np.float64)
        net_mask = np.abs(nld) > 1e-6
        if net_mask.any():
            pred_net_mag = float(np.abs(npd[net_mask]).mean())
            label_net_mag = float(np.abs(nld[net_mask]).mean())
            net_mag_ratio = pred_net_mag / label_net_mag if label_net_mag > 1e-8 else 0.0
        else:
            pred_net_mag = 0.0
            label_net_mag = 0.0
            net_mag_ratio = 0.0

        return EvalMetrics(
            loss=val_loss,
            vel_spearman=val_vel_spearman,
            vel_sign_acc_all=val_vel_sign_acc_all,
            mid_rewind_sign=val_mid_rewind_sign,
            fwd_sign=val_fwd_sign,
            rew_sign=val_rew_sign,
            cum_sign=val_cum_sign,
            cum_pos_sign=val_cum_pos_sign,
            cum_neg_sign=val_cum_neg_sign,
            vel_corr=val_vel_corr,
            spearman=0.0,  # filled in by evaluator after dense scan
            vel_magnitude_ratio=mag_ratio,
            pred_magnitude_mean=pred_mag,
            label_magnitude_mean=label_mag,
            vel_net_magnitude_ratio=net_mag_ratio,
            pred_net_magnitude_mean=pred_net_mag,
            label_net_magnitude_mean=label_net_mag,
        )
