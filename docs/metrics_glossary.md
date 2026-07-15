# Metrics Glossary

Reference for every metric that appears in WARP-RM training/validation logs. All implementations: `warp_rm/core/metrics.py`.

## Log-line shorthand (the `val_loss: … | d[…] | c[…] | corr: … | cum_sp: …` line)

Emitted every eval step via `EvalMetrics.print_summary()` at `metrics.py:89`.

| Shorthand | Full name | Field in result.json |
|---|---|---|
| `val_loss` | validation loss (mean per window) | `metrics.val_loss` |
| `vel_spearman` | velocity Spearman (see below) | `metrics.val_vel_spearman` |
| `d[all: ...]` | velocity sign accuracy — **all** windows | `metrics.val_vel_sign_acc_all` |
| `d[fwd: ...]` | velocity sign accuracy — forward-moving segments only | `metrics.val_fwd_sign` |
| `d[rew: ...]` | velocity sign accuracy — rewind segments only | `metrics.val_rew_sign` |
| `d[mid: ...]` | velocity sign accuracy — mid-rewind windows only | `metrics.val_mid_rewind_sign` |
| `c[all: ...]` | **cumulative** sign accuracy — all frames | `metrics.val_cum_sign` |
| `c[+: ...]` | cumulative sign accuracy — frames where true cum-progress > 0 | `metrics.val_cum_pos_sign` |
| `c[-: ...]` | cumulative sign accuracy — frames where true cum-progress < 0 | `metrics.val_cum_neg_sign` |
| `corr:` | velocity **Pearson** correlation | `metrics.val_vel_corr` |
| `cum_sp:` | dense-scan cumulative Spearman | `metrics.val_spearman` |
| `mag_ratio:` | velocity magnitude ratio (pred/label) | `metrics.val_vel_magnitude_ratio` |
| `(p= / l= )` | raw mean abs of pred-diffs / label-diffs | `metrics.val_pred_magnitude_mean` / `val_label_magnitude_mean` |
| `calib:` | calibration score (composite of corr + mag_ratio) | `metrics.val_calibration_score` |

Note the `d[…]` bracket = **velocity-diff** (per-step), `c[…]` = **cumulative** (per-frame abs-progress). Easy to mix up.

---

## Composite score

```
composite = val_vel_spearman + val_vel_sign_acc_all + val_cum_sign + val_spearman
```

- Sum of 4 sign/rank metrics (so the max is ~4.0).
- **Magnitude-agnostic by design** — magnitude calibration is reported as a peer metric, not baked in.
- Reason: magnitudes can drift (during data/loss changes) without breaking ranking. Keeping composite sign/rank-only means the leaderboard isn't silently invalidated by magnitude shifts.
- Used for `best_metrics` selection (which ckpt gets saved).

Note: `metadata.json` in each sidecar also has a top-level `"composite"` field that is NOT this. It's `val_spearman` (dense-scan rank corr), stored as a single SOTA-proxy number for annotation runs. Don't confuse the two.

---

## Primary metrics (in composite)

### `val_vel_spearman`
- **Velocity Spearman correlation.** Rank-order correlation between predicted per-step velocity `preds[j+1] − preds[j]` and label per-step velocity `labels[j+1] − labels[j]`, across all eval windows.
- Range: `[-1, 1]`; 1.0 = perfect rank.
- Measures "does the model rank timesteps' velocities the same way the labels do?" — scale-invariant.

### `val_vel_sign_acc_all`
- **Velocity sign accuracy** over all windows. Fraction of non-zero-label steps where `sign(pred_diff) == sign(label_diff)`.
- Range: `[0, 1]`; 0.5 = chance.
- Companion: `val_fwd_sign` / `val_rew_sign` — same metric split by label direction (forward vs rewind).
- `val_mid_rewind_sign` — same metric but only on **mid-rewind** windows (the trickiest case where a trajectory reverses mid-window).

### `val_cum_sign`
- **Cumulative sign accuracy.** For each frame where `|label_cum| > 1e-6`, check `sign(pred_cum) == sign(label_cum)`.
- Range: `[0, 1]`.
- `val_cum_pos_sign` / `val_cum_neg_sign` — split by label-cum-sign (positive vs negative cum-progress).
- Distinguishes "model knows we're currently past-zero vs pre-zero in cumulative progress" — catches drift-to-one-side errors.

### `val_spearman` (aka `cum_sp` in logs)
- **Dense-scan cumulative Spearman.** Spearman rank correlation between dense-inference abs_progress and ground-truth abs_progress across a scan of the whole episode.
- Different from `val_vel_spearman` in two ways:
  1. Operates on **cumulative** progress, not velocity diffs.
  2. Uses **dense scan** (many overlapping windows) — closer to how the model is used in deployment.
- Range: `[-1, 1]`.
- This is what `metadata.json["composite"]` reports in annotation sidecars.
- Defaults to 0.0 in `EvalMetrics.compute()`; filled in by the evaluator after the dense scan.

---

## Magnitude scorecard (NOT in composite — reported as peer metrics)

### `val_vel_corr` (aka `corr` in logs)
- **Velocity Pearson correlation** (magnitude-sensitive companion of `vel_spearman`).
- Formula: `np.corrcoef(pred_diffs, label_diffs)[0,1]`.
- Where `vel_spearman` answers "rank OK?", `vel_corr` answers "shape including scale OK?" — if predictions are a constant times labels, Pearson is still 1.0, but if they're non-linearly mapped, it drops.

### `val_vel_magnitude_ratio` (aka `mag_ratio` in logs)
- `mean(|pred_diff|) / mean(|label_diff|)` — bulk scale of predicted per-step velocity vs label per-step velocity.
- Range: `[0, ∞]`; 1.0 = perfectly calibrated; `< 1` = under-scaled (conservative predictions); `> 1` = over-scaled (overshooting).
- Old checkpoints without this metric report `0.0` (backward-compat sentinel).

### `val_pred_magnitude_mean` / `val_label_magnitude_mean`
- The raw numerator / denominator of `val_vel_magnitude_ratio`, kept separately for diagnosis (lets you see whether labels got smaller or predictions got smaller).

### `val_calibration_score` (aka `calib` in logs)
- **Magnitude-calibration summary in [0, 1]**. Combines shape (`vel_corr`) and scale (`mag_ratio`):
  ```
  calib = max(0, vel_corr) × exp(-|log(vel_magnitude_ratio)|)
  ```
- Penalty is log-symmetric: `mag_ratio=0.5` or `2.0` → 0.5 credit; `mag_ratio=0.1` or `10.0` → 0.1 credit; `mag_ratio=1.0` → 1.0 credit.
- 1.0 = perfectly calibrated in both shape and scale. 0.0 = either flat Pearson or extreme magnitude drift.
- **NOT part of composite**; used as a peer indicator when picking between runs with similar composites.

---

## Structural properties (NOT in composite)

### `val_speed_invariance`
- Dense scan at `std_step` vs `2 × std_step`, Pearson over intersection of covered frames.
- 1.0 = predictions depend only on video content (tempo-invariant); `< 1.0` = predictions shift with stride.
- This is a **deploy property** — the model should output the same abs-progress for the same frames regardless of how fast the video is fed in.
- Old checkpoints default to `0.0`.

---

## Deprecated metrics (not part of the current interface)

Earlier development used three further metric families that are **deprecated and
not part of the public interface**: human-annotated negative-segment precision/recall,
pairwise segment-vs-segment agreement (ELO), and the per-episode quality index
`Q` (and its `P`/`S`/`E`/`M`/`F` sub-scores). None of them ever contributed to the
composite. WARP-RM's signal is the dense per-frame velocity; `score_episodes.py`
summarizes it per episode as `mean_velocity` (pace) and `delta_v_std` (smoothness).

---

## Where each metric lives

| Artifact | What's in it |
|---|---|
| train log | Per-step `Step N | loss:` lines + every-eval full `val_loss: … | d[…] | c[…]` summary via `EvalMetrics.print_summary()` |
| checkpoint `.pt` meta | `{step, val_loss, val_vel_spearman, val_spearman, val_cum_sign, val_calibration_score, …, feature_stride, standard_stride_src, sampler, attention}` (stamped by the trainer; auto-detected at load) |
| `score_episodes.py` TSV | Per-episode: `episode | n_frames | n_feat | mean_velocity | delta_v_std` |
