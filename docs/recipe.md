# WARP-RM recipe — full data → supervision walkthrough

This is the configuration the defaults encode: `python scripts/train.py
--lerobot-repo <ds>` reproduces it. The paper checkpoint adds the
absolute-progress head (`--ablation full`); the recommended default drops it
(`--ablation no_abs`), which was more robust in downstream policy eval.

## 1. Data corpus & filtering

- **Source**: a LeRobot dataset of *successful* demonstrations of one task. The
  paper trains on the 1,950 shortest T-shirt-folding demos (`≤59.8 s`) of a
  larger d405 corpus; that dataset is not yet public, so bring your own (see
  [`dataset_schema.md`](dataset_schema.md)).
- **Filter** (`--shortest-frac 0.25`): keep the shortest 25% of episodes by
  `n_frames`. Shorter demos are smoother, more consistent-tempo, less hesitant —
  a cleaner reference for "expert pace" (v̂ ≈ 1).
  > **This assumes every episode is a *complete* demo.** The filter selects the
  > shortest episodes, so any truncated or aborted recording is exactly what it
  > keeps — a 2-second fragment survives while a good 45-second demo is dropped,
  > and `--clean-val` then draws the validation set from that same junk tail.
  > On corpora with thousands of episodes the effect is diluted; on a few hundred
  > it dominates. Check the low tail before trusting the default, and set
  > `--min-frames` to cut it. A principled floor is one standard-stride window,
  > `(window_size - 1) · SSS + 2 · feature_stride` (471 frames at N=32, SSS=15,
  > fs=3) — below that the sampler cannot lay down a standard-pace path.
- **Explicit pruning** (`--exclude-episode-index`): drop specific episodes by
  index — e.g. demos where a human reaches into frame, which teach the model that
  a hand is part of task progress.
- **Train/val split** (`--clean-val`, default): `N_VAL = 20` validation episodes
  drawn from the shortest 15% of the kept pool, so val is the cleanest subset and
  the composite metric stays comparable across runs.

## 2. Backbone features (precomputed, frozen)

- **Encoder**: DINOv3 ViT-B/16, frozen → 768-d per frame.
- **Feature stride** (`feature_stride=3`): one feature every 3 source frames.
- **Cache**: `~/.cache/warp_rm/features/dinov3_fs3/<dataset>/<key>.npy`
  (override root with `WARP_RM_FEATURE_CACHE`). The stride is stamped into the
  checkpoint, so scoring/inference auto-resolve it — never assume the global default.

## 3. Time-warp sampling — `ARSampler`

Per step the sampler emits `window_size = 32` integer feature indices
(`warp_rm/data/samplers.py:ARSampler`):

1. **Path budget**: total path length ~ `Uniform([⅓L, 5⁄3 L])` where
   `L = (N-1)·S`, `S = 1.5 s` (`ar_center_stride_sec=1.5`, `ar_half_range_sec=1.0`).
2. **Per-step speed**: AR(1) log-speed, `log sₜ = α·log sₜ₋₁ + ε`, `α=0.5`,
   stationary std `σ∞ = ln 2` → smooth speed variation (slow-mo ↔ fast-forward).
3. **Reversals**: number of in-window sign flips ~ `Poisson(λ=1.0)`.
4. **Placement**: offsets cumsum'd; the start index is sampled so all indices
   land in `[0, n_feat-1]`.
5. **Full-window flip** with `p=0.5` → equal forward/backward coverage.

`--sampler iid` swaps the AR(1) speed process for an i.i.d. log-normal draw with
the same marginal (the paper's IID ablation). Sampling never affects inference.

## 4. Labels — `RelativeCumulativeLabeler`

```
label[0] = 0
label[j] = label[j-1] + (idx[j] - idx[j-1]) · feature_stride
                        / ((N-1) · SOURCE_STANDARD_STRIDE)
```
With `SOURCE_STANDARD_STRIDE = 45`, `N = 32`: standard-pace forward → `[0,1]`,
2× pace → `[0,2]`, reversed → `[-1,0]`, mid-rewind → non-monotonic / negative.
These are cumulative signed progress values relative to the window's first frame.

### 4a. Changing `--source-standard-stride` — the sampler follows it

**`--source-standard-stride` (SSS) and `--ar-center-stride-sec` are two halves of
one calibration, and the code now couples them** (`derive_ar_budget`,
`warp_rm/data/samplers.py`). SSS sets the label denominator (§4); the AR
sampler's centre stride sets how far a window actually travels (§3). Changing
one without the other rescales every label by `45 / SSS_new`, so SSS is the
single knob and the sampler derives from it.

`--ar-center-stride-sec` and `--ar-half-range-sec` default to `-1`, meaning
*derive*. Pass a number to either one to override it; every run prints a
`[path-budget]` line reporting the realized band and whether each value was
derived or explicit.

The invariant that keeps "standard pace ⇒ velocity 1.0":

```
label(window) = path_src / ((N-1) · SSS)      # §4, summed over the window
path_src      = center_stride_sec · fps · (N-1)   # §3, ARSampler.__init__

⇒ centred at 1.0  ⟺  center_stride_sec = SSS / fps
```

Default: `45 / 30 = 1.5 s` ✓ — the value that used to be hardcoded, so adopting
the derivation changes nothing at the default SSS. The half-range derives as
`⅔ × centre` (the ratio these defaults have always had, `1.0 / 1.5`), which keeps
the *relative* speed band invariant to SSS. The table below is what the code now
produces, not a list to apply by hand:

| SSS | `--ar-center-stride-sec` | `--ar-half-range-sec` |
|---|---|---|
| 45 (default) | 1.5 | 1.0 |
| 25 | 0.8333 | 0.5556 |
| 15 | 0.5 | 0.3333 |

Measured C51 target distribution (6k sampled windows, 32-frame window, fs=3) —
labels are clamped to the head's `[-3, 3]` support at `warp_rm/core/loss.py:56`,
so overflow is silently destroyed, not merely rare:

| config | mean \|label\| | p99.9 | tokens clamped |
|---|---|---|---|
| SSS=45, centre 1.5 s (reference) | 0.33 | 1.20 | 0.00% |
| SSS=15, sampler **left at 1.5 s** | 0.86 | 3.81 | **1.16%** (8.5% of windows) |
| SSS=15, centre 0.5 s | 0.37 | 1.59 | 0.00% |
| SSS=25, centre 0.8333 s | 0.36 | 1.57 | 0.00% |

Recalibrated, the distribution lands on top of the reference. Left alone, the
fast tail is clipped and the model learns a compressed velocity scale.

Two further traps when picking SSS:

- **Make `feature_stride` divide SSS.** `standard_feat_steps = SSS // fs` is
  integer division, and it — not SSS — is what dense inference strides by. SSS=25
  with fs=3 truncates `8.33 → 8`, so inference measures velocity against 24
  source frames while the labels were normalised by 25: a constant ~4% scale
  error. SSS=45/fs=3 and SSS=15/fs=3 are exact. (The paper's sim config,
  `--feature-stride 1 --source-standard-stride 15`, is exact for the same
  reason.)
- **Check SSS against your episode lengths.** A standard-stride window spans
  `(N-1)·SSS` source frames — 1395 (46.5 s) at the default. On episodes shorter
  than that, `dense_inference_*` shrinks the stride and applies
  `vel_scale = standard_feat_steps / std_step` to compensate
  (`warp_rm/visualization/inference.py`). That rescale is *episode-length
  dependent*, so velocities stop being directly comparable across episodes.
  A dataset whose median episode is 43 s should use SSS≈15 (15.5 s window), not
  the 45 default, which would push every episode onto the fallback path.

## 5. Architecture — `TransformerAggregator`

Input `(B, 32, 768)`:
1. **Temporal-diff tokens**: concat `[fₜ, fₜ − fₜ₋₁]` → `(B, 32, 1536)` (`f₀` diff = 0).
2. Linear `1536 → 768`.
3. Sinusoidal positional encoding (max_seq_len 32).
4. **Bidirectional** Transformer encoder: 12 layers, 8 heads, ffn 3072,
   dropout 0.15 (full attention — the model scores already-observed trajectories).
5. **Stochastic depth** `p=0.1` during training.

Heads: a **relative-progress C51** head (30 bins over `[-3, 3]`, prediction =
softmax-weighted bin centers — the primary continuous signal) and, in
`--ablation full`, an **absolute-progress C51** auxiliary head (50 bins over
`[0,1]`). `--ablation no_abs` (the default) keeps only the relative head plus
temporal diffs + stochastic depth.

## 6. Supervision — `RMLoss`

Categorical cross-entropy between the two-hot target distribution (label
projected onto bin centers) and the predicted softmax, on the relative head
(plus the absolute head under `full`). `warp_rm/core/loss.py`.

## 7. Training schedule

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-3, grad clip 1.0 |
| Batch size | 1024 (≥40 GB GPU; `--batch-size 256` fallback) |
| LR | 4e-4 (linear-scaled from 1e-4 @ bs256), 1000-step warmup, cosine |
| Steps | 15,000 (`--max-steps`) |
| Eval every | 200 steps |

Best validation typically lands early; the paper checkpoint's best-val step is **2800**.

## 8. Inference / scoring

`dense_inference_*` slides the trained window (N=32) across each episode and
averages overlapping per-token predictions into a dense per-frame signal:

- `warp_rm_signed_magnitude[t]` — per-frame velocity (forward +, regression −) — **the signal**
- `warp_rm_progress[t]` — velocity-integrated absolute progress in `[0,1]`

This dense per-frame velocity is the model's output and what **WARP-BC** consumes
for action-chunk reweighting (`write_warp_rm_annotations.py` injects it as the
`warp_rm_signed_magnitude` column). `score_episodes.py` reports a per-episode
summary of it (`mean_velocity` = pace, `delta_v_std` = smoothness) for quick
ranking. (Episode-wide quality indices and human-annotation / negative-segment
eval were used in development but are deprecated and outside the current public interface.)
