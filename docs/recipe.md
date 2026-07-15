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
