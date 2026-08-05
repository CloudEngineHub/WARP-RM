# Reproducing the simulated bottles-in-bin results

Audit of what the public artifacts let you reproduce, end to end. Read against the
pinned companion commits:

| repo | branch | pin |
|---|---|---|
| [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi) | `paper-repro` | `91f99d6` <!-- TODO: verify after pushing paper-repro --> |
| [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi) | `release-candidate` | `204eb92dd2af37c4d1189b587d5fbff978383930` |
| [`uynitsuj/abc-rabc`](https://github.com/uynitsuj/abc-rabc) | `release-candidate` | `9a9fbb5bbf109b726b4130b18cd9826a4e262d45` |

> [!IMPORTANT]
> **Two openpi branches, two jobs.** `release-candidate` **serves** the released
> policies (`scripts/serve_paper_sim_policy.py`) and carries no trainer.
> `paper-repro` is the **training** branch: `scripts/train.py`, the RABC data path,
> and the two paper-arm configs. Earlier revisions of this document called pi0
> training "not reproducible" — that was a statement about `release-candidate`
> only.

```
 (1) train WARP-RM     (2) score + inject      (3) train pi0 w/ WARP-BC    (4) rollout + score
     this repo      ->     this repo        ->   openpi @ paper-repro   ->    abc-rabc
```

All four links are runnable from public artifacts. Fresh runs match the paper to
**tolerances, not bit-identically** — see [Limits](#limits-of-fresh-repro) at the end.

## 1–2. Reward model, scoring, injection

See the paper-sim command in the [README](../README.md#quickstart--train-a-warp-rm-on-a-lerobot-dataset).
The flags are recoverable from the published head's stamped metadata; the one
exception (`--ar-center-stride-sec`) is discussed at the bottom.

> [!WARNING]
> **The published sim dataset is 640x480, not square** — verified on the bitstream
> (`ffprobe` -> `h264 640x480`), and agreed by `meta/info.json`'s `shape` and its
> encoder-written `info` block for all three cameras. So training or scoring
> `sim-bottles-mjwarp-v1` **as published** with default flags squashes every frame
> horizontally by **1.33x**. Pass `--crop-mode center` to avoid that. See
> [`dataset_schema.md`](dataset_schema.md#frame-geometry--crop-mode).
>
> This is not cosmetic for step 3: the WARP-BC gate thresholds the raw injected
> velocity, so the crop you score under decides **which action chunks train**
> (see [§3 What must be pinned](#what-must-be-pinned)).
>
> **Open question — was the published dataset the training corpus?** The checkpoint
> carries no `crop_mode` stamp, so its training geometry is unrecorded. Two signals
> suggest the corpora may differ: the head's `branch_tag` is
> `wr_A_rm_perobj_s25_mjwarp_sss15_20k`, and the paper-arm configs reference
> `sim_put_bottles_mjwarp_rmperobj` / `..._rmsss15` — none of which is
> `sim-bottles-mjwarp-v1`. A 224x224 center-cropped conversion may have been used
> upstream (the usual mcap->LeRobot path applies `crop='min(iw,ih)':'min(iw,ih)'`
> then `scale=224:224`) without being the artifact that got published. Until this is
> resolved, treat "matches the published head" as unverified either way.

## 3. WARP-BC: build the weighted dataset, then train pi0

### The gate

`openpi/src/openpi/transforms.py::ComputeRABCWeights`, as configured by both paper
arms (`rabc_use_final_action_condition=True`, `rabc_threshold=1.00`, default
`rabc_clip_max=1.0`):

```python
# transforms.py, final-action branch
final_vel = float(vel[-1])                                  # last frame of the horizon window
w = float(np.clip(final_vel, None, clip_max)) if final_vel > thr else 0.0
```

So WARP-BC is a **binary keep/drop filter on the chunk-final velocity**: keep iff
`vel[-1] > 1.0`, otherwise weight 0. Because `clip_max == thr == 1.0`, every kept
sample gets weight exactly `1.0`, and the loss reduction in
`scripts/train.py:155-159`

```python
weighted_loss = per_sample_loss * observation.sample_weights
loss = jnp.sum(weighted_loss) / (jnp.sum(observation.sample_weights) + 1e-6)
```

is arithmetically a plain mean over kept samples. Zero-weight samples are also
dropped up front by the data loader's subset filter (`rabc_reject_zero_weighted`),
so they cost no compute. On this dataset the gate keeps **~31.5% of chunks**
(32.55% of frames have `vel > 1.0`).

> [!NOTE]
> This is *not* the soft `clip(mean(vel over horizon), 0, 1)` weighting that
> `ComputeRABCWeights` applies with its **defaults** (`threshold=None`,
> `use_final_action_condition=False`). The defaults are not the paper recipe.
> `release-candidate` never instantiates the transform at all, and nothing there
> consumes `sample_weights` — read the wiring on `paper-repro`.

### Build the weighted dataset

No extra data release is needed: `sim-bottles-mjwarp-v1` **is** the training corpus
(LeRobot-v3, 2,438 episodes, 2,228,979 frames — the frame count the arm configs
were built against). It ships without reward columns, so the only step is to score
it with the published head and inject the column.

```bash
hf download uynitsuj/sim-bottles-mjwarp-v1 --repo-type dataset \
  --local-dir sim-bottles-mjwarp-v1
hf download uynitsuj/warp-rm-sim-bottles-sss15 --local-dir warp-rm-sss15

# inject per-frame warp_rm_signed_magnitude (+ warp_rm_progress) in place
python scripts/data/write_warp_rm_annotations.py \
    --checkpoint warp-rm-sss15/warp_rm_sss15.pt \
    --lerobot-repo sim-bottles-mjwarp-v1

# openpi resolves a repo_id as $HF_LEROBOT_HOME/<repo_id> (data_loader.py:365)
ln -s "$PWD/sim-bottles-mjwarp-v1" \
  "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/sim_put_bottles_mjwarp_rmsss15"
```

`warp_rm_signed_magnitude` is the **first** key in openpi's velocity-column
preference chain on `paper-repro` (`config.py:1003,1072`; `data_loader.py:437`), so
no renaming is required. Only `release-candidate` lacks that resolution.

The vanilla arm reads `sim_put_bottles_mjwarp_rmperobj`. It ignores the reward
column entirely (`rabc_enabled=False`), so the same scored copy works — symlink it
under that name too.

Then compute normalization statistics (the in-repo `assets/` entry for these repos
is an empty placeholder, `{"norm_stats": {}}`):

```bash
# in openpi @ paper-repro
uv run scripts/compute_norm_stats.py --config-name pi0_put_bottles_mjwarp_rabc_sss15
uv run scripts/compute_norm_stats.py --config-name pi0_put_bottles_mjwarp_no_rabc
```

Stats land in `assets/<config-name>/<repo_id>/norm_stats.json`
(`assets_dirs = ./assets/<config-name>`, `compute_norm_stats.py` writes
`assets_dirs / repo_id`). **Check the file is non-empty before training** — the
in-repo entries are `{"norm_stats": {}}` placeholders, and an empty file is not a
loud failure.

Alternatively copy `assets/` out of the released checkpoint
(`uynitsuj/paper-sim-policy-checkpoints/<arm>/assets/`) to guarantee the released
normalization. Because `rmsss15` is a column-only copy of `rmperobj`, one set of
state/action statistics is valid for both arms.

### Train the two arms

```bash
# WARP-BC arm
uv run scripts/train.py pi0_put_bottles_mjwarp_rabc_sss15 \
    --exp-name warp_rabc_sss15 --checkpoint-base-dir ./checkpoints
# vanilla BC baseline
uv run scripts/train.py pi0_put_bottles_mjwarp_no_rabc \
    --exp-name vanilla --checkpoint-base-dir ./checkpoints
```

(`checkpoint_base_dir` defaults to an author-local absolute path; override it.)

Both configs are committed on `paper-repro` and carry the full recipe: pi0 with
`action_horizon=30`, `batch_size=32`, `num_workers=8`, pi0_base init
(`gs://openpi-assets/checkpoints/pi0_base/params`), cosine decay over 30k,
`num_train_steps=30_000`, `save_interval`/`keep_period` 10k. They differ only in
`repo_id` and `rabc_enabled`.

### What must be pinned

Two settings are not recoverable from the checkpoints and change the result:

1. **Scoring geometry** — the gate thresholds the *raw* injected column, so
   `--crop-mode` changes which ~31.5% of chunks survive. Use the same geometry the
   published head was trained under; see the open question in §1–2.
2. **Prompt** — both arms set `prompt_from_task=True`. On the `rmsss15` copy's
   LeRobot-v3 task table this path resolves to the degenerate prompt `"0"`, which
   is what the released arms trained on, while the evaluator serves
   `"Put the plastic bottles in the bin"` (`abc_minimal/eval_policy.py:804`). It is
   inert for these unconditioned arms — but a rebuilt dataset whose task table
   resolves to a real string trains a different model, so check `meta/tasks` rather
   than assuming.

## 4. Rollout and scoring

Deliberately two steps: rollouts write full-horizon qpos traces, and a separate
offline scorer produces the paper metrics.

```bash
# ── 1. serve one arm (openpi @ release-candidate)
hf download uynitsuj/paper-sim-policy-checkpoints --local-dir paper-sim-policy-checkpoints
uv run scripts/serve_paper_sim_policy.py \
    --policy warp_rabc_sss15 \
    --checkpoint-dir paper-sim-policy-checkpoints/warp_rabc_sss15 --port 8000
# (--policy vanilla --checkpoint-dir paper-sim-policy-checkpoints/vanilla for the baseline)

# ── 2. roll out the n=128 set, 4 shards x 32 worlds (abc-rabc)
for i in 0 1 2 3; do
  python -m abc_minimal.eval_policy \
      --policy-backend pi0 --pi0-host 127.0.0.1 --pi0-port 8000 \
      --num-worlds 32 --seed $((20260511 + 32*i)) \
      --no-early-stop \
      --output-dir local_eval_out/fullhz_warp_sh$i
done

# ── 3. score offline (arms must be named vanilla / warp)
python score_bottles.py --trace-dir local_eval_out --arm-glob 'fullhz_{arm}_sh*'
python score_bottles.py --trace-dir local_eval_out --self-test   # locked paper table
```

Details that matter:

- **`--policy-backend pi0` is required.** The default backend is `dit` and raises
  `policy_backend='dit' requires --checkpoint`. In pi0 mode the evaluator forces the
  blocking full-chunk scheme (`execute_chunk_dim=30`, `num_chunks=60`, no
  fast-inference, no RTC) = 1800 actions = the 60 s horizon; do not override those.
- **`--no-early-stop` is required.** Early stop truncates the qpos trace the moment
  the live primitive first reads 6/6 — and truncates the faster arm more — which
  makes offline re-scoring invalid.
- **The world set is the seeds.** World seed = `--seed + world_index`
  (`eval_policy.py:999`), so the paper's n=128 is base seeds 20260511 / 20260543 /
  20260575 / 20260607 x 32 worlds = seeds 20260511-20260638, **identical for both
  arms** (that pairing is what makes the paired *t* valid).
- **Directory names are load-bearing.** `score_bottles.py` globs
  `<trace-dir>/fullhz_{arm}_sh*/qpos_trace_*.npz` with arms `vanilla` and `warp`.
  Each `eval_policy` run writes one flat `--output-dir`, so name them accordingly
  (or pass your own `--arm-glob`). Each run also drops a `summary.json` recording
  the full resolved config.

**Success criterion.** A bottle counts as placed iff *any part of it* (5 points
sampled along its long axis from the free-joint pose) lies inside a **rim-tight
cylinder**, continuously for **>= 0.5 s** (persistence), **and** is still inside at
the **60 s** horizon (final-standing).

**Metrics.** bottles/scene (paired *t* vs baseline); pooled from-0 per-bottle
placement interval; throughput = `sum(count) / sum(effective_time) * 3600`, where
effective-time is the last-placement time if all bottles were placed, else the 60 s
horizon; and `>= k` placed rates. `--all-rules` emits the appendix robustness
ladder (loose-center / tight-center / tight-lowest / tight-anypart).

> [!CAUTION]
> `eval_policy`'s **live** in-bin count is a loose center-point check used only for
> rollout control (`early_stop`). It is **not** the paper metric and will not match.
> Always score offline.

To verify the published table without re-running anything, score the canonical
traces instead:

```bash
hf download uynitsuj/paper-sim-n128-traces --repo-type dataset \
  --local-dir paper-sim-n128-traces
python score_bottles.py --trace-dir paper-sim-n128-traces --self-test
```

## Limits of fresh repro

Every link runs from public artifacts, but three sources of variance mean fresh
numbers land within tolerances rather than on the published values:

| step | why it varies |
|---|---|
| WARP-RM training | `--batch-size` / `--lr` are not stamped in the published head; checkpoint selection is best-composite, so your run may select a step other than 14,400 |
| pi0 training | stochastic (data order, JAX nondeterminism); no seed is recorded in either arm config |
| rollouts | policy sampling + physics are stochastic; compare with the paired n=128 protocol, not trace-by-trace |

The canonical traces plus `--self-test` remain the deterministic artifact for the
published table.

## The one unrecoverable RM parameter

`--ar-center-stride-sec` is **not stamped** in the published head (verified: no key
matching `center|stride|half|ar_|sampler|alpha|lambda|flip` at any depth other than
`feature_stride`, `sampler`, `standard_stride_src`). It is not a free parameter —
it must track `sss/fps`, else every label rescales by `45/sss`
([recipe.md §4a](recipe.md)).

For this dataset the ambiguity turns out to be **low-impact**, because the kept
episodes are shorter than one standard window (465 src frames), so the path budget
is clipped either way:

| setting | requested path | clipped | mean \|final label\| |
|---|---|---|---|
| centre 1.5 s (default) | 465–2325 | 100% | 0.771 |
| centre 0.5 s (`= sss/fps`) | 155–775 | 100% | 0.594 |

Both are consistent with the stamped `val_vel_magnitude_ratio = 1.0903`, so the
checkpoint cannot distinguish them — a 1.3x label-scale difference, not 3x. On
corpora with **long** episodes the same ambiguity is severe. Checkpoints written
after 2026-08 stamp `ar_center_stride_sec` / `ar_half_range_sec`, closing this.
