# Reproducing the simulated bottles-in-bin results

Audit of what the public artifacts actually let you reproduce, end to end. Read
against the pinned companion commits:

| repo | pin |
|---|---|
| [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi) | `204eb92dd2af37c4d1189b587d5fbff978383930` |
| [`uynitsuj/abc-rabc`](https://github.com/uynitsuj/abc-rabc) | `9a9fbb5bbf109b726b4130b18cd9826a4e262d45` |

The chain has four links. **Three are reproducible; the pi0 training step is not.**

```
 (1) train WARP-RM        (2) score + inject       (3) train pi0 w/ WARP-BC     (4) rollout + score
     this repo        ->      this repo        ->        openpi            ->      abc-rabc
     REPRODUCIBLE            REPRODUCIBLE           NOT REPRODUCIBLE           REPRODUCIBLE
```

## 1–2. Reward model, scoring, injection — reproducible

See the paper-sim command in the [README](../README.md#quickstart--train-a-warp-rm-on-a-lerobot-dataset).
The flags are recoverable from the published head's stamped metadata; the one
exception (`--ar-center-stride-sec`) is discussed at the bottom.

> [!WARNING]
> **The published sim dataset is 640x480, not square** — verified on the bitstream
> (`ffprobe` -> `h264 640x480`), and agreed by `meta/info.json`'s `shape` and its
> encoder-written `info` block for all three cameras. So training on
> `sim-bottles-mjwarp-v1` **as published** with default flags squashes every frame
> horizontally by **1.33x**. Pass `--crop-mode center` to avoid that. See
> [`dataset_schema.md`](dataset_schema.md#frame-geometry--crop-mode).
>
> **Open question — was the published dataset the training corpus?** This is *not*
> a claim that the released head was trained on anamorphic frames; the checkpoint
> carries no `crop_mode` stamp, so its training geometry is unrecorded. Two signals
> suggest the corpora may differ: the head's `branch_tag` is
> `wr_A_rm_perobj_s25_mjwarp_sss15_20k`, and openpi's paper-sim configs reference
> `sim_put_bottles_mjwarp_rmperobj` / `..._rmsss15` — none of which is
> `sim-bottles-mjwarp-v1`. A 224x224 center-cropped conversion may have been used
> upstream (the usual mcap->LeRobot path applies
> `crop='min(iw,ih)':'min(iw,ih)'` then `scale=224:224`) without being the artifact
> that got published. Resolving this needs the author; until then, treat "matches
> the published head" as unverified either way.

## 3. The WARP-BC weighting — formula documented, training not reproducible

### The formula

`openpi/src/openpi/transforms.py::ComputeRABCWeights` — per training sample:

```python
vel    = data["rorm_velocity"]              # (action_horizon,) per-frame velocity
weight = np.sum(vel) / max(len(vel), 1)     # mean over the action horizon
weight = np.clip(weight, clip_min, clip_max)   # defaults: 0.0, 1.0
data["sample_weights"] = np.float32(weight)
```

So the chunk weight is the **mean per-frame velocity over the action horizon,
clipped to [0, 1]** — a scalar per sample. `action_horizon=30` in both paper-sim
configs. There is no thresholding, no sigmoid, and no episode-level term: earlier
`q_threshold` / `multiplicative` schemes do **not** appear at this commit.

`sample_weights` reaches the loss via `openpi/src/openpi/models/model.py:103,128`.

An alternative path, `LeRobotYamRormDataConfig` (`training/config.py:582`), instead
reads a **precomputed** per-frame column `rorm_weight` (documented in-line as
`clip(velocity, 0, inf)`) and repacks it straight to `sample_weights`. Note its
docstring claims it "computes per-sample RABC weights via the ComputeRABCWeights
transform", but the code path does not invoke that transform — it only repacks.
Treat the docstring as stale.

> [!IMPORTANT]
> **Column names do not match.** This repo writes `warp_rm_signed_magnitude` and
> `warp_rm_progress`. openpi reads `rorm_velocity` / `rorm_weight`. No alias exists
> in either repo, so the output of
> [`write_warp_rm_annotations.py`](../scripts/data/write_warp_rm_annotations.py) is
> not directly consumable — you must rename the column, or write it under the
> legacy name via
> [`inject_rm_column.py --column rorm_weight`](../scripts/data/inject_rm_column.py).

### Why pi0 training is not reproducible

openpi contains exactly two paper-sim configs, and **both are inference-only**:

```python
TrainConfig(name="pi0_warp_rm_sim_bottles_vanilla",     data=LeRobotYamDataConfig(repo_id="sim_put_bottles_mjwarp_rmperobj", ...))
TrainConfig(name="pi0_warp_rm_sim_bottles_rabc_sss15",  data=LeRobotYamDataConfig(repo_id="sim_put_bottles_mjwarp_rmsss15",  ...))
```

Three blockers:

1. They use `LeRobotYamDataConfig`, **not** `LeRobotYamRormDataConfig` — so no
   reward weighting is applied. The in-repo comment is explicit: *"the training-only
   reward weighting is already represented in the released parameter trees; inference
   uses the ordinary YAM observation/action transforms below."*
2. Their `repo_id`s (`sim_put_bottles_mjwarp_rmperobj`, `…_rmsss15`) are
   pre-weighted local dataset variants, not the published
   `sim-bottles-mjwarp-v1`. They are not obtainable.
3. No optimiser/schedule/steps/seed for the sim runs is recorded in either config
   (both inherit `TrainConfig` defaults).

The configs that *do* wire up RABC weighting are the real-robot tshirt ones
(`pi0_yam_tshirt_rabc`, `pi05_yam_tshirt_rabc`, …), which are a different
experiment. So the released policy checkpoints can be **served and evaluated**,
but not **retrained**.

## 4. Rollout and scoring — reproducible

`abc-rabc` documents this precisely (`score_bottles.py` docstring). It is a
deliberate two-step process:

```bash
# 1. rollouts -> fp16 qpos traces for the full 60 s horizon of every world
python -m abc_minimal.eval_policy --no-early-stop
# 2. offline scoring -> the paper metrics
python score_bottles.py --trace-dir local_eval_out --arm-glob 'fullhz_{arm}_sh*'
python score_bottles.py --trace-dir local_eval_out --self-test   # locked paper table
```

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
