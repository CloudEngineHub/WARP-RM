# WARP-RM — Warp-Augmented Relative Progress Reward Model

[Project Page](https://uynitsuj.github.io/warp-rm/) &ensp;|&ensp; [arXiv:2606.28320](https://arxiv.org/abs/2606.28320)

WARP-RM learns a dense, **signed relative-progress** signal from robot
manipulation videos: per frame, *how fast and in which direction* the task is
advancing. A frozen DINOv3 backbone feeds a bidirectional Transformer with a
categorical progress-velocity head, trained fully self-supervised via
the **WARP** time-warp augmentation (variable playback speed + reversals,
sampled from a smooth AR(1) process). The output — v̂ₜ ≈ 1 at expert pace,
≈ 0 while stalling, < 0 while regressing — is a per-frame reward used
downstream to filter and reweight behavior-cloning action chunks (**WARP-BC**).

> This repository is the **reward-model**: training, scoring, dense inference, and
> annotation injection — it stops at the injected per-frame reward column.
> Downstream **WARP-BC** chunk filtering and pi0 training live in
> [`uynitsuj/openpi@paper-repro`](https://github.com/uynitsuj/openpi/tree/paper-repro);
> the bottles simulator and paper scorer live in
> [`uynitsuj/abc-rabc`](https://github.com/uynitsuj/abc-rabc/tree/release-candidate).
> The full four-step recipe — WARP-RM → injected column → pi0 → sim eval — is in
> [`docs/reproduce_sim.md`](docs/reproduce_sim.md).

## Install

```bash
git clone https://github.com/uynitsuj/WARP-RM.git && cd WARP-RM
uv sync                       # core (training + scoring)
uv sync --extra wandb         # optional: Weights & Biases logging
```

Requires Python ≥ 3.10 and a CUDA GPU. Torch is pinned per CUDA runtime
(`cu128` default; run `scripts/setup_torch.sh` to auto-pick `cu128`/`cu130`).

The frozen backbone is pulled from the **gated** repo
`facebook/dinov3-vitb16-pretrain-lvd1689m`, so accept Meta's terms on that model
page and authenticate before the first run (or prime your HF cache):

```bash
hf auth login
```

## Quickstart — train a WARP-RM on a LeRobot dataset

WARP-RM trains and scores on **LeRobot v2.1 / v3.0** datasets of *successful*
demonstrations (see [`docs/dataset_schema.md`](docs/dataset_schema.md) for the
expected layout). The defaults **are** the recipe:

```bash
# Real T-shirt folding recipe: bidirectional attention, fs3/sss45, window 32, 15k steps.
python scripts/train.py --lerobot-repo /path/to/your/lerobot/dataset

# Paper simulation results repro RM checkpoint configuration (WARP-RM sss15).
# Flags below are reconstructed from the published head's stamped metadata
# (uynitsuj/warp-rm-sim-bottles-sss15); best-composite selection landed on
# step 14,400 of this 20k-step run. See "Paper simulation" for what is not
# recoverable from the stamp.
python scripts/train.py --ablation no_abs --feature-stride 1 \
    --source-standard-stride 15 --max-steps 20000 \
    --shortest-frac 0.25 \
    --object-counts-json /path/to/sim-bottles-mjwarp-v1/meta/object_counts.json \
    --lerobot-repo /path/to/your/lerobot/dataset

# List ablation configs:
python scripts/train.py --list-configs
```

Batch size defaults to 1024 (needs a ≥40 GB GPU); pass `--batch-size 256` on
smaller cards. Vision backbone features are cached on first use under
`~/.cache/warp_rm/features/` (override with `WARP_RM_FEATURE_CACHE`);
standalone caching via `scripts/data/precompute_features.py`.

Then score the dataset with the trained checkpoint and inject the per-frame
reward columns back into it:

```bash
python scripts/data/write_warp_rm_annotations.py \
    --checkpoint checkpoints/best_model_<tag>.pt \
    --lerobot-repo /path/to/your/lerobot/dataset
# writes per-frame `warp_rm_signed_magnitude` (+ `warp_rm_progress`) columns
```

See [`docs/recipe.md`](docs/recipe.md) for the full data → supervision
walkthrough and [`docs/metrics_glossary.md`](docs/metrics_glossary.md) for
every metric in the training logs.

## Inspect predictions in the browser

`scripts/webui/server.py` serves an interactive inspector: pick a checkpoint
and a dataset, scrub an episode's video on the left, and watch the reconstructed
progress / per-frame velocity / zoomed-velocity panels track the playhead on
the right, shaded by the per-frame WARP-BC weight.

```bash
uv sync --extra webui                      # fastapi + uvicorn (ffmpeg also required)
python scripts/webui/server.py \
    --checkpoint checkpoints/best_model_<tag>.pt \
    --dataset-root /path/to/datasets /another/root   # space-separated; scanned recursively
# → http://127.0.0.1:8000
```

Two signal sources are switchable in the header: **ckpt** runs live dense
inference from the loaded checkpoint, and **sidecar** reads the
`warp_rm_progress` / `warp_rm_signed_magnitude` columns already injected into
the dataset's parquets — no GPU required. Sorting the episode list by mean
velocity reuses the summary cache that
[`scripts/eval/score_episodes.py`](scripts/eval/score_episodes.py) writes, so an
offline scoring pass populates the table for free. DINOv3 features are read from
(and written to) the same cache the trainer uses, so a precached dataset costs
no extra GPU here.

See [`docs/webui.md`](docs/webui.md) for the endpoint reference and cache layout.

## Paper simulation

The paper's public sim reproduction uses these public artifacts beyond this
repo (the RM itself trains and scores any compatible LeRobot dataset):

- **MuJoCo-Warp data + full-state supplement**:
  [`uynitsuj/sim-bottles-mjwarp-v1`](https://huggingface.co/datasets/uynitsuj/sim-bottles-mjwarp-v1).
  Derived dataset (re-rendered at 30 Hz) from ABC's public bottles simulation
  data. It contains the LeRobot dataset, portable scene XML, and full qpos for
  the 2,436 re-renderable episodes.
- **Paper RM head**:
  [`uynitsuj/warp-rm-sim-bottles-sss15`](https://huggingface.co/uynitsuj/warp-rm-sim-bottles-sss15).
  The frozen DINOv3 backbone is obtained separately under Meta's terms.
- **ABC evaluator**: [`uynitsuj/abc-rabc@release-candidate`](https://github.com/uynitsuj/abc-rabc/tree/release-candidate)
  provides the MuJoCo-Warp evaluator and the deterministic `score_bottles.py`.
  Pin: `9a9fbb5bbf109b726b4130b18cd9826a4e262d45`
  ("Public WARP-RM simulator and paper scorer release", 2026-07-14).
- **OpenPI repository** — two branches, two jobs.
  [`uynitsuj/openpi@paper-repro`](https://github.com/uynitsuj/openpi/tree/paper-repro)
  provides pi0 **training**: `scripts/train.py`, the WARP-BC chunk filter that
  consumes `warp_rm_signed_magnitude`, and the two paper-arm configs
  (`pi0_put_bottles_mjwarp_rabc_sss15`, `pi0_put_bottles_mjwarp_no_rabc`).
  Pin: `91f99d6` <!-- TODO: verify after pushing paper-repro -->.
  [`uynitsuj/openpi@release-candidate`](https://github.com/uynitsuj/openpi/tree/release-candidate)
  **serves** the released policies. Pin:
  `204eb92dd2af37c4d1189b587d5fbff978383930`
  ("Add public WARP-RM paper simulation policy serving", 2026-07-14).
  Its upstream Pi0 terms apply to the corresponding published policy parameters.
- **Policy checkpoints and canonical traces**:
  [`uynitsuj/paper-sim-policy-checkpoints`](https://huggingface.co/uynitsuj/paper-sim-policy-checkpoints)
  and [`uynitsuj/paper-sim-n128-traces`](https://huggingface.co/datasets/uynitsuj/paper-sim-n128-traces).

> [!NOTE]
> `release-candidate` and `paper-repro` are **branches, not tags** — they can move.
> The SHAs above are the commits these instructions were verified against; check
> them out explicitly if you need the numbers to line up:
>
> ```bash
> git clone https://github.com/uynitsuj/abc-rabc.git
> git -C abc-rabc checkout 9a9fbb5bbf109b726b4130b18cd9826a4e262d45
> # serving
> git clone https://github.com/uynitsuj/openpi.git openpi-serve
> git -C openpi-serve checkout 204eb92dd2af37c4d1189b587d5fbff978383930
> # training
> git clone https://github.com/uynitsuj/openpi.git openpi-train
> git -C openpi-train checkout 91f99d6   # TODO: verify after pushing paper-repro
> ```

For the deterministic n=128 audit, download the public trace artifact and run
the public ABC scorer:

```bash
hf download uynitsuj/paper-sim-n128-traces --repo-type dataset \
  --local-dir ../traces/paper-sim-n128
git clone https://github.com/uynitsuj/abc-rabc.git && cd abc-rabc
git checkout 9a9fbb5bbf109b726b4130b18cd9826a4e262d45
python score_bottles.py --trace-dir ../traces/paper-sim-n128 --self-test
```

The self-test verifies every published table value from the canonical traces.
It uses only the published data, checkpoints, traces, and source repositories;
fresh rollouts are evaluated against tolerances rather than expected
to be trajectory-identical.

### Scope: what this repo does and does not get you

**From this repo alone:** training a WARP-RM on the sim dataset, scoring it, and
injecting the per-frame reward columns. Plus *verifying* the published n=128 table
via the trace self-test above — which checks published traces with a published
scorer, and is not the same as regenerating them.

**From the companion repos** — every step is runnable, but the details live there:

| piece | where |
|---|---|
| WARP-BC chunk filter | **binary gate**: keep iff chunk-final `velocity > 1.0` (~31.5% of chunks), kept weight 1.0 — see [`reproduce_sim.md`](docs/reproduce_sim.md#the-gate). Not the soft mean-velocity form that `ComputeRABCWeights` applies with its defaults. |
| weighted dataset | build it yourself: `sim-bottles-mjwarp-v1` + the injected column, no extra download — see [`reproduce_sim.md`](docs/reproduce_sim.md#build-the-weighted-dataset) |
| pi0 **training** | `openpi@paper-repro`; both arm configs are committed — see [`reproduce_sim.md`](docs/reproduce_sim.md#train-the-two-arms) |
| rollout + success criterion + metrics | `abc-rabc`; the n=128 seed set and trace layout are in [`reproduce_sim.md`](docs/reproduce_sim.md#4-rollout-and-scoring) |
| column naming | `paper-repro` resolves `warp_rm_signed_magnitude` first — **no rename needed**. Only `release-candidate` lacks that resolution (it reads `rorm_velocity`/`rorm_weight`). |

#### Config recovered from the published checkpoint

The published head stamps its own configuration, so most of the recipe is
recoverable rather than guessed. Read from
`uynitsuj/warp-rm-sim-bottles-sss15/warp_rm_sss15.pt`
(sha256 verified against its `MANIFEST.json`):

| stamped | value |
|---|---|
| `ablation` / `sampler` / `attention` | `no_abs` / `ar` / `bidirectional` |
| `feature_stride` / `standard_stride_src` | `1` / `15` → `std_feat_steps = 15` (exact) |
| `cameras` / `fusion` | `['top_camera-images-rgb']` (the default) / `concat` |
| `rel_bin_min` / `rel_bin_max` | `-3.0` / `3.0` |
| `label_mode` / `progress_shape` | `relative` / `uniform` |
| `step` | **14400** |
| `val_vel_spearman` / `val_cum_sign` / `val_spearman` | 0.8949 / 0.9908 / 0.9982 |
| composite | **3.8515** |
| `branch_tag` | `wr_A_rm_perobj_s25_mjwarp_sss15_20k` |

Two things follow. **Step 14,400 is not a manual pick** — the trainer saves
best-composite only, and that is simply where the best composite landed; your run
may select a different step. And `branch_tag`'s `perobj_s25` indicates the
**per-object stratified** shortest filter at frac 0.25, which needs
`--object-counts-json`; `meta/object_counts.json` **is** published in the dataset,
so this is reproducible once passed explicitly (added to the command above).

**What the stamp does *not* record:** `--ar-center-stride-sec` /
`--ar-half-range-sec` (not a free parameter — must track `sss/fps`; on this dataset
the ambiguity is low-impact, quantified in
[`docs/reproduce_sim.md`](docs/reproduce_sim.md)), `--batch-size` / `--lr`, and the
episode count surviving the filter. Checkpoints written after 2026-08 stamp the
sampler calibration.

> [!WARNING]
> **The published sim dataset is 640x480, not square** (verified on the bitstream),
> so training on it with default flags squashes frames 1.33x — pass
> `--crop-mode center`. Whether the *released head* saw squashed or pre-cropped
> frames is unrecorded (no `crop_mode` stamp) and the published dataset may not be
> the training corpus; see [`docs/reproduce_sim.md`](docs/reproduce_sim.md).

**Full recipe — the WARP-BC gate, the weighted-dataset build, both pi0 arm configs,
the n=128 rollout protocol, and the eval success criterion — is in
[`docs/reproduce_sim.md`](docs/reproduce_sim.md).**

## Repository layout

```
WARP-RM/
├── scripts/
│   ├── train.py                      # training entrypoint (defaults = the recipe)
│   ├── config.py                     # shared hyperparameters
│   ├── data/{precompute_features,write_warp_rm_annotations,inject_rm_column}.py
│   ├── eval/{score_episodes,render}.py
│   ├── webui/                        # inspector web UI (server.py + static/)
│   └── visualize_samplers.py         # visualize the WARP time-warp sampler
├── warp_rm/                          # the package
│   ├── data/      {dataset, samplers, labelers, lerobot_dataset, video_reader, ...}.py
│   ├── models/    aggregators/transformer.py, backbones/dinov3.py
│   ├── core/      {trainer, loss, metrics, ablation, ...}.py
│   ├── eval/      scoring.py
│   ├── visualization/ {inference, renderer, plotting}.py
│   └── utils/     {caching, environment, schema}.py
└── docs/{recipe,dataset_schema,metrics_glossary,webui}.md
```

## Configuration

| Setting | Default | Override |
|---|---|---|
| Feature cache root | `~/.cache/warp_rm/features/` | `--cache-dir`, or `WARP_RM_FEATURE_CACHE` |
| W&B logging | off | `--wandb` (needs `--extra wandb`) |
| S3 bucket (optional cloud artifacts) | unset | `WARP_RM_S3_BUCKET` |
| Checkpoint S3 upload | off | `WARP_RM_CKPT_S3_PREFIX` |

## Citation

```bibtex
@article{yu2026warp,
  title={WARP-RM: A Warp-Augmented Relative Progress Reward Model for Data Curation},
  author={Yu, Justin and Goldberg, Andrew and Kondap, Kavish and El-Refai, Karim and Ransing, Ethan and Chen, Qianzhong and Schwager, Mac and Shentu, Fred and Wu, Philipp and Goldberg, Ken},
  journal={arXiv preprint arXiv:2606.28320},
  year={2026}
}
```

## License

This project is available under the [MIT License](LICENSE).
