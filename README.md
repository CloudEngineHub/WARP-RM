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

> This repository is the **reward-model**: training, scoring, dense
> inference, and annotation injection. The downstream **WARP-BC** chunk
> reweighting and simulation evaluator live in the public companion repositories
> linked below.

## Install

```bash
git clone https://github.com/uynitsuj/WARP-RM.git && cd WARP-RM
uv sync                       # core (training + scoring)
uv sync --extra wandb         # optional: Weights & Biases logging
```

Requires Python ≥ 3.10 and a CUDA GPU. Torch is pinned per CUDA runtime
(`cu128` default; run `scripts/setup_torch.sh` to auto-pick `cu128`/`cu130`).

## Quickstart — train a WARP-RM on a LeRobot dataset

WARP-RM trains and scores on **LeRobot v2.1 / v3.0** datasets of *successful*
demonstrations (see [`docs/dataset_schema.md`](docs/dataset_schema.md) for the
expected layout). The defaults **are** the recipe:

> [!IMPORTANT]
> **Is your dataset's video already square?** Check
> `meta/info.json → features.<camera>.shape` before your first run.
>
> The default `--crop-mode squash` resizes each frame straight to 224×224
> ignoring aspect ratio. That is a **no-op for square video** (the resize is
> skipped), which is why it is the default — every dataset produced by the usual
> LeRobot conversion scripts is already center-cropped to square at encode time.
>
> But if your videos are stored at a **native non-square** ratio — e.g.
> `[720, 1280, 3]`, very common for off-the-shelf recordings — the default
> **horizontally compresses every frame by 1.78×** and hands DINOv3 distorted
> geometry (circles become ellipses). Pass `--crop-mode center` to center-crop to
> the largest centered square first, matching what the conversion scripts do:
>
> ```bash
> python scripts/train.py --crop-mode center --lerobot-repo /path/to/dataset
> ```
>
> The mode is part of the feature-cache key and is stamped into the checkpoint,
> so scoring, rendering and the inspector all reproduce it automatically — you
> only set it at train time. See [`docs/dataset_schema.md`](docs/dataset_schema.md#frame-geometry--crop-mode).

```bash
# Real T-shirt folding recipe: bidirectional attention, fs3/sss45, window 32, 15k steps.
python scripts/train.py --lerobot-repo /path/to/your/lerobot/dataset

# Paper simulation results repro RM checkpoint configuration (WARP-RM sss15):
# selected checkpoint was step 14,400 of this 20k-step no-abs run.
python scripts/train.py --ablation no_abs --feature-stride 1 \
    --source-standard-stride 15 --max-steps 20000 \
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
  provides the MuJoCo-Warp evaluator and deterministic `score_bottles.py`.
- **OpenPI repository**: [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi/tree/release-candidate)
  provides optional pi0 + RABC policy support. Its upstream Pi0 terms apply
  to the corresponding published policy parameters.
- **Policy checkpoints and canonical traces**:
  [`uynitsuj/paper-sim-policy-checkpoints`](https://huggingface.co/uynitsuj/paper-sim-policy-checkpoints)
  and [`uynitsuj/paper-sim-n128-traces`](https://huggingface.co/datasets/uynitsuj/paper-sim-n128-traces).

For the deterministic n=128 audit, download the public trace artifact and run
the public ABC scorer:

```bash
hf download uynitsuj/paper-sim-n128-traces --repo-type dataset \
  --local-dir ../traces/paper-sim-n128
git clone --branch release-candidate https://github.com/uynitsuj/abc-rabc.git
cd abc-rabc
python score_bottles.py --trace-dir ../traces/paper-sim-n128 --self-test
```

The self-test verifies every published table value from the canonical traces.
It uses only the published data, checkpoints, traces, and source repositories;
fresh rollouts are evaluated against tolerances rather than expected
to be trajectory-identical.

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
