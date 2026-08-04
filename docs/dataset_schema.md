# Dataset schema — what WARP-RM reads and writes

WARP-RM consumes **LeRobot v2.1 / v3.0** datasets of *successful* demonstrations
and (optionally) writes per-frame reward columns back into them. This page is
the contract for "bring your own data."

## Expected layout

A LeRobot dataset root looks like:

```
<dataset>/
├── meta/
│   ├── info.json            # features, fps, total_episodes, codebase_version
│   ├── episodes.jsonl       # per-episode index + length (v2.1) / episodes/ (v3.0)
│   └── stats.json           # normalization stats
├── data/.../*.parquet       # per-frame rows (one parquet group per chunk)
└── videos/.../*.mp4         # encoded camera streams
```

WARP-RM reads frames from **one camera stream**, selected by `--camera-key`
(default `top_camera-images-rgb`). The key must be a video feature present in
`meta/info.json → features`. Episode discovery (`warp_rm/data/lerobot_dataset.py`)
honors both v2.1 and v3.0 (chunked) layouts.

Notes:
- **fps** is read from `meta/info.json` (default 30). The canonical inference
  stride and the label normalization assume a fixed fps across the dataset.
- **Scalar features** in `info.json` must declare `shape: [1]` (not `[]`), or
  LeRobot's feature loader raises during stats/normalization.
- Videos are required for feature *extraction*; once features are cached
  (`~/.cache/warp_rm/features/dinov3_fs<stride>/<dataset>/`), scoring/training
  can run features-only (`--require-cached`).

## Frame geometry — `--crop-mode`

**Check this before your first training run on any new dataset.** Look at
`meta/info.json → features.<camera>.shape`, which is `[height, width, channels]`:

| `shape` | What to pass | Why |
|---|---|---|
| `[224, 224, 3]` (or any H == W) | nothing — the default is correct | Already square; the resize is skipped entirely, so `squash` is a no-op |
| `[720, 1280, 3]` (any H ≠ W) | **`--crop-mode center`** | Otherwise every frame is squashed to 224×224, compressing the long axis by `max(H,W)/min(H,W)` (1.78× for 16:9) |

The two modes (implemented in `warp_rm/data/preprocess.py`, applied identically
by the training precompute, the online loaders and the on-the-fly inference
path):

- **`squash`** (default) — `cv2.resize` straight to 224×224, ignoring aspect
  ratio. Correct and free for datasets already converted to square video, which
  is every dataset produced by the standard LeRobot conversion scripts: they
  center-crop at *encode* time with
  `crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2'`.
- **`center`** — center-crop to the largest centered square, then resize
  (INTER_AREA when downscaling, which matters at large reductions: a 1280×720
  source is a 5.7× reduction where bilinear aliases). Reproduces the same
  geometry the conversion scripts bake in, without re-encoding the dataset.

What squashing actually costs, stated no more strongly than we can support: a
16:9 frame is scaled anisotropically by 1.78×, so a wheel photographed round
reaches the backbone as an ellipse. DINO-family pretraining does use
`RandomResizedCrop` with aspect-ratio jitter (commonly ~[3/4, 4/3]), so the
backbone is not naive to *mild* anisotropy — but 1.78× is outside that usual
range, and we have not measured DINOv3's feature degradation at that ratio.
Treat it as an untested regime rather than a known failure.

The argument that does not depend on pretraining details: squashing spends the
224×224 budget unevenly (the horizontal axis is decimated 5.7× vs the vertical's
3.2×), and whatever geometry you train on must be reproduced at score time or
the features shift under the model. Cropping keeps the pixel budget isotropic
and matches what the conversion scripts already bake into square datasets.

Practical notes:

- The mode is **mixed into the feature-cache key**, so `squash` and `center`
  features can never collide; switching modes triggers a fresh extraction rather
  than silently reusing the wrong pixels.
- It is **stamped into the checkpoint** (`crop_mode`), and
  `score_episodes.py` / `render.py` / the inspector read it back automatically.
  You set it once at train time. Checkpoints trained before this flag existed
  carry no stamp and resolve to `squash`, which is what they were trained with.
- `scripts/data/precompute_features.py` takes the same `--crop-mode` and must be
  given the same value you train with, or you will build a second cache instead
  of reusing the first.
- Cropping *discards* the frame edges. If your task's action happens off-center
  near the left/right edge, prefer re-encoding the dataset to square with the
  framing you want over cropping at load time.

## Per-frame columns WARP-RM writes

`scripts/data/write_warp_rm_annotations.py` injects three per-frame columns into
the parquet (canonical write-side names, defined in `warp_rm/utils/schema.py`):

| Column | Meaning |
|---|---|
| `warp_rm_signed_magnitude` | per-frame signed progress velocity (forward = +, regression = −) — **the reward signal** |
| `warp_rm_progress` | per-frame absolute progress in `[0, 1]` (velocity-integrated) |

The public format has no legacy aliases or quality/weight columns. Derive any
per-chunk sample weight downstream from `warp_rm_signed_magnitude`.

## The reward signal

`warp_rm_signed_magnitude` (the dense per-frame velocity) is the model's output
and what WARP-BC reweights action chunks by. `score_episodes.py` reports a
per-episode summary (`mean_velocity`, `delta_v_std`) of it for quick ranking.
Episode-wide quality indices are deprecated. See [`recipe.md`](recipe.md) §8.
