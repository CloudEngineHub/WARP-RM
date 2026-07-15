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
