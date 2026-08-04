# Inspector web UI

`scripts/webui/server.py` serves an interactive viewer for WARP-RM predictions
on LeRobot episodes: video scrubber on the left, three live Plotly panels on the
right (reconstructed progress / per-frame velocity / zoomed velocity), all
shaded by the per-frame WARP-BC weight. The panels mirror what
[`scripts/eval/render.py`](../scripts/eval/render.py) bakes into a video, except
you can scrub, switch checkpoints, and compare episodes without re-rendering.

## Run it

```bash
uv sync --extra webui
python scripts/webui/server.py \
    --checkpoint checkpoints/best_model_<tag>.pt \
    --dataset-root ~/data/lerobot
```

Then open <http://127.0.0.1:8000>.

`ffmpeg` and `ffprobe` must be on `PATH` — LeRobot **v3.0** packs many episodes
into one shard MP4, so the server slices a per-episode clip on first request.
Without slicing, the `<video>` element would load an hours-long shard and
in-episode seeks would land in the wrong episode. LeRobot **v2.1** (one file per
episode) skips this path entirely and serves the source MP4 directly.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind address. `0.0.0.0` exposes on the network. |
| `--checkpoint` | none | Load at startup. Optional — the header dropdown can switch at runtime. |
| `--dataset-root` | see below | Directories scanned **recursively** for `meta/info.json`. Several go **space-separated after one flag** (`--dataset-root A B`); repeating the flag keeps only the last. |
| `--gpu` | none | Sets `CUDA_VISIBLE_DEVICES` for inference. |

Dataset roots default to `~/.cache/huggingface/lerobot`, `~/data/lerobot` and
`~/datasets`, or the `os.pathsep`-separated `WARP_RM_LEROBOT_ROOTS` env var when
set. Checkpoints are discovered from `checkpoints/**/*.pt` under the repo root.

## Signal sources

The header's **source** toggle picks where the plotted signal comes from. The
choice persists in `localStorage`.

- **`ckpt`** — live dense inference from the loaded checkpoint. Needs a GPU
  (CPU works, slowly) and a checkpoint.
- **`sidecar`** — reads the `warp_rm_progress` / `warp_rm_signed_magnitude`
  columns already written into the dataset's parquets by
  [`write_warp_rm_annotations.py --mode inject`](../scripts/data/write_warp_rm_annotations.py).
  No checkpoint, no GPU, no inference. The header shows which checkpoint
  produced those columns, read from `meta/warp_rm_meta.json`, so an injected
  signal is never confused with the currently-loaded one.

A dataset that hasn't been injected returns 404 with the command to fix it,
rather than an empty plot.

## Weighting schemes

The **scheme** control previews the per-frame weight a downstream WARP-BC sample
would receive, applied directly to velocity (no chunk integration):

- `off` — raw signed velocity; shading uses `max(0, v)`.
- `velocity_only` — `clip(v, clip_min, clip_max)`, or `0` below `thresh min`
  when that field is set.

The `clip min` / `clip max` inputs also anchor the red→green colour ramp, so the
gradient stays in lock-step with the scheme regardless of mode.

## Episode list

Sorting by **mean velocity** computes the same per-episode summary that
[`score_episodes.py`](../scripts/eval/score_episodes.py) writes — mean per-frame
progress velocity (pace) and the std of its first difference (smoothness). This
is checkpoint-derived, so it requires a loaded checkpoint in either source mode.

Summaries are pulled in this order, and only the last one costs GPU time:

1. the on-disk JSON cache (shared with `score_episodes.py --webui-cache`),
2. a sidecar under `assets/warp_rm_annotations/<dataset>/` whose
   `metadata.json` pins the currently-loaded checkpoint,
3. dense inference, cached on write.

So running `score_episodes.py` offline over a dataset populates the table for
free, and the server picks up an external writer's updates mid-session without a
checkpoint reload (it watches the cache file's mtime).

## Caches

All under `~/.cache/warp_rm/` (override the root with `WARP_RM_WEBUI_CACHE`;
features follow `WARP_RM_FEATURE_CACHE` instead):

| Path | Contents | Keyed on |
|---|---|---|
| `features/<backbone>_fs<stride>/<dataset>/` | DINOv3 features | Same canonical key as training — see [`warp_rm/utils/caching.py`](../warp_rm/utils/caching.py) |
| `webui_infer/<hash>.npz` | Per-frame progress / velocity / weights | ckpt path + mtime, episode video + frame offset + length, feature stride, window, source stride |
| `webui_infer/<hash>.abs.npz` | Absolute-progress head outputs | Sibling of the above, so adding it never invalidates the main cache |
| `webui_infer/quality_<hash>.json` | Per-episode velocity summaries | ckpt path + mtime |
| `webui_clips/<hash>.mp4` | Sliced per-episode clips | Source shard + frame offset + length |

The feature cache is **shared with the trainer**: a dataset already precached by
`scripts/train.py` or `scripts/data/precompute_features.py` costs zero GPU here,
and features extracted by the inspector are reused by a later training run.

The feature stride is read from the loaded checkpoint's stamped `feature_stride`
rather than assumed, so an FS=2 checkpoint cannot silently score against the FS=3
cache (same key, different content). The same applies to `crop_mode`: a
checkpoint trained with `--crop-mode center` resolves to the center-cropped cache
entries, and one trained (or predating the flag) with `squash` resolves to the
squashed ones. Both are printed on load, so you can see which geometry the
inspector is feeding the model.

## API

Every endpoint takes `repo` (absolute dataset path) and, where relevant,
`ep_idx` and `camera_key` (default `top_camera-images-rgb`).

| Endpoint | Purpose |
|---|---|
| `GET /api/checkpoints` | Discovered checkpoints + the currently-loaded one's metadata |
| `POST /api/checkpoint` | Load a checkpoint (`{"path": "..."}`) |
| `GET /api/datasets` | Discovered LeRobot datasets, with video keys and splits |
| `GET /api/episodes` | Episode list for one dataset (optional `split`) |
| `GET /api/inference` | Live per-frame progress / velocity / weights (+ abs-head overlay) |
| `GET /api/dataset_signals` | The injected `warp_rm_*` columns, same response shape |
| `GET /api/summary` | Velocity summary for one episode (runs inference on a miss) |
| `GET /api/summary/all` | Every *already-computed* summary; never runs inference |
| `GET /api/video` | Per-episode MP4 with HTTP Range support |

Interactive docs are at `/api/docs`.
