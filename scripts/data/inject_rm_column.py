#!/usr/bin/env python3
"""Move WARP-RM per-frame velocity between a LeRobot v3 dataset column and
per-episode float64 ``.bin`` sidecar files.

Behavior-cloning pipelines consume the WARP-RM signal in one of two shapes:

  * a per-frame **parquet column** (``warp_rm_signed_magnitude``) inside the
    LeRobot dataset itself (e.g. openpi-style RABC reads the column), or
  * a per-episode **float64 .bin sidecar** written next to each staged
    episode's states/actions (e.g. a DiT pipeline reading
    ``velocity_<name>.bin`` per episode directory).

This script converts between the two:

  extract:  scored LeRobot v3 (column already injected by
            ``scripts/data/write_warp_rm_annotations.py --mode inject``)
            → ``<staged-dir>/<split>/<episode_id>/<out-name>``
            (float64, one value per frame, frame_index order).

  inject:   per-episode ``.bin`` sidecars → a data/meta-only copy of a
            LeRobot v3 dataset whose existing float32 column is rewritten
            in place (arrow schema preserved). Videos are NOT copied —
            symlink or sync them separately. A full read-back validation
            compares the written column against the .bin per episode.

Join key: the ``source_episode_id`` column of ``meta/episodes`` parquets
matches the staged episode directory name; rows are located via
``(data/chunk_index, data/file_index)``, filtered by ``episode_index`` and
ordered by ``frame_index``.

Usage:
    # column -> .bin sidecars (for .bin-consuming policy trainers)
    python scripts/data/inject_rm_column.py extract \
        --data-dir /path/to/scored_lerobot --staged-dir /path/to/staged \
        --out-name velocity_warp_rm.bin

    # .bin sidecars -> fresh LeRobot copy with the column rewritten
    python scripts/data/inject_rm_column.py inject \
        --base /path/to/source_lerobot --dest ~/.cache/huggingface/lerobot/<repo_id> \
        --staged-dir /path/to/staged --binfile velocity_warp_rm.bin
"""
from __future__ import annotations

import dataclasses
import glob
import json
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

DEFAULT_COLUMN = "warp_rm_signed_magnitude"
DEFAULT_SPLITS = "train_sim,val_sim"


# ────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────

def _load_episodes_meta(dataset_root: Path):
    import pandas as pd  # noqa: F401 (pq.to_pandas needs it)
    metas = sorted(glob.glob(f"{dataset_root}/meta/episodes/**/*.parquet",
                             recursive=True))
    if not metas:
        raise SystemExit(f"no meta/episodes parquets under {dataset_root}")
    return pq.read_table(metas).to_pandas()


def _data_path(dataset_root: Path, chunk_index: int, file_index: int) -> Path:
    """Resolve a v3 data shard path from meta/info.json's data_path template."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    template = info.get(
        "data_path",
        "data/chunk-{chunk_index:03d}/file-{file_index:05d}.parquet",
    )
    return dataset_root / template.format(chunk_index=chunk_index,
                                          file_index=file_index)


def _staged_episode_dirs(staged_dir: Path, splits: list[str]) -> dict[str, Path]:
    out = {}
    for sub in splits:
        for ep in (staged_dir / sub).glob("*"):
            if ep.is_dir():
                out[ep.name] = ep
    return out


# ────────────────────────────────────────────────────────────────────
# extract: LeRobot column -> per-episode .bin sidecars
# ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Extract:
    """Write per-frame velocity .bin sidecars from a scored LeRobot v3 dataset."""

    data_dir: str
    """Scored LeRobot v3 root (column already injected)."""

    staged_dir: str
    """Staged episode tree with <split>/<episode_id>/ directories."""

    out_name: str = "velocity_warp_rm.bin"
    """Sidecar filename written into each matched episode directory."""

    column: str = DEFAULT_COLUMN
    splits: str = DEFAULT_SPLITS
    """Comma-separated split subdirectories under --staged-dir."""


def run_extract(cfg: Extract) -> None:
    root = Path(cfg.data_dir)
    meta = _load_episodes_meta(root)
    staged = _staged_episode_dirs(Path(cfg.staged_dir),
                                  [s for s in cfg.splits.split(",") if s])
    cache: dict = {}
    written = mismatch = nomatch = 0
    for _, row in meta.iterrows():
        sid = str(row["source_episode_id"])
        ep_dir = staged.get(sid)
        if ep_dir is None:
            nomatch += 1
            continue
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key not in cache:
            cache[key] = pq.read_table(
                _data_path(root, *key),
                columns=["episode_index", "frame_index", cfg.column],
            ).to_pandas()
        df = cache[key]
        rows = df[df["episode_index"] == int(row["episode_index"])].sort_values(
            "frame_index")
        vel = rows[cfg.column].to_numpy().astype(np.float64).reshape(-1)
        if len(vel) != int(row["length"]):
            print(f"[WARN] {sid}: column rows {len(vel)} != meta length "
                  f"{int(row['length'])}")
            mismatch += 1
        (ep_dir / cfg.out_name).write_bytes(vel.tobytes())
        written += 1
    print(f"wrote {written} '{cfg.out_name}' sidecars ({mismatch} length "
          f"mismatches, {nomatch} LeRobot episodes not in staged set)")


# ────────────────────────────────────────────────────────────────────
# inject: per-episode .bin sidecars -> LeRobot copy with column rewritten
# ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Inject:
    """Rewrite the velocity column of a LeRobot v3 copy from .bin sidecars."""

    base: str
    """Source LeRobot v3 root (must already carry the target float32 column —
    run write_warp_rm_annotations.py --mode inject once if it doesn't)."""

    dest: str
    """Output root: data/ + meta/ are copied here and the column rewritten.
    Videos are NOT copied; symlink or sync them separately."""

    staged_dir: str
    """Staged episode tree with <split>/<episode_id>/<binfile> sidecars."""

    binfile: str
    """Sidecar filename to read in each episode directory (float64/frame)."""

    column: str = DEFAULT_COLUMN
    splits: str = DEFAULT_SPLITS
    validate_only: bool = False
    """Skip copy/inject; just re-validate an existing --dest copy."""


def _build_sid_to_bin(staged_dir: Path, splits: list[str],
                      binfile: str) -> dict[str, np.ndarray]:
    out = {}
    for sid, ep in _staged_episode_dirs(staged_dir, splits).items():
        p = ep / binfile
        if p.exists():
            out[sid] = np.frombuffer(p.read_bytes(), dtype=np.float64)
    return out


def run_inject(cfg: Inject) -> int:
    base, dest = Path(cfg.base), Path(cfg.dest)
    splits = [s for s in cfg.splits.split(",") if s]
    sid2bin = _build_sid_to_bin(Path(cfg.staged_dir), splits, cfg.binfile)
    print(f"[inject] {dest}  <- {cfg.binfile}  ({len(sid2bin)} staged .bin files)")

    if not cfg.validate_only:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for sub in ("data", "meta"):
            shutil.copytree(base / sub, dest / sub)
        print(f"[inject] copied data/ + meta/ -> {dest}")

    meta = _load_episodes_meta(dest)

    # Group episodes by their data parquet so each shard is rewritten once.
    file_groups: dict[tuple, list] = {}
    for _, row in meta.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        file_groups.setdefault(key, []).append(row)

    n_ep_injected = 0
    nomatch = 0
    for key, rows in sorted(file_groups.items()):
        ppath = _data_path(dest, *key)
        table = pq.read_table(ppath)
        pdf = table.to_pandas()
        if cfg.column not in pdf.columns:
            raise SystemExit(
                f"[FATAL] {ppath.name} has no '{cfg.column}' column — inject it "
                f"first with write_warp_rm_annotations.py --mode inject")
        new_col = pdf[cfg.column].to_numpy().astype(np.float32).copy()
        for row in rows:
            sid = str(row["source_episode_id"])
            b = sid2bin.get(sid)
            if b is None:
                nomatch += 1
                continue
            ei = int(row["episode_index"])
            mask = (pdf["episode_index"].to_numpy() == ei)
            fi = pdf["frame_index"].to_numpy()[mask]
            order = np.argsort(fi, kind="stable")
            positions = np.nonzero(mask)[0][order]
            if len(positions) != len(b):
                raise SystemExit(
                    f"[FATAL] len mismatch ep {ei} sid {sid}: table "
                    f"{len(positions)} vs bin {len(b)}")
            new_col[positions] = b.astype(np.float32)
            n_ep_injected += 1
        if cfg.validate_only:
            continue
        # Replace the column preserving the arrow type (float32).
        col_idx = table.schema.get_field_index(cfg.column)
        field = table.schema.field(col_idx)
        table = table.set_column(col_idx, field, pa.array(new_col, type=field.type))
        pq.write_table(table, ppath)

    print(f"[inject] injected {n_ep_injected} episodes ({nomatch} sid nomatch)")

    # ── Validation: read back dest, compare column vs .bin per episode ──
    print("[validate] reading back dest and comparing to .bin ...")
    stats = {"n_ep": 0, "exact": 0, "len_mismatch": 0, "max_abs_diff": 0.0,
             "n_frames": 0, "sum_sq": 0.0}
    cache: dict = {}
    for _, row in meta.iterrows():
        sid = str(row["source_episode_id"])
        b = sid2bin.get(sid)
        if b is None:
            continue
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key not in cache:
            cache[key] = pq.read_table(
                _data_path(dest, *key),
                columns=["episode_index", "frame_index", cfg.column],
            ).to_pandas()
        df = cache[key]
        r = df[df["episode_index"] == int(row["episode_index"])].sort_values(
            "frame_index")
        col = r[cfg.column].to_numpy().astype(np.float64)
        bb = b.astype(np.float32).astype(np.float64)  # what got written, upcast
        stats["n_ep"] += 1
        if len(col) != len(bb):
            stats["len_mismatch"] += 1
            continue
        d = np.abs(col - bb)
        stats["n_frames"] += len(col)
        stats["max_abs_diff"] = max(stats["max_abs_diff"],
                                    float(d.max()) if len(d) else 0.0)
        stats["sum_sq"] += float(np.sum(d * d))
        if len(d) == 0 or d.max() == 0.0:
            stats["exact"] += 1

    rmse = ((stats["sum_sq"] / stats["n_frames"]) ** 0.5
            if stats["n_frames"] else float("nan"))
    print(f"[validate] episodes matched: {stats['n_ep']}")
    print(f"[validate] exact-match (float32): {stats['exact']}/{stats['n_ep']}  "
          f"len_mismatch: {stats['len_mismatch']}")
    print(f"[validate] max|col-bin32|: {stats['max_abs_diff']:.3e}  RMSE: {rmse:.3e}")
    ok = (stats["n_ep"] > 0 and stats["exact"] == stats["n_ep"]
          and stats["len_mismatch"] == 0)
    print(f"[validate] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> Optional[int]:
    cfg = tyro.cli(Union[Extract, Inject])
    if isinstance(cfg, Extract):
        run_extract(cfg)
        return 0
    return run_inject(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
