"""
Structured per-run JSONL logger.

Every training run writes a stream of structured rows to
`logs/runs/{YYYY-MM-DD}/{tag}.jsonl` instead of relying on
a human-edited research log as the source of truth. Rows are
queryable (pandas.read_json, jq, duckdb) and race-free because each
run has its own file.

Row types:
  - "start": config snapshot at training start
  - "eval":  metrics + optional Q-distribution summary at each eval step
  - "checkpoint": path + metric snapshot when a new best is saved
  - "finish": final metrics + status + git_sha + upload_status="pending"

After training, an optional `scripts/upload_run.py` uploader (not part
of this package; `spawn_uploader` no-ops when it is absent) can be
spawned detached to upload the JSONL to S3 and rewrite the
`upload_status` field to "success"/"failed".
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scripts.config import warp_rm_s3_bucket


DEFAULT_RUNS_DIR = Path("logs/runs")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_branch() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


class RunLogger:
    """Per-run JSONL writer. Append-only, one file per (date, tag).

    Typical usage:
        logger = RunLogger(tag="yam_tshirt_online_q_linrank4_30k",
                           config={"ablation": "c51_abs_only", ...})
        logger.start()
        ...
        logger.eval(step=5000, metrics={...}, q_percentiles={...})
        logger.checkpoint(step=5000, path=".../best_model_*.pt",
                          composite=3.78)
        ...
        logger.finish(status="succeeded", final_metrics={...})
        logger.spawn_uploader(s3_prefix="s3://.../warp_rm/runs_jsonl/")
    """

    def __init__(
        self,
        tag: str,
        config: Optional[dict[str, Any]] = None,
        runs_dir: Path = DEFAULT_RUNS_DIR,
    ):
        self.tag = tag
        self.config = dict(config or {})
        day = _today_dir()
        self.path = runs_dir / day / f"{tag}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = False
        self._finished = False

    # ────────────────────────────────────────────────────────────────
    # Row writers
    # ────────────────────────────────────────────────────────────────

    def _write(self, row: dict[str, Any]) -> None:
        row["ts"] = _iso_now()
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def start(self) -> None:
        """Emit a 'start' row with config + host metadata."""
        if self._started:
            return
        self._write({
            "row_type": "start",
            "tag": self.tag,
            "config": self.config,
            "git_sha": _git_sha(),
            "git_branch": _git_branch(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "argv": sys.argv,
            "wall_start": time.time(),
        })
        self._started = True

    def eval(
        self,
        step: int,
        metrics: dict[str, float],
        q_percentiles: Optional[dict[str, float]] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit one eval row. `q_percentiles` is the per-step Q-distribution
        summary if online-quality is on (keys: p10, p25, p50, p75, p90,
        min, max, mean)."""
        row = {
            "row_type": "eval",
            "tag": self.tag,
            "step": step,
            "metrics": metrics,
        }
        if q_percentiles is not None:
            row["q_percentiles"] = q_percentiles
        if extras:
            row["extras"] = extras
        self._write(row)

    def checkpoint(
        self,
        step: int,
        path: str,
        composite: Optional[float] = None,
    ) -> None:
        self._write({
            "row_type": "checkpoint",
            "tag": self.tag,
            "step": step,
            "path": path,
            "composite": composite,
        })

    def finish(
        self,
        status: str,
        final_metrics: dict[str, float],
        training_seconds: Optional[float] = None,
    ) -> None:
        """Emit a 'finish' row with upload_status='pending'. The
        uploader rewrites this in place to 'success'/'failed'."""
        if self._finished:
            return
        self._write({
            "row_type": "finish",
            "tag": self.tag,
            "status": status,
            "final_metrics": final_metrics,
            "training_seconds": training_seconds,
            "wall_end": time.time(),
            "upload_status": "pending",
        })
        self._finished = True

    # ────────────────────────────────────────────────────────────────
    # Detached upload
    # ────────────────────────────────────────────────────────────────

    def spawn_uploader(
        self,
        s3_prefix: str = f"s3://{warp_rm_s3_bucket()}/warp_rm/runs_jsonl/",
        repo_root: Optional[Path] = None,
    ) -> Optional[int]:
        """Fire-and-forget spawn of `scripts/upload_run.py` as a
        detached process. Returns the PID (for diagnostic / telemetry)
        or None if the uploader script is missing.

        The caller does not wait; on crash the pending row stays,
        next invocation can retry via `scripts/upload_run.py --retry`.
        """
        root = repo_root or Path(__file__).resolve().parent.parent.parent
        uploader = root / "scripts" / "upload_run.py"
        if not uploader.exists():
            return None
        cmd = [
            sys.executable,
            str(uploader),
            "--jsonl", str(self.path),
            "--s3-prefix", s3_prefix,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return proc.pid
        except Exception:
            return None
