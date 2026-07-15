"""Thin wandb wrapper so Trainer and Trainer log without importing wandb directly.

Init is optional — if project is None (or wandb isn't installed) the logger
is a no-op. All methods are guarded so training never fails because of logging.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional


class WandbLogger:
    """Guarded wandb logger. Silently no-ops when disabled or unauthenticated.

    Opt-in-by-default pattern (matches openpi/sky): construct with a project
    name; if the user isn't logged in to wandb and no API key is available,
    this gracefully degrades to a no-op instead of prompting or crashing.
    Pass `disabled=True` (or project=None / "") to explicitly disable.
    """

    def __init__(
        self,
        project: Optional[str],
        run_name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        disabled: bool = False,
    ):
        self.enabled = False
        self._run = None
        if disabled or not project:
            if disabled:
                print("[wandb] disabled via flag")
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[wandb] not installed — skipping logging")
            return

        # Avoid interactive prompts / hangs: verify auth before calling init.
        if not _wandb_has_auth(wandb):
            print("[wandb] no API key (run `uv run wandb login` or set "
                  "WANDB_API_KEY) — skipping logging")
            return

        try:
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=dict(config or {}),
                entity=entity,
                tags=tags or [],
                reinit="finish_previous",
            )
            self.enabled = True
            print(f"[wandb] logging to {self._run.project}/{self._run.name}")
        except Exception as e:
            print(f"[wandb] init failed: {e}")
            self.enabled = False

    def log(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            if step is not None:
                self._run.log(dict(data), step=step)
            else:
                self._run.log(dict(data))
        except Exception as e:
            print(f"[wandb] log failed: {e}")

    def log_metrics(self, metrics: Any, step: int, prefix: str = "val") -> None:
        """Log an EvalMetrics-like object. Reads .to_dict() + .composite_score."""
        if not self.enabled:
            return
        payload = {}
        try:
            d = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
            for k, v in d.items():
                payload[f"{prefix}/{k}"] = v
            if hasattr(metrics, "composite_score"):
                payload[f"{prefix}/composite"] = metrics.composite_score
        except Exception as e:
            print(f"[wandb] metrics payload build failed: {e}")
            return
        self.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass


def _default_run_name(tag: str) -> str:
    return tag


def _wandb_has_auth(wandb_mod) -> bool:
    """Check for an available wandb API key without prompting."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return False
    try:
        key = wandb_mod.Api().api_key
    except Exception:
        return False
    return bool(key)
