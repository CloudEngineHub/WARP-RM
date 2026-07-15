"""Canonical per-frame schema for WARP-RM score annotations."""

from __future__ import annotations

COL_PROGRESS = "warp_rm_progress"
COL_SIGNED_MAGNITUDE = "warp_rm_signed_magnitude"


def has_warp_rm_columns(df) -> bool:
    """True when both canonical WARP-RM score columns are present."""
    cols = set(df.columns)
    return COL_PROGRESS in cols and COL_SIGNED_MAGNITUDE in cols
