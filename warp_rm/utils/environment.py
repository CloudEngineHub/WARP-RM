"""
Environment utilities: RNG seeding, git branch tracking.
"""

import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_branch_tag() -> str:
    """Extract git branch name (removes 'autoresearch/' prefix)."""
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        )
        branch = r.stdout.strip()
        return branch.replace("autoresearch/", "") if branch.startswith("autoresearch/") else branch
    except Exception:
        return "unknown"
