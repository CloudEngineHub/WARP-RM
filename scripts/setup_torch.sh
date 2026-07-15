#!/usr/bin/env bash
# Auto-detect the NVIDIA driver's max supported CUDA runtime, flip
# pyproject.toml's `default-groups` line to the matching torch build, then
# sync. The `cu128` / `cu130` dependency groups are declared in pyproject
# under [dependency-groups].
#
# Note: `default-groups` must live in pyproject.toml (uv rejects it in
# uv.toml), so this script edits pyproject in place. The diff is expected —
# commit it or leave it local per machine, your call.
#
# Usage:
#     scripts/setup_torch.sh           # auto-detect
#     scripts/setup_torch.sh cu128     # force a specific group

set -euo pipefail

cd "$(dirname "$0")/.."

forced="${1:-}"

if [[ -n "$forced" ]]; then
    group="$forced"
    echo "[setup_torch] forced group: $group"
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[setup_torch] no nvidia-smi — leaving pyproject untouched, running plain 'uv sync'"
        exec uv sync
    fi

    cuda_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    if [[ -z "$cuda_ver" ]]; then
        echo "[setup_torch] could not parse CUDA Version from nvidia-smi — aborting" >&2
        nvidia-smi | head -5 >&2
        exit 1
    fi

    major="${cuda_ver%%.*}"
    minor="${cuda_ver#*.}"

    if (( major >= 13 )); then
        group="cu130"
    elif (( major == 12 && minor >= 8 )); then
        group="cu128"
    elif (( major == 12 )); then
        echo "[setup_torch] driver CUDA=$cuda_ver is older than 12.8; cu128 wheels may not run. Forcing cu128."
        group="cu128"
    else
        echo "[setup_torch] driver CUDA=$cuda_ver too old for supported builds" >&2
        exit 1
    fi

    echo "[setup_torch] driver CUDA=$cuda_ver -> default-groups = [\"$group\"]"
fi

if ! grep -qE '^default-groups = \["cu(128|130)"\]$' pyproject.toml; then
    echo "[setup_torch] could not find the 'default-groups = [\"cuNNN\"]' line in pyproject.toml" >&2
    exit 1
fi

# Flip the line in place. Only touches the single default-groups entry.
current=$(grep -oE 'default-groups = \["cu(128|130)"\]' pyproject.toml | grep -oE 'cu(128|130)' || true)
sed -i -E "s/^default-groups = \\[\"cu(128|130)\"\\]$/default-groups = [\"$group\"]/" pyproject.toml

# Switching between cu128<->cu130 leaves behind dangling shared libs because
# a chunk of nvidia-* wheels are shared across both groups but pinned to
# different versions. Force a full reinstall on group change to avoid
# "libcudnn.so.9: cannot open shared object file" at import time.
if [[ -n "$current" && "$current" != "$group" ]]; then
    echo "[setup_torch] group change ($current -> $group); running full reinstall"
    exec uv sync --reinstall
fi

exec uv sync

