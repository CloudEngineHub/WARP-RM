"""
Video reading abstractions with backend auto-detection and LRU caching.

Supports decord (preferred), av, and cv2 (fallback).
"""

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# LRU caches for video containers
# ---------------------------------------------------------------------------

_decord_cache: dict[str, object] = {}
_av_cache: dict[str, tuple] = {}
_MAX_CACHE_SIZE = 32
_DECORD_NUM_THREADS: int = int(os.environ.get("DECORD_NUM_THREADS", "2"))


def _evict_if_full(cache: dict, key: str) -> None:
    if len(cache) >= _MAX_CACHE_SIZE and key not in cache:
        cache.pop(next(iter(cache)))


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _get_decord_reader(path: str):
    if path not in _decord_cache:
        _evict_if_full(_decord_cache, path)
        from decord import VideoReader, cpu
        _decord_cache[path] = VideoReader(path, ctx=cpu(0), num_threads=_DECORD_NUM_THREADS)
    return _decord_cache[path]


def _read_decord(video_path: str, indices: list[int]) -> list[np.ndarray]:
    # Bypass the LRU cache: it isn't thread-safe (concurrent readers can be
    # evicted mid-use, corrupting ffmpeg state → NAL/heap aborts). Precache
    # touches each video once, so caching buys nothing here anyway.
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=_DECORD_NUM_THREADS)
    arr = vr.get_batch(indices).asnumpy()
    return [arr[i] for i in range(len(indices))]


def _get_av_container(path: str):
    if path not in _av_cache:
        _evict_if_full(_av_cache, path)
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        _av_cache[path] = (container, stream)
    return _av_cache[path]


def _read_av(video_path: str, indices: list[int]) -> list[np.ndarray]:
    import av as _av  # noqa: F811
    container, stream = _get_av_container(video_path)
    fps = float(stream.average_rate)
    tb = float(stream.time_base)
    H, W = stream.height, stream.width
    sorted_unique = sorted(set(indices))
    needed = set(sorted_unique)
    results: dict[int, np.ndarray] = {}
    fallback = np.zeros((H, W, 3), dtype=np.uint8)
    first_idx, last_idx = sorted_unique[0], sorted_unique[-1]
    seek_ts = int(first_idx / fps / tb)
    try:
        container.seek(seek_ts, backward=True, any_frame=False, stream=stream)
    except Exception:
        container.seek(0)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        fn = int(round(float(frame.pts) * tb * fps))
        if fn in needed:
            results[fn] = frame.to_ndarray(format="rgb24")
            needed.discard(fn)
        if not needed or fn > last_idx + 60:
            break
    return [results.get(i, fallback) for i in indices]


def _read_cv2(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """Read frames at the given indices.

    For mostly-sequential indices (precompute_features always strides
    forward), use grab()/retrieve() — grab() is decode-and-discard, ~10x
    cheaper than set(POS_FRAMES) which re-seeks to the prior keyframe per
    call. We fall back to set() only when an index goes BACKWARD relative
    to the current decoder head, since cv2 can't grab backwards."""
    # Loudly raise on missing/unreadable videos. Previously this silently
    # returned all-black frames via the fallback below — caused entire cloud
    # training runs to collapse on missing-video-sync setups (DINOv3 forward
    # on black frames produces near-constant features → model can't learn).
    # Diagnosed 2026-04-29: hlm sweep collapsed because launch_train.py
    # gated videos sync on precache_mode=True, and the cv2 fallback masked
    # the missing files.
    if not Path(video_path).is_file():
        raise FileNotFoundError(
            f"video not found: {video_path}. cv2's silent zero-frame fallback "
            f"would otherwise produce all-black DINOv3 features and break "
            f"training silently. If running on cloud, ensure launch_train.py "
            f"syncs `<dataset>/videos/` (i.e. --precache-mode or always-sync)."
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2.VideoCapture failed to open: {video_path}. See note in "
            f"_read_cv2 — silent fallback would corrupt training."
        )
    frames: list[np.ndarray] = []
    last_pos = -1
    try:
        for idx in indices:
            target = int(idx)
            # Forward seek by grabbing-and-discarding — fast because cv2
            # only decodes; no keyframe re-seek per frame.
            if target == last_pos + 1:
                ok = cap.grab()
            elif target > last_pos and target - last_pos < 200:
                # Short forward jump: grab through, then retrieve target.
                ok = True
                for _ in range(target - last_pos - 1):
                    if not cap.grab():
                        ok = False
                        break
                if ok:
                    ok = cap.grab()
            else:
                # Backward or far-forward seek: pay the keyframe re-seek.
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok = cap.grab()
            if ok:
                ok2, frame = cap.retrieve()
                if ok2:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    last_pos = target
                    continue
            # Fallback: reuse last frame or emit a black frame.
            frames.append(frames[-1] if frames else np.zeros((480, 848, 3), np.uint8))
            last_pos = target
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BACKEND: str | None = None


def _detect_backend() -> str:
    env = os.environ.get("DATALOADER_BACKEND", "").lower()
    if env in ("decord", "av", "cv2"):
        return env
    try:
        import decord  # noqa: F401
        return "decord"
    except ImportError:
        pass
    try:
        import av  # noqa: F401
        return "av"
    except ImportError:
        pass
    return "cv2"


def read_frames(video_path: str | Path, indices: Sequence[int]) -> list[np.ndarray]:
    """Read specific frame indices from a video file."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _detect_backend()
    video_path, indices = str(video_path), list(indices)
    if not indices:
        return []
    if _BACKEND == "decord":
        try:
            return _read_decord(video_path, indices)
        except Exception:
            pass
    if _BACKEND in ("av", "decord"):
        try:
            return _read_av(video_path, indices)
        except Exception:
            pass
    return _read_cv2(video_path, indices)


def clear_video_cache():
    """Release all cached video containers."""
    for container, _ in _av_cache.values():
        try:
            container.close()
        except Exception:
            pass
    _av_cache.clear()
    _decord_cache.clear()
