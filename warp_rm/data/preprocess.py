"""Frame geometry preprocessing shared by every decode path.

One helper, used by the training precompute (`warp_rm/utils/caching.py`), the
online loaders (`warp_rm/data/dataset.py`) and the on-the-fly inference path
(`warp_rm/visualization/renderer.py`), so a dataset is preprocessed the same way
whether its features are cached, streamed, or computed at score time. A mismatch
between those paths silently shifts the feature distribution under the model.

Crop modes
----------
``squash`` (default)
    Resize straight to ``image_size`` x ``image_size``, ignoring aspect ratio —
    the historical behavior. Correct for datasets whose videos are *already*
    square (everything converted by the LeRobot conversion scripts, which
    center-crop at encode time), and a no-op for those since the resize is
    skipped when the frame is already the target size.

``center``
    Center-crop to the largest centered square, then resize. Matches the ffmpeg
    filter the conversion scripts apply at encode time::

        crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2'

    Use this for datasets stored at their native non-square aspect ratio (e.g.
    1280x720), where ``squash`` would compress the horizontal axis by 1.78x and
    hand DINOv3 distorted geometry.

The mode is mixed into the feature-cache key (see
`warp_rm.utils.caching._ep_cache_key_stable`), so squashed and cropped features
can never collide under one key.

Interpolation
-------------
``squash`` keeps ``cv2.resize``'s default (INTER_LINEAR) so previously cached
features stay bit-identical. ``center`` uses INTER_AREA when downscaling, which
is the correct filter for large reductions — a native 1280x720 source is a 5.7x
reduction, where INTER_LINEAR aliases badly. This only matters for datasets fed
at native resolution; pre-resized 224x224 datasets skip the resize entirely.
"""

import cv2
import numpy as np

CROP_MODES = ("squash", "center")


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Crop the largest centered square. No-op on already-square input."""
    h, w = frame.shape[:2]
    if h == w:
        return frame
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def resize_frame(frame: np.ndarray, image_size: int,
                 crop_mode: str = "squash") -> np.ndarray:
    """Bring one decoded RGB frame to ``image_size`` x ``image_size``.

    Args:
        frame: (H, W, 3) uint8 RGB.
        image_size: Target square edge length.
        crop_mode: ``"squash"`` or ``"center"`` — see the module docstring.

    Returns:
        (image_size, image_size, 3) array, same dtype as the input.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(
            f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}"
        )

    if crop_mode == "center":
        frame = center_crop_square(frame)
        h, w = frame.shape[:2]
        if h != image_size or w != image_size:
            interp = cv2.INTER_AREA if h > image_size else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (image_size, image_size), interpolation=interp)
        return frame

    # squash: historical path, byte-identical to the pre-crop-mode code.
    h, w = frame.shape[:2]
    if h != image_size or w != image_size:
        frame = cv2.resize(frame, (image_size, image_size))
    return frame
