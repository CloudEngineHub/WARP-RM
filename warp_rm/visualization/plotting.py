"""
Matplotlib + OpenCV plot rendering helpers for reward visualization videos.

Renders static base plots (progress curves, velocity signals) and provides
fast per-frame overlays (scrubber line, value dots, zoomed velocity window).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Colour palette ────────────────────────────────────────────────────────
C_RECON = "#9B59B6"       # purple  - reconstructed absolute progress
C_GT = "#AAAAAA"          # grey    - ground truth
C_VEL_POS = "#2ECC71"     # green   - positive velocity (forward)
C_VEL_NEG = "#E74C3C"     # red     - negative velocity (rewind)
C_ABS_DIRECT = "#3498DB"  # blue    - direct absolute progress (abs_progress_head)
BG = "#1A1A2E"            # dark background
FG = "#EAEAEA"            # foreground text

_DATA_BG_BGR = (26, 13, 13)
_ZERO_BGR = (102, 102, 102)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# ── Base plot rendering ──────────────────────────────────────────────────

def _style_axis(ax, sp_color="#444466"):
    ax.set(facecolor="#0D0D1A")
    for sp in ax.spines.values():
        sp.set_color(sp_color)
    ax.tick_params(colors=FG, labelsize=7)
    ax.grid(True, color="#222244", linewidth=0.5, alpha=0.7)


def render_base_plot_with_gt(
    abs_progress: np.ndarray,
    raw_velocity: np.ndarray,
    gt_progress: np.ndarray,
    plot_w_px: int,
    plot_h_px: int,
    title: str = "",
) -> tuple[np.ndarray, float, float, list[tuple[float, float]]]:
    """
    3-panel plot: Ground Truth | Reconstructed Progress | Velocity.

    Returns (img_rgb, ax_x0, ax_x1, ax_bboxes).
    """
    dpi = 100
    fig = plt.figure(figsize=(plot_w_px / dpi, plot_h_px / dpi), facecolor=BG)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.65,
                           left=0.13, right=0.97, top=0.93, bottom=0.05)

    if title:
        fig.suptitle(title, fontsize=9, color=FG, fontweight="bold", y=0.985)

    t = np.linspace(0, 1, len(abs_progress))
    t_v = np.linspace(0, 1, len(raw_velocity))

    # Row 0: Ground truth
    ax0 = fig.add_subplot(gs[0])
    _style_axis(ax0)
    ax0.set_ylabel("Progress", color=C_GT, fontsize=7, labelpad=3)
    ax0.set_title("Ground Truth: Frame Index / N", color=C_GT, fontsize=8,
                  fontweight="bold", pad=2, loc="left")
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.plot(t, gt_progress, color=C_GT, linewidth=1.8, zorder=3)
    ax0.fill_between(t, 0, gt_progress, alpha=0.15, color=C_GT, zorder=2)
    ax0.set_xticklabels([])

    # Row 1: Reconstructed absolute progress
    ax1 = fig.add_subplot(gs[1])
    _style_axis(ax1)
    ax1.set_ylabel("Progress", color=C_RECON, fontsize=7, labelpad=3)
    ax1.set_title("Reconstructed Absolute Progress", color=C_RECON, fontsize=8,
                  fontweight="bold", pad=2, loc="left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.plot(t, abs_progress, color=C_RECON, linewidth=1.8, zorder=3)
    ax1.fill_between(t, 0, abs_progress, alpha=0.15, color=C_RECON, zorder=2)
    ax1.set_xticklabels([])

    # Row 2: Per-step velocity
    ax2 = fig.add_subplot(gs[2])
    _style_axis(ax2)
    ax2.set_ylabel("Velocity", color=FG, fontsize=7, labelpad=3)
    ax2.set_title("Per-step Velocity (positive=forward, negative=rewind)",
                  color=FG, fontsize=8, fontweight="bold", pad=2, loc="left")
    ax2.set_xlim(0, 1)
    ax2.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    ax2.plot(t_v, raw_velocity, color=FG, linewidth=1.0, alpha=0.6, zorder=3)
    ax2.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity >= 0), alpha=0.4, color=C_VEL_POS, zorder=2)
    ax2.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity < 0), alpha=0.4, color=C_VEL_NEG, zorder=2)
    ax2.set_xlabel("Normalized time", color=FG, fontsize=7)

    axes = [ax0, ax1, ax2]
    return _finalize_plot(fig, axes, plot_w_px, plot_h_px)


def render_base_plot_no_gt(
    abs_progress: np.ndarray,
    raw_velocity: np.ndarray,
    plot_w_px: int,
    plot_h_px: int,
    title: str = "",
    zoom_half_w: float = 0.05,
) -> tuple[np.ndarray, float, float, list[tuple[float, float]],
           tuple[int, int, int, int], tuple[float, float]]:
    """
    3-panel plot without GT: Reconstructed | Velocity | Zoomed Velocity.

    Returns (img_rgb, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim).
    """
    dpi = 100
    fig = plt.figure(figsize=(plot_w_px / dpi, plot_h_px / dpi), facecolor=BG)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.80,
                           left=0.13, right=0.97, top=0.93, bottom=0.06)

    if title:
        fig.suptitle(title, fontsize=9, color=FG, fontweight="bold", y=0.985)

    t = np.linspace(0, 1, len(abs_progress))
    t_v = np.linspace(0, 1, len(raw_velocity))

    vel_min = float(raw_velocity.min())
    vel_max = float(raw_velocity.max())
    vel_ylim = (min(vel_min, 0.0), max(vel_max, 1e-9))

    # Row 0: Reconstructed absolute progress
    ax0 = fig.add_subplot(gs[0])
    _style_axis(ax0)
    ax0.set_ylabel("Progress", color=C_RECON, fontsize=7, labelpad=3)
    ax0.set_title("Reconstructed Absolute Progress", color=C_RECON, fontsize=8,
                  fontweight="bold", pad=2, loc="left")
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.plot(t, abs_progress, color=C_RECON, linewidth=1.8, zorder=3)
    ax0.fill_between(t, 0, abs_progress, alpha=0.15, color=C_RECON, zorder=2)
    ax0.set_xticklabels([])

    # Row 1: Per-step velocity
    ax1 = fig.add_subplot(gs[1])
    _style_axis(ax1)
    ax1.set_ylabel("Velocity", color=FG, fontsize=7, labelpad=3)
    ax1.set_title("Per-step Velocity (positive=forward, negative=rewind)",
                  color=FG, fontsize=8, fontweight="bold", pad=2, loc="left")
    ax1.set_xlim(0, 1); ax1.set_ylim(*vel_ylim)
    ax1.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    ax1.plot(t_v, raw_velocity, color=FG, linewidth=1.0, alpha=0.6, zorder=3)
    ax1.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity >= 0), alpha=0.4, color=C_VEL_POS, zorder=2)
    ax1.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity < 0), alpha=0.4, color=C_VEL_NEG, zorder=2)
    ax1.set_xticklabels([])

    # Row 2: Zoomed velocity (static axes, data redrawn per-frame via OpenCV)
    ax2 = fig.add_subplot(gs[2])
    _style_axis(ax2)
    ax2.set_ylabel("Velocity", color=FG, fontsize=7, labelpad=3)
    ax2.set_title(f"Zoomed Velocity  ({int(zoom_half_w * 200)}% window, slides with playhead)",
                  color=FG, fontsize=8, fontweight="bold", pad=2, loc="left")
    ax2.set_xlim(-zoom_half_w, zoom_half_w)
    ax2.set_ylim(*vel_ylim)
    ax2.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    ax2.axvline(0, color=(0.9, 0.4, 0.0), linewidth=1.5, linestyle="-", zorder=4, alpha=0.9)
    ax2.set_xlabel("Relative position", color=FG, fontsize=7)
    ticks = [-zoom_half_w, -zoom_half_w / 2, 0, zoom_half_w / 2, zoom_half_w]
    labels = [f"{v:+.0%}" if v != 0 else "now" for v in ticks]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=6)

    axes = [ax0, ax1, ax2]

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(plot_h_px, plot_w_px, 4)[..., :3].copy()

    ax_x0 = float(axes[0].get_window_extent().x0)
    ax_x1 = float(axes[0].get_window_extent().x1)
    ax_bboxes = []
    for ax in axes:
        bb = ax.get_window_extent()
        y_top = plot_h_px - bb.y1
        y_bot = plot_h_px - bb.y0
        ax_bboxes.append((float(y_top), float(y_bot)))

    # Zoom rect for row 2 per-frame blanking
    bb2 = ax2.get_window_extent()
    zoom_rect = (
        int(bb2.x0) + 1, int(plot_h_px - bb2.y1) + 1,
        int(bb2.x1) - 1, int(plot_h_px - bb2.y0) - 1,
    )

    plt.close(fig)
    return img, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim


def render_base_plot_enhanced(
    abs_progress: np.ndarray,
    raw_velocity: np.ndarray,
    plot_w_px: int,
    plot_h_px: int,
    title: str = "",
    direct_abs: np.ndarray | None = None,
    zoom_half_w: float = 0.05,
) -> tuple[np.ndarray, float, float, list[tuple[float, float]],
           tuple[int, int, int, int], tuple[float, float]]:
    """
    4-panel plot for enhanced models: Recon Abs | Direct Abs | Velocity | Zoomed Velocity.

    Matches the WARP-RM experiments/render_longest.py layout.

    Returns (img_rgb, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim).
    """
    dpi = 100
    _sp_color = "#444466"
    _ax_style = dict(facecolor="#0D0D1A")

    vel_min = float(raw_velocity.min())
    vel_max = float(raw_velocity.max())
    vel_ylim = (min(vel_min, 0.0), max(vel_max, 1e-9))

    fig = plt.figure(figsize=(plot_w_px / dpi, plot_h_px / dpi), facecolor=BG)
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.80,
                           left=0.13, right=0.97, top=0.93, bottom=0.06)

    if title:
        fig.suptitle(title, fontsize=9, color=FG, fontweight="bold", y=0.985)

    t = np.linspace(0, 1, len(abs_progress))
    t_v = np.linspace(0, 1, len(raw_velocity))

    # Row 0: Reconstructed Absolute Progress (from velocity integration)
    ax0 = fig.add_subplot(gs[0])
    ax0.set(**_ax_style)
    for sp in ax0.spines.values(): sp.set_color(_sp_color)
    ax0.tick_params(colors=FG, labelsize=7)
    ax0.set_ylabel("Progress", color=C_RECON, fontsize=7, labelpad=3)
    ax0.set_title("Reconstructed Absolute Progress", color=C_RECON, fontsize=8,
                  fontweight="bold", pad=2, loc="left")
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.grid(True, color="#222244", linewidth=0.5, alpha=0.7)
    ax0.plot(t, abs_progress, color=C_RECON, linewidth=1.8, zorder=3)
    ax0.fill_between(t, 0, abs_progress, alpha=0.15, color=C_RECON, zorder=2)
    ax0.set_xticklabels([])

    # Row 1: Direct Absolute Progress (from abs_progress_head)
    ax1 = fig.add_subplot(gs[1])
    ax1.set(**_ax_style)
    for sp in ax1.spines.values(): sp.set_color(_sp_color)
    ax1.tick_params(colors=FG, labelsize=7)
    ax1.set_ylabel("Progress", color=C_ABS_DIRECT, fontsize=7, labelpad=3)
    ax1.set_title("Direct Absolute Progress (abs_progress_head)", color=C_ABS_DIRECT,
                  fontsize=8, fontweight="bold", pad=2, loc="left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(True, color="#222244", linewidth=0.5, alpha=0.7)
    if direct_abs is not None:
        t_da = np.linspace(0, 1, len(direct_abs))
        ax1.plot(t_da, direct_abs, color=C_ABS_DIRECT, linewidth=1.8, zorder=3)
        ax1.fill_between(t_da, 0, direct_abs, alpha=0.15, color=C_ABS_DIRECT, zorder=2)
    ax1.set_xticklabels([])

    # Row 2: Per-step Velocity
    ax2 = fig.add_subplot(gs[2])
    ax2.set(**_ax_style)
    for sp in ax2.spines.values(): sp.set_color(_sp_color)
    ax2.tick_params(colors=FG, labelsize=7)
    ax2.set_ylabel("Velocity", color=FG, fontsize=7, labelpad=3)
    ax2.set_title("Per-step Velocity", color=FG, fontsize=8,
                  fontweight="bold", pad=2, loc="left")
    ax2.set_xlim(0, 1); ax2.set_ylim(*vel_ylim)
    ax2.grid(True, color="#222244", linewidth=0.5, alpha=0.7)
    ax2.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    ax2.plot(t_v, raw_velocity, color=FG, linewidth=1.0, alpha=0.6, zorder=3)
    ax2.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity >= 0), alpha=0.4, color=C_VEL_POS, zorder=2)
    ax2.fill_between(t_v, 0, raw_velocity,
                     where=(raw_velocity < 0), alpha=0.4, color=C_VEL_NEG, zorder=2)
    ax2.set_xticklabels([])

    # Row 3: Zoomed Velocity (placeholder — drawn per-frame)
    ax3 = fig.add_subplot(gs[3])
    ax3.set(**_ax_style)
    for sp in ax3.spines.values(): sp.set_color(_sp_color)
    ax3.tick_params(colors=FG, labelsize=7)
    ax3.set_ylabel("Velocity", color=FG, fontsize=7, labelpad=3)
    ax3.set_title("Zoomed Velocity (sliding window)", color=FG,
                  fontsize=8, fontweight="bold", pad=2, loc="left")
    ax3.set_xlim(0, 1); ax3.set_ylim(*vel_ylim)
    ax3.grid(True, color="#222244", linewidth=0.5, alpha=0.7)
    ax3.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    ax3.set_xlabel("Normalized time", color=FG, fontsize=7)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(plot_h_px, plot_w_px, 4)[..., :3].copy()

    axes = [ax0, ax1, ax2, ax3]
    ax_x0 = float(ax0.get_window_extent().x0)
    ax_x1 = float(ax0.get_window_extent().x1)

    ax_bboxes = []
    for ax in axes:
        bb = ax.get_window_extent()
        y_top = plot_h_px - bb.y1
        y_bot = plot_h_px - bb.y0
        ax_bboxes.append((float(y_top), float(y_bot)))

    # zoom_rect for Row 3 (zoomed velocity panel)
    bb3 = ax3.get_window_extent()
    zoom_rect = (int(bb3.x0), int(plot_h_px - bb3.y1),
                 int(bb3.x1), int(plot_h_px - bb3.y0))

    plt.close(fig)
    return img, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim


def _finalize_plot(fig, axes, plot_w_px, plot_h_px):
    """Extract image and axis metadata from a matplotlib figure."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(plot_h_px, plot_w_px, 4)[..., :3].copy()

    ax_x0 = float(axes[0].get_window_extent().x0)
    ax_x1 = float(axes[0].get_window_extent().x1)
    ax_bboxes = []
    for ax in axes:
        bb = ax.get_window_extent()
        y_top = plot_h_px - bb.y1
        y_bot = plot_h_px - bb.y0
        ax_bboxes.append((float(y_top), float(y_bot)))

    plt.close(fig)
    return img, ax_x0, ax_x1, ax_bboxes


# ── Per-frame overlays (fast, OpenCV-only) ────────────────────────────────

def draw_scrubber(
    base_img: np.ndarray,
    t_norm: float,
    ax_x0: float,
    ax_x1: float,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a vertical scrubber line + triangle at normalized time position."""
    img = base_img.copy()
    x = int(ax_x0 + t_norm * (ax_x1 - ax_x0))
    x = max(int(ax_x0), min(int(ax_x1), x))
    cv2.line(img, (x, 0), (x, img.shape[0]), color_bgr, thickness)
    tri = np.array([[x - 6, 0], [x + 6, 0], [x, 10]], np.int32)
    cv2.fillPoly(img, [tri], color_bgr)
    return img


def draw_value_dots(
    base_img: np.ndarray,
    t_norm: float,
    ax_x0: float,
    ax_x1: float,
    ax_bboxes: list[tuple[float, float]],
    signals: list[tuple[np.ndarray, np.ndarray]],
    colors: list[tuple[int, int, int]],
    ylims: list[tuple[float, float]],
) -> np.ndarray:
    """Draw interpolated value dots on each axis at the scrubber position."""
    for i, (t_arr, y_arr) in enumerate(signals):
        if len(y_arr) == 0:
            continue
        val = float(np.interp(t_norm, t_arr, y_arr))
        x = int(ax_x0 + t_norm * (ax_x1 - ax_x0))
        x = max(int(ax_x0), min(int(ax_x1), x))
        y_range = ylims[i][1] - ylims[i][0]
        y_frac = (np.clip(val, ylims[i][0], ylims[i][1]) - ylims[i][0]) / (y_range if y_range else 1.0)
        y_px = int(ax_bboxes[i][1] - y_frac * (ax_bboxes[i][1] - ax_bboxes[i][0]))
        y_px = max(int(ax_bboxes[i][0]), min(int(ax_bboxes[i][1]), y_px))
        cv2.circle(base_img, (x, y_px), 5, colors[i], -1)
        cv2.circle(base_img, (x, y_px), 5, (255, 255, 255), 1)
    return base_img


def draw_zoomed_velocity(
    base_img: np.ndarray,
    t_norm: float,
    vel_frames: np.ndarray,
    zoom_rect: tuple[int, int, int, int],
    vel_ylim: tuple[float, float],
    zoom_half_w: float = 0.05,
) -> np.ndarray:
    """Redraw the zoomed velocity panel for the current playhead position."""
    x0, y0, x1, y1 = zoom_rect
    pw = x1 - x0
    ph = y1 - y0
    if pw <= 0 or ph <= 0:
        return base_img

    img = base_img  # already a copy from draw_scrubber
    img[y0:y1, x0:x1] = _DATA_BG_BGR

    v_lo, v_hi = vel_ylim
    v_range = v_hi - v_lo if v_hi > v_lo else 1.0

    def _vy(v):
        frac = (np.clip(float(v), v_lo, v_hi) - v_lo) / v_range
        return int(np.clip(y1 - frac * ph, y0, y1 - 1))

    def _vx(t):
        frac = (t - (t_norm - zoom_half_w)) / (2.0 * zoom_half_w)
        return int(np.clip(x0 + frac * pw, x0, x1 - 1))

    # Zero line
    zy = _vy(0.0)
    cv2.line(img, (x0, zy), (x1, zy), _ZERO_BGR, 1, cv2.LINE_AA)

    # Current-position cursor
    cx = (x0 + x1) // 2
    cv2.line(img, (cx, y0), (cx, y1), (0, 102, 230), 2, cv2.LINE_AA)

    # Collect pixel points for visible window
    n = len(vel_frames)
    t_lo = t_norm - zoom_half_w
    t_hi = t_norm + zoom_half_w
    idx_lo = max(0, int(np.floor(t_lo * (n - 1))))
    idx_hi = min(n - 1, int(np.ceil(t_hi * (n - 1))))

    if idx_hi <= idx_lo:
        return img

    pts = []
    for idx in range(idx_lo, idx_hi + 1):
        t_abs = idx / max(n - 1, 1)
        pts.append((_vx(t_abs), _vy(vel_frames[idx])))

    if len(pts) < 2:
        return img

    # Filled areas (positive/negative)
    overlay = img[y0:y1, x0:x1].copy()
    c_pos = hex_to_bgr(C_VEL_POS)
    c_neg = hex_to_bgr(C_VEL_NEG)
    for i in range(len(pts) - 1):
        px0, py0 = pts[i]
        px1, py1 = pts[i + 1]
        vi = float(vel_frames[idx_lo + i])
        vi1 = float(vel_frames[idx_lo + i + 1])
        zero_y_l = _vy(0.0)
        poly = np.array([
            [px0 - x0, py0 - y0], [px1 - x0, py1 - y0],
            [px1 - x0, zero_y_l - y0], [px0 - x0, zero_y_l - y0],
        ], dtype=np.int32)
        fill_col = c_pos if (vi + vi1) >= 0 else c_neg
        cv2.fillPoly(overlay, [poly], fill_col)
    cv2.addWeighted(overlay, 0.35, img[y0:y1, x0:x1], 0.65, 0, img[y0:y1, x0:x1])

    # Velocity line
    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_arr], False, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def compose_frame(video_frame_bgr: np.ndarray, plot_img_rgb: np.ndarray,
                  target_h: int) -> np.ndarray:
    """Horizontally stack resized video frame and plot image."""
    h_v, w_v = video_frame_bgr.shape[:2]
    w_new = int(w_v * target_h / h_v)
    vid_res = cv2.resize(video_frame_bgr, (w_new, target_h))
    plot_bgr = cv2.cvtColor(plot_img_rgb, cv2.COLOR_RGB2BGR)
    plot_bgr = cv2.resize(plot_bgr, (plot_bgr.shape[1], target_h))
    return np.hstack([vid_res, plot_bgr])
