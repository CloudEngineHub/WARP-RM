#!/usr/bin/env python3
"""Full-episode rollouts in the inspector's layout, as standalone mp4.

Camera frame beside reconstructed progress / per-frame velocity / WARP-BC weight,
with a synced playhead. Palette matches scripts/webui/static/style.css.

Three things this handles that a naive renderer gets wrong:
  * LeRobot v3.0 frame offsets — episodes live at an offset inside a packed shard,
    so reading ep.video_path from frame 0 shows the WRONG episode's footage under
    the right episode's traces.
  * H.264 output — cv2's mp4v does not play in browsers.
  * Split provenance — every clip is stamped TRAIN / HELD-OUT (val) / UNSEEN, and
    the trainer's clean-val split is reproduced exactly (seed 42), because a
    memorised curve and a generalised one look identical otherwise.

Only the SUPERVISED head is drawn: has_abs_progress_head() reports False for a
no_abs checkpoint, so the untrained abs head's flat ~0.5 line is never plotted.
"""
import argparse, json, random, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "/home/justinyu/WARP-RM")
from scripts.config import default_feature_cache_dir
from scripts.eval.render import load_checkpoint
from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.data.video_reader import read_frames
from warp_rm.utils.caching import _ep_cache_path
from warp_rm.visualization.inference import (
    dense_inference_relative, has_abs_progress_head, _plan_episode_windows)

BG, FG, GRID, PROG, CUR = "#0D0D1A", "#EAEAEA", "#222244", "#9B59B6", "#F4D35E"
N = 32

def ramp(t):
    t = np.clip(t, 0, 1)
    out = np.zeros((len(t), 3))
    lo = t < 0.5
    out[lo] = np.stack([np.ones(lo.sum()), t[lo]*1.6, np.full(lo.sum(), .098)], 1)
    s = (t[~lo]-.5)*2
    out[~lo] = np.stack([1-s*.82, .8+s*.1, .1+s*.2], 1)
    return np.clip(out, 0, 1)

def plot_panel(prog, vel, wt, W, H, note):
    fig, ax = plt.subplots(3, 1, figsize=(W/100, H/100), dpi=100,
                           facecolor=BG, gridspec_kw=dict(hspace=.42))
    x = np.linspace(0, 1, len(prog))
    for a in ax:
        a.set_facecolor(BG); a.grid(color=GRID, lw=.5)
        for sp in a.spines.values(): sp.set_color("#444466")
        a.tick_params(colors="#C5C5D4", labelsize=7); a.set_xlim(0, 1)
    ax[0].plot(x, prog, color=PROG, lw=1.8); ax[0].set_ylim(0, 1)
    ax[0].set_title("reconstructed progress", color=PROG, fontsize=9, loc="left")
    lo, hi = min(vel.min(), 0)-.08, max(vel.max(), .1)+.08
    pts = np.array([x, vel]).T.reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1),
                        colors=ramp(np.clip(wt[:-1], 0, 1)), lw=1.5)
    ax[1].add_collection(lc); ax[1].axhline(0, color="#C5C5D4", lw=1.2)
    ax[1].set_ylim(lo, hi)
    ax[1].set_title("per-frame velocity (signed) — coloured by WARP-BC weight",
                    color=FG, fontsize=9, loc="left")
    ax[2].fill_between(x, 0, wt, color="#3f8c59", alpha=.75, lw=0)
    ax[2].set_ylim(0, max(1.05, wt.max()*1.05))
    ax[2].set_title("WARP-BC weight = max(0, v)", color=FG, fontsize=9, loc="left")
    if note: fig.text(.01, .003, note, color="#8A8A9E", fontsize=6.5)
    fig.tight_layout(pad=.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img

def render(ep, feat, model, dev, step, out, split, label, fps=15, target=400):
    n_feat = len(feat)
    all_fi, used, vscale = _plan_episode_windows(n_feat, N, step)
    prog_f, vel_f = dense_inference_relative(model, feat, dev, window_size=N,
                                             standard_feat_steps=step)
    n = ep.n_frames
    t_src, t_feat = np.linspace(0, 1, n), np.linspace(0, 1, len(prog_f))
    prog = np.interp(t_src, t_feat, prog_f).astype(np.float32)
    vel = np.interp(t_src, t_feat, vel_f).astype(np.float32)
    wt = np.clip(vel, 0, None)
    note = (f"fs=1 sss=15 win=32 step={step}"
            + (f"  SHORT-EPISODE FALLBACK step {step}->{used} vel_scale={vscale:.3f}"
               if used != step else "")
            + f"  windows={len(all_fi)}")
    PH, PW = 560, 680
    panel = plot_panel(prog, vel, wt, PW, PH, note)
    stride = max(1, n // target)
    idxs = list(range(0, n, stride))
    frames = read_frames(ep.video_path, [ep.frame_offset + i for i in idxs])
    tmp = Path(tempfile.mkdtemp())
    for k, (i, fr) in enumerate(zip(idxs, frames)):
        h, w = fr.shape[:2]
        cam = cv2.resize(fr, (PW, int(PW*h/w)), interpolation=cv2.INTER_AREA)
        left = np.full((PH, PW, 3), 13, np.uint8)
        y0 = 46; left[y0:y0+cam.shape[0]] = cam[:PH-y0]
        col = {"TRAIN": (255,170,90), "HELD-OUT (val)": (140,200,255)}.get(split, (140,255,170))
        cv2.putText(left, f"ep{ep.episode_index}  {split}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1, cv2.LINE_AA)
        cv2.putText(left, f"{label}" if label else "", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .44, (200,200,215), 1, cv2.LINE_AA)
        cv2.putText(left, f"frame {i}/{n-1}   v={vel[i]:+.3f}  w={wt[i]:.3f}  p={prog[i]:.3f}",
                    (8, PH-10), cv2.FONT_HERSHEY_SIMPLEX, .44, (235,235,235), 1, cv2.LINE_AA)
        p = panel.copy()
        # playhead across all three axes (axes span ~9%..97% of panel width)
        px = int(0.092*PW + (i/max(1, n-1))*(0.965-0.092)*PW)
        cv2.line(p, (px, 30), (px, PH-24), (244, 211, 94), 1)
        cv2.imwrite(str(tmp/f"{k:05d}.png"), cv2.cvtColor(np.hstack([left, p]), cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),
                    "-i",str(tmp/"%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(episode=int(ep.episode_index), split=split, label=label,
                n_frames=int(n), step=int(step), step_used=int(used),
                vel_scale=float(vscale), n_windows=len(all_fi),
                mean_v=float(vel.mean()), min_v=float(vel.min()),
                frac_negative=float((vel<0).mean()))

LABELS = {30:"hand in frame, OOD", 40:"clean", 81:"regression at a failure",
          84:"clean", 134:"decisive segments towards the end"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best_model_warp_rm_bike_rotor_fs1_sss15_win32_center_no_abs.pt")
    ap.add_argument("--out", default="/home/justinyu/WARP-RM_rollouts")
    a = ap.parse_args()
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DS = "/home/justinyu/Downloads/PlaceBikeRotorToolOnRotor"
    eps = discover_lerobot_episodes(DS, camera_key="observation.images.top", require_video=True)
    by = {e.episode_index: e for e in eps}
    kept = sorted([e for e in eps if e.episode_index not in {15,16,30,32} and e.n_frames>=471],
                  key=lambda e: e.n_frames)[:round(0.5*len([e for e in eps if e.episode_index not in {15,16,30,32} and e.n_frames>=471]))]
    pool = sorted(kept, key=lambda e: e.n_frames)[:max(20, round(.15*len(kept)))]
    sh = pool.copy(); random.Random(42).shuffle(sh)
    val = {e.episode_index for e in sh[:20]}; tr = {e.episode_index for e in kept} - val
    model, ckpt = load_checkpoint(a.ckpt, dev)
    fs, sss = int(ckpt["feature_stride"]), int(ckpt["standard_stride_src"])
    crop, step = ckpt.get("crop_mode","squash"), sss//fs
    print(f"  abs head drawn: {has_abs_progress_head(model)} (no_abs -> False, correct)")
    cd = default_feature_cache_dir("dinov3", fs)
    man = []
    for i in a.episodes:
        e = by[i]
        p = _ep_cache_path(cd, e, "dinov3", fs, None, crop)
        if not p.exists(): print(f"  ep{i}: features missing, skipping"); continue
        split = "TRAIN" if i in tr else ("HELD-OUT (val)" if i in val else "UNSEEN (never trained)")
        r = render(e, np.load(p), model, dev, step, outd/f"ep{i:03d}_{split.split()[0].lower()}.mp4",
                   split, LABELS.get(i,""))
        man.append(r)
        print(f"  ep{i:>3} {split:<22} mean_v={r['mean_v']:+.3f} min_v={r['min_v']:+.3f} "
              f"%v<0={100*r['frac_negative']:.1f}%"
              + ("  [SHORT-EP FALLBACK]" if r['step_used']!=r['step'] else ""))
    (outd/"manifest.json").write_text(json.dumps(man, indent=2))
    print(f"  -> {outd}")

if __name__ == "__main__":
    main()
