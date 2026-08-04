#!/usr/bin/env python3
"""Render direction-ambiguity windows as synced animations.

Two clips per corpus:
  anim_ambiguity_windows_<c>.mp4  2 CLEAR above 2 AMBIGUOUS (sign-flip failures)
  anim_late_lowv_<c>.mp4          4 windows from the LAST 30% of episodes with the
                                  lowest v_fwd -- the population behind the
                                  late-episode dense-velocity degradation.

Each panel: exact model input, all 32 window frames at 10 fps, amber border on
the START frame, header carrying v_fwd / v_rev / whether the sign flips.
Encoded H.264 via ffmpeg (cv2's mp4v does not play in browsers). Writes
manifest.json so figures are reproducible rather than re-sampled.
"""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, "/home/justinyu/WARP-RM")
from scripts.config import default_feature_cache_dir
from scripts.eval.render import load_checkpoint
from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.data.video_reader import read_frames
from warp_rm.data.preprocess import resize_frame
from warp_rm.utils.caching import _ep_cache_path

N, TILE, FPS = 32, 224, 10
OUT = Path("/home/justinyu/WARP-RM_analysis")
CORPORA = {
 "bike": dict(repo="/home/justinyu/Downloads/PlaceBikeRotorToolOnRotor",
   cam="observation.images.top", fs=1, sss=15, crop="center", shortest=0.5,
   min_frames=471, exclude={15,16,30,32},
   ckpt="checkpoints/best_model_warp_rm_bike_rotor_fs1_sss15_win32_center_no_abs.pt"),
 "tshirt": dict(repo="/home/justinyu/data/lerobot/hlm_plus_d405_singlefold_gop10",
   cam="top_camera-images-rgb", fs=3, sss=45, crop="squash", shortest=0.25,
   ckpt="checkpoints/best_model_warp_rm_full_hlm_plus_d405_singlefold_gop10_shortest25_win32_15k_full.pt"),
}
AMBER = (255, 191, 0)

def header(w, txt, col, h=30):
    im = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(im, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    return im

def panel(frames, j, meta, crop):
    """One panel at animation step j."""
    img = resize_frame(frames[j], TILE, crop).copy()
    if j == 0:                                   # amber START border
        cv2.rectangle(img, (0, 0), (TILE - 1, TILE - 1), AMBER, 4)
        cv2.putText(img, "START", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 2, cv2.LINE_AA)
    cv2.putText(img, f"{j+1}/{N}", (TILE - 52, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (235, 235, 235), 1, cv2.LINE_AA)
    col = (140, 255, 170) if not meta["ambiguous"] else (255, 150, 150)
    h1 = header(TILE, f"{meta['kind']}  ep{meta['episode']} @{meta['start']}", col)
    h2 = header(TILE, f"v_fwd={meta['v_fwd']:+.3f} v_rev={meta['v_rev']:+.3f} "
                      f"flip={'NO' if meta['ambiguous'] else 'YES'}", col, 26)
    return np.vstack([h1, h2, img])

def encode(panels_frames, path):
    tmp = Path(tempfile.mkdtemp())
    for j in range(N):
        cells = [p[j] for p in panels_frames]
        top = np.hstack(cells[:2]); bot = np.hstack(cells[2:4])
        g = np.vstack([top, np.full((6, top.shape[1], 3), 70, np.uint8), bot])
        cv2.imwrite(str(tmp / f"{j:04d}.png"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmp / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmp / "%04d.png"), "-vf", "scale=640:-1:flags=lanczos",
                    str(path.with_suffix(".gif"))], check=False)
    shutil.rmtree(tmp, ignore_errors=True)

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {}
    for name, c in CORPORA.items():
        step = c["sss"] // c["fs"]; span = (N - 1) * step
        eps = discover_lerobot_episodes(c["repo"], camera_key=c["cam"], require_video=True)
        if c.get("exclude"): eps = [e for e in eps if e.episode_index not in c["exclude"]]
        if c.get("min_frames"): eps = [e for e in eps if e.n_frames >= c["min_frames"]]
        eps = sorted(eps, key=lambda e: e.n_frames)[:max(1, round(c["shortest"] * len(eps)))]
        cd = default_feature_cache_dir("dinov3", c["fs"])
        pool = [(e, p) for e in eps
                for p in [_ep_cache_path(cd, e, "dinov3", c["fs"], None, c["crop"])] if p.exists()]
        model, _ = load_checkpoint(c["ckpt"], dev)
        rng = np.random.default_rng(0); cand = []
        for e, p in [pool[i] for i in rng.permutation(len(pool))[:120]]:
            f = np.load(p)
            if len(f) < span + 1: continue
            n_start = len(f) - span
            for _ in range(14):
                s = int(rng.integers(0, n_start))
                cand.append((e, s, f[s:s + span + 1:step][:N], s / max(1, n_start)))
        W = np.stack([w for _, _, w, _ in cand]).astype(np.float32)
        with torch.no_grad():
            vv = []
            for arr in (W, W[:, ::-1].copy()):
                a = []
                for i in range(0, len(arr), 512):
                    o = model(torch.from_numpy(arr[i:i + 512]).to(dev))
                    pr = (o[0] if isinstance(o, tuple) else o).cpu().numpy()
                    a.append(pr[:, -1] - pr[:, 0])
                vv.append(np.concatenate(a))
        vf, vr = vv; amb = np.sign(vr) == np.sign(vf)
        pos = np.array([q for _, _, _, q in cand])
        order = np.argsort(np.abs(vf))

        sets = {
          "anim_ambiguity_windows": (
            [("CLEAR", i) for i in order[::-1] if not amb[i]][:2] +
            [("AMBIGUOUS", i) for i in order if amb[i]][:2]),
          "anim_late_lowv": (
            [("LATE lowest v", i) for i in np.argsort(vf) if pos[i] > 0.70][:4]),
        }
        for tag, picks in sets.items():
            picks = [p for p in picks if p is not None][:4]
            if len(picks) < 4:
                picks += [("CLEAR", i) for i in order[::-1] if not amb[i]][:4 - len(picks)]
            pf, man = [], []
            for kind, i in picks:
                e, s, _, q = cand[i]
                idxs = [e.frame_offset + min(s * c["fs"] + j * step * c["fs"], e.n_frames - 1)
                        for j in range(N)]
                frames = read_frames(e.video_path, idxs)
                meta = dict(kind=kind, episode=int(e.episode_index), start=int(s),
                            pos_in_episode=round(float(q), 3), v_fwd=float(vf[i]),
                            v_rev=float(vr[i]), ambiguous=bool(amb[i]))
                pf.append([panel(frames, j, meta, c["crop"]) for j in range(N)])
                man.append(meta)
            out = OUT / f"{tag}_{name}.mp4"
            encode(pf, out)
            manifest[f"{tag}_{name}"] = man
            print(f"  wrote {out.name}  ({', '.join(m['kind'] for m in man)})")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
