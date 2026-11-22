#!/usr/bin/env python3
"""Prediction oracle (decoder roadmap, Phase 0): how much would a better P-frame predictor save?

Runs `vdwt pred-dump` (each frame's motion-compensated prediction) and `vdwt encode --recon`
at the same qp and options, then applies ideal versions of candidate prediction tools to the
prediction offline and prices them:

  gain-offset    per-frame, per-plane gain and offset fitted to the source (roadmap 3.1)
  offset-field   a smooth luma offset field: one offset per 64x64 region, bilinear between
                 region centres, so no block edges (3.1, local)
  gain-field     the same with gain and offset per region
  best-of-2      control: per 16x16 region the better of the prediction and the previous
                 source frame (an upper bound for a second hypothesis that needs no motion)
  dense-8        upper bound for any finer or richer motion model (3.2, 3.3): per 8x8 block
                 the better of the prediction and the best whole-pel match in the previous
                 reconstruction within +-R pixels, side information ignored (--dense R; slow)

Bits are modelled per 8x8 block by reverse water-filling at the frame's real distortion D
(the recon's MSE): a block whose residual variance s2 exceeds D costs 32*log2(s2/D). The
model's absolute scale is poor at low rates, so only its relative change is used: "% stream"
is that change applied to the P-frames' real band bytes (from the frame log), plus the
oracle's side information, over the whole stream. It is an upper-bound style estimate (ideal
fits on the source), so an idea that cannot clear the kill line here is dropped without
writing codec code.

    tools/pred_oracle.py CLIP [--qp 40] [--frames 60] [--opts=--tier=medium]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import read_y4m, scratch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def block_bits(res, d):
    """Water-filling bits of residual `res` (2-D float) at distortion `d`, per 8x8 block."""
    h, w = (res.shape[0] // 8) * 8, (res.shape[1] // 8) * 8
    b = res[:h, :w].reshape(h // 8, 8, w // 8, 8)
    var = b.var(axis=(1, 3)) + b.mean(axis=(1, 3)) ** 2 * 0.0  # mean is coded by LL anyway
    var = np.maximum(var, 1e-9)
    return float(np.sum(np.where(var > d, 32 * np.log2(var / d), 0.0)))


def fit_gain_offset(p, s):
    p, s = p.ravel().astype(np.float64), s.ravel().astype(np.float64)
    vp = p.var()
    a = 1.0 if vp < 1e-6 else ((p - p.mean()) * (s - s.mean())).mean() / vp
    a = min(max(a, 0.5), 2.0)
    return a, s.mean() - a * p.mean()


def region_field(p, s, size, gain):
    """Per-region (gain,) offset fitted to the source, bilinear between region centres."""
    h, w = p.shape
    ny, nx = math.ceil(h / size), math.ceil(w / size)
    A, B = np.ones((ny, nx)), np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            pp = p[j * size:(j + 1) * size, i * size:(i + 1) * size]
            ss = s[j * size:(j + 1) * size, i * size:(i + 1) * size]
            if gain:
                A[j, i], B[j, i] = fit_gain_offset(pp, ss)
            else:
                B[j, i] = ss.mean() - pp.mean()
    # Quantize as a codec would: offsets to 1/2, gains to 1/64.
    B = np.round(B * 2) / 2
    A = np.round(A * 64) / 64

    def upsample(f):
        yc = (np.arange(h) + 0.5) / size - 0.5
        xc = (np.arange(w) + 0.5) / size - 0.5
        y0 = np.clip(np.floor(yc).astype(int), 0, ny - 1)
        x0 = np.clip(np.floor(xc).astype(int), 0, nx - 1)
        y1, x1 = np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1)
        fy = np.clip(yc - y0, 0, 1)[:, None]
        fx = np.clip(xc - x0, 0, 1)[None, :]
        top = f[y0][:, x0] * (1 - fx) + f[y0][:, x1] * fx
        bot = f[y1][:, x0] * (1 - fx) + f[y1][:, x1] * fx
        return top * (1 - fy) + bot * fy

    side_bits = ny * nx * (10 if gain else 4)
    return upsample(A) * p + upsample(B), side_bits


def best_of_two(p, prev, s, size=16):
    out = p.copy()
    h, w = p.shape
    for y in range(0, h, size):
        for x in range(0, w, size):
            sl = (slice(y, y + size), slice(x, x + size))
            if np.mean((prev[sl] - s[sl]) ** 2) < np.mean((p[sl] - s[sl]) ** 2):
                out[sl] = prev[sl]
    side = math.ceil(h / size) * math.ceil(w / size) * 0.5
    return out, side


def dense_match(p, ref, s, r):
    """Per 8x8 block, the better of `p` and the best integer shift of `ref` within +-r."""
    h, w = (s.shape[0] // 8) * 8, (s.shape[1] // 8) * 8
    s, p = s[:h, :w], p[:h, :w]
    pad = np.pad(ref, r, mode="edge")

    def bsse(x):
        return ((x - s) ** 2).reshape(h // 8, 8, w // 8, 8).sum(axis=(1, 3))

    best = bsse(p)
    out = p.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cand = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            e = bsse(cand)
            better = e < best
            if better.any():
                best = np.where(better, e, best)
                m = np.repeat(np.repeat(better, 8, 0), 8, 1)
                out = np.where(m, cand, out)
    return out, 0


def psnr(mse):
    return 99.0 if mse <= 0 else 10 * math.log10(255 * 255 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--qp", type=int, default=40)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--opts", default="")
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/vdwt"))
    ap.add_argument("--dense", type=int, default=0, help="dense-8 search range (0: off)")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="oracle_", dir=scratch())
    pred, rec, stream, log = (os.path.join(work, n) for n in ("p.y4m", "r.y4m", "s.vdwt", "f.csv"))
    opts = args.opts.split()
    subprocess.run([args.bin, "pred-dump", args.clip, pred, "--qp", str(args.qp), "--frames",
                    str(args.frames)] + opts, check=True, capture_output=True)
    subprocess.run([args.bin, "encode", args.clip, stream, "--qp", str(args.qp), "--frames",
                    str(args.frames), "--recon", rec, "--frame-log", log] + opts, check=True,
                   capture_output=True)
    import csv
    with open(log) as f:
        rows = list(csv.DictReader(f))
    p_band_bits = 8 * sum(
        int(r["bytes"]) - sum(int(r[k] or 0) for k in
                              ("motion_bytes", "header_bytes", "layout_bytes", "loop_bytes"))
        for r in rows if r["type"] not in ("I", "R"))
    src, _, _ = read_y4m(args.clip, args.frames)
    prd, _, _ = read_y4m(pred, args.frames)
    dec, _, _ = read_y4m(rec, args.frames)
    stream_bits = 8 * os.path.getsize(stream)
    for f in (pred, rec, stream, log):
        os.unlink(f)
    os.rmdir(work)

    names = ["baseline", "gain-offset", "offset-field", "gain-field", "best-of-2"]
    if args.dense:
        names.append("dense-8")
    bits = {n: 0.0 for n in names}
    mse = {n: 0.0 for n in names}
    side = {n: 0.0 for n in names}
    frames = 0
    for t in range(1, min(len(src), len(prd), len(dec))):
        p = [prd[t][c].astype(np.float64) for c in range(3)]
        if all(np.all(p[c] == 128) for c in range(3)):
            continue  # intra frame
        s = [src[t][c].astype(np.float64) for c in range(3)]
        d = [max(np.mean((dec[t][c].astype(np.float64) - s[c]) ** 2), 0.25) for c in range(3)]
        prev = [dec[t - 1][c].astype(np.float64) for c in range(3)]
        frames += 1
        for c in range(3):
            cands = {"baseline": (p[c], 0)}
            a, b = fit_gain_offset(p[c], s[c])
            cands["gain-offset"] = (a * p[c] + b, 24)
            if c == 0:
                cands["offset-field"] = region_field(p[c], s[c], 64, False)
                cands["gain-field"] = region_field(p[c], s[c], 64, True)
                cands["best-of-2"] = best_of_two(p[c], prev[c], s[c])
                if args.dense:
                    q, sb = dense_match(p[c], prev[c], s[c], args.dense)
                    hh, ww = q.shape
                    cands["dense-8"] = (np.pad(q, ((0, p[c].shape[0] - hh), (0, p[c].shape[1] - ww)), mode="edge"), sb)
            else:
                cands["offset-field"] = cands["gain-field"] = cands["best-of-2"] = (p[c], 0)
                if args.dense:
                    cands["dense-8"] = (p[c], 0)
            for n, (q, sb) in cands.items():
                r = s[c] - q
                bits[n] += block_bits(r, d[c])
                mse[n] += np.mean(r ** 2) * (6 if c == 0 else 1) / 8
                side[n] += sb
    if not frames:
        print("no P-frames")
        return
    base = bits["baseline"]
    print(f"{args.clip} qp {args.qp} {args.opts}: {frames} P-frames, stream {stream_bits / 8:.0f} B,"
          f" P-frame bands {p_band_bits / 8:.0f} B ({100 * p_band_bits / stream_bits:.0f}%)")
    print(f"{'oracle':14}{'pred PSNR611':>13}{'Δ dB':>7}{'Δ bytes':>10}{'side':>8}{'% model':>9}{'% stream':>10}")
    for n in names:
        net = bits[n] - base + side[n]
        print(f"{n:14}{psnr(mse[n] / frames):13.2f}"
              f"{psnr(mse[n] / frames) - psnr(mse['baseline'] / frames):7.2f}"
              f"{(bits[n] - base) / 8:10.0f}{side[n] / 8:8.0f}{100 * net / base:8.2f}%"
              f"{100 * ((bits[n] - base) / base * p_band_bits + side[n]) / stream_bits:9.2f}%")


if __name__ == "__main__":
    main()
