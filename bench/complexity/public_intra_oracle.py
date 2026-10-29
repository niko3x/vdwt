#!/usr/bin/env python3
"""The directional intra oracle: how much could a directional intra coder save?

The published gap is in the intra frame (x265's I-frame is about half VDWT's at equal
PSNR), and the project's own plan (T2a) is an "LL-conditioned directional intra oracle".
This measures that oracle's upper bound without touching the codec.

The trick: an intra frame codes `source - prediction`. The current prediction is flat
(DC), so the coded residual is `source - 128`. If the prediction were the best of a few
directional predictions, the coded residual would be `source - prediction`. Feeding the
codec the image `residual + 128` therefore produces exactly the bitstream a directional
intra coder would produce for that residual -- and the codec needs no change.

For every block the oracle picks the direction with the smallest residual energy, using
the source pixels it would really have (the already reconstructed neighbours). The
direction is not signalled: this is the upper bound of any directional scheme, block-free
or not.

    public_intra_oracle.py [--qp 42,50,58] [--block 8,16] [--clips name=path,...]
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

CLIPS = [
    ("fourpeople", ROOT / "videos/public/fourpeople_720p60.y4m"),
    ("stockholm", ROOT / "videos/public/stockholm_720p60.y4m"),
    ("parkrun", ROOT / "videos/public/parkrun_720p50.y4m"),
    ("shields", ROOT / "videos/public/shields_720p50.y4m"),
    ("aerial", ROOT / "videos/public/aerial_720p60.y4m"),
]


def read_y4m(path, n=1):
    f = open(path, "rb")
    hdr = f.readline().decode().split()
    w = int([t for t in hdr if t.startswith("W")][0][1:])
    h = int([t for t in hdr if t.startswith("H")][0][1:])
    fs = w * h * 3 // 2
    frames = []
    for _ in range(n):
        f.readline()
        buf = f.read(fs)
        if len(buf) < fs:
            break
        y = np.frombuffer(buf[: w * h], dtype=np.uint8).reshape(h, w)
        cw, ch = w // 2, h // 2
        u = np.frombuffer(buf[w * h : w * h + cw * ch], dtype=np.uint8).reshape(ch, cw)
        v = np.frombuffer(buf[w * h + cw * ch :], dtype=np.uint8).reshape(ch, cw)
        frames.append((y, u, v))
    return frames


def write_y4m(path, frames, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F60:1 Ip A1:1 C420jpeg\n".encode())
        for y, u, v in frames:
            f.write(b"FRAME\n")
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())


def shift_right(a):
    """pred[x, y] = a[x-1, y] (the left neighbour)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1:] = a[:, :-1]
    return out


def shift_down(a):
    out = np.empty_like(a)
    out[0, :] = a[0, :]
    out[1:, :] = a[:-1, :]
    return out


def shift_diag(a):
    """pred[x, y] = a[x-1, y-1]."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 0]
    out[0, :] = a[0, :]
    out[1:, 1:] = a[:-1, :-1]
    return out


def shift_anti(a):
    """pred[x, y] = a[x+1, y-1]."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 0]
    out[0, :] = a[0, :]
    out[1:, :-1] = a[:-1, 1:]
    return out


CANDIDATES = [("left", shift_right), ("up", shift_down), ("diag", shift_diag),
              ("anti", shift_anti)]


def global_prediction(plane, err=None):
    """One direction for the whole plane: the cheapest possible direction field."""
    a = plane.astype(np.float64)
    ref = plane if err is None else np.clip(plane.astype(np.float64) + err, 0, 255)
    best = None
    for name, fn in CANDIDATES:
        cand = fn(ref).astype(np.float64)
        err = float(((a - cand) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, cand, name)
    dc = float(a.mean())
    if ((a - dc) ** 2).sum() < best[0]:
        return np.full_like(a, dc), "dc"
    return best[1], best[2]


def smooth_error(h, w, sigma, seed=7):
    """A smooth random error field, like the codec's own reconstruction error."""
    if sigma <= 0:
        return np.zeros((h, w))
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w))
    k = max(3, int(sigma * 4) | 1)
    ker = np.hanning(k)
    ker /= ker.sum()

    def conv(a, axis):
        if axis == 0:
            return np.stack([np.convolve(a[:, j], ker, mode="same")
                             for j in range(a.shape[1])], axis=1)
        return np.stack([np.convolve(a[i, :], ker, mode="same")
                         for i in range(a.shape[0])], axis=0)

    n = conv(conv(n, 0), 1)
    rms = float(np.sqrt((n ** 2).mean()))
    return n * (sigma / rms) if rms > 0 else n


def oracle_prediction(plane, block, err=None):
    """The best directional prediction per block of the plane."""
    h, w = plane.shape
    ref = plane if err is None else np.clip(plane.astype(np.float64) + err, 0, 255)
    pred = np.zeros_like(plane, dtype=np.float64)
    for by in range(0, h, block):
        for bx in range(0, w, block):
            b = plane[by : by + block, bx : bx + block].astype(np.float64)
            best = None
            for _, fn in CANDIDATES:
                cand = fn(ref)[by : by + block, bx : bx + block].astype(np.float64)
                e = float(((b - cand) ** 2).sum())
                if best is None or e < best[0]:
                    best = (e, cand)
            # The flat prediction of the block itself, as the DC candidate.
            dc = float(b.mean())
            err_dc = float(((b - dc) ** 2).sum())
            if err_dc < best[0]:
                pred[by : by + block, bx : bx + block] = dc
            else:
                pred[by : by + block, bx : bx + block] = best[1]
    return pred


def encoded_bytes(y4m, qp, tmp):
    out = tmp / "oracle.vdwt"
    subprocess.run([str(BIN), "encode", str(y4m), str(out), "--qp", str(qp), "--frames", "1",
                    "--threads", "2", "--intra-only"], check=True, capture_output=True)
    return out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qp", default="42,50,58")
    ap.add_argument("--block", default="8,16,32")
    ap.add_argument("--clips", default="")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="rms of the reconstruction error in the prediction's neighbours")
    args = ap.parse_args()

    clips = CLIPS
    if args.clips:
        clips = [(n, Path(p)) for n, p in (c.split("=", 1) for c in args.clips.split(","))]
    qps = [int(q) for q in args.qp.split(",")]
    blocks = [int(b) for b in args.block.split(",")]

    tmp = Path(tempfile.mkdtemp())
    print(f"{'clip':12s} {'qp':>3s} {'blk':>4s} {'DC':>8s} {'oracle':>8s} {'intra -':>8s} "
          f"{'intra%':>7s} {'stream':>8s} {'all -':>7s} {'signal':>7s} {'net -':>7s}")
    for clip, src in clips:
        if not src.exists():
            continue
        frames = read_y4m(src, 1)
        y, u, v = frames[0]
        h, w = y.shape
        for qp in qps:
            base = tmp / f"{clip}_{qp}_base.y4m"
            write_y4m(base, frames, w, h)
            base_bytes = encoded_bytes(base, qp, tmp)
            for block in blocks:
                planes = []
                if block == 0:
                    # One direction for the whole plane (2 bits per frame per component).
                    for p in (y, u, v):
                        e = smooth_error(p.shape[0], p.shape[1], args.noise)
                        pred, _ = global_prediction(p, e)
                        r = p.astype(np.float64) - pred
                        planes.append(np.clip(r + 128, 0, 255).astype(np.uint8))
                else:
                    for p in (y, u, v):
                        e = smooth_error(p.shape[0], p.shape[1], args.noise)
                        pred = oracle_prediction(p, block, e)
                        r = p.astype(np.float64) - pred
                        planes.append(np.clip(r + 128, 0, 255).astype(np.uint8))
                orc = tmp / f"{clip}_{qp}_orc{block}.y4m"
                write_y4m(orc, [(planes[0], planes[1], planes[2])], w, h)
                orc_bytes = encoded_bytes(orc, qp, tmp)
                # The whole stream at the same qp, and the intra frame's share of it.
                full = tmp / f"{clip}_{qp}_full.vdwt"
                subprocess.run([str(BIN), "encode", str(src), str(full), "--qp", str(qp),
                                "--frames", "20", "--threads", "2"],
                               check=True, capture_output=True)
                stream = full.stat().st_size
                stats = subprocess.run([str(BIN), "band-stats", str(src), "--qp", str(qp),
                                        "--frames", "1"], check=True, capture_output=True,
                                       text=True).stdout.splitlines()
                intra = sum(int(l.split(",")[9]) for l in stats[1:] if l.split(",")[1] == "I")
                intra += 64  # the motion section and the headers of the first frame
                share = 100.0 * intra / stream
                all_gain = 100.0 * (base_bytes - orc_bytes) / stream
                # A real coder must signal the direction: 2 bits per luma block of the
                # field's granularity (a coarse field; the chroma inherits the luma's).
                if block == 0:
                    sig = 6  # two bits per component, three components
                else:
                    nb = ((w + block - 1) // block) * ((h + block - 1) // block)
                    sig = (nb * 2 + 7) // 8
                net = base_bytes - orc_bytes - sig
                print(f"{clip:12s} {qp:3d} {block:4d} {base_bytes:8d} {orc_bytes:8d} "
                      f"{100*(base_bytes-orc_bytes)/base_bytes:7.1f}% {share:6.1f}% "
                      f"{stream:8d} {all_gain:6.2f}% {sig:6d} {100*net/stream:6.2f}%")
    print("\nThe oracle is not signalled: it is the upper bound of any directional intra "
          "coder,\nblock-free or not. The residual is clipped to the codec's 8-bit range, "
          "which costs the\noracle a little where the directional prediction fails badly.")


if __name__ == "__main__":
    main()
