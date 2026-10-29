#!/usr/bin/env python3
"""Compare VDWT encoder settings against a baseline on one clip.

    tools/sweep.py CLIP.y4m BASELINE_OPTS VARIANT_OPTS... [--frames 5] [--qp 24,32,40,48,56]

Each OPTS is one quoted string of `vdwt rd` options ("" for defaults). For every variant it prints
the size change at equal qp and the BD-PSNR (luma and 6:1:1 YUV): the average PSNR gain at equal
bitrate, interpolated piecewise-linearly in log rate over the range both curves cover.
Positive BD-PSNR and negative size change are better.
"""
import math
import os
import subprocess
import sys

VDWT = os.environ.get("VDWT", "target/release/vdwt")


def rd(clip, opts, frames, qps):
    cmd = [VDWT, "rd", clip, "--frames", str(frames), "--qp", qps] + opts.split()
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    points = []
    for line in out.splitlines():
        f = line.split()
        if len(f) == 9 and f[0].isdigit():
            points.append((float(f[1]), float(f[3]), float(f[6])))  # kbps, PSNR-Y, PSNR-YUV
    return points


def bd_psnr(a, b, col):
    """Mean PSNR difference b - a at equal log-rate (piecewise-linear interpolation)."""
    ca = sorted((math.log(p[0]), p[col]) for p in a)
    cb = sorted((math.log(p[0]), p[col]) for p in b)
    lo, hi = max(ca[0][0], cb[0][0]), min(ca[-1][0], cb[-1][0])
    if hi <= lo:
        return float("nan")

    def interp(c, x):
        for (x0, y0), (x1, y1) in zip(c, c[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
        return c[-1][1]

    n = 200
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    return sum(interp(cb, x) - interp(ca, x) for x in xs) / len(xs)


def main():
    args = sys.argv[1:]
    frames, qps = 5, "24,32,40,48,56"
    if "--frames" in args:
        i = args.index("--frames")
        frames = int(args[i + 1])
        del args[i : i + 2]
    if "--qp" in args:
        i = args.index("--qp")
        qps = args[i + 1]
        del args[i : i + 2]
    clip, base, *variants = args
    ref = rd(clip, base, frames, qps)
    print(f"baseline [{base}]: " + ", ".join(f"{r:.0f} kbps/{y:.2f} dB" for r, y, _ in ref))
    for v in variants:
        pts = rd(clip, v, frames, qps)
        size = 100.0 * (sum(p[0] for p in pts) / sum(p[0] for p in ref) - 1.0)
        print(
            f"{v or '(defaults)':40s} size {size:+6.2f}%   "
            f"BD-PSNR Y {bd_psnr(ref, pts, 1):+6.3f} dB   YUV {bd_psnr(ref, pts, 2):+6.3f} dB"
        )


if __name__ == "__main__":
    main()
