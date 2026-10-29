#!/usr/bin/env python3
"""Summarise auto_vs_fixed.txt: BD of auto against fixed band2 2.0, positive = auto better."""
import math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/complexity/auto_vs_fixed.txt"
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]


def interp(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return pts[-1][1]


def bd_rate(ref, test):
    a = sorted((q, math.log(r)) for r, q in ref if r > 0)
    b = sorted((q, math.log(r)) for r, q in test if r > 0)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi - lo < 0.5:
        return None
    xs = [lo + (hi - lo) * i / 200 for i in range(201)]
    diff = [interp(b, x) - interp(a, x) for x in xs]
    return -100 * (math.exp(sum(diff) / len(diff)) - 1)


def main():
    rows = defaultdict(dict)
    strength = defaultdict(list)
    for line in DATA.read_text().splitlines():
        clip, mode, tier, qp, kbps, vmaf, yuv, _, s = line.split()
        rows[(clip, tier)][mode] = rows[(clip, tier)].get(mode, [])
        rows[(clip, tier)][mode].append((float(kbps), float(vmaf), float(yuv)))
        if mode == "auto":
            strength[(clip, tier)].append(int(s) / 8)

    clips = sorted({c for c, _ in rows})
    print(f"{'clip':9s} {'tier':11s} {'auto s':>6s} {'VMAF BD':>8s} {'PSNR BD':>8s} {'hybrid':>8s}")
    per_tier = defaultdict(list)
    per_clip = defaultdict(list)
    for clip in clips:
        for tier in TIERS:
            got = rows.get((clip, tier), {})
            if "band2-2.0" not in got or "auto" not in got:
                continue
            f = got["band2-2.0"]
            a = got["auto"]
            vb = bd_rate([(k, v) for k, v, _ in f], [(k, v) for k, v, _ in a])
            pb = bd_rate([(k, y) for k, _, y in f], [(k, y) for k, _, y in a])
            if vb is None or pb is None:
                continue
            hb = 0.7 * vb + 0.3 * pb
            s = sum(strength[(clip, tier)]) / len(strength[(clip, tier)])
            print(f"{clip:9s} {tier:11s} {s:6.2f} {vb:+8.2f} {pb:+8.2f} {hb:+8.2f}")
            per_tier[tier].append(hb)
            per_clip[clip].append(hb)
    print()
    for tier in TIERS:
        v = per_tier[tier]
        if v:
            print(f"mean hybrid, {tier:11s}: {sum(v)/len(v):+6.2f}%  (n={len(v)})")
    print()
    for clip in clips:
        v = per_clip[clip]
        if v:
            print(f"mean hybrid, {clip:9s}: {sum(v)/len(v):+6.2f}%  (n={len(v)})")
    allv = [x for v in per_clip.values() for x in v]
    if allv:
        print(f"\noverall mean hybrid: {sum(allv)/len(allv):+6.2f}%  (n={len(allv)})")


if __name__ == "__main__":
    main()
