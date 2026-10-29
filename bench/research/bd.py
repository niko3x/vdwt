#!/usr/bin/env python3
"""Compare two run.py CSVs: BD-rate of `test` against `ref`.

    bd.py REF.csv TEST.csv [--tier TIER]

Without `--tier`, scores each tier separately and averages the tier BD-rates. Pooling tiers into
one curve is not a valid BD-rate: the tiers cover different quality ranges, and the pooled number
can be inflated by extrapolation (e.g. four luma levels measured +4.80% pooled but +0.50% per
tier in this campaign). With `--tier`, prints per-clip rows for that tier.

Hybrid = 70% VMAF + 30% YUV-PSNR. Positive saving = test needs fewer bits.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/complexity"))
from plot import bd_rate


def bd_safe(ref, test):
    """bd_rate with endpoint clamping (the shared one can step just past the last point)."""
    def interp(curve, x):
        if x <= curve[0][0]:
            return curve[0][1]
        if x >= curve[-1][0]:
            return curve[-1][1]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
        return None
    a = sorted((p, math.log(r)) for r, p in ref if r > 0)
    b = sorted((p, math.log(r)) for r, p in test if r > 0)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi - lo < 0.5:
        return None
    xs = [lo + (hi - lo) * i / 200 for i in range(201)]
    diff = [interp(b, x) - interp(a, x) for x in xs]
    return 100.0 * (math.exp(sum(diff) / len(diff)) - 1.0)


def load(path):
    rows = defaultdict(list)
    for line in open(path):
        p = line.split()
        if not p or p[0] == "tag":
            continue
        _, clip, tier, qp, nbytes, kbps, vmaf, yuv, py, ms = p
        rows[(clip, tier)].append((float(kbps), None if vmaf == "-" else float(vmaf), float(yuv)))
    return rows


def score(ref, test, clip, tier):
    a = ref.get((clip, tier), [])
    b = test.get((clip, tier), [])
    if len(a) < 2 or len(b) < 2:
        return None
    av = [(k, v) for k, v, _ in a]
    bv = [(k, v) for k, v, _ in b]
    ay = [(k, y) for k, _, y in a]
    by = [(k, y) for k, _, y in b]
    rv, ry = bd_safe(av, bv), bd_safe(ay, by)
    if rv is None or ry is None:
        return None
    return 0.7 * -rv + 0.3 * -ry, -rv, -ry


def main():
    ref, test = load(sys.argv[1]), load(sys.argv[2])
    tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else None
    clips = sorted({c for c, _ in ref if (c, tier) in test}) if tier else sorted(
        {c for c, t in ref if (c, t) in test})
    tiers = [tier] if tier else sorted({t for c, t in ref if (c, t) in test})
    all_hy = []
    for t in tiers:
        hy, vs, ys = [], [], []
        for clip in clips:
            s = score(ref, test, clip, t)
            if s is None:
                continue
            h, v, y = s
            hy.append(h)
            vs.append(v)
            ys.append(y)
            if tier:
                print(f"{clip:9s} hybrid {h:+7.2f}%  VMAF {v:+7.2f}%  YUV {y:+7.2f}%")
        if hy:
            all_hy += hy
            tag = f"{t:10s}" if not tier else "mean"
            print(f"{tag} hybrid {sum(hy)/len(hy):+7.2f}%  VMAF {sum(vs)/len(vs):+7.2f}%  "
                  f"YUV {sum(ys)/len(ys):+7.2f}%  ({len(hy)} clips)")
    if not tier and all_hy:
        print(f"{'overall':10s} hybrid {sum(all_hy)/len(all_hy):+7.2f}%  "
              f"({len(all_hy)} clip-tier cells)")


if __name__ == "__main__":
    main()
