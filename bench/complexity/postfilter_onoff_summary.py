#!/usr/bin/env python3
"""Post-filter on/off: BD of fixed 2.0 and auto against the unfiltered encode."""
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ["tasmania", "a720", "b720", "canyon", "forest"]
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]


def load():
    rows = defaultdict(lambda: defaultdict(list))
    for name in ("auto_vs_fixed.txt", "postfilter_off.txt"):
        path = ROOT / "bench/complexity" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            c, m, t, q, k, v, y, _, s = line.split()
            rows[(c, t)][m].append((float(k), float(v), float(y)))
    return rows


def interp(p, x):
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return p[-1][1]


def bd_rate(ref, test):
    a = sorted((q, math.log(r)) for r, q in ref)
    b = sorted((q, math.log(r)) for r, q in test)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi - lo < 0.5:
        return None
    xs = [lo + (hi - lo) * i / 200 for i in range(201)]
    return -100 * (math.exp(sum(interp(b, x) - interp(a, x) for x in xs) / len(xs)) - 1)


def main():
    rows = load()
    out = {}
    for c in CLIPS:
        for t in TIERS:
            got = rows[(c, t)]
            if "none" not in got:
                continue
            base = got["none"]
            for mode in ("band2-2.0", "auto"):
                if mode not in got:
                    continue
                vb = bd_rate([(k, v) for k, v, _ in base], [(k, v) for k, v, _ in got[mode]])
                pb = bd_rate([(k, y) for k, _, y in base], [(k, y) for k, _, y in got[mode]])
                out[(c, t, mode)] = (vb, pb, 0.7 * vb + 0.3 * pb)

    for mode, label in (("band2-2.0", "fixed 2.0"), ("auto", "auto")):
        print(f"\n{label} vs unfiltered: hybrid BD (positive = filter helps), VMAF / PSNR")
        print(f"{'clip':9s} " + " ".join(f"{t:>19s}" for t in TIERS))
        for c in CLIPS:
            cells = []
            for t in TIERS:
                v = out.get((c, t, mode))
                cells.append(f"{v[0]:+5.1f}/{v[1]:+5.1f}/{v[2]:+5.1f}" if v else " " * 17)
            print(f"{c:9s} " + " ".join(f"{x:>19s}" for x in cells))
        means = []
        for t in TIERS:
            vals = [out[(c, t, mode)] for c in CLIPS if (c, t, mode) in out]
            means.append(tuple(sum(v[i] for v in vals) / len(vals) for i in range(3)))
        print(f"{'mean':9s} " + " ".join(f"{m[0]:+5.1f}/{m[1]:+5.1f}/{m[2]:+5.1f}" for m in means))
        allv = [out[(c, t, mode)][2] for c in CLIPS for t in TIERS if (c, t, mode) in out]
        print(f"overall hybrid: {sum(allv)/len(allv):+.2f}% (n={len(allv)})")


if __name__ == "__main__":
    main()
