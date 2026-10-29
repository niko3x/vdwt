#!/usr/bin/env python3
"""Controlled A/B: baseline vs held-out context tables on the same build and x265 rows."""
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ["a720", "b720", "canyon", "forest", "tasmania"]
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]


def load_x265():
    rows = defaultdict(list)
    for name in ("hybrid_samtime.txt", "hybrid_drone.txt"):
        for line in (ROOT / "bench/complexity" / name).read_text().splitlines():
            c, e, q, k, v, y, m = line.split()
            if e == "x265-medium":
                rows[c].append((float(k), float(v), float(y)))
    return rows


def load_vdwt(path):
    rows = defaultdict(lambda: defaultdict(list))
    for line in (ROOT / path).read_text().splitlines():
        c, e, q, k, v, y, m = line.split()
        rows[c][e.replace("vdwt-", "")].append((float(k), float(v), float(y)))
    return rows


def interp(p, x):
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return p[-1][1]


def bd_rate(ref, test):
    a = sorted((q, math.log(r)) for r, q in ref)
    b = sorted((q, math.log(r)) for r, q in test)
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    xs = [lo + (hi - lo) * i / 200 for i in range(201)]
    return -100 * (math.exp(sum(interp(b, x) - interp(a, x) for x in xs) / len(xs)) - 1)


def score(old, new, ref):
    ov = bd_rate([(k, v) for k, v, _ in ref], [(k, v) for k, v, _ in old])
    op = bd_rate([(k, y) for k, _, y in ref], [(k, y) for k, _, y in old])
    nv = bd_rate([(k, v) for k, v, _ in ref], [(k, v) for k, v, _ in new])
    np_ = bd_rate([(k, y) for k, _, y in ref], [(k, y) for k, _, y in new])
    return (0.7 * ov + 0.3 * op, 0.7 * nv + 0.3 * np_)


def main():
    ref = load_x265()
    old = load_vdwt("bench/complexity/baseline_vdwt.txt")
    new = load_vdwt("bench/complexity/heldout_vdwt.txt")
    print(f"{'clip':9s} {'tier':11s} {'baseline':>9s} {'held-out':>9s} {'delta':>7s}")
    deltas = defaultdict(list)
    for c in CLIPS:
        for t in TIERS:
            o, n = score(old[c][t], new[c][t], ref[c])
            print(f"{c:9s} {t:11s} {o:+9.2f} {n:+9.2f} {n-o:+7.2f}")
            deltas[t].append(n - o)
    print()
    for t in TIERS:
        print(f"mean delta, {t:11s}: {sum(deltas[t])/5:+.2f}%")
    allv = [x for v in deltas.values() for x in v]
    print(f"overall mean delta: {sum(allv)/len(allv):+.2f}%")
    sam = [deltas[t][i] for i in (0, 1) for t in TIERS]
    drone = [deltas[t][i] for i in (2, 3, 4) for t in TIERS]
    print(f"samtime mean delta: {sum(sam)/len(sam):+.2f}%")
    print(f"drone mean delta:   {sum(drone)/len(drone):+.2f}%")


if __name__ == "__main__":
    main()
