#!/usr/bin/env python3
"""Chart and table for cnn_pareto.py: BD-rate (YUV-PSNR 6:1:1) against VDWT medium without the
CNN, over decode (or encode) time per frame.

    plot_cnn_pareto.py ROWS.txt decode|encode OUT.svg
"""
import math, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "complexity"))
from plot import bd_rate  # noqa: E402

TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]
FAMS = [("vdwt", "VDWT", "#2E6DB4"), ("vdwtS", "VDWT + small CNN (12×4, ~4k MAC/px)", "#D9822B"),
        ("vdwtC", "VDWT + large CNN (32×6, ~49k MAC/px)", "#7B1E3A")]


def main():
    rows_path, kind, out = sys.argv[1:4]
    col = 6 if kind == "decode" else 5
    pts, ms = defaultdict(list), defaultdict(list)
    for f in (l.split() for l in open(rows_path) if l.strip()):
        pts[(f[0], f[1])].append((float(f[3]), float(f[4])))
        ms[f[1]].append(float(f[col]))
    clips = sorted({c for c, _ in pts})
    ref = "vdwt-medium"

    def bd(enc, base=ref):
        return sum(bd_rate(pts[(c, base)], pts[(c, enc)]) for c in clips) / len(clips)

    data = {f"{p}-{t}": (sum(ms[f"{p}-{t}"]) / len(ms[f"{p}-{t}"]), bd(f"{p}-{t}"))
            for p, _, _ in FAMS for t in TIERS}
    print(f"{'tier':11s} {'no CNN':>16s} {'small CNN':>26s} {'large CNN':>26s}   ({kind} ms/frame, BD vs medium)")
    for t in TIERS:
        cells = [f"{data[f'{p}-{t}'][0]:7.1f} ms {data[f'{p}-{t}'][1]:+6.1f}%" for p, _, _ in FAMS]
        gain = [bd(f"{p}-{t}", f"vdwt-{t}") for p in ("vdwtS", "vdwtC")]
        print(f"{t:11s} {cells[0]:>16s} {cells[1]:>18s} ({gain[0]:+5.1f}%) {cells[2]:>18s} ({gain[1]:+5.1f}%)")

    W, H, L, R, T, B = 1200, 700, 110, 40, 120, 90
    xs = [v[0] for v in data.values()]
    x0, x1 = math.log10(min(xs) / 1.5), math.log10(max(xs) * 1.5)
    ys = [v[1] for v in data.values()]
    y0, y1 = math.floor(min(ys) / 5) * 5 - 5, math.ceil(max(ys) / 5) * 5 + 5
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (v - y0) / (y1 - y0) * (H - T - B)  # lower BD (better) at the top
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='sans-serif'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='{L}' y='40' font-size='26' font-weight='bold'>VDWT with and without the CNN restoration filter</text>",
         f"<text x='{L}' y='68' font-size='15' fill='#555'>3 public Xiph clips never seen in CNN training (Netflix Aerial, vidyo4, park_joy), 20 frames, 720p; "
         f"YUV-PSNR 6:1:1 BD-rate vs VDWT medium</text>",
         f"<text x='{L}' y='90' font-size='15' fill='#555'>CNN applied to decoded frames with a per-frame switch (offline); its time is single-thread float PyTorch on the x86 laptop</text>"]
    for v in range(int(y0), int(y1) + 1, 5):
        s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='#e5e5e5'/>"
                 f"<text x='{L - 10}' y='{Y(v) + 5:.1f}' font-size='14' text-anchor='end'>{v:+d}%</text>")
    for e in range(math.floor(x0), math.ceil(x1) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if x0 <= math.log10(v) <= x1:
                s.append(f"<line y1='{T}' y2='{H - B}' x1='{X(v):.1f}' x2='{X(v):.1f}' stroke='#eee'/>"
                         f"<text x='{X(v):.1f}' y='{H - B + 22}' font-size='14' text-anchor='middle'>{v:g}</text>")
    s.append(f"<text x='{(L + W - R) / 2}' y='{H - 30}' font-size='16' text-anchor='middle'>"
             f"{kind.capitalize()} time per frame (ms, log scale; single thread)</text>")
    s.append(f"<text x='30' y='{(T + H - B) / 2}' font-size='16' text-anchor='middle' "
             f"transform='rotate(-90 30 {(T + H - B) / 2})'>BD-rate vs VDWT medium (higher up = fewer bits)</text>")
    for i, (p, name, colour) in enumerate(FAMS):
        line = " ".join(f"{X(data[f'{p}-{t}'][0]):.1f},{Y(data[f'{p}-{t}'][1]):.1f}" for t in TIERS)
        s.append(f"<polyline points='{line}' fill='none' stroke='{colour}' stroke-width='2.5'/>")
        for t in TIERS:
            x, y = X(data[f"{p}-{t}"][0]), Y(data[f"{p}-{t}"][1])
            s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='{colour}'/>"
                     f"<text x='{x + 8:.1f}' y='{y - 8:.1f}' font-size='12' fill='{colour}'>{t}</text>")
        s.append(f"<rect x='{W - R - 380}' y='{H - B - 95 + 28 * i}' width='16' height='16' fill='{colour}'/>"
                 f"<text x='{W - R - 356}' y='{H - B - 82 + 28 * i}' font-size='15'>{name}</text>")
    s.append("</svg>")
    Path(out).write_text("\n".join(s))


if __name__ == "__main__":
    main()
