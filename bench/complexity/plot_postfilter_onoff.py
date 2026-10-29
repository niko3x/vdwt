#!/usr/bin/env python3
"""Post-filter on/off Pareto: VMAF against bitrate for unfiltered, fixed 2.0 and auto."""
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "videos/postfilter_onoff_pareto.svg"
CLIPS = ["tasmania", "a720", "b720", "canyon", "forest"]
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]
MODES = [("none", "unfiltered", "#9E9E9E"), ("band2-2.0", "fixed band2 2.0", "#D9643A"),
         ("auto", "auto", "#4CAF82")]


def load():
    rows = defaultdict(list)
    for name in ("auto_vs_fixed.txt", "postfilter_off.txt"):
        path = ROOT / "bench/complexity" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            c, m, t, q, k, v, y, _, s = line.split()
            rows[(c, m)].append((float(k), float(v), float(y)))
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


def tier_rows(rows):
    """(clip, mode) -> tier -> [(kbps, vmaf, yuv)] by re-reading the raw tables."""
    d = defaultdict(lambda: defaultdict(list))
    for name in ("auto_vs_fixed.txt", "postfilter_off.txt"):
        path = ROOT / "bench/complexity" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            c, m, t, q, k, v, y, _, s = line.split()
            d[(c, m)][t].append((float(k), float(v), float(y)))
    return d


def draw():
    rows = load()
    tiers = tier_rows(rows)
    W, H = 2000, 1440
    font = "font-family='Inter, Helvetica, Arial, sans-serif'"
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='70' y='92' {font} font-size='44' font-weight='700' fill='#222'>"
         "Post-filter on/off: VMAF against bitrate</text>",
         f"<text x='70' y='137' {font} font-size='23' fill='#555'>"
         "15 points per mode (5 tiers × qp 42/50/58, 720p, 20 frames); the three modes share each "
         "rate, so the filter gain is vertical</text>",
         f"<text x='70' y='171' {font} font-size='21' fill='#777'>"
         "Same encode, only the display filter changes; VMAF is luma-only, not the 70/30 hybrid</text>"]

    lx = 70
    for i, (_, label, col) in enumerate(MODES):
        x = lx + i * 330
        s.append(f"<circle cx='{x}' cy='216' r='13' fill='{col}'/>")
        s.append(f"<text x='{x+24}' y='225' {font} font-size='24' fill='#222'>{label}</text>")
    s.append(f"<text x='{W-70}' y='225' {font} font-size='20' fill='#777' text-anchor='end'>"
             "higher and left is better</text>")

    pw, ph = 500, 380
    pos = [(90 + (i % 3) * 620, 300 + (i // 3) * 520) for i in range(5)]
    for k, clip in enumerate(CLIPS):
        px, py = pos[k]
        pts = []
        for mode, _, _ in MODES:
            for t in TIERS:
                pts.extend(tiers[(clip, mode)][t])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs) / 1.25, max(xs) * 1.25
        y0, y1 = min(ys) - 3, max(ys) + 3
        sx = lambda x: px + 60 + (math.log10(x) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * pw
        sy = lambda y: py + ph - (y - y0) / (y1 - y0) * ph

        s.append(f"<text x='{px+10}' y='{py-18}' {font} font-size='26' font-weight='600' "
                 f"fill='#222'>{clip}</text>")
        s.append(f"<rect x='{px+60}' y='{py}' width='{pw}' height='{ph}' fill='#fff' "
                 f"stroke='#DDD' stroke-width='2'/>")
        for v in range(0, 101, 10):
            if y0 <= v <= y1:
                y = sy(v)
                s.append(f"<line x1='{px+60}' y1='{y:.1f}' x2='{px+60+pw}' y2='{y:.1f}' "
                         f"stroke='#EEE' stroke-width='1.5'/>")
                s.append(f"<text x='{px+50}' y='{y+6:.1f}' {font} font-size='17' fill='#777' "
                         f"text-anchor='end'>{v}</text>")
        ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for v in ticks:
            if x0 <= v <= x1:
                x = sx(v)
                s.append(f"<line x1='{x:.1f}' y1='{py}' x2='{x:.1f}' y2='{py+ph}' "
                         f"stroke='#EEE' stroke-width='1.5'/>")
                lab = f"{v // 1000}k" if v >= 1000 else str(v)
                s.append(f"<text x='{x:.1f}' y='{py+ph+24}' {font} font-size='17' fill='#777' "
                         f"text-anchor='middle'>{lab}</text>")
        s.append(f"<text x='{px+60+pw/2}' y='{py+ph+48}' {font} font-size='18' fill='#777' "
                 f"text-anchor='middle'>kbps (log)</text>")
        for mode, _, col in MODES:
            for kbps, vmaf, _ in rows[(clip, mode)]:
                s.append(f"<circle cx='{sx(kbps):.1f}' cy='{sy(vmaf):.1f}' r='7' fill='{col}' "
                         f"fill-opacity='0.85' stroke='#fff' stroke-width='1'/>")

        # Mean hybrid BD against unfiltered over the five tiers.
        notes = []
        for mode, label, _ in (("band2-2.0", "fixed", None), ("auto", "auto", None)):
            vals = []
            for t in TIERS:
                base = tiers[(clip, "none")][t]
                test = tiers[(clip, mode)][t]
                vb = bd_rate([(k, v) for k, v, _ in base], [(k, v) for k, v, _ in test])
                pb = bd_rate([(k, y) for k, _, y in base], [(k, y) for k, _, y in test])
                vals.append(0.7 * vb + 0.3 * pb)
            notes.append(f"{label} {sum(vals)/len(vals):+.1f}")
        s.append(f"<text x='{px+60}' y='{py+ph+78}' {font} font-size='17' fill='#555'>"
                 f"hybrid BD vs off: " + ", ".join(notes) + "%</text>")

    s.append(f"<text x='70' y='{H-36}' {font} font-size='18' fill='#777'>"
             "Data: auto_vs_fixed.txt + postfilter_off.txt; generator plot_postfilter_onoff.py. "
             "b720: auto wins VMAF but fixed wins the 70/30 hybrid (PSNR).</text>")
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    draw()
