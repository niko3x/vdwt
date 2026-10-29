#!/usr/bin/env python3
"""Heatmap of auto against fixed band2 2.0: hybrid BD-rate per clip and tier."""
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/complexity/auto_vs_fixed.txt"
OUT = ROOT / "videos/auto_vs_fixed_hybrid.svg"
CLIPS = ["tasmania", "a720", "b720", "canyon", "forest"]
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]


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


def colour(v):
    t = max(-8.0, min(8.0, v)) / 8.0
    red, cream, green = (217, 100, 58), (250, 249, 245), (76, 175, 130)
    if t >= 0:
        a, b = cream, green
        k = t
    else:
        a, b = cream, red
        k = -t
    return tuple(round(a[i] + (b[i] - a[i]) * k) for i in range(3))


def main():
    rows = defaultdict(lambda: defaultdict(list))
    strength = {}
    for line in DATA.read_text().splitlines():
        c, m, t, q, k, v, y, _, s = line.split()
        rows[(c, t)][m].append((float(k), float(v), float(y)))
        if m == "auto" and q == "42":
            strength[c] = int(s) / 8

    hybrid = {}
    for c in CLIPS:
        for t in TIERS:
            f = rows[(c, t)]["band2-2.0"]
            a = rows[(c, t)]["auto"]
            vb = bd_rate([(k, v) for k, v, _ in f], [(k, v) for k, v, _ in a])
            pb = bd_rate([(k, y) for k, _, y in f], [(k, y) for k, _, y in a])
            hybrid[(c, t)] = 0.7 * vb + 0.3 * pb

    W, H = 1960, 1080
    left, top = 360, 350
    cw, ch = 270, 120
    font = "font-family='Inter, Helvetica, Arial, sans-serif'"
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='70' y='92' {font} font-size='44' font-weight='700' fill='#222'>"
         "Auto against fixed band2: hybrid BD-rate</text>",
         f"<text x='70' y='137' {font} font-size='23' fill='#555'>"
         "5 clips, 5 tiers, qp 42/50/58, 20 frames; 70% VMAF + 30% YUV-PSNR BD-rate</text>",
         f"<text x='70' y='171' {font} font-size='21' fill='#777'>"
         "Green = auto needs fewer bits than fixed band2 2.0 at equal quality; red = more</text>"]

    for j, t in enumerate(TIERS):
        x = left + j * cw + cw / 2
        s.append(f"<text x='{x}' y='{top-22}' {font} font-size='22' fill='#333' "
                 f"text-anchor='middle'>{t}</text>")
    for i, c in enumerate(CLIPS):
        y = top + i * ch + ch / 2 + 8
        s.append(f"<text x='{left-26}' y='{y}' {font} font-size='24' fill='#222' "
                 f"text-anchor='end'>{c}</text>")
        s.append(f"<text x='{left-26}' y='{y+28}' {font} font-size='18' fill='#888' "
                 f"text-anchor='end'>qp42 {strength[c]:.1f}</text>")
    for i, c in enumerate(CLIPS):
        for j, t in enumerate(TIERS):
            v = hybrid[(c, t)]
            r, g, b = colour(v)
            x, y = left + j * cw, top + i * ch
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "#222" if lum > 150 else "#fff"
            s.append(f"<rect x='{x}' y='{y}' width='{cw-6}' height='{ch-6}' fill='rgb({r},{g},{b})' "
                     f"stroke='#FAF9F5' stroke-width='3'/>")
            s.append(f"<text x='{x+(cw-6)/2}' y='{y+ch/2+10}' {font} font-size='34' fill='{fg}' "
                     f"text-anchor='middle'>{v:+.1f}%</text>")

    # Means: right column per clip, bottom row per tier.
    mx = left + 5 * cw + 24
    s.append(f"<text x='{mx+100}' y='{top-22}' {font} font-size='20' fill='#333' "
             f"text-anchor='middle'>mean</text>")
    means = []
    for i, c in enumerate(CLIPS):
        v = sum(hybrid[(c, t)] for t in TIERS) / len(TIERS)
        means.append(v)
        y = top + i * ch + ch / 2 + 8
        s.append(f"<text x='{mx+100}' y='{y}' {font} font-size='26' fill='#222' "
                 f"text-anchor='middle'>{v:+.2f}%</text>")
    s.append(f"<text x='{mx+100}' y='{top+5*ch+8}' {font} font-size='20' fill='#333' "
             f"text-anchor='middle'>mean</text>")
    for j, t in enumerate(TIERS):
        v = sum(hybrid[(c, t)] for c in CLIPS) / len(CLIPS)
        x = left + j * cw + cw / 2
        s.append(f"<text x='{x}' y='{top+5*ch+40}' {font} font-size='26' fill='#222' "
                 f"text-anchor='middle'>{v:+.2f}%</text>")
    overall = sum(means) / len(means)
    s.append(f"<text x='{mx+100}' y='{top+5*ch+40}' {font} font-size='26' font-weight='700' "
             f"fill='#222' text-anchor='middle'>{overall:+.2f}%</text>")

    s.append(f"<text x='70' y='{H-40}' {font} font-size='18' fill='#777'>"
             "Auto raises strength to 1.0-4.5; b720 (VMAF ~96 at qp 42) is the regression. "
             "Data: auto_vs_fixed.txt</text>")
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
