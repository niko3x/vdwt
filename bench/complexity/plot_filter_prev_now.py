#!/usr/bin/env python3
"""Prev-vs-now post-filter Pareto on the samtime clips (VMAF and PSNR-Y vs bitrate)."""
import csv, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "bench/complexity/filter_prev_now.csv"
OUT = ROOT / "videos/samtime_filter_prev_vs_now.svg"
CLIPS = ["a720", "b720"]
W, H = 2000, 1500
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


def read():
    d = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        d[(r["clip"], r["filter"])].append(
            (float(r["kbps"]), float(r["vmaf"]), float(r["psnr_y"])))
    for v in d.values():
        v.sort()
    return d


def interp(curve, x):
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return None


def bd_rate(ref, test):
    """Bits saved by `test` at equal quality (%), positive = test better."""
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


def nice_y(lo, hi):
    span = hi - lo
    step = 5 if span > 18 else 2 if span > 8 else 1
    y0 = math.floor(lo / step) * step
    y1 = math.ceil(hi / step) * step
    return y0, y1, step


def panel(s, x0, x1, y0, y1, clip, metric, data, show_x, show_y):
    px_min = min(p[0] for p in data[(clip, "prev")])
    px_max = max(p[0] for p in data[(clip, "prev")])
    xmin, xmax = px_min * 0.82, px_max * 1.18
    idx = 1 if metric == "vmaf" else 2
    vals = [p[idx] for f in ("prev", "now") for p in data[(clip, f)]]
    ymin, ymax, step = nice_y(min(vals) - 1, max(vals) + 1)
    px = lambda x: x0 + (x1 - x0) * (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
    py = lambda y: y1 - (y1 - y0) * (y - ymin) / (ymax - ymin)

    s.append(f"<rect x='{x0}' y='{y0}' width='{x1-x0}' height='{y1-y0}' fill='#FFFDF8' stroke='#CCC' stroke-width='1.5'/>")
    y = ymin
    while y <= ymax + 1e-9:
        yy = py(y)
        s.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x1}' y2='{yy:.1f}' stroke='#E7E3D8' stroke-width='1'/>")
        s.append(f"<text x='{x0-18}' y='{yy+7:.1f}' {FONT} font-size='22' text-anchor='end' fill='#444'>{y:g}</text>")
        y += step
    for xt in (50, 100, 200, 400):
        if xmin <= xt <= xmax:
            xx = px(xt)
            s.append(f"<line x1='{xx:.1f}' y1='{y0}' x2='{xx:.1f}' y2='{y1}' stroke='#EFEBE2' stroke-width='1'/>")
            s.append(f"<text x='{xx:.1f}' y='{y1+34}' {FONT} font-size='22' text-anchor='middle' fill='#444'>{xt}</text>")

    # axes
    s.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='2'/>")
    s.append(f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333' stroke-width='2'/>")
    title = f"{clip} - {'VMAF' if metric == 'vmaf' else 'PSNR-Y (dB)'}"
    s.append(f"<text x='{x0}' y='{y0-18}' {FONT} font-size='27' font-weight='700' fill='#222'>{title}</text>")
    if show_x:
        s.append(f"<text x='{(x0+x1)/2:.0f}' y='{y1+78}' {FONT} font-size='23' text-anchor='middle' fill='#444'>bitrate (kb/s, log)</text>")
    if show_y:
        s.append(f"<text x='{x0-72}' y='{(y0+y1)/2:.0f}' {FONT} font-size='23' text-anchor='middle' fill='#444' transform='rotate(-90 {x0-72} {(y0+y1)/2:.0f})'>{'VMAF' if metric == 'vmaf' else 'PSNR-Y (dB)'}</text>")

    styles = {"prev": ("#8A8A8A", "10 8", "circle"), "now": ("#1A237E", "", "square")}
    for f in ("prev", "now"):
        pts = sorted((px(p[0]), py(p[idx])) for p in data[(clip, f)])
        col, dash, shape = styles[f]
        d = f" stroke-dasharray='{dash}'" if dash else ""
        s.append(f"<polyline points='" + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) +
                 f"' fill='none' stroke='{col}' stroke-width='5' stroke-linejoin='round'{d}/>")
        for a, b in pts:
            if shape == "circle":
                s.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='9' fill='{col}' stroke='#222' stroke-width='1.5'/>")
            else:
                s.append(f"<rect x='{a-8:.1f}' y='{b-8:.1f}' width='16' height='16' fill='{col}' stroke='#222' stroke-width='1.5'/>")

    # score in the corner
    ref = [(p[0], p[idx]) for p in data[(clip, "prev")]]
    test = [(p[0], p[idx]) for p in data[(clip, "now")]]
    bd = bd_rate(ref, test)
    dv = sum(b[idx] for b in data[(clip, "now")]) / 3 - sum(b[idx] for b in data[(clip, "prev")]) / 3
    tag = (f"VMAF BD-rate {bd:+.1f}%" if metric == "vmaf" and bd is not None
           else f"mean {dv:+.2f} dB")
    s.append(f"<text x='{x0+14}' y='{y0+30}' {FONT} font-size='21' fill='#555'>{tag}</text>")


def main():
    data = read()
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='70' y='86' {FONT} font-size='44' font-weight='700' fill='#222'>Samtime: previous vs new post-filter</text>",
         f"<text x='70' y='130' {FONT} font-size='23' fill='#555'>VDWT ultraheavy, 100 frames per clip, 25 fps, qp 42/50/58. Both filters run on the same decoded streams at strength 2.0.</text>",
         f"<text x='70' y='164' {FONT} font-size='22' fill='#777'>Mean now - prev: +1.42 VMAF, -0.84 dB PSNR-Y; VMAF BD-rate +6.0% (new needs fewer bits at equal VMAF).</text>",
         f"<line x1='70' y1='212' x2='155' y2='212' stroke='#8A8A8A' stroke-width='5' stroke-dasharray='10 8'/>",
         f"<circle cx='112' cy='212' r='9' fill='#8A8A8A' stroke='#222'/>",
         f"<text x='172' y='220' {FONT} font-size='24' fill='#222'>previous: c + 2.0*(blur3 - blur5)</text>",
         f"<line x1='660' y1='212' x2='745' y2='212' stroke='#1A237E' stroke-width='5'/>",
         f"<rect x='698' y='205' width='16' height='16' fill='#1A237E' stroke='#222'/>",
         f"<text x='762' y='220' {FONT} font-size='24' fill='#222'>new: c + 2.0*(blur3 - [1 2 3 2 1]/9)</text>"]

    panel(s, 170, 940, 300, 700, "a720", "vmaf", data, True, True)
    panel(s, 1120, 1890, 300, 700, "b720", "vmaf", data, True, False)
    panel(s, 170, 940, 890, 1290, "a720", "psnr_y", data, True, True)
    panel(s, 1120, 1890, 890, 1290, "b720", "psnr_y", data, True, False)

    s.append(f"<text x='70' y='1400' {FONT} font-size='21' fill='#666'>Display-only filter: not signalled and not a coding gain. VMAF rises while PSNR-Y falls because sharpening adds MSE.</text>")
    s.append(f"<text x='70' y='1436' {FONT} font-size='21' fill='#666'>Points are ultraheavy qp 42 (right), 50, 58 (left); curves connect equal-qp points.</text>")
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
