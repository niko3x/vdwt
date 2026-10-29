#!/usr/bin/env python3
"""VMAF and PSNR-Y against band2 strength for the three drone clips, qp 50."""
import csv, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "bench/complexity/drone_filter_sweep.csv"
OUT = ROOT / "videos/drone_filter_strength_curve.svg"
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"
W, H = 2000, 980
CLIPS = [("forest", "#2E7D32"), ("canyon", "#8D6E63"), ("tasmania", "#D9643A")]
STRENGTHS = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def load():
    d = {}
    for r in csv.DictReader(open(CSV)):
        if r["qp"] != "50" or r["filter"] not in ("none", "band2"):
            continue
        d[(r["clip"], float(r["strength"]))] = (float(r["vmaf"]), float(r["psnr_y"]))
    return d


def panel(s, x0, x1, y0, y1, d, idx, title, ymin, ymax, step):
    px = lambda x: x0 + (x1 - x0) * x / 8.0
    py = lambda y: y1 - (y1 - y0) * (y - ymin) / (ymax - ymin)
    s.append(f"<rect x='{x0}' y='{y0}' width='{x1-x0}' height='{y1-y0}' fill='#FFFDF8' stroke='#CCC' stroke-width='1.5'/>")
    v = ymin
    while v <= ymax + 1e-9:
        yy = py(v)
        s.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x1}' y2='{yy:.1f}' stroke='#E7E3D8' stroke-width='1'/>")
        s.append(f"<text x='{x0-16}' y='{yy+7:.1f}' {FONT} font-size='21' text-anchor='end' fill='#444'>{v:g}</text>")
        v += step
    for xt in (0, 2, 3, 4, 5, 6, 8):
        xx = px(xt)
        s.append(f"<line x1='{xx:.1f}' y1='{y0}' x2='{xx:.1f}' y2='{y1}' stroke='#EFEBE2' stroke-width='1'/>")
        s.append(f"<text x='{xx:.1f}' y='{y1+32}' {FONT} font-size='21' text-anchor='middle' fill='#444'>{xt:g}</text>")
    s.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='2'/>")
    s.append(f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333' stroke-width='2'/>")
    s.append(f"<text x='{x0}' y='{y0-16}' {FONT} font-size='25' font-weight='700' fill='#222'>{title}</text>")
    s.append(f"<text x='{(x0+x1)/2:.0f}' y='{y1+68}' {FONT} font-size='21' text-anchor='middle' fill='#444'>band2 strength</text>")
    for name, col in CLIPS:
        pts = [(px(s_), py(d[(name, s_)][idx])) for s_ in STRENGTHS]
        s.append("<polyline points='" + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) +
                 f"' fill='none' stroke='{col}' stroke-width='4' stroke-linejoin='round'/>")
        for a, b in pts:
            s.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='7' fill='{col}' stroke='#222' stroke-width='1.5'/>")
    # sweet spot and peak guides
    for x, label in ((2.5, "sweet spot 2.5"), (5.0, "VMAF peak 5.0")):
        xx = px(x)
        s.append(f"<line x1='{xx:.1f}' y1='{y0}' x2='{xx:.1f}' y2='{y1}' stroke='#999' stroke-width='1.5' stroke-dasharray='6 6'/>")
        s.append(f"<text x='{xx+8:.1f}' y='{y0+24}' {FONT} font-size='18' fill='#777'>{label}</text>")


def main():
    d = load()
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='70' y='78' {FONT} font-size='38' font-weight='700' fill='#222'>Drone strength sweep: how far band2 goes</text>",
         f"<text x='70' y='116' {FONT} font-size='21' fill='#555'>VDWT ultraheavy qp 50, 20 frames. VMAF rises then rolls over; PSNR-Y falls. Strength 3.5 and 4.0 collapse to the same coefficients.</text>",
         f"<text x='70' y='146' {FONT} font-size='20' fill='#777'>Sweet spot 2.0-2.5 for forest; smooth canyon/tasmania keep gaining to 5.0. Beyond 6.0 is over-sharpening.</text>"]
    lx = 70
    for name, col in CLIPS:
        s.append(f"<circle cx='{lx}' cy='188' r='8' fill='{col}' stroke='#222'/>")
        s.append(f"<text x='{lx+18}' y='196' {FONT} font-size='21' fill='#222'>{name}</text>")
        lx += 180
    panel(s, 150, 950, 260, 760, d, 0, "VMAF", 45, 95, 10)
    panel(s, 1120, 1920, 260, 760, d, 1, "PSNR-Y (dB)", 22, 38, 2)
    s.append(f"<text x='70' y='910' {FONT} font-size='19' fill='#666'>Display-only filter; not signalled. The sweet spot is content-dependent, which is the argument for a content-adaptive blend.</text>")
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
