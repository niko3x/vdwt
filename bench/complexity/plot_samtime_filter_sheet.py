#!/usr/bin/env python3
"""Samtime qp50 strength contact sheet: a720 face and b720 audience crops."""
import base64, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "target/bench_tmp/filter_sweep"
OUT = ROOT / "videos/samtime_filter_strength_sheet.svg"
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"
W, H = 3540, 1270
TW, TH = 540, 405

COLS = [
    ("none", "none", 0.0),
    ("prev band 2.0", "prev", 2.0),
    ("band2 2.0", "band2", 2.0),
    ("band2 2.5", "band2", 2.5),
    ("band2 3.0", "band2", 3.0),
    ("band2 5.0", "band2", 5.0),
]
ROWS = [("a720", "portrait, q50"), ("b720", "stage, q50")]


def load():
    d = {}
    for r in csv.DictReader(open(ROOT / "bench/complexity/filter_strength_sweep.csv")):
        if r["qp"] != "50":
            continue
        key = ({"band": "prev", "none": "none"}.get(r["filter"], "band2"), float(r["strength"]))
        d[(r["clip"],) + key] = (float(r["vmaf"]), float(r["psnr_y"]), float(r["kbps"]))
    return d


def main():
    d = load()
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='50' y='58' {FONT} font-size='34' font-weight='700' fill='#222'>Samtime: band2 strength sweep</text>",
         f"<text x='50' y='94' {FONT} font-size='19' fill='#555'>VDWT ultraheavy qp 50, 100 frames at 25 fps; crop 400x300 at 1.35x. VMAF and PSNR-Y are over the whole clip; deltas are against prev band 2.0.</text>",
         f"<text x='50' y='122' {FONT} font-size='19' fill='#777'>Both clips keep gaining VMAF to 5.0, but the PSNR cost is steep; 2.5-3.0 is the practical knee.</text>"]

    for row, (clip, note) in enumerate(ROWS):
        y = 170 + row * 550
        kbps = d[(clip, "none", 0.0)][2]
        s.append(f"<text x='50' y='{y-14}' {FONT} font-size='25' font-weight='700' fill='#222'>{clip} - {note} ({kbps:.0f} kb/s)</text>")
        base = d[(clip, "none", 0.0)]
        prev = d[(clip, "prev", 2.0)]
        for i, (label, kind, strength) in enumerate(COLS):
            x = 50 + i * 580
            name = {"none": "none", "prev": "prev"}.get(kind, f"band2_{strength}")
            png = TMP / f"samcrop_{clip}_{name}.png"
            href = base64.b64encode(png.read_bytes()).decode()
            s.append(f"<image x='{x}' y='{y}' width='{TW}' height='{TH}' href='data:image/png;base64,{href}'/>")
            s.append(f"<rect x='{x}' y='{y}' width='{TW}' height='{TH}' fill='none' stroke='#333' stroke-width='2'/>")
            v, p, _ = d[(clip, kind, strength)]
            colour = "#1A237E" if kind == "band2" else "#555"
            s.append(f"<text x='{x}' y='{y+TH+32}' {FONT} font-size='23' font-weight='700' fill='{colour}'>{label}</text>")
            if kind == "none":
                dv, dp = 0.0, 0.0
                tag = "none"
            elif kind == "prev":
                dv, dp = v - base[0], p - base[1]
                tag = "vs none"
            else:
                dv, dp = v - prev[0], p - prev[1]
                tag = "vs prev"
            s.append(f"<text x='{x}' y='{y+TH+60}' {FONT} font-size='19' fill='#333'>VMAF {v:.2f}  ({tag} {dv:+.2f})</text>")
            s.append(f"<text x='{x}' y='{y+TH+86}' {FONT} font-size='19' fill='#333'>PSNR-Y {p:.2f} dB  ({dp:+.2f})</text>")

    s.append(f"<text x='50' y='1240' {FONT} font-size='18' fill='#666'>Display-only filter, not signalled. b720 q42 is the weakest point: band2 5.0 loses 4.4 dB PSNR for +0.9 VMAF over prev.</text>")
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
