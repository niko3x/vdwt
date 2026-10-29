#!/usr/bin/env python3
"""Compose the forest drone strength sweep into one labelled contact sheet."""
import base64, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "target/bench_tmp/filter_sweep"
OUT_SVG = ROOT / "videos/forest_filter_strength_sheet.svg"
OUT_PNG = ROOT / "videos/forest_filter_strength_sheet.png"
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"
W, H = 3540, 780
CROP = (400, 300, 850, 100)  # w, h, x, y in the 720p frame
TILE_W, TILE_H = 540, 405

ROWS = [
    ("none", 0.0, "none", "crop_forest_q50_none.png"),
    ("band 2.0", 2.0, "prev", "crop_forest_q50_prev.png"),
    ("band2 2.0", 2.0, "band2", "crop_forest_q50_band2_2.0.png"),
    ("band2 2.5", 2.5, "band2", "crop_forest_q50_band2_2.5.png"),
    ("band2 3.5", 3.5, "band2", "crop_forest_q50_band2_3.5.png"),
    ("band2 8.0", 8.0, "band2", "crop_forest_q50_band2_8.0.png"),
]


def read():
    d = {}
    for r in csv.DictReader(open(ROOT / "bench/complexity/drone_filter_sweep.csv")):
        if r["clip"] == "forest" and r["qp"] == "50":
            key = ({"band": "prev", "none": "none"}.get(r["filter"], "band2"), float(r["strength"]))
            if r["filter"] == "none":
                key = ("none", 0.0)
            d[key] = (float(r["vmaf"]), float(r["psnr_y"]), float(r["kbps"]))
    return d


def main():
    data = read()
    base_v, base_p, _ = data[("none", 0.0)]
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='40' y='58' {FONT} font-size='34' font-weight='700' fill='#222'>Forest drone frame: band2 strength sweep</text>",
         f"<text x='40' y='94' {FONT} font-size='19' fill='#555'>VDWT ultraheavy qp 50 (3.1 Mb/s), 20 frames at 60 fps; crop 400x300 at 1.35x. VMAF and PSNR-Y are over the whole clip.</text>",
         f"<text x='40' y='122' {FONT} font-size='19' fill='#777'>Sweet spot: 2.0-2.5; VMAF peaks at 2.5 here, PSNR cost grows past 3.0, and 8.0 is over-sharpened.</text>"]

    for i, (label, strength, kind, png) in enumerate(ROWS):
        x = 40 + i * 580
        y = 160
        href = base64.b64encode((TMP / png).read_bytes()).decode()
        s.append(f"<image x='{x}' y='{y}' width='{TILE_W}' height='{TILE_H}' "
                 f"href='data:image/png;base64,{href}'/>")
        s.append(f"<rect x='{x}' y='{y}' width='{TILE_W}' height='{TILE_H}' fill='none' stroke='#333' stroke-width='2'/>")
        v, p, kbps = data[(kind, strength)]
        colour = "#1A237E" if kind == "band2" else "#666"
        s.append(f"<text x='{x}' y='{y+TILE_H+34}' {FONT} font-size='24' font-weight='700' fill='{colour}'>{label}</text>")
        s.append(f"<text x='{x}' y='{y+TILE_H+64}' {FONT} font-size='20' fill='#333'>VMAF {v:.2f}  ({v-base_v:+.2f})</text>")
        s.append(f"<text x='{x}' y='{y+TILE_H+90}' {FONT} font-size='20' fill='#333'>PSNR-Y {p:.2f} dB  ({p-base_p:+.2f})</text>")
    s.append(f"<text x='40' y='760' {FONT} font-size='18' fill='#666'>Filter is display-only and not signalled. The wider band helps smooth canyon/tasmania texture but over-sharpens this forest frame at high strength.</text>")
    s.append("</svg>")
    OUT_SVG.write_text("\n".join(s))
    print("wrote", OUT_SVG)


if __name__ == "__main__":
    main()
