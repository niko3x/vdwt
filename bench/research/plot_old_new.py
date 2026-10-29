#!/usr/bin/env python3
"""Old-vs-new VDWT Pareto charts from bench/research CSV grids.

    plot_old_new.py OUTDIR

Reads target/research/csv/{old,new,aq,free}.csv and the x265 reference rows, computes the hybrid
BD-rate (70% VMAF + 30% YUV-PSNR) against x265 medium, and writes SVG charts. The `free_*` charts
compare the VERSION 9 baseline, the AQ+filter build and the adaptive-tilt build. Rasterise with
`rsvg-convert -w 2200 FILE.svg -o FILE.png`.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/complexity"))
from plot_hybrid import Panel

CSV = ROOT / "target/research/csv"
X265_FILES = [ROOT / "bench/complexity/hybrid_samtime.txt",
              ROOT / "bench/complexity/hybrid_drone.txt"]
CLIPS = ["a720", "b720", "canyon", "forest", "tasmania"]
SAMTIME = ["a720", "b720"]
DRONE = ["canyon", "forest", "tasmania"]
TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]
X265 = ["medium", "slow", "veryslow"]


def bd(ref, test):
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


def load_tag(path):
    rows = defaultdict(list)
    for line in open(path):
        p = line.split()
        if not p or p[0] == "tag":
            continue
        _, clip, tier, qp, nbytes, kbps, vmaf, yuv, py, ms = p
        rows[(clip, tier)].append((float(kbps), float(vmaf), float(yuv), float(ms), int(qp)))
    return rows


def load_x265():
    rows = defaultdict(list)
    for path in X265_FILES:
        for line in open(path):
            c, e, q, k, v, y, m = line.split()
            if e.startswith("x265-"):
                rows[(c, e[5:])].append((float(k), float(v), float(y), float(m), int(q)))
    return rows


def hybrid_bd(ref, test):
    rv = [(k, v) for k, v, y, m, q in ref]
    ry = [(k, y) for k, v, y, m, q in ref]
    tv = [(k, v) for k, v, y, m, q in test]
    ty = [(k, y) for k, v, y, m, q in test]
    a, b = bd(rv, tv), bd(ry, ty)
    if a is None or b is None:
        return None
    return 0.7 * -a + 0.3 * -b


def load_ms(path):
    if not path.exists():
        return {}
    rows = defaultdict(list)
    for line in open(path):
        p = line.split()
        if not p or p[0] == "tag":
            continue
        clip, tier, ms = p[1], p[2], p[-1]
        rows[(clip, tier)].append(float(ms))
    return {k: sum(v) / len(v) for k, v in rows.items()}


def points(tag_rows, xrows, clips, tiers=TIERS, times=None):
    out = {}
    for tier in tiers:
        vals, times_ms = [], []
        for clip in clips:
            test = tag_rows.get((clip, tier), [])
            ref = xrows.get((clip, "medium"), [])
            h = hybrid_bd(ref, test) if test else None
            if h is not None:
                vals.append(h)
                if times:
                    times_ms.append(times[(clip, tier)])
                else:
                    times_ms += [m for *_, m, q in test]
        if vals:
            out[tier] = (sum(vals) / len(vals), sum(times_ms) / len(times_ms), len(vals))
    return out


class Chart(Panel):
    def __init__(self, title, subtitle, clips, yrange=(-25, 20), xrange=(5, 600),
                 legend=None, note=None):
        super().__init__(title, subtitle, yrange, clips, xrange=xrange,
                         xticks=(5, 10, 20, 50, 100, 200, 500))
        self.subtitle = subtitle
        self.legend = legend or [("#777777", "old"), ("#E08A1E", "AQ only"),
                                 ("#C62828", "new"), ("#2E8B57", "x265")]
        self.note = note or ("Old: VERSION 9 baseline. New: causal frame AQ + perceptual "
                             "in-loop filter. AQ: frame AQ only.")

    def header(self):
        s = self.svg
        s.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.W}' height='{self.H}' "
                 f"viewBox='0 0 {self.W} {self.H}'>")
        s.append(f"<rect width='{self.W}' height='{self.H}' fill='#FAF9F5'/>")
        s.append(f"<rect x='4' y='4' width='{self.W-8}' height='{self.H-8}' fill='none' "
                 "stroke='#222' stroke-width='3'/>")
        s.append(self.text(70, 90, 44, weight="700") + f"{self.title}</text>")
        s.append(self.text(70, 137, 24, fill="#555") + self.subtitle + "</text>")
        s.append(self.text(70, 171, 23, fill="#555")
                 + "70% VMAF (vmaf_v0.6.1) + 30% YUV-PSNR BD-rate vs x265 medium; "
                 "positive = fewer bits at equal quality</text>")
        s.append(self.text(70, 205, 22, fill="#777") + self.note + "</text>")
        for x, (colour, label) in zip((250, 410, 590, 760), self.legend):
            s.append(f"<circle cx='{x}' cy='277' r='15' fill='{colour}' stroke='#222' stroke-width='2.5'/>")
            s.append(self.text(x + 26, 287, 27) + label + "</text>")

    def draw_family(self, pts, colour, labels, name, offsets=None):
        ordered = [(t, pts[t]) for t in TIERS if t in pts]
        if len(ordered) < 2:
            return
        coords = [(self.px(v[1]), self.py(v[0]), t, v) for t, v in ordered]
        self.svg.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
                        + f"' fill='none' stroke='{colour}' stroke-width='5' stroke-linejoin='round'/>")
        for x, y, tier, v in coords:
            r = 17 if tier in labels else 12
            self.svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{colour}' "
                            "stroke='#222' stroke-width='2.5'/>")
            if tier in labels:
                dx, dy = (offsets or {}).get(tier, (0, -28))
                self.label(x, y, labels[tier], colour, dx=dx, dy=dy)

    def draw_x265(self, pts):
        coords = [(self.px(v[1]), self.py(v[0]), n, v) for n, v in pts.items()]
        self.svg.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
                        + "' fill='none' stroke='#2E8B57' stroke-width='4' stroke-dasharray='8 7'/>")
        for x, y, name, v in coords:
            self.svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='12' fill='#2E8B57' "
                            "stroke='#222' stroke-width='2.5'/>")
            if name == "medium":
                continue
            dx, dy = {"slow": (18, 30), "veryslow": (0, 34)}.get(name, (0, 32))
            self.label(x, y, name, "#2E8B57", dx=dx, dy=dy, size=23)

    def footer(self, source):
        self.svg.append(self.text(70, 1345, 19, fill="#888") + source + "</text>")


def x265_points(xrows, clips):
    out = {}
    for preset in X265:
        vals, times = [], []
        for clip in clips:
            ref = xrows[(clip, "medium")]
            test = xrows[(clip, preset)]
            h = hybrid_bd(ref, test)
            if h is not None:
                vals.append(h)
                times += [m for *_, m, q in test]
        if vals:
            out[preset] = (sum(vals) / len(vals), sum(times) / len(times), len(vals))
    return out


def chart(outdir, name, clips, title, yrange=(-25, 20)):
    old = load_tag(CSV / "old.csv")
    new = load_tag(CSV / "new.csv")
    aq = load_tag(CSV / "aq.csv")
    xrows = load_x265()
    told, tnew, taq = load_ms(CSV / "timing_old.csv"), load_ms(CSV / "timing_new.csv"), load_ms(CSV / "timing_aq.csv")
    oldp = points(old, xrows, clips, times=told or None)
    newp = points(new, xrows, clips, times=tnew or None)
    aqp = points(aq, xrows, clips, times=taq or tnew or None)
    xp = x265_points(xrows, clips)
    c = Chart(title, f"{len(clips)} clips ({', '.join(clips)}), 20 frames, 720p, 4 threads", clips,
              yrange=yrange)
    c.header()
    c.axes()
    c.draw_family(oldp, "#777777", {"ultraheavy": "old ultra"}, "old",
                  {"ultraheavy": (60, 40)})
    c.draw_family(aqp, "#E08A1E", {"ultraheavy": "AQ ultra"}, "aq",
                  {"ultraheavy": (64, 16)})
    c.draw_family(newp, "#C62828", {"ultraheavy": "new ultra"}, "new",
                  {"ultraheavy": (56, -30)})
    c.draw_x265(xp)
    c.footer("Data: target/research/csv; x265 rows from bench/complexity/hybrid_{samtime,drone}.txt")
    c.finish(outdir / f"{name}.svg")
    print(name, {k: (round(v[0], 2), round(v[1], 1)) for k, v in newp.items()})
    return c


def bars(outdir):
    """Per-clip ultraheavy hybrid BD, old vs new vs AQ."""
    old = load_tag(CSV / "old.csv")
    new = load_tag(CSV / "new.csv")
    aq = load_tag(CSV / "aq.csv")
    xrows = load_x265()
    # reuse Chart's canvas but draw vertical bars
    c = Chart("Per-clip hybrid BD-rate, ultraheavy", "20 frames, 720p; positive = fewer bits", CLIPS,
              yrange=(-40, 25), xrange=(1, 6))
    c.header()
    # axes are custom here
    left, right, top, bottom = 260, 1900, 380, 1230
    c.svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    c.svg.append(f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    for y in range(-40, 21, 5):
        yy = bottom - (bottom - top) * (y + 40) / 65
        c.svg.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD'/>")
        c.svg.append(c.text(left - 20, yy + 8, 24, anchor="end") + f"{y:+d}%</text>")
    zero = bottom - (bottom - top) * 40 / 65
    c.svg.append(f"<line x1='{left}' y1='{zero:.1f}' x2='{right}' y2='{zero:.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")
    width = (right - left) / len(CLIPS)
    for i, clip in enumerate(CLIPS):
        vals = []
        for tag in (old, aq, new):
            test = tag.get((clip, "ultraheavy"), [])
            h = hybrid_bd(xrows[(clip, "medium")], test) if test else None
            vals.append(h)
        x0 = left + i * width + width * 0.18
        bw = width * 0.20
        for j, (v, colour) in enumerate(zip(vals, ["#777777", "#E08A1E", "#C62828"])):
            if v is None:
                continue
            y = bottom - (bottom - top) * (v + 40) / 65
            x = x0 + j * bw
            y0 = zero
            c.svg.append(f"<rect x='{x:.1f}' y='{min(y,y0):.1f}' width='{bw*0.9:.1f}' "
                         f"height='{abs(y0-y):.1f}' fill='{colour}'/>")
            c.svg.append(c.text(x + bw * 0.45, y - 10 if v >= 0 else y + 26, 20, anchor="middle")
                         + f"{v:+.1f}</text>")
        c.svg.append(c.text(left + i * width + width / 2, bottom + 40, 27, anchor="middle") + clip + "</text>")
    c.svg.append(c.text((left + right) / 2, bottom + 100, 26, anchor="middle") + "clip (bars: old / AQ only / new)</text>")
    c.finish(outdir / "bd_bars.svg")
    print("wrote bd_bars.svg")


def rd(outdir):
    """VMAF against bitrate for a720 and forest, old/new medium+ultraheavy."""
    old = load_tag(CSV / "old.csv")
    new = load_tag(CSV / "new.csv")
    xrows = load_x265()
    W, H = 2000, 1500
    left, right, top, bottom = 250, 1900, 380, 1300
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
           f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
           f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>"]
    font = "font-family='Inter, Helvetica, Arial, sans-serif'"
    svg.append(f"<text x='70' y='90' {font} font-size='44' font-weight='700'>Rate-quality curves: old vs new</text>")
    svg.append(f"<text x='70' y='137' {font} font-size='24' fill='#555'>VMAF against bitrate, 20 frames, 720p; solid = new, dashed = old</text>")
    panels = [("a720", 250, 460, 830, 1220), ("forest", 1120, 460, 1700, 1220)]
    # common log-rate mapping 20..12000 kbps, VMAF 20..100
    xmin, xmax, ymin, ymax = 20.0, 12000.0, 20.0, 100.0
    px = lambda x: 0
    for (clip, xl, xt, xr, xb) in panels:
        svg.append(f"<line x1='{xl}' y1='{xt}' x2='{xl}' y2='{xb}' stroke='#222' stroke-width='2'/>")
        svg.append(f"<line x1='{xl}' y1='{xb}' x2='{xr}' y2='{xb}' stroke='#222' stroke-width='2'/>")
        def pxx(k):
            return xl + (xr - xl) * (math.log10(k) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        def pyy(v):
            return xb - (xb - xt) * (v - ymin) / (ymax - ymin)
        for k in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
            xx = pxx(k)
            svg.append(f"<line x1='{xx:.1f}' y1='{xt}' x2='{xx:.1f}' y2='{xb}' stroke='#EEE'/>")
            svg.append(f"<text x='{xx:.1f}' y='{xb+36}' {font} font-size='22' fill='#333' text-anchor='middle'>{k}</text>")
        for v in range(30, 100, 10):
            yy = pyy(v)
            svg.append(f"<line x1='{xl}' y1='{yy:.1f}' x2='{xr}' y2='{yy:.1f}' stroke='#EEE'/>")
            svg.append(f"<text x='{xl-18}' y='{yy+7:.1f}' {font} font-size='22' fill='#333' text-anchor='end'>{v}</text>")
        svg.append(f"<text x='{(xl+xr)/2:.0f}' y='{xb+78}' {font} font-size='25' fill='#333' text-anchor='middle'>kbit/s</text>")
        svg.append(f"<text x='{xl}' y='{xt-20}' {font} font-size='28' font-weight='600'>{clip}</text>")
        for tag, colour, dash, label in [(old, "#777777", "8 7", "old"), (new, "#C62828", "", "new")]:
            for tier, marker in [("medium", "circle"), ("ultraheavy", "square")]:
                pts = sorted((k, v) for k, v, y, m, q in tag.get((clip, tier), []))
                if len(pts) < 2:
                    continue
                coords = [(pxx(k), pyy(v)) for k, v in pts]
                svg.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                           + f"' fill='none' stroke='{colour}' stroke-width='4' stroke-dasharray='{dash}'/>")
                for x, y in coords:
                    if marker == "circle":
                        svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='9' fill='{colour}' stroke='#222' stroke-width='1.5'/>")
                    else:
                        svg.append(f"<rect x='{x-8:.1f}' y='{y-8:.1f}' width='16' height='16' fill='{colour}' stroke='#222' stroke-width='1.5'/>")
        # x265 medium
        pts = sorted((k, v) for k, v, y, m, q in xrows[(clip, "medium")])
        coords = [(pxx(k), pyy(v)) for k, v in pts]
        svg.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                   + "' fill='none' stroke='#2E8B57' stroke-width='4' stroke-dasharray='2 6'/>")
    svg.append(f"<text x='250' y='1380' {font} font-size='20' fill='#666'>green = x265 medium; red = new (AQ + perceptual filter); grey = old; circle = medium, square = ultraheavy</text>")
    svg.append("</svg>")
    (outdir / "rd_curves.svg").write_text("\n".join(svg))
    print("wrote rd_curves.svg")


def free_chart(outdir, name, clips, title, yrange):
    """Old / AQ+filter / adaptive-tilt Pareto for one clip group."""
    old = load_tag(CSV / "old.csv")
    new = load_tag(CSV / "new.csv")
    free = load_tag(CSV / "free.csv")
    if not free:
        return None
    xrows = load_x265()
    told = load_ms(CSV / "timing_old.csv")
    tnew = load_ms(CSV / "timing_new.csv")
    # The tilt changes only the sequence-header weights, not any code path, so it cannot change
    # encode time. Use the same timing run as `new`; a separate timing_free run was taken under a
    # different desktop load and is not comparable.
    oldp = points(old, xrows, clips, times=told or None)
    newp = points(new, xrows, clips, times=tnew or None)
    freep = points(free, xrows, clips, times=tnew or None)
    xp = x265_points(xrows, clips)
    c = Chart(title, f"{len(clips)} clips ({', '.join(clips)}), 20 frames, 720p, 4 threads",
              clips, yrange=yrange,
              legend=[("#777777", "before"), ("#C62828", "AQ+filter"),
                      ("#1565C0", "+ tilt"), ("#2E8B57", "x265")],
              note="Before: VERSION 9 baseline. AQ + filter: the 2026-10-26 perceptual defaults. "
                   "Adaptive tilt: finest-level step multiplier chosen from qp (free, encoder-only).")
    c.header()
    c.axes()
    c.draw_family(oldp, "#777777", {"ultraheavy": "before ultra"}, "old",
                  {"ultraheavy": (60, 40)})
    c.draw_family(newp, "#C62828", {"ultraheavy": "AQ+filter ultra"}, "new",
                  {"ultraheavy": (48, -34)})
    c.draw_family(freep, "#1565C0", {"ultraheavy": "tilt ultra"}, "free",
                  {"ultraheavy": (-110, -22)})
    c.draw_x265(xp)
    c.footer("Data: target/research/csv; hybrid = 70% VMAF (vmaf_v0.6.1) + 30% YUV-PSNR BD-rate; "
             "encode times shared with AQ+filter (weights-only change)")
    c.finish(outdir / f"{name}.svg")
    print(name, {k: (round(v[0], 2), round(v[1], 1)) for k, v in freep.items()})
    return c


def free_bars(outdir):
    """Per-clip ultraheavy hybrid BD, before vs AQ+filter vs adaptive tilt."""
    old = load_tag(CSV / "old.csv")
    new = load_tag(CSV / "new.csv")
    free = load_tag(CSV / "free.csv")
    if not free:
        return
    xrows = load_x265()
    c = Chart("Per-clip hybrid BD-rate, ultraheavy", "20 frames, 720p; positive = fewer bits",
              CLIPS, yrange=(-45, 30), xrange=(1, 6),
              legend=[("#777777", "before"), ("#C62828", "AQ+filter"), ("#1565C0", "+ tilt")],
              note="Before: VERSION 9 baseline. AQ+filter: causal frame AQ + perceptual in-loop "
                   "filter. + tilt: adaptive finest-level step weights.")
    c.header()
    left, right, top, bottom = 260, 1900, 380, 1230
    c.svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    c.svg.append(f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    lo, hi = -45, 30
    for y in range(lo, hi + 1, 5):
        yy = bottom - (bottom - top) * (y - lo) / (hi - lo)
        c.svg.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD'/>")
        c.svg.append(c.text(left - 20, yy + 8, 24, anchor="end") + f"{y:+d}%</text>")
    zero = bottom - (bottom - top) * (0 - lo) / (hi - lo)
    c.svg.append(f"<line x1='{left}' y1='{zero:.1f}' x2='{right}' y2='{zero:.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")
    width = (right - left) / len(CLIPS)
    for i, clip in enumerate(CLIPS):
        vals = []
        for tag in (old, new, free):
            test = tag.get((clip, "ultraheavy"), [])
            h = hybrid_bd(xrows[(clip, "medium")], test) if test else None
            vals.append(h)
        x0 = left + i * width + width * 0.14
        bw = width * 0.22
        for j, (v, colour) in enumerate(zip(vals, ["#777777", "#C62828", "#1565C0"])):
            if v is None:
                continue
            y = bottom - (bottom - top) * (v - lo) / (hi - lo)
            x = x0 + j * bw
            c.svg.append(f"<rect x='{x:.1f}' y='{min(y,zero):.1f}' width='{bw*0.88:.1f}' "
                         f"height='{abs(zero-y):.1f}' fill='{colour}'/>")
            c.svg.append(c.text(x + bw * 0.44, y - 10 if v >= 0 else y + 26, 20, anchor="middle")
                         + f"{v:+.1f}</text>")
        c.svg.append(c.text(left + i * width + width / 2, bottom + 40, 27, anchor="middle") + clip + "</text>")
    c.svg.append(c.text((left + right) / 2, bottom + 100, 26, anchor="middle")
                 + "clip (bars: before / AQ+filter / adaptive tilt)</text>")
    c.finish(outdir / "bd_bars_free.svg")
    print("wrote bd_bars_free.svg")


def chart_free2(outdir, name, clips, title, yrange):
    """Before / AQ+filter+tilt / band-CDF build Pareto for one clip group."""
    old = load_tag(CSV / "old.csv")
    free = load_tag(CSV / "free.csv")
    free2 = load_tag(CSV / "free2.csv")
    if not free2:
        return None
    xrows = load_x265()
    told = load_ms(CSV / "timing_old.csv")
    tnew = load_ms(CSV / "timing_new.csv")
    oldp = points(old, xrows, clips, times=told or None)
    freep = points(free, xrows, clips, times=tnew or None)
    f2p = points(free2, xrows, clips, times=tnew or None)
    xp = x265_points(xrows, clips)
    c = Chart(title, f"{len(clips)} clips ({', '.join(clips)}), 20 frames, 720p, 4 threads",
              clips, yrange=yrange,
              legend=[("#777777", "before"), ("#C62828", "tilt"),
                      ("#1565C0", "+ band"), ("#2E8B57", "x265")],
              note="Before: VERSION 9 baseline. AQ+filter+tilt: the 2026-10-26 perceptual defaults. "
                   "+ band CDFs: VERSION 10 per-band entropy contexts and B qp 11.")
    c.header()
    c.axes()
    c.draw_family(oldp, "#777777", {"ultraheavy": "before ultra"}, "old",
                  {"ultraheavy": (60, 40)})
    c.draw_family(freep, "#C62828", {"ultraheavy": "tilt ultra"}, "free",
                  {"ultraheavy": (48, -34)})
    c.draw_family(f2p, "#1565C0", {"ultraheavy": "band ultra"}, "free2",
                  {"ultraheavy": (-100, -22)})
    c.draw_x265(xp)
    c.footer("Data: target/research/csv; hybrid = 70% VMAF (vmaf_v0.6.1) + 30% YUV-PSNR BD-rate")
    c.finish(outdir / f"{name}.svg")
    print(name, {k: (round(v[0], 2), round(v[1], 1)) for k, v in f2p.items()})
    return c


def bars_free2(outdir):
    """Per-clip ultraheavy hybrid BD, before vs tilt vs band-CDF builds."""
    old = load_tag(CSV / "old.csv")
    free = load_tag(CSV / "free.csv")
    free2 = load_tag(CSV / "free2.csv")
    if not free2:
        return
    xrows = load_x265()
    c = Chart("Per-clip hybrid BD-rate, ultraheavy", "20 frames, 720p; positive = fewer bits",
              CLIPS, yrange=(-45, 30), xrange=(1, 6),
              legend=[("#777777", "before"), ("#C62828", "tilt"), ("#1565C0", "band CDFs")],
              note="Before: VERSION 9 baseline. Tilt: adaptive finest level. Band CDFs: VERSION 10 "
                   "per-band entropy contexts and B qp 11.")
    c.header()
    left, right, top, bottom = 260, 1900, 380, 1230
    c.svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    c.svg.append(f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    lo, hi = -45, 30
    for y in range(lo, hi + 1, 5):
        yy = bottom - (bottom - top) * (y - lo) / (hi - lo)
        c.svg.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD'/>")
        c.svg.append(c.text(left - 20, yy + 8, 24, anchor="end") + f"{y:+d}%</text>")
    zero = bottom - (bottom - top) * (0 - lo) / (hi - lo)
    c.svg.append(f"<line x1='{left}' y1='{zero:.1f}' x2='{right}' y2='{zero:.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")
    width = (right - left) / len(CLIPS)
    for i, clip in enumerate(CLIPS):
        vals = []
        for tag in (old, free, free2):
            test = tag.get((clip, "ultraheavy"), [])
            vals.append(hybrid_bd(xrows[(clip, "medium")], test) if test else None)
        x0 = left + i * width + width * 0.14
        bw = width * 0.22
        for j, (v, colour) in enumerate(zip(vals, ["#777777", "#C62828", "#1565C0"])):
            if v is None:
                continue
            y = bottom - (bottom - top) * (v - lo) / (hi - lo)
            x = x0 + j * bw
            c.svg.append(f"<rect x='{x:.1f}' y='{min(y,zero):.1f}' width='{bw*0.88:.1f}' "
                         f"height='{abs(zero-y):.1f}' fill='{colour}'/>")
            c.svg.append(c.text(x + bw * 0.44, y - 10 if v >= 0 else y + 26, 20, anchor="middle")
                         + f"{v:+.1f}</text>")
        c.svg.append(c.text(left + i * width + width / 2, bottom + 40, 27, anchor="middle") + clip + "</text>")
    c.svg.append(c.text((left + right) / 2, bottom + 100, 26, anchor="middle")
                 + "clip (bars: before / tilt / band CDFs)</text>")
    c.finish(outdir / "bd_bars_free2.svg")
    print("wrote bd_bars_free2.svg")


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    chart(outdir, "pareto_old_new", CLIPS, "VDWT old vs new: compression at equal quality")
    chart(outdir, "pareto_samtime_old_new", SAMTIME, "Samtime (2 clips): old vs new", (-45, 15))
    chart(outdir, "pareto_drone_old_new", DRONE, "Drone videos (3 clips): old vs new", (-10, 20))
    bars(outdir)
    rd(outdir)
    if (CSV / "free.csv").exists():
        free_chart(outdir, "pareto_free_old_new", CLIPS,
                   "VDWT before / after: free bitrate gains", (-22, 20))
        free_chart(outdir, "pareto_samtime_free_old_new", SAMTIME,
                   "Samtime (2 clips): before / after free gains", (-45, 12))
        free_chart(outdir, "pareto_drone_free_old_new", DRONE,
                   "Drone videos (3 clips): before / after free gains", (-10, 22))
        free_bars(outdir)
    if (CSV / "free2.csv").exists():
        chart_free2(outdir, "pareto_free2_old_new", CLIPS,
                    "VDWT before / after: entropy free gains", (-22, 20))
        chart_free2(outdir, "pareto_samtime_free2_old_new", SAMTIME,
                    "Samtime (2 clips): before / after free gains", (-45, 12))
        chart_free2(outdir, "pareto_drone_free2_old_new", DRONE,
                    "Drone videos (3 clips): before / after free gains", (-10, 22))
        bars_free2(outdir)


if __name__ == "__main__":
    main()
