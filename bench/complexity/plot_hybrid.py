#!/usr/bin/env python3
"""Plot the hybrid VMAF/YUV Pareto: 70% VMAF + 30% YUV-PSNR BD-rate saving.

    plot_hybrid.py samtime OUT.svg [clean] [decode]
    plot_hybrid.py drone   OUT.svg [clean] [decode]

The full chart shows every VDWT tier and every x265 preset; the clean chart keeps only
VDWT ultraheavy and x265 medium/slow/veryslow for slides. The optional `decode` word reads
`decode_samtime.txt` / `decode_drone.txt` and puts decode ms/frame on the x-axis instead of
encode time. Both use the same hybrid metric: positive means VDWT needs fewer bits.
"""
import math, sys
from collections import defaultdict
sys.path.insert(0, "bench/complexity")
from plot import bd_rate

VDWT_TIERS = ["realtime", "light", "medium", "heavy", "ultraheavy"]
X265_LABELS = {"ultrafast", "medium", "slow", "veryslow"}


def load(path):
    d = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        if not line.strip():
            continue
        c, e, q, k, v, y, m = line.split()
        d[e][c].append((float(k), float(v), float(y), float(m)))
    return d


def decode_times(path):
    total = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        _, e, _, ms = line.split()
        total[e].append(float(ms))
    return {e: sum(v) / len(v) for e, v in total.items()}


def savings(d):
    clips = sorted({c for e in d.values() for c in e})
    ref = d["x265-medium"]
    out = {}
    for e, per in d.items():
        bv, by, ms = [], [], []
        for c in clips:
            if c not in per or c not in ref:
                continue
            rv = sorted((b, v) for b, v, y, m in ref[c])
            ry = sorted((b, y) for b, v, y, m in ref[c])
            tv = sorted((b, v) for b, v, y, m in per[c])
            ty = sorted((b, y) for b, v, y, m in per[c])
            a, b = bd_rate(rv, tv), bd_rate(ry, ty)
            if a is None or b is None:
                continue
            bv.append(a)
            by.append(b)
            ms += [m for *_, m in per[c]]
        if bv:
            sv, sy = -sum(bv) / len(bv), -sum(by) / len(by)
            out[e] = (0.7 * sv + 0.3 * sy, sv, sy, sum(ms) / len(ms))
    return clips, out


class Panel:
    def __init__(self, title, subtitle, yrange, clips, xrange=(8.0, 600.0),
                 xticks=(10, 20, 50, 100, 200, 500),
                 xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86)"):
        self.W, self.H = 2000, 1400
        self.left, self.right, self.top, self.bottom = 250, 1915, 380, 1230
        self.title = title
        self.subtitle = subtitle
        self.clips = clips
        (self.ymin, self.ymax) = yrange
        self.xmin, self.xmax = xrange
        self.xticks = xticks
        self.xlabel = xlabel
        self.font = "font-family='Inter, Helvetica, Arial, sans-serif'"
        self.svg = []

    def px(self, x):
        return self.left + (self.right - self.left) * (
            math.log10(x) - math.log10(self.xmin)
        ) / (math.log10(self.xmax) - math.log10(self.xmin))

    def py(self, y):
        return self.bottom - (self.bottom - self.top) * (y - self.ymin) / (
            self.ymax - self.ymin
        )

    def text(self, x, y, size, fill="#222", weight="400", anchor="start", extra=""):
        return (
            f"<text x='{x:.1f}' y='{y:.1f}' {self.font} font-size='{size}' "
            f"font-weight='{weight}' fill='{fill}' text-anchor='{anchor}' {extra}>"
        )

    def header(self):
        s = self.svg
        s.append(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.W}' height='{self.H}' "
            f"viewBox='0 0 {self.W} {self.H}'>"
        )
        s.append(f"<rect width='{self.W}' height='{self.H}' fill='#FAF9F5'/>")
        s.append(
            f"<rect x='4' y='4' width='{self.W-8}' height='{self.H-8}' fill='none' "
            "stroke='#222' stroke-width='3'/>"
        )
        s.append(self.text(70, 90, 44, weight="700") + f"{self.title}</text>")
        s.append(
            self.text(70, 137, 24, fill="#555")
            + f"{len(self.clips)} clips ({', '.join(self.clips)}), 20 frames each, 720p; "
            "hybrid BD-rate vs x265 medium</text>"
        )
        s.append(
            self.text(70, 171, 24, fill="#555")
            + "70% VMAF + 30% YUV-PSNR; VMAF vmaf_v0.6.1; VDWT v13 vs x265 --tune psnr, "
            "4 threads</text>"
        )
        s.append(
            self.text(70, 205, 22, fill="#777")
            + "Positive = VDWT needs fewer bits at equal hybrid quality; negative = more bits</text>"
        )
        # legend
        s.append(f"<circle cx='262' cy='277' r='16' fill='#1A237E' stroke='#222' stroke-width='2.5'/>")
        s.append(self.text(287, 287, 28) + "VDWT v13</text>")
        s.append(f"<circle cx='505' cy='277' r='16' fill='#2E8B57' stroke='#222' stroke-width='2.5'/>")
        s.append(self.text(530, 287, 28) + "x265</text>")
        if self.subtitle:
            s.append(self.text(730, 287, 23, fill="#555") + f"{self.subtitle}</text>")

    def axes(self):
        s = self.svg
        s.append(f"<line x1='{self.left}' y1='{self.top}' x2='{self.left}' y2='{self.bottom}' stroke='#222' stroke-width='2'/>")
        s.append(f"<line x1='{self.left}' y1='{self.bottom}' x2='{self.right}' y2='{self.bottom}' stroke='#222' stroke-width='2'/>")
        step = 10
        start = math.ceil(self.ymin / step) * step
        for y in range(start, int(self.ymax) + 1, step):
            yy = self.py(y)
            s.append(f"<line x1='{self.left}' y1='{yy:.1f}' x2='{self.right}' y2='{yy:.1f}' stroke='#DDD' stroke-width='1'/>")
            s.append(self.text(self.left - 22, yy + 8, 24, fill="#333", anchor="end") + f"{y:+d}%</text>")
        for x in self.xticks:
            xx = self.px(x)
            s.append(f"<line x1='{xx:.1f}' y1='{self.top}' x2='{xx:.1f}' y2='{self.bottom}' stroke='#EEE' stroke-width='1'/>")
            s.append(self.text(xx, self.bottom + 40, 24, fill="#333", anchor="middle") + f"{x}</text>")
        s.append(self.text((self.left + self.right) / 2, self.bottom + 98, 26, fill="#333", anchor="middle")
                 + f"{self.xlabel}</text>")
        s.append(
            f"<text x='55' y='{(self.top+self.bottom)/2:.0f}' {self.font} font-size='26' "
            f"text-anchor='middle' fill='#333' transform='rotate(-90 55 {(self.top+self.bottom)/2:.0f})'>"
            "Hybrid bitrate saving vs x265 medium (%)</text>"
        )
        s.append(f"<line x1='{self.left}' y1='{self.py(0):.1f}' x2='{self.right}' y2='{self.py(0):.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")

    def family(self, pts, colour, label_all=False, labels=None):
        """One encoder family, sorted by time. `pts` are (name, (hybrid, vmaf, yuv, ms))."""
        s = self.svg
        pts = sorted(pts, key=lambda q: q[1][3])
        coords = [(self.px(v[3]), self.py(v[0]), n, v) for n, v in pts]
        s.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
                 + f"' fill='none' stroke='{colour}' stroke-width='5' stroke-linejoin='round'/>")
        for x, y, name, v in coords:
            labelled = label_all or (labels is not None and name in labels)
            r = 16 if labelled else 11
            s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{colour}' stroke='#222' stroke-width='2.5'/>")

    def label(self, x, y, name, colour, dx=0, dy=-24, size=27):
        # A pale halo keeps the label readable over a line or another point.
        halo = "paint-order='stroke' stroke='#FAF9F5' stroke-width='7' stroke-linejoin='round'"
        self.svg.append(
            f"<text x='{x+dx:.1f}' y='{y+dy:.1f}' {self.font} font-size='{size}' fill='{colour}' "
            f"text-anchor='middle' {halo}>{name}</text>"
        )

    def footer(self, source):
        self.svg.append(self.text(70, 1345, 19, fill="#888") + f"{source}</text>")

    def finish(self, path):
        self.svg.append("</svg>")
        open(path, "w").write("\n".join(self.svg))


def draw(kind, out, clean=False, decode=False):
    group = kind.split("_")[0]
    heldout = kind.endswith("_heldout")
    path = f"bench/complexity/hybrid_{kind}.txt"
    if group == "samtime":
        title = "Samtime: hybrid VMAF + YUV-PSNR"
        yrange = (-30, 15) if clean else (-55, 20)
    else:
        title = "Drone: hybrid VMAF + YUV-PSNR"
        yrange = (-10, 22) if clean else (-42, 22)
    if heldout:
        title += " — held-out contexts"
    d = load(path)
    clips, s = savings(d)
    if decode:
        title += " — decode"
        dt = decode_times(f"bench/complexity/decode_{kind}.txt")
        s = {e: (h, sv, sy, dt[e]) for e, (h, sv, sy, _) in s.items() if e in dt}
        xrange = (0.8, 6.0)
        xticks = (1, 2, 5)
        xlabel = "Decode time per frame (ms, log; 720p, 4 threads, x86)"
        subtitle = ("decode: in-process VDWT, FFmpeg x265; "
                    + ("ultraheavy vs key presets" if clean else "all tiers and presets"))
    else:
        xrange = (8.0, 600.0)
        xticks = (10, 20, 50, 100, 200, 500)
        xlabel = "Encode time per frame (ms, log; 720p, 4 threads, x86)"
        subtitle = "clean: ultraheavy vs key x265 presets" if clean else "all tiers and presets"
    p = Panel(title, subtitle, yrange, clips, xrange, xticks, xlabel)
    p.header()
    p.axes()

    vd_names = ["ultraheavy"] if clean else VDWT_TIERS
    x_names = ["medium", "slow", "veryslow"] if clean else sorted(
        e.replace("x265-", "") for e in s if e.startswith("x265-")
    )
    vd = [(n, s["vdwt-" + n]) for n in vd_names if "vdwt-" + n in s]
    xp = [("x265-" + n, s["x265-" + n]) for n in x_names if "x265-" + n in s]

    # x265 behind the VDWT markers.
    p.family(xp, "#2E8B57", label_all=False, labels=set(x_names))
    p.family(vd, "#1A237E", label_all=True)

    # Label positions: hand-tuned once, then verified on the rendered PNG. Decode-time charts
    # cluster horizontally (preset does not change x265 decode speed much), so they need their
    # own offsets to keep every label readable.
    if decode and clean:
        if group == "samtime":
            vd_pos = {"ultraheavy": (-50, -28)}
            x_pos = {"medium": (-60, 34), "slow": (-60, 34), "veryslow": (55, -34)}
        else:
            vd_pos = {"ultraheavy": (50, 34)}
            x_pos = {"medium": (-60, 34), "slow": (-60, -30), "veryslow": (60, 34)}
        for n, v in vd:
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = vd_pos.get(n, (0, -28))
            p.label(x, y, n, "#1A237E", dx=dx, dy=dy)
        for n, v in xp:
            name = n.replace("x265-", "")
            if name not in X265_LABELS:
                continue
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = x_pos.get(name, (30, 30))
            p.label(x, y, name, "#2E8B57", dx=dx, dy=dy, size=26)
    elif decode:
        if group == "samtime":
            vd_pos = {"realtime": (-52, -30), "light": (-55, -30), "medium": (-55, -30),
                      "heavy": (50, -30), "ultraheavy": (50, -30)}
            x_pos = {"ultrafast": (40, 34), "medium": (-60, 34), "slow": (50, -30),
                     "veryslow": (75, -32)}
        else:
            vd_pos = {"realtime": (-55, 34), "light": (-50, -30), "medium": (50, -32),
                      "heavy": (-55, -30), "ultraheavy": (50, -30)}
            x_pos = {"ultrafast": (40, 34), "medium": (-60, 34), "slow": (-60, -30),
                     "veryslow": (60, 34)}
        for n, v in vd:
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = vd_pos.get(n, (0, -28))
            p.label(x, y, n, "#1A237E", dx=dx, dy=dy)
        for n, v in xp:
            name = n.replace("x265-", "")
            if name not in X265_LABELS:
                continue
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = x_pos.get(name, (30, 30))
            p.label(x, y, name, "#2E8B57", dx=dx, dy=dy, size=26)
    elif clean:
        for n, v in vd:
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = ((-75, 40) if group == "drone" else (-50, -28))
            p.label(x, y, n, "#1A237E", dx=dx, dy=dy)
        for n, e in [("medium", "x265-medium"), ("slow", "x265-slow"), ("veryslow", "x265-veryslow")]:
            if e not in s:
                continue
            x, y = p.px(s[e][3]), p.py(s[e][0])
            dx, dy = {"medium": (30, 34), "slow": (34, -30), "veryslow": (-20, 34)}[n]
            p.label(x, y, n, "#2E8B57", dx=dx, dy=dy)
    else:
        vpos = {
            "realtime": (34, 36) if group == "drone" else (34, -24),
            "light": (-40, -30) if group == "drone" else (-40, -24),
            "medium": (42, -28), "heavy": (44, -30),
            "ultraheavy": (44, 36) if group == "drone" else (44, -30),
        }
        for n, v in vd:
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = vpos.get(n, (0, -28))
            p.label(x, y, n, "#1A237E", dx=dx, dy=dy)
        xpos = {
            "ultrafast": (30, 36), "superfast": (30, 30), "veryfast": (-46, 30),
            "faster": (30, -24), "fast": (30, 30),
            "medium": (30, 34) if group == "drone" else (36, -30),
            "slow": (34, -28), "slower": (34, 30), "veryslow": (-20, 34),
        }
        for n, v in xp:
            if n.replace("x265-", "") not in X265_LABELS:
                continue
            name = n.replace("x265-", "")
            x, y = p.px(v[3]), p.py(v[0])
            dx, dy = xpos.get(name, (30, 30))
            p.label(x, y, name, "#2E8B57", dx=dx, dy=dy, size=26)

    source = path.split("/")[-1]
    if decode:
        source += " + decode_%s.txt" % kind
    src = "Data: bench/complexity/%s; generated 2026-10-25." % source
    p.footer(src)
    p.finish(out)
    print(out)
    for e, (h, sv, sy, t) in sorted(s.items(), key=lambda q: -q[1][0]):
        print(f"{e:22s} hybrid {h:+6.2f}%  vmaf {sv:+6.2f}%  yuv {sy:+6.2f}%  {t:7.2f} ms")


if __name__ == "__main__":
    flags = set(sys.argv[3:])
    draw(sys.argv[1], sys.argv[2], "clean" in flags, "decode" in flags)
