#!/usr/bin/env python3
"""Tasmania: VMAF-only encode/decode Pareto for the post-filtered comparison."""
import math, sys
from collections import defaultdict

ROOT = "bench/complexity"
VDWT = ["realtime", "light", "medium", "heavy", "ultraheavy"]
X265 = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]


def load_enc():
    d = defaultdict(list)
    for line in open(f"{ROOT}/tassie_pf_encode.txt"):
        c, e, q, k, v, y, m = line.split()
        d[e].append((float(k), float(v), float(y), float(m)))
    return d


def load_dec():
    d = {}
    for line in open(f"{ROOT}/tassie_pf_decode.txt"):
        c, e, q, ms = line.split()
        d.setdefault(e, []).append(float(ms))
    return {e: sum(v) / len(v) for e, v in d.items()}


def interp(curve, x):
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return None


def bd_rate(ref, test):
    """Bits saved by test at equal quality (%), positive = test better."""
    a = sorted((q, math.log(r)) for r, q in ref if r > 0)
    b = sorted((q, math.log(r)) for r, q in test if r > 0)
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi - lo < 0.5:
        return None
    xs = [lo + (hi - lo) * i / 200 for i in range(201)]
    diff = [interp(b, x) - interp(a, x) for x in xs]
    return -100 * (math.exp(sum(diff) / len(diff)) - 1)


def scores(enc):
    ref = [(k, v) for k, v, y, m in enc["x265-medium"]]
    out = {}
    for e, rows in enc.items():
        v = bd_rate(ref, [(k, v) for k, v, y, m in rows])
        out[e] = (v, sum(m for *_, m in rows) / len(rows))
    return out


def draw(path, mode, sc, dec):
    W, H = 2000, 1400
    left, right, top, bottom = 250, 1915, 390, 1215
    if mode == "encode":
        xmin, xmax = 8.0, 600.0
        xticks = (10, 20, 50, 100, 200, 500)
        xlabel = "Encode time per frame (ms, log; 720p, 4 threads, x86)"
    else:
        xmin, xmax = 0.8, 8.0
        xticks = (1, 2, 5)
        xlabel = "Decode time per frame (ms, log; 720p, 4 threads, x86; filter included)"
    ymin, ymax = -45, 20
    px = lambda x: left + (right - left) * (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
    py = lambda y: bottom - (bottom - top) * (y - ymin) / (ymax - ymin)
    font = "font-family='Inter, Helvetica, Arial, sans-serif'"
    title = f"Tasmania: post-filtered VMAF {mode} Pareto"
    subtitle = ("custom band 2.0 on both codecs; 20-frame quality, 60-frame timing; "
                "VMAF only, not the 70/30 hybrid")
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
         f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
         f"<text x='70' y='92' {font} font-size='44' font-weight='700' fill='#222'>{title}</text>",
         f"<text x='70' y='137' {font} font-size='23' fill='#555'>{subtitle}</text>",
         f"<text x='70' y='171' {font} font-size='21' fill='#777'>Positive = fewer bits than x265 medium+filter at equal VMAF</text>",
         f"<circle cx='262' cy='270' r='16' fill='#1A237E' stroke='#222' stroke-width='2.5'/>",
         f"<text x='287' y='280' {font} font-size='27' fill='#222'>VDWT + custom band</text>",
         f"<circle cx='585' cy='270' r='16' fill='#2E8B57' stroke='#222' stroke-width='2.5'/>",
         f"<text x='610' y='280' {font} font-size='27' fill='#222'>x265 + custom band</text>"]
    s += [f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>",
          f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>"]
    for y in range(-40, 21, 10):
        yy = py(y)
        s.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD' stroke-width='1'/>")
        s.append(f"<text x='{left-22}' y='{yy+8:.1f}' {font} font-size='24' text-anchor='end' fill='#333'>{y:+d}%</text>")
    for x in xticks:
        xx = px(x)
        s.append(f"<line x1='{xx:.1f}' y1='{top}' x2='{xx:.1f}' y2='{bottom}' stroke='#EEE' stroke-width='1'/>")
        s.append(f"<text x='{xx:.1f}' y='{bottom+42}' {font} font-size='24' text-anchor='middle' fill='#333'>{x}</text>")
    s.append(f"<text x='{(left+right)/2:.0f}' y='{bottom+100}' {font} font-size='26' text-anchor='middle' fill='#333'>{xlabel}</text>")
    s.append(f"<text x='55' y='{(top+bottom)/2:.0f}' {font} font-size='26' text-anchor='middle' fill='#333' transform='rotate(-90 55 {(top+bottom)/2:.0f})'>VMAF bitrate saving vs x265 medium+filter (%)</text>")
    s.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{right}' y2='{py(0):.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")

    def family(names, colour):
        pts = sorted(((n, sc[n], dec[n] if mode == "decode" else sc[n][1]) for n in names), key=lambda q: q[2])
        coords = [(px(x), py(v[0]), n) for n, v, x in pts]
        s.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords) +
                 f"' fill='none' stroke='{colour}' stroke-width='5' stroke-linejoin='round'/>")
        for x, y, n in coords:
            s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='{colour}' stroke='#222' stroke-width='2.5'/>")

    family([f"x265-{p}" for p in X265], "#2E8B57")
    family([f"vdwt-{t}" for t in VDWT], "#1A237E")
    halo = "paint-order='stroke' stroke='#FAF9F5' stroke-width='7' stroke-linejoin='round'"

    def label(n, dx, dy, colour):
        if n not in sc:
            return
        x = px(dec[n] if mode == "decode" else sc[n][1])
        y = py(sc[n][0])
        s.append(f"<text x='{x+dx:.1f}' y='{y+dy:.1f}' {font} font-size='26' fill='{colour}' text-anchor='middle' {halo}>{n.replace('vdwt-','').replace('x265-','')}</text>")

    if mode == "encode":
        pos = {"realtime": (34, 34), "light": (30, -28), "medium": (36, 30), "heavy": (40, -28), "ultraheavy": (48, -30)}
        for t in VDWT:
            label(f"vdwt-{t}", *pos[t], "#1A237E")
        xpos = {"ultrafast": (30, 34), "medium": (38, 34), "slow": (38, -28), "veryslow": (-45, -28)}
        for p in xpos:
            label(f"x265-{p}", *xpos[p], "#2E8B57")
    else:
        pos = {"realtime": (40, 34), "light": (-50, 34), "medium": (-50, -30), "heavy": (40, -30), "ultraheavy": (-50, -30)}
        for t in VDWT:
            label(f"vdwt-{t}", *pos[t], "#1A237E")
        xpos = {"ultrafast": (40, 34), "medium": (40, -30), "slow": (40, -30), "veryslow": (50, -30)}
        for p in xpos:
            label(f"x265-{p}", *xpos[p], "#2E8B57")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def main():
    enc = load_enc()
    dec = load_dec()
    sc = scores(enc)
    for e in sorted(sc, key=lambda e: sc[e][0]):
        print(f"{e:22s} {sc[e][0]:+7.2f}%")
    draw("videos/tasmania_pf_vmaf_encode.svg", "encode", sc, dec)
    draw("videos/tasmania_pf_vmaf_decode.svg", "decode", sc, dec)


if __name__ == "__main__":
    main()
