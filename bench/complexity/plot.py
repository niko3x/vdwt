#!/usr/bin/env python3
"""Compression against complexity for drone footage, in the style of an accuracy-vs-cost chart.

    plot.py RD.txt SPEED.txt OUT.svg
    plot.py --unrestricted RD6.txt OUT.svg   (RD6 lines carry a 6th field: CPU ms/frame)
    plot.py --decode RD6.txt OUT.svg         (the same, the 6th field being decode CPU ms/frame)
    plot.py --psnr|--yuv611 GROUP encode|decode RD7.txt OUT.svg
                                             (psnr_pareto.py rows: 6th field encode, 7th decode
                                             CPU ms/frame; GROUP drone, samtime or talking)
    plot.py --samtime RD6.txt OUT.svg        (the samtime podcast/stage chart)

RD.txt:    "clip encoder param kbps psnr_y" per line (one rate-quality point).
SPEED.txt: "encoder ms_per_frame" per line (encode time, 720p, 4 threads).

Each encoder's compression is its BD-rate (PSNR-Y, piecewise-linear in log rate over the
overlapping quality range) against x264 medium on every clip, averaged over the clips; the
chart shows the bitrate saving (minus the BD-rate), so higher is better. No dependencies: it
writes SVG (render with `rsvg-convert -w 2000 OUT.svg -o OUT.png`).
"""

import math
import sys
from collections import defaultdict

REFERENCE = "x264-medium"
PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
           "veryslow", "placebo"]
# Family: (label, colour, members, point labels: member -> (text, dx, dy) from the point).
UP, DOWN = -32, 52
FAMILIES = [
    ("VDWT", "#D9643A", ["vdwt-realtime", "vdwt-light", "vdwt-medium", "vdwt-heavy"],
     {"vdwt-realtime": ("realtime", -30, UP), "vdwt-light": ("light", 18, UP),
      "vdwt-medium": ("medium", -10, UP), "vdwt-heavy": ("heavy", 22, UP)}),
    ("VDWT rt (lossy links)", "#8E2C1C", ["vdwt-rt"], {"vdwt-rt": ("rt", 36, 10)}),
    ("x265", "#4CAF82", [f"x265-{p}" for p in PRESETS],
     {"x265-ultrafast": ("ultrafast", -85, 10), "x265-superfast": ("superfast", 40, DOWN),
      "x265-medium": ("medium", 30, DOWN), "x265-slow": ("slow", 0, DOWN),
      "x265-slower": ("slower", 0, DOWN), "x265-veryslow": ("veryslow", 0, DOWN),
      "x265-placebo": ("placebo", 0, DOWN)}),
    ("x264", "#E8A838", [f"x264-{p}" for p in PRESETS],
     {"x264-ultrafast": ("ultrafast", 0, UP), "x264-superfast": ("superfast", 0, DOWN),
      "x264-faster": ("faster", 30, DOWN), "x264-medium": ("medium", 18, DOWN),
      "x264-slow": ("slow", 0, DOWN), "x264-veryslow": ("veryslow", 0, DOWN),
      "x264-placebo": ("placebo", 0, DOWN)}),
    # Intel's three target usages; the driver fails levels 2, 3, 5 and 6 after one frame.
    ("HEVC hardware (Intel VAAPI)", "#5F5F5F", ["hw-1", "hw-4", "hw-7"],
     {"hw-1": ("TU1", 0, UP), "hw-4": ("TU4", -10, UP), "hw-7": ("TU7", 0, UP)}),
]
# The y range shown; points beyond it sit on the edge with their value.
Y_RANGE = (-60, 60)


def load_rd(path):
    rd = defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) in (5, 6, 7):
            try:
                rd[(f[0], f[1])].append((float(f[3]), float(f[4])))
            except ValueError:
                pass
    return rd


def interp(curve, x):
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return None


def bd_rate(ref, test):
    """BD-rate (%) of `test` against `ref`, each [(kbps, psnr)]; None without overlap."""
    a = sorted((p, math.log(r)) for r, p in ref if r > 0)
    b = sorted((p, math.log(r)) for r, p in test if r > 0)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi - lo < 0.5:
        return None
    # min(): the last sample must not pass `hi` by a rounding step (interp returns None there).
    xs = [min(lo + (hi - lo) * i / 200, hi) for i in range(201)]
    diff = [interp(b, x) - interp(a, x) for x in xs]
    return 100 * (math.exp(sum(diff) / len(diff)) - 1)


UNRESTRICTED = dict(
    reference="x265-medium",
    families=[
        ("VDWT", "#D9643A", ["vdwt-realtime", "vdwt-light", "vdwt-medium", "vdwt-heavy", "vdwt-ultraheavy"],
         {"vdwt-realtime": ("realtime", 0, DOWN), "vdwt-light": ("light", 0, DOWN),
          "vdwt-medium": ("medium", -10, UP), "vdwt-heavy": ("heavy", 10, UP),
          "vdwt-ultraheavy": ("ultraheavy", -10, UP)}),
        ("x265", "#4CAF82", [f"x265-{p}" for p in PRESETS],
         {"x265-ultrafast": ("ultrafast", 0, DOWN), "x265-superfast": ("superfast", -75, 10),
          "x265-veryfast": ("veryfast", -75, 22), "x265-faster": ("faster", -60, -8),
          "x265-fast": ("fast", -30, DOWN), "x265-medium": ("medium", 40, DOWN),
          "x265-slow": ("slow", 0, DOWN), "x265-slower": ("slower", 0, UP),
          "x265-veryslow": ("veryslow", 0, DOWN), "x265-placebo": ("placebo", 0, UP)}),
    ],
    title="Podcast, stage and drone footage",
    subtitle=["{n} clips ({clips}), 20 frames each, 720p; x265 in its zero-latency settings "
              "(no B-frames, lookahead or cutree, --tune psnr); PSNR-Y BD-rate against x265 medium",
              "VDWT at fixed qp 24-48 with its tier defaults (ultraheavy: pairs of P-frames or a "
              "B-frame group); x265 at fixed qp 22-40; single thread, 4 rates per clip"],
    xlabel="Encode time per frame (ms, log; 720p, 1 thread; startup included)",
    ylabel="Bitrate saving vs x265 medium (%)",
    yrange=(-100, 40),
    heading="Compression vs encode complexity",
)
# The pre-round HEAD, for the speed charts: same harness, 4 threads, grey and behind.
OLD_VDWT = ("VDWT old (HEAD)", "#A8A59C",
            ["vdwt-old-realtime", "vdwt-old-light", "vdwt-old-medium", "vdwt-old-heavy",
             "vdwt-old-ultraheavy"], {})
# The previous speed round, and this one, on the same harness.
SPEED_VDWT = ("VDWT current", "#D9643A",
              ["vdwt-new-realtime", "vdwt-new-light", "vdwt-new-medium", "vdwt-new-heavy",
               "vdwt-new-ultraheavy"],
              {"vdwt-new-realtime": ("realtime", -30, UP), "vdwt-new-light": ("light", 18, UP),
               "vdwt-new-medium": ("medium", -10, UP), "vdwt-new-heavy": ("heavy", 22, UP),
               "vdwt-new-ultraheavy": ("ultraheavy", -10, UP)})
NEW_VDWT = ("VDWT v12 (24 Oct)", "#880E4F",
            ["vdwt-v2-realtime", "vdwt-v2-light", "vdwt-v2-medium", "vdwt-v2-heavy",
             "vdwt-v2-ultraheavy"],
            {"vdwt-v2-realtime": ("realtime", -30, UP), "vdwt-v2-light": ("light", 18, UP),
             "vdwt-v2-medium": ("medium", -10, UP), "vdwt-v2-heavy": ("heavy", 22, UP),
             "vdwt-v2-ultraheavy": ("ultraheavy", -10, UP)})
X265 = UNRESTRICTED["families"][1]
DECODE = dict(UNRESTRICTED,
    xlabel="Decode time per frame (ms, log; 720p, 1 thread; x265 via FFmpeg; startup included)",
    heading="Compression vs decode complexity",
)
SAMTIME = dict(UNRESTRICTED,
    families=[OLD_VDWT, SPEED_VDWT, NEW_VDWT, X265],
    title="Samtime: talking head and stage",
    subtitle=["{n} clips ({clips}), 20 frames each, 720p; x265 at its preset defaults (B-frames, "
              "lookahead, cutree, scenecut); PSNR-Y BD-rate against x265 medium",
              "x265 --tune psnr, 4 threads; VDWT tiers at their defaults; fixed qp, 3 rates per clip",
              "grey: old HEAD; orange: current version; wine: v12; identical harness"],
    xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86 laptop)",
    yrange=(-60, 25),
)
DRONE = dict(UNRESTRICTED,
    families=[OLD_VDWT, SPEED_VDWT, NEW_VDWT, X265],
    title="Drone footage, no restrictions",
    subtitle=["{n} drone clips ({clips}), 20 frames each, 720p; every tool on: B-frames, lookahead, "
              "cutree, scenecut; PSNR-Y BD-rate against x265 medium",
              "x265 --tune psnr at its preset defaults, 4 threads; VDWT tiers at their defaults "
              "(ultraheavy: P-frames or a B-frame per pair); fixed qp, 3 rates per clip",
              "grey: old HEAD; orange: current version; wine: v12; identical harness"],
    xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86 laptop)",
    yrange=(-40, 40),
)


CLEAN_VDWT = ("VDWT", "#880E4F",
              ["vdwt-v2-realtime", "vdwt-v2-light", "vdwt-v2-medium", "vdwt-v2-heavy",
               "vdwt-v2-ultraheavy"],
              {"vdwt-v2-realtime": ("realtime", -20, UP),
               "vdwt-v2-light": ("light", 18, UP),
               "vdwt-v2-medium": ("medium", -10, UP),
               "vdwt-v2-heavy": ("heavy", 18, UP),
               "vdwt-v2-ultraheavy": ("ultraheavy", -10, UP)})
CLEAN_SAMTIME = dict(UNRESTRICTED,
    families=[CLEAN_VDWT, X265],
    title="Samtime: VDWT vs x265",
    subtitle=["a720 + b720, 20 frames each, 720p; PSNR-Y BD-rate against x265 medium",
              "VDWT v3 fixed qp 42/50/58, 4 threads; x265 --tune psnr, preset defaults"],
    xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86)",
    yrange=(-60, 25),
)
CLEAN_DRONE = dict(UNRESTRICTED,
    families=[CLEAN_VDWT, X265],
    title="Drone: VDWT vs x265",
    subtitle=["canyon + forest + tasmania, 20 frames each, 720p; PSNR-Y BD-rate against x265 medium",
              "VDWT v3 fixed qp 42/50/58, 4 threads; x265 --tune psnr, preset defaults"],
    xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86)",
    yrange=(-40, 40),
)


PODCAST = dict(UNRESTRICTED,
    title="Talking head (The Action Lab, studio wall), no restrictions",
    subtitle=["{n} clips ({clips}), 20 frames at 59.94 fps, 720p from 1440p; PSNR-Y BD-rate against "
              "x265 medium; x265 --tune psnr with B-frames, lookahead and cutree",
              "VDWT tiers at their defaults, fixed qp 36/44/52; x265 fixed qp 30/36/42; single thread, "
              "CPU time including startup"],
    xlabel="Encode CPU time per frame (ms, log scale; 720p, single thread, x86 laptop)",
    yrange=(-80, 40),
)


PSNR_GROUPS = {
    "drone": ("Drone footage", "{n} drone FPV clips ({clips})", (-140, 40)),
    "samtime": ("Samtime: talking head and stage zoom", "{n} clips from a tech-news video ({clips}: "
                "a talking head; a stage zoom with a crowd and a caption)", (-120, 20)),
    "talking": ("Talking head, studio wall", "{n} clips of The Action Lab ({clips}), 59.94 fps from 1440p",
                (-100, 20)),
    "xiph_aerial": ("Aerial footage (public Xiph clips)", "{n} clips: Netflix Aerial, old_town_cross, "
                    "stockholm", (-140, 40)),
    "xiph_talking": ("Video-call talking heads (public Xiph clips)", "{n} clips: vidyo1, vidyo3, vidyo4",
                     (-100, 40)),
    "pexels_moto": ("Motocross FPV drone (Pexels, Eno Eltora)", "1080p60 FPV footage", (-60, 40)),
    "pexels_moto_vmaf": ("Motocross FPV drone (Pexels, Eno Eltora): VMAF", "1080p60 FPV footage", (-80, 60)),
    "pexels_bike": ("Bike ride (Pexels, Adrián Vizuete Moreno)", "1080p60 first-person ride", (-60, 40)),
    "xiph_general": ("General content (public Xiph clips)", "{n} clips: parkrun, shields, crowd_run, "
                     "park_joy", (-140, 40)),
}


def hw_config(kind):
    """VDWT tiers against the hardware HEVC and H.264 encoders (hw_codecs_pareto.py rows)."""
    vdwt = UNRESTRICTED["families"][0]
    lv = lambda p: {f"{p}-1": ("TU1", 0, UP), f"{p}-4": ("TU4", 0, DOWN), f"{p}-7": ("TU7", 0, UP)}
    return dict(UNRESTRICTED,
        reference="hw-1",
        title="Bike ride (Pexels, Adrián Vizuete Moreno): VDWT vs hardware encoders",
        subtitle=["1080p60 first-person ride, 120 frames; YUV-PSNR (6:1:1) BD-rate against hardware HEVC TU1 "
                  "(its best quality level)",
                  "Intel VAAPI HEVC and H.264 in CQP 36/42/48 at levels TU1/TU4/TU7 (default GOP); VDWT tiers "
                  "at their defaults, fixed qp 42/50/58"],
        xlabel="Encode time per frame (ms, log scale; 1080p; VDWT: single-thread CPU; hardware: wall time)",
        ylabel="Bitrate saving vs hardware HEVC TU1 (%)",
        heading="Compression vs encode complexity",
        yrange=(-60, 60),
        families=[vdwt, ("HEVC hardware", "#5F5F5F", ["hw-1", "hw-4", "hw-7"],
                         dict(lv("hw"), **{"hw-7": ("TU7", 58, 10)})),
                  ("H.264 hardware", "#3A7BD5", ["h264hw-1", "h264hw-4", "h264hw-7"],
                   dict(lv("h264hw"), **{"h264hw-4": ("TU4", -58, 10), "h264hw-7": ("TU7", -58, 10)}))],
        lines=True,
    )


def psnr_config(group, kind, metric="PSNR-Y"):
    if group == "hw_codecs":
        return hw_config(kind)
    title, clips_line, yrange = PSNR_GROUPS[group.removesuffix("+cnn")]
    qps = "36/44/52 (x265 30/36/42)" if "talking" in group else "42/50/58 (x265 36/42/48)"
    return dict(UNRESTRICTED,
        title=title if group.startswith("pexels") else f"{title}, no restrictions",
        subtitle=[clips_line + f", 20 frames{'' if group.startswith('pexels') else ' each, 720p'}; "
                  f"{metric} BD-rate against x265 medium",
                  ("x265 4.1 --tune psnr, all tools, no placebo; VDWT tiers at defaults; both on an Orange Pi 3B; "
                   f"fixed qp {qps}; Intel hardware HEVC CQP")
                  if group.endswith("_vmaf") else
                  "x265 4.3 --tune psnr with every tool (B-frames, lookahead, cutree; no placebo); VDWT tiers at "
                  f"their defaults; fixed qp {qps}; single thread"],
        xlabel=("Encode time per frame (ms, log; 1080p; VDWT, x265: one Cortex-A55 core; "
                "hardware HEVC: laptop wall time)") if group.endswith("_vmaf") else
               ("Encode" if kind == "encode" else "Decode") + " CPU time per frame (ms, log scale; " +
               ("1080p" if group.startswith("pexels") else "720p") + ", single thread, x86 laptop" + ("; x265 streams decoded by FFmpeg)" if kind == "decode"
                                                     else "; hardware HEVC: wall time)"),
        heading=f"Compression vs {kind} complexity",
        yrange=yrange,
        families=psnr_families(kind, group),
        extra_note=("CNN: a 79k-parameter restoration network trained on 48 Xiph clips, applied to "
                    "decoded frames with a per-frame switch (offline; decoder cost not shown)")
                   if group.endswith("+cnn") else None,
        lines=kind == "encode",
    )


def psnr_families(kind, group=None):
    """UNRESTRICTED's families with labels placed for the PSNR charts. Decode times cluster
    (x265 2–4 ms, VDWT 6–10 ms), so the decode charts label fewer x265 presets, to the right."""
    (vn, vc, vm, vl), (xn, xc, xm, xl) = UNRESTRICTED["families"]
    cnn = bool(group) and group.endswith("+cnn")
    group = group.removesuffix("+cnn") if group else group
    vl = dict(vl, **{"vdwt-realtime": ("realtime", -70, UP)})
    if group == "talking":  # the VDWT tiers sit among x265's fast presets
        vl = dict(vl, **{"vdwt-realtime": ("realtime", -85, 10), "vdwt-light": ("light", 55, 30),
                         "vdwt-medium": ("medium", -80, 10), "vdwt-heavy": ("heavy", 70, 30),
                         "vdwt-ultraheavy": ("ultraheavy", 20, DOWN)})
    if group == "samtime":
        vl = dict(vl, **{"vdwt-realtime": ("realtime", 0, DOWN), "vdwt-light": ("light", 60, 30),
                         "vdwt-medium": ("medium", -55, UP), "vdwt-heavy": ("heavy", 35, UP),
                         "vdwt-ultraheavy": ("ultraheavy", 30, UP)})
        xl = dict(xl, **{"x265-ultrafast": ("ultrafast", -100, 10)})
    if group == "pexels_moto_vmaf":
        vl = dict(vl, **{"vdwt-medium": ("medium", -40, UP), "vdwt-heavy": ("heavy", 10, UP), "vdwt-ultraheavy": ("ultraheavy", 30, 48)})
    if group == "pexels_bike":
        xl = dict(xl, **{"x265-fast": ("fast", -10, 48)})
    if kind == "decode":
        vl = {"vdwt-realtime": ("realtime", 0, DOWN), "vdwt-light": ("light", 70, 10),
              "vdwt-medium": ("medium", 85, 10), "vdwt-heavy": ("heavy", -70, 10),
              "vdwt-ultraheavy": ("ultraheavy", 0, UP)}
        xl = {"x265-ultrafast": ("ultrafast", 0, DOWN), "x265-medium": ("medium", 80, 10),
              "x265-veryslow": ("veryslow", 95, 10)}
    hw = ("HEVC hardware (Intel VAAPI)", "#5F5F5F", ["hw-1", "hw-4", "hw-7"],
          {"hw-1": ("TU1", 0, UP), "hw-4": ("TU4", 0, DOWN), "hw-7": ("TU7", 0, UP)})
    fams = [(vn, vc, vm, vl), (xn, xc, xm, xl), hw]
    if cnn:
        tiers = ["realtime", "light", "medium", "heavy", "ultraheavy"]
        fams = [("VDWT morning", "#A8A59C", [f"vdwt0-{t}" for t in tiers], {}),
                ("VDWT now", vc, vm, vl),
                ("VDWT + CNN", "#7B1E3A", [f"vdwtC-{t}" for t in tiers],
                 {"vdwtC-realtime": ("realtime", -60, UP), "vdwtC-ultraheavy": ("ultraheavy", 0, UP)}),
                (xn, xc, xm, xl), ("HW HEVC", hw[1], hw[2], hw[3])]
    return fams


def main():
    global REFERENCE, FAMILIES, Y_RANGE
    cfg = None
    if sys.argv[1] in ("--psnr", "--yuv611", "--vmaf"):
        group, kind, rd_path, out = sys.argv[2:6]
        cfg = psnr_config(group, kind, {"--psnr": "PSNR-Y", "--yuv611": "YUV-PSNR (6:1:1)",
                                        "--vmaf": "VMAF (vmaf_v0.6.1)"}[sys.argv[1]])
        REFERENCE, FAMILIES, Y_RANGE = cfg["reference"], cfg["families"], cfg["yrange"]
        rd = load_rd(rd_path)
        col = 5 if kind == "encode" else 6
        t = defaultdict(list)
        for f in (l.split() for l in open(rd_path)):
            if len(f) == 7:
                t[f[1]].append(float(f[col]))
        speed = {e: sum(v) / len(v) for e, v in t.items()}
    elif sys.argv[1] in ("--unrestricted", "--decode", "--drone", "--samtime",
                       "--clean-samtime", "--clean-drone", "--podcast"):
        cfg = {"--unrestricted": UNRESTRICTED, "--decode": DECODE, "--drone": DRONE,
               "--samtime": SAMTIME, "--clean-samtime": CLEAN_SAMTIME,
               "--clean-drone": CLEAN_DRONE, "--podcast": PODCAST}[sys.argv[1]]
        rd_path, out = sys.argv[2:4]
        REFERENCE, FAMILIES, Y_RANGE = cfg["reference"], cfg["families"], cfg.get("yrange", (-40, 40))
        rd = load_rd(rd_path)
        cpu = defaultdict(list)
        for f in (l.split() for l in open(rd_path)):
            if len(f) == 6:
                cpu[f[1]].append(float(f[5]))
        speed = {e: sum(v) / len(v) for e, v in cpu.items()}
    else:
        rd_path, speed_path, out = sys.argv[1:4]
        rd = load_rd(rd_path)
        speed = {f[0]: float(f[1]) for f in (l.split() for l in open(speed_path)) if len(f) == 2}
    clips = sorted({c for c, _ in rd})
    saving = {}
    for enc in {e for _, e in rd}:
        rates = [bd_rate(rd[(c, REFERENCE)], rd[(c, enc)]) for c in clips]
        rates = [r for r in rates if r is not None]
        if rates:
            saving[enc] = -sum(rates) / len(rates)
    for enc in sorted(saving, key=lambda e: -saving[e]):
        print(f"{enc:16s} saving {saving[enc]:+7.1f}%  {'decode' if cfg is DECODE else 'encode'} {speed.get(enc, float('nan')):8.2f} ms/frame")

    # Layout.
    W, H = 2000, 1400
    left, right, top, bottom = 230, 1920, 310, 1185
    xs_all = [speed[e] for e in saving if e in speed]
    ys_all = [saving[e] for e in saving if e in speed]
    nice = [m * 10 ** k for k in range(-1, 6) for m in (1, 2, 5)]
    xmin = max(v for v in nice if v <= min(xs_all))
    xmax = min(v for v in nice if v >= max(xs_all) * 1.15)
    ystep = 20
    ymin, ymax = Y_RANGE
    px = lambda x: left + (right - left) * (math.log10(x) - math.log10(xmin)) / (
        math.log10(xmax) - math.log10(xmin))
    py = lambda y: bottom - (bottom - top) * (y - ymin) / (ymax - ymin)
    font = "font-family='Inter, Helvetica, Arial, sans-serif'"
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
           f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
           f"<rect x='4' y='4' width='{W - 8}' height='{H - 8}' fill='none' stroke='#222' stroke-width='3'/>",
           f"<text x='62' y='95' {font} font-size='38' font-weight='700' fill='#111'>{cfg['title'] if cfg else 'Drone footage, I+P only'}</text>",
           f"<text x='{W - 55}' y='95' {font} font-size='34' fill='#777' text-anchor='end'>{cfg['heading'] if cfg else 'Compression vs encode complexity'}</text>",
           f"<text x='62' y='135' {font} font-size='24' fill='#888'>{len(clips)} drone clips ({', '.join(clips)}), 30 frames "
           "each; no B-frames, lookahead or reordering; PSNR-Y BD-rate against x264 medium</text>",
           f"<text x='62' y='167' {font} font-size='24' fill='#888'>x264 tune psnr,zerolatency; x265 tune psnr with "
           "the zero-latency settings; hardware HEVC in CQP; VDWT at fixed qp, its rt at matched bitrates</text>"]
    if cfg:
        svg[5:7] = [f"<text x='62' y='{135 + 32 * i}' {font} font-size='24' fill='#888'>"
                    + line.format(n=len(clips), clips=", ".join(clips)) + "</text>" for i, line in enumerate(cfg["subtitle"])]
    # Legend.
    lx = 250
    for name, colour, _, _ in FAMILIES:
        svg.append(f"<circle cx='{lx}' cy='225' r='17' fill='{colour}' stroke='#222' stroke-width='2.5'/>")
        svg.append(f"<text x='{lx + 32}' y='237' {font} font-size='34' font-weight='600' fill='#111'>{name}</text>")
        lx += 100 + 18 * len(name)
    # Grid and axes.
    y = ymin
    while y <= ymax + 1e-9:
        svg.append(f"<line x1='{left}' y1='{py(y)}' x2='{right}' y2='{py(y)}' stroke='#D5D3CC' stroke-width='2'/>")
        svg.append(f"<text x='{left - 22}' y='{py(y) + 11}' {font} font-size='30' fill='#555' text-anchor='end'>{y:+.0f}".replace("+0<", "0<") + "</text>")
        y += ystep
    svg.append(f"<line x1='{left}' y1='{py(0)}' x2='{right}' y2='{py(0)}' stroke='#999' stroke-width='2.5' stroke-dasharray='10 8'/>")
    svg.append(f"<line x1='{left}' y1='{top - 50}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='3'/>")
    svg.append(f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='3'/>")
    decade = xmin
    while decade <= xmax * 1.0001:  # 1-2-5 ticks
        for m in (1, 2, 5):
            x = decade * m
            if xmin <= x <= xmax * 1.0001:
                label = f"{x:g}"
                svg.append(f"<text x='{px(x)}' y='{bottom + 52}' {font} font-size='30' fill='#555' text-anchor='middle'>{label}</text>")
                svg.append(f"<line x1='{px(x)}' y1='{bottom}' x2='{px(x)}' y2='{bottom + 10}' stroke='#222' stroke-width='2'/>")
        decade *= 10
    svg.append(f"<text x='{(left + right) / 2}' y='{bottom + 118}' {font} font-size='34' fill='#222' text-anchor='middle'>"
               + (cfg["xlabel"] if cfg else "Encode time per frame (ms, log scale; 720p, 4 threads, x86 laptop; hardware: wall time)") + "</text>")
    svg.append(f"<text transform='translate(118,{(top + bottom) / 2}) rotate(-90)' {font} font-size='34' fill='#222' "
               "text-anchor='middle'>" + (cfg["ylabel"] if cfg else "Bitrate saving vs x264 medium (%)") + "</text>")
    # Series: lines, then points and labels on top.
    for name, colour, members, labels in FAMILIES:
        pts = sorted((px(speed[e]), py(max(saving[e], ymin)), e) for e in members if e in saving and e in speed)
        if len(pts) > 1 and (cfg is None or cfg.get("lines", True)):
            path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
            if name == "VDWT prev":
                svg.append(f"<path d='{path}' fill='none' stroke='{colour}' stroke-width='3.5' stroke-dasharray='13 10' stroke-linejoin='round'/>")
            else:
                svg.append(f"<path d='{path}' fill='none' stroke='{colour}' stroke-width='6' stroke-linejoin='round'/>")
    for name, colour, members, labels in FAMILIES:
        for e in members:
            if e not in saving or e not in speed:
                continue
            x, v = px(speed[e]), saving[e]
            clipped = v < ymin
            y = py(max(v, ymin))
            if clipped:
                # Off the chart: an arrow on the edge, labelled with the value.
                svg.append(f"<path d='M{x - 16:.1f},{y - 22:.1f} L{x + 16:.1f},{y - 22:.1f} L{x:.1f},{y + 4:.1f} Z' fill='{colour}' stroke='#222' stroke-width='2.5'/>")
                text = (labels.get(e, ("", 0, 0))[0] + f" ({v:+.0f}%)").strip()
                # Neighbouring off-chart points stack their labels upwards instead of overlapping.
                ly = y - 4 - 34 * sum(1 for o in members if o in saving and o in speed and saving[o] < ymin
                                      and speed[o] < speed[e])
                svg.append(f"<text x='{x + 26:.1f}' y='{ly:.1f}' {font} font-size='28' fill='#222'>{text}</text>")
                continue
            r = 11 if name == "VDWT prev" else (13 if name.startswith("VDWT old") else 17)
            svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{colour}' stroke='#222' stroke-width='2.5'/>")
            if e in labels:
                text, dx, dy = labels[e]
                svg.append(f"<text x='{x + dx:.1f}' y='{y + dy:.1f}' {font} font-size='28' fill='#222' text-anchor='middle'>{text}</text>")
    if cfg and cfg.get("extra_note"):
        svg.append(f"<text x='62' y='{H - 30}' {font} font-size='22' fill='#888'>{cfg['extra_note']}</text>")
    svg.append("</svg>")
    open(out, "w").write("\n".join(svg))


if __name__ == "__main__":
    main()
