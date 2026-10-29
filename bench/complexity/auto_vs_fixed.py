#!/usr/bin/env python3
"""Auto post-filter against fixed band2 2.0: 20-frame quality on five clips.

Rows: clip mode tier qp kbps vmaf yuv ms strength
`mode` is `band2-2.0` or `auto`; `strength` is the signalled value in 1/8 units for frame 0.
Resumable: rows already in the output file are skipped.
"""
import json, os, re, subprocess, sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT / "target/bench_tmp/auto_vs_fixed"
OUT = ROOT / "bench/complexity/auto_vs_fixed.txt"
TMP.mkdir(parents=True, exist_ok=True)

CLIPS = [
    ("tasmania", ROOT/"videos/nr/clips/tasmania.y4m"),
    ("a720", ROOT/"videos/samtime/a720.y4m"),
    ("b720", ROOT/"videos/samtime/b720.y4m"),
    ("canyon", ROOT/"videos/nr/clips/canyon.y4m"),
    ("forest", ROOT/"videos/nr/clips/forest.y4m"),
]
TIERS = [("realtime", ["--realtime"]), ("light", ["--tier", "light"]),
         ("medium", ["--tier", "medium"]), ("heavy", ["--tier", "heavy"]),
         ("ultraheavy", ["--tier", "ultraheavy"])]
QPS = [42, 50, 58]
MODES = [("band2-2.0", ["--postfilter", "band2", "--postfilter-strength", "2.0"]),
         ("auto", ["--postfilter", "auto"])]


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def fps_of(path):
    h = open(path, "rb").readline().decode().split()
    f = [x for x in h if x.startswith("F")][0][1:]
    n, d = f.split(":")
    return int(n) / int(d)


def metrics(ref, dist, tag):
    vm = TMP / f"{tag}.json"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dist), "-i", str(ref),
         "-lavfi", f"libvmaf=n_threads=4:log_fmt=json:log_path={vm}",
         "-frames:v", "20", "-f", "null", "-"])
    v = json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p = run([str(BIN), "psnr", str(ref), str(dist)]).stdout
    m = re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)", p)
    vm.unlink(missing_ok=True)
    return v, float(m.group(1))


def signalled_strength(log):
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["frame"]) == 0:
            return int(r["postfilter"])
    return 0


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            p = line.split()
            if len(p) >= 9:
                done.add(tuple(p[:4]))
    out = open(OUT, "a")
    for clip, src in CLIPS:
        if only is not None and clip not in only:
            continue
        ref = TMP / f"{clip}_ref.y4m"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-frames:v", "20", "-pix_fmt", "yuv420p", "-strict", "-1", str(ref)])
        fps = fps_of(src)
        for mode, opts in MODES:
            for tier, tier_opts in TIERS:
                for qp in QPS:
                    key = (clip, mode, tier, str(qp))
                    if key in done:
                        continue
                    tag = f"{clip}_{mode}_{tier}_{qp}"
                    enc = TMP / f"{tag}.vdwt"
                    rec = TMP / f"{tag}.y4m"
                    log = TMP / f"{tag}.csv"
                    run([str(BIN), "encode", str(src), str(enc), "--qp", str(qp), "--frames", "20",
                         "--threads", "4", *tier_opts, "--recon", str(rec),
                         "--frame-log", str(log), *opts])
                    kbps = enc.stat().st_size * 8 * fps / 20 / 1000
                    v, y = metrics(ref, rec, tag)
                    s = signalled_strength(log)
                    out.write(f"{clip} {mode} {tier} {qp} {kbps:.1f} {v:.4f} {y:.4f} 0.0 {s}\n")
                    out.flush()
                    print(f"{clip:8s} {mode:9s} {tier:10s} q{qp}: {kbps:7.1f} kbps "
                          f"VMAF {v:5.2f} YUV {y:5.2f} s/8={s}", flush=True)
                    for p in (enc, rec, log):
                        p.unlink(missing_ok=True)
        ref.unlink(missing_ok=True)
        print("finished", clip, flush=True)
    out.close()


if __name__ == "__main__":
    main()
