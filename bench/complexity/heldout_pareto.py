#!/usr/bin/env python3
"""Re-run the VDWT side of the samtime + drone Pareto on the held-out context tables.

Rows: clip vdwt-tier qp kbps vmaf yuv ms. x265 rows are reused from the baseline hybrid tables.
"""
import json, math, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT / "target/bench_tmp/heldout"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "bench/complexity/heldout_vdwt.txt"
TMP.mkdir(parents=True, exist_ok=True)

CLIPS = [
    ("a720", ROOT/"videos/samtime/a720.y4m"),
    ("b720", ROOT/"videos/samtime/b720.y4m"),
    ("canyon", ROOT/"videos/nr/clips/canyon.y4m"),
    ("forest", ROOT/"videos/nr/clips/forest.y4m"),
    ("tasmania", ROOT/"videos/nr/clips/tasmania.y4m"),
]
TIERS = [("realtime", ["--realtime"]), ("light", ["--tier", "light"]),
         ("medium", ["--tier", "medium"]), ("heavy", ["--tier", "heavy"]),
         ("ultraheavy", ["--tier", "ultraheavy"])]
QPS = [42, 50, 58]


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


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            p = line.split()
            if len(p) >= 7:
                done.add(tuple(p[:4]))
    only_tiers = set(os.environ["ONLY_TIERS"].split(",")) if os.environ.get("ONLY_TIERS") else None
    out = open(OUT, "a")
    for clip, src in CLIPS:
        if only is not None and clip not in only:
            continue
        ref = TMP / f"{clip}_ref.y4m"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-frames:v", "20", "-pix_fmt", "yuv420p", "-strict", "-1", str(ref)])
        fps = fps_of(src)
        for tier, opts in TIERS:
            if only_tiers and tier not in only_tiers:
                continue
            for qp in QPS:
                key = (clip, f"vdwt-{tier}", str(qp))
                if key in done:
                    continue
                tag = f"{clip}_{tier}_{qp}"
                enc = TMP / f"{tag}.vdwt"
                rec = TMP / f"{tag}.y4m"
                run([str(BIN), "encode", str(src), str(enc), "--qp", str(qp), "--frames", "20",
                     "--threads", "4", *opts, "--recon", str(rec)])
                kbps = enc.stat().st_size * 8 * fps / 20 / 1000
                v, y = metrics(ref, rec, tag)
                out.write(f"{clip} vdwt-{tier} {qp} {kbps:.1f} {v:.4f} {y:.4f} 0.0\n")
                out.flush()
                print(f"{clip:8s} {tier:10s} q{qp}: {kbps:7.1f} kbps VMAF {v:5.2f} YUV {y:5.2f}",
                      flush=True)
                enc.unlink(missing_ok=True)
                rec.unlink(missing_ok=True)
        ref.unlink(missing_ok=True)
        print("finished", clip, flush=True)
    out.close()


if __name__ == "__main__":
    main()
