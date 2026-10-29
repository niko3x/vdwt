#!/usr/bin/env python3
"""Run a VDWT encode grid and record bytes, VMAF and PSNR.

    run.py --bin BIN --tag TAG --out CSV [--clips a,b] [--tiers medium,ultraheavy]
           [--qps 42,50,58] [--frames 20] [--vmaf] [-- extra encoder args]

Rows are `tag clip tier qp bytes kbps vmaf yuv psnr_y ms`; vmaf is empty when not measured.
Re-running the same output resumes: rows already present are skipped.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "target/research/work"
REF = ROOT / "target/research/ref"
CLIPS = {
    "a720": ROOT / "videos/samtime/a720.y4m",
    "b720": ROOT / "videos/samtime/b720.y4m",
    "canyon": ROOT / "videos/nr/clips/canyon.y4m",
    "forest": ROOT / "videos/nr/clips/forest.y4m",
    "tasmania": ROOT / "videos/nr/clips/tasmania.y4m",
    # Held out from the tuning grid: used to check that a gain generalizes.
    "coast": ROOT / "videos/nr/clips/coast.y4m",
    "gen_a": ROOT / "videos/nr/clips/gen_a.y4m",
    "gen_b": ROOT / "videos/nr/clips/gen_b.y4m",
    "madeira": ROOT / "videos/madeira_720p60.y4m",
}
TIERS = {
    "realtime": ["--realtime"],
    "light": ["--tier", "light"],
    "medium": ["--tier", "medium"],
    "heavy": ["--tier", "heavy"],
    "ultraheavy": ["--tier", "ultraheavy"],
}
HEADER = "tag clip tier qp bytes kbps vmaf yuv psnr_y ms\n"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def fps_of(path):
    h = open(path, "rb").readline().decode().split()
    f = [x for x in h if x.startswith("F")][0][1:]
    n, d = f.split(":")
    return int(n) / int(d)


def ref_clip(clip, frames):
    src = CLIPS[clip]
    out = REF / f"{clip}_{frames}.y4m"
    if not out.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-strict", "-1", str(out)])
    return out


def metrics(binary, ref, dist, tag, vmaf, model):
    yuv = psnr_y = 0.0
    p = run([binary, "psnr", str(ref), str(dist)]).stdout
    m = re.search(r"average\s+Y\s+([\d.]+)\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)", p)
    if m:
        psnr_y, yuv = float(m.group(1)), float(m.group(2))
    v = None
    if vmaf:
        vm = WORK / f"{tag}.json"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dist), "-i", str(ref),
             "-lavfi", f"libvmaf=model=version={model}:n_threads=4:log_fmt=json:log_path={vm}",
             "-f", "null", "-"])
        v = json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
        vm.unlink(missing_ok=True)
    return v, yuv, psnr_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "target/release/vdwt"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="a720,b720,canyon,forest,tasmania")
    ap.add_argument("--tiers", default="medium,ultraheavy")
    ap.add_argument("--qps", default="42,50,58")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--vmaf", action="store_true")
    ap.add_argument("--vmaf-model", default="vmaf_v0.6.1")
    ap.add_argument("--extra", default="", help="extra encoder arguments, shell-split")
    a = ap.parse_args()
    extra = shlex.split(a.extra)

    WORK.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            p = line.split()
            if len(p) >= 4 and p[0] != "tag":
                done.add(tuple(p[:4]))
    new = not out.exists()
    f = open(out, "a")
    if new:
        f.write(HEADER)
    try:
        for clip in a.clips.split(","):
            ref = ref_clip(clip, a.frames)
            fps = fps_of(CLIPS[clip])
            for tier in a.tiers.split(","):
                for qp in [int(q) for q in a.qps.split(",")]:
                    key = (a.tag, clip, tier, str(qp))
                    if key in done:
                        continue
                    tag = f"{a.tag}_{clip}_{tier}_{qp}"
                    enc = WORK / f"{tag}.vdwt"
                    rec = WORK / f"{tag}.y4m"
                    t = time.perf_counter()
                    run([a.bin, "encode", str(CLIPS[clip]), str(enc), "--qp", str(qp),
                         "--frames", str(a.frames), "--threads", str(a.threads), *TIERS[tier],
                         *extra, "--recon", str(rec)])
                    ms = (time.perf_counter() - t) * 1000.0 / a.frames
                    nbytes = os.path.getsize(enc)
                    kbps = nbytes * 8 * fps / a.frames / 1000
                    v, yuv, py = metrics(a.bin, ref, rec, tag, a.vmaf, a.vmaf_model)
                    f.write(f"{a.tag} {clip} {tier} {qp} {nbytes} {kbps:.1f} "
                            f"{'' if v is None else f'{v:.4f}'} {yuv:.4f} {py:.4f} {ms:.2f}\n")
                    f.flush()
                    vtxt = "" if v is None else f" VMAF {v:6.2f}"
                    print(f"{clip:9s} {tier:10s} q{qp}: {kbps:8.1f} kbps{vtxt} "
                          f"YUV {yuv:6.2f} {ms:7.2f} ms", flush=True)
                    enc.unlink(missing_ok=True)
                    rec.unlink(missing_ok=True)
    finally:
        f.close()


if __name__ == "__main__":
    main()
