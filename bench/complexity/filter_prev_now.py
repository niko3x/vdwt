#!/usr/bin/env python3
"""Score the previous band-pass against the wider band-pass on the samtime clips.

Both filters run on the same decoded VDWT streams (100 frames, ultraheavy, qp 42/50/58).
Output: bench/complexity/filter_prev_now.csv with clip,qp,kbps,filter,vmaf,psnr_y.
"""
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT / "target/bench_tmp/filter_sweep"
OUT = ROOT / "bench/complexity/filter_prev_now.csv"
CLIPS = ["a720", "b720"]
QPS = [42, 50, 58]
MODES = {"band": "prev", "band2": "now"}


def run(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)


def psnr_y(ref, dist):
    p = run([BIN, "psnr", ref, dist]).stdout
    m = re.search(r"average\s+Y\s+([\d.]+)", p)
    if not m:
        raise RuntimeError(p)
    return float(m.group(1))


def vmaf(ref, dist, tag):
    js = TMP / f"{tag}.json"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", dist, "-i", ref,
         "-lavfi", f"libvmaf=log_fmt=json:log_path={js}:n_threads=8",
         "-f", "null", "-"])
    v = json.load(open(js))["pooled_metrics"]["vmaf"]["mean"]
    js.unlink()
    return v


def main():
    rows = []
    for clip in CLIPS:
        src = ROOT / f"videos/samtime/{clip}.y4m"
        for qp in QPS:
            stream = TMP / f"{clip}_q{qp}.vdwt"
            recon = TMP / f"{clip}_q{qp}.y4m"
            kbps = stream.stat().st_size * 8 * 25 / 100 / 1000
            for mode, label in MODES.items():
                dst = TMP / f"prevnow_{clip}_{qp}_{mode}.y4m"
                run([BIN, "postfilter", recon, dst,
                     "--postfilter", mode, "--postfilter-strength", "2.0"])
                v = vmaf(src, dst, f"prevnow_{clip}_{qp}_{mode}")
                py = psnr_y(src, dst)
                dst.unlink()
                rows.append((clip, qp, kbps, label, v, py))
                print(f"{clip} q{qp} {label:4s} {kbps:7.1f} kbps  "
                      f"VMAF {v:6.2f}  PSNR-Y {py:6.2f}", flush=True)

    OUT.write_text("clip,qp,kbps,filter,vmaf,psnr_y\n" +
                   "".join(f"{c},{q},{k:.2f},{f},{v:.4f},{y:.4f}\n" for c, q, k, f, v, y in rows))

    by = {(c, q, f): (k, v, y) for c, q, k, f, v, y in rows}
    dv = [by[(c, q, "now")][1] - by[(c, q, "prev")][1] for c in CLIPS for q in QPS]
    dp = [by[(c, q, "now")][2] - by[(c, q, "prev")][2] for c in CLIPS for q in QPS]
    print(f"\nmean now-prev: dVMAF {sum(dv)/len(dv):+.2f}  dPSNR-Y {sum(dp)/len(dp):+.2f}")


if __name__ == "__main__":
    main()
