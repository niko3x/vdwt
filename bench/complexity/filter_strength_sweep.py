#!/usr/bin/env python3
"""Sweep band2 strength on the samtime clips and keep one q42 still per point."""
import json, math, os, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT / "target/bench_tmp/filter_sweep"
OUT = ROOT / "bench/complexity/filter_strength_sweep.csv"
CLIPS = ["a720", "b720"]
QPS = [42, 50, 58]
STRENGTHS = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


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


def still(src, dst):
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-vf", "select=eq(n\\,50)", "-frames:v", "1", dst])


def main():
    rows = []
    for clip in CLIPS:
        src = ROOT / f"videos/samtime/{clip}.y4m"
        for qp in QPS:
            stream = TMP / f"{clip}_q{qp}.vdwt"
            recon = TMP / f"{clip}_q{qp}.y4m"
            kbps = stream.stat().st_size * 8 * 25 / 100 / 1000
            for s in STRENGTHS:
                if s == 0:
                    v, py = vmaf(src, recon, f"sw_{clip}_{qp}_0"), psnr_y(src, recon)
                    rows.append((clip, qp, kbps, "none", 0.0, v, py))
                    print(f"{clip} q{qp} none          {kbps:7.1f} kbps  "
                          f"VMAF {v:6.2f}  PSNR-Y {py:6.2f}", flush=True)
                    if qp == 50:
                        still(recon, TMP / f"samsheet_{clip}_none.png")
                    continue
                dst = TMP / f"sw_{clip}_{qp}_{s}.y4m"
                run([BIN, "postfilter", recon, dst, "--postfilter", "band2",
                     "--postfilter-strength", str(s)])
                v = vmaf(src, dst, f"sw_{clip}_{qp}_{s}")
                py = psnr_y(src, dst)
                rows.append((clip, qp, kbps, "band2", s, v, py))
                print(f"{clip} q{qp} band2 {s:4.1f}   {kbps:7.1f} kbps  "
                      f"VMAF {v:6.2f}  PSNR-Y {py:6.2f}", flush=True)
                if qp == 50 and s in (2.0, 2.5, 3.0, 3.5, 5.0, 8.0):
                    still(dst, TMP / f"samsheet_{clip}_band2_{s}.png")
                dst.unlink()
            # previous filter, same stream
            dst = TMP / f"sw_{clip}_{qp}_prev.y4m"
            run([BIN, "postfilter", recon, dst, "--postfilter", "band",
                 "--postfilter-strength", "2.0"])
            v = vmaf(src, dst, f"sw_{clip}_{qp}_prev")
            py = psnr_y(src, dst)
            rows.append((clip, qp, kbps, "band", 2.0, v, py))
            print(f"{clip} q{qp} band  2.0    {kbps:7.1f} kbps  VMAF {v:6.2f}  PSNR-Y {py:6.2f}", flush=True)
            if qp == 50:
                still(dst, TMP / f"samsheet_{clip}_prev.png")
            dst.unlink()

    OUT.write_text("clip,qp,kbps,filter,strength,vmaf,psnr_y\n" +
                   "".join(f"{c},{q},{k:.2f},{f},{s:g},{v:.4f},{y:.4f}\n"
                           for c, q, k, f, s, v, y in rows))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
