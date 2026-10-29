#!/usr/bin/env python3
"""Hybrid benchmark: VMAF + YUV-PSNR for current VDWT and x265 presets.

Rows: clip encoder qp kbps vmaf yuv_psnr ms_per_frame

    hybrid_bench.py [CLIPS] [OUT] [TUNE]

TUNE matches each side's tuning to the metric so neither is handicapped: `perceptual` (default,
for the VMAF-weighted hybrid) runs VDWT with --tune perceptual and x265 at its default tuning
(psy-rd, AQ); `psnr` runs VDWT at its defaults and x265 with --tune psnr. PRESETS (env,
comma-separated) limits the x265 presets.
One row at a time: each 20-frame Y4M is deleted after measurement, so /tmp stays small.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT/"target/bench_tmp/hybrid"
TMP.mkdir(parents=True, exist_ok=True)

CLIPS = [
    ("a720", ROOT/"videos/samtime/a720.y4m"),
    ("b720", ROOT/"videos/samtime/b720.y4m"),
    ("canyon", ROOT/"videos/nr/clips/canyon.y4m"),
    ("forest", ROOT/"videos/nr/clips/forest.y4m"),
    ("tasmania", ROOT/"videos/nr/clips/tasmania.y4m"),
]
TIERS = [("realtime", ["--realtime"]), ("light", ["--tier","light"]),
         ("medium", ["--tier","medium"]), ("heavy", ["--tier","heavy"]),
         ("ultraheavy", ["--tier","ultraheavy"])]
VDWT_QPS = [42, 50, 58]
X265_QPS = [36, 42, 48]
PRESETS = os.environ.get("PRESETS", "ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow").split(",")

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def fps_of(path):
    h = open(path, "rb").readline().decode().split()
    f = [x for x in h if x.startswith("F")][0][1:]
    n, d = f.split(":"); return int(n)/int(d)

def ref20(path, out):
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(path),"-frames:v","20",
         "-pix_fmt","yuv420p","-strict","-1",str(out)])

def clean(*paths):
    for p in paths:
        try: os.remove(p)
        except FileNotFoundError: pass

def metrics(ref, dist, tag):
    vm = TMP/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    j = json.load(open(vm))
    v = j["pooled_metrics"]["vmaf"]["mean"]
    p = run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m = re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)", p)
    clean(vm)
    return v, float(m.group(1))

def add(f, clip, enc, qp, kbps, ms, ref, tag):
    dist = TMP/f"{tag}.y4m"
    vm, yuv = metrics(ref, dist, tag)
    f.write(f"{clip} {enc} {qp} {kbps:.1f} {vm:.4f} {yuv:.4f} {ms:.2f}\n"); f.flush()
    print(f"{clip:9s} {enc:18s} q{qp}: {kbps:8.1f} kbps VMAF {vm:6.2f} YUV {yuv:6.2f} {ms:6.1f} ms", flush=True)
    clean(dist)

def main():
    clips = sys.argv[1].split(",") if len(sys.argv)>1 else [c for c,_ in CLIPS]
    out = Path(sys.argv[2] if len(sys.argv)>2 else ROOT/"bench/complexity/hybrid.txt")
    tune = sys.argv[3] if len(sys.argv)>3 else "perceptual"
    assert tune in ("perceptual", "psnr"), "TUNE is perceptual or psnr"
    vdwt_tune = ["--tune", tune]
    x265_tune = ["--tune", "psnr"] if tune == "psnr" else []
    with open(out, "a") as f:
        for clip, src in CLIPS:
            if clip not in clips: continue
            ref = TMP/f"{clip}_ref.y4m"; ref20(src, ref); fps = fps_of(src)
            for tier, opts in TIERS:
                for qp in VDWT_QPS:
                    tag=f"{clip}_{tier}_{qp}"; enc=TMP/f"{tag}.vdwt"; rec=TMP/f"{tag}.y4m"
                    t=time.perf_counter()
                    run([str(BIN),"encode",str(src),str(enc),"--qp",str(qp),"--frames","20",
                         "--threads","4",*opts,*vdwt_tune,"--recon",str(rec)])
                    ms=(time.perf_counter()-t)*1000/20
                    kbps=os.path.getsize(enc)*8*fps/20/1000
                    add(f, clip, f"vdwt-{tier}", qp, kbps, ms, ref, tag)
                    clean(enc)
            for preset in PRESETS:
                for qp in X265_QPS:
                    tag=f"{clip}_x265_{preset}_{qp}"; h=TMP/f"{tag}.hevc"; dist=TMP/f"{tag}.y4m"
                    t=time.perf_counter()
                    run(["x265","--input",str(ref),"--preset",preset,*x265_tune,"--qp",str(qp),
                         "--pools","4","--frame-threads","1","-o",str(h)])
                    ms=(time.perf_counter()-t)*1000/20
                    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(h),"-frames:v","20",
                         "-pix_fmt","yuv420p","-strict","-1",str(dist)])
                    kbps=os.path.getsize(h)*8*fps/20/1000
                    add(f, clip, f"x265-{preset}", qp, kbps, ms, ref, tag)
                    clean(h)
            clean(ref)
            print("finished", clip, flush=True)

if __name__=="__main__": main()
