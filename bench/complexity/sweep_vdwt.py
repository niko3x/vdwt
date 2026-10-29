#!/usr/bin/env python3
"""Sweep one VDWT encoder option and score it against the stored x265-medium rows.

Rows use the same 20-frame setup as hybrid_bench.py and are written in the same format.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT/"target/bench_tmp/sweep"; TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "bench/complexity"))
from plot import bd_rate

CLIPS = {
    "a720": (ROOT/"videos/samtime/a720.y4m", ROOT/"bench/complexity/hybrid_samtime.txt"),
    "b720": (ROOT/"videos/samtime/b720.y4m", ROOT/"bench/complexity/hybrid_samtime.txt"),
    "canyon": (ROOT/"videos/nr/clips/canyon.y4m", ROOT/"bench/complexity/hybrid_drone.txt"),
    "forest": (ROOT/"videos/nr/clips/forest.y4m", ROOT/"bench/complexity/hybrid_drone.txt"),
    "tasmania": (ROOT/"videos/nr/clips/tasmania.y4m", ROOT/"bench/complexity/hybrid_drone.txt"),
}
TIERS = {
    "realtime": ["--realtime"], "light": ["--tier","light"],
    "medium": ["--tier","medium"], "heavy": ["--tier","heavy"],
    "ultraheavy": ["--tier","ultraheavy"],
}
QPS = [42, 50, 58]

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def fps_of(path):
    h = open(path, "rb").readline().decode().split()
    f = [x for x in h if x.startswith("F")][0][1:]
    n, d = f.split(":"); return int(n)/int(d)

def ref20(src, out):
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-frames:v","20",
         "-pix_fmt","yuv420p","-strict","-1",str(out)])

def metrics(ref, dist, tag):
    vm = TMP/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    v = json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p = run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m = re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)", p)
    os.remove(vm)
    return v, float(m.group(1))

def x265_rows(clip):
    path = CLIPS[clip][1]
    rows = []
    for line in open(path):
        c,e,q,k,v,y,m = line.split()
        if c == clip and e == "x265-medium":
            rows.append((float(k),float(v),float(y)))
    return rows

def bd_row(rows, ref):
    rv = sorted((b,v) for b,v,y in ref); ry = sorted((b,y) for b,v,y in ref)
    tv = sorted((b,v) for b,v,y,_ in rows); ty = sorted((b,y) for b,v,y,_ in rows)
    return bd_rate(rv,tv), bd_rate(ry,ty)

def main():
    # usage: sweep_vdwt.py TAG CLIPS TIERS -- [extra encoder args]
    tag = sys.argv[1]
    clips = sys.argv[2].split(",")
    tiers = sys.argv[3].split(",")
    extra = sys.argv[5:]
    out = ROOT/f"bench/complexity/sweep_{tag}.txt"
    with open(out,"w") as f:
        for clip in clips:
            src = CLIPS[clip][0]
            ref = TMP/f"{clip}_ref.y4m"; ref20(src, ref); fps=fps_of(src)
            for tier in tiers:
                for qp in QPS:
                    name=f"{clip}_{tier}_{qp}_{tag}"
                    enc=TMP/f"{name}.vdwt"; rec=TMP/f"{name}.y4m"
                    t=time.perf_counter()
                    run([str(BIN),"encode",str(src),str(enc),"--qp",str(qp),"--frames","20",
                         "--threads","4",*TIERS[tier],*extra,"--recon",str(rec)])
                    ms=(time.perf_counter()-t)*1000/20
                    kbps=os.path.getsize(enc)*8*fps/20/1000
                    v,y=metrics(ref,rec,name)
                    f.write(f"{clip} vdwt-{tier}-{tag} {qp} {kbps:.1f} {v:.4f} {y:.4f} {ms:.2f}\n"); f.flush()
                    print(f"{clip:9s} {tier:11s} q{qp}: {kbps:8.1f} VMAF {v:6.2f} YUV {y:6.2f} {ms:6.1f} ms", flush=True)
                    os.remove(enc); os.remove(rec)
            os.remove(ref)
    # score per clip and pooled (average of per-clip BD, as plot_hybrid does)
    data=defaultdict(lambda: defaultdict(list))
    for line in open(out):
        c,e,q,k,v,y,m=line.split(); data[e][c].append((float(k),float(v),float(y),float(m)))
    print()
    for e,per in data.items():
        hs=[]
        for c,rows in per.items():
            a,b=bd_row(rows,x265_rows(c))
            if a is None or b is None: continue
            h=0.7*(-a)+0.3*(-b); hs.append(h)
            print(f"{c:9s} hybrid {h:+6.2f}%  vmaf {-a:+6.2f}%  yuv {-b:+6.2f}%")
        print(f"{e:24s} mean hybrid {sum(hs)/len(hs):+6.2f}%  ({len(hs)} clips)")

if __name__=="__main__": main()
