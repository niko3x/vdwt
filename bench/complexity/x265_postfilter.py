#!/usr/bin/env python3
"""x265 medium rows after the same display post-filter, for a fair comparison."""
import json, os, re, subprocess, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "bench/complexity")
from sweep_vdwt import CLIPS, x265_rows
from plot import bd_rate
ROOT=Path(__file__).resolve().parents[2]; BIN=ROOT/"target/release/vdwt"
TMP=ROOT/"target/bench_tmp/x265pf"; TMP.mkdir(parents=True, exist_ok=True)
def run(c): return subprocess.run(c,check=True,capture_output=True,text=True)
def metrics(ref,dist,tag):
    vm=TMP/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    v=json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p=run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m=re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)",p)
    os.remove(vm); return v,float(m.group(1))
amt=sys.argv[1] if len(sys.argv)>1 else "0.6"
out=ROOT/f"bench/complexity/hybrid_x265pf{amt.replace('.','')}.txt"
with open(out,"w") as f:
    for clip,(src,_) in CLIPS.items():
        fps=float([x for x in open(src,'rb').readline().decode().split() if x.startswith('F')][0][1:].split(':')[0])
        ref=TMP/f"{clip}_ref.y4m"
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(ref)])
        for qp in (36,42,48):
            h=TMP/f"{clip}_{qp}.hevc"; d=TMP/f"{clip}_{qp}.y4m"; sh=TMP/f"{clip}_{qp}_sh.y4m"
            run(["x265","--input",str(ref),"--preset","medium","--tune","psnr","--qp",str(qp),
                 "--pools","4","--frame-threads","1","-o",str(h)])
            run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(h),"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(d)])
            run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(d),"-vf",f"unsharp=5:5:{amt}:5:5:0","-pix_fmt","yuv420p","-strict","-1",str(sh)])
            kbps=os.path.getsize(h)*8*fps/20/1000
            v,y=metrics(ref,sh,f"{clip}_{qp}")
            f.write(f"{clip} x265-medium-pf {qp} {kbps:.1f} {v:.4f} {y:.4f} 0\n"); f.flush()
            print(clip,qp,round(kbps,1),round(v,2),round(y,2),flush=True)
            for p in (h,d,sh): os.remove(p)
        os.remove(ref)
# BD of x265-pf against x265-medium (unfiltered), and VDWT-pf against x265-medium-pf
vd=defaultdict(lambda: defaultdict(list)); x=defaultdict(lambda: defaultdict(list))
for line in open(ROOT/"bench/complexity/hybrid_postfilter.txt"):
    c,e,q,k,v,y,m=line.split(); vd[e][c].append((float(k),float(v),float(y)))
for line in open(out):
    c,e,q,k,v,y,m=line.split(); x[e][c].append((float(k),float(v),float(y)))
base=defaultdict(lambda: defaultdict(list))
for path in (ROOT/"bench/complexity/hybrid_samtime.txt", ROOT/"bench/complexity/hybrid_drone.txt"):
    for line in open(path):
        c,e,q,k,v,y,m=line.split(); base[e][c].append((float(k),float(v),float(y)))
print("x265-pf vs x265-medium (unfiltered):")
for c in CLIPS:
    a=bd_rate(sorted((b,v) for b,v,y in base['x265-medium'][c]),sorted((b,v) for b,v,y in x['x265-medium-pf'][c]))
    b=bd_rate(sorted((b,y) for b,v,y in base['x265-medium'][c]),sorted((b,y) for b,v,y in x['x265-medium-pf'][c]))
    print(c,round(0.7*(-a)+0.3*(-b),2))
for tier in ('realtime','light','medium','heavy','ultraheavy'):
    hs=[]
    for c in CLIPS:
        e=f"vdwt-{tier}-pf"
        a=bd_rate(sorted((b,v) for b,v,y in x['x265-medium-pf'][c]),sorted((b,v) for b,v,y in vd[e][c]))
        b=bd_rate(sorted((b,y) for b,v,y in x['x265-medium-pf'][c]),sorted((b,y) for b,v,y in vd[e][c]))
        hs.append(0.7*(-a)+0.3*(-b))
    print(tier,round(sum(hs)/len(hs),2))
