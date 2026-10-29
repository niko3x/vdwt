#!/usr/bin/env python3
"""Simulate a post-filter: encode VDWT, sharpen the reconstruction, score against x265 rows."""
import json, os, re, subprocess, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "bench/complexity")
from sweep_vdwt import CLIPS, TIERS, QPS, x265_rows
from plot import bd_rate

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT/"target/release/vdwt"; TMP=ROOT/"target/bench_tmp/sharp"; TMP.mkdir(parents=True, exist_ok=True)

def run(cmd): return subprocess.run(cmd,check=True,capture_output=True,text=True)
def fps_of(p):
    h=open(p,'rb').readline().decode().split(); f=[x for x in h if x.startswith('F')][0][1:]
    n,d=f.split(':'); return int(n)/int(d)
def metrics(ref,dist,tag):
    vm=TMP/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    v=json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p=run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m=re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)",p)
    os.remove(vm); return v,float(m.group(1))
def bd(rows,ref):
    a=bd_rate(sorted((b,v) for b,v,y in ref),sorted((b,v) for b,v,y,_ in rows))
    b=bd_rate(sorted((b,y) for b,v,y in ref),sorted((b,y) for b,v,y,_ in rows))
    return 0.7*(-a)+0.3*(-b),-a,-b
def main():
    amt=sys.argv[1]; clips=sys.argv[2].split(',')
    data=defaultdict(list)
    for clip in clips:
        src=CLIPS[clip][0]; ref=TMP/f"{clip}_ref.y4m"
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(ref)])
        fps=fps_of(src)
        for qp in QPS:
            tag=f"{clip}_{qp}"; enc=TMP/f"{tag}.vdwt"; rec=TMP/f"{tag}.y4m"; sh=TMP/f"{tag}_sh.y4m"
            run([str(BIN),"encode",str(src),str(enc),"--qp",str(qp),"--frames","20","--threads","4","--tier","ultraheavy","--recon",str(rec)])
            kbps=os.path.getsize(enc)*8*fps/20/1000
            run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(rec),"-vf",f"unsharp=5:5:{amt}:5:5:0","-pix_fmt","yuv420p","-strict","-1",str(sh)])
            v,y=metrics(ref,sh,tag)
            data[clip].append((kbps,v,y,0.0))
            print(clip,qp,round(kbps,1),round(v,2),round(y,2),flush=True)
            for p in (enc,rec,sh): os.remove(p)
        os.remove(ref)
    hs=[]
    for clip,rows in data.items():
        h,a,b=bd(rows,x265_rows(clip)); hs.append(h)
        print(f"{clip:9s} hybrid {h:+6.2f}% vmaf {a:+6.2f}% yuv {b:+6.2f}%")
    print("mean",round(sum(hs)/len(hs),2))
if __name__=="__main__": main()
