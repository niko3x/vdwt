#!/usr/bin/env python3
"""Post-filter lab: apply candidate filters to decoded VDWT/x265 clips and score the hybrid BD.

The lab uses the same 20-frame clips as hybrid_bench.py. For every candidate it scores
VDWT+candidate against x265 with (a) no extra filter, (b) the standard 5x5 unsharp mask, and
(c) the same candidate. That separates a codec gain from a filter that simply helps every codec.
"""
import json, math, os, re, subprocess, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "bench/complexity")
from plot import bd_rate

ROOT=Path(__file__).resolve().parents[2]; BIN=ROOT/"target/release/vdwt"
LAB=ROOT/"target/bench_tmp/pf_lab"
QPS={"vdwt":[42,50,58],"x265":[36,42,48]}
CLIPS=["a720","b720","canyon","forest","tasmania"]

def run(c): return subprocess.run(c,check=True,capture_output=True,text=True)

def read_y4m(path):
    with open(path,"rb") as f:
        header=f.readline().decode().strip()
        frames=[]
        while True:
            line=f.readline()
            if not line: break
            assert line.startswith(b"FRAME"), line[:20]
            w=int(re.search(r"W(\d+)",header).group(1)); h=int(re.search(r"H(\d+)",header).group(1))
            y=np.frombuffer(f.read(w*h),dtype=np.uint8).reshape(h,w).copy()
            uv=np.frombuffer(f.read(w*h//2),dtype=np.uint8).copy()
            frames.append((y,uv))
        return header,frames

def write_y4m(path,header,frames):
    with open(path,"wb") as f:
        f.write((header+"\n").encode())
        for y,uv in frames:
            f.write(b"FRAME\n"); f.write(y.tobytes()); f.write(uv.tobytes())

def blur3(a):
    # [1 2 1]/4 separable
    p = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    h = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4.0
    q = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    return (q[:-2, :] + 2 * q[1:-1, :] + q[2:, :]) / 4.0


def blur5(a):
    # [1 4 6 4 1]/16 separable
    p = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    h = (p[:, :-4] + 4 * p[:, 1:-3] + 6 * p[:, 2:-2] + 4 * p[:, 3:-1] + p[:, 4:]) / 16.0
    q = np.pad(h, ((2, 2), (0, 0)), mode="edge")
    return (q[:-4, :] + 4 * q[1:-3, :] + 6 * q[2:-2, :] + 4 * q[3:-1, :] + q[4:, :]) / 16.0


def filt(y,name,param,qp):
    c=y.astype(np.float32)
    if name=="none": out=c
    elif name=="unsharp5": out=c+param*(c-blur5(c))
    elif name=="multi":
        a3,a5=param; out=c+a3*(c-blur3(c))+a5*(c-blur5(c))
    elif name=="band":
        out=c+param*(blur3(c)-blur5(c))
    elif name=="edge":
        b3=blur3(c); b5=blur5(c); g=np.abs(c-b3)
        mask=g/(g+8.0)
        out=c+param*mask*(c-b5)
    elif name=="edge2":
        # Gate on a coarser activity so a single noisy pixel cannot trigger sharpening.
        b5=blur5(c); g=np.abs(blur3(c)-b5)
        mask=g/(g+4.0)
        out=c+param*mask*(c-b5)
    else: raise ValueError(name)
    return np.clip(np.rint(out),0,255).astype(np.uint8)

def metrics(ref,dist,tag):
    vm=LAB/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    v=json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p=run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m=re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)",p)
    os.remove(vm); return v,float(m.group(1))

def bd(rows,ref):
    a=bd_rate(sorted((b,v) for b,v,y in ref),sorted((b,v) for b,v,y in rows))
    b=bd_rate(sorted((b,y) for b,v,y in ref),sorted((b,y) for b,v,y in rows))
    return 0.7*(-a)+0.3*(-b),-a,-b

def main():
    candidates={
        "none":("none",0),
        "unsharp5_040":("unsharp5",0.40),
        "unsharp5_060":("unsharp5",0.60),
        "multi_030_030":("multi",(0.30,0.30)),
        "multi_050_020":("multi",(0.50,0.20)),
        "multi_020_050":("multi",(0.20,0.50)),
        "band_080":("band",0.80),
        "band_120":("band",1.20),
        "edge_080":("edge",0.80),
        "edge_120":("edge",1.20),
        "edge2_080":("edge2",0.80),
        "edge2_120":("edge2",1.20),
    }
    # Load every decoded clip once.
    data={}  # (clip,codec,qp) -> header, frames, kbps
    for clip in CLIPS:
        src=ROOT/f"videos/{'samtime' if clip in ('a720','b720') else 'nr/clips'}/{clip}.y4m"
        header,ref=read_y4m(LAB/f"{clip}_ref.y4m")
        # fps from header
        fs=[x for x in header.split() if x.startswith("F")][0][1:]; n,d=fs.split(":"); fps=int(n)/int(d)
        for codec,qps in QPS.items():
            for qp in qps:
                path=LAB/f"{clip}_{codec}_{qp}.y4m"
                h,frames=read_y4m(path)
                stream=LAB/f"{clip}_{codec}_{qp}.vdwt" if codec=="vdwt" else LAB/f"{clip}_{codec}_{qp}.hevc"
                kbps=stream.stat().st_size*8*fps/20/1000
                data[(clip,codec,qp)]=(h,frames,kbps)
        data[(clip,"ref",0)]=(header,ref)

    rows=defaultdict(lambda: defaultdict(list))  # candidate -> (clip,codec) -> rows
    for ci,(name,(kind,param)) in enumerate(candidates.items()):
        print("candidate",name,flush=True)
        for clip in CLIPS:
            header,ref=data[(clip,"ref",0)]
            for codec,qps in QPS.items():
                for qp in qps:
                    h,frames,kbps=data[(clip,codec,qp)]
                    out=[]
                    for y,uv in frames:
                        out.append((filt(y,kind,param,qp),uv))
                    dst=LAB/"filtered.y4m"
                    write_y4m(dst,h,out)
                    v,yuv=metrics(LAB/f"{clip}_ref.y4m",dst,f"{clip}_{codec}_{qp}")
                    rows[name][(clip,codec)].append((kbps,v,yuv))
                    print(f"  {clip:9s} {codec:5s} q{qp}: VMAF {v:6.2f} YUV {yuv:6.2f}",flush=True)
                    os.remove(dst)

    # Score each candidate against the three x265 references.
    print("\nBD summary (positive = VDWT needs fewer bits):")
    for name in candidates:
        fair=[]; std=[]; base=[]
        for clip in CLIPS:
            vd=rows[name][(clip,"vdwt")]
            x=rows[name][(clip,"x265")]
            fair.append(bd(vd,x)[0])
            std.append(bd(vd,rows["unsharp5_060"][(clip,"x265")])[0])
            base.append(bd(vd,rows["none"][(clip,"x265")])[0])
        print(f"{name:16s} vs same {sum(fair)/5:+6.2f}%  vs std-unsharp {sum(std)/5:+6.2f}%  vs x265-base {sum(base)/5:+6.2f}%")

if __name__=="__main__": main()
