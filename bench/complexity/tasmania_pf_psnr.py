#!/usr/bin/env python3
"""Tasmania, custom band-pass on both: PSNR-Y rows for the PSNR Pareto."""
import os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; BIN=ROOT/"target/release/vdwt"
SRC=ROOT/"videos/nr/clips/tasmania.y4m"; TMP=ROOT/"target/bench_tmp/tassie_psnr"; TMP.mkdir(parents=True,exist_ok=True)
QPS={"vdwt":[42,50,58],"x265":[36,42,48]}
TIERS={"realtime":["--realtime"],"light":["--tier","light"],"medium":["--tier","medium"],
       "heavy":["--tier","heavy"],"ultraheavy":["--tier","ultraheavy"]}
PRESETS=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
def run(c): return subprocess.run(c,check=True,capture_output=True,text=True)
def psnr_y(ref,dist):
    out=run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m=re.search(r"average\s+Y\s+([\d.]+)",out)
    return float(m.group(1))
def post(inp,out):
    run([str(BIN),"postfilter",str(inp),str(out),"--postfilter","band","--postfilter-strength","2.0"])
ref=TMP/"ref20.y4m"
run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(SRC),"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(ref)])
rows=[]
for tier,opts in TIERS.items():
    for qp in QPS["vdwt"]:
        stream=TMP/f"v_{tier}_{qp}.vdwt"; rec=TMP/f"v_{tier}_{qp}.y4m"
        run([str(BIN),"encode",str(SRC),str(stream),"--qp",str(qp),"--frames","20","--threads","4",*opts,
             "--recon",str(rec),"--postfilter","band","--postfilter-strength","2.0"])
        kbps=stream.stat().st_size*8*60/20/1000; y=psnr_y(ref,rec)
        rows.append(f"tasmania vdwt-{tier} {qp} {kbps:.1f} {y:.4f}\n")
        print("vdwt",tier,qp,round(kbps,1),round(y,2),flush=True)
        for p in (stream,rec): p.unlink(missing_ok=True)
for preset in PRESETS:
    for qp in QPS["x265"]:
        hevc=TMP/f"x_{preset}_{qp}.hevc"; dec=TMP/f"x_{preset}_{qp}.y4m"; filt=TMP/f"x_{preset}_{qp}_f.y4m"
        run(["x265","--input",str(ref),"--preset",preset,"--tune","psnr","--qp",str(qp),
             "--pools","4","--frame-threads","1","-o",str(hevc)])
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(hevc),"-frames:v","20",
             "-pix_fmt","yuv420p","-strict","-1",str(dec)])
        post(dec,filt)
        kbps=hevc.stat().st_size*8*60/20/1000; y=psnr_y(ref,filt)
        rows.append(f"tasmania x265-{preset} {qp} {kbps:.1f} {y:.4f}\n")
        print("x265",preset,qp,round(kbps,1),round(y,2),flush=True)
        for p in (hevc,dec,filt): p.unlink(missing_ok=True)
Path("bench/complexity/tassie_pf_encode_psnr.txt").write_text("".join(rows))
