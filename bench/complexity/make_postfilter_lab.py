#!/usr/bin/env python3
"""Decode the 20-frame benchmark clips to Y4M for post-filter experiments."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; BIN=ROOT/"target/release/vdwt"
OUT=ROOT/"target/bench_tmp/pf_lab"; OUT.mkdir(parents=True, exist_ok=True)
CLIPS={"a720":"videos/samtime/a720.y4m","b720":"videos/samtime/b720.y4m",
       "canyon":"videos/nr/clips/canyon.y4m","forest":"videos/nr/clips/forest.y4m",
       "tasmania":"videos/nr/clips/tasmania.y4m"}
def run(c): subprocess.run(c,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for clip,src in CLIPS.items():
    ref=OUT/f"{clip}_ref.y4m"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",src,"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(ref)])
    for qp in (42,50,58):
        enc=OUT/f"{clip}_vdwt_{qp}.vdwt"; rec=OUT/f"{clip}_vdwt_{qp}.y4m"
        run([str(BIN),"encode",src,enc,"--qp",str(qp),"--frames","20","--threads","4","--tier","ultraheavy","--recon",rec])
        print("vdwt",clip,qp,flush=True)
    for qp in (36,42,48):
        hevc=OUT/f"{clip}_x265_{qp}.hevc"; dec=OUT/f"{clip}_x265_{qp}.y4m"
        run(["x265","--input",ref,"--preset","medium","--tune","psnr","--qp",str(qp),"--pools","4","--frame-threads","1","-o",hevc])
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",hevc,"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",dec])
        print("x265",clip,qp,flush=True)
print("done")
