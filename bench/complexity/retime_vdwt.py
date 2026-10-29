#!/usr/bin/env python3
"""Re-time the VDWT rows in the hybrid tables with encode-only runs (no --recon write)."""
import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT/"target/release/vdwt"
TMP = ROOT/"target/bench_tmp/retime"; TMP.mkdir(parents=True, exist_ok=True)
CLIPS = {c: ROOT/p for c, p in {
    "a720":"videos/samtime/a720.y4m", "b720":"videos/samtime/b720.y4m",
    "canyon":"videos/nr/clips/canyon.y4m", "forest":"videos/nr/clips/forest.y4m",
    "tasmania":"videos/nr/clips/tasmania.y4m"}.items()}
TIERS = {"realtime":["--realtime"], "light":["--tier","light"], "medium":["--tier","medium"],
         "heavy":["--tier","heavy"], "ultraheavy":["--tier","ultraheavy"]}
for table in ["hybrid_samtime.txt", "hybrid_drone.txt"]:
    path = ROOT/"bench/complexity"/table
    out = []
    for line in open(path):
        f = line.split()
        if len(f) != 7 or not f[1].startswith("vdwt-"):
            out.append(line); continue
        clip, enc, qp = f[0], f[1], f[2]
        tier = enc.split("-")[1]
        dst = TMP/f"{clip}_{tier}_{qp}.vdwt"
        t = time.perf_counter()
        subprocess.run([str(BIN),"encode",str(CLIPS[clip]),str(dst),"--qp",qp,"--frames","20",
                        "--threads","4",*TIERS[tier]], check=True, capture_output=True)
        ms = (time.perf_counter()-t)*1000/20
        f[6] = f"{ms:.2f}"
        out.append(" ".join(f)+"\n")
        print(table, clip, tier, qp, f"{ms:.2f}", flush=True)
        dst.unlink(missing_ok=True)
    path.write_text("".join(out))
