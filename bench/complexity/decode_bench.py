#!/usr/bin/env python3
"""Measure decode time per frame for the hybrid tables' encoders.

Rows: "clip encoder qp decode_ms". Quality/rate come from the matching hybrid table; this
script only times decoding. VDWT is decoded by the library's own CLI (in-process average),
x265 by FFmpeg from a 50x concatenated raw HEVC stream so process startup is amortised.
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT/"target/release/vdwt"
TMP = ROOT/"target/bench_tmp/decode"; TMP.mkdir(parents=True, exist_ok=True)
VDWT_QPS = [42, 50, 58]
X265_QPS = [36, 42, 48]
LOOPS = 50
CLIPS = {
    "a720": ROOT/"videos/samtime/a720.y4m",
    "b720": ROOT/"videos/samtime/b720.y4m",
    "canyon": ROOT/"videos/nr/clips/canyon.y4m",
    "forest": ROOT/"videos/nr/clips/forest.y4m",
    "tasmania": ROOT/"videos/nr/clips/tasmania.y4m",
}
TIERS = {"realtime":["--realtime"], "light":["--tier","light"], "medium":["--tier","medium"],
         "heavy":["--tier","heavy"], "ultraheavy":["--tier","ultraheavy"]}
PRESETS = ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

def ref20(src, out):
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-frames:v","20",
         "-pix_fmt","yuv420p","-strict","-1",str(out)])

def vdwt_decode_ms(path):
    p = run([str(BIN),"decode",str(path),"/dev/null","--threads","4"])
    for line in p.stdout.splitlines():
        if line.startswith("decoded "):
            return float(line.split(": ")[1].split()[0])
    raise RuntimeError(p.stdout + p.stderr)

def main():
    clips = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CLIPS)
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT/"bench/complexity/decode.txt")
    with open(out_path, "w") as out:
        for clip in clips:
            src = CLIPS[clip]
            ref = TMP/f"{clip}_ref.y4m"
            ref20(src, ref)
            for tier, opts in TIERS.items():
                for qp in VDWT_QPS:
                    enc = TMP/f"{clip}_{tier}_{qp}.vdwt"
                    run([str(BIN),"encode",str(src),str(enc),"--qp",str(qp),"--frames","20",
                         "--threads","4",*opts])
                    ms = vdwt_decode_ms(enc)
                    out.write(f"{clip} vdwt-{tier} {qp} {ms:.3f}\n"); out.flush()
                    print(f"{clip:9s} vdwt-{tier:11s} q{qp}: {ms:6.3f} ms/frame", flush=True)
                    enc.unlink(missing_ok=True)
            for preset in PRESETS:
                for qp in X265_QPS:
                    hevc = TMP/f"{clip}_{preset}_{qp}.hevc"
                    run(["x265","--input",str(ref),"--preset",preset,"--tune","psnr","--qp",str(qp),
                         "--pools","4","--frame-threads","1","-o",str(hevc)])
                    data = hevc.read_bytes()
                    loop = TMP/"loop.hevc"
                    loop.write_bytes(data * LOOPS)
                    t = time.perf_counter()
                    run(["ffmpeg","-hide_banner","-loglevel","error","-threads","4","-i",str(loop),
                         "-f","null","-"])
                    ms = (time.perf_counter()-t)*1000/(20*LOOPS)
                    out.write(f"{clip} x265-{preset} {qp} {ms:.3f}\n"); out.flush()
                    print(f"{clip:9s} x265-{preset:11s} q{qp}: {ms:6.3f} ms/frame", flush=True)
                    hevc.unlink(missing_ok=True)
            ref.unlink(missing_ok=True)
            (TMP/"loop.hevc").unlink(missing_ok=True)

if __name__ == "__main__":
    main()
