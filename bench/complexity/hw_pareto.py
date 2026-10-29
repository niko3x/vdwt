#!/usr/bin/env python3
"""Hardware HEVC (Intel VAAPI, the laptop's iGPU) rows for the Pareto charts, appended to
psnr_pareto.py's file for the group.

    hw_pareto.py GROUP OUT.txt

Compression levels 1, 4 and 7 (the driver fails the others), CQP at x265's qps, the encoder's
default GOP (with B-frames: no restrictions). Quality is YUV-PSNR 6:1:1 as in psnr_pareto.py.
Time is wall time per frame, as a GPU's CPU time means nothing: the difference between encoding
the clip looped 5 times and once, over the 80 extra frames, so start-up is amortised. Encodes
run one at a time.
"""
import os, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psnr_pareto import CLIPS, FRAMES, GROUPS, ROOT, kbps, psnr_y  # noqa: E402

LEVELS = [1, 4, 7]
TMP = ROOT / "target/bench_tmp/hw_pareto"


def encode(src, out, level, qp):
    t = time.perf_counter()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-vaapi_device", "/dev/dri/renderD128",
                    "-i", str(src), "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi",
                    "-compression_level", str(level), "-rc_mode", "CQP", "-qp", str(qp), str(out)],
                   check=True)
    return time.perf_counter() - t


def main():
    group, out_path = sys.argv[1], Path(sys.argv[2])
    clips, _, xq = GROUPS[group]
    TMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        looped = TMP / f"{clip}_x5.y4m"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-stream_loop", "4", "-i",
                        str(CLIPS / f"{clip}.y4m"), "-strict", "-1", str(looped)], check=True)
        for level in LEVELS:
            for qp in xq:
                out = TMP / f"{clip}_hw{level}_{qp}.hevc"
                once = encode(CLIPS / f"{clip}.y4m", out, level, qp)
                five = encode(looped, TMP / "x5.hevc", level, qp)
                ms = max(five - once, 0.0) * 1000 / (4 * FRAMES)
                with tempfile.NamedTemporaryFile(suffix=".y4m", dir=TMP) as dec:
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out),
                                    "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1",
                                    dec.name], check=True)
                    y = psnr_y(clip, dec.name)
                row = f"{clip} hw-{level} {qp} {kbps(clip, out):.1f} {y:.2f} {ms:.2f} nan"
                print(row, flush=True)
                rows.append(row)
        looped.unlink()
    with open(out_path, "a") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
