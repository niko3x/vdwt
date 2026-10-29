#!/usr/bin/env python3
"""VDWT tiers against the GPU's hardware HEVC and H.264 encoders (Intel VAAPI) on one clip.

    hw_codecs_pareto.py CLIP.y4m OUT.txt

Rows as psnr_pareto.py writes them ("clip encoder qp kbps psnr encode_ms decode_ms"), with
encoders vdwt-TIER, hw-LEVEL (HEVC) and h264hw-LEVEL. VDWT: fixed qp 42/50/58, single-thread CPU
time per frame. Hardware: CQP 36/42/48 at compression levels 1, 4 and 7, default GOP, wall time
per frame. Quality is YUV-PSNR 6:1:1 over every frame of the clip.
"""
import os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
TMP = ROOT / "target/bench_tmp/hw_codecs"
TIERS = {"realtime": ["--realtime"], "light": ["--tier", "light"], "medium": ["--tier", "medium"],
         "heavy": ["--tier", "heavy"], "ultraheavy": ["--tier", "ultraheavy"]}
HW = {"hw": "hevc_vaapi", "h264hw": "h264_vaapi"}


def info(clip):
    head = open(clip, "rb").readline().decode().split()
    n, d = next(t for t in head if t.startswith("F"))[1:].split(":")
    w, h = (int(next(t for t in head if t.startswith(k))[1:]) for k in "WH")
    frames = os.path.getsize(clip) // (w * h * 3 // 2 + 6)
    return int(n) / int(d), frames


def yuv611(clip, decoded):
    out = subprocess.run([str(BIN), "psnr", str(clip), str(decoded)], capture_output=True, text=True,
                         check=True).stdout
    f = next(l for l in out.splitlines() if l.startswith("average")).split()
    return (6 * float(f[2]) + float(f[4]) + float(f[6])) / 8


def main():
    clip, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    name = clip.stem
    fps, frames = info(clip)
    TMP.mkdir(parents=True, exist_ok=True)
    kbps = lambda p: os.path.getsize(p) * 8 * fps / frames / 1000
    rows = []
    for tier, targs in TIERS.items():
        for qp in (42, 50, 58):
            s, rec = TMP / f"{name}_{tier}_{qp}.vdwt", TMP / f"{name}_{tier}_{qp}.y4m"
            p = subprocess.Popen([str(BIN), "encode", str(clip), str(s), "--qp", str(qp), "--threads", "1",
                                  "--recon", str(rec), *targs], stdout=subprocess.DEVNULL)
            _, status, ru = os.wait4(p.pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            ms = (ru.ru_utime + ru.ru_stime) * 1000 / frames
            rows.append(f"{name} vdwt-{tier} {qp} {kbps(s):.1f} {yuv611(clip, rec):.2f} {ms:.1f} nan")
            rec.unlink()
            print(rows[-1], flush=True)
    for prefix, codec in HW.items():
        for level in (1, 4, 7):
            for qp in (36, 42, 48):
                s = TMP / f"{name}_{prefix}{level}_{qp}.mp4"
                t = time.perf_counter()
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-vaapi_device", "/dev/dri/renderD128",
                                "-i", str(clip), "-vf", "format=nv12,hwupload", "-c:v", codec,
                                "-compression_level", str(level), "-rc_mode", "CQP", "-qp", str(qp), str(s)],
                               check=True)
                ms = (time.perf_counter() - t) * 1000 / frames
                with tempfile.NamedTemporaryFile(suffix=".y4m", dir=TMP) as dec:
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(s), "-pix_fmt", "yuv420p",
                                    "-f", "yuv4mpegpipe", "-strict", "-1", dec.name], check=True)
                    q = yuv611(clip, dec.name)
                rows.append(f"{name} {prefix}-{level} {qp} {kbps(s):.1f} {q:.2f} {ms:.2f} nan")
                print(rows[-1], flush=True)
    out_path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
