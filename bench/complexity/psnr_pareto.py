#!/usr/bin/env python3
"""YUV-PSNR (6:1:1) Pareto data: every VDWT tier (current build, defaults) and every x265 preset
but placebo (--tune psnr, all tools on), single thread, 20 frames per clip.

    psnr_pareto.py GROUP OUT.txt      GROUP: drone, samtime, talking or xiph_* (public clips)

Rows: "clip encoder qp kbps psnr encode_cpu_ms decode_cpu_ms" (per frame), psnr being
(6·Y + U + V) / 8 of the average per-plane PSNR, as JVET and AOM report it. Decoding is timed
only with DECODE=1 (else the column is nan). x265 encode times are reused from an earlier
run's rows in KNOWN (a file of these rows) when its stream is cached. Encodes run in
parallel (4 at a time, one thread each; CPU time comes from wait4, so it is per process). Then
every stream is decoded one at a time on a quiet machine: VDWT with `vdwt decode --threads 1`
(its own per-frame average), x265 with FFmpeg's single-threaded HEVC decoder on the stream
concatenated 20 times, so start-up is amortised. x265 streams are cached in videos/nr/hevc;
for the drone clips their encode times come from bench/complexity/nr_rd.txt (same settings)
unless FRESH is set, which re-encodes and re-times them (nr_rd.txt ran 8 jobs at a time).
"""
import os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("VDWT_BIN", ROOT / "target/release/vdwt"))
# Encoder name prefix in the rows (VDWT_PREFIX=vdwt0 for an older build), and VDWT_ONLY=1 to skip
# x265.
PREFIX = os.environ.get("VDWT_PREFIX", "vdwt")
CLIPS = ROOT / "videos/nr/clips"
HEVC = ROOT / "videos/nr/hevc"
VDWT = ROOT / "target/bench_tmp/psnr_pareto"
GROUPS = {  # clips, VDWT qps, x265 qps
    "drone": (["tasmania", "coast", "canyon", "forest"], [42, 50, 58], [36, 42, 48]),
    "samtime": (["gen_a", "gen_b"], [42, 50, 58], [36, 42, 48]),
    "talking": (["pod_a", "pod_b", "pod_c"], [36, 44, 52], [30, 36, 42]),
    # Public Xiph clips (bench/complexity/fetch_xiph.sh), for published charts.
    "xiph_aerial": (["x_aerial", "x_oldtown", "x_stockholm"], [42, 50, 58], [36, 42, 48]),
    "xiph_talking": (["x_vidyo1", "x_vidyo3", "x_vidyo4"], [36, 44, 52], [30, 36, 42]),
    "xiph_general": (["x_parkrun", "x_shields", "x_crowdrun", "x_parkjoy"], [42, 50, 58],
                     [36, 42, 48]),
    # Pexels demo clips, 1080p60 (videos/pexels; the originals are never committed).
    "pexels_moto": (["p_moto"], [42, 50, 58], [36, 42, 48]),
    "pexels_bike": (["p_bike"], [42, 50, 58], [36, 42, 48]),
}
TIERS = {"realtime": ["--realtime"], "light": ["--tier", "light"], "medium": ["--tier", "medium"],
         "heavy": ["--tier", "heavy"], "ultraheavy": ["--tier", "ultraheavy"]}
PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
           "veryslow"]
FRAMES = 20


def cpu_run(cmd):
    """Runs `cmd`; returns its user+system CPU seconds."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"failed: {cmd}")
    return ru.ru_utime + ru.ru_stime


def fps(clip):
    head = open(CLIPS / f"{clip}.y4m", "rb").readline().decode().split()
    n, d = next(t for t in head if t.startswith("F"))[1:].split(":")
    return int(n) / int(d)


def psnr_y(clip, decoded):
    """YUV-PSNR (6:1:1) of `decoded` against the clip."""
    out = subprocess.run([str(BIN), "psnr", str(CLIPS / f"{clip}.y4m"), str(decoded)],
                         capture_output=True, text=True, check=True).stdout
    f = next(l for l in out.splitlines() if l.startswith("average")).split()
    y, u, v = float(f[2]), float(f[4]), float(f[6])
    return (6 * y + u + v) / 8


def kbps(clip, path):
    return os.path.getsize(path) * 8 * fps(clip) / FRAMES / 1000


def vdwt_job(clip, tier, qp):
    out = VDWT / f"{PREFIX}_{clip}_{tier}_{qp}.vdwt"
    rec = VDWT / f"{PREFIX}_{clip}_{tier}_{qp}.y4m"
    cpu = cpu_run([str(BIN), "encode", str(CLIPS / f"{clip}.y4m"), str(out), "--qp", str(qp),
                   "--threads", "1", "--recon", str(rec), *TIERS[tier]])
    y = psnr_y(clip, rec)
    rec.unlink()
    return [clip, f"{PREFIX}-{tier}", qp, kbps(clip, out), y, cpu * 1000 / FRAMES, out]


def x265_job(clip, preset, qp, known):
    out = HEVC / f"{clip}_x265-{preset}_{qp}.hevc"
    cpu = known.get((clip, f"x265-{preset}", qp))
    if cpu is None or not out.exists():
        tmp = out.with_suffix(".tmp")
        cpu = cpu_run(["x265", "--input", str(CLIPS / f"{clip}.y4m"), "--preset", preset,
                       "--tune", "psnr", "--qp", str(qp), "--pools", "1", "--frame-threads", "1",
                       "-o", str(tmp)]) * 1000 / FRAMES
        tmp.rename(out)
    with tempfile.NamedTemporaryFile(suffix=".y4m", dir=VDWT) as dec:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-pix_fmt",
                        "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1", dec.name], check=True)
        y = psnr_y(clip, dec.name)
    return [clip, f"x265-{preset}", qp, kbps(clip, out), y, cpu, out]


def vdwt_decode_ms(path):
    out = subprocess.run([str(BIN), "decode", str(path), "/dev/null", "--threads", "1"],
                         capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("decoded "))
    return float(line.split(": ")[1].split()[0])


def x265_decode_ms(path, loops=20):
    with tempfile.NamedTemporaryFile(suffix=".hevc", dir=VDWT) as cat:
        data = open(path, "rb").read()
        cat.write(data * loops)
        cat.flush()
        cpu = cpu_run(["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", cat.name,
                       "-f", "null", "-"])
    return cpu * 1000 / (FRAMES * loops)


def main():
    group, out_path = sys.argv[1], Path(sys.argv[2])
    clips, vq, xq = GROUPS[group]
    VDWT.mkdir(parents=True, exist_ok=True)
    known = {}
    for l in open(os.environ["KNOWN"]) if os.environ.get("KNOWN") else []:
        f = l.split()
        if len(f) == 7 and f[1].startswith("x265"):
            known[(f[0], f[1], int(f[2]))] = float(f[5])
    jobs = [(vdwt_job, c, t, q) for c in clips for t in TIERS for q in vq]
    if not os.environ.get("VDWT_ONLY"):
        jobs += [(x265_job, c, p, q, known) for c in clips for p in PRESETS for q in xq]
    # The slowest jobs first, so the pool drains evenly.
    jobs.sort(key=lambda j: 0 if j[0] is x265_job and j[2] in ("placebo", "veryslow", "slower") else 1)
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda j: j[0](*j[1:]), jobs))
    with open(out_path, "w") as f:
        for r in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            dec = float("nan")
            if os.environ.get("DECODE"):
                dec = vdwt_decode_ms(r[6]) if r[1].startswith(PREFIX) else x265_decode_ms(r[6])
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]:.1f} {r[4]:.2f} {r[5]:.1f} {dec:.2f}\n")
            f.flush()
            print(*r[:6], f"dec {dec:.2f}", flush=True)


if __name__ == "__main__":
    main()
