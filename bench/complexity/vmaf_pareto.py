#!/usr/bin/env python3
"""VMAF Pareto rows: VDWT tiers and x265 presets encoded on a remote ARM board, hardware HEVC
(laptop VAAPI) from hw_pareto.py's cached streams, all scored with VMAF on this machine.

    vmaf_pareto.py CLIP OUT.txt [vdwt|x265|hw ...]

CLIP is a name in videos/nr/clips (20 frames). Encodes run on the host `orangepi` (ssh), four
single-thread jobs at a time, timed as each process's own CPU time; the streams are copied back
and decoded here (VDWT is bit-exact across platforms; x265 streams are decoded by FFmpeg). Rows:
"clip encoder qp kbps vmaf encode_ms nan", appended to OUT.txt. VMAF: libvmaf, vmaf_v0.6.1,
mean over frames.
"""
import os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
CLIPS = ROOT / "videos/nr/clips"
HW = ROOT / "target/bench_tmp/hw_pareto"
TMP = ROOT / "target/bench_tmp/vmaf_pareto"
HOST, RBIN = "orangepi", "~/vdwt-lat"
TIERS = {"realtime": "--realtime", "light": "--tier light", "medium": "--tier medium",
         "heavy": "--tier heavy", "ultraheavy": "--tier ultraheavy"}
PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
           "veryslow"]
VQ, XQ, FRAMES = [42, 50, 58], [36, 42, 48], 20


def fps(clip):
    head = open(CLIPS / f"{clip}.y4m", "rb").readline().decode().split()
    n, d = next(t for t in head if t.startswith("F"))[1:].split(":")
    return int(n) / int(d)


def remote_cpu(cmd):
    """Runs cmd on the board; returns its user+system CPU seconds (GNU time is not assumed)."""
    py = ("import os,subprocess,sys;p=subprocess.Popen(sys.argv[1],shell=True,stdout=subprocess.DEVNULL,"
          "stderr=subprocess.DEVNULL);_,s,r=os.wait4(p.pid,0);assert os.waitstatus_to_exitcode(s)==0,sys.argv[1];"
          "print(r.ru_utime+r.ru_stime)")
    out = subprocess.run(["ssh", HOST, "python3", "-c", repr(py), repr(cmd)], capture_output=True,
                         text=True, check=True).stdout
    return float(out.strip().splitlines()[-1])


def vmaf(clip, decoded):
    with tempfile.NamedTemporaryFile(suffix=".json", dir=TMP) as log:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(decoded), "-i", str(CLIPS / f"{clip}.y4m"),
                        "-lavfi", f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path={log.name}:n_threads=4",
                        "-f", "null", "-"], check=True)
        import json
        return json.load(open(log.name))["pooled_metrics"]["vmaf"]["mean"]


def decode_to_y4m(stream, out):
    if str(stream).endswith(".vdwt"):
        subprocess.run([str(BIN), "decode", str(stream), str(out)], check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(stream), "-pix_fmt", "yuv420p",
                        "-strict", "-1", str(out)], check=True)


def score(clip, name, qp, stream, ms):
    kbps = os.path.getsize(stream) * 8 * fps(clip) / FRAMES / 1000
    with tempfile.NamedTemporaryFile(suffix=".y4m", dir=TMP) as dec:
        decode_to_y4m(stream, dec.name)
        v = vmaf(clip, dec.name)
    return f"{clip} {name} {qp} {kbps:.1f} {v:.2f} {ms:.1f} nan"


def remote_job(clip, kind, setting, qp):
    if kind == "vdwt":
        name, rel = f"vdwt-{setting}", f"vm_{clip}_{setting}_{qp}.vdwt"
        cmd = f"{RBIN} encode ~/{clip}.y4m ~/{rel} --qp {qp} --threads 1 {TIERS[setting]}"
    else:
        name, rel = f"x265-{setting}", f"vm_{clip}_x265_{setting}_{qp}.hevc"
        cmd = (f"x265 --input ~/{clip}.y4m --preset {setting} --tune psnr --qp {qp} --pools 1 "
               f"--frame-threads 1 -o ~/{rel}")
    cpu = remote_cpu(cmd)
    local = TMP / rel
    subprocess.run(["scp", "-q", f"{HOST}:~/{rel}", str(local)], check=True)
    return score(clip, name, qp, local, cpu * 1000 / FRAMES)


def main():
    clip, out_path = sys.argv[1], Path(sys.argv[2])
    parts = sys.argv[3:] or ["vdwt", "x265", "hw"]
    TMP.mkdir(parents=True, exist_ok=True)
    rows = []
    if "hw" in parts:
        # Hardware HEVC streams and wall times from hw_pareto.py's run on this laptop.
        times = {}
        for l in open(ROOT / f"bench/complexity/yuv611_pexels_{clip[2:]}.txt"):
            f = l.split()
            if len(f) == 7 and f[1].startswith("hw-"):
                times[(f[1], int(f[2]))] = float(f[5])
        for level in (1, 4, 7):
            for qp in XQ:
                rows.append(score(clip, f"hw-{level}", qp, HW / f"{clip}_hw{level}_{qp}.hevc",
                                  times[(f"hw-{level}", qp)]))
                print(rows[-1], flush=True)
    jobs = []
    if "vdwt" in parts:
        jobs += [(clip, "vdwt", t, q) for t in TIERS for q in VQ]
    if "x265" in parts:
        jobs += [(clip, "x265", p, q) for p in PRESETS for q in XQ]
    if jobs:
        subprocess.run(["scp", "-q", str(CLIPS / f"{clip}.y4m"), f"{HOST}:~/{clip}.y4m"], check=True)
        jobs.sort(key=lambda j: 0 if j[2] in ("veryslow", "slower", "ultraheavy", "slow") else 1)
        with ThreadPoolExecutor(4) as pool:
            for r in pool.map(lambda j: remote_job(*j), jobs):
                rows.append(r)
                print(r, flush=True)
    with open(out_path, "a") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
