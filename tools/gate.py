#!/usr/bin/env python3
"""The benchmark gate: is a codec change better *in general*?

Every clip in videos/gate/ (fetch with tools/fetch_gate.sh) plus the Luckfox 288p segment is
encoded at three constant-bitrate points (the codec's real use: a low-delay link) and, for VDWT,
on a four-point fixed-qp curve. Each encode is scored with YUV-PSNR 6:1:1 (per-frame PSNR
averaged per plane, then (6Y+U+V)/8) and VMAF v0.6.1; results land in bench/gate/<tag>.json.

    tools/gate.py run TAG [--bin target/release/vdwt] [--tiers medium,ultraheavy] [--quick]
                         [--clips a,b] [--opts "--extra vdwt options"] [-j 2]
    tools/gate.py refs [--codecs x265,svtav1]     # reference lines, cached as ref_<codec>.json
    tools/gate.py compare BASE_TAG NEW_TAG        # BD-rate per clip, tier and metric
    tools/gate.py where TAG                       # BD-rate of VDWT against the references

BD-rate is the average bitrate change at equal quality (negative: the new one needs fewer
bits), over the quality range both curves cover, piecewise-cubic-free (linear in log rate).
--quick runs medium only and CBR only (about a third of the time). A change passes the gate when
the mean BD-rate improves and no clip loses more than 1% on either metric.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "gate")
FRAMES = 120

# name: (path, CBR rates in kbit/s). 720p rates sit near 0.008, 0.025 and 0.07 bits per pixel at
# 60 fps; Luckfox is the 80 kbit/s challenge with a point either side.
CLIPS = {
    "aerial": ("videos/gate/aerial.y4m", [500, 1500, 4000]),
    "driving": ("videos/gate/driving.y4m", [500, 1500, 4000]),
    "coaster": ("videos/gate/coaster.y4m", [500, 1500, 4000]),
    "fourpeople": ("videos/gate/fourpeople.y4m", [300, 800, 2000]),
    "kristen": ("videos/gate/kristen.y4m", [300, 800, 2000]),
    "crowdrun": ("videos/gate/crowdrun.y4m", [1500, 4000, 10000]),
    "boxing": ("videos/gate/boxing.y4m", [500, 1500, 4000]),
    "ducks": ("videos/gate/ducks.y4m", [1500, 4000, 10000]),
    "oldtown": ("videos/gate/oldtown.y4m", [500, 1500, 4000]),
    "crosswalk": ("videos/gate/crosswalk.y4m", [500, 1500, 4000]),
    "luckfox": ("videos/luckfox/src_mid_288p_500.y4m", [50, 80, 150]),
}
QPS = [32, 40, 48, 56]


# ---------------------------------------------------------------------------------------------
# Y4M and metrics


def read_y4m(path, limit=FRAMES):
    with open(path, "rb") as f:
        header = f.readline().split()
        w = int([t for t in header if t.startswith(b"W")][0][1:])
        h = int([t for t in header if t.startswith(b"H")][0][1:])
        rate = [t for t in header if t.startswith(b"F")][0][1:].split(b":")
        fps = int(rate[0]) / int(rate[1])
        cw, ch = (w + 1) // 2, (h + 1) // 2
        size = w * h + 2 * cw * ch
        frames = []
        while len(frames) < limit:
            line = f.readline()
            if not line:
                break
            buf = f.read(size)
            if len(buf) < size:
                break
            a = np.frombuffer(buf, np.uint8)
            frames.append(
                (
                    a[: w * h].reshape(h, w),
                    a[w * h : w * h + cw * ch].reshape(ch, cw),
                    a[w * h + cw * ch :].reshape(ch, cw),
                )
            )
    return frames, fps, (w, h)


def psnr_611(ref, dec):
    n = min(len(ref), len(dec))
    sums = [0.0, 0.0, 0.0]
    for r, d in zip(ref[:n], dec[:n]):
        for p in range(3):
            mse = np.mean((r[p].astype(np.int32) - d[p].astype(np.int32)) ** 2)
            sums[p] += 100.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)
    y, u, v = (s / n for s in sums)
    return (6 * y + u + v) / 8, y


def vmaf(ref_path, dec_path, n):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        log = t.name
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", dec_path, "-i", ref_path,
        "-lavfi", f"[0:v]trim=end_frame={n}[d];[1:v]trim=end_frame={n}[r];"
        f"[d][r]libvmaf=n_threads=4:log_fmt=json:log_path={log}",
        "-f", "null", "-",
    ]
    subprocess.run(cmd, check=True)
    with open(log) as f:
        score = json.load(f)["pooled_metrics"]["vmaf"]["mean"]
    os.unlink(log)
    return score


# ---------------------------------------------------------------------------------------------
# Encoders: each returns (bytes, seconds) and leaves the decoded clip at `dec`.



def enc_vdwt(binary, src, dec, mode, value, tier, opts, work):
    stream = os.path.join(work, "s.vdwt")
    cmd = [binary, "encode", src, stream, "--frames", str(FRAMES), "--tier", tier, "--recon", dec]
    cmd += ["--bitrate", f"{value}k"] if mode == "cbr" else ["--qp", str(value)]
    cmd += opts.split()
    t = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    return os.path.getsize(stream), time.time() - t


def enc_x265(src, dec, kbps, fps, work):
    stream = os.path.join(work, "s.hevc")
    buf = max(1, round(kbps * 3 / fps))  # three frames of link, as VDWT's ultraheavy delay limit
    params = f"bitrate={kbps}:vbv-maxrate={kbps}:vbv-bufsize={buf}:keyint=600:bframes=0:rc-lookahead=0"
    t = time.time()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-frames:v", str(FRAMES),
         "-c:v", "libx265", "-preset", "medium", "-tune", "zerolatency", "-x265-params", params,
         stream], check=True)
    secs = time.time() - t
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", stream, "-pix_fmt",
                    "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1", dec], check=True)
    return os.path.getsize(stream), secs


def enc_svtav1(src, dec, kbps, fps, work):
    stream = os.path.join(work, "s.ivf")
    buf_ms = max(1, round(3000 / fps))
    t = time.time()
    subprocess.run(
        ["SvtAv1EncApp", "-i", src, "-b", stream, "-n", str(FRAMES), "--preset", "8",
         "--rc", "2", "--tbr", str(kbps), "--pred-struct", "1", "--buf-sz", str(buf_ms),
         "--keyint", "600", "--progress", "0"], check=True, capture_output=True)
    secs = time.time() - t
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", stream, "-pix_fmt",
                    "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1", dec], check=True)
    return os.path.getsize(stream), secs


def score(job):
    """One encode and its scores; `job` = (clip, codec, tier, mode, value, encoder closure)."""
    clip, codec, tier, mode, value, encode = job
    path = os.path.join(ROOT, CLIPS[clip][0])
    ref, fps, _ = read_y4m(path)
    work = tempfile.mkdtemp(prefix="gate_", dir=os.environ.get("GATE_TMP"))
    try:
        dec = os.path.join(work, "d.y4m")
        size, secs = encode(path, dec, fps, work)
        decoded, _, _ = read_y4m(dec)
        n = min(len(ref), len(decoded))
        p611, py = psnr_611(ref, decoded)
        v = vmaf(path, dec, n)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    kbps = size * 8 / (n / fps) / 1000
    row = dict(clip=clip, codec=codec, tier=tier, mode=mode, value=value, kbps=kbps,
               psnr611=p611, psnr_y=py, vmaf=v, enc_fps=n / secs)
    print(f"{clip:10} {codec:7} {tier:10} {mode} {value:>6}: {kbps:8.1f} kbps "
          f"{p611:6.2f} dB VMAF {v:5.2f} ({n / secs:5.1f} fps)", flush=True)
    return row


# ---------------------------------------------------------------------------------------------
# BD-rate


def bd_rate(a, b, metric):
    """Mean log-rate difference of b against a at equal `metric`, in percent."""
    def curve(rows):
        pts = sorted((r[metric], math.log(r["kbps"])) for r in rows)
        # Keep it monotone: a point that is no better than a cheaper one is dominated.
        out = []
        for q, lr in pts:
            if not out or q > out[-1][0]:
                out.append((q, lr))
        return out

    ca, cb = curve(a), curve(b)
    if len(ca) < 2 or len(cb) < 2:
        return float("nan")
    lo, hi = max(ca[0][0], cb[0][0]), min(ca[-1][0], cb[-1][0])
    if hi <= lo:
        return float("nan")

    def interp(c, q):
        for (q0, r0), (q1, r1) in zip(c, c[1:]):
            if q0 <= q <= q1:
                return r0 + (r1 - r0) * (q - q0) / (q1 - q0)
        return c[-1][1]

    n = 100
    qs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    d = sum(interp(cb, q) - interp(ca, q) for q in qs) / len(qs)
    return 100 * (math.exp(d) - 1)


def load(name):
    with open(os.path.join(OUT, f"{name}.json")) as f:
        return json.load(f)


def group(rows):
    g = {}
    for r in rows:
        g.setdefault((r["clip"], r["tier"], r["mode"]), []).append(r)
    return g


def table(pairs, title, flag=True):
    """pairs: {(clip, label): (rows_a, rows_b)} -> printed BD-rate table and means; `flag` marks
    clips that lose more than 1% (the gate's rule for a change)."""
    print(f"\n{title}\n{'clip':12}{'curve':22}{'BD PSNR611':>12}{'BD VMAF':>10}")
    per = {}
    for (clip, label), (a, b) in sorted(pairs.items()):
        p, v = bd_rate(a, b, "psnr611"), bd_rate(a, b, "vmaf")
        mark = "  <-- regression" if flag and (p > 1.0 or v > 1.0) else ""
        print(f"{clip:12}{label:22}{p:11.2f}%{v:9.2f}%{mark}")
        per.setdefault(label, []).append((p, v))
    for label, vals in sorted(per.items()):
        ps = [p for p, _ in vals if not math.isnan(p)]
        vs = [v for _, v in vals if not math.isnan(v)]
        print(f"{'MEAN':12}{label:22}{sum(ps) / max(1, len(ps)):11.2f}%"
              f"{sum(vs) / max(1, len(vs)):9.2f}%   ({len(ps)} clips)")


# ---------------------------------------------------------------------------------------------


def available(clips):
    out = []
    for c in clips:
        if os.path.exists(os.path.join(ROOT, CLIPS[c][0])):
            out.append(c)
        else:
            print(f"skipping {c}: {CLIPS[c][0]} missing (tools/fetch_gate.sh)", file=sys.stderr)
    return out


def run(args):
    binary = os.path.abspath(args.bin)
    tiers = ["medium"] if args.quick else args.tiers.split(",")
    clips = available(args.clips.split(",") if args.clips else list(CLIPS))
    jobs = []
    for clip in clips:
        for tier in tiers:
            points = [("cbr", r) for r in CLIPS[clip][1]]
            if not args.quick:
                points += [("qp", q) for q in QPS]
            for mode, value in points:
                def enc(src, dec, fps, work, mode=mode, value=value, tier=tier):
                    return enc_vdwt(binary, src, dec, mode, value, tier, args.opts, work)
                jobs.append((clip, "vdwt", tier, mode, value, enc))
    rows = list(ThreadPoolExecutor(args.j).map(score, jobs))
    save(args.tag, rows, dict(bin=args.bin, opts=args.opts, tiers=tiers))


def save(tag, rows, meta):
    """Writes `rows` into bench/gate/<tag>.json, replacing earlier rows of the same clips (so
    a run over a few clips extends an existing tag)."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.json")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            clips = {r["clip"] for r in rows}
            old = [r for r in json.load(f)["rows"] if r["clip"] not in clips]
    with open(path, "w") as f:
        json.dump(dict(meta, rows=old + rows), f, indent=1)
    print(f"wrote bench/gate/{tag}.json")


def refs(args):
    clips = available(args.clips.split(",") if args.clips else list(CLIPS))
    for codec in args.codecs.split(","):
        fn = {"x265": enc_x265, "svtav1": enc_svtav1}[codec]
        jobs = []
        for clip in clips:
            for r in CLIPS[clip][1]:
                def enc(src, dec, fps, work, r=r, fn=fn):
                    return fn(src, dec, r, fps, work)
                jobs.append((clip, codec, "ref", "cbr", r, enc))
        rows = list(ThreadPoolExecutor(args.j).map(score, jobs))
        save(f"ref_{codec}", rows, {})


def compare(args):
    a, b = group(load(args.base)["rows"]), group(load(args.new)["rows"])
    pairs = {(c, f"{t} {m}"): (a[k], b[k]) for k in a if k in b for (c, t, m) in [k]}
    table(pairs, f"{args.new} against {args.base} (negative: fewer bits at equal quality)")


def where(args):
    v = group(load(args.tag)["rows"])
    for codec in ("x265", "svtav1"):
        try:
            r = group(load(f"ref_{codec}")["rows"])
        except FileNotFoundError:
            continue
        pairs = {}
        for (clip, tier, mode), rows in v.items():
            if mode == "cbr" and (clip, "ref", "cbr") in r:
                pairs[(clip, f"{tier} vs {codec}")] = (r[(clip, "ref", "cbr")], rows)
        table(pairs, f"{args.tag} against {codec} (negative: VDWT needs fewer bits)", flag=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("tag")
    p.add_argument("--bin", default=os.path.join(ROOT, "target/release/vdwt"))
    p.add_argument("--tiers", default="medium,ultraheavy")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--clips", default="")
    p.add_argument("--opts", default="")
    p.add_argument("-j", type=int, default=2)
    p = sub.add_parser("refs")
    p.add_argument("--codecs", default="x265,svtav1")
    p.add_argument("--clips", default="")
    p.add_argument("-j", type=int, default=2)
    p = sub.add_parser("compare")
    p.add_argument("base")
    p.add_argument("new")
    p = sub.add_parser("where")
    p.add_argument("tag")
    args = ap.parse_args()
    {"run": run, "refs": refs, "compare": compare, "where": where}[args.cmd](args)


if __name__ == "__main__":
    main()
