#!/usr/bin/env python3
"""VDWT with and without the CNN restoration filter: bitrate saving against decode latency.

    cnn_pareto.py OUT.txt SMALL.pt LARGE.pt

Every VDWT tier on the public clips the CNN never saw in training (Netflix Aerial, vidyo4,
park_joy; bench/complexity/fetch_xiph.sh), 20 frames, three qps. Each decode is filtered by a
small (12x4) and a large (32x6) network with the per-frame switch, as an in-loop filter would be
signalled (offline here: references stay unfiltered, so the in-loop gain is a lower bound).
Rows "clip encoder qp kbps psnr encode_ms decode_ms" as psnr_pareto.py writes them; encoder
vdwt-TIER, vdwtS-TIER (small CNN) or vdwtC-TIER (large). VDWT times are single-thread CPU; the
CNN's time is single-thread float CPU on this machine (PyTorch), added to encode (the encoder
runs the filter to decide the switch) and decode.
"""
import os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_bound import read_y4m  # noqa: E402
from sbs import Net, cond  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
CLIPS = ROOT / "videos/nr/clips"
TMP = ROOT / "target/bench_tmp/cnn_pareto"
SET = {"x_aerial": [42, 50, 58], "x_parkjoy": [42, 50, 58], "x_vidyo4": [36, 44, 52]}
TIERS = {"realtime": (["--realtime"], 0), "light": (["--tier", "light"], 1),
         "medium": (["--tier", "medium"], 1), "heavy": (["--tier", "heavy"], 2),
         "ultraheavy": (["--tier", "ultraheavy"], 2)}
FRAMES = 20


def load(path):
    m = torch.load(path, map_location="cpu")
    if "luma" in m:  # cnn_32x6_long.pt layout
        ch, depth, ls, cs = m["ch"], m["depth"], m["luma"], m["chroma"]
    else:  # tiny-search layout: a Plain module's state dict
        ls = {k[3:]: v for k, v in m.items() if k.startswith("ln.")}
        cs = {k[3:]: v for k, v in m.items() if k.startswith("cn.")}
        ch, depth = ls["body.0.weight"].shape[0], len([k for k in ls if k.endswith("weight")])
    ln, cn = Net(5, 1, ch, depth).eval(), Net(10, 2, ch, depth).eval()
    ln.load_state_dict(ls); cn.load_state_dict(cs)
    return ln, cn


@torch.no_grad()
def filt(net, y, u, v, tier, qp):
    ln, cn = net
    n, _, h, w = y.shape
    c = lambda h, w: cond(tier, qp, h, w).expand(n, -1, -1, -1)
    oy = y + ln(torch.cat([y, c(h, w)], 1))
    oc = torch.cat([u, v], 1) + cn(torch.cat([u, v, F.pixel_unshuffle(y, 2), c(h // 2, w // 2)], 1))
    return oy, oc[:, :1], oc[:, 1:]


def plane_psnr(a, b):
    m = ((a.astype(np.float64) - b) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(m, 1e-9))


def cpu_run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out = p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, cmd
    return ru.ru_utime + ru.ru_stime, out


def frame_ms(net):
    """Single-thread float CPU time of one 720p frame."""
    torch.set_num_threads(1)
    x = [torch.rand(1, 1, 720, 1280), torch.rand(1, 1, 360, 640), torch.rand(1, 1, 360, 640)]
    filt(net, *x, 1, 44)
    t = time.perf_counter()
    for _ in range(3):
        filt(net, *x, 1, 44)
    torch.set_num_threads(os.cpu_count())
    return (time.perf_counter() - t) / 3 * 1000


def main():
    out_path, small, large = sys.argv[1:4]
    nets = {"vdwtS": load(small), "vdwtC": load(large)}
    cost = {k: frame_ms(n) for k, n in nets.items()}
    print("CNN single-thread ms/frame:", {k: round(v, 1) for k, v in cost.items()}, flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, qps in SET.items():
        src = read_y4m(CLIPS / f"{clip}.y4m", FRAMES)
        fps = 60.0 if "vidyo" in clip or "aerial" in clip else 50.0
        for tier, (targs, ti) in TIERS.items():
            for qp in qps:
                s, rec = TMP / f"{clip}_{tier}_{qp}.vdwt", TMP / f"{clip}_{tier}_{qp}.y4m"
                enc, _ = cpu_run([str(BIN), "encode", str(CLIPS / f"{clip}.y4m"), str(s), "--qp", str(qp),
                                  "--threads", "1", "--frames", str(FRAMES), "--recon", str(rec), *targs])
                _, dout = cpu_run([str(BIN), "decode", str(s), "/dev/null", "--threads", "1"])
                dec = float(next(l for l in dout.splitlines() if l.startswith("decoded")).split(": ")[1].split()[0])
                kbps = os.path.getsize(s) * 8 * fps / FRAMES / 1000
                r = read_y4m(rec)
                rec.unlink()
                base = [[plane_psnr(r[k][f], src[k][f]) for k in range(3)] for f in range(FRAMES)]
                p611 = lambda per: (6 * np.mean([p[0] for p in per]) + np.mean([p[1] for p in per]) + np.mean([p[2] for p in per])) / 8
                rows.append((clip, f"vdwt-{tier}", qp, kbps, p611(base), enc * 1000 / FRAMES, dec))
                t = lambda a: torch.from_numpy(a.astype(np.float32) / 255)[:, None]
                y, u, v = t(r[0]), t(r[1]), t(r[2])
                for name, net in nets.items():
                    # One frame at a time: a 20-frame batch thrashes memory on the CPU.
                    outs = [filt(net, y[f:f + 1], u[f:f + 1], v[f:f + 1], ti, qp) for f in range(FRAMES)]
                    o8 = [torch.cat([o[k] for o in outs]).clamp(0, 1).mul(255).round().byte().numpy()[:, 0]
                          for k in range(3)]
                    per = []
                    for f in range(FRAMES):
                        a = [plane_psnr(o8[k][f], src[k][f]) for k in range(3)]
                        b = base[f]
                        per.append(a if 6 * a[0] + a[1] + a[2] > 6 * b[0] + b[1] + b[2] else b)
                    rows.append((clip, f"{name}-{tier}", qp, kbps, p611(per),
                                 enc * 1000 / FRAMES + cost[name], dec + cost[name]))
                print(*rows[-3][:5], "| S", round(rows[-2][4], 2), "| C", round(rows[-1][4], 2), flush=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]:.1f} {r[4]:.2f} {r[5]:.1f} {r[6]:.2f}\n")


if __name__ == "__main__":
    main()
