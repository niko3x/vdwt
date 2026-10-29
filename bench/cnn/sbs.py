#!/usr/bin/env python3
"""Side-by-side: VDWT (+ CNN restoration filter) against hardware HEVC (VAAPI) at the same rate.

    sbs.py SRC.y4m VDWT_RECON.y4m VDWT.vdwt FRAMELOG.csv MODEL.pt|- OUT_PREFIX

MODEL `-` skips the CNN (plain VDWT against HEVC).

The CNN runs on every decoded frame with that frame's qp and the ultraheavy tier, with the
per-frame switch (kept only where it raises YUV-PSNR 6:1:1, as an encoder flag would signal).
HEVC is encoded with VBR at the VDWT stream's actual bitrate. Writes OUT_PREFIX.mp4 (near
lossless x264 of the two decodes side by side) and prints the PSNRs.
"""
import csv, os, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_bound import read_y4m  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
TIER = 2  # ultraheavy


class Net(nn.Module):
    def __init__(self, cin, cout, ch, depth):
        super().__init__()
        L = [nn.Conv2d(cin, ch, 3, padding=1, padding_mode="replicate"), nn.ReLU()]
        for _ in range(depth - 2):
            L += [nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"), nn.ReLU()]
        L += [nn.Conv2d(ch, cout, 3, padding=1, padding_mode="replicate")]
        self.body = nn.Sequential(*L)

    def forward(self, x):
        return self.body((x - 0.5) * 4) / 16


def cond(tier, qp, h, w):
    q = torch.full((1, 1, h, w), qp / 128 + 0.25)
    t = F.one_hot(torch.tensor(tier), 3).float().view(1, 3, 1, 1).expand(1, 3, h, w)
    return torch.cat([q, t], 1)


def psnr(a, b):
    m = ((a.astype(np.float64) - b) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(m, 1e-9))


def p611(planes, src, f):
    return (6 * psnr(planes[0][f], src[0][f]) + psnr(planes[1][f], src[1][f]) + psnr(planes[2][f], src[2][f])) / 8


def write_y4m(path, header, planes):
    with open(path, "wb") as f:
        f.write(header)
        for y, u, v in zip(*planes):
            f.write(b"FRAME\n" + y.tobytes() + u.tobytes() + v.tobytes())


@torch.no_grad()
def main():
    src_p, rec_p, vdwt_p, log_p, model_p, out = sys.argv[1:7]
    header = open(src_p, "rb").readline()
    rate = next(t for t in header.split() if t.startswith(b"F"))[1:].split(b":")
    fps = int(rate[0]) / int(rate[1])
    if model_p == "-":
        return plain(src_p, rec_p, vdwt_p, out, fps)
    src, rec = read_y4m(src_p), read_y4m(rec_p)
    n = len(rec[0])
    kbps = os.path.getsize(vdwt_p) * 8 / (n / fps) / 1000
    qps = {int(r["frame"]): int(r["qp"]) for r in csv.DictReader(open(log_p))}

    if True:
        cnn, on = apply_cnn(model_p, rec, src, qps, n)
        label = "VDWT ultraheavy + CNN"
    write_y4m(out + "_vdwt_cnn.y4m", header, cnn)
    side_by_side(src_p, out, src, rec, cnn, on, n, fps, kbps, label)


BIN = Path(__file__).resolve().parents[2] / "target/release/vdwt"


def yuv611(src_p, dec_p):
    """YUV-PSNR 6:1:1 and frame count from `vdwt psnr` (streams; no clip held in memory). Each
    plane's per-frame PSNR is capped at 60 dB, so a flat frame (infinite PSNR) cannot dominate."""
    out = subprocess.run([str(BIN), "psnr", src_p, dec_p], capture_output=True, text=True, check=True).stdout
    frames = [l.split() for l in out.splitlines() if l.startswith("frame")]
    plane = lambda k: np.mean([min(float(f[k]), 60.0) for f in frames])
    return (6 * plane(3) + plane(5) + plane(7)) / 8, len(frames)


def plain(src_p, rec_p, vdwt_p, out, fps):
    """VDWT (no CNN) against hardware HEVC at the same rate, streaming every clip."""
    pv, n = yuv611(src_p, rec_p)
    kbps = os.path.getsize(vdwt_p) * 8 / (n / fps) / 1000
    hevc = out + "_hevc_hw.mp4"
    # The hardware VBR misses its target on some clips (+22% on the bike at 600k): rescale the
    # target until the real rate is within 2% of VDWT's.
    target = kbps
    for _ in range(6):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-vaapi_device", "/dev/dri/renderD128", "-i", src_p,
                        "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi", "-rc_mode", "VBR",
                        "-b:v", f"{target:.0f}k", "-maxrate", f"{target * 1.5:.0f}k", "-compression_level", "1",
                        hevc], check=True)
        hevc_kbps = os.path.getsize(hevc) * 8 / (n / fps) / 1000
        if abs(hevc_kbps / kbps - 1) <= 0.02:
            break
        target *= kbps / hevc_kbps
    dec = out + "_hevc_hw.y4m"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", hevc, "-pix_fmt", "yuv420p", "-strict", "-1", dec],
                   check=True)
    ph, _ = yuv611(src_p, dec)
    os.remove(dec)
    print(f"VDWT ultraheavy       {kbps:6.0f} kbps  {pv:.2f} dB (6:1:1)")
    print(f"HEVC hardware (VAAPI) {hevc_kbps:6.0f} kbps  {ph:.2f} dB")
    head = open(src_p, "rb").readline().split()
    w, h = (int(next(t for t in head if t.startswith(k))[1:]) for k in (b"W", b"H"))
    font = f"fontsize={30 * min(w, h) // 720}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20"
    fc = (f"[0:v]drawtext=text='VDWT ultraheavy  {kbps:.0f} kbps  {pv:.2f} dB':{font}[a];"
          f"[1:v]drawtext=text='HEVC hardware  {hevc_kbps:.0f} kbps  {ph:.2f} dB':{font}[b];[a][b]hstack")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", rec_p, "-i", hevc, "-filter_complex", fc,
                    "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", out + ".mp4"], check=True)


def apply_cnn(model_p, rec, src, qps, n):
    m = torch.load(model_p, map_location="cpu")
    ln, cn = Net(5, 1, m["ch"], m["depth"]).eval(), Net(10, 2, m["ch"], m["depth"]).eval()
    ln.load_state_dict(m["luma"]); cn.load_state_dict(m["chroma"])
    out_planes, on = [[], [], []], 0
    for f in range(n):
        y, u, v = (torch.from_numpy(p[f].astype(np.float32) / 255)[None, None] for p in rec)
        h, w = y.shape[2:]
        qp = qps[f]
        oy = y + ln(torch.cat([y, cond(TIER, qp, h, w)], 1))
        oc = torch.cat([u, v], 1) + cn(torch.cat([u, v, F.pixel_unshuffle(y, 2), cond(TIER, qp, h // 2, w // 2)], 1))
        to8 = lambda x: (x.clamp(0, 1) * 255).round().byte().numpy()[0, 0]
        cand = [to8(oy), to8(oc[:, :1]), to8(oc[:, 1:])]
        keep = p611([[c] for c in cand], [[s[f]] for s in src], 0) > p611(rec, src, f)
        on += keep
        for k in range(3):
            out_planes[k].append(cand[k] if keep else rec[k][f])
    return [np.stack(p) for p in out_planes], on


def side_by_side(src_p, out, src, rec, cnn, on, n, fps, kbps, label):
    hevc = out + "_hevc_hw.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-vaapi_device", "/dev/dri/renderD128", "-i", src_p,
                    "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi", "-rc_mode", "VBR",
                    "-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps * 1.5:.0f}k", "-compression_level", "1", hevc], check=True)
    hevc_kbps = os.path.getsize(hevc) * 8 / (n / fps) / 1000
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", hevc, "-pix_fmt", "yuv420p", "-strict", "-1",
                    out + "_hevc_hw.y4m"], check=True)
    hw = read_y4m(out + "_hevc_hw.y4m")

    avg = lambda planes: np.mean([p611(planes, src, f) for f in range(n)])
    pv, pc, ph = avg(rec), avg(cnn), avg(hw)
    print(f"VDWT ultraheavy       {kbps:6.0f} kbps  {pv:.2f} dB (6:1:1)")
    if label.endswith("CNN"):
        print(f"VDWT ultraheavy + CNN {kbps:6.0f} kbps  {pc:.2f} dB  (CNN on {on}/{n} frames)")
    print(f"HEVC hardware (VAAPI) {hevc_kbps:6.0f} kbps  {ph:.2f} dB")

    font = f"fontsize={30 * src[0].shape[1] // 720}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20"
    fc = (f"[0:v]drawtext=text='{label}  {kbps:.0f} kbps  {pc:.2f} dB':{font}[a];"
          f"[1:v]drawtext=text='HEVC hardware  {hevc_kbps:.0f} kbps  {ph:.2f} dB':{font}[b];[a][b]hstack")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", out + "_vdwt_cnn.y4m", "-i", out + "_hevc_hw.y4m",
                    "-filter_complex", fc, "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p",
                    out + ".mp4"], check=True)
    for p in (out + "_vdwt_cnn.y4m", out + "_hevc_hw.y4m"):
        os.remove(p)


if __name__ == "__main__":
    main()
