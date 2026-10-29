#!/usr/bin/env python3
"""Side-by-side demo: VDWT against a hardware HEVC encoder at the same bitrate.

The source is a 4:2:0 y4m clip. Both sides are encoded to the same target rate, decoded
back to y4m, and stacked left/right with a caption bar. VDWT can carry its display
post-filter on the decode side; the hardware encoder is stock (its own deblock/SAO).

Usage:
  side_by_side.py SRC.y4m OUT.mp4 [--tier ultraheavy] [--bitrate 300k]
                  [--postfilter band] [--postfilter-strength 2.0]
                  [--hevc-bitrate 300k] [--hevc-device /dev/dri/renderD128]
                  [--also-plain] [--work DIR]
"""
import argparse, json, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/vdwt"
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True, **kw)


def y4m_info(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
    tok = header.split()
    assert tok[0] == "YUV4MPEG2", header
    w = int(next(t[1:] for t in tok if t.startswith("W")))
    h = int(next(t[1:] for t in tok if t.startswith("H")))
    fn, fd = next(t[1:] for t in tok if t.startswith("F")).split(":")
    fps = int(fn) / int(fd)
    return w, h, fps, len(header) + 1, header


def y4m_frames(path):
    w, h, fps, hdr_len, _ = y4m_info(path)
    frame = w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)
    size = path.stat().st_size
    return round((size - hdr_len) / (6 + frame))


def human_kbps(path, frames, fps):
    return path.stat().st_size * 8 / (frames / fps) / 1000


def vdwt_encode(src, out, tier, bitrate):
    t = time.perf_counter()
    p = run([BIN, "encode", src, out, "--tier", tier, "--bitrate", bitrate])
    return time.perf_counter() - t, p.stdout


def vdwt_decode(enc, out, postfilter, strength):
    cmd = [BIN, "decode", enc, out]
    if postfilter != "none":
        cmd += ["--postfilter", postfilter, "--postfilter-strength", str(strength)]
    t = time.perf_counter()
    p = run(cmd)
    return time.perf_counter() - t, p.stdout


def hevc_encode_x265(src, out, bitrate):
    kbps = int(round(float(str(bitrate).rstrip("kK"))))
    t = time.perf_counter()
    run(["x265", "--input", src, "--preset", "medium", "--bitrate", str(kbps),
         "--vbv-maxrate", str(kbps), "--vbv-bufsize", str(2 * kbps),
         "--pools", "4", "--frame-threads", "1", "--keyint", "60", "-o", out])
    return time.perf_counter() - t


def hevc_encode(src, out, bitrate, device):
    # CBR at the requested rate: the same budget VDWT is given.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-vaapi_device", device, "-i", src,
           "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi",
           "-rc_mode", "CBR", "-b:v", bitrate, "-maxrate", bitrate,
           "-bufsize", bitrate, "-profile:v", "main", "-g", "60", out]
    t = time.perf_counter()
    run(cmd)
    return time.perf_counter() - t


def hevc_decode(hevc, out):
    t = time.perf_counter()
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", hevc,
         "-pix_fmt", "yuv420p", "-strict", "-1", out])
    return time.perf_counter() - t


def psnr_y(a, b):
    p = run([BIN, "psnr", a, b])
    m = re.search(r"average\s+Y\s+([\d.]+)", p.stdout)
    if not m:
        raise RuntimeError(p.stdout)
    return float(m.group(1))


def compose(left, right, out, label_l, label_r, work, fps, w, h):
    bar = 64
    lab_l = work / "label_left.txt"
    lab_r = work / "label_right.txt"
    lab_l.write_text(label_l + "\n")
    lab_r.write_text(label_r + "\n")
    vf = (f"pad={w}:{h + bar}:0:0:black,"
          f"drawtext=fontfile='{FONT}':textfile='{lab_l}':fontcolor=white:fontsize=22:"
          f"x=(w-text_w)/2:y=h-40")
    vf2 = (f"pad={w}:{h + bar}:0:0:black,"
           f"drawtext=fontfile='{FONT}':textfile='{lab_r}':fontcolor=white:fontsize=22:"
           f"x=(w-text_w)/2:y=h-40")
    fc = f"[0:v]{vf}[l];[1:v]{vf2}[r];[l][r]hstack=inputs=2[v]"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", left, "-i", right,
         "-filter_complex", fc, "-map", "[v]", "-r", f"{fps:g}", "-c:v", "libx264",
         "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tier", default="ultraheavy")
    ap.add_argument("--bitrate", default="300k")
    ap.add_argument("--postfilter", default="band")
    ap.add_argument("--postfilter-strength", type=float, default=2.0)
    ap.add_argument("--hevc-bitrate", default="300k")
    ap.add_argument("--hevc-encoder", choices=["vaapi", "x265"], default="vaapi")
    ap.add_argument("--hevc-label", default="Intel UHD 620 hardware HEVC")
    ap.add_argument("--hevc-device", default="/dev/dri/renderD128")
    ap.add_argument("--also-plain", action="store_true",
                    help="also render the unfiltered VDWT side for reference")
    ap.add_argument("--work", type=Path)
    args = ap.parse_args()

    w, h, fps, _, _ = y4m_info(args.src)
    frames = y4m_frames(args.src)
    work = args.work or (ROOT / "target" / "side_by_side" / args.out.stem)
    work.mkdir(parents=True, exist_ok=True)

    vdwt_enc = work / "vdwt.vdwt"
    t_enc, out = vdwt_encode(args.src, vdwt_enc, args.tier, args.bitrate)
    print(out.strip().splitlines()[-1], flush=True)

    left = work / "vdwt_filtered.y4m"
    t_dec, out = vdwt_decode(vdwt_enc, left, args.postfilter, args.postfilter_strength)
    print(out.strip().splitlines()[-1], flush=True)

    hevc = work / "hevc.hevc"
    if args.hevc_encoder == "x265":
        t_hevc = hevc_encode_x265(args.src, hevc, args.hevc_bitrate)
    else:
        t_hevc = hevc_encode(args.src, hevc, args.hevc_bitrate, args.hevc_device)
    right = work / "hevc.y4m"
    t_hdec = hevc_decode(hevc, right)

    vdwt_kbps = human_kbps(vdwt_enc, frames, fps)
    hevc_kbps = human_kbps(hevc, frames, fps)
    stats = {
        "tier": args.tier, "frames": frames, "fps": fps,
        "vdwt_kbps": round(vdwt_kbps, 1), "hevc_kbps": round(hevc_kbps, 1),
        "vdwt_encode_s": round(t_enc, 2), "hevc_encode_s": round(t_hevc, 2),
        "vdwt_decode_s": round(t_dec, 2), "hevc_decode_s": round(t_hdec, 2),
    }

    tag_l = f"VDWT {args.tier} + {args.postfilter} filter"
    tag_r = args.hevc_label
    compose(left, right, args.out, f"{tag_l}  |  {vdwt_kbps:.0f} kb/s",
            f"{tag_r}  |  {hevc_kbps:.0f} kb/s", work, fps, w, h)
    stats["vdwt_psnr_y_filtered"] = round(psnr_y(args.src, left), 2)
    stats["hevc_psnr_y"] = round(psnr_y(args.src, right), 2)

    if args.also_plain:
        plain = work / "vdwt_plain.y4m"
        vdwt_decode(vdwt_enc, plain, "none", 0)
        plain_out = args.out.with_name(args.out.stem + "_plain" + args.out.suffix)
        plain_dir = work / "plain"
        plain_dir.mkdir(exist_ok=True)
        compose(plain, right, plain_out,
                f"VDWT {args.tier} (no filter)  |  {vdwt_kbps:.0f} kb/s",
                f"{tag_r}  |  {hevc_kbps:.0f} kb/s", plain_dir, fps, w, h)
        stats["vdwt_psnr_y_plain"] = round(psnr_y(args.src, plain), 2)

    print(json.dumps(stats, indent=2), flush=True)

    (args.out.with_suffix(".json")).write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
