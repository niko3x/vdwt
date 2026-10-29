#!/usr/bin/env python3
"""Side-by-side comparison video from two decoded y4m clips.

    side_by_side.py LEFT.y4m RIGHT.y4m OUT.mp4 "left label" "right label" [AUDIO_SRC]

Both y4m files must have the same size and frame rate. Frames are composited horizontally in
Python (numpy) and encoded once with libx264, so the comparison is not degraded by re-encoding
the two sides separately. If AUDIO_SRC is given, its first clip duration becomes the audio.
"""
import subprocess
import sys

import numpy as np


def y4m_info(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", "replace").strip()
    fields = {}
    for token in header.split()[1:]:
        if len(token) > 1:
            fields[token[0]] = token[1:]
    w, h = fields["W"], fields["H"]
    fn, fd = fields["F"].split(":")
    return int(w), int(h), int(fn) / int(fd)


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def main():
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    left, right, out, label_l, label_r = sys.argv[1:6]
    audio = sys.argv[6] if len(sys.argv) > 6 else None
    w, h, fps = y4m_info(left)
    w2, h2, fps2 = y4m_info(right)
    if (w, h) != (w2, h2):
        raise SystemExit(f"size mismatch: {w}x{h} vs {w2}x{h2}")
    if abs(fps - fps2) > 0.01:
        raise SystemExit(f"fps mismatch: {fps} vs {fps2}")

    frame = w * h * 3
    dec = ["ffmpeg", "-loglevel", "error", "-i", "{in}", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    left_p = subprocess.Popen([a.replace("{in}", left) for a in dec], stdout=subprocess.PIPE)
    right_p = subprocess.Popen([a.replace("{in}", right) for a in dec], stdout=subprocess.PIPE)

    font = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    box = "box=1:boxcolor=black@0.55:boxborderw=10"
    style = f"fontfile={font}:fontsize=30:fontcolor=white:{box}"
    vf = (f"drawtext={style}:text='{label_l}':x=24:y=24,"
          f"drawtext={style}:text='{label_r}':x=w/2+24:y=24")

    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2*w}x{h}", "-r", f"{fps:.6f}",
           "-i", "-"]
    if audio:
        cmd += ["-ss", "0", "-t", "10", "-i", audio]
    cmd += ["-map", "0:v:0"]
    if audio:
        cmd += ["-map", "1:a:0?"]
    cmd += ["-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n = 0
    while True:
        a = read_exact(left_p.stdout, frame)
        b = read_exact(right_p.stdout, frame)
        if a is None or b is None:
            break
        A = np.frombuffer(a, np.uint8).reshape(h, w, 3)
        B = np.frombuffer(b, np.uint8).reshape(h, w, 3)
        fused = np.empty((h, 2 * w, 3), np.uint8)
        fused[:, :w] = A
        fused[:, w:] = B
        enc.stdin.write(fused.tobytes())
        n += 1
    enc.stdin.close()
    rc = enc.wait()
    left_p.kill(); right_p.kill()
    print(f"{n} frames, {2*w}x{h} @ {fps:.0f} fps -> {out} (ffmpeg {rc})")
    if rc != 0 or n == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
