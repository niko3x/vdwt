#!/usr/bin/env python3
"""Synthetic drone-style clips with known camera motion, for testing VDWT's motion estimation.

A virtual camera flies over a large still image (the canvas). Its pose per frame is known, so the
true motion between any two frames is known too. The camera can pan, yaw, zoom (altitude), tilt
(perspective), vibrate, and read out with a rolling shutter; sensor noise, exposure drift and an
independently moving object are optional. Only numpy and ffmpeg (to read images) are needed.

    tools/synth_motion.py forest.jpg out.y4m --preset orbit --size 640x360 --frames 180
    tools/synth_motion.py forest.jpg out.y4m --preset shaky --truth truth.csv

The canvas is the image mirrored 3x3, so the camera has room to move and the seams stay
continuous. Rendering uses Keys bicubic sampling, supersampled 2x2 where the camera zooms out, so
frames carry no aliasing that the source would not have.

Ground truth (--truth) lists each frame's pose: the canvas point under the frame centre (cx, cy),
yaw theta (radians), zoom (canvas pixels per output pixel) and perspective terms p1, p2 (1/pixel).
An output point (x, y) maps to the canvas as follows: u = x - (W-1)/2, v = y - (H-1)/2,
d = 1 + p1*u + p2*v, then canvas = (cx, cy) + zoom * R(theta) * (u/d, v/d). With a rolling shutter
row y is captured later by `rs * y / H` frame times; the listed pose is that of row 0.

`--check FRAMES.csv` compares the global motion that `vdwt encode --frame-log` estimated with the
truth, at the four corners of the codec's motion grid:

    tools/synth_motion.py --truth truth.csv --check frames.csv
"""

import argparse
import math
import subprocess
import sys

import numpy as np


def load_rgb(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    w, h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32)


def mirror_tile(img, n=3):
    row = [img if i % 2 == 0 else img[:, ::-1] for i in range(n)]
    strip = np.concatenate(row, axis=1)
    return np.concatenate([strip if j % 2 == 0 else strip[::-1] for j in range(n)], axis=0)


def keys(t, a=-0.5):
    """Keys cubic weights for taps at offsets -1, 0, 1, 2 from floor(position)."""
    t2, t3 = t * t, t * t * t
    return (a * (t3 - 2 * t2 + t),
            (a + 2) * t3 - (a + 3) * t2 + 1,
            -(a + 2) * t3 + (2 * a + 3) * t2 - a * t,
            a * (t2 - t3))


def sample(img, xs, ys):
    """Bicubic samples of img (H, W, C) at float coordinates (pixel centres are integers)."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(xs), np.floor(ys)
    wx, wy = keys(xs - x0), keys(ys - y0)
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    out = np.zeros(xs.shape + (img.shape[2],), np.float32)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, h - 1)
        acc = np.zeros_like(out)
        for i in range(4):
            xx = np.clip(x0 + i - 1, 0, w - 1)
            acc += wx[i][..., None] * img[yy, xx]
        out += wy[j][..., None] * acc
    return out


class Path:
    """Camera pose as a smooth function of time (in frames)."""

    def __init__(self, preset, w, h, canvas_w, canvas_h, frames, seed):
        rng = np.random.default_rng(seed)
        self.rs = 0.0
        self.jitter = 0.0
        self.jitter_rot = 0.0
        # Pose at t = 0, then per-frame velocities and optional oscillations.
        cx0, cy0 = canvas_w / 2, canvas_h / 2
        self.p = dict(cx=cx0, cy=cy0, theta=0.0, zoom=1.0, p1=0.0, p2=0.0)
        self.v = dict(cx=0.0, cy=0.0, theta=0.0, zoom=0.0, p1=0.0, p2=0.0)
        self.osc = []  # (key, amplitude, period in frames)
        if preset == "survey":      # steady forward flight
            self.v.update(cx=0.6, cy=-1.8)
        elif preset == "pan":       # fast lateral pan
            self.v.update(cx=6.0, cy=0.5)
        elif preset == "orbit":     # yaw while drifting
            self.v.update(cx=1.0, cy=-0.5, theta=math.radians(0.35))
        elif preset == "climb":     # altitude gain: the view widens
            self.p.update(zoom=0.75)
            self.v.update(zoom=0.35 / frames, cy=-0.6, theta=math.radians(0.1))
        elif preset == "tilt":      # gimbal tilting: perspective changes
            self.v.update(cy=-1.0)
            self.osc += [("p2", 4e-4, 160.0), ("p1", 1.5e-4, 110.0)]
        elif preset == "static":    # a still camera over a still scene: only noise changes
            pass
        elif preset == "hover":     # hovering: slow drift and a little vibration
            self.v.update(cx=0.05, cy=0.03)
            self.jitter = 0.15
        elif preset == "shaky":     # hover with vibration and a rolling shutter
            self.v.update(cx=0.3)
            self.rs, self.jitter, self.jitter_rot = 0.6, 1.2, math.radians(0.08)
        elif preset == "mixed":     # everything at once
            self.v.update(cx=2.0, cy=-1.0, theta=math.radians(0.25), zoom=0.2 / frames)
            self.osc += [("p2", 2e-4, 200.0)]
            self.rs, self.jitter, self.jitter_rot = 0.5, 0.6, math.radians(0.04)
        else:
            raise SystemExit(f"unknown preset {preset}")
        # Vibration: a few incommensurate sinusoids between 12 and 35 Hz at 60 fps.
        self.jit = [(rng.uniform(3.5, 5.0), rng.uniform(0, 2 * math.pi)) for _ in range(6)]
        # Start far enough from the canvas edge for the whole clip.
        span_x = abs(self.v["cx"]) * frames
        span_y = abs(self.v["cy"]) * frames
        self.p["cx"] -= math.copysign(span_x / 2, self.v["cx"]) if self.v["cx"] else 0
        self.p["cy"] -= math.copysign(span_y / 2, self.v["cy"]) if self.v["cy"] else 0

    def pose(self, t):
        """Pose arrays for times t (array, in frames)."""
        out = {k: self.p[k] + self.v[k] * t for k in self.p}
        for key, amp, period in self.osc:
            out[key] = out[key] + amp * np.sin(2 * math.pi * t / period)
        if self.jitter:
            jx = sum(math.sqrt(2 / len(self.jit)) * np.sin(2 * math.pi * t / p + ph)
                     for p, ph in self.jit[:3])
            jy = sum(math.sqrt(2 / len(self.jit)) * np.cos(2 * math.pi * t / p + ph)
                     for p, ph in self.jit[3:])
            out["cx"] = out["cx"] + self.jitter * jx
            out["cy"] = out["cy"] + self.jitter * jy
            out["theta"] = out["theta"] + self.jitter_rot * jx * jy
        return out


def to_canvas(pose, x, y, w, h):
    u = x - (w - 1) / 2
    v = y - (h - 1) / 2
    d = 1 + pose["p1"] * u + pose["p2"] * v
    u, v = u / d, v / d
    c, s = np.cos(pose["theta"]), np.sin(pose["theta"])
    z = pose["zoom"]
    return pose["cx"] + z * (c * u - s * v), pose["cy"] + z * (s * u + c * v)


def from_canvas(pose, xc, yc, w, h):
    c, s = math.cos(pose["theta"]), math.sin(pose["theta"])
    du, dv = (xc - pose["cx"]) / pose["zoom"], (yc - pose["cy"]) / pose["zoom"]
    u1, v1 = c * du + s * dv, -s * du + c * dv
    d = 1 / (1 - pose["p1"] * u1 - pose["p2"] * v1)
    return u1 * d + (w - 1) / 2, v1 * d + (h - 1) / 2


def render(canvas, path, t, w, h, sprite, chunk=48):
    ss = 2 if max(path.p["zoom"], path.pose(np.array([t]))["zoom"][0]) > 1.02 else 1
    out = np.empty((h, w, 3), np.float32)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        ys = (np.arange(y0, y1)[:, None] + offs[None, :]).reshape(-1)
        xs = (np.arange(w)[:, None] + offs[None, :]).reshape(-1)
        X, Y = np.meshgrid(xs, ys)
        pose = path.pose(t + path.rs * Y / h)
        xc, yc = to_canvas(pose, X, Y, w, h)
        rgb = sample(canvas, xc, yc)
        if sprite is not None:
            img, (sx, sy), (vx, vy) = sprite
            ox, oy = sx + vx * t, sy + vy * t
            lx, ly = xc - ox, yc - oy
            inside = (lx >= 0) & (ly >= 0) & (lx <= img.shape[1] - 1) & (ly <= img.shape[0] - 1)
            if inside.any():
                rgb[inside] = sample(img, lx[inside], ly[inside])
        rgb = rgb.reshape(y1 - y0, ss, w, ss, 3).mean(axis=(1, 3))
        out[y0:y1] = rgb
    return out


def rgb_to_yuv420(rgb):
    """BT.601 limited range, chroma averaged over 2x2 (centre sited)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255
    h, w = y.shape
    ph, pw = h + h % 2, w + w % 2

    def sub(c):
        c = np.pad(c, ((0, ph - h), (0, pw - w)), mode="edge")
        return c.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))

    q = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return q(y), q(sub(u)), q(sub(v))


def generate(args):
    w, h = (int(v) for v in args.size.lower().split("x"))
    canvas = mirror_tile(load_rgb(args.image))
    path = Path(args.preset, w, h, canvas.shape[1], canvas.shape[0], args.frames, args.seed)
    sprite = None
    if args.sprite:
        img = load_rgb(args.sprite)
        scale = max(1, int(max(img.shape[:2]) / args.sprite_size))
        img = img[::scale, ::scale].copy()
        start = to_canvas(path.pose(np.array([0.0])), np.array([w * 0.2]), np.array([h * 0.3]), w, h)
        sprite = (img, (float(start[0][0]), float(start[1][0])), (args.sprite_speed, args.sprite_speed * 0.3))
    rng = np.random.default_rng(args.seed + 1)
    truth = ["frame,cx,cy,theta,zoom,p1,p2"]
    with open(args.output, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{args.fps}:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(args.frames):
            rgb = render(canvas, path, float(t), w, h, sprite)
            if args.gain:
                rgb *= 1 + args.gain * math.sin(2 * math.pi * t / 240)
            if args.noise > 0:
                rgb += rng.normal(0, args.noise, rgb.shape).astype(np.float32)
            y, u, v = rgb_to_yuv420(rgb)
            f.write(b"FRAME\n")
            for plane in (y, u, v):
                f.write(plane.tobytes())
            p = path.pose(np.array([float(t)]))
            truth.append(",".join([str(t)] + [f"{float(p[k][0]):.9g}" for k in
                                              ("cx", "cy", "theta", "zoom", "p1", "p2")]))
            if t % 30 == 0:
                print(f"frame {t}/{args.frames}", file=sys.stderr)
    if args.truth:
        with open(args.truth, "w") as f:
            f.write("\n".join(truth) + "\n")
            f.write(f"# size {w}x{h} rs {path.rs}\n")


def read_csv(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: empty")
    head = lines[0].split(",")
    return [dict(zip(head, l.split(","))) for l in lines[1:]]


def truth_size(path):
    """The frame size recorded in a truth file's trailing comment."""
    with open(path) as f:
        for line in f:
            if line.startswith("# size "):
                w, h = line.split()[2].split("x")
                return int(w), int(h)
    raise SystemExit(f"{path}: no '# size' line")


def check(args):
    """Compares estimated global motion (corner MVs in the frame log) with the truth."""
    truth = read_csv(args.truth)
    w, h = truth_size(args.truth)
    log = read_csv(args.check)
    grid = args.mesh
    gx, gy = -(-w // grid) * grid, -(-h // grid) * grid
    corners = [(0, 0), (gx, 0), (0, gy), (gx, gy)]
    pose = lambda i: {k: float(truth[i][k]) for k in ("cx", "cy", "theta", "zoom", "p1", "p2")}
    errs, zero_errs = [], []
    ref = None
    for row in log:
        i, kind = int(row["frame"]), row.get("type")
        if kind == "P" and ref is not None:
            cur_p, ref_p = pose(i), pose(ref)
            for k, (x, y) in enumerate(corners):
                xc, yc = to_canvas(cur_p, x, y, w, h)
                rx, ry = from_canvas(ref_p, xc, yc, w, h)
                tx, ty = rx - x, ry - y
                ex = float(row[f"gm{k}x"]) / 8 - tx
                ey = float(row[f"gm{k}y"]) / 8 - ty
                errs.append(math.hypot(ex, ey))
                zero_errs.append(math.hypot(tx, ty))
        if kind in ("I", "P"):
            ref = i  # the reference of the next P-frame
    if not errs:
        raise SystemExit("no P-frames in the log")
    e = np.array(errs)
    print(f"{len(errs) // 4} P-frames: corner error mean {e.mean():.3f} px, rms "
          f"{math.sqrt((e ** 2).mean()):.3f}, p95 {np.percentile(e, 95):.3f}, max {e.max():.3f}"
          f"  (true corner motion mean {np.mean(zero_errs):.2f} px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--preset", default="orbit",
                    help="survey, pan, orbit, climb, tilt, static, hover, shaky or mixed")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--noise", type=float, default=1.5, help="sensor noise sigma (RGB levels)")
    ap.add_argument("--gain", type=float, default=0.0, help="exposure drift amplitude, e.g. 0.05")
    ap.add_argument("--sprite", help="image for an independently moving object")
    ap.add_argument("--sprite-size", type=int, default=96)
    ap.add_argument("--sprite-speed", type=float, default=3.0, help="canvas pixels per frame")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--truth", help="write (or, with --check, read) the ground-truth CSV")
    ap.add_argument("--check", help="frame log from `vdwt encode --frame-log` to evaluate")
    ap.add_argument("--mesh", type=int, default=32, help="codec motion grid spacing")
    args = ap.parse_args()
    if args.check:
        if not args.truth:
            raise SystemExit("--check needs --truth")
        check(args)
    else:
        if not args.image or not args.output:
            raise SystemExit("an input image and an output .y4m path are needed")
        generate(args)


if __name__ == "__main__":
    main()
