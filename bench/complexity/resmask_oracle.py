#!/usr/bin/env python3
"""Oracle for a residual weight field: final = prediction + w * (reconstruction - prediction),
w per motion-grid node (0 or 1), bilinear between nodes as the codec ramps intra weights.

    resmask_oracle.py SOURCE.y4m PRED.y4m RECON.y4m SPACING

PRED is `vdwt pred-dump` (mid-grey on intra frames, which are skipped), RECON the same encode's
--recon. Each node keeps or drops its residual by the error over its cell, then the ramped field
is applied and every P-frame is scored. Prints the YUV-PSNR 6:1:1 change, the share of nodes
masked and the flag count, so the gain can be set against one flag per node.
"""
import math, sys
import numpy as np


def read(path):
    f = open(path, "rb")
    t = f.readline().split()
    w = int(next(x for x in t if x.startswith(b"W"))[1:])
    h = int(next(x for x in t if x.startswith(b"H"))[1:])
    out = []
    while f.readline():
        b = f.read(w * h * 3 // 2)
        if len(b) < w * h * 3 // 2:
            break
        a = np.frombuffer(b, np.uint8).astype(np.float64)
        out.append((a[: w * h].reshape(h, w), a[w * h: w * h * 5 // 4].reshape(h // 2, w // 2),
                    a[w * h * 5 // 4:].reshape(h // 2, w // 2)))
    return out


def field(nodes, h, w, c):
    """Bilinear interpolation of node values (node (i, j) at pixel (i*c, j*c))."""
    ny, nx = nodes.shape
    ys, xs = np.arange(h) / c, np.arange(w) / c
    j0 = np.clip(np.floor(ys).astype(int), 0, ny - 2)
    i0 = np.clip(np.floor(xs).astype(int), 0, nx - 2)
    ty, tx = (ys - j0)[:, None], (xs - i0)[None, :]
    a, b = nodes[j0][:, i0], nodes[j0][:, i0 + 1]
    cc, d = nodes[j0 + 1][:, i0], nodes[j0 + 1][:, i0 + 1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + d * tx) * ty


def psnr(m):
    return 10 * math.log10(255 ** 2 / max(m, 1e-9))


def main():
    src, pred, rec = (read(p) for p in sys.argv[1:4])
    c = int(sys.argv[4])
    h, w = src[0][0].shape
    ny, nx = math.ceil(h / c) + 1, math.ceil(w / c) + 1
    gains, masked, frames = [], 0, 0
    for s, p, r in zip(src, pred, rec):
        if np.all(p[0] == 128):
            continue  # intra frame
        frames += 1
        # Per node: luma error over its cell with and without the residual.
        e1 = (s[0] - r[0]) ** 2
        e0 = (s[0] - p[0]) ** 2
        nodes = np.ones((ny, nx))
        for j in range(ny):
            for i in range(nx):
                y0, y1 = max(j * c - c // 2, 0), min(j * c + c // 2, h)
                x0, x1 = max(i * c - c // 2, 0), min(i * c + c // 2, w)
                if y0 < y1 and x0 < x1 and e0[y0:y1, x0:x1].sum() < e1[y0:y1, x0:x1].sum():
                    nodes[j, i] = 0
        masked += int((nodes == 0).sum())
        wy = field(nodes, h, w, c)
        wc = field(nodes, h // 2, w // 2, c / 2)
        out = [p[0] + wy * (r[0] - p[0]), p[1] + wc * (r[1] - p[1]), p[2] + wc * (r[2] - p[2])]
        out = [np.clip(np.round(o), 0, 255) for o in out]
        before = [((a - b) ** 2).mean() for a, b in zip(s, r)]
        after = [((a - b) ** 2).mean() for a, b in zip(s, out)]
        b6 = (6 * psnr(before[0]) + psnr(before[1]) + psnr(before[2])) / 8
        a6 = (6 * psnr(after[0]) + psnr(after[1]) + psnr(after[2])) / 8
        gains.append(a6 - b6)
    print(f"{sys.argv[1].split('/')[-1]:12s} spacing {c}: P-frames {frames}, 6:1:1 gain "
          f"{np.mean(gains):+.3f} dB (max {max(gains):+.3f}), nodes masked "
          f"{100 * masked / (frames * nx * ny):.1f}% of {nx * ny} per frame")


if __name__ == "__main__":
    main()
