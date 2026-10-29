#!/usr/bin/env python3
"""Oracle for a motion-compensated temporal pre-filter (MCTF) on the first (intra) frame.

    mctf_oracle.py IN.y4m OUT.y4m STRENGTH [REFS]

Frame 0 of IN is replaced by a weighted average of itself and its next REFS frames (default 2),
each aligned by 8x8 full-search block matching (±12 px, luma; chroma follows at half the
vector). Per block and reference the weight is STRENGTH * exp(-mse / (2 * s2)), s2 the block's
expected noise power (VVC VTM's temporal filter idea): well-matched references average in,
mismatched ones (occlusion, new content) fall away. The other frames are unchanged. Encode OUT
and measure PSNR against IN: the filter must pay for itself against the original source.
"""
import sys
import numpy as np

B, R = 8, 12


def read(path):
    f = open(path, "rb")
    hdr = f.readline()
    t = hdr.split()
    w = int(next(x for x in t if x.startswith(b"W"))[1:])
    h = int(next(x for x in t if x.startswith(b"H"))[1:])
    frames = []
    while f.readline():
        b = f.read(w * h * 3 // 2)
        if len(b) < w * h * 3 // 2:
            break
        y = np.frombuffer(b[: w * h], np.uint8).reshape(h, w).astype(np.float32)
        u = np.frombuffer(b[w * h: w * h * 5 // 4], np.uint8).reshape(h // 2, w // 2).astype(np.float32)
        v = np.frombuffer(b[w * h * 5 // 4:], np.uint8).reshape(h // 2, w // 2).astype(np.float32)
        frames.append([y, u, v])
    return hdr, frames


def block_vectors(cur, ref):
    h, w = cur.shape
    pad = np.pad(ref, R, mode="edge")
    by, bx = h // B, w // B
    cb = cur[: by * B, : bx * B].reshape(by, B, bx, B)
    best = np.full((by, bx), np.inf)
    vec = np.zeros((by, bx, 2), int)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sh = pad[R + dy: R + dy + by * B, R + dx: R + dx + bx * B].reshape(by, B, bx, B)
            e = ((cb - sh) ** 2).mean(axis=(1, 3))
            m = e < best
            best[m] = e[m]
            vec[m] = (dy, dx)
    return vec, best


def warp(ref, vec, scale):
    """`ref` displaced by per-block vectors (block size B*scale)."""
    b = int(B * scale)
    h, w = ref.shape
    out = ref.copy()
    pad = np.pad(ref, R, mode="edge")
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            dy, dx = (vec[i, j] * scale).round().astype(int)
            y0, x0 = i * b, j * b
            out[y0: y0 + b, x0: x0 + b] = pad[R + y0 + dy: R + y0 + dy + b, R + x0 + dx: R + x0 + dx + b]
    return out


def filter_frame(frames, i, strength, offsets):
    cur = frames[i]
    lap = np.abs(cur[0][1:-1, 1:-1] * 4 - cur[0][:-2, 1:-1] - cur[0][2:, 1:-1] - cur[0][1:-1, :-2] - cur[0][1:-1, 2:])
    s2 = max((np.median(lap) / 0.6745 / np.sqrt(20)) ** 2, 1.0) * 4
    acc = [p.copy() for p in cur]
    wsum = [np.ones_like(p) for p in cur]
    for o in offsets:
        if not 0 <= i + o < len(frames):
            continue
        ref = frames[i + o]
        vec, err = block_vectors(cur[0], ref[0])
        wblk = strength * np.exp(-err / (2 * s2))
        wy = np.kron(wblk, np.ones((B, B)))
        wc = np.kron(wblk, np.ones((B // 2, B // 2)))
        for k, (p, wk) in enumerate([(ref[0], wy), (ref[1], wc), (ref[2], wc)]):
            al = warp(p, vec, 1.0 if k == 0 else 0.5)
            hh, ww = wk.shape
            acc[k][:hh, :ww] += wk * al[:hh, :ww]
            wsum[k][:hh, :ww] += wk
    return [np.clip(np.round(a / s), 0, 255).astype(np.uint8) for a, s in zip(acc, wsum)]


def main():
    """mctf_oracle.py IN OUT STRENGTH [MODE]: MODE first (frame 0 from frames 1-2) or all (every
    frame from its neighbours at -2..+2)."""
    src, dst, strength = sys.argv[1], sys.argv[2], float(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "first"
    hdr, frames = read(src)
    out = []
    for i in range(len(frames)):
        if mode == "all":
            out.append(filter_frame(frames, i, strength, [-2, -1, 1, 2]))
        elif i == 0:
            out.append(filter_frame(frames, 0, strength, [1, 2]))
        else:
            out.append([np.clip(p, 0, 255).astype(np.uint8) for p in frames[i]])
    with open(dst, "wb") as f:
        f.write(hdr)
        for planes in out:
            f.write(b"FRAME\n")
            for p in planes:
                f.write(p.tobytes())


if __name__ == "__main__":
    main()
