#!/usr/bin/env python3
"""Oracle for block-free edge-directed (adaptive directional) lifting on intra frames.

    adl_oracle.py FRAME.y4m:INDEX ...

Compares a plain 5-level 5/3 lifting wavelet with the same wavelet whose predict steps follow a
local direction (ADL: the odd row/column is predicted from the even neighbours displaced by ±d
along the other axis, d in half pixels, linear interpolation). Directions are chosen by the
encoder per region of each level and step (an oracle: minimum |highpass|), as a codec would
carry them on a smooth mesh like the motion field; their side information is charged at the
entropy of the chosen indices. Rate is the zeroth-order entropy of the dead-zone-quantized
coefficients per subband (the same estimator for both arms), distortion the true PSNR after the
inverse transform. Prints the BD-rate of ADL against plain lifting, with and without side info.
The update steps stay non-directional (invertible either way).
"""
import math, os, sys
import numpy as np

LEVELS = 5
# Displacement in half pixels per row of distance; region side in samples of the band being
# split; a direction other than 0 must cut the region's |highpass| by at least BIAS (the RD
# preference for the free "no direction" choice).
DIRS = [int(v) for v in os.environ.get("DIRS", "-4,-3,-2,-1,0,1,2,3,4").split(",")]
REGION = int(os.environ.get("REGION", "16"))
BIAS = float(os.environ.get("BIAS", "0"))


def read_frame(spec):
    path, idx = spec.rsplit(":", 1)
    f = open(path, "rb")
    hdr = f.readline().split()
    w = int(next(t for t in hdr if t.startswith(b"W"))[1:])
    h = int(next(t for t in hdr if t.startswith(b"H"))[1:])
    fs = w * h * 3 // 2
    for _ in range(int(idx) + 1):
        f.readline()
        b = f.read(fs)
    return np.frombuffer(b, np.uint8)[: w * h].reshape(h, w).astype(np.float64)


def shifted(row, d2):
    """`row` displaced by d2 half-pixels (linear interpolation, edges clamped)."""
    n = row.shape[-1]
    x = np.arange(n) + d2 / 2.0
    x = np.clip(x, 0, n - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    t = x - i0
    return row[..., i0] * (1 - t) + row[..., i1] * t


def region_map(choice, n):
    """Per-sample direction index from a per-region choice (regions of REGION samples)."""
    return np.repeat(choice, REGION)[:n]


DECSIDE = os.environ.get("DECSIDE") == "1"


def est_dirs(even, rows):
    """Decoder-side directions: per region of odd rows, the displacement under which the even
    rows above and below match best (from even samples only, which the decoder rebuilds before
    predicting the odd ones). A direction other than 0 must beat it by BIAS."""
    above = even[:rows]
    below = np.vstack([even[1:], even[-1:]])[:rows]
    w = even.shape[1]
    nreg, nrr = math.ceil(w / REGION), math.ceil(rows / REGION)
    costs = []
    for d in DIRS:
        e = np.abs(shifted(above, d) - shifted(below, -d))
        e = np.pad(e, ((0, nrr * REGION - rows), (0, nreg * REGION - w)), mode="edge")
        costs.append(e.reshape(nrr, REGION, nreg, REGION).sum(axis=(1, 3)))
    cost = np.stack(costs)
    zero = DIRS.index(0)
    cost[np.arange(len(DIRS)) != zero] *= 1 / (1 - BIAS) if BIAS < 1 else np.inf
    return cost.argmin(axis=0)


def predict_rows(x, dirs_idx=None, oracle=False):
    """One vertical lifting predict step on rows: odd rows -= mean of the even rows above and
    below, displaced by ±d. Returns (high, low_unupdated, chosen indices per (odd row, region))."""
    even, odd = x[0::2], x[1::2]
    below = np.vstack([even[1:], even[-1:]])[: odd.shape[0]]
    above = even[: odd.shape[0]]
    w = x.shape[1]
    nreg = math.ceil(w / REGION)
    if oracle:
        cands = []
        for d in DIRS:
            pred = 0.5 * (shifted(above, d) + shifted(below, -d))
            cands.append(odd - pred)
        cands = np.stack(cands)  # (D, rows, w)
        # Region cost: sum |h| over REGION columns of a band of REGION odd rows.
        pad = nreg * REGION - w
        c = np.abs(np.pad(cands, ((0, 0), (0, 0), (0, pad)), mode="edge"))
        rows = odd.shape[0]
        nrr = math.ceil(rows / REGION)
        c = np.pad(c, ((0, 0), (0, nrr * REGION - rows), (0, 0)), mode="edge")
        cost = c.reshape(len(DIRS), nrr, REGION, nreg, REGION).sum(axis=(2, 4))
        zero = DIRS.index(0)
        cost[np.arange(len(DIRS)) != zero] *= 1 / (1 - BIAS) if BIAS < 1 else np.inf
        dirs_idx = cost.argmin(axis=0)  # (nrr, nreg)
    high = np.empty_like(odd)
    for rr in range(dirs_idx.shape[0]):
        r0, r1 = rr * REGION, min((rr + 1) * REGION, odd.shape[0])
        dmap = region_map(dirs_idx[rr], w)
        h = odd[r0:r1].copy()
        for k, d in enumerate(DIRS):
            m = dmap == k
            if m.any():
                pred = 0.5 * (shifted(above[r0:r1], d) + shifted(below[r0:r1], -d))
                h[:, m] -= pred[:, m]
        high[r0:r1] = h
    return high, even, dirs_idx


def unpredict_rows(high, even, dirs_idx):
    below = np.vstack([even[1:], even[-1:]])[: high.shape[0]]
    above = even[: high.shape[0]]
    w = high.shape[1]
    odd = np.empty_like(high)
    for rr in range(dirs_idx.shape[0]):
        r0, r1 = rr * REGION, min((rr + 1) * REGION, high.shape[0])
        dmap = region_map(dirs_idx[rr], w)
        o = high[r0:r1].copy()
        for k, d in enumerate(DIRS):
            m = dmap == k
            if m.any():
                pred = 0.5 * (shifted(above[r0:r1], d) + shifted(below[r0:r1], -d))
                o[:, m] += pred[:, m]
        odd[r0:r1] = o
    return odd


def update(even, high):
    """even += (h_above + h_below)/4 (non-directional 5/3 update)."""
    hp = np.vstack([high[:1], high])[: even.shape[0]]          # h of the odd row above
    hn = np.vstack([high, high[-1:]])[: even.shape[0]]          # h of the odd row below
    return even + 0.25 * (hp + hn)


def split_rows(x, directional, dirs=None):
    zero = np.zeros((math.ceil((x.shape[0] // 2) / REGION), math.ceil(x.shape[1] / REGION)), int) + DIRS.index(0)
    if directional and DECSIDE:
        high, even, d = predict_rows(x, est_dirs(x[0::2], x.shape[0] // 2))
    elif directional and dirs is None:
        high, even, d = predict_rows(x, oracle=True)
    else:
        high, even, d = predict_rows(x, dirs if dirs is not None else zero)
    low = update(even, high)
    return low, high, d


def merge_rows(low, high, d):
    even = low - 0.25 * (np.vstack([high[:1], high])[: low.shape[0]] + np.vstack([high, high[-1:]])[: low.shape[0]])
    if DECSIDE and d.max() >= 0 and getattr(merge_rows, "decoded", False):
        d = est_dirs(even, high.shape[0])  # the decoder's own estimate, from decoded samples
    odd = unpredict_rows(high, even, d)
    x = np.empty((even.shape[0] + odd.shape[0], even.shape[1]))
    x[0::2], x[1::2] = even, odd
    return x


def forward(img, directional):
    bands, side = [], []
    ll = img
    for _ in range(LEVELS):
        lo, hi, dv = split_rows(ll, directional)              # vertical step
        lol, loh, dl = split_rows(lo.T, directional)          # horizontal step on each half
        hil, hih, dh = split_rows(hi.T, directional)
        bands.append((loh.T, hil.T, hih.T))
        side += [dv, dl, dh]
        ll = lol.T
        bands_dirs = (dv, dl, dh)
        bands[-1] = bands[-1] + (bands_dirs,)
    return ll, bands, side


def inverse(ll, bands, decoded=False):
    merge_rows.decoded = decoded
    for loh, hil, hih, (dv, dl, dh) in reversed(bands):
        lo = merge_rows(ll.T, loh.T, dl).T
        hi = merge_rows(hil.T, hih.T, dh).T
        ll = merge_rows(lo, hi, dv)
    return ll


def entropy_bits(q):
    _, counts = np.unique(q, return_counts=True)
    p = counts / q.size
    return float(-(counts * np.log2(p)).sum())


def code(ll, bands, step):
    """Dead-zone quantization (θ = 0.25 as the codec's intra default), reconstruction offset
    0.2; returns (bits, dequantized ll, dequantized bands)."""
    def q(b):
        a = np.abs(b) / step
        m = np.floor(a + 0.25)
        return np.sign(b) * m
    def dq(m):
        return np.sign(m) * (np.abs(m) + 0.2 * (m != 0)) * step
    bits, out = 0.0, []
    qll = np.round(ll / step)
    bits += entropy_bits(qll)
    for loh, hil, hih, d in bands:
        qs = [q(b) for b in (loh, hil, hih)]
        bits += sum(entropy_bits(m) for m in qs)
        out.append(tuple(dq(m) for m in qs) + (d,))
    return bits, qll * step, out


def side_bits(side):
    return sum(entropy_bits(d) for d in side)


def bd(ref, test):
    """BD-rate (%) of test against ref; each [(bits, psnr)], piecewise-linear in log rate."""
    a = sorted((p, math.log(r)) for r, p in ref)
    b = sorted((p, math.log(r)) for r, p in test)
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    def it(c, x):
        for (x0, y0), (x1, y1) in zip(c, c[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
        return c[-1][1] if x > c[-1][0] else c[0][1]
    xs = [lo + (hi - lo) * i / 100 for i in range(101)]
    return 100 * (math.exp(sum(it(b, x) - it(a, x) for x in xs) / len(xs)) - 1)


def main():
    steps = [int(s) for s in os.environ.get("STEPS", "19,27,38,54,76").split(",")]
    for spec in sys.argv[1:]:
        img = read_frame(spec) - 128.0
        h, w = img.shape
        img = img[: h // 32 * 32, : w // 32 * 32]
        res = {}
        for directional in (False, True):
            ll, bands, side = forward(img, directional)
            rec0 = inverse(ll, bands)
            assert np.abs(rec0 - img).max() < 1e-6, "not invertible"
            sb = side_bits(side) if directional and not DECSIDE else 0.0
            pts, pts_side = [], []
            for s in steps:
                bits, qll, qb = code(ll, bands, s)
                rec = inverse(qll, qb, decoded=directional and os.environ.get("NOMISMATCH") != "1")
                mse = ((np.clip(np.round(rec), -128, 127) - img) ** 2).mean()
                psnr = 10 * math.log10(255 ** 2 / mse)
                pts.append((bits, psnr))
                pts_side.append((bits + sb, psnr))
            res[directional] = (pts, pts_side, sb)
        base = res[False][0]
        adl, adl_side, sb = res[True]
        print(f"{spec.split('/')[-1]:28s} ADL vs 5/3: {bd(base, adl):+6.2f}% without side info, "
              f"{bd(base, adl_side):+6.2f}% with {sb / 8:.0f} bytes of directions "
              f"(plain intra {base[0][0] / 8:.0f}..{base[-1][0] / 8:.0f} bytes)", flush=True)


if __name__ == "__main__":
    main()
