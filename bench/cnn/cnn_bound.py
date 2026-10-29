#!/usr/bin/env python3
"""Float upper bound for an in-loop CNN restoration filter (applied offline to decoded frames).

    cnn_bound.py train DIR_TRAIN SOURCES_DIR MODEL.pt [ITERS]
    cnn_bound.py eval  DIR_TEST  SOURCES_DIR MODEL.pt KBPS.txt

Training pairs are DIR_TRAIN/<clip>_q<qp>.y4m (VDWT reconstructions) against
SOURCES_DIR/<clip>.y4m. Two small residual networks: luma (decoded Y + a qp plane, 6 convs of
32 channels) and chroma (decoded U, V + the luma folded to chroma resolution + qp). Evaluation
filters every decoded frame of DIR_TEST and prints the YUV-PSNR 6:1:1 BD-rate of the filtered
decode against the unfiltered one at the same bitrates (KBPS.txt: "clip qp kbps"). Applied
outside the loop, the result is a lower bound for the in-loop gain.
"""
import math, os, re, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("THREADS", "4")))


def read_y4m(path, n=None):
    f = open(path, "rb")
    t = f.readline().split()
    w = int(next(x for x in t if x.startswith(b"W"))[1:])
    h = int(next(x for x in t if x.startswith(b"H"))[1:])
    ys, us, vs = [], [], []
    while (n is None or len(ys) < n) and f.readline():
        b = f.read(w * h * 3 // 2)
        if len(b) < w * h * 3 // 2:
            break
        a = np.frombuffer(b, np.uint8)
        ys.append(a[: w * h].reshape(h, w))
        us.append(a[w * h: w * h * 5 // 4].reshape(h // 2, w // 2))
        vs.append(a[w * h * 5 // 4:].reshape(h // 2, w // 2))
    return np.stack(ys), np.stack(us), np.stack(vs)


def net(cin, cout, ch=32, depth=6):
    layers = [nn.Conv2d(cin, ch, 3, padding=1, padding_mode="replicate"), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"), nn.ReLU()]
    layers += [nn.Conv2d(ch, cout, 3, padding=1, padding_mode="replicate")]
    m = nn.Sequential(*layers)
    # He initialisation, the output layer scaled down: the network starts near the identity
    # but its gradients are not starved (a zero-initialised output layer learned ~nothing).
    for l in m:
        if isinstance(l, nn.Conv2d):
            nn.init.kaiming_normal_(l.weight, nonlinearity="relu")
            nn.init.zeros_(l.bias)
    m[-1].weight.data *= 0.1
    return Scaled(m)


class Scaled(nn.Module):
    """Inputs centred and scaled to about unit range (samples are in [0, 1]); the correction is
    produced 16 times larger than applied, so its useful range is not a few 1e-3."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return self.body((x - 0.5) * 4) / 16


def fold(y):
    """Luma (N, 1, H, W) to chroma resolution as 4 channels (space-to-depth)."""
    return torch.nn.functional.pixel_unshuffle(y, 2)


def luma_in(y, qp):
    return torch.cat([y, torch.full_like(y, qp / 128.0 + 0.25)], 1)


def chroma_in(u, v, y, qp):
    return torch.cat([u, v, fold(y), torch.full_like(u, qp / 128.0 + 0.25)], 1)


def pairs(dir_, src_dir):
    out = []
    for p in sorted(Path(dir_).glob("*_q*.y4m")):
        clip, qp = re.match(r"(.+)_q(\d+)\.y4m", p.name).groups()
        rec = read_y4m(p)
        src = read_y4m(Path(src_dir) / f"{clip}.y4m", len(rec[0]))
        out.append((clip, int(qp), rec, src))
    return out


def train(dir_, src_dir, model_path, iters):
    data = pairs(dir_, src_dir)
    rng = np.random.default_rng(1)
    ln, cn = net(2, 1).to(DEV), net(7, 2).to(DEV)
    opt = torch.optim.Adam(list(ln.parameters()) + list(cn.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    P, BATCH = 64, 16
    for it in range(iters):
        ry, ru, rv, sy, su, sv, qs = [], [], [], [], [], [], []
        for _ in range(BATCH):
            clip, qp, rec, src = data[rng.integers(len(data))]
            f = rng.integers(len(rec[0]))
            h, w = rec[0].shape[1:]
            y0, x0 = 2 * rng.integers((h - P) // 2), 2 * rng.integers((w - P) // 2)
            sl = (f, slice(y0, y0 + P), slice(x0, x0 + P))
            cl = (f, slice(y0 // 2, (y0 + P) // 2), slice(x0 // 2, (x0 + P) // 2))
            ry.append(rec[0][sl]); ru.append(rec[1][cl]); rv.append(rec[2][cl])
            sy.append(src[0][sl]); su.append(src[1][cl]); sv.append(src[2][cl])
            qs.append(qp)
        t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32) / 255.0)[:, None].to(DEV)
        Ry, Ru, Rv, Sy, Su, Sv = map(t, (ry, ru, rv, sy, su, sv))
        q = torch.tensor(qs, dtype=torch.float32, device=DEV)[:, None, None, None] / 128.0 + 0.25
        oy = Ry + ln(torch.cat([Ry, q.expand_as(Ry)], 1))
        oc = torch.cat([Ru, Rv], 1) + cn(torch.cat([Ru, Rv, fold(Ry), q.expand_as(Ru)], 1))
        # 6:1:1: luma counts 6, each chroma plane 1.
        loss = 6 * ((oy - Sy) ** 2).mean() + ((oc - torch.cat([Su, Sv], 1)) ** 2).mean() * 2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0 or it == iters - 1:
            print(f"iter {it} loss {loss.item() * 255 ** 2 / 8:.2f} (6:1:1 MSE)", flush=True)
    torch.save({"luma": ln.state_dict(), "chroma": cn.state_dict()}, model_path)


def psnr(a, b):
    m = ((a.astype(np.float64) - b) ** 2).mean(axis=(1, 2))
    return float(np.mean(10 * np.log10(255 ** 2 / np.maximum(m, 1e-9))))


@torch.no_grad()
def evaluate(dir_, src_dir, model_path, kbps_path):
    m = torch.load(model_path)
    ln, cn = net(2, 1).to(DEV), net(7, 2).to(DEV)
    ln.load_state_dict(m["luma"]); cn.load_state_dict(m["chroma"])
    kbps = {(l.split()[0], int(l.split()[1])): float(l.split()[2]) for l in open(kbps_path) if l.strip()}
    rd = {}
    for clip, qp, rec, src in pairs(dir_, src_dir):
        out = [[], [], []]
        for f in range(len(rec[0])):
            t = lambda a: torch.from_numpy(a[f].astype(np.float32) / 255.0)[None, None].to(DEV)
            y, u, v = t(rec[0]), t(rec[1]), t(rec[2])
            oy = y + ln(luma_in(y, qp))
            oc = torch.cat([u, v], 1) + cn(chroma_in(u, v, y, qp))
            to8 = lambda x: (x.clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
            out[0].append(to8(oy)); out[1].append(to8(oc[:, :1])); out[2].append(to8(oc[:, 1:]))
        p = lambda planes: (6 * psnr(planes[0], src[0]) + psnr(planes[1], src[1]) + psnr(planes[2], src[2])) / 8
        before, after = p(rec), p([np.stack(o) for o in out])
        rd.setdefault(clip, []).append((kbps[(clip, qp)], before, after))
        print(f"{clip:9s} q{qp}: {before:.3f} -> {after:.3f} dB ({after - before:+.3f})", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "complexity"))
    from plot import bd_rate
    total = []
    for clip, pts in sorted(rd.items()):
        b = bd_rate([(k, x) for k, x, _ in pts], [(k, y) for k, _, y in pts])
        total.append(b)
        print(f"{clip:9s} BD-rate {b:+.2f}%")
    print(f"average   BD-rate {np.mean(total):+.2f}%")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 4000)
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
