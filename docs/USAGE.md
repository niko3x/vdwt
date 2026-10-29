# VDWT command reference

`vdwt help` prints every command and option of your build. This page covers the ones meant for
everyday use.

## encode

```
vdwt encode IN OUT.vdwt [--qp 44 | --bitrate RATE[@FRAME],...] [options]
```

`IN` is a `.y4m` file (4:2:0, 4:2:2, 4:4:4 or monochrome, 8–16 bits; coded as 8-bit 4:2:0), `-`
for Y4M on standard input, or any other video file `ffmpeg` can read.

| Option | Meaning |
|---|---|
| `--qp N` | Fixed quality. Higher is coarser; +8 doubles the quantizer step. |
| `--bitrate RATE` | Constant bitrate that follows the link, e.g. `3M`, `600k`, or a schedule of changes at given frames: `4M@0,1500k@40,300k@110`. |
| `--max-delay F` | Queueing-delay limit in frame durations (default 1.5). |
| `--quality-cap DB` | Stop refining a frame beyond this estimated luma PSNR and bank the spare capacity (default 48; 0: off). |
| `--tier NAME` | `light`, `medium` (default), `heavy` or `ultraheavy`. |
| `--realtime` | The realtime tier, for small ARM CPUs. |
| `--frames N` | Encode only the first N frames. |
| `--recon FILE.y4m` | Also write the encoder's reconstruction. |
| `--frame-log FILE.csv` | Per-frame type, size, qp and rate-control state. |
| `--threads N` | Worker threads (default: up to 4). The stream does not depend on it. |
| `--intra-period N` | An intra frame every N frames (0: only the first frame and scene cuts). |
| `--intra-only` | Code every frame intra. |
| `--tune psnr` / `perceptual` | `psnr` (default) maximizes PSNR; `perceptual` keeps more fine detail at some PSNR cost. |
| `--lossless` | Exact reconstruction. |
| `--postfilter auto` | Let the encoder choose a display sharpening strength per scene and signal it in the stream. |

## decode

```
vdwt decode IN.vdwt OUT.y4m [--threads N] [--postfilter auto|none]
```

`OUT` may be `-` for standard output; the timing report then goes to standard error.
`--postfilter auto` applies the strength the encoder signalled; `none` shows the unfiltered
picture.

## rt-encode / rt-decode (lossy links)

```
vdwt rt-encode IN.y4m OUT.vdrt --bitrate RATE [--mtu 1200] [--refresh 30]
vdwt rt-decode IN.vdrt OUT.y4m [--loss 0.05] [--burst 1] [--ref SOURCE.y4m]
```

The `rt` mode splits every frame into independent tiles carried in self-contained packets of at
most `--mtu` bytes, refreshing each tile every `--refresh` frames, so a lost packet damages one
tile for a short time instead of the stream. `rt-decode` can simulate packet loss (`--loss`,
`--burst`) and report PSNR against the source (`--ref`).

## Utilities

| Command | What it does |
|---|---|
| `vdwt info` | CPU features and the kernels in use |
| `vdwt selftest` | Checks every optimized kernel bit for bit on this CPU |
| `vdwt psnr A.y4m B.y4m` | Per-frame and average PSNR (Y, U, V and 6:1:1 YUV) |
| `vdwt rd IN.y4m --qp 32,40,48` | Rate-quality table over several qps, checking that decoding matches |
| `vdwt bench` | Speed of the core kernels on one core |

## Examples

Live-style encode at 1.5 Mbit/s that drops to 300 kbit/s at frame 300:

```bash
vdwt encode drone.mp4 drone.vdwt --realtime --bitrate 1500k@0,300k@300
```

Best compression for a file:

```bash
vdwt encode talk.mov talk.vdwt --tier ultraheavy --qp 44
```

Decode straight into a lossless MKV:

```bash
vdwt decode talk.vdwt - | ffmpeg -i - -c:v ffv1 talk.mkv
```
