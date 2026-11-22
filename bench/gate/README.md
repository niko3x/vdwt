# Benchmark gate

`tools/gate.py` decides whether a codec change is better in general, not on one clip. Clips:
`bash tools/fetch_gate.sh` (the first 120 frames of ten standard sequences from
media.xiph.org — Netflix Aerial, DrivingPOV, RollerCoaster, BoxingPractice, Crosswalk; JVET
FourPeople, KristenAndSara; SVT crowd_run, ducks_take_off, old_town_cross — scaled to
1280×720), plus the Luckfox 512×288 segment (`videos/luckfox/src_mid_288p_500.y4m`).

Each clip is encoded at three CBR rates (the codec's real use, a low-delay link) and, for VDWT,
on a four-point fixed-qp curve, at medium and ultraheavy; scores are YUV-PSNR 6:1:1 and VMAF
v0.6.1. A change passes when the mean BD-rate improves and no clip loses more than 1%.

    python3 tools/gate.py run TAG [--bin BINARY] [--quick] [--clips a,b] [--opts "..."]
    python3 tools/gate.py compare BASE_TAG NEW_TAG
    python3 tools/gate.py refs                 # x265 / SVT-AV1 reference lines
    python3 tools/gate.py where TAG            # VDWT against them

A full run is about 20 minutes on the i5-10210U (126 encodes, three at a time); `--quick`
(medium, CBR) about 5.

Gate v2 options (decoder roadmap, Phase 0):

- `--tiers realtime,light,medium,heavy,ultraheavy`: `realtime` is medium with `--realtime`.
- `--long` (and `refs --long`, `where --long`): the long set from `bash tools/fetch_gate.sh
  --long` — 300 frames of FourPeople, KristenAndSara, crowd_run, ducks_take_off and
  old_town_cross (the 4K sources would be 4 GB of download each) plus 300 frames of Luckfox.
  Tags end in `_long`; references are `ref_<codec>_long.json`.
- Every VDWT encode is also decoded once on one thread; rows carry `dec_fps`, and `compare`
  prints the decode-time change per tier. A normative change states it. Times are wall clock
  with `-j` jobs in parallel, so compare runs made the same way on an idle machine.
- `--cvvdp`: also ColorVideoVDP (JOD, `standard_fhd`), which sees chroma and temporal
  artifacts VMAF does not. It needs pycvvdp (installed in `~/.cache/vdwt-cvvdp`, CPU torch;
  `$CVVDP` overrides the path) and costs about 2 minutes per 720p 120-frame encode here, so it
  is for phase-end checks. `compare` and `where` add a CVVDP column when both sides have it.

Offline tools for deciding what to build before building it:

- `vdwt ctx-sim CLIPS [--qp Q | --bitrate R] [--frames N] [--only names]` replays every
  written band's and motion section's symbols through exact copies of the adaptive CDFs and
  prices context refinements (`crates/vdwt/src/ctxsim.rs`); its baseline matches the coded
  segments to within the range coder's flush bytes.
- `tools/pred_oracle.py CLIP [--qp Q] [--dense R]` prices ideal prediction tools (gain/offset,
  smooth offset fields, a second hypothesis, a dense block-motion upper bound) against the
  real prediction (`vdwt pred-dump`).

## Results in this directory

| file | what |
|---|---|
| `592f920.json` | VDWT at 592f920 (rate-aware motion grid), the baseline |
| `49c379e.json` | VDWT at 49c379e (joint band streams, VERSION 12) |
| `e2e268a.json` | VDWT at e2e268a (plus moving-cut keyframes and chroma 0.45 at 720p): −2.1 to −2.9% PSNR, −4.8 to −5.5% VMAF against 592f920; ultraheavy −1.6 / −0.5% against x265 (see `overnight-2026-11-21.md`) |
| `ref_x265.json` | x265 4.3 `--preset medium --tune zerolatency`, CBR, VBV of three frames, no B-frames |
| `ref_svtav1.json` | SVT-AV1 preset 8, low-delay (`--pred-struct 1`), CBR, 3-frame buffer |
| `ref_x265qp.json` | x265 as above at fixed QP 22–37 (coaster, oldtown, aerial, kristen only) |

`49c379e` against `592f920` (BD-rate, PSNR / VMAF; 11 clips): medium CBR −1.66 / −1.84%,
medium qp −1.61 / −1.81%, ultraheavy CBR −1.20 / −1.27%, ultraheavy qp −1.21 / −1.38%.

`49c379e` against the references (CBR, mean of 11 clips, PSNR / VMAF): medium +11.8 / +12.5%
and ultraheavy −0.4 / +3.0% against x265; medium +59 / +73% and ultraheavy +35 / +54% against
SVT-AV1 (whose CBR overshoots the low rates by up to 90%). VDWT wins clearly on Luckfox
(−28 / −42% against x265) and Crosswalk, and loses most on RollerCoaster (smooth sky with
grain, +55%: intra alone needs 3.5× x265's bits there) and old_town_cross at medium.
