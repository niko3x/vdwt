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

## Results in this directory

| file | what |
|---|---|
| `592f920.json` | VDWT at 592f920 (rate-aware motion grid), the baseline |
| `49c379e.json` | VDWT at 49c379e (joint band streams, VERSION 12) |
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
