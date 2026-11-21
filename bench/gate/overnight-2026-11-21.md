# Overnight optimization round, 2026-11-20/21

Every change below was judged by the benchmark gate (`bench/gate/README.md`): 10 standard
720p clips plus the Luckfox 288p segment, CBR at three rates and a four-point fixed-qp curve,
medium and ultraheavy, BD-rate on YUV-PSNR 6:1:1 and VMAF. A change ships only when the mean
improves and no clip loses more than 1%. Negative numbers are fewer bits at equal quality.

## Shipped

Commit hashes are the development branch's; the public `main` carries the same changes under
its own hashes.

| commit | change | gate result |
|---|---|---|
| 592f920 | Rate-aware motion grid: probes run under rate control; a moving camera below 0.03 bpp gets ≥ 32 px | Luckfox mid 80k +0.52 / +0.68 dB (medium / ultraheavy), coast 400k +0.17 / +0.27; still cameras and fixed qp unchanged (pushed to the public repo earlier as aaeca82) |
| 49c379e | **Joint band streams (VERSION 12)**: a frame whose bands total ≤ 4 KB codes them all in one range stream (one mode bit, a presence symbol per band) instead of separate segments plus a length layout; the rate model learns the written size | −1.2 to −1.9% on every tier and mode, all 11 clips improve; about −5% at Luckfox 80k |
| 5f47d68 | Bit-exact speedups: one translation for overlapped-prediction cells whose four nodes agree; the 16-symbol CDF update as fixed-width i16 lanes | decode −6% on static content (x86); CDF update 11× faster than the scalar loop ARM ran (NEON now; checked in aarch64 assembly), also faster than the removed AVX2 intrinsics; streams byte-identical |
| ee4e02a | The benchmark gate itself (`tools/gate.py`, `tools/fetch_gate.sh`, `bench/gate/`) | — |
| 0623fa4 | Fixed qp: a scene cut on a moving camera takes intra offset −3, not −8; P recon offset at the frame's own qp | Luckfox (65 cuts) −1.4% PSNR / −4.7% VMAF medium; nothing else changes; CBR byte-identical |
| e2e268a | **Chroma step 0.45 from 480 lines up** (0.4 below) | every tier and mode on the 10 720p clips: −0.9 to −1.3% PSNR, −3.4 to −4.1% VMAF, no clip worse; Luckfox keeps 0.4 (0.45 cost it 2.9%) |

| c9574b1 | Library B-frame qp offset 9 (the command line already used 9; the gate confirms it against 7 and 11) | CLI streams unchanged |

**Measured together** (`e2e268a.json` against `592f920.json`, all 11 clips, no clip worse):

| | YUV-PSNR 6:1:1 | VMAF |
|---|---|---|
| medium CBR | −2.86% | −5.48% |
| medium fixed qp | −2.67% | −5.33% |
| ultraheavy CBR | −2.39% | −4.94% |
| ultraheavy fixed qp | −2.14% | −4.81% |

## Where VDWT stands now (e2e268a, CBR, mean of 11 clips, PSNR / VMAF BD-rate)

| | vs x265 medium zerolatency | vs SVT-AV1 preset 8 low-delay |
|---|---|---|
| VDWT medium | +10.6 / +8.5% (was +11.8 / +12.5%) | +56 / +66% (was +59 / +73%) |
| VDWT ultraheavy | **−1.6 / −0.5%** (was −0.4 / +3.0%) | +33 / +47% (was +35 / +54%) |

Wins: Luckfox (−28 / −42% against x265), Crosswalk (beats SVT-AV1 on PSNR at ultraheavy),
crowd_run on PSNR. Losses: RollerCoaster (+55%), old_town_cross, Boxing, Aerial at medium.

## Diagnoses

- **RollerCoaster** (smooth sky with fine grain): the gap is spatial. Intra-only, VDWT needs
  3.5× x265's bits for the same luma PSNR, and is about 2 dB behind JPEG 2000 there, while
  on Aerial, Kristen, old_town_cross and crowd_run VDWT intra matches or beats JPEG 2000. No
  coder option (blocks, parents, LL prediction, dead zone, isolated zeroing, levels) moves it;
  transform precision is not the cause (its floor is far above 46 dB).
- **Talking heads at medium** (x265 +32% at fixed qp): VDWT's quality drifts down over the
  clip (luma 43.7 → 39.7 dB in 110 frames) while x265 holds 41.7. The background holds in
  both (46 dB); the loss is in moving faces and hands — the smooth mesh with overlapped
  prediction against HEVC's small blocks. Finer grids (8 px) add only +0.02 dB. This is the
  block-free model's limit, not a setting.
- **Medium vs ultraheavy** (13% apart): the B-frame pairs; heavy codes like medium. They
  cost a frame of latency, so they cannot move down for free.
- **CBR vs fixed qp on 2-second clips**: the opening intra frame is held to 1.3 frame
  budgets by the 1.5-frame delay limit and quality climbs over ~100 frames. Inherent to the
  latency limit (x265 has the same VBV), not a bug.

## Tried and rejected (gate or direct measurement)

| idea | result |
|---|---|
| Motion re-search at the planned qp when the plan lands ≥ 6 qp coarser | ±0.05 dB, noise |
| Re-quantize at a coarser qp before dropping bands on overflow | −0.24 dB (dropping the tiny finest bands is cheaper) |
| Scene-cut margin from the reference's error | no effect: Luckfox's cuts are all certain cuts |
| `--b-qp-offset` 11 (library) vs 9 (CLI) | noise |
| Context-aware motion leaf costs (mode costs from the trained priors) | gate: worse at fixed qp, coaster regresses |
| Retrained context tables (corpus + studio clips) | −0.01 to −0.17%: carried contexts make the initial tables matter only after intra frames |
| P-residual sign context from the prediction's sign | residual sign agrees with it 50%: −0.07% of the stream |
| Temporal significance context (previous frame's nonzero map) | 0.2–2% of significance bits even against a crude baseline |
| Serial rate estimates instead of pool batches | no speedup on x86 |
| Realtime: candidates-only refinement on a static camera | +40% encode time |
| Realtime: keep the loop filter under rate control | −0.2% PSNR, +1.4% VMAF, coaster/crosswalk regress |
| Wider P dead zone on coarse levels and LL (θ 0.25) | +2.5 to +5% PSNR (VMAF −1.4 to −2.7%) |
| Planner θ ramp at the planned qp | +0.15% ultraheavy, two regressions |
| Defaults sweep (medium CBR): inter θ 0.10/0.16/0.20, recon δ 0.05/0.15/0.20, isolated 1.0/2.0, motion λ 0.8/1.25, intra θ 0.20/0.30, sharp filter, intra offset ±2 | all current defaults at or near the optimum |
| Mesh 8 on talking heads | +0.02–0.03 dB |
| CBR: moving-camera cut keyframes at a normal frame's budget (share 1.0, not 1.3) | Luckfox +0.6% medium, −1.0% ultraheavy: noise |
| Chroma 0.5 / 0.55 at 720p | VMAF −6.5 / −8.7% but crowd_run +1.2 / +2.3% and Luckfox +6 / +9% on PSNR |
| Inter θ 0.10, inter δ 0.05 | −0.3 / −0.4% mean, ducks and Luckfox regress |
| Ultraheavy: B qp offset 7 / 11, static B offset 4 / 8, motion λ 0.7 / 0.9 | all worse or with regressions (static 8: −0.6% at fixed qp, FourPeople VMAF +1.1% under CBR) |

## Next, by expected value

1. **In-loop CNN filter** (offline −8 to −11%): int8 quantization-aware training for
   bit-exactness, then decoder integration; on the Pi a 12×4 model with SDOT kernels or the NPU.
   The only single lever of the size of the SVT-AV1 gap.
2. **The moving-region model** (talking heads, Boxing): the smooth mesh drifts on deforming
   faces. A research item within the block-free rule (e.g. finer motion only where the
   prediction error is high, or a second overlapped hypothesis per node).
3. **Grain on smooth content** (RollerCoaster): VDWT intra is 2 dB behind JPEG 2000 only there;
   worth a closer look at how the quad coder handles dense low-amplitude finest bands.
4. Smaller: fold the motion section into the joint stream (about 0.5% at 80 kbit/s); a
   chroma-only loop filter in realtime mode; Pi timing of tonight's NEON CDF update and the
   equal-vector OBMC path.
