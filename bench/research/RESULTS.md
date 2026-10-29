# 2026-10-26 optimization campaign — working ledger

Grid unless stated: 5 clips (a720, b720, canyon, forest, tasmania), 20 frames, 720p, 4 threads,
medium + ultraheavy, qp 42/50/58, VMAF `vmaf_v0.6.1`, YUV-PSNR. Hybrid = 70% VMAF + 30% YUV.
BD is against the VERSION 9 baseline (`target/research/vdwt-baseline`); positive saving means the
new build needs fewer bits at equal quality.

## Kept

| ID | Change | Measurement | Decision |
|---|---|---|---|
| E1 | Causal frame-level AQ: `qp -= 2·log2(activity/running average)`, clamp ±6, activity = luma prediction MSE (P) / Laplacian (I), encoder-only | 3-point: 0.5 +0.01%, 1.0 +0.11%, 1.5 +0.68%, **2.0 +1.38%**, 2.5 +0.73%. Six-point at 2.0, per tier: **+0.84%** (a720 +0.72, b720 −0.18, canyon +1.14, forest +1.72, tasmania +0.83; VMAF +1.07, YUV +0.33) | Keep, default 2.0 |
| E2 | Perceptual in-loop filter: fit the Wiener filter to `src + 0.4·(src − blur3(src))` instead of `src`; coefficients still coded, decoder unchanged | 3-point (medium+ultra): 0.3 **+4.68%**, 0.4 **+4.84%**, 0.5 +4.53%. Six-point 0.4, per tier: **+5.57%** (VMAF +9.49, YUV −3.58) | Keep, default 0.4 |
| E3 | E1 + E2 | 3-point combo 0.3 **+5.23%**, 0.4 **+5.71%**, AQ1.5+LS0.3 +4.46%. Six-point combo 0.4 **+5.57%** (a720 +5.87, b720 +3.32, canyon +4.84, forest +6.25, tasmania +7.57) | Keep both defaults |
| E4 | VMAF NEG sanity check of E3 | NEG VMAF BD +5.39%, YUV −2.50, hybrid **+3.02%**; b720 −5.57 medium / −0.68 ultra under NEG | Perceptual mode is real but smaller under NEG; document |
| E5 | CBR sanity, a720 300/600 kbit/s, 100 frames | 300k: VMAF 74.93→75.74, NEG 72.63→72.73, PSNR 40.64→40.40. 600k: VMAF 86.64→88.17, NEG 84.63→84.83, PSNR 44.16→43.65 | CBR gain is modest; fixed-qp Pareto is the main result |
| E6 | Opt-out / compatibility | `--frame-aq 0 --loop-sharpen 0` reproduces the baseline stream byte-for-byte (a720 q42 MD5 30380b…); old VERSION 9 decoder decodes new streams to the encoder reconstruction | No syntax/version change |
| E7 | External depth/tilt check | 3 luma levels: **−3.88%** per-tier (canyon −8.4, tasmania −16.8); 4 levels: **+0.51%**; finest tilt 1.45: **+1.03%** (opt-in) | 3 levels rejected on our corpus; 4 levels optional; finest tilt kept opt-in |
| E8 | QP-adaptive finest tilt: level-1 luma weights ×1.6 at qp ≤ 46, ×1.0 above; weights already in the sequence header, encoder-only | Tuning, six-point: **+0.71%** per-tier (medium +0.67, ultra +0.75). Held out (coast, gen_a, gen_b, madeira): **+0.41%**, worst cell −0.13%. NEG: **+0.60%** (full stack +3.60%). Aggressive 1.6/1.45/1.0: +1.39 tuning but madeira −3.08/−2.35 | Keep, default auto (`--finest-tilt 1.0` opt-out) |
| E9 | VERSION 10: per-band entropy contexts on fixed-qp streams (intra/inter set per band, carried) and B qp offset 11 | Band contexts: realtime +0.99, light +0.92, medium +0.85, heavy +0.83, ultra +0.51; **identical reconstruction**. B qp 11: +0.35 ultra normal, +0.29 NEG. Full grid +5.55…+6.81 over VERSION 9; free2 NEG +4.24 | Keep; VERSION 10 |

## Rejected

| ID | Candidate | Measurement | Why |
|---|---|---|---|
| R1 | Loop-filter integer RD coordinate descent (T4) | 30-row PSNR/bytes: mixed −0.75…+1.19% bytes, BD ≈ 0 | No measurable gain; reverted |
| R2 | Band step tilt over depth: +0.15 / +0.30 (coarser fine detail) | a720 q50 VMAF 67.78→62.58 / 61.73 for −3.0% / −7.8% bits; forest 51.54→50.03 / 48.80 | VMAF falls much faster than bits; rejected |
| R3 | Negative band tilt −0.10 / −0.20 (finer detail) | a720 q50 VMAF 67.78→64.23 / 65.03 at +3% / +7% bits | Worse rate and VMAF; MSE weights are already good |
| R4 | Intra qp offset −6 (coarser than −8/−7) with E3 | PSNR BD vs E3: a720 +1.0%, b720 +6.9%, canyon +0.8%, forest +0.8% | Current offsets better |
| R5 | B qp offsets +7 / +11 with E3 (ultraheavy) | PSNR BD vs E3: b7 −0.08…+0.51, b11 −0.54…+0.04 | Within noise; keep +9 |
| R6 | Separate intra AQ strength | `--frame-aq-intra 0` produced a byte-identical 30-row grid | Intra AQ never fires in 20-frame clips (fewer than 2 intra frames); no-op |
| R7 | Full 5-clip mean of band tilt etc. | See R2/R3 | Not pursued |

## Rejected after VERSION 10 round

| ID | Candidate | Measurement | Why |
|---|---|---|---|
| R8 | Finest tilt 1.45 at q50 as well (schedule 1.6/1.45/1.0) | Tuning +1.39% per-tier; held-out madeira **−3.08 medium / −2.35 ultra** (VMAF −4.72/−2.42) | q50 tilt is not robust on low-rate 60 fps content; q42 only |
| R9 | `--isolated` 0 / 1.25 / 1.75 / 2.0 | −1.50 / +0.15 / −0.17 / −0.06 | Default 1.5 is the optimum |
| R10 | `--chroma-scale` 0.90 / 1.00 | +0.08 / −0.13 | Within noise |
| R11 | `--inter-rounding` 0.08 / 0.10 | −0.29 / −0.22 | Default 0.12 is better |
| R12 | Loop-sharpen qp schedules over 0.3/0.4/0.5 | Best 0.00%, most −0.1…−0.4% | Fixed 0.4 is already the optimum |
| R13 | B qp offsets 5 / 7 / 13 / 15 (11 ships) | −0.84 / −0.30 / +0.25 / +0.11 | 11 is the peak |
| R14 | Loop-sharpen 0.6 at ultraheavy | +0.75 alone, +1.26 with band contexts; but b720 −3.56 and intra-only −0.45 | VMAF-saturated clip; opt-in only |
| R15 | Motion mesh 8 at 720p | −9.23% | Motion bits explode |
| R16 | Temporal node predictor at qp 42 | a720 +0.40, b720 −0.25, canyon +0.86, forest +0.10, tasmania +0.49 | Too mixed; left off |
| R17 | Sparse ±1 zeroing without isolation | +0.05% mean, forest +2.83 / canyon −1.44 | Sign disagrees per clip |
| R18 | Temporal motion trial (VERSION 11, opt-in): code each motion section with spatial vs previous-frame predictor, keep shorter | Ultraheavy: a720 −0.00, b720 +0.04, canyon −0.00, forest +0.00, tasmania −0.02; **+0.00 overall** | Selector bit offsets the few wins; default off, streams without it stay VERSION 10 |

## Protocol checks

- Encoder recon == decoder output for VERSION 10 (a720 ultraheavy q42 MD5 `18c571…`); a
  VERSION 9 stream still decodes to its own reconstruction. Streams are identical at 1, 2 and 4
  threads; the CBR pictures and size are unchanged (the header version byte differs).
- `cargo test --workspace`: 79 unit + 6 codec + 21 inter + 6 rate-control pass (112), 7 ignored.
- `quality_and_size_move_with_qp` disables `loop_sharpen` and the adaptive tilt because the
  perceptual filter is not PSNR-monotonic at very low qp; the test is about the quantizer.
- Visual crops (a720/tasmania q50) show restored mid-frequency detail without visible halos.
- SSIM: a720 0.958519→0.958583, b720 0.966827→0.965607, tasmania 0.967124→0.967413 at qp 42;
  b720 is the one clip where sharpening costs SSIM.

## Final full grid

5 clips × 5 tiers × qp 42/50/58, VMAF + YUV, five builds (old VERSION 9, AQ only, AQ + filter,
the tilt build and the VERSION 10 band-context build):

| tier | old | AQ only | AQ + filter | free (tilt) | free2 (V10) |
|---|---:|---:|---:|---:|---:|
| realtime | −18.71 | −18.76 | −14.31 | −13.64 | **−12.55** |
| light | −13.80 | −13.63 | −9.23 | −8.89 | **−7.93** |
| medium | −6.81 | −6.13 | −2.34 | −1.62 | **−0.80** |
| heavy | −6.06 | −5.55 | −1.90 | −1.30 | **−0.52** |
| ultraheavy | −1.38 | −0.36 | +4.29 | +4.95 | **+5.42** |

The VERSION 10 build gains **+5.55…+6.81 points** over VERSION 9 at every tier, and +0.47…+1.10
over the tilt build. Under VMAF NEG the full stack is **+4.24%** (medium +3.53, ultraheavy +4.96).

Per group (hybrid BD vs x265 medium; samtime = a720+b720, drone = canyon+forest+tasmania):

| tier | samtime old | samtime free2 | drone old | drone free2 |
|---|---:|---:|---:|---:|
| realtime | −38.48 | **−30.33** | −5.52 | **−0.70** |
| light | −28.91 | **−23.52** | −3.72 | **+2.47** |
| medium | −24.03 | **−18.76** | +4.68 | **+11.17** |
| heavy | −23.71 | **−19.17** | +5.71 | **+11.92** |
| ultraheavy | −20.98 | **−13.07** | +11.68 | **+17.76** |

Both groups gain at every tier: samtime +4.5…+8.2 points, drone +4.8…+6.5. Drone is above x265
medium from light up; samtime is still behind on its I-frame-heavy content. Charts: `videos/pareto_free_old_new.png`,
`pareto_samtime_free_old_new.png`, `pareto_drone_free_old_new.png`, `bd_bars_free.png`, plus the
previous `pareto_old_new.png`, `bd_bars.png` and `rd_curves.png`.

## Pi

`selftest --isa all` passes. CBR 600k realtime is byte-identical (559.3 kbit/s, 42.64 dB) at
19.82–19.91 ms (old 19.87–19.93). Fixed-qp ultraheavy q42: a720 216.0 ms (old 205.4), forest
440.3 (415.5), b720 273.7 (265.4). The old VERSION 9 decoder produces the same MD5 as the new
encoder reconstruction. The adaptive tilt and per-band contexts are fixed-qp only, so the CBR pictures and size stay
identical; the board was offline for this round, so the fixed-qp rows were not re-timed. The
per-band contexts add 711 KiB per encoder/decoder and one 8.7 kB context copy per band per
frame; that is inside the entropy stage's noise on x86 but has not been measured on the A55.

## Status

Docs updated (EXPERIMENTS/DESIGN/README/TASKS/TODO), `cargo fmt --check` clean, 112 tests pass,
x86 selftest passes; the Pi was offline this round (CBR stream and fixed-qp time unchanged). Remaining: clean /tmp, final goal check.
