# VDWT: low-latency wavelet video codec

Status: v0.31 (2026-10-26; VERSION 10). The main encoder/decoder and the separate `rt` codec are implemented; the main codec has five tiers (`realtime`, `light`, `medium`, `heavy`, `ultraheavy`). 720p60 realtime on the Orange Pi 3B is 19.9 ms/frame (50 fps); `--no-guide` reaches 17.55 ms (57 fps). Ultraheavy is 208.5 ms/frame at 37.60 dB. Current numbers and the Pi's thermal headroom are in §10; older rounds are in the internal lab notes.

Target use: **drone video downlinks** — low bitrate, minimal latency and edge ARM CPUs — with an optional general-purpose quality tier.

Contents: [1 Goals](#1-goals-and-non-goals) · [2 CPU budget](#2-cpu-budget) · [3 Architecture](#3-architecture-overview) · [4 Transform](#4-transform-dwt-over-the-whole-image) · [5 Quantization and entropy](#5-quantization-and-entropy-coding) · [6 Inter prediction](#6-inter-prediction-global-motion--mesh-implemented-motion) · [7 Parallelism](#7-parallelism-4-cores-and-latency) · [8 Bitstream](#8-bitstream-and-packetization-implemented-bitstreamrs) · [9 Implementation](#9-rust-implementation) · [10 Status](#10-status-and-measurements) · [11 What is new](#11-what-is-new-in-vdwt) · [12 Open questions](#12-open-questions)

## 1. Goals and non-goals

**Goals**
- From-scratch encoder **and** decoder in Rust.
- **I and P frames by default; bounded B groups only in ultraheavy.**
  - The default pipeline has no B-frames, reordering, lookahead or frame buffering; a frame goes out as soon as it is encoded.
  - `ultraheavy` may code a pair as a B-frame group (one B between two anchors); a pair buffers two frames (the four-frame format allows three), and the delay is bounded. The other tiers and the default stay I/P only.
- **Whole-image transform.** The DWT runs over the full frame (no tiles), using the **JPEG 2000 / HTJ2K wavelet**:
  - the reversible CDF 5/3 integer lifting from ISO/IEC 15444-1 Annex F, bit-exact;
  - a fixed-point 9/7 for lossy coding (§4.4).
- **Block-free by design.** No visible block structure anywhere in the pipeline:
  - motion is a smooth global model plus a smooth mesh field;
  - transform and prediction both cover the whole image;
  - so there is no deblocking filter and no blocking artifacts. At low bitrate the codec degrades to **blur, not blocks**.
- **Entropy coder that keeps the HTJ2K (Part 15) HT quad structure but uses an adaptive range coder (§5.2).** It works on 2×2 quads, codes zero runs MEL-style, and does it in a single pass.
- **Real time: 720p60 4:2:0 8-bit on 4× Cortex-A55 with NEON**, using at most about 50% of the CPU.
- **Low-bitrate operating point:** roughly **1–6 Mbps at 720p60**, to be confirmed.
- **Radio-link friendly:**
  - rate control that follows link capacity frame by frame;
  - packet loss that degrades gracefully;
  - intra refresh instead of large I-frames.
- **Integer-only pipeline.** Encoder reconstruction and decoder output are **bit-exact** on all platforms.

**Non-goals (for now)**
- HTJ2K codestream compatibility.
- Matching HEVC or AV1 efficiency. The target is the best rate-distortion **at a fixed, tiny CPU cost and latency**.

## 2. CPU budget

| Quantity | Value |
|---|---|
| Samples per frame (1280×720, 4:2:0) | 1,382,400 |
| Samples per second at 60 fps | 82.9 M |
| 4× A55 @ 1.8 GHz | 7.2 G cycles/s |
| Cycles per sample at 100% of all cores | **~87** |
| Encoder target (≤50% CPU) | **~40–45 cycles/sample** |
| Wall-clock per frame | 16.7 ms; encode target ≤ 8 ms |

Cortex-A55 constraints:
- It is **in-order and dual issue**, so kernels need manual unrolling and interleaved independent chains.
- The **NEON datapath is narrow**, which favours **int16 lanes** (8 per vector).
- It supports ARMv8.2 dot product (UDOT) and FP16.
- **Caches are small and DRAM bandwidth is a few GB/s.** One full-frame i16 pass costs about 110 MB/s per luma plane, so **count the passes**.
- Typical SoCs: Rockchip RK3566/RK3568, Amlogic S905X3/X4.

First-cut encoder budget per sample (P-frame, averaged over Y and C). Microbenchmarks will validate it:

| Stage | cycles/sample |
|---|---|
| Input load and ¼-res pyramid | 1 |
| Global motion estimation and fit (on ¼ res) | 1 |
| Mesh node refinement | 2 |
| Mesh MC warp (bilinear, Y and C) | 4 |
| Residual | 0.5 |
| Forward DWT (5/3; 9/7 roughly 2×) | 3–4 (6–8) |
| Quantization and rate statistics | 2 |
| Block coding (scales with bitrate, light at low rates) | 6–12 |
| Dequant, inverse DWT, reconstruction | 4–6 |
| Threading, rate control, bitstream | 2–3 |
| **Total** | **~27–40** |

This was the initial budget, and it is kept as design rationale rather than a current plan.
The A55 measurements in §10.2 superseded it: the pool, the real-time search and the NEON
kernels are in place, and the remaining gap is quantified per stage there.

## 3. Architecture overview

```
            ┌──────────────────── encoder ───────────────────────────────────┐
  YUV in →  │ pyramid ─► global motion (affine) ─► mesh refinement (nodes)    │
            │                                            │                   │
            │                         smooth per-pixel motion field          │
            │                                            ▼                   │
            │ ref_recon ───────────────────────► mesh warp = prediction      │
            │                                            │                   │
            │ cur ─────────────────────────(−)◄──────────┘                   │
            │                               │                                │
            │              residual ─► whole-image DWT ─► Q ─► VEC coder ─► bitstream
            │                                                       │        │
            │ ref_recon ◄─ (+) pred ◄─ whole-image inverse DWT ◄─ dequant ◄──┘
            └────────────────────────────────────────────────────────────────┘
```

**Hybrid closed loop.**
- Prediction is a **smooth warp** of the previous reconstructed frame.
- The residual is coded with a **whole-image wavelet transform**.
- The encoder contains the decoder's reconstruction loop, so there is no drift.

**I-frame.** The residual is the source minus 128, which makes the I-frame a whole-image HTJ2K-style intra picture.

**Why block-free motion.** A whole-image DWT represents smooth signals very efficiently. Block motion compensation would inject sharp edges along a 16×16 grid into the residual, and the wavelet would spend bits on those edges in every subband. Drone motion is dominated by camera motion, which a few global parameters plus a smooth correction field describe well:
- pan and translation;
- yaw rotation;
- zoom from altitude changes;
- perspective;
- rolling-shutter wobble.

A **continuous motion field** gives a residual with no artificial edges, which is the ideal input for a whole-image DWT.

## 4. Transform: DWT over the whole image

Lossy streams use the 9/7 of §4.4 (sequence header `kernel` 1); lossless streams, and any stream with `--wavelet 53`, use the HTJ2K 5/3 below (`kernel` 0). `dwt::Transform` runs either.

### 4.1 5/3 reversible lifting (lossless, and `--wavelet 53`)
```
predict: d[n] = x[2n+1] − ⌊(x[2n] + x[2n+2]) / 2⌋
update : s[n] = x[2n]   + ⌊(d[n−1] + d[n] + 2) / 4⌋
```
This uses whole-sample symmetric extension at the frame borders.

NEON int16 mapping, with no widening and no overflow in the sums:
- `⌊(a+b)/2⌋` is a single `SHADD`.
- `⌊(a+b+2)/4⌋` equals `SRSHR(SHADD(a,b),1)`. This identity has been verified exhaustively.
- LLVM fuses the update's shift and add into one `SRSRA`. Checked in the generated A55 assembly: the vertical update is `SHADD` + `SRSRA`, **2 instructions per 8 samples**.
- The horizontal pass uses `LD2`/`ST2`, which deinterleave for free.
- The vertical pass is ordinary row-vector arithmetic, streamed line by line.

On x86 (development), `⌊(a+b)/2⌋` is computed as `(a & b) + ((a ^ b) >> 1)` because AVX2 has no signed halving add.

**Implementation** (`crates/vdwt/src/dwt/`):
- **Order:** each level transforms vertically, then horizontally, the Annex F order, with Mallat-layout coefficients.
- **Streaming:** one pass per level goes down the region two rows at a time.
  - It runs the vertical predict/update in place, then sends each finished low and high row through the horizontal transform straight into its subbands.
  - A level touches each sample about once while it is still in cache.
  - The same row-pair schedule is the basis for a stripe (sub-frame) mode later.
- **Buffers:** low bands ping-pong between the input plane and one quarter-size scratch plane. The engine needs no other buffers and allocates nothing per frame.
- **Kernels:** six row kernels per ISA (scalar, AVX2, NEON) sit behind one table: vertical predict, update and their inverses, plus horizontal forward and inverse.
- **Verification:**
  - `dwt/oracle.rs` is a literal, independent transcription of Annex F (PSE extension, `1D_SD`/`1D_SR`).
  - `dwt::verify` checks every kernel against scalar on all short lengths. It then checks the full transform against the oracle for random sizes, 0–6 levels, and pixel, residual and full-`i16` (wrapping) value ranges.
  - `vdwt selftest` runs the same checks on the target CPU.

### 4.2 int16 safety
The 5/3 analysis L1 gains are 1.5 for lowpass and 2 for highpass. Worst case after 5 levels on a residual in [−255, 255]:
- LL5 ≈ 14.7 k
- HH5 ≈ 26.1 k

Both are below 32 767, so **8-bit input is provably safe in i16.** 6 levels or 10-bit input need re-analysis, i32, or normative wrapping arithmetic.

### 4.3 Decomposition
- Luma: **5 levels** (LL5 is about 40×23 at 720p). Chroma: 4 levels. Six levels are not used.
- Four levels were tried in the 2026-10-24 round (§10.5): at fixed qp 24–48 they saved 0.3–1.1% of the bits on several clips, but the direct comparison at the low-rate end of the drone range (qp 50–58) showed **2–4% more bits and 0.1–0.35 dB lower PSNR**; with five levels the same build reproduces the old streams exactly there. Five stays, and `--levels` still selects any depth.
- The 2026-10-26 recheck with the perceptual defaults scores **per tier**: four levels is **+0.51%**
  hybrid (small, content-dependent; tasmania −1.0), three levels is **−3.88%** (canyon −8.4,
  tasmania −16.8; b720/forest positive). Four levels remains a realtime speed option (`--levels 4`
  ties PSNR at CBR for 7–14% less x86 encode time, VMAF −0.1…−0.2); three is not a default.
- Arbitrary frame sizes with J2K-style even/odd phase handling.
- **Full-frame transform.** Parallelism comes from splitting rows and columns, not tiles (§7).
- **Subband dropping.** At very low rates, rate control may zero the finest HH (or whole level-1) bands outright. That saves bits for nearly free, and the result is controlled softness instead of artifacts.

### 4.4 9/7 in exact integer lifting (implemented: `dwt/d97.rs`, normative)
The four lifting steps of the CDF 9/7 (Annex F, irreversible), with Q14 constants and one rounding per step:
```
d[n] += f(α, s[n],   s[n+1])        f(c, a, b) = ⌊(c·(a + b) + 2^13) / 2^14⌋, wrapped to i16
s[n] += f(β, d[n−1], d[n])          α, β, γ, δ = −25987, −868, 14466, 7266 (/2^14)
d[n] += f(γ, s[n],   s[n+1])
s[n] += f(δ, d[n−1], d[n])
```
- **Exactly invertible:** each step adds a function of the other channel only, so the inverse subtracts the same values in reverse order. Encoder and decoder agree bit for bit on every platform; no floats are involved. The final scaling by K is left out (it is not invertible in integers); the step weights absorb it.
- **Fraction bits:** samples enter scaled by 2^3 and leave as `(v + 4) >> 3`. This is where most of the gain comes from. Integer lifting rounds at every step; on small values, P-frame residuals above all, that rounding is noise on the scale of a quantizer step. Measured on the three real clips at `effort 1` (BD-rate, luma):

  | | beach | coast | canyon |
  |---|---|---|---|
  | 5/3 with 2 fraction bits | −14.9% | −13.7% | −17.7% |
  | 9/7, 0 fraction bits, intra frames only (intra-only coding) | +6.3% | +5.6% | −4.8% |
  | 9/7, 1 / 2 / 3 / 4 fraction bits | −8.6 / −17.3 / −21.9 / −21.7% | −7.5 / −15.8 / −21.4 / −21.9% | −13.1 / −22.8 / −28.3 / −28.7% |

  The 9/7 itself adds 2–5% over the 5/3 at equal precision. 3 bits is the choice: a fourth gains nothing, and 3 keeps 8-bit input inside i16 (LL5 ≤ about 8 100; the steps of the deepest levels stay below 2^15).
- **Steps:** coefficients carry the 2^3 scale, so a 9/7 stream's steps are `Δ_b · 2^3` (`quant::step_q12`'s `frac_bits`); the header weights stay in sample units, `1/√G_b` from the 9/7's synthesis gains.
- **Schedule:** per level, the region's rows are reordered into low and high halves, lifted vertically a whole row at a time (each step writes one half and reads the other, so its rows run in parallel), then each row is split and lifted horizontally. One element-wise kernel does every step: AVX2 forms `c·a + c·b` exactly with `madd` and wraps to 16 bits. Rows are spread over the worker pool, components one after another.
- **Speed:** 720p luma forward 1.2 ms and inverse 1.4 ms on one x86 core (0.7 ms each on 4 threads). There is no sparse inverse: the 9/7 always synthesizes every block (§6.5 applies to the 5/3 only).
- **Verification:** `dwt::d97::verify` checks the ISA kernels against the portable ones on the pool and the exact round trip, in `cargo test` and in `vdwt selftest`.
- **NEON (§10.2):** lifting, split and merge kernels; each step multiplies into 32 bits and narrows with a rounding shift, bit-exact with the portable code.
- **Open:** a streaming schedule that fuses the four vertical steps into one pass.

## 5. Quantization and entropy coding

### 5.1 Quantization (implemented: `quant.rs`)
- **Dead-zone uniform scalar quantizer** per subband: `q = sign(c)·⌊|c|/Δ_b + θ⌋`.
  - The encoder chooses θ (default 0.25); it is not part of the format.
  - Reconstruction is `(|q| + δ)·Δ_b`, rounded and saturated to `i16`. δ (default 0.2) is sent in each frame header.
- **Steps:** `Δ_b = 2^(qp/8) · w_b`, in Q12 fixed point, clamped to [1, 32768].
  - The weights `w_b` (Q12) travel in the sequence header, so the encoder may use any weighting without a decoder change.
  - The default is MSE-optimal: `w_b = 1/√G_b`, where `G_b` is the energy of the wavelet's synthesis basis, computed numerically. A 9/7 stream's steps also carry its coefficients' 2^3 scale (§4.4).
  - The cap at 32768 keeps the reconstruction in branch-free 32-bit arithmetic.
  - **Finest-level tilt (`--finest-tilt`; auto under `--tune perceptual`, else 1.0).** The weights are MSE-derived. The
    auto policy multiplies only level 1's luma weights by 1.6 at qp ≤ 46 and leaves them at 1.0
    above. At qp 42 the finest band is dense enough that coarsening it saves several percent of
    the bits for a fraction of a VMAF point; at qp 50 the same tilt removed detail that low-rate
    content needs (a held-out 60 fps aerial lost 2–3% on the BD curve), so q50 and above keep the
    MSE weights. Measured **+0.71%** per-tier hybrid on the five tuning clips (medium +0.67,
    ultraheavy +0.75) and **+0.41%** on four held-out clips, worst cell −0.13% (noise). Under VMAF NEG
    it is **+0.60%** (VMAF +0.92, YUV −0.14) and the full free stack is **+3.60%** NEG.
    On PSNR-Y it costs 0.2–1% (2026-10-28), so the default is 1.0 and `auto` is part of
    `--tune perceptual`; any value in
    0.5–2.0 is free-form. The weights travel in the sequence header, so no decoder change. Rate
    control keeps 1.0, so every CBR stream is byte-identical.
- **Lossless mode:** qp 0, unit weights, δ = 0. Verified exact.
- **Intra frames followed by P-frames** use qp − 4 at a fixed qp (`intra_qp_offset`), a step ratio of √2 like x264's default I/P ratio. It gains about 0.1 dB where the scene persists (+0.25 dB at −8) and changes nothing on fast FPV footage.
- **P-frames** use θ = 0.12 and δ = 0.1 by default (0.04 in the first rounds; 0.12 saves 1.0–1.7% on real footage).
  - Residuals are mostly small, noise-like values. A much wider dead zone than for intra frames pays off: together the two settings gain about 0.5 dB BD-PSNR on the synthetic drone clips over the intra values.
  - θ is the encoder's choice, and δ travels in every frame header, so no decoder change was needed.
- **Causal frame activity adaptation (encoder-only, default 2.0).** At a fixed qp, a P-frame's qp
  moves by `−2·log2(activity / running average)`, clamped to ±6, where activity is the luma
  prediction MSE sampled every other pixel. Hard frames get a finer qp and easy frames a coarser
  one; the average move is near zero, so the operating point is preserved. Intra frames use a
  spatial Laplacian activity and the same rule. The decoder needs no change: the resulting qp is
  already in every frame header. Six-point hybrid BD against the previous build: **+0.84%**
  (a720 +0.67, b720 −0.17, canyon +0.99, forest +1.97, tasmania +1.24). ``--frame-aq 0``
  restores fixed qp exactly.
- **Isolated values (encoder only):** a ±1 detail value alone in its 4×4 block, with the 8 blocks around it empty (an earlier round: or the other three blocks of its 8×8 quadtree node empty, with zeroed blocks counting as empty for later ones: a further 0.3–0.5% fewer bits), is zeroed when its coefficient is below 1.5 steps (`isolated_threshold`). Locating it costs a dozen bits or more of block and node flags, more than the ~1·Δ² of distortion it removes is worth at the λ ≈ 0.12·Δ² that motion decisions use. Gain: +0.03 to +0.07 dB BD-PSNR on all six evaluation clips (1–2.6% fewer bits); 1.1 gains about two thirds of that, and the gain is flat from 1.5 to 2.2.
- **Rejected: RD zeroing of whole 4×4 blocks** (drop a block when the error its values remove is worth less than λ times their estimated bits, from the rate model's location cost and magnitude bits): at most +0.02 dB at λ = 0.12·Δ², worse above. With the dead zone and the isolated-value rule, residual quantization is already near its RD optimum; x264's advantage lies elsewhere (HISTORY.md).
- **Rejected: rate-distortion optimized quantization.** An encoder pass with the band coder's live contexts rewrote each 2×2 quad to the lowest `D + λ·R` among the choices that lower magnitudes by one, with exact costs for ρ, U, the bit lengths and the raw bits. At the best λ (0.15–0.2 squared steps per bit) it saved 0–0.5%; above that it lost (+5–7% at λ = 0.4), and starting it from round-to-nearest lost 7–24%. Lower-magnitude choices by level (a separate θ above 1) and zeroing 4×4 blocks of two or three ±1 values both lost too. The dead zone already makes nearly every choice a greedy RD pass can see; what is left is the effect of sparser values on later contexts, which a greedy pass cannot weigh.
- **Tuning results** on the two test clips (HISTORY.md):
  - θ from 0.1 to 0.25 is flat (within ±0.06 dB BD-PSNR); θ ≥ 0.35 loses.
  - **Chroma steps:** `chroma_step_scale` defaults to **0.4** since 2026-10-28, when YUV-PSNR 6:1:1 became the main metric. On nine clips (drone, samtime, talking head) every tier needs 10–11% fewer bits at equal 6:1:1 than at 0.95 (0.6: −6.5%, 0.5: −8.6%, 0.4: −10.0%, 0.3: −9.5% at medium; 5 chroma levels add nothing on top). The earlier 0.95 (0.85, 0.7 before) was tuned on PSNR-Y and the VMAF hybrid, which do not see what coarse chroma costs. At a fixed link (1.2 Mbit/s) luma falls 0.3–0.7 dB and chroma rises 2.3–3.4 dB, 6:1:1 +0.1 to +0.6 dB.

### 5.2 Entropy coder "VEC" (VDWT Entropy Coder) (implemented: `entropy/`)
**Decision: an adaptive multi-symbol range coder with wavelet-specific contexts over a significance quadtree, keeping HT's 2×2 quads and quad exponents.**

**Why not the alternatives:**

| Option | Verdict |
|---|---|
| EBCOT/MQ (JPEG 2000 Part 1) | Bit-plane passes and one binary decision per bit. Too slow. |
| HT cleanup pass (HTJ2K) | Very fast, but static VLC tables cost about 5–10% (or more) versus EBCOT, and that loss hurts most at low bitrate. Still worth measuring as a speed baseline. |
| Static rANS | Frequency tables cost header bits every frame. Adaptive rANS is awkward (the encoder runs backward) and has patent noise around it. |
| **Adaptive multi-symbol range coder** (AV1-style CDF adaptation, 15-bit probabilities, alphabets ≤ 16) | **Chosen.** Efficient, no table overhead, old and royalty-free. |

**Why it is affordable:** the cost scales with the **number of coded symbols, not the number of pixels.** Measured on x86, coding takes 0.8–1.8 ms per 720p frame at 1.5–4 Mbps (HISTORY.md).

**Range coder** (`range.rs`):
- 32-bit range, byte output with carry propagation, no dummy first byte.
- **Raw bits** (the lower mantissa bits) are coded as equiprobable batches of up to 16 bits inside the same stream, so each band is one segment with one length.
- **Termination** picks the value with the most trailing zero bytes and drops trailing zeros; the decoder reads zeros past the end. Cost: ≤ 1 byte plus pending carry bytes.

**CDF adaptation** (`cdf.rs`):
- The AV1 rate schedule: fast for the first 16 and 32 symbols, then slower.
- Updates pull toward a target that keeps `MIN_PROB = 4/32768` for every other symbol. This **provably** keeps every probability ≥ MIN_PROB despite rounding, so no symbol ever becomes uncodable. A test confirms it.

**Coding hierarchy per band** (`band.rs`):
1. A **nonzero flag per 64×64 block** (`block_log2`, default 6; 16–128 measured within ±0.3%).
2. A **quadtree** below each nonzero block, with one flag per child, down to 4×4 nodes.
   - When the earlier children of a node are all zero, the last child's flag is **inferred**.
   - The last quad of such a node uses a ρ table that excludes 0.
3. **Per 2×2 quad:** the significance pattern ρ as one 16-ary symbol.
4. **Per significant quad:** U − 1, where U is the bit length of the largest magnitude (HT's quad exponent).
5. **Per significant sample when U > 1:** U − e, its own bit length e. For the last sample it is inferred when no earlier sample reached U.
6. **Mantissa:** below each magnitude's leading 1, the first bit in a context of the bit length (2, 3, 4, 5 or more; it is 0 about 70% of the time), the other e − 2 as raw bits.
7. **Sign:** in a context of two neighbours' signs (none, +, −): the one along the edge and the one across it. Along an edge neighbouring coefficients share a sign, across it they alternate. In LH bands (horizontal edges) the left neighbour is along and the upper one across; HL bands swap them, so the two orientations, which share a class, see the same statistics. P-frame classes start from their own weights: in residuals only the neighbour along the edge tells anything.

The traversal is Z-order within blocks, and blocks go in raster order. That way the left, above and above-left neighbours of every quad are always coded first.

**Contexts** (60 CDFs per class, plus 9 sign and 4 mantissa CDFs):
- **Spatial:** nonzero count and largest bit length among the five coded neighbour samples.
- **Cross-scale (parent):** a 2×2 quad at level k maps to **one parent coefficient** in the same orientation at level k+1. Nodes use the parent region's significance (from a per-band significance pyramid).
- **Classes:**
  - 14 for intra frames, from luma/chroma × (LL, level 1, level 2, deeper) × (HL/LH, HH);
  - the same 14 again for P-frame residuals, whose LL bands start from sparse priors instead of dense ones;
  - three more context sets for the motion section (§6.2).
- **Measured effect:** parent contexts save **6.1%** on testsrc2 and are neutral (+0.3%) on terrain. Sign and mantissa contexts save 6.4% of intra luma bits at qp 28 on 1080p drone frames (signs 5.4%), 1.4% of the stream at 720p60 1.2 Mbps and 3.0% at 1080p60 4 Mbps; decoding time is unchanged. They bring VDWT intra level with OpenJPEG (9/7, five levels) at equal luma bytes per frame (−0.3 to +0.14 dB, mean 0.0), from 0.2–0.8 dB behind. Richer sign neighbourhoods (above-left, above-right, the parent's sign, the second neighbours) did worse in a static estimate: their counts spread thinner than the correlation they add.

**Prediction guide (P-frames).** The decoder has the motion-compensated prediction before it decodes the residual, and residual values gather where the prediction has texture: motion errors show at edges. So the encoder and the decoder both transform the prediction (one forward DWT per component), and every 2×2 quad, node and block of a detail band gets a **guide class** from the largest prediction coefficient it covers against the band's step: below Δ/4, below Δ, below 4Δ, above. The class selects one of four copies of the block-flag, node-flag, ρ and exponent contexts; the magnitude contexts are shared. Trained tables cover all four.
- **Gain:** 2.1–2.9% fewer bits on the real clips. An ideal static model suggested 6–10% of location bits; the coder's neighbour and parent contexts already capture much of it. A 3×3 neighbourhood maximum (better in the ideal model) lost 0.5%, as did three classes; sign contexts would gain about 1%.
- **Cost:** one forward DWT of the prediction (1–1.2 ms per 720p frame on one core) and the class pyramids (built straight from 2×2 blocks).

**LL prediction:** LL bands are coded as residuals of the LOCO-I median predictor on quantized values. It saves **6.2%** on testsrc2 and 0.5% on terrain. It applies to intra frames only: a P-frame's LL residual is noise-like, and prediction would not help.

**Initial distributions (trained, normative):**
- Every context set (28 band classes and the 3 motion planes) starts from tables in `entropy/trained.rs`; CDFs with too little training data keep the hand-set heuristics (binomial ρ, geometric exponents, dense LL priors), which already saved 2–10% against flat priors.
- **Training:** `vdwt train-contexts` encodes clips at qp 24–56 (P-frame runs, and the first frames intra-only), re-codes every band and motion section through a recording coder, and counts each CDF's symbols per context set. The shipped tables were retrained on a disjoint corpus (POV bike, coastline, three Madeira segments; 88.8M symbols, `bench/contexts/build_corpus.sh`); probabilities are the counts with half a count added.
- **Held out:** the retrain is 0.78% worse than the earlier in-domain tables, which had seen the evaluation clips, but it is still 4.63% below untrained hand-set CDFs at ultraheavy. That 0.78% is the leakage, not a codec gain; the held-out tables are what ships, and the Pareto is regenerated from them.
- **Gain** at equal quality (bytes only; reconstruction is unchanged): 4.6% at ultraheavy against no trained CDFs; the samtime tiers lose slightly more than drone because the corpus has no studio content.
- Starting the trained CDFs at the slower adaptation rates (as if they had already seen 16 or 32 symbols) changed nothing (±0.3%), so they adapt from the start like the hand-set ones.
- **Continuity:** on fixed-qp streams a sequence flag (`continuous_cdfs`) makes the motion-tree CDFs carry from one coded frame to the next. That is safe because no frame can be skipped after its contexts advance. Rate-controlled streams reset them.
- **Per-band contexts (VERSION 10):** on the same fixed-qp streams every band keeps its own intra and inter context set across frames (`band_contexts`, flags bit 3). A band sees its own coefficient statistics, so this is a pure rate gain with **identical reconstruction**: +0.99% at realtime, +0.92 light, +0.85 medium, +0.83 heavy, +0.51 ultraheavy. Each band clones and writes back its set once per frame, so bands still code in parallel and the result is thread-count independent; no cross-band merge is needed. Memory is 8,676 bytes per set × 42 bands × 2 = **711 KiB** per encoder and decoder. Rate-controlled streams keep the old reset (so CBR is unchanged), and an intra frame resets both sets so a corrupt frame cannot poison later pictures.

**Unit of independence: the band** (not the code-block).
- Every band is one range-coded segment, starting from its class's default contexts. On fixed-qp VERSION 10 streams it then carries its own adapted contexts into the next frame (§5.1); otherwise the copy is fresh each frame.
- Per-code-block segments would cost a length and a flush for each of about 400 code-blocks per frame. At 60 fps that overhead alone is 50–200 kbps, which is fatal at low bitrate.
- **Parallelism:** 42 bands per 720p frame. The encoder can code them all in parallel. The decoder goes resolution by resolution and runs in parallel within each, because children need decoded parents.
- **Loss resilience:** a lost band blurs. Finer segmentation (rows of blocks within a band) is the packetization knob.

### 5.3 Single-pass rate control (implemented: `ratectl.rs`)
**Estimator: frame bits at any qp without quantizing or coding.**
- After the transform, every band gets a **magnitude histogram in 1/8-octave bins**, matching qp's resolution.
  - Bins in the lowest four octaves hold one integer each, so they are exact.
  - Bands of at least 128×128 are sampled 2×2.
  - Building them is table-driven, with four interleaved sub-histograms: about 1 ms per 720p frame on one x86 core.
- From a histogram, the nonzero count `n1` and the raw magnitude bits `Σe` follow for any step Δ and rounding θ. Measured per band, with the 2×2 sampling: 5% rms on terrain, and 12–17% on structured content (photos, graphics). Errors largely cancel at the frame level.
- **Predicted LL bands are not histogrammed.** They are quantized and predicted for real at each candidate qp: about 1,400 values at 720p. A histogram of unquantized residuals missed the ±1 residuals that quantization noise creates, and underestimated tiny frames by up to 56%.
- **Band bits:** `Σe + β·n1 + 20`, with **`β = a + b·log2(1/p)`** per context class (p = nonzero density).
  - Sparser nonzeros cost more to locate. A constant β mis-predicted frames by −30%.
  - `(a, b)` were least-squares fitted on coded bands with at least 16 nonzeros, using `vdwt band-stats`. The trained fit refit them for the trained tables, on the training clips: frame estimates are now unbiased with 2.7% rms (intra) and 4.1% rms (P-frames) error; the old fit overestimated P-frames by 9.5% and left 2–4% of the link unused.
  - With exact statistics the model predicts frame sizes within 7.5–9% rms. What remains is content bias of about ±10%, stable across qp.
  - **One learned correction factor** removes that bias: an exponential moving average, weight 0.3, of actual over modelled size at the chosen qp.

**Controller: the link as a transmit queue.**
- The link drains `R/fps` bits per frame interval, where R can change **every frame** (`Encoder::set_bitrate`).
- **Hard cap per frame:** `max_delay·R/fps − queue`. Every sent frame finishes transmitting within `max_delay` frame durations of being queued; the default is 1.5.
- **Target:** `R/fps − queue`, which drains the queue. It is held below `cap/1.15` so estimation error rarely hits the cap.
  - **Keyframes** (intra frames that P-frames will predict from) aim at `1.3·R/fps − queue`, still below `cap/1.15`: their quality carries into every P-frame after them. At the default 1.5-frame limit that is exactly where the cap binds. With a 3-frame limit, 2 frame budgets helped the synthetic clips but cost 0.2 dB (2 dB on the worst frame) on real footage, whose P-frames then starve while the queue drains (HISTORY.md).
- **qp choice:** binary search over 0–127 for the qp just under the target, then the nearer of it and its neighbour. qp steps are about 9% in rate, and always rounding down left about 5% of the link unused.
- **Credit (2026-10-24 round, §10.5):** unused capacity is not discarded. The queue may go negative down to one delay limit's worth of bits; the cap `max_delay·budget − queue` grows by the credit, so an easy stretch pays for the hard frame after it. The bound keeps every frame's queueing delay within the limit.
- **Quality ceiling:** from the same band histograms the controller estimates the frame's distortion at the candidate qp (`MagnitudeHistogram::distortion`, mapped to pixels through the MSE step weights). A frame is not coded finer than `quality_cap` dB of luma PSNR (default 50, `--quality-cap`; 0 disables); the bits it leaves are banked as credit. The ceiling waits until the reference is within 2 dB of it, so a P-frame repairing a coarse keyframe is never capped, and keyframes get a 2 dB bonus. This is what stops a constant-bitrate encoder from spending a whole static scene's link on refininng an unchanging picture; it is also why the static/quiet content in §10.5 uses less of the link while its measured PSNR changes by less than 0.1 dB.

**Enforcement after coding, exact, no re-encode.** Every band is a separate segment, so the encoder knows the frame's true size before writing it. When it exceeds the cap:
1. **Drop bands from the end of the coding order** (finest first). Bands depend only on earlier, coarser ones, so the remainder decodes exactly. The reconstruction zeroes the dropped bands too.
2. If the LL bands alone still do not fit, send a **Repeat frame**: 2 bytes, and the decoder shows its last picture again.
3. If not even that fits, **skip the frame**: write nothing, and the receiver keeps its last picture. This happens after a sudden rate drop, while the backlog queued at the old rate drains at the new one.
- **No grey frames:** when rate control squeezes a non-flat picture so hard that even its LL bands quantize to zero, a Repeat goes out instead of a featureless grey frame.

**Results** (HISTORY.md):
- 97–99% of the link used at steady rates (less when the quality ceiling banks the surplus, §10.5).
- Worst queueing delay 1.07–1.39 frames against a 1.5 limit.
- Reacts to rate changes on the very next frame.
- Survives a collapse to 10 kbit/s at 60 fps (about 21 bytes per frame).
- The decoder matches the encoder's reconstruction through drops, repeats and skips.

**P-frames:**
- **Model:** P-frame residual bands have their own `(a, b)` per class, fitted on about 12,000 bands from the synthetic drone clips. That gives 5.5% rms frame-size error with exact statistics, and 6.6% rms (+3.7% bias) with histograms. Intra frames and P-frames learn separate correction factors.
- **Motion section:** it is coded before the qp choice and counts as fixed overhead. Its motion decisions use λ from the qp chosen for the previous frame, even when that frame was a Repeat. Otherwise a stale, fine λ kept a ~100-byte motion section that could never fit a starved link.
- **Dropping:** P-frames may drop every band, LL included, and then fall back to global motion alone (§6.6) before resorting to a Repeat frame.
- **Graceful degradation:** on a starved link qp rises until a P-frame is little more than its motion, and the picture keeps moving. At 12 kbit/s and 60 fps that is 20–26 bytes per frame (HISTORY.md).

**Still open:**
- Proactive subband dropping as a quality tool at very low rates. The mechanism exists; a policy needs perceptual evaluation.
- Using the delay budget to smooth quality. The controller always aims at an empty queue, which is the low-latency choice.
- Speeding up the estimator (NEON, fused with the transform).

### 5.4 In-loop Wiener filter (implemented: `loopfilter.rs`)
After reconstruction, each plane of each frame may be filtered before it is shown and predicted from. The filter is a symmetric 5×5 Wiener filter in zero-DC form: every sample `r` gets `Σ_k c_k·(r[p + d_k] + r[p − d_k] − 2·r[p])` added, over the twelve offsets `d_k` of one half of the 5×5 square, in 1/64, rounded, clamped to 0..=255; samples outside the plane repeat the edge. Flat areas pass unchanged.
- **Coefficients:** `|c_k| ≤ 64` and `Σ|c_k| ≤ 64`. The bound keeps every whole sum inside 16 bits, so the AVX2 kernel works on 32 samples at a time in 16-bit lanes (one `maddubs` per pair of samples, wrapping adds), bit-exact with the portable code (`vdwt selftest` checks it). They are coded per frame and plane as signed Exp-Golomb differences from the plane's previous coefficients (zero after an intra frame): an unchanged coefficient costs 1 bit, a change of ±1 3 bits, ±2–3 5 bits.
- **Encoder:** least squares over the plane's interior give the coefficients minimizing the squared error to a **target**. By default the target is the source (MSE). Under `--tune perceptual` it is the source sharpened by `src + 0.4·(src − blur3(src))` (`--loop-sharpen 0.4`): the coded filter then restores mid-frequency detail that the wavelet quantizer smoothed, instead of only minimizing MSE. This is a perceptual target, not a display filter — the coefficients are coded and the filter stays in the prediction loop, so it changes references and residual statistics. It is also fully encoder-side: old decoders apply the same coefficients unchanged. It was the default from 2026-10-26 to 2026-10-28; PSNR-Y BD against the MSE target is +0.1–0.7% on the drone clips and **+5.5% to +13.4%** on gen_b (realtime/medium/ultraheavy), so it moved behind `--tune perceptual`. Six-point hybrid BD is **+5.57%** against the MSE target (VMAF +9.49%, YUV-PSNR −3.58%); the normal-VMAF gain is larger than the NEG-model gain (+5.39%), and a720/tasmania SSIM is flat or slightly better while b720 loses 0.0012 SSIM. An out-of-range solution shrinks as a whole (any fraction of the least-squares solution still saves error). A plane is filtered when the error saved exceeds λ·bits, with λ = 0.12·Δ² as for motion; where the fit samples every fourth or eighth pair of rows (light, medium, heavy) the luma λ is 6× that (chroma keeps the plain λ: with the 6× there too, 6:1:1 lost 0.3–0.4%), because a sampled fit overstates what it saves on the whole plane (2026-10-28: −0.9%, −0.6%, −0.3% PSNR-Y BD on nine clips, CBR unchanged; ultraheavy's every-second-pair fit and realtime's reused fits gained nothing). A still-camera skip was tried in the 2026-10-24 round and **removed**: it gained 1.4% on a talking head but lost 3–4 points of BD-rate on the samtime slow zoom, which sits at 0.97 px/frame. The filter stays on every frame and decides for itself. Under rate control the segment must fit what the frame cap leaves; chroma goes first. Effort 1 fits on every fourth pair of rows, effort 0 and realtime mode on every eighth (§10.2) (pairs, so both row parities of the wavelet count: fitting on even rows only lost 0.5–1.4%). On 360p clips half the rows cost 0.4%; on the 720p60 and 1080p60 FPV clips at the target links a quarter costs nothing measurable (±0.01 dB) and saves 1–3 ms per frame.
- **Why it pays:** wavelet coding noise and the repeated interpolation of the motion compensation blur the reconstruction in ways a per-frame filter undoes, and the filtered picture is a better reference too. Content shapes the filter: canyon's rock strata get a strongly diagonal one. Fitted on the reconstruction alone it gains 0.01–0.26 dB PSNR-Y; in the loop it saves 0.7–4.1% of the bits on luma and 5–6.4% on luma and chroma together (chroma reconstructions are blurrier).
- **Cost:** decoder about 1.3 ms per 720p frame on one x86 core (AVX2), encoder fit about 8 ms (both parallel across rows). Per-phase filters (one per position in the 2×2 wavelet lattice) gained only 0–0.08 dB and are not used. NEON kernels for the filter and the fit's normal equations (§10.2).

## 6. Inter prediction: global motion + mesh (implemented: `motion/`)

Prediction is a motion-compensated **warp or overlapped prediction** (chosen per frame, §6.3, §6.3.1) of the previous reconstruction.
- **Nodes** sit every 16 luma pixels by default (8 to 64 can be chosen in the sequence header), which is 81×46 nodes at 720p. Each carries a vector in 1/8 luma pel: the global prediction (§6.1) plus an offset that a quadtree of leaves shares (§6.2).
- **Between nodes** the field is the bilinear interpolation of the four surrounding vectors. It is continuous everywhere, so the prediction has no block edges.
- **Patches and windows** appear only inside motion estimation; the prediction never uses blocks.

### 6.1 Global motion (implemented: `gme.rs`)
**Model and coding.**
- The global model is the vector at each of the four corners of the node grid. Every node's vector is predicted by interpolating the corners bilinearly; this is normative integer arithmetic, rounded to nearest.
- That represents any affine motion exactly (pan, yaw, zoom), plus a bilinear twist that approximates mild perspective.
- It is coded as the change from the previous P-frame's model: about 3 bytes per frame for steady motion.

**Estimation (encoder only).**
- It runs against the **source** of the reference frame, not its reconstruction. The camera motion between the two frames is the same, and estimation then works at any quality. Matching against a qp-90 reconstruction failed and made every frame look like a scene cut.
- Steps:
  1. A ¼-resolution pyramid, from two 2×2 box downsamples.
  2. Up to 320 of the most textured 8×8 patches are block-matched with SAD. Patches are ranked by the smaller eigenvalue of the structure tensor, so straight edges rank below corners and texture.
     - Each search starts from the best of three guesses: the previous model, a whole-frame translation from matching row and column profiles (±128 px), and zero.
     - It searches ±3 around that guess, recentring while the minimum sits on the edge, then refines to sub-pixel with an SSD parabola.
  3. RANSAC: 100 rounds of 3-point affine fits with a 2-pixel inlier distance, then least squares on the inliers.
  4. A robust Gauss–Newton refinement of the six affine parameters on the ½-resolution frames, with Huber weights, using textured samples only.
- **Accuracy** against the ground truth of the synthetic clips (§9.3), at the grid corners:
  - 0.07–0.08 px rms for pan, yaw and zoom;
  - about 1 px where the true motion has perspective or rolling-shutter shear, which an affine model cannot represent. The mesh absorbs it.
- **Cost:** about 5 ms per 720p frame on one x86 core, 3.9 ms on 4 threads (was 9–10 ms).
  - The refinement uses the 16,000 most textured samples (a gradient histogram picks the threshold), not all 36,000 with any texture: 0.07 px rms instead of 0.06 on rigid motion, in under half the time. It sums only the 21 distinct entries of the symmetric normal matrix.
  - Patch matching runs one patch per task. The refinement sums 16 fixed chunks of samples, one per task, and adds them in order, so the model does not depend on the thread count.
  - The scene-cut check moves each 16×16 block by the model's displacement at its centre, in integer arithmetic (was a floating-point sample per pixel).
- **Not adopted: spatially stratified patches and samples.** An external audit suggested capping patches and refinement samples per grid cell, so that textured foreground in one part of the frame cannot dominate the fit. Tried with 2×2, 4×3 and 8×6 cells and caps of 0.6–2× the even share: BD-PSNR within ±0.02 dB on every clip except +0.05 dB on canyon at 720p, motion bytes within ±3%, and the corner vectors jumped more from frame to frame, not less (beach: 10.6 → 11–15 eighth-pels). Below 720p the caps rarely bind anyway: a 640×384 frame has only 209 patch positions (cap 320) and 15,360 samples (cap 16,000). The corner swings on beach are the affine fit to strong parallax, which the mesh corrects.

### 6.2 Motion quadtree (implemented: `tree.rs`, `partition.rs`, `refine.rs`)
**Temporal motion trial (VERSION 11, opt-in).** With `--temporal-motion` the encoder codes a
P-frame's motion section twice — once from the spatial median, once from the same node's offset in
the previous coded frame (minus that frame's global model) — and keeps the shorter. The section's
first bit says which. Reconstruction is identical; only the predictor changes. On the five private
clips at ultraheavy it measured **+0.00%** hybrid (a720 −0.00, b720 +0.04, canyon −0.00, forest
+0.00, tasmania −0.02): the spatial median already captures the redundancy, and the selector bit
offsets the few wins. It stays opt-in; default streams are VERSION 10 and byte-identical without
the flag.

**Coding (normative, `tree.rs`).** The node grid is cut into roots of 64×64 luma pixels, in raster order. Each root splits recursively into quadrants (Z order) down to single nodes. A leaf gives all its nodes one offset δ from the global prediction, in quarter pels, by one of five modes: **Global** (δ = 0), **Left** or **Up** (copy the δ of the node left of or above the leaf's top-left node), **New** (the component-wise median of the left, upper and upper-right neighbours' δ, or the first of them that is coded, plus a coded difference: per component a magnitude class and raw mantissa and sign bits), or **Intra** (every node of the leaf is an intra node, §6.4). Split flags, modes and difference classes have adaptive contexts (leaf size, neighbours split finer, neighbours moved), reset every frame and started from distributions trained on the training clips (0.4–0.5% fewer bits than hand-set ones; with no moved neighbour, Left and Up cannot occur and get almost no probability). New-vector differences are about 70% of the motion bits and their class symbols code within 1–3% of their entropy; a temporal predictor (the previous frame's offset) would save an estimated 9% of them, not done.
- Background that follows the global model costs a fraction of a bit per root; a region moving together (an object, a layer of parallax) costs one vector and then a mode symbol per leaf; fine leaves go only where motion changes.
- **Why:** with the earlier coding (Δx, Δy and intra planes, VEC-coded as deltas from the global model) a refined node cost about 16 bits, so a 16-pixel grid lost to 32 pixels. Unpenalised vectors predict much better on dense grids (coast, qp 38: 32.8 dB on the 32-pixel mesh, 33.9 dB on 16 pixels, 35.0 dB on 8 pixels), but their bits swamped the gain.
- **Measured:** the quadtree alone saved 10–18% on the three real clips (16-pixel grid) over the previous round; quarter-pel offsets another 2–2.6%. 8-pixel nodes are within 1–6% of 16 pixels but not better: the tent blending of 8-pixel cells is too short.

**Search (encoder, `partition.rs`).** Every node stands for the C×C pixels around it (C the grid spacing: 32 pixels by default from 480 lines up, otherwise 16, HISTORY.md). Roots are decided in coding order, quadrants in Z order, top-down: each region's best single offset (candidates: none, the neighbours', the enclosing region's best, the previous frame's; an integer diamond search with steps of 64 down to 1 pixel at the root, 2 and 1 pixel in quadrants; Lucas–Kanade; single-axis projections; offsets reach an eighth of the picture width from the global prediction, at least 48 pixels) against the cost of its four quadrants decided the same way. A region whose whole error could not pay for a split is not split. Errors are block matching; rates come from the coding modes. Root rows run as a wavefront (each waits for the root above and to its right), so the result does not depend on the thread count.
- Windows of large regions skip rows, never columns: every row is a contiguous run (the first pool's strided windows cost 4 ns per sample at fractional positions, 8× a contiguous one). An AVX2 kernel computes a window's error in 16-bit lanes, 16 samples at a time, from raw row pointers (the per-row overhead dominated on 16-wide windows).
- The candidates are costed exactly first; a region whose best candidate costs at most 16 bits' worth (λ·16) is not searched. A new vector costs at least 6.5 bits, so below that a search can never pay; up to 16 the bits are unchanged and 20–30% of the searches are skipped.
- Per-node exact errors are cached per root row (24 vectors per node), and node windows are built in parallel.

**Refinement (encoder, `refine.rs`).** Block matching misjudges vectors under overlapped prediction, which blends four nodes per pixel. This pass re-chooses the offset of every leaf of up to 2×2 nodes by the squared error of the *blended* prediction over all the pixels its nodes touch, neighbours held fixed, plus λ·R; candidates are none, the four neighbours' offsets, and ±¼ and ±1 pixel along each axis (and ±½ at effort 2); from the best it then descends in whole-pixel steps while they win, then in up to two rounds of quarter-pixel steps. It mostly turns new vectors into copies of a neighbour and corrects misses of one or more pixels; 70% of leaves move in the first pass, to every kind of candidate.
- **Parallelism:** leaves go in raster order. A leaf interacts only with leaves starting in the two node rows above or below and fewer than 4 columns away, so each leaf waits only for those (a wavefront over pairs of node rows, the lower one 4 columns behind; pairs balance the rows of 2×2 leaves against the lighter ones). The result is that of one raster scan, for any thread count. The earlier strip search changed the scan order at strip edges: narrowing them to balance 4 threads cost 0.3–0.5%.
- **Kernels:** a node's contribution is `w·s` with `s` the bilinear sample (16-bit, from one multiply per tap) and `w` a precomputed table of tent weights; one AVX2 pass computes a single node's candidate error directly against the target, without building the prediction. Errors are summed a few rows at a time and a candidate stops once it cannot win (exact; it rarely triggers, since candidates are close). 2×2 leaves go in bands of 16 rows. Twice as fast as the strip search, bit-exact.
- **Gain** (BD-rate, real clips): one pass 3.4–4.4% fewer bits, two passes 4.2–5.5%. The descent saves another 2.2% at 720p60 and 3.3% at 1080p60 on the Tasmania clip: the block-matching offsets are often pixels from the best ones for the blended prediction. A second pass instead saved 1.6% at 1080p for 40% more encoding time, the descent 3.6% for a third more. Refining larger leaves gained nothing; ranking candidates on every other row lost most of the gain; diagonal steps changed nothing; half-pixel steps save 0.1–0.4% for a quarter more time (effort 2 only).
- **Cost:** about as much as the rest of the search per pass. `effort` (0–2) sets the passes; at 2 quadrants also search as widely as whole regions (0.5–1% fewer bits).

**Decision constants.** λ = 0.12·Δ² as before (re-checked: ×0.7 and ×1.4 lose). Nodes whose whole error could not pay for the smallest offset keep the global vector.

### 6.3 Motion compensation (implemented: `warp.rs`, normative)
The warp is exact integer arithmetic, defined per pixel.
- **Vector:** the bilinear blend of the cell's four node vectors, `M = (C−u)(C−v)·m00 + u(C−v)·m10 + (C−u)v·m01 + uv·m11`, kept exact (unit `2^-(f+2c)` pixel).
- **Position:** in 1/64 pixel, `p = 64·x + ((M + 2^(k−1)) >> k)`.
- **Filter:** a separable T-tap filter at phase `p & 63`. Taps have 7 bits and sum to 128, and reference coordinates are clamped to the plane (edge extension): `out = clamp((Σ_j h_j·Σ_i h_i·R + 2^13) >> 14, 0, 255)`.
- **Chroma (4:2:0):** the same vectors, read as 1/16 chroma pel, with cells of 16 chroma pixels.
- **Shortcut:** along a row inside a cell, the position advances by a constant integer step. A row segment whose taps all lie inside the reference skips the clamping.

**Filters** (chosen per P-frame, in the frame header): bilinear, cubic (Keys a = −1/2, Catmull-Rom), sharp (a = −3/4), and smooth (the cubic B-spline).
- **Neither of the first two wins everywhere** (BD-PSNR against cubic):
  - **Rigid motion** (the synthetic clips, like a high-altitude survey): bilinear loses 1.5 dB, 44–48% more bits at equal qp, because its blur builds up over repeated warps (§6.8). Sharp loses 0.8 dB: its overshoot builds up too.
  - **Real FPV footage** (parallax, motion blur): bilinear *gains* 0.2–0.4 dB. Wherever the smooth field cannot follow the true motion exactly, a smoother prediction hedges the misregistration.
- **Decision: the encoder picks per frame.** It warps every 8th luma row with both cubic and bilinear and keeps the filter with the smaller squared error, at a cost of a quarter of one luma warp.
  - This matches the better filter on both kinds of content: within 0.03 dB of cubic on the synthetic clips, and 0.2–0.4 dB above it on the real ones.
  - It settles the "bilinear versus 4-tap" question: both stay, chosen per frame.
  - `--filter` forces one.
- **Smooth.** Bilinear wins on real footage by smoothing, so a smoother filter was tried as a third candidate: the cubic B-spline, an approximating filter that blurs even whole-pixel positions (`[1, 4, 1] / 6`). Its taps come from an integer Mitchell–Netravali generator (B = 1, C = 0), which reproduces the Catmull-Rom table exactly at B = 0, C = 1/2.
  - **It wins on real footage:** the encoder picks it for 54 of 59 coast frames, for +0.13 dB BD-PSNR-Y (+0.18 dB YUV) on coast, +0.11 dB on beach and +0.02 dB on canyon over the cubic/bilinear choice. Its blur cleans coding noise and sensor noise out of the prediction, as a loop filter would. Synthetic clips still pick cubic and are unchanged.
  - **Its cost:** it is a 4-tap filter replacing a 2-tap one on real footage, so the warp takes about 50% longer there (8.5 → 13.4 ms per 720p frame on one x86 core, in the decoder too), plus an eighth of a luma warp for the third candidate. That is the opposite side of the trade the per-run warp below would make, and is kept on the same principle: quality now, revisited with A55 numbers. `--filter bilinear` is the fast setting.
  - **Rejected:** Mitchell–Netravali with B = C = 1/3 (between Catmull-Rom and the B-spline) lost 0.05 dB to bilinear on the real clips.

**Kernels.** NEON has no gather, and on this machine's x86 core the "Downfall" (GDS) mitigation makes gathers slow, so both SIMD kernels avoid them:
- The field is smooth, so 8 neighbouring pixels read from a narrow window: their integer positions span a dozen columns at most, and one or two rows.
- Each group of 8 pixels computes its positions with vector arithmetic, loads 16 bytes of each reference row it needs, and hands every lane its samples with a byte shuffle (`pshufb` on x86, `tbl` on NEON). Taps come straight from the table: on NEON single-lane `ld4` loads deinterleave them.
- Samples are centred on 128, so every horizontal sum fits 16 bits; the vertical sum needs 32.
- A group straddling two rows computes both alignments and picks per lane. Wider groups (a steep zoom) and the tail of a cell row take the portable path.
- `vdwt selftest` checks every ISA's kernel against the portable code. The NEON kernel passes it on a Cortex-A55 (§10.2).
- The warp runs in bands of 32 rows, one task each.

**Cost.**
- x86, one core, one 720p frame: 4.2 ms (bilinear) or 8.7 ms (cubic) with AVX2, from 5.3 and 8.5 ms with gathers; the portable code takes 8 and 15 ms. On 4 threads the decoder's warp takes 1.8–2.6 ms.
- On the A55 the NEON cubic kernel is estimated at about 115 instructions per 8 pixels, mostly 128-bit: roughly 8–9 cycles per sample, twice the 4 cycles §2 assumed for bilinear.

**A faster definition, not adopted (yet).** Evaluating the field once per *run* of 8 horizontal pixels (at the run's centre) makes every run an ordinary separable filter with constant taps: one 16-byte load per row, `ext` for the alignments, `umlal`/`umlsl` by tap magnitudes (outer taps are never positive, inner ones never negative). That is about 55 mostly 64-bit NEON operations per 8 pixels, half the per-pixel kernel, and the same trade as the 4×4 sub-blocks of VVC's affine prediction. On x86 it halved the warp (4.1 ms against 8.5 ms, cubic). The cost (BD-PSNR-Y against per-pixel vectors):

| Runs | synthetic (survey) | real FPV (coast) |
|---|---|---|
| 8 pixels | 0.00 dB | −0.12 dB |
| 8 pixels, vector at the exact run centre | −0.01 dB | −0.12 dB |
| 4 pixels | 0.00 dB | −0.05 dB |

Real footage has parallax, so its fields change fast between nodes. 4-pixel runs would cost NEON about as much as per-pixel vectors. The format keeps per-pixel vectors until A55 measurements show whether the warp's share of the budget justifies about 2.5% more bits on real footage. The change is a ready patch (not in the repository).

### 6.3.1 Overlapped prediction (implemented: `obmc.rs`, normative)
The warp interpolates *vectors*, so the predicted picture is always a continuous deformation of the reference. Real FPV footage has motion discontinuities everywhere (parallax of trees, rocks, terrain edges), which a continuous field can only stretch across. Measured on coast against the same reconstructed references, 16×16 blocks with free full-search quarter-pixel vectors predict 1.8 dB better (luma PSNR, before the residual) than the 32-pixel mesh, and even a mesh searched with no rate penalty gains only 0.2 dB: the limit is the model, not the search.

**Definition.** Each node's vector predicts by plain translation (`s_n`, with two fractional bits), and every pixel blends its cell's four translated samples with the same bilinear weights the warp uses for vectors:

```text
s_n = clamp((Σ_j h[fy_n][j] · Σ_i h[fx_n][i] · R(ix_n − o + i, iy_n − o + j) + 2^11) >> 12, 0, 1023)
out = clamp((Σ_n w_n · s_n + 2^(2c+1)) >> (2c + 2), 0, 255)
```

An intra node contributes mid-grey (512). This is overlapped block motion compensation with tent windows, as in Dirac: no block edges, so it suits a wavelet residual, and neighbouring nodes can move independently. Each `s_n` uses one phase for its whole cell, so its filter is an ordinary separable one (each reference row filtered horizontally once per cell). Rounding `s_n` to 8 bits instead of 10 lost 0.04–0.07 dB (mostly chroma).

**Choice.** A frame flag selects warp or overlapped prediction. The encoder evaluates, on every 8th luma row, the warp with cubic and overlapped prediction with cubic, bilinear and smooth, and keeps the lowest squared error (pruning the warp's other filters costs nothing: it wins only on rigid motion, where cubic wins). Real footage picks overlapped prediction for nearly every frame; synthetic rigid motion keeps the warp.

**Gain** (BD-PSNR-Y over warp-only, 640×360 evaluation clips): coast +0.60 dB (11% fewer bits at equal qp), beach +0.42 dB, canyon +0.21 dB; synthetic clips about neutral. Against the previous round, at equal PSNR: 14% (beach), 21% (coast) and 6% (canyon) fewer bits.

**Kernels.** Each node is translated once per cell row over both cells it touches (2C wide), with constant taps: a horizontal pass on samples centred on 128 (sums fit 16 bits; centring subtracts exactly 512·2^12 before the final shift) and a vertical pass with 16×16→32-bit multiply-adds; then a separable blend. AVX2 runs 16 pixels per register and SSE/NEON 8; cells reaching past the plane's sides get padded rows and the same kernels. Bit-exact with the portable code (`vdwt selftest`); the NEON version runs on the A55 (§10.2).
- **Cost:** 6.3 ms for a 720p frame on one x86 core at the 16-pixel grid (4.4 ms at 32 pixels), from 20 ms; faster than the cubic warp (10 ms). The kernels run near this laptop core's shuffle-port limit.

**Tried with it and rejected:**
- A search window of a whole cell each way (what overlapped prediction reaches) instead of half: −0.3 to −0.8 dB on real footage, at stride 2 or 4.
- The 16-pixel grid: still −0.1 to −0.6 dB with the earlier node coding. With the motion quadtree (§6.2) it became the default.
- Coding node deltas as residuals of their neighbours' (median edge predictor) instead of plain changes from the global model: −0.03 to −0.11 dB. Most nodes keep the global vector, so the planes are mostly zero, and neighbour prediction only adds nonzeros at region edges.
- Retraining the context tables for overlapped prediction: no change.

### 6.4 Intra nodes (implemented)
- Each node carries an intra flag.
- The inter weight interpolates the flags bilinearly, like the vectors. Where it drops, the prediction fades towards mid-grey: soft ramps, never block edges.
- **Encoder rule:** a node goes intra when its motion-compensated error exceeds both 1.5× the neighbourhood's activity (plus a noise margin) and the error of plain grey.
  - The second condition matters at very low rates. There, no residual can repair a fade to grey.
- Intra nodes appear where new content enters at the frame edges and around occlusions.

### 6.5 Zero-region skipping (implemented: `dwt/sparse.rs`, 5/3 only)
- Inverse lifting of all-zero coefficients yields exactly zero.
- The sparse inverse transform tracks, level by level, which 8×8 blocks of each level's output can be nonzero. A block depends on its own and the neighbouring 4×4 coefficient blocks of the four input bands.
- The lifting then runs only over spans of such blocks, with 2-coefficient margins, so the result stays bit-exact. P-frames add the residual only in nonzero blocks.
- Verified bit-exact against the full inverse: 400 random sparse cases, on every ISA.
- **Measured: it rarely triggers on real content.**
  - On moving clips, and even on a static noisy one, 86–100% of the 8×8 blocks stay nonzero down to about 0.003 bpp.
  - The reason: the residual's LL band has by far the finest step (the MSE weights put Δ_LL5 near Δ/32), so it is nonzero almost everywhere. Every nonzero coarse coefficient keeps its whole synthesis support alive.
  - When it does not trigger, it costs nothing measurable.
- **A better target:** skip only the detail synthesis where a level's three detail bands are zero. Synthesis then reduces to interpolating the low band, and zero detail bands are common at low rates.

### 6.6 Scene cuts and emergency frames (implemented)
**Scene cuts.**
- After global estimation, the ¼-resolution source is predicted from the reference source with the global model.
- If more than half of the 16×16 blocks are predicted worse than their own activity plus 3 levels, the frame is a suspect.
- Above 75% the frame is coded intra. Between the threshold and 75%, it is coded intra only if its real P-frame prediction (after the node search) also predicts more than half of its 16×16 luma blocks worse than their activity plus 3 levels. Fast flight with rotation or parallax fails the global model while the node vectors still follow it : on real FPV footage the false cuts scored 0.50–0.51 with 1–10% of blocks failing the prediction, the real cuts 0.87–0.97.

**Emergency P-frames.** Rate control can shed every band of a P-frame, LL included, leaving its motion alone. When even that does not fit, the encoder falls back in steps:
1. drop the node deltas and intra flags, leaving the global model alone: a few bytes, after redoing the prediction;
2. then a Repeat frame;
3. then a skip.

On a rotating clip whose link collapses to 12 kbit/s, this keeps 33 of 40 frames moving at 24–25 dB. Repeat frames had frozen it at 14–15 dB.

### 6.7 Gradual intra refresh
Implemented in the `rt` codec (§10.3): the picture is swept top to bottom one root row at a time,
and clean-zone rows predict through a virtual boundary so a lost tile cannot leak. The main
codec has no refresh yet and uses `--intra-period` as its stopgap. If it gets one, exact cleanup
needs two adjacent intra node columns per frame; a single column leaves the ramps on either side
sampling the old reference.
### 6.8 Low-bitrate artifact profile (what to expect and what to watch)
VDWT should degrade **like HTJ2K** within a frame: blur first, ringing and halos near strong edges, and texture wash-out on foliage and water. It should not show blocks. Inter coding adds video-specific risks that HTJ2K never has:
- **Accumulated warp blur.** Each P-frame resamples the previous reconstruction.
  - Repeated bilinear interpolation acts as a low-pass filter that compounds over frames, and low-rate residuals are too small to restore the detail.
  - Measured: about 1.5 dB BD-PSNR over 60 frames of rigid motion. That is why the encoder picks the cubic filter there (§6.3).
  - Further mitigations: snapping the field to integer or half-pel where the mesh allows it, and intra refresh.
- **Rubbery or stretched objects.** Objects that move independently of the camera (cars, people, other drones) are only approximated by a smooth field. They distort instead of blocking. Intra nodes (§6.4) limit the damage.
- **Temporal shimmer.** Quantization decisions on texture coefficients flip from frame to frame and cause flicker.
  - Mitigations: dead-zone and reconstruction-offset tuning, and a temporally aware quantizer later.
  - P-frames already use a much wider dead zone than intra frames (θ = 0.12 against 0.25), which suppresses small residual flips.
- **Refresh band visibility.** At very low rates the sweeping intra band can show as a moving quality stripe. Tune the band width and quantization for intra regions.

### 6.9 B-frames (implemented: `bframe.rs`, the ultraheavy tier, normative)
A **pair** is coded with its later frame as a P anchor from the picture before the pair, flagged
**keep reference**, and its earlier frame as a B-frame between that picture and the anchor. The
decoder copies its picture aside when an anchor carries the flag, so I/P-only streams cost no
memory or work. A B-frame carries no loop filter, reconstructs into its own buffer and changes no
other decoder state. A pair adds one frame of latency (the encoder holds the two frames); the
four-frame group that the format still allows (three frames of latency) is not chosen by the CLI.
- **References and motion:** a B-frame at position `k` of `n` names its two references by position (0 the past anchor, `n` the next, else the kept middle frame). A direction's vectors are predicted from the anchor's own field (its global motion at intra nodes), which spans the group, scaled by `(k − position)/n` and rounded half up in integers; a coded field sends its offsets from that with the motion quadtree (§6.2), a direct one uses it as it is, for no bits. The six group fields share one 16-bit extension. On fixed-qp streams the quadtree CDFs carry from one coded section to the next, separately for each direction; a blend frame resets them because its two modified fields are encoded before the mode is chosen.
- **Prediction modes** (one per frame): past, next or their average, each by a coded field or directly; or **blend**: both fields coded, then every 64-pixel root's choice of past, next or both (an adaptive three-symbol code in the context of the root to its left), the two predictions weighted per node (64 past, 0 next, 32 both, of 64) with the weights interpolated bilinearly across each cell, so regions switch smoothly. In roots that use one direction the encoder zeroes the other field's offsets (one global leaf, a few bits). All arithmetic is integer; averages round half up.
- **Encoder:** both directions searched (the P-frame mesh search and refinement, priced at the B-frame's qp); per root the direction of least luma squared error plus λ times the motion bits it keeps; then the mode of least cost over the frame. qp: anchor + 9, or + 6 on a static camera (+ 8 when the anchor is qp 50 or coarser); rate control may override within the group budget (§10.4.1).
- **Static cameras (§10.5):** a pair is also coded as a B-frame group when the camera speed is below 1.5 px/frame, even if the motion share is small: on a talking head the B-frame predicts almost exactly and costs a fraction of its P-frame sibling (fixed qp: 8–12% fewer bits at equal PSNR; CBR: +0.3 dB at 1.5 Mbit/s). Its offset is 6, or 8 when the anchor qp is at least 50; under rate control its qp comes from its share of the group budget, clamped to at most 2 dB below the anchor and 6 above.
- **No cut guard:** a pair is not broken at a scene cut. The B before the cut predicts from the next anchor, and the cut frame itself is intra; this is cheaper than the I+P fallback on samtime in both rate and BD, so the old mean-luma cut guard is gone.
- **Groups of 4 on a still camera (2026-10-28):** at a fixed qp the ultraheavy tier steps 4 frames at a time; `Encoder::encode_pair` codes them as one hierarchical group of 4 (the middle frame a reference for the outer two) when the smoothed camera speed is below 0.25 px/frame (`STILL_GROUP_SPEED`), else as two pairs exactly as before. 100-frame talking head (The Action Lab, 1440p→720p): **−8.1%** against pairs (+29.6% → +18.9% behind x265 medium); a720 −0.4%, b720 and the four drone clips ±0.0% (forest lost 3% with the 1.5 px/frame static rule, whose smoothed speed passes 1 while it settles). Rate-controlled streams keep pairs and their delay limit.
- **Pairs:** the ultraheavy tier codes frames in pairs, each as two P-frames or a B pair, by the last P-frame's motion share (§10.4.1) or the static rule above. With rate control the anchor aims at the pair's budgets less the B-frame's learned share, and the tier's default delay limit is 6 frames.

## 7. Parallelism (4 cores) and latency

**Implemented (`pool.rs`).** The encoder and the decoder each own a fixed pool of threads (`threads`, the caller's included), created once: a 60 fps frame has several parallel stages, and spawning threads for each would cost more than some of them take. `Pool::run` hands out the indices of a stage with an atomic counter, the calling thread working too, and returns when all are done. Stages, per frame:

| Stage | Tasks |
|---|---|
| Global motion: patch matching; refinement sums; scene-cut check | per patch; 16 fixed chunks of samples; per row of blocks |
| Mesh refinement | per leaf-row task, as a wavefront (§6.2) |
| Warp (filter choice, prediction; decoder too) | bands of 32 rows of every plane |
| Residual and forward DWT | per component |
| Quantization, LL prediction, significance pyramids, statistics | per band, largest first |
| Band coding | per band, largest first |
| Band decoding (decoder) | per band, each waiting only for its parent band (claimed earlier) |
| Dequantization, inverse DWT, reconstruction | per component |

**The stream does not depend on the thread count**, nor does the decoder's output; tests check 1, 2, 3 and 4 threads. Tasks write disjoint outputs, and the only floating-point reduction (the global refinement) adds fixed chunks in a fixed order. Every band has its own value plane, so bands are coded and decoded concurrently; the band coder reads its parent band's values from the parent's plane.

**Still serial:** the pyramids, patch selection, RANSAC, the global model, the motion section, rate control and the bitstream. The component stages are only as parallel as luma is large: luma is two thirds of the samples, so the transform and reconstruction scale to about 1.5×. Splitting luma by rows (with a halo, or a barrier per level) is the next step there.

**Not done:** pinning threads to cores. The A55 scaling is measured in §10.2.

**Latency**
- Capture of a full frame, plus encode (≤ 8 ms), plus radio, plus decode (≤ 8 ms) and display. No buffering anywhere.
- Sources over MIPI CSI and ISP deliver whole frames, so frame-level processing does not add latency beyond capture.
- A later **stripe mode** (line-based DWT with fewer vertical levels) can cut latency to sub-frame for sources that stream lines.

## 8. Bitstream and packetization (implemented: `bitstream.rs`)

```
stream    := sequence_header frame*
sequence  := "VDWT" version:u8(=10; 11 with the opt-in temporal motion trial; 9 still decodes) width:u16 height:u16 chroma_format:u8(=1, 4:2:0)
             bit_depth:u8(=8) kernel:u8(0 = 5/3, 1 = 9/7 with 3 fraction bits)
             luma_levels:u8 chroma_levels:u8
             block_log2:u8 mesh_log2:u8(3..6)
             flags:u8(bit0 guide, bit1 continuous motion-tree CDFs,
                      bit2 range-coded band layout, bit3 per-band carried CDFs,
                      bit4 reserved, bit5 temporal motion trial [opt-in, VERSION 11])
             fps_num:u32 fps_den:u32
             luma_weights:u16[1+3·Ly] chroma_weights:u16[1+3·Lc]       (Q12 step weights)
frame     := varint(len) payload
payload   := type_frame:u8(bits0-1 type: 0 = I, 1 = Repeat, 2 = P, 3 = B;
                        bit2 short header: qp, recon offset and post-filter are inherited;
                        bits3-7 frame number 0..30, 31 escapes to a varint)
             [frame_num:varint]                                         only when escaped
             (Repeat: the payload ends here; it inherits the previous coded frame's display
              state, including the post-filter strength)
             [qp:u8 recon_offset:u8(δ·256)]                             omitted when short
             flags:u8(bit0 parent contexts, bit1 LL prediction,
                      bits2-3 P-frame filter: 0 bilinear, 1 cubic, 2 sharp, 3 smooth;
                      bit4 P-frame prediction: 0 warp, 1 overlapped;
                      bit5 loop-filter segment present;
                      bit6 keep reference: on an anchor, a B-frame group follows that needs
                           the replaced picture; on a B-frame, it is the group's middle reference;
                      bit7 a post-filter byte follows)
             [postfilter:u8(strength in 1/8 units, 1..=40; absent = off; the encoder picks one
                            per scene, constant until the next intra frame)]
             [motion: varint(len) range-coded motion]                   P-frames only
             b_extension:u16(bits0-2 mode, 3-5 k, 6-8 n, 9-11 past position,
                             12-14 next position)                       B-frames only
             [field: varint(len) quadtree per used direction, past first]
                                                                        B-frames, searched modes
             [blend: varint(len) root choices]                          B-frames, blend mode
             layout: varint(len) range-coded length of every band, in coding order
             segments: range-coded band data, concatenated in coding order; layout gives the
                       length of each, and length 0 means the band is absent
             [loop: varint(len) loop filter]                            if flag bit 5
loop      := per plane Y, U, V (bits MSB first): on:1, then if on 12 coefficient
             differences as signed Exp-Golomb codes (0, 1, −1, 2, …), §5.4
motion    := 8 global corner-vector changes, each a magnitude class (16-ary symbol) plus raw
             mantissa and sign bits; then per 64-pixel root, in raster order, a quadtree of
             split flags and leaves (mode: global, left, up, new, intra; new carries a
             quarter-pel difference per component as a class plus raw bits), §6.2
```
- All integers are little-endian. Unknown frame types are rejected as unsupported.
- **Band coding order:** resolution by resolution, coarsest first, then Y, U, V within each, then HL, LH, HH. With 4:2:0 and one fewer chroma level, each resolution pairs luma and chroma bands of the same spatial frequency.
- **Layout:** one range-coded length per band replaces the old presence mask and per-band length
  varints. A length is coded as its bit length (0 = absent) plus a raw mantissa, with static
  bit-length priors per band class and a within-frame adaptive model. The section is stateless,
  so a corrupt frame cannot desynchronise later frames. A typical fixed-qp layout is roughly 8–40 bytes
  for 42 bands, depending on rate and detail; the band data then follows with no per-band prefixes.
- **Short headers** are a fixed-qp optimisation: when qp, recon offset and post-filter strength
  equal the previous coded frame's, the encoder sets bit 2 and omits those three bytes. The
  decoder keeps the same last-header state. Rate-controlled streams do not enable the option.
- **Repeat** frames write only the type byte (plus the escaped frame number when needed): no qp,
  recon offset, flags or post-filter byte. They inherit display state and do not update the
  short-header state. With the frame's own length prefix a Repeat is **2 bytes**.
- **Coarse-to-fine order** means the most important data comes first, which suits unequal error protection (FEC) on the radio link.
- **Graceful loss:** losing a fine band only **blurs** with no garbage blocks; intra refresh then cleans up drift.
- **Repeat frames** (type 1) show the last picture again. A Repeat before any picture is an error.
- **P-frames** carry their motion section before the layout. Node vectors are clamped to ±4096 pixels and global corners to ±2048, so a corrupt stream cannot overflow the warp.
- **B-frames** pack the six group fields into one 16-bit extension (3 bits each), then the coded
  fields and, for blend mode, the per-root choices. Their motion-tree CDFs carry from one coded
  B section to the next on fixed-qp streams; a blend frame resets them because its sections are
  encoded before the mode decision. Direct modes leave the state untouched.
- **Still to come in the main codec:** finer segments. Packetization and loss handling live in
  the separate `rt` codec (§10.3).
- **Robustness:** the decoder never panics on malformed input.
  - Tested in debug builds with overflow checks: 2000 truncated, bit-flipped and random intra payloads, 1500 corrupt P-frames, and mutated sequence headers.
  - The P-frame fuzzing found one overflow: a corrupt residual added to the prediction. The add now saturates.

## 9. Rust implementation

### 9.1 Workspace layout
The library and CLI have **no external dependencies**, so the A55 build is a single static binary.
```
Cargo.toml                   workspace; release = thin LTO, 1 codegen unit, line tables
.cargo/config.toml           aarch64-unknown-linux-musl: rust-lld, target-cpu=cortex-a55
crates/vdwt/                 library
  src/aligned.rs             ✔ 64-byte aligned buffers
  src/plane.rs  frame.rs     ✔ planes (aligned rows), 4:2:0 frames, u8 <-> centred i16
  src/cpu.rs                 ✔ ISA selection (scalar / avx2 / neon)
  src/dwt/                   ✔ 5/3 DWT: mod.rs (engine + streaming schedule), geometry.rs
                               (subbands), kernels.rs (table), scalar.rs, avx2.rs, neon.rs,
                               oracle.rs (Annex F reference), verify.rs (shared bit-exact checks)
  src/metrics.rs             ✔ PSNR
  src/rng.rs                 ✔ SplitMix64 for tests and synthetic content
  src/quant.rs               ✔ dead-zone quantizer, normative Q12 steps, MSE step weights
  src/entropy/               ✔ cdf.rs (adaptive CDFs), range.rs (range coder),
                               band.rs (VEC: quadtree, contexts, shared enc/dec traversal,
                               training recorder), trained.rs (generated default tables)
  src/training.rs            ✔ symbol statistics for `vdwt train-contexts`
  src/bitstream.rs           ✔ headers, varints, frame and segment syntax
  src/codec.rs               ✔ band coding order, context classes, LL prediction, dequantization
  src/encoder.rs decoder.rs  ✔ intra frames, P-frames and emergency frames
  src/error.rs               ✔ error type
  tests/codec.rs             ✔ end-to-end: enc recon == dec, lossless, ISA-identical streams,
                               corrupt-stream robustness
  src/ratectl.rs             ✔ magnitude histograms, fitted bits model, transmit-queue controller
  tests/ratectl.rs           ✔ delay limit, link tracking, forced band dropping, repeats, skips,
                               P-frames shedding bands down to motion alone
  src/motion/                ✔ mod.rs (node grid, global model, motion section coding),
                               warp.rs (normative warp, filters, portable kernel), warp_avx2.rs,
                               warp_neon.rs (shuffle kernels), obmc.rs (overlapped prediction),
                               pyramid.rs, gme.rs (global ME + fit), mesh.rs (nodes, intra flags)
  src/dwt/sparse.rs          ✔ inverse transform with zero-region skipping
  tests/inter.rs             ✔ P-frames: long runs bit-exact with every filter and grid, odd
                               sizes, lossless, scene cuts, ISA-identical streams, corrupt streams
  src/pool.rs                ✔ worker pool for the parallel stages
  src/profile.rs             ✔ per-stage frame times (`FrameStats::times`, `Decoder::times`)
crates/vdwt-cli/             `vdwt` binary
  src/main.rs                ✔ encode (fixed qp, or --bitrate schedules with --max-delay and
                               --frame-log; motion options), decode, rd, band-stats, pred-dump,
                               train-contexts, info,
                               selftest, bench (DWT and motion compensation), dwt-roundtrip,
                               dwt-dump, psnr
  src/y4m.rs                 ✔ 8-bit 4:2:0 reader and writer
tools/ref_rd.sh              ✔ reference RD points: intra (x264, x265, JPEG 2000, SVT-AV1) or
                               low-delay P-frames (x264 ultrafast/veryfast/medium, x265 medium)
tools/ref_drone.sh           ✔ x264 at full strength in drone mode (I/P only, no lookahead, single
                               thread): every preset, with encode speed
tools/sweep.py               ✔ setting-vs-baseline comparison: size change and BD-PSNR
tools/synth_motion.py        ✔ drone-style clips with known camera motion, and a checker for the
                               encoder's global motion against that ground truth
tools/                       later: CDF-default training
```

### 9.2 SIMD strategy
- Stable `core::arch::aarch64` NEON intrinsics, with `-C target-cpu=cortex-a55` for the target build.
- AVX2 on x86 for development (this machine is an i5-10210U), with runtime detection.
- Each SIMD body is a `#[target_feature]` function behind a thin safe wrapper. On AArch64 this is needed even though NEON is always on: Rust does not count build-wide features when deciding whether an intrinsic call is safe.
- A scalar oracle for every kernel, and randomized tests requiring SIMD and scalar to be **bit-exact**.
- No allocation per frame. Buffers are 64-byte aligned and allocated once.

### 9.3 Toolchain and testing
- **Cross-building works today without a C cross toolchain.** `cargo build --release --target aarch64-unknown-linux-musl -p vdwt-cli` links with Rust's bundled `rust-lld` into a static binary. It runs on any aarch64 Linux, or on Android via `adb push`.
- **ARM tests:** `qemu-user` is optional (not installed here); with it, `cargo test --target aarch64-unknown-linux-musl` runs under emulation (correctness only). On a real board, `vdwt selftest` runs the same bit-exactness checks; the NEON kernels run and
pass on the Orange Pi 3B (§10.2).
- **Performance numbers come from a real A55 board.** Copy the binary and run `vdwt info`, `vdwt selftest`, then `vdwt bench --ghz <clock>`.
- **Test content** is drone footage at 720p:
  - fast pan and yaw;
  - altitude changes (zoom);
  - foliage and water (hard textures);
  - sky (smooth);
  - vibration and rolling shutter.

  The first sequence set still has to be chosen, for example public aerial datasets or your own captures.

  Until then, `tools/synth_motion.py` renders drone-style clips with known motion. A virtual camera flies over a large still image (mirror-tiled, rendered with supersampled bicubic sampling), with presets for survey flight, fast pan, yaw, climb, gimbal tilt, static, hover, and vibration with a rolling shutter. Sensor noise, exposure drift and an independently moving object are optional.
  - It writes the true camera pose per frame, and `--check` scores the encoder's global motion against it.
  - What synthetic clips cannot provide: parallax, real sensor noise and processing, motion blur. Real footage is still needed for tuning and for credible comparisons (§12).
- **Correctness gates:**
  - DWT lossless round trip;
  - lossless codec mode;
  - encoder recon == decoder output (bit-exact, all platforms), over long P-frame runs;
  - packet-loss simulation;
  - `cargo fuzz` on the decoder.
- **Metrics: a stack, never PSNR alone.** MSE-optimal coding drifts toward blur, because when detail cannot be afforded, the blurred average minimizes squared error. Tuning on PSNR alone would make VDWT blurrier than it needs to be. Which metric applies depends on what a change affects:

  | Change type | Judge by |
  |---|---|
  | Lossless-equivalent (entropy coder, bitstream syntax, any change with identical reconstruction) | **Bytes only.** Distortion is unchanged, so no metric is needed. |
  | Prediction accuracy (ME, mesh, interpolation filter) | PSNR-Y per frame, **PSNR versus frame index** for accumulated blur, and SSIMULACRA2 |
  | Artifact character (5/3 vs 9/7, subband weights, dead-zone, subband dropping, refresh) | **Perceptual metrics and eyes**: SSIMULACRA2, MS-SSIM, VMAF (NEG mode, so sharpening cannot game it), XPSNR, then side-by-side viewing |
  | Temporal quality (shimmer, refresh stripe, pumping) | A frame-difference flicker measure, VMAF's motion features, and viewing in motion at 60 fps (not single frames) |

  - BD-rate is reported **per metric** in the 1–6 Mbps range.
  - A feature is adopted when the perceptual metrics agree and viewing confirms it. PSNR is a sanity check, not the decider.
  - Inside the encoder, rate and mode decisions use cheap weighted MSE. Perceptual intent enters through subband visual weights (CSF) and, later, texture masking.
  - Tools: FFmpeg `libvmaf` and `xpsnr`, and the Rust `ssimulacra2` crate. A small A/B viewer tool goes in `tools/`.
- **Reference codecs** (`MODE=inter tools/ref_rd.sh`), all low delay (P-frames only, no B-frames, no lookahead, PSNR tuning):
  - x264 `ultrafast` and `veryfast`, the realistic real-time competitors on edge CPUs;
  - x264 `medium` and x265 `medium`, as efficiency references;
  - VDWT intra-only.

## 10. Status and measurements

This is the current state. The detailed historical log (old rounds, superseded defaults and
rejected variants) is in the internal lab notes; the sections below keep only what the
current code and decisions rest on.

### 10.1 Tiers and defaults

| Tier | Effort | Search | In-loop | B-frames | Use |
|---|---|---|---|---|---|
| `realtime` | `--realtime` | coarse: static stripes, high-rate shortcuts, 2-row windows; cubic interpolation without the per-frame trial (bilinear until 2026-10-28: −4.1% YUV-PSNR 6:1:1 BD on nine clips, talking heads −3.5 to −12.3%, for 2–5% more x86 encode time; on the A55 with 4 threads +2–3% encode, 47.7 / 45.2 ms at fixed qp / 600 kbit/s, and +4–6% decode, 23.6 ms) | loop filter off under rate control; quality guide on, flat opt-in | no | small CPUs, 720p60 |
| `light` | 0 | full search, static stripes, one candidates-only refinement pass (no offset and the neighbours', no descent), every-eighth loop fit | loop filter + guide | no | fastest general tier |
| `medium` | 1 | full search + one refinement pass | loop filter + guide | no | default |
| `heavy` | 2 | wide search, two refinement passes | loop filter + guide | no | quality per time |
| `ultraheavy` | 3 | wide search, four refinement passes (two on a static camera below 50 fps), B pairs, adaptive mesh probe | loop filter + guide | pairs | offline/general-purpose quality |

Light's candidates-only pass (2026-10-28) gains 4.9% PSNR-Y BD on nine clips (drone 2.7–6.4%,
talking heads 5.5–9.3%) for 29% more x86 encode time (22.6 → 29.2 ms/frame single-thread on
Tasmania); the full pass would gain 7.4% for 2.3×. Realtime mode keeps no pass: the same
candidates-only pass gained 5.3% (talking heads 7–10%) but doubled its A55 encode time on drone
footage (4 threads, 720p: 47.7 → 95.5 ms fixed qp, 45 → 94 ms at 600 kbit/s; pod_b +36%).

`--effort 0..3` name the same tiers; `--realtime` is the separate small-CPU mode. Every tier uses
the integer 9/7 wavelet (lossless and `--wavelet 53` use 5/3), 5 luma and 4 chroma levels, a
64-pixel band root and the VEC range coder. Streams are independent of the thread count (1, 2, 3
and 4 threads are tested).

The encoder defaults include causal frame activity adaptation (`--frame-aq 2.0`, §5.1), which
also helps PSNR-Y (0.2–0.9%). The perceptual in-loop-filter fit target (`--loop-sharpen 0.4`, §5.4)
and the finest tilt are `--tune perceptual` only since 2026-10-28: on PSNR-Y every tier is
3.8–4.9% (drone) and 8.5–16.1% (samtime gen_a/gen_b) below the 2026-10-23 build with them off,
against 3.0–4.0% and up to +0.8% with them on. `--frame-aq 0 --loop-sharpen 0` reproduces the VERSION 9 baseline stream
byte for byte with the pre-VERSION-10 build; VERSION 10's per-band contexts change the entropy
state. Neither perceptual tool changes the syntax or the decoder; the qp and the filter
coefficients are already coded.

### 10.2 Current performance

**Laptop (i5-10210U, 4 threads, 20 frames, fixed qp 42/50/58), BD-rate against x265 medium
(positive = VDWT needs fewer bits than x265 medium; negative = more):**

| tier | samtime BD | drone BD | encode ms/frame (samtime) |
|---|---:|---:|---:|
| realtime | **−21.8%** | +0.5% | 9.2 |
| light | **−17.0%** | +2.9% | 12.8 |
| medium | **−12.1%** | +10.3% | 15.5 |
| heavy | **−12.4%** | +10.7% | 20.7 |
| ultraheavy | **−7.1%** | +17.9% | 33.1 |

2026-10-26 VERSION 10 free build (perceptual defaults + adaptive tilt + per-band contexts +
B qp 11), hybrid BD against x265 medium on the same five clips (three qps; positive = fewer bits):

| tier | all 5 clips | samtime | drone | clean encode ms/frame |
|---|---:|---:|---:|---:|
| realtime | −12.55 | −30.33 | −0.70 | 17.3 |
| light | −7.93 | −23.52 | +2.47 | 22.0 |
| medium | −0.80 | −18.76 | +11.17 | 35.8 |
| heavy | −0.52 | −19.17 | +11.92 | 55.2 |
| ultraheavy | **+5.42** | **−13.07** | **+17.76** | 87.4 |

Against the VERSION 9 baseline the gain is **+5.55 to +6.81 points** at every tier (five-clip
mean). AQ alone is +0.84%, the perceptual filter is most of the pair's +5.57%, the adaptive tilt
adds +0.71% on the six-point grid, per-band contexts +0.5–0.99% depending on tier, and B qp 11
+0.35% at ultraheavy. The encode times are wall times including process startup and `--recon`,
measured clean; the tilt and B qp change only coded parameters, and the per-band contexts add
only copies inside the entropy stage, so the build shares the earlier timing run. The published
before/after charts are `videos/pareto_free2_old_new.png`,
`pareto_samtime_free2_old_new.png` and `pareto_drone_free2_old_new.png`, with
`bd_bars_free2.png`; the earlier +4.95 ultraheavy build is `pareto_free_old_new.png`. The
underlying data is `target/research/csv/{old,new,free,free2}.csv`, and
`bench/research/plot_old_new.py` regenerates them.

The earlier published charts are `videos/pareto_samtime_encode.png` and
`videos/pareto_drone_encode.png`, with decode-time counterparts `*_decode.png` (and
`*_decode_clean.png`). They use the hybrid **70% VMAF + 30% YUV-PSNR** BD-rate against x265
medium (`vmaf_v0.6.1`), not PSNR-Y; the PSNR-Y table above is kept for continuity and the old
charts are archived as `*_psnr_y.svg/png`. Those charts show the held-out context-table rows at
**−22.0%** (samtime ultraheavy) and **+10.4%** (drone ultraheavy); the current before/after grid
gives **−13.07%** and **+17.76%** for the same two groups. A22's chroma 1.2 was reverted after it measured worse under
this standard. Decode timings are measured separately from the same streams: VDWT with its own
CLI in-process average, x265 through FFmpeg from a looped raw stream; average ms/frame is the
x-axis. `bash bench/complexity/plot_all.sh` regenerates all eight charts; `decode_bench.py`
refreshes the timings.
They show the current v13 build against x265; historical builds are not comparable under the hybrid metric and are archived separately;
the x86 wall times are the final benchmark, the Pi below is the iteration reference.

**Orange Pi 3B (RK3566, 4× Cortex-A55, 1.8 GHz, fan), 720p, 4 threads:**

| workload | result |
|---|---|
| realtime, 600 kbit/s, 120 frames | **19.8 ms/frame (51 fps)**, 42.64 dB, 559.3 kbit/s |
| realtime, `--no-guide`, same | **17.4 ms/frame (57 fps)**, 41.99 dB, 555.2 kbit/s |
| realtime, fixed q42, 120 frames | 38.4 ms/frame, 40.40 dB, 345.6 kbit/s |
| heavy a720 q42, 20 frames | **130.8 ms/frame**, 37.63 dB, 576.7 kbit/s |
| ultraheavy a720 q42, 20 frames | **204.8 ms/frame**, 37.04 dB, 483.0 kbit/s |
| ultraheavy forest q42 (60 fps) | **409.9 ms/frame**, 28.84 dB, 9839.2 kbit/s |
| ultraheavy b720 q42, 20 frames | **266.7 ms/frame**, 36.52 dB, 763.3 kbit/s |

Re-measured 2026-10-26 on the current build; the rates and PSNR include A19, the final-intra
change, chroma 0.95, the held-out context tables and the VERSION 9 syntax. A same-session A/B
against the old binary gives 19.80–19.92 vs 19.89–19.92 ms at 600 kbit/s and 204.8–206.4 vs
205.2–206.7 ms for ultraheavy, so the syntax changes cost no measurable speed. The fixed-qp realtime row is not the same
workload as the CBR rows: under rate control realtime skips the in-loop filter (A15) and repeats
the 67 static frames, while fixed q42 codes 118 P frames with the filter, hence 38.4 vs 19.8 ms.
The 60 fps budget is 16.7 ms: the default is 19.8 ms (51 fps), `--no-guide` 17.4 ms (57 fps).

The 2026-10-26 perceptual defaults were checked on the same board: `selftest --isa all` passes;
the CBR realtime stream is **byte-identical** (559.3 kbit/s, 42.64 dB) and stays at **19.82–19.91
ms** (old 19.87–19.93) because rate control skips the loop filter and AQ is disabled there.
Fixed-qp ultraheavy grows from 205.4 to **216.0 ms** on a720 q42, from 415.5 to **440.3** on
forest, and from 265.4 to **273.7** on b720: the perceptual filter's source pass, not the entropy
coding. The Pi reproduces the encoder reconstruction bit-for-bit, and the A55 decoder from before that round (VERSION 9) decodes that stream to the same MD5; the
VERSION 10 per-band contexts are later and were not run on the board.
The zero-copy output change (§10.7) removed the per-frame reconstruction clone and is the largest
recent win; the CBR stage split is global motion 1.7, mesh 4.9, prediction 2.8, guide transform
4.1, quantize 3.4, entropy 0.6, reconstruct 2.5 ms, plus the frame loop. The A15–A17 code paths are present: rate-controlled realtime skips the loop filter, and the CBR
global-motion stage is 1.7 ms against 2.8 ms at fixed qp, consistent with the held-shot GME
bypass.

**Thermal.** 240 s of four `yes` hogs followed by 240 s of ultraheavy encoding held all four
cores at 1.8 GHz: CPU peaked at 62.2 °C (trip points 70/75/95), GPU at 50 °C, encoder 267.6
ms/frame mean and flat (the then-current four-pass build). Fan connected; no throttle. Logs in `bench/complexity/pi_stress.csv`,
plot in `videos/pi_thermal_stress.png`.

### 10.2.1 Vector kernels (2026-10-26)

The scalar quantizer is auto-vectorized by LLVM, but the hand-written AVX2 kernel is still
**3.09×** faster in isolation (1.54 → 0.50 ms for 921k coefficients) and the dequantizer is
**1.18×**. On the A55 the same NEON quantizer measured **0.72×** — slower than the compiler's
auto-vectorized scalar loop (1.63 vs 2.26 ms) — so aarch64 keeps the scalar quantizer; only the
AVX2 path is dispatched. The mesh-search gradient fill gained **1.61×** on x86 and **1.50×** on
the A55 and is used on both.

All kernels are bit-exact against the scalar paths (`simd_tests` for quantize/dequantize, a
gradient test for mesh); a720 q42 ultraheavy streams are byte-identical with `--isa scalar` and
`--isa neon`, and the x86 stream is unchanged. End-to-end on the Pi this moved realtime CBR from
19.95 to 19.6 ms and ultraheavy a720 q42 from 205.3 to 203.9 ms (about 1%); on x86 the whole
encode gain is 1–2%, as expected from the quantize stage's share.

### 10.3 The rt codec (separate pipeline)

`vdwt::rt` is a separate realtime stream (`VDRT`) for links that lose packets. It shares the
kernels but not the syntax: the main stream is one range-coded band segment per frame and cannot
resynchronize after a loss, while `rt` cuts each frame into 64-pixel-root-row tiles, gives every
packet its own header/global motion/clean-zone rows, and sweeps a gradual intra refresh over 30
frames. Clean-zone rows predict through a virtual boundary so a lost tile never leaks. A 30-frame
sweep costs 0.2 dB without loss and gains 2.5–3 dB at 5–10% loss; after one sweep the decoder is
exactly the encoder's again. The full tile/loss/refresh design and its A55 measurements are in
the internal lab notes.

### 10.4 Tiers on general content, and ultraheavy

The tiers are the four graded efforts plus `realtime`. On two general clips (A: a talking head
with cuts; B: a slow zoom on a stage), every tier still needs more bits than x265 medium at fixed
qp; the gap shrinks from light through medium and heavy to ultraheavy, which closes most of it.
x265 veryslow remains 18–21% ahead of its own medium. The whole-image wavelet's intra coding is
the structural gap on sharp synthetic content (it beats OpenJPEG but trails directional intra
prediction); P-frames close most of it, and ultraheavy closes the most.

**Static/podcast gap.** On a720/b720 at medium q42 the motion section is 43% / 33% of the
P-frame payload, and 1,869 / 2,578 of 3,726 nodes are refined. Snapping small leaf offsets to the
global prediction hurt, but a **near-zero global model is itself the problem**: when the average
corner speed is below 1.25 px/frame (0.70 in ultraheavy, which keeps b720's pair motion), the
affine fit is mostly foreground/compromise, so the encoder codes exact zero global motion.
Background leaves then reach exact (0, 0), and the samtime BD-rate improves from −40.1 to −26.3
(realtime), −29.4 to −18.9 (light), −23.8 to −13.9 (medium), −22.7 to −13.9 (heavy), and −17.1
to −7.6 (ultraheavy; last-intra coarsening, a +1 P-frame qp offset, a sharper opening
I-frame after a hard cut, and the unreferenced first shot coded at qp). The rest of this round is a higher chroma step (0.95 everywhere:
about 1% fewer bits at the same PSNR-Y, chroma about 0.5 dB lower) and no half-pel
refinement on a static camera at heavy. The remaining gap is the whole-image wavelet's cost on text and sharp edges and
its residual leakage across a large static background; closing it would need directional intra
or block-local transform tools, which conflict with the block-free design.

A 2026-10-25 sweep of subband weights, small-coefficient gates, loop-filter margins, mesh/block
sizes, rounding, intra offsets, chroma scales, temporal AQ, cross-orientation contexts and
display post-filters found **no fair coding gain**; the current settings remain. Two optional
display filters live in `postfilter.rs`: a generic 5×5 unsharp (`--postfilter unsharp`, 0.6), a
custom mid-frequency band-pass (`--postfilter band`, 2.0) and a wider band-pass
(`--postfilter band2`, 2.0, with `[1 2 3 2 1]/9` on the low-pass side). Against x265 stock the
first two raise VDWT's ultraheavy hybrid mean from −1.87% to +7.22% and **+8.63%**; against x265
plus the normal unsharp they measure −2.74% and **−0.35%**. `band2` improves samtime VMAF by
**+1.42** over `band` 2.0 while costing **−0.84 dB** PSNR-Y (VMAF BD-rate +6.0%). Drone
validation is split: forest qp 42 prefers the narrower band, while canyon/tasmania keep gaining
to strength 5.0 at about 2.2 dB PSNR cost; samtime also plateaus around 5.0 but the knee is 2.5-3.0,
so 2.0-2.5 is the safe range. The custom filter wins
all three drone clips against x265 but still loses samtime. Applying the same band-pass to x265
gives it a larger gain, so VDWT+band against x265+band is about −4%: these are display filters,
not codec improvements, and are excluded from the Pareto. The detailed rows are in
the internal lab notes; the previous-vs-new chart is
`videos/samtime_filter_prev_vs_now.png`.

The filter can also run as a signalled codec feature: `--postfilter auto` picks one `band2`
strength per scene at each intra frame from the scene's nominal qp and mid-band texture
(`T = mean |blur3 − blur_w|` on a half-resolution luma sample), writes it in the frame header in
1/8 units, and holds it until the next intra frame. The fit is
`2.4 + 0.15·(qp − 42) − 0.3·T`, clamped to 0..5 and rounded to half steps, so a qp-42 talking
head gets 2.0-2.5, forest about 1.0, and a qp-58 stream about 4.5. `decode --postfilter auto`
applies the signalled value; a fixed `--postfilter-strength` overrides it. The field is
display-only and never enters prediction or reference pictures. Measured against fixed 2.0 on
five clips, auto gains 8-10% VMAF BD while losing 18-20% PSNR BD; the 70/30 hybrid is +0.56%
overall (+1.51% at ultraheavy) but b720 is −7.5%, so auto stays opt-in and 2.0 remains the safe
fixed strength until the overshoot clamp and saturation guard land. Against unfiltered, fixed
and auto gain +29.0 / +36.5% VMAF BD and +8.0 / +6.4% hybrid; b720 is below unfiltered under the
hybrid even with fixed 2.0 (−9.2%), so that content is better left unfiltered.

#### 10.4.1 Ultraheavy: current decisions

- **Pairs.** Two frames are coded as a B-frame group when the previous P-frame's motion section
  was over 30% of its bytes, or when the camera is static. Otherwise they are two P-frames.
  Groups of 4 remain in the format; since 2026-10-28 fixed-qp ultraheavy chooses them on a still
  camera (below).
- **No cut guard.** A pair is not broken at a scene cut: the cheap B before the cut predicts from
  the next anchor, and the cut frame itself is intra. This is cheaper than the I+P fallback and
  is what the old HEAD did; the guard was removed after the samtime BD improved without a speed cost.
- **B syntax and prediction.** Each direction names its two references by position; vectors are
  scaled from the anchor field and coded as offsets. Modes are past, next, their average, the
  direct fields, or a per-root blend (past/next/both) with bilinear weight interpolation.
- **qp.** The anchor's plus 6 for the kept middle B and plus 9 for the others; a static camera's
  B uses 6, or 8 at anchor qp ≥ 50. Under rate control the group budget and the B share decide
  the final qp within `[anchor + 2, anchor + 6]`.
- **Motion grid.** A three-frame probe halves the grid (32→16) when the field has many single-node
  leaves yet changes smoothly. For ultraheavy the probe always gets the last word; the static
  shortcut is only used when it already chose 16 (below 50 fps). At 60 fps this fixes forest,
  where keeping 32 cost 8% of the bits at qp 50.
- **Search and refinement.** Wide diamond, four refinement passes, except two passes on a static
  camera below 50 fps (same BD, 16% faster on the Pi). No root-row stripes: ultra gained no time
  from them.
- **B-mode selection.** One pooled pass per prediction pair computes the past/next/both errors
  per root, reused by the blend; only the winning candidate is copied. Prediction fell from
  14.6 to 12.3 ms single-thread, 10.6 to 6.7 ms on four.

### 10.5 Static content, repeats and link allocation

The encoder tracks the mean global displacement over the frame corners, smoothed. Below
1.5 px/frame the content counts as static:

- the static mesh probe chooses 16 pixels below 50 fps and keeps 32 at 60 — except at a fixed qp
  when every probed pair moves less than 1 px/frame (talking heads probe at 0.2–0.5; an FPV
  opening such as forest reads 0.0 then 1.4): 3–12% fewer bits on the 60 fps talking head
  (realtime −8 to −12%, light −4 to −7%, medium and heavy −4 to −6%, ultraheavy −3%);
- medium, heavy and ultraheavy run the adaptive probe when the shortcut kept 32 (medium and heavy
  since 2026-10-28: forest −3.6/−3.8%, canyon medium −1.6%, nothing worse); light and realtime do
  not, as the finer grid costs them 36–46% more encode time on forest;
- ultraheavy pairs static frames as B even when the motion share is small; the B qp comes from
  its share of the group budget;
- identical source frames become 7-byte Repeat frames once the picture is 45 dB, so a still
  video at 1.5 Mbit/s drops to about 148 kbit/s at 45 dB;
- the transmit queue may borrow one delay limit of credit, and a 48 dB quality ceiling with a
  2 dB reference gate stops quiet passages from soaking up the link (`--quality-cap`).

A still-camera filter skip was tried and removed: it gained on a talking head but lost 3–4 points
on the slow zoom. The four-level transform default was tried and reverted (2–4% regression at
qp 58). Both are in the internal lab notes.

### 10.6 Round history

Current code is the sum of: the 9/7 wavelet and trained contexts; the motion quadtree and
overlapped prediction; the loop filter; the 2026-10-24 static-content round (Repeats, link
credit, quality cap, static mesh probe, B pairs); the speed round (overlapped always, realtime
GME, loop-fit subsampling, flat guide opt-in); the parallel round (stripes, rate-aware loop
filter, high-rate search shortcuts, B-mode pass, compression recovery, two-pass static
refinement, zero-copy output, static global-motion zero, chroma step 0.85, no half-pel on a
static camera, a one-shot −2 opening keyframe when frame 1 starts a new scene, final intra
frames in a known-length stream coarsened to `qp`, and no half-pel at ultraheavy on a truly still
camera); and the 2026-10-26 encoder-only perceptual round (causal frame AQ and a sharpened
in-loop-filter target, six-point hybrid **+5.57%**, no syntax change). Per-round tables, the old
tier names and every rejected parameter are in
the internal lab notes. The complete accepted/rejected/open experiment table is
the internal lab notes; the charts in `videos/` are the current comparison.

### 10.7 Parallelism, Pi and rejected variants

The encoder and decoder own a fixed thread pool; every stage is data-parallel and the stream is
thread-count independent. On the final Pi build, realtime takes 52.9 / 30.0 / 19.9 ms at
1 / 2 / 4 threads (2.66× at four) and ultraheavy 533 / 307 / 208 ms (2.57×). The remainder is
the frame dependency chain, the partly serial global-motion fit and rate control, and bitstream
assembly, not an unpainted stage. The search is a root-row wavefront; refine is a leaf wavefront.
Stripes are used at light/medium/heavy and on a static camera; ultraheavy keeps its single
wavefront. The Pi (in-order A55) is the speed reference for each change; the x86 laptop supplies
the final published curves. **Static global-motion zero** is the compression win of this round: when a near-zero affine fit
is detected, using exact zero instead of the compromised model makes background leaves reach
(0, 0) and improves samtime BD by 4–10 points (§10.4). **A one-shot −2 opening keyframe** (instead of −8) is a free win when frame 0 is unreferenced
because frame 1 cuts; it applies to every tier and accounts for about 2 BD points on samtime.
The earlier −4 variant was already better than no override; −2 gains another ~0.5–0.7. Removing
the boost entirely (qp + 0) lost 3.9 dB on the displayed first frame and was rejected.
**A final-intra coarsening** (intra frames in the last two frames of a known-length stream go to
`qp` instead of `qp − 7`) is the largest ultraheavy win of this round: a720 ultra q42 drops from
558.3 to 497.9 kbps for a 0.41 dB drop on one displayed frame and the final P-frame, moving
samtime ultraheavy BD from −11.2 to −8.7.
**Zero-copy output** is the speed win: the CLI
processes `Encoder::recon()` (or `group_recon()`) before the next encode instead of cloning a
1.4 MB picture per display frame, which drops the Pi realtime encode from 22.5 to 20.0 ms with a
byte-identical stream and helps every tier.

Measured and not kept: halves of the B search on two 2-thread pools (10% slower on the Pi); two
full pools oversubscribing four cores (3% slower); forcing the 128-bit/one-LK realtime search
at every rate (3–5 BD points on the charts for 7% of the time); snapping small static offsets
to the global prediction (2.5–90% more bits, 0.25–0.7 dB lower PSNR); cheaper B-pair QP offsets
(+6→+3 and a −1 anchor: +0.2–0.3 dB PSNR but 0.7–0.8 BD points worse); dividing pair-anchor
camera speed by the group length (b720 ultra +2.1% bits, BD 16.42→16.80); preserving searched vectors
at B-blend root boundaries (no measurable change); the reported ultraheavy “mesh probe override”
is not present in the CLI, which keeps the static 16-pixel grid; one refine wavefront for both B directions
(bit-identical, 0.8% faster on a720, neutral on forest); root stripes for ultraheavy (no time,
−0.5 BD samtime); coarse search at light (1.3 BD drone); flat guide as the realtime default
(0.2–0.9 dB); mesh 16 for heavy/ultraheavy (mixed); warp prediction, 5/3 wavelet, coarser chroma,
32/128-pixel block roots (no net gain). A 30-minute no-fan soak has not been run. Every accepted, rejected and open candidate is listed
in the internal lab notes.

**Next.** The realtime gap is 20.0→16.7 ms (17.62 with `--no-guide`). The remaining ideas are an
adaptive flat-guide decision for low-detail/static sequences and less search/transform work;
zero-copy output is already in. AVX2/NEON is out of scope.

## 11. What is new in VDWT

Prior art: Dirac/VC-2 (wavelet + OBMC block MC), Snow, JPEG XS (intra and line-based), HTJ2K (intra).

VDWT combines:
- a **block-free inter pipeline**: global affine motion plus a smooth mesh warp feeding a whole-image HTJ2K DWT, with soft intra/inter weighting instead of block modes;
- an **HT-structured adaptive range coder with cross-scale parent contexts**, tuned for low-rate P residuals;
- **histogram-shift rate estimation**, giving one-pass link-following CBR, now with link credit and a
  distortion ceiling estimated from the same histograms;
- **content-adaptive static-scene coding**: B-frame pairs on a still camera; identical frames
  as Repeat frames under rate control;
- **zero-region skipping** in reconstruction;
- **coarse-to-fine packetization**, so loss turns into blur instead of blocks;
- **intra refresh instead of I-frames** in the `rt` codec;
- a design built from the start around int16 NEON on in-order ARM cores.

## 12. Open questions

1. **720p60 realtime on the A55.** Current 20.0 ms/frame at 600 kbit/s (17.62 with
   `--no-guide`). The measured gap is the guide transform, quantize/reconstruct and the search;
   an adaptive flat-guide decision and less search work are the next candidates.
2. **Ultraheavy search time.** Refinement is still over half of the 216 ms static encode and
   scales only ~2.5× on four cores. The wavefront is dependency-bound; less search work, not
   another pool, is the lever. Two passes already apply to static cameras below 50 fps.
3. **GPU and decode.** The decoder is not the current bottleneck at 720p; the GPU is unused.
4. **Loss and 10-bit.** The `rt` codec has the loss path; the main codec has no FEC/feedback and
   no 10-bit profile yet.
5. **Content.** Real 720p60 raw footage from the target camera would still improve tuning; the
   current charts use samtime (a720/b720) and the three drone clips.
