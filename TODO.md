# TODO — known follow-ups

Codec bitrate-efficiency R&D (new coding tools, ordered queue with expected gains) is tracked in
the internal lab notes; this file keeps release blockers and product follow-ups.

## 0. Metric standard

Two results, each against a matched reference (2026-10-28):

- **PSNR-Y (primary):** VDWT defaults (`--tune psnr`) against x265 `--tune psnr`.
- **Hybrid 70% VMAF + 30% YUV-PSNR (perceptual):** VDWT `--tune perceptual` against x265 at its
  default tuning (psy-rd and AQ on): `hybrid_bench.py CLIPS OUT perceptual`.

The hybrid files made before 2026-10-28 (`hybrid_samtime.txt`, `hybrid_drone.txt` and kin) scored
x265 `--tune psnr` on VMAF — x265 with its perceptual tools off — and VDWT with sharpening on, so
they flatter VDWT; fair rows are in `hybrid_fair.txt`. Do not mix hybrid and PSNR-Y numbers.

Measured 2026-10-28 (`hybrid_fair.txt`, x265 medium/veryslow; hybrid saving, positive = VDWT needs
fewer bits), the same VDWT `--tune perceptual` rows against the fair and the old reference:

| | samtime fair / old ref | drone fair / old ref |
|---|---|---|
| realtime | −26.0% / −30.3% | +1.4% / −0.7% |
| light | −19.7% / −23.5% | +4.4% / +2.5% |
| medium | −15.0% / −18.8% | +12.9% / +11.2% |
| heavy | −15.4% / −19.2% | +13.6% / +11.9% |
| ultraheavy | **−9.6%** / −13.1% | **+19.4%** / +17.8% |
| x265 veryslow | +8.5% | +17.0% |

x265's perceptual tuning gains it about 3.8 points of hybrid on samtime and loses it about 1.7 on
drone, so the old reference was too easy on samtime and slightly too hard on drone.

## 1. Held-out context-table retraining (A21) — done 2026-10-25

The earlier tables were trained on a corpus that contained the evaluation clips; the published
Pareto was therefore provisional. The retrain used a disjoint corpus (POV bike, coastline, three
Madeira segments; 260 frames, 88,846,227 symbols; `bench/contexts/build_corpus.sh`) and shipped.

- [x] Build a disjoint training corpus: not a720/b720/canyon/forest/tasmania.
- [x] Run `vdwt train-contexts` on it at qp 24–56, with P-frame and intra-only runs.
- [x] Re-run the full samtime + drone Pareto on the held-out tables.
- [x] Confirm the gain survives: held-out is 0.78% worse than the in-domain tables (leakage) but
      4.63% below untrained CDFs at ultraheavy, so 89% of the retrain gain is real.
- [x] Record the corpus, command, symbol counts and the before/after BD in EXPERIMENTS.md.
- [x] Update DESIGN.md §5.2; the shipped tables and Pareto now use the held-out corpus.

Follow-up, not blocking: add studio/talking-head content and retrain to recover the ~1% samtime
gap; the current corpus is natural footage plus one POV clip.

## 2. A22 chroma step — resolved (reverted)

A22 was reverted to 0.95 after the hybrid measurement: **−21.06%** with 0.95 versus
**−21.30%** with 1.2 on samtime ultraheavy. The PSNR-Y gain was mostly hidden chroma loss.
No further review is needed unless chroma is reconsidered.

## 3. A19 first-frame quality — review before shipping

A19 codes the unreferenced hard-cut opening frame at `qp`, saving bits but dropping b720 frame 0
by ~3.9 dB. Re-check whether that trade is acceptable for a first impression; if not, revert A19
to A8's `−2` for the opening frame.

## 4. Pi cross-check — done 2026-10-25

The current build was measured on the Orange Pi 3B (RK3566, 4× A55, 1.8 GHz, fan, 720p, 4
threads):

| workload | result |
|---|---|
| realtime CBR 600k, 120 frames | 19.8 ms/frame (51 fps), 42.64 dB, 559.3 kbit/s |
| realtime CBR 600k, `--no-guide` | 17.4 ms/frame (57 fps), 41.99 dB, 555.2 kbit/s |
| realtime fixed q42, 120 frames | 38.4 ms/frame, 40.40 dB, 345.6 kbit/s |
| heavy a720 q42, 20 | 130.8 ms, 37.63 dB, 576.7 kbit/s |
| ultraheavy a720 q42, 20 | 204.8 ms, 37.04 dB, 483.0 kbit/s |
| ultraheavy forest q42, 20 | 409.9 ms, 28.84 dB, 9839.2 kbit/s |
| ultraheavy b720 q42, 20 | 266.7 ms, 36.52 dB, 763.3 kbit/s |

The A15–A17 code paths are present and there is no speed regression: CBR realtime is 19.8 ms
(was 19.9), `--no-guide` 17.4 (was 17.55), heavy 130.8 (was 135.2), ultra a720 204.8 (was 208.5),
forest 409.9 (was 409.5). The fixed-qp realtime row is slower because CBR skips the in-loop filter
and repeats 67 static frames; it is not the same workload. Bitrate/PSNR now include A19, the
final-intra change, chroma 0.95, the held-out tables and the VERSION 9 syntax. Thermal stress was not re-run; the old
62.2 °C result still stands unless the cooling changes.

- [x] Realtime fixed-qp and CBR.
- [x] Heavy/ultraheavy on a720, forest and b720.
- [x] Check A15–A17: A15 is the CBR loop-filter skip; the held-shot GME bypass is present and the
      CBR global-motion stage is 1.7 vs 2.8 ms at fixed qp.
- [ ] Re-run the thermal stress only if the fan or enclosure changes.
- [x] Integrate the external vector kernels: AVX2 quantizer/dequantizer and mesh gradients on
      both ISAs, bit-exact. The NEON quantizer measured 0.72× on the A55 (slower than the
      auto-vectorized scalar loop) and is not dispatched; the NEON mesh gradients are used.
- [x] Re-check the Pi after the VERSION 9 syntax changes: `selftest --isa all` passes, an
      encode/decode MD5 matches, and CBR/ultraheavy stay within noise of the old binary
      (EXPERIMENTS.md, 2026-10-26).

## 5. Intra is the remaining structural gap

The 2026-10-25 sweep (EXPERIMENTS.md) found no encoder-side coding change that improves the
hybrid BD. On b720 the I-frame is about 2× x265's size at equal PSNR, while VDWT's P/B frames
are about half of x265's; the gap is directional intra prediction. The block-free constraint rules
out block/directional block intra, so the next serious option is a block-free directional or
wavelet-packet intra transform (whole-subband or orientation-field based), not more context
tuning. Carrying band and motion CDF state across frames measured +0.65 hybrid points but is
unsafe with dropped bands/skipped frames unless every unsent band's state is snapshotted; not
worth the complexity for that gain.

The public block-directional oracle (2026-10-26) priced the prize at up to 21.9% of the whole
stream on fourpeople and 9.1% on aerial, but it predicts from full-resolution source neighbours
that a decoder does not have for free. The block-free LL pyramid was then measured: at 1/8 it is
**5.4% larger and 0.24 dB worse** than the normal intra frame, and every scale from 1/32 to 1/4
loses. A simple LL-conditioned scheme is dead; the only remaining intra option is a true
edge-directed lifting transform, which would have to cut the residual by >12% just to break even.
Do not start T2b on the block numbers.

The cheap inter shape was then tested: a global horizontal shear of the luma residual, signalled
in a few bits, needs only row offsets rather than a per-pixel crossbar. It reaches **−1.32%** on
a720 with per-frame oracle selection (−0.89% fixed) but is worse on forest and canyon, and the
useful slope is not predicted by `pred` orientation. That is far below the 5% hardware kill
threshold, so the MUX-heavy dense field is not justified either. The structural directional
search is closed on the current evidence; VERSION 9 stays frozen and the remaining work is RTL
synthesis and product validation. Details in TASKS.md and EXPERIMENTS.md 2026-10-26.

## 6. IMPORTANT — the display post-filter auto rule is mis-calibrated

> **Status: open quality bug.** The filter is display-only, off by default and outside the codec
> Pareto. `auto` is opt-in and the tasmania/canyon gains are real, but the strength rule was
> fitted on five private clips and picks the wrong strength on public content. Do **not** promote
> auto, change the default, or claim the VMAF gains are general until the rule is refit and gated.

### What the filter is for

`--sharpen` and `--postfilter band/band2` are display post-filters, not coding gains: the same
filter improves x265 too, and the fair VDWT+filter vs x265+filter comparison is worse than the
unfiltered one. The filter's value is the detail the **codec destroyed**: motion-blurred drone
frames gain, a sharp tripod shot gains little VMAF and only spends PSNR. `band2` widens the band
to `[1 2 3 2 1]/9`; it beat `band` 2.0 by +1.42 VMAF at −0.84 dB PSNR-Y on samtime. Every
filter number stays outside the codec Pareto.

### Findings from the public diagnostic (2026-10-25)

1. **The texture term aliases.** `auto_strength` computes `T = mean|blur3 − blur_w|` over every
   other pixel. Dense high-frequency detail can cancel at those sample points, so detailed
   content gets a *low* T and smooth ramps a *high* T. The rule then subtracts `0.3·T`, giving
   detailed content **more** sharpening — the opposite of the comment and the opposite of the
   measured optimum. Public examples: parkrun/shields T≈0.40 → auto 3.5 (optimum 1.0); fourpeople
   T=1.44 → auto 3.0 (optimum 4.5).
2. **The gain curve has a content-specific peak, and auto sails past it.** At qp 42, parkrun peaks
   at s=1.0 and is already negative by s=3.0 while auto picks 2.5; fourpeople keeps gaining to
   4.5 while auto picks 2.0. Auto was past the peak in 8 of 15 public points.
3. **The physical failure is overshoot/clipping.** At s=4.5, 14–47% of pixels leave the source's
   local range; the fraction that hard-clips at 0/255 is **6.18% on parkrun (snow)** vs **0.59%
   on aerial**. This is the local overshoot clamp and saturation guard that were already open.
4. **It is motion, not “drones”.** Synthetic clips made from real aerial footage (mean frame
   difference 9.6–10.8) gain +6.4 to +11.0 VMAF; clips made by flying over a still image are
   predicted almost perfectly (31 kbit/s, 44.7 dB) and the filter buys ≈0 because there is
   nothing left to restore.
5. **The fair comparison is confirmed.** At s=4.5 the same filter gives x265 +8.94 VMAF vs
   VDWT +7.20, so the “VDWT+band vs x265 stock” framing flatters VDWT; the fair both-filtered
   result is 2–3 VMAF points worse.
6. **Three-point BD is unusable on the PSNR axis for filter sweeps.** The curve is flat enough
   that it reported ±85%; the earlier “parkrun −68%” was that artifact. Use per-strength point
   deltas or at least five rates for filter decisions; keep BD only for the codec Pareto.

### Our cross-checks and corrections

- The Rust metric does **not** blur a decimated image: it computes full-res 3×3 and 5×5 blurs and
  samples every other pixel. The aliasing critique stands, but the effect is content-specific,
  not universal. On our five clips full-res and sampled T are nearly identical:
  tasmania 0.34/0.33, canyon 0.53/0.52, forest 3.95/3.95, a720 0.77/0.76, b720 0.35/0.34.
- **b720 is not mispicked for VMAF.** Unfiltered VMAF 84.64 → band2 2.5 = 96.11, and 2.5 is the
  VMAF peak (3.0 = 95.98, 3.5 = 95.42); auto already picks 2.5. Its hybrid loss is the PSNR bill
  (35.98 → 29.84 dB). Do not retune auto to chase b720’s hybrid number; the correct action there
  is “leave the filter off under the hybrid.”
- Measured on b720/forest at qp 42: hard-clip **1.75%/2.52%** at s=2.5/4.5 (b720) and
  **1.87%/3.01%** (forest); pixels outside the source 3×3 range are **51–57%** (b720) and
  **30–39%** (forest). The guard has real work to do.
- The recon/source mid-band ratio `T_rec/T_src` separates the cases we care about: tasmania 0.60
  and canyon 0.53 (detail destroyed → boost), forest/a720 0.83, b720 0.88 (little lost → little
  gain). It is a better feature than T alone.
- The public report’s suggested recon-based rule is feasible: `header.write()` already runs after
  `self.reconstruct()`, so an intra frame can choose its strength from the picture that will
  actually be filtered.

### Problems to fix

- [ ] **Replace the sampled T with a full-res feature** (or a robust gradient/Laplacian energy)
      that does not alias dense detail; keep or invert the term only after fitting.
- [ ] **Add the reconstruction-loss term** `loss = 1 − T_rec/T_src`; raise strength when the codec
      lost mid-band detail, lower it when the reconstruction is already sharp.
- [ ] **Add the clipping/overshoot cap**: apply the candidate strength to the reconstruction,
      count hard clips and local overshoot, and cap the strength (target hard clips ≤0.5%,
      overshoot bounded). This is the concrete ringing fix.
- [ ] **Move the auto decision after reconstruction** for intra frames and keep the scene value
      in the header; the header is written after reconstruct, so the ordering is safe.
- [ ] **Refit on both corpora** — public clips for shape, private samtime/drone for the gate —
      and record the fit in EXPERIMENTS.md.
- [ ] **Gate on point deltas, not 3-point BD**: parkrun/fourpeople/aerial/shields/stockholm must
      improve; our five clips must not regress, especially tasmania/canyon and the b720 hybrid
      “off” rule.
- [ ] **Do not change the default or promote auto** until the gate passes; auto stays opt-in.
- [ ] Re-run the auto-vs-fixed and on/off tables after the new rule; do not mix numbers against
      different references, the hybrid BD is not additive.

### Proposed rule shape (to be fitted, not guessed)

```
base = 2.0 + ramp(qp)                  # low rate needs more recovery
loss = 1 − T_rec / T_src               # high when motion destroyed detail
s    = base + a·loss − b·T_src_full    # detailed source gets less, lossy recon gets more
s    = max s with clip(s) ≤ threshold  # hard-clip/overshoot cap
```

### Turn-on policy (unchanged until the gate passes)

Auto remains fully supported and signalled; the tasmania/canyon gains are kept. Under **VMAF**
the filter is on and auto is the stronger mode. Under the **70/30 hybrid** use auto for
natural/drone footage (tasmania +22.2%, canyon +14.8% vs off), either mode for a720/forest, and
leave near-saturated **b720 unfiltered** (fixed −9.2%, auto −20.9% vs off). The codec default
remains unfiltered.

- [x] Post-filter on/off benchmark + Pareto (unfiltered vs fixed 2.0 vs auto) per clip and tier,
      so "turn it on" is a measured decision, not a rule of thumb.

### Division of work

The public-corpus agent owns the recon-based rule and the parkrun/fourpeople/aerial validation.
The private five-clip gate (tasmania/canyon gains, b720 off, full hybrid table) belongs here; only
merge after both pass. Filter work is display-quality work, not the codec-efficiency queue in
the internal lab notes.


## 7. Miscellaneous open items

See EXPERIMENTS.md `Open, not applied` (O6–O19), especially loop-filter coordinate descent (O10)
and quadtree pruning (O9).

## 8. Perceptual encoder defaults (2026-10-26)

**Superseded 2026-10-28:** frame AQ stays default; the sharpened target and the finest tilt
moved behind `--tune perceptual` because they cost PSNR-Y (gen_b 5.5–13.4%). Original note:
causal frame AQ (`--frame-aq 2.0`) and a sharpened in-loop-filter fit target
(`--loop-sharpen 0.4`) were made defaults. They are encoder-only and VERSION 9 streams still decode
on old builds; `--frame-aq 0 --loop-sharpen 0` restores the previous stream exactly. Six-point
hybrid BD is +5.57% (AQ alone +0.84%), VMAF +9.49%, YUV-PSNR -3.58%. The normal-VMAF gain is
larger than the NEG gain (+5.39% VMAF, +3.02% hybrid), and b720 loses 5.57% at medium and 0.68% at ultraheavy under NEG while
gaining under normal VMAF; this is documented rather than hidden.

- [ ] If a validated content gate appears, use it to leave very smooth, near-saturated content
      (b720) on the MSE filter target. Do not add an unvalidated heuristic like the display
      post-filter auto rule.
- [ ] Re-run the CBR tables after the next rate-control change; AQ is inactive under rate control
      and the filter's CBR gain is smaller than the fixed-qp Pareto.
