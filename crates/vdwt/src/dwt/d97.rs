//! The CDF 9/7 wavelet in exactly invertible integer lifting form (DESIGN.md §4.4).
//!
//! The four lifting steps of the 9/7 (ISO/IEC 15444-1 Annex F, irreversible), with Q14
//! coefficients:
//!
//! ```text
//! d[n] += f(α, s[n],   s[n+1])      f(c, a, b) = ⌊(c·(a + b) + 2^13) / 2^14⌋
//! s[n] += f(β, d[n−1], d[n])
//! d[n] += f(γ, s[n],   s[n+1])
//! s[n] += f(δ, d[n−1], d[n])
//! ```
//!
//! Samples are `i16` with wrapping arithmetic, and each step adds a function of the other channel
//! only, so the inverse subtracts the same values in reverse order and reconstruction is exact.
//! The final scaling by K is left out (it is not invertible in integers); the step weights absorb
//! it. Borders use whole-sample symmetric extension, as the 5/3 does.
//!
//! Samples enter scaled by `2^FRAC_BITS` and leave rounded back. Integer lifting rounds at every
//! step, and on small values (P-frame residuals above all) that rounding is noise on the scale of
//! a quantizer step: the fraction bits cut it by 8×.
//!
//! Each level is one streaming pass: rows enter an eight-row ring that runs all four vertical
//! lifting steps a few rows behind the input, and each finished row is split and lifted
//! horizontally straight into place (the inverse mirrors this). Everything is one element-wise
//! kernel (`lift`) that vectorizes. On a pool, a level runs as bands of rows, each starting a few
//! rows early so that its first rows match a pass from the top. The Mallat layout is the 5/3's.

use std::ops::Range;

use crate::plane::SharedRows;
use crate::pool::Pool;
use crate::spare::{give, take};
use crate::{Isa, Plane};

use super::{BlockMap, Geometry, OUT_LOG2};

/// Fraction bits the transform carries.
pub const FRAC_BITS: u32 = 3;
const Q: u32 = 14;
/// α, β, γ, δ in Q14: −1.586134342, −0.052980118, 0.882911076, 0.443506852.
const STEPS: [i32; 4] = [-25987, -868, 14466, 7266];
/// The rt codec's faster lifting (DESIGN.md §10.3): the same steps in Q15 on 16-bit lanes,
/// `g(c, a, b) = ⌊(c'·sat(a + b) + 2^14) / 2^15⌋ − [c < −2^15]·sat(a + b)` with `c' = c + 2^15`
/// for α (below −1) and `c` otherwise: one rounding multiply-high per 8 or 16 lanes (`SQRDMULH`,
/// `PMULHRSW`) instead of widening to 32 bits. The sum saturates at 16 bits.
const FAST_STEPS: [i32; 4] = [-51975, -1736, 28931, 14533];

/// `g(c, a, b)` (see [`FAST_STEPS`]), wrapped to 16 bits.
#[inline(always)]
pub(super) fn g(c: i32, a: i16, b: i16) -> i16 {
    let s = a.saturating_add(b);
    let (whole, frac) = fast_parts(c);
    let m = ((i32::from(s) * frac + (1 << 14)) >> 15) as i16;
    if whole { m.wrapping_sub(s) } else { m }
}

/// A fast step's whole part (−1 or 0) as a flag, and its Q15 fraction.
#[inline(always)]
pub(super) fn fast_parts(c: i32) -> (bool, i32) {
    if c < -32768 {
        (true, c + 32768)
    } else {
        (false, c)
    }
}

/// `f(c, a, b)`, wrapped to 16 bits.
#[inline(always)]
fn f(c: i32, a: i16, b: i16) -> i16 {
    ((c * (i32::from(a) + i32::from(b)) + (1 << (Q - 1))) >> Q) as i16
}

/// `dst[i] ±= f(c, a[i], b[i])` over equal-length slices.
type LiftFn = fn(&mut [i16], &[i16], &[i16], i32, bool);

#[inline(always)]
fn lift_any(dst: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    let n = dst.len();
    let (a, b) = (&a[..n], &b[..n]);
    if sub {
        for i in 0..n {
            dst[i] = dst[i].wrapping_sub(f(c, a[i], b[i]));
        }
    } else {
        for i in 0..n {
            dst[i] = dst[i].wrapping_add(f(c, a[i], b[i]));
        }
    }
}

fn lift_portable(dst: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    lift_any(dst, a, b, c, sub);
}

/// `dst[i] ±= g(c, a[i], b[i])` (the fast steps).
pub(super) fn lift_fast_portable(dst: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    let n = dst.len();
    let (a, b) = (&a[..n], &b[..n]);
    for i in 0..n {
        let t = g(c, a[i], b[i]);
        dst[i] = if sub {
            dst[i].wrapping_sub(t)
        } else {
            dst[i].wrapping_add(t)
        };
    }
}

/// The scalar tail of a SIMD lifting kernel.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
pub(super) fn lift_tail(dst: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    lift_any(dst, a, b, c, sub);
}

fn split_portable(src: &[i16], even: &mut [i16], odd: &mut [i16]) {
    for (o, v) in even.iter_mut().zip(src.iter().step_by(2)) {
        *o = *v;
    }
    for (o, v) in odd.iter_mut().zip(src.iter().skip(1).step_by(2)) {
        *o = *v;
    }
}

fn merge_portable(even: &[i16], odd: &[i16], dst: &mut [i16]) {
    for (o, v) in dst.iter_mut().step_by(2).zip(even) {
        *o = *v;
    }
    for (o, v) in dst.iter_mut().skip(1).step_by(2).zip(odd) {
        *o = *v;
    }
}

/// The kernels for one ISA; all compute identical results.
#[derive(Clone, Copy)]
struct Kernels {
    lift: LiftFn,
    /// α, β, γ, δ as `lift` takes them.
    steps: [i32; 4],
    /// A row into its even and odd samples.
    split: fn(&[i16], &mut [i16], &mut [i16]),
    /// The inverse of `split`.
    merge: fn(&[i16], &[i16], &mut [i16]),
}

const PORTABLE: Kernels = Kernels {
    lift: lift_portable,
    steps: STEPS,
    split: split_portable,
    merge: merge_portable,
};

fn kernels_for(isa: Isa, fast: bool) -> Option<Kernels> {
    let k = kernels_exact(isa)?;
    if !fast {
        return Some(k);
    }
    let lift: LiftFn = match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => super::avx2::lift97_fast,
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        Isa::Neon => super::neon::lift97_fast,
        _ => lift_fast_portable,
    };
    Some(Kernels {
        lift,
        steps: FAST_STEPS,
        ..k
    })
}

fn kernels_exact(isa: Isa) -> Option<Kernels> {
    match isa {
        Isa::Scalar => Some(PORTABLE),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => crate::cpu::avx2_detected().then_some(Kernels {
            lift: super::avx2::lift97,
            steps: STEPS,
            split: super::avx2::split,
            merge: super::avx2::merge,
        }),
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        Isa::Neon => Some(Kernels {
            lift: super::neon::lift97,
            steps: STEPS,
            split: super::neon::split,
            merge: super::neon::merge,
        }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// `d[i] ±= f(c, s[i], s[i+1])`, with `s[ns]` mirrored to `s[ns−1]`.
#[inline]
fn lift_d(lift: LiftFn, d: &mut [i16], s: &[i16], c: i32, sub: bool) {
    let nd = d.len();
    if nd == 0 {
        return;
    }
    let m = nd.min(s.len() - 1);
    lift(&mut d[..m], &s[..m], &s[1..m + 1], c, sub);
    if m < nd {
        lift(&mut d[m..], &s[m..], &s[m..], c, sub);
    }
}

/// `s[i] ±= f(c, d[i−1], d[i])`, with `d[−1]` mirrored to `d[0]` and `d[nd]` to `d[nd−1]`.
#[inline]
fn lift_s(lift: LiftFn, s: &mut [i16], d: &[i16], c: i32, sub: bool) {
    let (ns, nd) = (s.len(), d.len());
    if nd == 0 {
        return;
    }
    lift(&mut s[..1], &d[..1], &d[..1], c, sub);
    let m = ns.min(nd);
    if m > 1 {
        lift(&mut s[1..m], &d[..m - 1], &d[1..m], c, sub);
    }
    if ns > nd {
        lift(&mut s[nd..], &d[nd - 1..], &d[nd - 1..], c, sub);
    }
}

/// Forward lifting of one row already split into `s` and `d`.
fn analyze(k: &Kernels, s: &mut [i16], d: &mut [i16]) {
    for pair in k.steps.chunks(2) {
        lift_d(k.lift, d, s, pair[0], false);
        lift_s(k.lift, s, d, pair[1], false);
    }
}

/// Inverse of [`analyze`].
fn synthesize(k: &Kernels, s: &mut [i16], d: &mut [i16]) {
    for pair in k.steps.chunks(2).rev() {
        lift_s(k.lift, s, d, pair[1], true);
        lift_d(k.lift, d, s, pair[0], true);
    }
}

/// Rows at or above this many samples per level are split across the pool.
const PARALLEL_SAMPLES: usize = 1 << 15;
/// Lifting pairs of extra rows a band computes before its first output pair: from there on its
/// rows match a pass from the top (each pair depends on the two before it, and one more of margin).
const WARMUP: usize = 3;

/// Runs level `l` of every plane in `planes` (level count and geometry) that has one, as one
/// batch on the pool: `f(plane, pairs)` for bands of its lifting pairs. Planes of at least
/// [`PARALLEL_SAMPLES`] split into bands in proportion to their share of the batch, about two per
/// thread in all (each costs [`WARMUP`] pairs of extra work); smaller ones run whole. Earlier
/// planes' bands come first. Each band writes only its own rows.
fn run_level(
    pool: Option<&Pool>,
    l: usize,
    planes: &[(usize, Geometry)],
    f: &(dyn Fn(usize, Range<usize>) + Sync),
) {
    let size = |geo: &Geometry| geo.ll_size(l - 1);
    let present = || {
        planes
            .iter()
            .enumerate()
            .filter(|(_, (levels, _))| l <= *levels)
    };
    let total: usize = present().map(|(_, (_, g))| size(g).0 * size(g).1).sum();
    let mut tasks: Vec<(usize, Range<usize>)> = Vec::new();
    for (i, (_, geo)) in present() {
        let (w, h) = size(geo);
        let n = h.div_ceil(2);
        let bands = match pool {
            Some(p) if w * h >= PARALLEL_SAMPLES => (2 * p.threads() * w * h)
                .div_ceil(total)
                .min(n / (4 * WARMUP))
                .max(1),
            _ => 1,
        };
        tasks.extend((0..bands).map(|t| (i, t * n / bands..(t + 1) * n / bands)));
    }
    match pool {
        Some(p) if tasks.len() > 1 => p.run(tasks.len(), &|t| f(tasks[t].0, tasks[t].1.clone())),
        _ => tasks.into_iter().for_each(|(i, pairs)| f(i, pairs)),
    }
}

/// Eight rows of one level, the vertical lifting window: low row `n` in slot `2(n mod 4)`, high
/// row `n` in the slot after it. It lifts with `lift`, subtracting when `sub` (the inverse).
struct Ring {
    buf: Vec<i16>,
    w: usize,
    lift: LiftFn,
    steps: [i32; 4],
    sub: bool,
}

impl Ring {
    fn new(w: usize, k: &Kernels, sub: bool) -> Self {
        Self {
            buf: take(8 * w),
            w,
            lift: k.lift,
            steps: k.steps,
            sub,
        }
    }

    fn slot(n: usize, high: bool) -> usize {
        2 * (n % 4) + usize::from(high)
    }

    fn row(&mut self, n: usize, high: bool) -> &mut [i16] {
        let at = Self::slot(n, high) * self.w;
        &mut self.buf[at..at + self.w]
    }

    /// `dst[n] ±= f(c, src[a], src[b])`, `dst` in the half `high`, the sources in the other.
    fn lift(&mut self, high: bool, n: usize, a: usize, b: usize, c: i32) {
        let w = self.w;
        let (d, sa, sb) = (
            Self::slot(n, high),
            Self::slot(a, !high),
            Self::slot(b, !high),
        );
        // The destination's half differs from the sources', so its slot does too.
        let ptr = self.buf.as_mut_ptr();
        // SAFETY: the three slots are in bounds, and `d` is neither `sa` nor `sb`.
        let (dst, x, y) = unsafe {
            (
                std::slice::from_raw_parts_mut(ptr.add(d * w), w),
                std::slice::from_raw_parts(ptr.add(sa * w), w),
                std::slice::from_raw_parts(ptr.add(sb * w), w),
            )
        };
        (self.lift)(dst, x, y, c, self.sub);
    }
}

// Rings and rows come from the thread's spares (`crate::spare`).
impl Drop for Ring {
    fn drop(&mut self) {
        give(std::mem::take(&mut self.buf));
    }
}

/// Forward vertical lifting of output pairs `pairs` of an `h`-row level, rolled through a
/// [`Ring`]: `load(y, row)` fills input row `y`, `emit(n, high, row)` takes finished low or high
/// row `n`.
fn forward_rows(
    ring: &mut Ring,
    h: usize,
    pairs: Range<usize>,
    load: &mut dyn FnMut(usize, &mut [i16]),
    emit: &mut dyn FnMut(usize, bool, &mut [i16]),
) {
    let (ns, nd) = (h.div_ceil(2), h / 2);
    let [alpha, beta, gamma, delta] = ring.steps;
    let t0 = pairs.start.saturating_sub(WARMUP);
    load(2 * t0, ring.row(t0, false));
    // Step t: α on high row t, β on low row t, γ on high row t − 1, δ on low row t − 1, which
    // finishes pair t − 1.
    for t in t0..=pairs.end.min(ns) {
        if t + 1 < ns {
            load(2 * t + 2, ring.row(t + 1, false));
        }
        if t < nd {
            load(2 * t + 1, ring.row(t, true));
            ring.lift(true, t, t, (t + 1).min(ns - 1), alpha);
        }
        if nd == 0 {
            // A single row: nothing to lift.
            if t < ns && pairs.contains(&t) {
                emit(t, false, ring.row(t, false));
            }
            continue;
        }
        if t < ns {
            ring.lift(false, t, t.saturating_sub(1), t.min(nd - 1), beta);
        }
        if t == 0 {
            continue;
        }
        let n = t - 1;
        if n < nd {
            ring.lift(true, n, n, t.min(ns - 1), gamma);
        }
        ring.lift(false, n, n.saturating_sub(1), n.min(nd - 1), delta);
        if n >= pairs.start {
            emit(n, false, ring.row(n, false));
            if n < nd {
                emit(n, true, ring.row(n, true));
            }
        }
    }
}

/// Inverse of [`forward_rows`]: `load(n, high, row)` fills low or high row `n`, `emit(y, row)`
/// takes finished output row `y`, for output pairs `pairs`.
fn inverse_rows(
    ring: &mut Ring,
    h: usize,
    pairs: Range<usize>,
    load: &mut dyn FnMut(usize, bool, &mut [i16]),
    emit: &mut dyn FnMut(usize, &mut [i16]),
) {
    let (ns, nd) = (h.div_ceil(2), h / 2);
    let [alpha, beta, gamma, delta] = ring.steps;
    if nd == 0 {
        for n in pairs {
            load(n, false, ring.row(n, false));
            emit(2 * n, ring.row(n, false));
        }
        return;
    }
    // Step t: δ on low row t, γ on high row t − 1, β on low row t − 1, α on high row t − 2,
    // which finishes pair t − 2.
    for t in pairs.start.saturating_sub(WARMUP)..=(pairs.end + 1).min(ns + 1) {
        if t < ns {
            load(t, false, ring.row(t, false));
        }
        if t < nd {
            load(t, true, ring.row(t, true));
        }
        if t < ns {
            ring.lift(false, t, t.saturating_sub(1), t.min(nd - 1), delta);
        }
        if t == 0 {
            continue;
        }
        let n = t - 1;
        if n < nd {
            ring.lift(true, n, n, t.min(ns - 1), gamma);
        }
        if n < ns {
            ring.lift(false, n, n.saturating_sub(1), n.min(nd - 1), beta);
        }
        if t == 1 {
            continue;
        }
        let n = t - 2;
        if n < nd {
            ring.lift(true, n, n, (n + 1).min(ns - 1), alpha);
        }
        if n >= pairs.start {
            emit(2 * n, ring.row(n, false));
            if n < nd {
                emit(2 * n + 1, ring.row(n, true));
            }
        }
    }
}

/// A forward transform for [`Dwt97::forward_many`]: of the plane whose row `y` `input(y, row)`
/// writes, into `coeffs` (of the plane's size).
pub struct ForwardJob<'a> {
    pub dwt: &'a mut Dwt97,
    pub input: &'a (dyn Fn(usize, &mut [i16]) + Sync),
    pub levels: usize,
    pub coeffs: &'a mut Plane<i16>,
}

/// An inverse transform for [`Dwt97::inverse_many`]: of `coeffs`, handing each output row `y` to
/// `output(y, row)` (once each, from any thread).
pub struct InverseJob<'a> {
    pub dwt: &'a mut Dwt97,
    pub coeffs: &'a Plane<i16>,
    pub levels: usize,
    pub output: &'a (dyn Fn(usize, &[i16]) + Sync),
    /// `zero(l, y, right)`: whether row `y` of level `l`'s region is known to be zero in its left
    /// half (an LL or LH band row) or its right one (HL or HH). Such rows are not read.
    pub zero: Option<&'a (dyn Fn(usize, usize, bool) -> bool + Sync)>,
}

/// Reusable 9/7 transform engine with scratch space for planes up to a maximum size.
pub struct Dwt97 {
    k: Kernels,
    isa: Isa,
    /// Level LLs, alternately: the forward transform's inputs, the inverse's outputs.
    scratch: Plane<i16>,
    scratch2: Plane<i16>,
    /// The whole output, for [`Self::inverse_sparse`]'s interface.
    full: BlockMap,
}

impl Dwt97 {
    /// An engine for a specific kernel family, or `None` if this CPU cannot run it.
    pub fn with_isa(max_width: usize, max_height: usize, isa: Isa) -> Option<Self> {
        Self::build(max_width, max_height, isa, false)
    }

    /// The rt codec's engine: the fast steps ([`FAST_STEPS`]), not the main codec's.
    pub fn fast(max_width: usize, max_height: usize, isa: Isa) -> Option<Self> {
        Self::build(max_width, max_height, isa, true)
    }

    fn build(max_width: usize, max_height: usize, isa: Isa, fast: bool) -> Option<Self> {
        Some(Self {
            k: kernels_for(isa, fast)?,
            isa,
            scratch: Plane::new(max_width.max(1), max_height.max(1)),
            scratch2: Plane::new(max_width.max(1), max_height.max(1)),
            full: BlockMap::default(),
        })
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// Forward transform of `input` into `coeffs` (same size), on `pool` if given. `input` is
    /// left unchanged.
    pub fn forward(
        &mut self,
        pool: Option<&Pool>,
        input: &Plane<i16>,
        levels: usize,
        coeffs: &mut Plane<i16>,
    ) {
        check_size(input, coeffs, levels);
        let w = input.width();
        self.forward_from(
            pool,
            &|y, row| row.copy_from_slice(input.row_span(y, 0, w)),
            levels,
            coeffs,
        );
    }

    /// Forward transform of the plane whose row `y` `input(y, row)` writes, into `coeffs` (of
    /// the plane's size), on `pool` if given.
    pub fn forward_from(
        &mut self,
        pool: Option<&Pool>,
        input: &(dyn Fn(usize, &mut [i16]) + Sync),
        levels: usize,
        coeffs: &mut Plane<i16>,
    ) {
        Self::forward_many(
            pool,
            &mut [ForwardJob {
                dwt: self,
                input,
                levels,
                coeffs,
            }],
        );
    }

    /// Forward transforms of several planes, level by level: each level of every plane is one
    /// batch on `pool` if given, so small planes and levels fill in around large ones.
    ///
    /// Each level is one pass: rows stream through a [`Ring`] that lifts them vertically, and
    /// each finished row is split and lifted horizontally into place.
    pub fn forward_many(pool: Option<&Pool>, jobs: &mut [ForwardJob<'_>]) {
        struct View<'v> {
            k: Kernels,
            input: &'v (dyn Fn(usize, &mut [i16]) + Sync),
            levels: usize,
            dst: SharedRows<'v, i16>,
            /// Level LLs, alternately (see [`Dwt97::scratch`]).
            ll: [SharedRows<'v, i16>; 2],
        }
        let mut views = Vec::with_capacity(jobs.len());
        let mut planes = Vec::with_capacity(jobs.len());
        for job in jobs.iter_mut() {
            let (w, h) = (job.coeffs.width(), job.coeffs.height());
            if job.levels == 0 {
                for y in 0..h {
                    let row = job.coeffs.row_span_mut(y, 0, w);
                    (job.input)(y, row);
                    for v in row {
                        *v <<= FRAC_BITS;
                    }
                }
                continue;
            }
            planes.push((job.levels, Geometry::new(w, h, job.levels)));
            views.push(View {
                k: job.dwt.k,
                input: job.input,
                levels: job.levels,
                dst: job.coeffs.shared(),
                ll: [job.dwt.scratch.shared(), job.dwt.scratch2.shared()],
            });
        }
        let max = planes.iter().map(|p| p.0).max().unwrap_or(0);
        for l in 1..=max {
            run_level(pool, l, &planes, &|i, pairs| {
                let (v, (w, h)) = (&views[i], planes[i].1.ll_size(l - 1));
                let (k, (wl, ns)) = (v.k, (w.div_ceil(2), h.div_ceil(2)));
                // Each level reads the LL the previous one wrote into one scratch plane and
                // writes its own into the other, while its rows are at hand.
                let (src, next) = (&v.ll[l % 2], &v.ll[(l + 1) % 2]);
                let mut ring = Ring::new(w, &k, false);
                let mut load = |y: usize, row: &mut [i16]| {
                    if l == 1 {
                        (v.input)(y, row);
                        for v in row {
                            *v <<= FRAC_BITS;
                        }
                    } else {
                        // SAFETY: the previous level's LL is only read at this level.
                        row.copy_from_slice(unsafe { src.row(y, 0, w) });
                    }
                };
                let mut emit = |n: usize, high: bool, row: &mut [i16]| {
                    // SAFETY: each band writes only the rows of its own pairs.
                    let out = unsafe { v.dst.row_mut(if high { ns + n } else { n }, 0, w) };
                    let (s, d) = out.split_at_mut(wl);
                    (k.split)(row, s, d);
                    analyze(&k, s, d);
                    if !high && l < v.levels {
                        // SAFETY: as above; the next level reads it.
                        unsafe { next.row_mut(n, 0, wl) }.copy_from_slice(s);
                    }
                };
                forward_rows(&mut ring, h, pairs, &mut load, &mut emit);
            });
        }
    }

    /// Inverse transform of `coeffs` into `out` (same size), on `pool` if given.
    pub fn inverse(
        &mut self,
        pool: Option<&Pool>,
        coeffs: &Plane<i16>,
        levels: usize,
        out: &mut Plane<i16>,
    ) {
        check_size(coeffs, out, levels);
        let (w, dst) = (out.width(), out.shared());
        self.inverse_into(pool, coeffs, levels, &|y, row| {
            // SAFETY: each output row is written once.
            unsafe { dst.row_mut(y, 0, w) }.copy_from_slice(row);
        });
    }

    /// Inverse transform of `coeffs`, on `pool` if given, handing each output row `y` to
    /// `output(y, row)` (once each, from any thread).
    pub fn inverse_into(
        &mut self,
        pool: Option<&Pool>,
        coeffs: &Plane<i16>,
        levels: usize,
        output: &(dyn Fn(usize, &[i16]) + Sync),
    ) {
        Self::inverse_many(
            pool,
            &mut [InverseJob {
                dwt: self,
                coeffs,
                levels,
                output,
                zero: None,
            }],
        );
    }

    /// Inverse transforms of several planes, level by level as [`Self::forward_many`].
    pub fn inverse_many(pool: Option<&Pool>, jobs: &mut [InverseJob<'_>]) {
        struct View<'v> {
            k: Kernels,
            coeffs: &'v Plane<i16>,
            output: &'v (dyn Fn(usize, &[i16]) + Sync),
            zero: Option<&'v (dyn Fn(usize, usize, bool) -> bool + Sync)>,
            levels: usize,
            /// Level outputs, alternately (see [`Dwt97::scratch`]).
            ll: [SharedRows<'v, i16>; 2],
        }
        let half = 1i16 << (FRAC_BITS - 1);
        let round = |src: &[i16], dst: &mut [i16]| {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s.wrapping_add(half) >> FRAC_BITS;
            }
        };
        let mut views = Vec::with_capacity(jobs.len());
        let mut planes = Vec::with_capacity(jobs.len());
        for job in jobs.iter_mut() {
            let (w, h) = (job.coeffs.width(), job.coeffs.height());
            if job.levels == 0 {
                let mut row = vec![0i16; w];
                for y in 0..h {
                    round(job.coeffs.row_span(y, 0, w), &mut row);
                    (job.output)(y, &row);
                }
                continue;
            }
            planes.push((job.levels, Geometry::new(w, h, job.levels)));
            views.push(View {
                k: job.dwt.k,
                coeffs: job.coeffs,
                output: job.output,
                zero: job.zero,
                levels: job.levels,
                ll: [job.dwt.scratch.shared(), job.dwt.scratch2.shared()],
            });
        }
        let max = planes.iter().map(|p| p.0).max().unwrap_or(0);
        for l in (1..=max).rev() {
            run_level(pool, l, &planes, &|i, pairs| {
                let (v, (w, h)) = (&views[i], planes[i].1.ll_size(l - 1));
                let (k, (wl, ns)) = (v.k, (w.div_ceil(2), h.div_ceil(2)));
                // Levels alternate between the scratch planes: each reads its LL from the other.
                let (target, low) = (&v.ll[l % 2], &v.ll[(l + 1) % 2]);
                let mut ring = Ring::new(w, &k, true);
                let (mut s, mut d) = (take(wl), take(w - wl));
                let mut rounded = take(if l == 1 { w } else { 0 });
                let mut load = |n: usize, high: bool, row: &mut [i16]| {
                    let y = if high { ns + n } else { n };
                    let zero = |right: bool| v.zero.is_some_and(|z| z(l, y, right));
                    let from_coeffs = l == v.levels || high;
                    let left_zero = from_coeffs && zero(false);
                    let right_zero = zero(true);
                    if left_zero && right_zero {
                        row.fill(0);
                        return;
                    }
                    if left_zero {
                        s.fill(0);
                    } else {
                        s.copy_from_slice(if from_coeffs {
                            v.coeffs.row_span(y, 0, wl)
                        } else {
                            // SAFETY: the coarser level's output is only read at this level.
                            unsafe { low.row(y, 0, wl) }
                        });
                    }
                    if right_zero {
                        d.fill(0);
                    } else {
                        d.copy_from_slice(v.coeffs.row_span(y, wl, w - wl));
                    }
                    synthesize(&k, &mut s, &mut d);
                    (k.merge)(&s, &d, row);
                };
                let mut emit = |y: usize, row: &mut [i16]| {
                    if l == 1 {
                        round(row, &mut rounded);
                        (v.output)(y, &rounded);
                    } else {
                        // SAFETY: each band writes only the rows of its own pairs.
                        unsafe { target.row_mut(y, 0, w) }.copy_from_slice(row);
                    }
                };
                inverse_rows(&mut ring, h, pairs, &mut load, &mut emit);
                for v in [s, d, rounded] {
                    give(v);
                }
            });
        }
    }

    /// [`Self::inverse`] with the 5/3's sparse interface; every block of the output counts as
    /// possibly nonzero.
    pub fn inverse_sparse(
        &mut self,
        pool: Option<&Pool>,
        coeffs: &Plane<i16>,
        levels: usize,
        out: &mut Plane<i16>,
    ) -> &BlockMap {
        self.inverse(pool, coeffs, levels, out);
        self.full.reset(out.width(), out.height(), OUT_LOG2);
        self.full.fill(true);
        &self.full
    }

    /// The output map of the last [`Self::inverse_sparse`].
    pub fn output_map(&self) -> &BlockMap {
        &self.full
    }
}

impl std::fmt::Debug for Dwt97 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dwt97").field("isa", &self.isa).finish()
    }
}

fn check_size(a: &Plane<i16>, b: &Plane<i16>, levels: usize) -> Geometry {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "input and output planes differ in size"
    );
    Geometry::new(a.width(), a.height(), levels)
}

/// Energy of the 1-D synthesis basis of a low- or high-pass coefficient `level` levels deep
/// (linear, unrounded lifting), with coefficients in sample units (the steps carry the
/// `2^FRAC_BITS`).
pub fn synthesis_gain_1d(level: usize, high: bool) -> f64 {
    let n = 128;
    let (mut low, mut hi) = (vec![0.0; n], vec![0.0; n]);
    if high {
        hi[n / 2] = 1.0
    } else {
        low[n / 2] = 1.0
    }
    let mut x = synthesize_f64(&low, &hi);
    for _ in 1..level {
        x = synthesize_f64(&x, &vec![0.0; x.len()]);
    }
    x.iter().map(|v| v * v).sum()
}

/// One linear synthesis step of equal-length bands.
fn synthesize_f64(low: &[f64], high: &[f64]) -> Vec<f64> {
    let (mut s, mut d) = (low.to_vec(), high.to_vec());
    let n = s.len();
    for (k, &c) in STEPS.iter().enumerate().rev() {
        let c = f64::from(c) / f64::from(1u32 << Q);
        if k % 2 == 0 {
            for i in 0..n {
                let right = if i + 1 < n { s[i + 1] } else { s[i] };
                d[i] -= c * (s[i] + right);
            }
        } else {
            for i in 0..n {
                let left = if i > 0 { d[i - 1] } else { d[0] };
                s[i] -= c * (left + d[i]);
            }
        }
    }
    (0..2 * n)
        .map(|i| if i % 2 == 0 { s[i / 2] } else { d[i / 2] })
        .collect()
}

/// Checks the `isa` kernels against the portable ones, and the exact round trip, on random
/// planes.
pub fn verify(isa: Isa, cases: usize, seed: u64) -> Result<(), String> {
    use crate::rng::SplitMix64;
    let unsupported = || format!("{isa} is not supported on this CPU");
    let mut rng = SplitMix64::new(seed);
    let pool = Pool::new(3);
    for case in 0..cases {
        let (w, h) = if case % 8 == 7 {
            (rng.range_usize(1, 300), rng.range_usize(1, 200))
        } else if case % 16 == 3 {
            // Large enough to run on the pool.
            (rng.range_usize(200, 400), rng.range_usize(200, 300))
        } else {
            (rng.range_usize(1, 72), rng.range_usize(1, 72))
        };
        let levels = rng.range_usize(0, 6);
        let range = [255, 64, 4, 2000][rng.below(4) as usize];
        let mut input = Plane::new(w, h);
        for y in 0..h {
            for v in input.row_span_mut(y, 0, w) {
                *v = rng.range_i32(-range, range) as i16;
            }
        }
        // Odd cases check the rt codec's fast steps.
        let fast = case % 2 == 1;
        let mut a = Dwt97::build(w, h, isa, fast).ok_or_else(unsupported)?;
        let mut b = Dwt97::build(w, h, Isa::Scalar, fast).ok_or_else(unsupported)?;
        let (mut ca, mut cb) = (Plane::new(w, h), Plane::new(w, h));
        a.forward(Some(&pool), &input, levels, &mut ca);
        b.forward(None, &input, levels, &mut cb);
        if !ca.rows().eq(cb.rows()) {
            return Err(format!(
                "case {case}: 9/7 forward {w}x{h} L{levels} differs"
            ));
        }
        let mut out = Plane::new(w, h);
        a.inverse(Some(&pool), &ca, levels, &mut out);
        if !out.rows().eq(input.rows()) {
            return Err(format!(
                "case {case}: 9/7 round trip {w}x{h} L{levels} differs"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_agree_and_round_trip() {
        verify(Isa::best(), 200, 7).unwrap();
    }

    #[test]
    fn constant_rows_lift_to_k() {
        // A constant row lifts to s = K·x and d = 0.
        let (mut s, mut d) = (vec![1000i16; 8], vec![1000i16; 8]);
        analyze(&PORTABLE, &mut s, &mut d);
        assert!(d.iter().all(|&v| v.abs() <= 1), "{d:?}");
        assert!(s.iter().all(|&v| (v - 1230).abs() <= 1), "{s:?}");
        let fast = kernels_for(Isa::Scalar, true).unwrap();
        let (mut s, mut d) = (vec![1000i16; 8], vec![1000i16; 8]);
        analyze(&fast, &mut s, &mut d);
        assert!(d.iter().all(|&v| v.abs() <= 1), "{d:?}");
        assert!(s.iter().all(|&v| (v - 1230).abs() <= 1), "{s:?}");
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn speed() {
        let (w, h) = (1280, 720);
        let mut input = Plane::new(w, h);
        let mut rng = crate::rng::SplitMix64::new(1);
        for y in 0..h {
            for v in input.row_span_mut(y, 0, w) {
                *v = rng.range_i32(-20, 20) as i16;
            }
        }
        let (mut c, mut o) = (Plane::new(w, h), Plane::new(w, h));
        let pool = Pool::new(4);
        for (fast, pool) in [
            (false, None),
            (true, None),
            (false, Some(&pool)),
            (true, Some(&pool)),
        ] {
            let mut t = Dwt97::build(w, h, Isa::best(), fast).unwrap();
            eprintln!("fast {fast}");
            for _ in 0..3 {
                let s = std::time::Instant::now();
                for _ in 0..20 {
                    t.forward(pool, &input, 5, &mut c);
                }
                let f = s.elapsed().as_secs_f64() / 20.0 * 1e3;
                let s = std::time::Instant::now();
                for _ in 0..20 {
                    t.inverse(pool, &c, 5, &mut o);
                }
                let i = s.elapsed().as_secs_f64() / 20.0 * 1e3;
                let n = pool.map_or(1, |p| p.threads());
                eprintln!("720p luma, {n} threads: forward {f:.2} ms, inverse {i:.2} ms");
            }
        }
    }
}
