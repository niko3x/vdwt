//! Block matching for the motion search (encoder only): windows of current-frame samples, their
//! squared error against the displaced reference, and Lucas–Kanade refinement.
//!
//! **Speed.** All sample arithmetic is fixed point. Every sample of a window moves by the same
//! vector, so whole-pixel errors are plain differences and sub-pixel ones use the same four
//! bilinear weights throughout. Large windows skip rows, never columns, so every row is a
//! contiguous run the compiler vectorizes (on x86 the hot loops are compiled a second time for
//! AVX2 and picked at run time). The results do not depend on the ISA.

use crate::Plane;

/// Whether the CPU has AVX2 (checked once).
#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// Lucas–Kanade iterations.
pub(super) const ITERATIONS: usize = 4;
/// Furthest Lucas–Kanade may wander from where it starts, in pixels.
const MAX_LK_PX: f64 = 8.0;
/// Smallest structure-tensor eigenvalue per sample for Lucas–Kanade: below it the window is too
/// flat (or a straight edge) to pin down a sub-pixel vector.
const MIN_TEXTURE: f64 = 4.0;

/// Encoder settings for motion decisions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshParams {
    /// Rate-distortion multiplier: squared error per bit.
    pub lambda: f64,
    /// Search for node vectors (else every node keeps its global prediction).
    pub refine: bool,
    /// Quadrants search as widely as whole regions (else from their region's best, in steps of
    /// at most 2 pixels).
    pub wide: bool,
    /// Realtime mode (§10.2): exact node errors on every other row, searches only where the
    /// best candidate costs over 32 bits' worth, two Lucas–Kanade steps.
    pub realtime: bool,
    /// Root rows per stripe searched independently (0: one stripe): a stripe's first row
    /// neither waits for nor sees the row above, as the `rt` codec's tiles code them.
    pub stripe_rows: usize,
}

/// What the search decided.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MeshStats {
    pub refined: usize,
    pub intra: usize,
}

/// A window of current-frame samples: every `step`-th row of `x0..x0 + cols` from row `y0`,
/// with central differences (twice the gradient) for Lucas–Kanade.
#[derive(Default)]
pub(super) struct Window {
    x0: i32,
    y0: i32,
    step: usize,
    cols: usize,
    rows: usize,
    value: Vec<u8>,
    gx: Vec<i16>,
    gy: Vec<i16>,
    /// Σv and Σv².
    pub sum: u64,
    pub sum_sq: u64,
}

impl Window {
    /// Columns `x0..x1` of rows `y0..y1` (every `step`-th) of `cur`. Gradients need a one-pixel
    /// margin inside `cur`.
    pub(super) fn fill(
        &mut self,
        cur: &Plane<u8>,
        (x0, x1): (i32, i32),
        (y0, y1): (i32, i32),
        step: usize,
    ) {
        self.x0 = x0;
        self.y0 = y0;
        self.step = step;
        self.cols = (x1 - x0).max(0) as usize;
        self.rows = ((y1 - y0).max(0) as usize).div_ceil(step);
        self.value.clear();
        self.gx.clear();
        self.gy.clear();
        if self.cols == 0 {
            self.rows = 0;
        }
        for r in 0..self.rows {
            let y = y0 as usize + r * step;
            self.value
                .extend_from_slice(&cur.row(y)[x0 as usize..x1 as usize]);
        }
        (self.sum, self.sum_sq) = sums(&self.value, self.cols.max(1));
    }

    /// The samples' central differences, which only Lucas–Kanade needs.
    pub(super) fn fill_gradients(&mut self, cur: &Plane<u8>) {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { self.fill_gradients_avx2(cur) };
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // SAFETY: NEON is present.
            return unsafe { self.fill_gradients_neon(cur) };
        }
        #[allow(unreachable_code)]
        self.fill_gradients_any(cur)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fill_gradients_avx2(&mut self, cur: &Plane<u8>) {
        use std::arch::x86_64::*;

        let x0 = self.x0 as usize;
        let cols = self.cols;
        let total = cols * self.rows;
        self.gx.clear();
        self.gy.clear();
        self.gx.resize(total, 0);
        self.gy.resize(total, 0);
        for r in 0..self.rows {
            let y = self.y0 as usize + r * self.step;
            // SAFETY: the row is inside the plane, and the window keeps a one-pixel margin,
            // so `x0 - 1` and `x0 + cols` are inside it.
            unsafe {
                let row = cur.row(y);
                let up = cur.row(y - 1);
                let down = cur.row(y + 1);
                let base = r * cols;
                let mut k = 0;
                while k + 16 <= cols {
                    let l = _mm_loadu_si128(row.as_ptr().add(x0 - 1 + k).cast());
                    let rt = _mm_loadu_si128(row.as_ptr().add(x0 + 1 + k).cast());
                    let gx = _mm256_sub_epi16(_mm256_cvtepu8_epi16(rt), _mm256_cvtepu8_epi16(l));
                    _mm256_storeu_si256(self.gx.as_mut_ptr().add(base + k).cast(), gx);
                    let u = _mm_loadu_si128(up.as_ptr().add(x0 + k).cast());
                    let d = _mm_loadu_si128(down.as_ptr().add(x0 + k).cast());
                    let gy = _mm256_sub_epi16(_mm256_cvtepu8_epi16(d), _mm256_cvtepu8_epi16(u));
                    _mm256_storeu_si256(self.gy.as_mut_ptr().add(base + k).cast(), gy);
                    k += 16;
                }
                // The tail of the row.
                while k < cols {
                    self.gx[base + k] = i16::from(row[x0 + 1 + k]) - i16::from(row[x0 - 1 + k]);
                    self.gy[base + k] = i16::from(down[x0 + k]) - i16::from(up[x0 + k]);
                    k += 1;
                }
            }
        }
    }

    /// [`Self::fill_gradients`] compiled for NEON.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn fill_gradients_neon(&mut self, cur: &Plane<u8>) {
        use std::arch::aarch64::*;

        let x0 = self.x0 as usize;
        let cols = self.cols;
        let total = cols * self.rows;
        self.gx.clear();
        self.gy.clear();
        self.gx.resize(total, 0);
        self.gy.resize(total, 0);
        for r in 0..self.rows {
            let y = self.y0 as usize + r * self.step;
            // SAFETY: the row is inside the plane, and the window keeps a one-pixel margin,
            // so `x0 - 1` and `x0 + cols` are inside it.
            unsafe {
                let row = cur.row(y);
                let up = cur.row(y - 1);
                let down = cur.row(y + 1);
                let base = r * cols;
                let mut k = 0;
                while k + 16 <= cols {
                    let l = vld1q_u8(row.as_ptr().add(x0 - 1 + k));
                    let rt = vld1q_u8(row.as_ptr().add(x0 + 1 + k));
                    // Widen to 16 bits, then subtract: the difference of two bytes fits.
                    let gx_lo = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rt))),
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(l))),
                    );
                    let gx_hi = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_high_u8(rt)),
                        vreinterpretq_s16_u16(vmovl_high_u8(l)),
                    );
                    vst1q_s16(self.gx.as_mut_ptr().add(base + k).cast(), gx_lo);
                    vst1q_s16(self.gx.as_mut_ptr().add(base + k + 8).cast(), gx_hi);
                    let u = vld1q_u8(up.as_ptr().add(x0 + k));
                    let d = vld1q_u8(down.as_ptr().add(x0 + k));
                    let gy_lo = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(d))),
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u))),
                    );
                    let gy_hi = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_high_u8(d)),
                        vreinterpretq_s16_u16(vmovl_high_u8(u)),
                    );
                    vst1q_s16(self.gy.as_mut_ptr().add(base + k).cast(), gy_lo);
                    vst1q_s16(self.gy.as_mut_ptr().add(base + k + 8).cast(), gy_hi);
                    k += 16;
                }
                while k < cols {
                    self.gx[base + k] = i16::from(row[x0 + 1 + k]) - i16::from(row[x0 - 1 + k]);
                    self.gy[base + k] = i16::from(down[x0 + k]) - i16::from(up[x0 + k]);
                    k += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn fill_gradients_any(&mut self, cur: &Plane<u8>) {
        let x0 = self.x0 as usize;
        for r in 0..self.rows {
            let y = self.y0 as usize + r * self.step;
            let (up, row, down) = (cur.row(y - 1), cur.row(y), cur.row(y + 1));
            let (l, rt) = (
                &row[x0 - 1..x0 - 1 + self.cols],
                &row[x0 + 1..x0 + 1 + self.cols],
            );
            self.gx
                .extend(rt.iter().zip(l).map(|(&a, &b)| i16::from(a) - i16::from(b)));
            let (u, d) = (&up[x0..x0 + self.cols], &down[x0..x0 + self.cols]);
            self.gy
                .extend(d.iter().zip(u).map(|(&a, &b)| i16::from(a) - i16::from(b)));
        }
    }

    pub(super) fn len(&self) -> usize {
        self.value.len()
    }

    /// Pixels each sample stands for.
    pub(super) fn weight(&self) -> usize {
        self.step
    }

    /// Whether every sample displaced by `(ix, iy)` and its right and lower neighbours lie
    /// inside `reference`.
    fn inside(&self, reference: &Plane<u8>, ix: i32, iy: i32) -> bool {
        let last_row = self.y0 + ((self.rows.max(1) - 1) * self.step) as i32;
        self.x0 + ix >= 0
            && self.y0 + iy >= 0
            && self.x0 + self.cols.max(1) as i32 + ix < reference.width() as i32
            && last_row + iy + 1 < reference.height() as i32
    }

    /// Squared error, in 1/64 units, of the reference displaced by `mv` (1/8 pel) against the
    /// samples: the reference is interpolated bilinearly with weights in 1/64 and the error
    /// rounded to 1/8. Coordinates are clamped to the plane like the codec's edge extension.
    pub(super) fn ssd(&self, reference: &Plane<u8>, mv: [i32; 2]) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { self.ssd_avx2(reference, mv) };
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        return self.ssd_neon(reference, mv);
        #[allow(unreachable_code)]
        self.ssd_any(reference, mv)
    }

    /// [`Self::ssd`] on NEON, as [`Self::ssd_avx2`].
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn ssd_neon(&self, reference: &Plane<u8>, mv: [i32; 2]) -> u64 {
        let (ix, iy) = (mv[0] >> 3, mv[1] >> 3);
        let (fx, fy) = (mv[0] & 7, mv[1] & 7);
        if !self.inside(reference, ix, iy) {
            return self.ssd_any(reference, mv);
        }
        let w = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
        if self.rows == 0 {
            return 0;
        }
        let stride = reference.stride();
        let first = (self.y0 + iy) as usize * stride + (self.x0 + ix) as usize;
        let last = first + (self.rows - 1) * self.step * stride + stride + self.cols;
        // SAFETY: `inside` keeps every sample and its right and lower neighbours (up to
        // `last`) inside the plane's data.
        unsafe {
            neon::ssd(
                &self.value,
                self.cols,
                &reference.data()[first..=last],
                self.step * stride,
                stride,
                w,
            )
        }
    }

    /// [`Self::ssd`] on AVX2: 16 samples at a time in 16-bit arithmetic.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn ssd_avx2(&self, reference: &Plane<u8>, mv: [i32; 2]) -> u64 {
        let (ix, iy) = (mv[0] >> 3, mv[1] >> 3);
        let (fx, fy) = (mv[0] & 7, mv[1] & 7);
        if !self.inside(reference, ix, iy) {
            return self.ssd_any(reference, mv);
        }
        let w = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
        if self.rows == 0 {
            return 0;
        }
        let stride = reference.stride();
        let first = (self.y0 + iy) as usize * stride + (self.x0 + ix) as usize;
        let last = first + (self.rows - 1) * self.step * stride + stride + self.cols;
        // SAFETY: AVX2 is present; `inside` keeps every sample and its right and lower
        // neighbours (up to `last`) inside the plane's data.
        unsafe {
            x86::ssd(
                &self.value,
                self.cols,
                &reference.data()[first..=last],
                self.step * stride,
                stride,
                w,
            )
        }
    }

    #[inline(always)]
    fn ssd_any(&self, reference: &Plane<u8>, mv: [i32; 2]) -> u64 {
        let (ix, iy) = (mv[0] >> 3, mv[1] >> 3);
        let (fx, fy) = (mv[0] & 7, mv[1] & 7);
        let w = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
        if !self.inside(reference, ix, iy) {
            return self.ssd_clamped(reference, ix, iy, w);
        }
        let n = self.cols;
        let mut sum = 0u64;
        for (r, cur) in self.value.chunks_exact(n).enumerate() {
            let (a, b) = self.reference_rows(reference, r, ix, iy);
            let s: u32 = if fx | fy == 0 {
                a[..n]
                    .iter()
                    .zip(cur)
                    .map(|(&p, &v)| (i32::from(p) - i32::from(v)).pow(2) as u32)
                    .sum::<u32>()
                    * 64
            } else {
                (a[..n].iter().zip(&a[1..n + 1]))
                    .zip(b[..n].iter().zip(&b[1..n + 1]))
                    .zip(cur)
                    .map(|(((&a0, &a1), (&b0, &b1)), &v)| {
                        let p = w[0] * i32::from(a0)
                            + w[1] * i32::from(a1)
                            + w[2] * i32::from(b0)
                            + w[3] * i32::from(b1);
                        ((p - 64 * i32::from(v) + 4) >> 3).pow(2) as u32
                    })
                    .sum()
            };
            sum += u64::from(s);
        }
        sum
    }

    /// Reference rows `y` and `y + 1` for sample row `r` displaced by `(ix, iy)`, from the first
    /// sample's column to the last sample's right neighbour.
    #[inline(always)]
    fn reference_rows<'a>(
        &self,
        reference: &'a Plane<u8>,
        r: usize,
        ix: i32,
        iy: i32,
    ) -> (&'a [u8], &'a [u8]) {
        let y = (self.y0 + iy) as usize + r * self.step;
        let x = (self.x0 + ix) as usize;
        let n = self.cols + 1;
        let stride = reference.stride();
        // Rows are `stride` apart in `data`, and `inside` keeps both rows and every neighbour
        // inside the plane.
        let data = reference.data();
        let at = y * stride + x;
        (&data[at..at + n], &data[at + stride..at + stride + n])
    }

    fn ssd_clamped(&self, reference: &Plane<u8>, ix: i32, iy: i32, w: [i32; 4]) -> u64 {
        let mut sum = 0u64;
        self.for_each_clamped(reference, ix, iy, |k, [a, b, c, d]| {
            let p = w[0] * a + w[1] * b + w[2] * c + w[3] * d;
            sum += ((p - 64 * i32::from(self.value[k]) + 4) >> 3).pow(2) as u64;
        });
        sum
    }

    /// Calls `f` with every sample index and its four reference neighbours at `(ix, iy)`, with
    /// coordinates clamped to the plane.
    fn for_each_clamped(
        &self,
        reference: &Plane<u8>,
        ix: i32,
        iy: i32,
        mut f: impl FnMut(usize, [i32; 4]),
    ) {
        let (w, h) = (reference.width() as i32, reference.height() as i32);
        let cx = |x: i32| x.clamp(0, w - 1) as usize;
        for r in 0..self.rows {
            let y = self.y0 + (r * self.step) as i32 + iy;
            let (ra, rb) = (
                reference.row(y.clamp(0, h - 1) as usize),
                reference.row((y + 1).clamp(0, h - 1) as usize),
            );
            for c in 0..self.cols {
                let x = self.x0 + c as i32 + ix;
                let (x0, x1) = (cx(x), cx(x + 1));
                let n = [ra[x0], ra[x1], rb[x0], rb[x1]].map(i32::from);
                f(r * self.cols + c, n);
            }
        }
    }

    /// Lucas–Kanade's sums `(Σ gx·e, Σ gy·e)` at `q` (1/64 pel), with gradients doubled and the
    /// errors in 1/256.
    fn gradient_error(&self, reference: &Plane<u8>, q: [i32; 2]) -> (i64, i64) {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { self.gradient_error_avx2(reference, q) };
        }
        self.gradient_error_any(reference, q)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn gradient_error_avx2(&self, reference: &Plane<u8>, q: [i32; 2]) -> (i64, i64) {
        self.gradient_error_any(reference, q)
    }

    #[inline(always)]
    fn gradient_error_any(&self, reference: &Plane<u8>, q: [i32; 2]) -> (i64, i64) {
        let (ix, iy) = (q[0] >> 6, q[1] >> 6);
        let (fx, fy) = (q[0] & 63, q[1] & 63);
        let w = [
            (64 - fx) * (64 - fy),
            fx * (64 - fy),
            (64 - fx) * fy,
            fx * fy,
        ];
        // Interpolation in 1/4096, errors rounded to 1/256.
        let error = |n: [i32; 4], v: u8| {
            (w[0] * n[0] + w[1] * n[1] + w[2] * n[2] + w[3] * n[3] - 4096 * i32::from(v) + 8) >> 4
        };
        let (mut ex, mut ey) = (0i64, 0i64);
        if !self.inside(reference, ix, iy) {
            self.for_each_clamped(reference, ix, iy, |k, n| {
                let e = error(n, self.value[k]);
                ex += i64::from(i32::from(self.gx[k]) * e);
                ey += i64::from(i32::from(self.gy[k]) * e);
            });
            return (ex, ey);
        }
        let n = self.cols;
        let rows = self.value.chunks_exact(n);
        let gxs = self.gx.chunks_exact(n);
        let gys = self.gy.chunks_exact(n);
        for (r, ((cur, gx), gy)) in rows.zip(gxs).zip(gys).enumerate() {
            let (a, b) = self.reference_rows(reference, r, ix, iy);
            let (sx, sy) = (a[..n].iter().zip(&a[1..n + 1]))
                .zip(b[..n].iter().zip(&b[1..n + 1]))
                .zip(cur)
                .map(|(((&a0, &a1), (&b0, &b1)), &v)| error([a0, a1, b0, b1].map(i32::from), v))
                .zip(gx)
                .zip(gy)
                .map(|((e, &gx), &gy)| (i32::from(gx) * e, i32::from(gy) * e))
                .fold((0, 0), |(x, y), (dx, dy)| (x + dx, y + dy));
            ex += i64::from(sx);
            ey += i64::from(sy);
        }
        (ex, ey)
    }

    /// The structure tensor `(Σ gx², Σ gx·gy, Σ gy²)` in squared sample values per pixel.
    fn tensor(&self) -> (f64, f64, f64) {
        let (mut a, mut b, mut c) = (0i64, 0i64, 0i64);
        for (&x, &y) in self.gx.iter().zip(&self.gy) {
            let (x, y) = (i64::from(x), i64::from(y));
            a += x * x;
            b += x * y;
            c += y * y;
        }
        // Central differences are twice the gradient.
        (a as f64 / 4.0, b as f64 / 4.0, c as f64 / 4.0)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Sums the four 64-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn total(acc: __m256i) -> u64 {
        let mut lanes = [0u64; 4];
        // SAFETY: four 64-bit lanes.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
        lanes.iter().sum()
    }

    /// Adds eight 32-bit sums (each below 2^31) to four 64-bit ones.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen(acc: __m256i, sq: __m256i) -> __m256i {
        let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq));
        let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(sq));
        _mm256_add_epi64(acc, _mm256_add_epi64(lo, hi))
    }

    /// The squared error of [`Window::ssd`](super::Window::ssd) for samples `value` (rows of
    /// `cols`) against `reference`, which starts at the first sample's displaced position and
    /// holds rows `row_step` apart with their lower neighbours `stride` further:
    /// `Σ ((w·[a[k], a[k+1], b[k], b[k+1]] − 64·v[k] + 4) >> 3)²`, 16 samples at a time. The
    /// weights sum to 64, so every term fits 16 bits.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ssd(
        value: &[u8],
        cols: usize,
        reference: &[u8],
        row_step: usize,
        stride: usize,
        w: [i32; 4],
    ) -> u64 {
        let whole = w[0] == 64;
        let wv = w.map(|w| _mm256_set1_epi16(w as i16));
        let round = _mm256_set1_epi16(4);
        let rows = value.len() / cols;
        debug_assert!(rows == 0 || reference.len() > (rows - 1) * row_step + stride + cols);
        let (mut acc, mut acc32) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        // Pairs of squares are below 2^23: 128 of them fit 32 bits.
        let mut pending = 0;
        let mut tail = 0u64;
        let (vp, rp) = (value.as_ptr(), reference.as_ptr());
        for r in 0..rows {
            // SAFETY: (all below) row `r` of `value` holds `cols` samples, and the reference
            // rows `cols + 1` from `r·row_step` and `stride` further (checked above).
            let (v, a) = unsafe { (vp.add(r * cols), rp.add(r * row_step)) };
            let b = unsafe { a.add(stride) };
            let ld = |s: *const u8, at: usize| unsafe {
                _mm256_cvtepu8_epi16(_mm_loadu_si128(s.add(at).cast()))
            };
            let mut k = 0;
            while k + 16 <= cols {
                let e = if whole {
                    // 8·(a − v) after the shift: squares carry the factor 64 below.
                    _mm256_sub_epi16(ld(a, k), ld(v, k))
                } else {
                    let p = _mm256_add_epi16(
                        _mm256_add_epi16(
                            _mm256_mullo_epi16(wv[0], ld(a, k)),
                            _mm256_mullo_epi16(wv[1], ld(a, k + 1)),
                        ),
                        _mm256_add_epi16(
                            _mm256_mullo_epi16(wv[2], ld(b, k)),
                            _mm256_mullo_epi16(wv[3], ld(b, k + 1)),
                        ),
                    );
                    let e = _mm256_sub_epi16(p, _mm256_slli_epi16::<6>(ld(v, k)));
                    _mm256_srai_epi16::<3>(_mm256_add_epi16(e, round))
                };
                acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(e, e));
                k += 16;
                pending += 1;
                if pending == 128 {
                    acc = widen(acc, acc32);
                    acc32 = _mm256_setzero_si256();
                    pending = 0;
                }
            }
            for k in k..cols {
                let at = |s: *const u8, i: usize| i32::from(unsafe { *s.add(i) });
                let p =
                    w[0] * at(a, k) + w[1] * at(a, k + 1) + w[2] * at(b, k) + w[3] * at(b, k + 1);
                let e = (p - 64 * at(v, k) + 4) >> 3;
                tail += (e * e) as u64;
            }
        }
        acc = widen(acc, acc32);
        let sum = total(acc);
        (if whole { 64 * sum } else { sum }) + tail
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;

    /// Rows ahead [`ssd`] prefetches.
    const PREFETCH_ROWS: usize = 2;

    /// Hints that the cache line holding `p` will be read soon. It never faults.
    #[inline(always)]
    fn prefetch(p: *const u8) {
        // SAFETY: a prefetch hint has no architectural effect and never faults.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
        }
    }

    /// The squared error of [`Window::ssd`](super::Window::ssd) as the AVX2 kernel computes it,
    /// 16 samples at a time: the weights are at most 64, so the interpolation is byte
    /// multiplies widening to 16 bits.
    pub(super) unsafe fn ssd(
        value: &[u8],
        cols: usize,
        reference: &[u8],
        row_step: usize,
        stride: usize,
        w: [i32; 4],
    ) -> u64 {
        let whole = w[0] == 64;
        let rows = value.len() / cols;
        debug_assert!(rows == 0 || reference.len() > (rows - 1) * row_step + stride + cols);
        // SAFETY: (all below) NEON is present; row `r` of `value` holds `cols` samples, and
        // the reference rows `cols + 1` from `r·row_step` and `stride` further.
        unsafe {
            let wv = w.map(|w| vdupq_n_u8(w as u8));
            let (mut acc, mut acc32) = (vdupq_n_u64(0), vdupq_n_u32(0));
            // Squares are below 2^22, two per lane per step: 128 steps fit 32 bits.
            let mut pending = 0;
            let mut tail = 0u64;
            let (vp, rp) = (value.as_ptr(), reference.as_ptr());
            for r in 0..rows {
                let (v, a) = (vp.add(r * cols), rp.add(r * row_step));
                let b = a.add(stride);
                // The reference rows a few rows on: short rows a large stride apart, which the
                // hardware prefetcher does not follow.
                let ahead = rp.wrapping_add((r + PREFETCH_ROWS) * row_step);
                for p in [ahead, ahead.wrapping_add(stride)] {
                    prefetch(p);
                    prefetch(p.wrapping_add(cols));
                }
                let mut k = 0;
                while k + 16 <= cols {
                    let (a0, vv) = (vld1q_u8(a.add(k)), vld1q_u8(v.add(k)));
                    let e = if whole {
                        // 8·(a − v) after the shift: squares carry the factor 64 below.
                        [
                            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(a0), vget_low_u8(vv))),
                            vreinterpretq_s16_u16(vsubl_high_u8(a0, vv)),
                        ]
                    } else {
                        let (a1, b0, b1) = (
                            vld1q_u8(a.add(k + 1)),
                            vld1q_u8(b.add(k)),
                            vld1q_u8(b.add(k + 1)),
                        );
                        let lo = vmull_u8(vget_low_u8(a0), vget_low_u8(wv[0]));
                        let lo = vmlal_u8(lo, vget_low_u8(a1), vget_low_u8(wv[1]));
                        let lo = vmlal_u8(lo, vget_low_u8(b0), vget_low_u8(wv[2]));
                        let lo = vmlal_u8(lo, vget_low_u8(b1), vget_low_u8(wv[3]));
                        let hi = vmull_high_u8(a0, wv[0]);
                        let hi = vmlal_high_u8(hi, a1, wv[1]);
                        let hi = vmlal_high_u8(hi, b0, wv[2]);
                        let hi = vmlal_high_u8(hi, b1, wv[3]);
                        let e = |p: uint16x8_t, v: uint8x8_t| {
                            let d = vsubq_u16(p, vshll_n_u8::<6>(v));
                            vrshrq_n_s16::<3>(vreinterpretq_s16_u16(d))
                        };
                        [e(lo, vget_low_u8(vv)), e(hi, vget_high_u8(vv))]
                    };
                    let sq = vmull_s16(vget_low_s16(e[0]), vget_low_s16(e[0]));
                    let sq = vmlal_high_s16(sq, e[0], e[0]);
                    let sq = vmlal_s16(sq, vget_low_s16(e[1]), vget_low_s16(e[1]));
                    let sq = vmlal_high_s16(sq, e[1], e[1]);
                    acc32 = vaddq_u32(acc32, vreinterpretq_u32_s32(sq));
                    k += 16;
                    pending += 1;
                    if pending == 64 {
                        acc = vpadalq_u32(acc, acc32);
                        acc32 = vdupq_n_u32(0);
                        pending = 0;
                    }
                }
                for k in k..cols {
                    let at = |s: *const u8, i: usize| i32::from(*s.add(i));
                    let p = w[0] * at(a, k)
                        + w[1] * at(a, k + 1)
                        + w[2] * at(b, k)
                        + w[3] * at(b, k + 1);
                    let e = (p - 64 * at(v, k) + 4) >> 3;
                    tail += (e * e) as u64;
                }
            }
            let sum = vaddvq_u64(vpadalq_u32(acc, acc32));
            (if whole { 64 * sum } else { sum }) + tail
        }
    }
}

/// `(Σv, Σv²)` of `value`, summed in 32 bits per row of `cols`.
fn sums(value: &[u8], cols: usize) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        #[target_feature(enable = "avx2")]
        fn sums_avx2(value: &[u8], cols: usize) -> (u64, u64) {
            sums_any(value, cols)
        }
        // SAFETY: AVX2 is present.
        return unsafe { sums_avx2(value, cols) };
    }
    sums_any(value, cols)
}

#[inline(always)]
fn sums_any(value: &[u8], cols: usize) -> (u64, u64) {
    let (mut sum, mut sum_sq) = (0u64, 0u64);
    for row in value.chunks(cols) {
        sum += u64::from(row.iter().map(|&v| u32::from(v)).sum::<u32>());
        sum_sq += u64::from(
            row.iter()
                .map(|&v| u32::from(v) * u32::from(v))
                .sum::<u32>(),
        );
    }
    (sum, sum_sq)
}

/// Lucas–Kanade from `start` (pixels), at most `iterations` steps: returns the converged vector, or `None` when the window
/// is too flat or the iteration runs away.
pub(super) fn lucas_kanade(
    win: &Window,
    reference: &Plane<u8>,
    start: [f64; 2],
    iterations: usize,
) -> Option<[f64; 2]> {
    let (a, b, c) = win.tensor();
    let lambda_min = (a + c) / 2.0 - (((a - c) / 2.0).powi(2) + b * b).sqrt();
    if lambda_min < MIN_TEXTURE * win.len() as f64 {
        return None;
    }
    let det = a * c - b * b;
    let mut m = start;
    for _ in 0..iterations {
        let q = m.map(|v| (v * 64.0).round() as i32);
        let (ex, ey) = win.gradient_error(reference, q);
        // Doubled gradients times errors in 1/256.
        let (ex, ey) = (ex as f64 / 512.0, ey as f64 / 512.0);
        // δ = −G⁻¹·(Σ g·e)
        let dx = -(c * ex - b * ey) / det;
        let dy = -(a * ey - b * ex) / det;
        m[0] += dx;
        m[1] += dy;
        if (m[0] - start[0]).abs() > MAX_LK_PX || (m[1] - start[1]).abs() > MAX_LK_PX {
            return None;
        }
        if dx.abs() < 0.02 && dy.abs() < 0.02 {
            break;
        }
    }
    Some(m)
}

#[cfg(test)]
/// Temporary micro-benchmark: the scalar gradients against the vector ones.
#[test]
#[ignore]
fn bench_gradients() {
    use std::time::Instant;
    let mut plane = Plane::<u8>::new(1280, 720);
    let mut rng = crate::rng::SplitMix64::new(3);
    for v in plane.rows_mut().flatten() {
        *v = rng.below(256) as u8;
    }
    let mut w = Window::default();
    w.x0 = 1;
    w.y0 = 1;
    w.step = 1;
    w.cols = 64;
    w.rows = 64;
    let mut best_scalar = f64::MAX;
    let mut best_vector = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..50 {
            w.fill_gradients_any(&plane);
        }
        best_scalar = best_scalar.min(t.elapsed().as_secs_f64() * 1000.0 / 50.0);
        let t = Instant::now();
        for _ in 0..50 {
            w.fill_gradients(&plane);
        }
        best_vector = best_vector.min(t.elapsed().as_secs_f64() * 1000.0 / 50.0);
    }
    println!(
        "gradients 64x64 window: scalar {:.4} ms   vector {:.4} ms   speedup {:.2}x",
        best_scalar,
        best_vector,
        best_scalar / best_vector
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector gradients must equal the scalar ones exactly: the motion decisions they
    /// drive are the same.
    #[test]
    fn the_vector_gradients_match_the_scalar_ones() {
        let mut rng = crate::rng::SplitMix64::new(11);
        let mut plane = Plane::<u8>::new(64, 48);
        for v in plane.rows_mut().flatten() {
            *v = rng.below(256) as u8;
        }
        for (x0, x1, y0, y1, step) in [
            (1usize, 60usize, 1usize, 40usize, 1usize),
            (2, 61, 2, 41, 3),
            (1, 17, 1, 20, 2),
            (1, 33, 1, 33, 5),
        ] {
            let mut want = Window::default();
            want.x0 = x0 as i32;
            want.y0 = y0 as i32;
            want.step = step;
            want.cols = x1 - x0;
            want.rows = (y1 - y0).div_ceil(step);
            want.value.clear();
            want.gx.clear();
            want.gy.clear();
            for r in 0..want.rows {
                let y = y0 + r * step;
                want.value.extend_from_slice(&plane.row(y)[x0..x1]);
            }
            want.fill_gradients_any(&plane);
            let mut got = Window::default();
            got.x0 = want.x0;
            got.y0 = want.y0;
            got.step = want.step;
            got.cols = want.cols;
            got.rows = want.rows;
            got.fill_gradients(&plane);
            assert_eq!(want.gx, got.gx, "gx mismatch");
            assert_eq!(want.gy, got.gy, "gy mismatch");
        }
    }

    /// A textured plane: many random waves, so no shift makes it repeat.
    fn texture(w: usize, h: usize) -> Plane<u8> {
        let mut rng = crate::rng::SplitMix64::new(12);
        let waves: Vec<(f64, f64, f64)> = (0..20)
            .map(|_| {
                let f = 0.05 + rng.below(1000) as f64 / 4000.0;
                let a = rng.below(6283) as f64 / 1000.0;
                (f * a.cos(), f * a.sin(), rng.below(6283) as f64 / 1000.0)
            })
            .collect();
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                let v: f64 = waves
                    .iter()
                    .map(|&(a, b, ph)| (a * fx + b * fy + ph).sin())
                    .sum();
                p.set(x, y, (128.0 + 20.0 * v).clamp(0.0, 255.0) as u8);
            }
        }
        p
    }

    #[test]
    fn errors_match_a_direct_computation_and_lucas_kanade_converges() {
        let (reference, cur) = (texture(160, 120), texture(160, 120));
        for step in [1, 2, 4] {
            let mut win = Window::default();
            win.fill(&cur, (40, 104), (30, 94), step);
            win.fill_gradients(&cur);
            for mv in [[0, 0], [8, -16], [3, 5], [-13, 7], [-2000, 9]] {
                let mut direct = 0u64;
                let (fx, fy) = (mv[0] & 7, mv[1] & 7);
                let w = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
                for y in (30..94).step_by(step) {
                    for x in 40..104 {
                        let at = |dx: i32, dy: i32| {
                            let xx = (x + (mv[0] >> 3) + dx).clamp(0, 159) as usize;
                            let yy = (y + (mv[1] >> 3) + dy).clamp(0, 119) as usize;
                            i32::from(reference.get(xx, yy))
                        };
                        let p =
                            w[0] * at(0, 0) + w[1] * at(1, 0) + w[2] * at(0, 1) + w[3] * at(1, 1);
                        let v = i32::from(cur.get(x as usize, y as usize));
                        direct += ((p - 64 * v + 4) >> 3).pow(2) as u64;
                    }
                }
                assert_eq!(win.ssd(&reference, mv), direct, "step {step}, {mv:?}");
            }
            // The same texture, so Lucas–Kanade finds no motion from under a pixel away.
            let m = lucas_kanade(&win, &reference, [0.8, -0.6], ITERATIONS).expect("textured");
            assert!(m[0].abs() < 0.1 && m[1].abs() < 0.1, "step {step}: {m:?}");
        }
    }
}
