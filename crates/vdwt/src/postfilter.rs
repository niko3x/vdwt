//! Display post-filters (optional). They are applied after decoding, never inside the prediction
//! loop, so they change neither the bitstream nor any reference frame.
//!
//! Two filters are provided:
//! - [`sharpen`]: a 5×5 unsharp mask (the generic filter, also used on x265 for comparison);
//! - [`band`]: a mid-frequency band-pass, `c + a·(blur3(c) − blur5(c))`. It boosts the detail
//!   octave the 9/7 wavelet smooths most and leaves the highest, noisiest octave alone.
//!
//! Both blurs are separable binomial kernels (`[1 2 1]/4` and `[1 4 6 4 1]/16`). Samples outside
//! the plane repeat the nearest edge sample. Strengths are in 1/256 (`a = 2.0` is 512). Luma only:
//! VMAF is luma-only and chroma sharpening adds colour fringes for no measured gain.

use crate::{Frame, Plane};

/// How the encoder decides the signalled display-filter strength.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PostFilterMode {
    /// No display filter; the header carries zero.
    #[default]
    Off,
    /// A fixed strength in 1/8 units (0..=40), for experiments.
    Fixed(u8),
    /// One strength per scene: the encoder measures luma mid-band energy and QP at each intra
    /// frame and holds that value until the next intra frame.
    Auto,
}

/// Picks the scene's `band2` strength in 1/8 units from the source frame and its QP.
///
/// The fit is `2.4 + 0.15·(qp − 42) − 0.3·T`, clamped to 0..5 and rounded to half steps:
/// smooth footage (low `T`) gets more sharpening, detailed footage (forest-like, high `T`) gets
/// little, and high QP (low rate) gets more than low QP. `T` is the mean absolute difference
/// between the 3×3 and the 5-tap wide binomial blurs, on a half-resolution luma sample.
/// Calibrated on the samtime and drone sweeps.
pub fn auto_strength(frame: &Frame, qp: u8) -> u8 {
    let t = texture_energy(&frame.y);
    let raw = 2.4 + 0.15 * (f64::from(qp) - 42.0) - 0.3 * t;
    let half = (raw * 2.0).round().clamp(0.0, 10.0) as u8;
    half * 4
}

/// Mean |blur3 − blur_w| on a half-resolution luma sample; `blur_w = [1 2 3 2 1]/9`.
fn texture_energy(y: &Plane<u8>) -> f64 {
    let (w, h) = (y.width(), y.height());
    if w < 3 || h < 3 {
        return 0.0;
    }
    const K3: [i32; 3] = [1, 2, 1];
    const KW: [i32; 5] = [1, 2, 3, 2, 1];
    let mut sum = 0.0;
    let mut n = 0u32;
    for py in (1..h - 1).step_by(2) {
        for px in (1..w - 1).step_by(2) {
            let mut s3 = 0i32;
            let mut sw = 0i32;
            for dy in -1..=1 {
                let yy = (py as isize + dy).clamp(0, h as isize - 1) as usize;
                let row = y.row(yy);
                for dx in -1..=1 {
                    let xx = (px as isize + dx).clamp(0, w as isize - 1) as usize;
                    s3 += K3[(dy + 1) as usize] * K3[(dx + 1) as usize] * i32::from(row[xx]);
                }
            }
            for dy in -2..=2 {
                let yy = (py as isize + dy).clamp(0, h as isize - 1) as usize;
                let row = y.row(yy);
                for dx in -2..=2 {
                    let xx = (px as isize + dx).clamp(0, w as isize - 1) as usize;
                    sw += KW[(dy + 2) as usize] * KW[(dx + 2) as usize] * i32::from(row[xx]);
                }
            }
            let b3 = f64::from(s3) / 16.0;
            let bw = f64::from(sw) / 81.0;
            sum += (b3 - bw).abs();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / f64::from(n) }
}

/// `[1 4 6 4 1]`, summing to 16.
const KERNEL5: [i32; 5] = [1, 4, 6, 4, 1];

/// Sharpen `src` into `dst` with strength `amount_q8` (0 disables). Both frames must match.
pub fn sharpen(src: &Frame, dst: &mut Frame, amount_q8: u16) {
    assert_eq!((src.width(), src.height()), (dst.width(), dst.height()));
    if amount_q8 == 0 {
        dst.clone_from(src);
        return;
    }
    sharpen_plane(&src.y, &mut dst.y, amount_q8);
    dst.u.clone_from(&src.u);
    dst.v.clone_from(&src.v);
}

fn sharpen_plane(src: &Plane<u8>, dst: &mut Plane<u8>, amount_q8: u16) {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return;
    }
    // One horizontal pass into a scratch plane, then a vertical pass that also applies the
    // unsharp difference. The scratch is 1.4 MB at 720p; the filter runs once per displayed
    // frame, off the prediction path.
    let mut horizontal_plane = vec![0u8; w * h];
    for y in 0..h {
        horizontal(
            src.row(y),
            &mut horizontal_plane[y * w..(y + 1) * w],
            &KERNEL5,
            2,
            4,
        );
    }
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for (k, &weight) in KERNEL5.iter().enumerate() {
                let yy = (y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
                sum += weight * i32::from(horizontal_plane[yy * w + x]);
            }
            let blur = ((sum + 8) >> 4).clamp(0, 255);
            let c = i32::from(src.row(y)[x]);
            let out = c + ((i32::from(amount_q8) * (c - blur) + 128) >> 8);
            dst.row_mut(y)[x] = out.clamp(0, 255) as u8;
        }
    }
}

fn horizontal(row: &[u8], out: &mut [u8], kernel: &[i32], radius: isize, shift: u32) {
    let w = row.len();
    for (x, o) in out.iter_mut().enumerate().take(w) {
        let mut sum = 0i32;
        for (k, &weight) in kernel.iter().enumerate() {
            let xx = (x as isize + k as isize - radius).clamp(0, w as isize - 1) as usize;
            sum += weight * i32::from(row[xx]);
        }
        *o = ((sum + (1 << (shift - 1))) >> shift).clamp(0, 255) as u8;
    }
}

/// Mid-frequency band-pass: `c + strength·(blur3(c) − blur5(c))`, strength in 1/256.
pub fn band(src: &Frame, dst: &mut Frame, strength_q8: u16) {
    apply_band(src, dst, strength_q8, crate::loopfilter::band_coefs);
}

/// Wider band-pass: `c + strength·(blur3(c) − blur_w(c))`, with `blur_w = [1 2 3 2 1]/9`.
/// Strength is in 1/256; 2.0 is the default.
pub fn band_wide(src: &Frame, dst: &mut Frame, strength_q8: u16) {
    apply_band(src, dst, strength_q8, crate::loopfilter::band_wide_coefs);
}

fn apply_band(
    src: &Frame,
    dst: &mut Frame,
    strength_q8: u16,
    coefs: fn(u16) -> crate::loopfilter::Coefs,
) {
    assert_eq!((src.width(), src.height()), (dst.width(), dst.height()));
    if strength_q8 == 0 {
        dst.clone_from(src);
        return;
    }
    let (w, h) = (src.width(), src.height());
    if w > 0 && h > 0 {
        let coefs = coefs(strength_q8);
        crate::loopfilter::apply_serial(&src.y, &mut dst.y, &coefs);
    }
    dst.u.clone_from(&src.u);
    dst.v.clone_from(&src.v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_identity_and_flat_stays_flat() {
        let mut src = Frame::new(13, 7);
        for (k, p) in src.planes_mut().into_iter().enumerate() {
            p.fill(17 + k as u8);
        }
        let mut dst = Frame::new(13, 7);
        sharpen(&src, &mut dst, 0);
        assert!(
            src.planes()
                .iter()
                .zip(dst.planes())
                .all(|(a, b)| a.rows().eq(b.rows()))
        );
        sharpen(&src, &mut dst, 154);
        assert!(
            src.planes()
                .iter()
                .zip(dst.planes())
                .all(|(a, b)| a.rows().eq(b.rows()))
        );
        band(&src, &mut dst, 512);
        assert!(
            src.planes()
                .iter()
                .zip(dst.planes())
                .all(|(a, b)| a.rows().eq(b.rows()))
        );
        band_wide(&src, &mut dst, 512);
        assert!(
            src.planes()
                .iter()
                .zip(dst.planes())
                .all(|(a, b)| a.rows().eq(b.rows()))
        );
    }

    #[test]
    fn auto_strength_is_bounded_and_texture_lowers_it() {
        let mut flat = Frame::new(64, 64);
        flat.y.fill(128);
        let flat_strength = auto_strength(&flat, 42);
        assert!(flat_strength <= 40);
        let mut detail = Frame::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                detail.y.row_mut(y)[x] = if (x / 2 + y / 2) % 2 == 0 { 0 } else { 255 };
            }
        }
        let detail_strength = auto_strength(&detail, 42);
        assert!(detail_strength < flat_strength);
    }

    #[test]
    fn an_impulse_stays_bounded_and_symmetric() {
        let mut src = Frame::new(11, 11);
        src.y.fill(0);
        src.y.row_mut(5)[5] = 255;
        let mut dst = Frame::new(11, 11);
        sharpen(&src, &mut dst, 256);
        assert_eq!(dst.y.row(5)[5], 255);
        assert_eq!(dst.y.row(5)[4], dst.y.row(5)[6]);
        assert_eq!(dst.y.row(4)[5], dst.y.row(6)[5]);
        band_wide(&src, &mut dst, 512);
        assert_eq!(dst.y.row(5)[4], dst.y.row(5)[6]);
        assert_eq!(dst.y.row(4)[5], dst.y.row(6)[5]);
    }
}
