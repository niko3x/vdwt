//! Bit-exactness checks, shared by the unit tests and `vdwt selftest` (which runs them on the
//! target CPU, where the NEON kernels actually execute).

use super::kernels::Kernels;
use super::{Dwt53, oracle, scalar};
use crate::rng::SplitMix64;
use crate::{Isa, Plane};

/// Value ranges to draw from: centred 8-bit pixels, 8-bit residuals, and the full `i16` range,
/// which makes the lifting wrap and checks that every implementation wraps identically.
const RANGES: [(i32, i32); 3] = [(-128, 127), (-255, 255), (i16::MIN as i32, i16::MAX as i32)];

/// Checks the `isa` row kernels against the scalar ones on every short length (covering all
/// SIMD edge and tail paths), then runs `cases` random whole-plane transforms against
/// [`oracle`] plus a round trip.
pub fn verify(isa: Isa, cases: usize, seed: u64) -> Result<(), String> {
    let k = Kernels::for_isa(isa).ok_or_else(|| format!("{isa} is not supported on this CPU"))?;
    let mut rng = SplitMix64::new(seed);
    check_row_kernels(&k, &mut rng)?;
    for case in 0..cases {
        let (w, h) = if case % 8 == 7 {
            (rng.range_usize(1, 300), rng.range_usize(1, 200))
        } else {
            (rng.range_usize(1, 72), rng.range_usize(1, 72))
        };
        let levels = rng.range_usize(0, 6);
        let range = RANGES[rng.below(RANGES.len() as u64) as usize];
        check_transform(isa, w, h, levels, range, &mut rng)
            .map_err(|e| format!("case {case}: {e}"))?;
    }
    Ok(())
}

/// One whole-plane check at a given size, on 8-bit residual data.
pub fn verify_size(
    isa: Isa,
    width: usize,
    height: usize,
    levels: usize,
    seed: u64,
) -> Result<(), String> {
    check_transform(
        isa,
        width,
        height,
        levels,
        RANGES[1],
        &mut SplitMix64::new(seed),
    )
}

fn random_vec(rng: &mut SplitMix64, n: usize, (lo, hi): (i32, i32)) -> Vec<i16> {
    (0..n).map(|_| rng.range_i32(lo, hi) as i16).collect()
}

fn check_row_kernels(k: &Kernels, rng: &mut SplitMix64) -> Result<(), String> {
    let r = &scalar::KERNELS;
    let isa = k.isa;
    for n in (1usize..=100).chain([255, 256, 257, 1279, 1280, 1281]) {
        for range in RANGES {
            let (ns, nd) = (n.div_ceil(2), n / 2);
            let (a, b, x) = (
                random_vec(rng, n, range),
                random_vec(rng, n, range),
                random_vec(rng, n, range),
            );
            let vertical = [
                ("v_predict", k.v_predict, r.v_predict),
                ("v_update", k.v_update, r.v_update),
                ("v_unpredict", k.v_unpredict, r.v_unpredict),
                ("v_unupdate", k.v_unupdate, r.v_unupdate),
            ];
            for (name, f, g) in vertical {
                let (mut got, mut want) = (x.clone(), x.clone());
                f(&mut got, &a, &b);
                g(&mut want, &a, &b);
                first_diff(&got, &want).map_or(Ok(()), |i| {
                    Err(format!("{isa} {name}, n={n}: differs at {i}"))
                })?;
            }

            let (mut lo, mut hi) = (vec![0; ns], vec![0; nd]);
            let (mut lo_ref, mut hi_ref) = (vec![0; ns], vec![0; nd]);
            (k.h_forward)(&x, &mut lo, &mut hi);
            (r.h_forward)(&x, &mut lo_ref, &mut hi_ref);
            if (&lo, &hi) != (&lo_ref, &hi_ref) {
                return Err(format!("{isa} h_forward, n={n}: differs from scalar"));
            }

            let mut tmp = vec![0; ns];
            let mut back = vec![0; n];
            (k.h_inverse)(&lo, &hi, &mut back, &mut tmp);
            first_diff(&back, &x).map_or(Ok(()), |i| {
                Err(format!("{isa} h_inverse round trip, n={n}: differs at {i}"))
            })?;

            // Inverse of arbitrary coefficients, not just ones a forward pass produced.
            let (lo, hi) = (random_vec(rng, ns, range), random_vec(rng, nd, range));
            let mut want = vec![0; n];
            (k.h_inverse)(&lo, &hi, &mut back, &mut tmp);
            (r.h_inverse)(&lo, &hi, &mut want, &mut tmp);
            first_diff(&back, &want).map_or(Ok(()), |i| {
                Err(format!("{isa} h_inverse, n={n}: differs at {i}"))
            })?;
        }
    }
    Ok(())
}

fn check_transform(
    isa: Isa,
    w: usize,
    h: usize,
    levels: usize,
    range: (i32, i32),
    rng: &mut SplitMix64,
) -> Result<(), String> {
    let what = format!(
        "{isa}, {w}x{h}, {levels} levels, values {}..={}",
        range.0, range.1
    );
    let data = random_vec(rng, w * h, range);
    let expected = oracle::forward(&data, w, h, levels);
    if oracle::inverse(&expected, w, h, levels) != data {
        return Err(format!("oracle round trip failed ({what})"));
    }

    let mut dwt =
        Dwt53::with_isa(w, h, isa).ok_or_else(|| format!("{isa} is not supported on this CPU"))?;
    let mut input = to_plane(&data, w, h);
    let mut coeffs = Plane::new(w, h);
    dwt.forward(&mut input, levels, &mut coeffs);
    compare(&coeffs, &expected)
        .map_err(|e| format!("forward differs from oracle at {e} ({what})"))?;

    let mut out = Plane::new(w, h);
    dwt.inverse(&coeffs, levels, &mut out);
    compare(&out, &data).map_err(|e| format!("round trip differs at {e} ({what})"))
}

fn to_plane(data: &[i16], w: usize, h: usize) -> Plane<i16> {
    let mut p = Plane::new(w, h);
    for (row, src) in p.rows_mut().zip(data.chunks_exact(w)) {
        row.copy_from_slice(src);
    }
    p
}

/// `Err` names the first differing sample as `(x, y): got vs want`.
fn compare(p: &Plane<i16>, want: &[i16]) -> Result<(), String> {
    let w = p.width();
    for (y, (row, want)) in p.rows().zip(want.chunks_exact(w)).enumerate() {
        if let Some(x) = first_diff(row, want) {
            return Err(format!("({x}, {y}): {} vs {}", row[x], want[x]));
        }
    }
    Ok(())
}

fn first_diff(a: &[i16], b: &[i16]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_isa_matches_the_oracle() {
        for isa in Isa::supported() {
            verify(isa, 200, 0x5EED_0001).unwrap();
        }
    }

    #[test]
    fn hd_frame_sizes() {
        for isa in Isa::supported() {
            verify_size(isa, 1280, 720, 5, 7).unwrap();
            verify_size(isa, 640, 360, 4, 8).unwrap();
        }
    }

    #[test]
    fn engine_reuse_across_sizes() {
        let mut rng = SplitMix64::new(3);
        let mut dwt = Dwt53::new(97, 61);
        for (w, h, levels) in [(97, 61, 5), (12, 7, 3), (97, 1, 6), (1, 61, 6), (50, 50, 0)] {
            let data = random_vec(&mut rng, w * h, RANGES[1]);
            let mut input = to_plane(&data, w, h);
            let mut coeffs = Plane::new(w, h);
            dwt.forward(&mut input, levels, &mut coeffs);
            compare(&coeffs, &oracle::forward(&data, w, h, levels)).unwrap();
            let mut out = Plane::new(w, h);
            dwt.inverse(&coeffs, levels, &mut out);
            compare(&out, &data).unwrap();
        }
    }
}
