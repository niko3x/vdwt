//! A literal transcription of the ISO/IEC 15444-1 Annex F reversible 5/3 transform.
//!
//! Slow and allocation-heavy on purpose: it follows the standard's procedures (periodic symmetric
//! extension, `1D_SD` / `1D_SR` on interleaved signals, vertical-then-horizontal analysis) rather
//! than the streaming row schedule of the fast path, so the two are independent implementations.
//! Every optimized path must match it bit for bit.

use super::Geometry;

/// `⌊(a + b) / 2⌋`
fn floor_half(a: i16, b: i16) -> i16 {
    (i32::from(a) + i32::from(b)).div_euclid(2) as i16
}

/// `⌊(a + b + 2) / 4⌋`
fn floor_quarter_rounded(a: i16, b: i16) -> i16 {
    (i32::from(a) + i32::from(b) + 2).div_euclid(4) as i16
}

/// Periodic symmetric extension (Annex F, `PSE_O`) of index `i` into a signal on `0..n`, `n ≥ 2`.
fn pse(i: isize, n: usize) -> usize {
    let period = 2 * (n as isize - 1);
    let m = i.rem_euclid(period);
    (if m >= n as isize { period - m } else { m }) as usize
}

/// `1D_SD` on an interleaved signal that starts at an even index; output stays interleaved
/// (low-pass at even positions, high-pass at odd ones).
fn sd_1d(x: &mut [i16]) {
    let n = x.len();
    if n < 2 {
        return; // a lone sample at an even index passes through unchanged
    }
    let src = x.to_vec();
    let xe = |i: isize| src[pse(i, n)];
    // y[i + 1] holds Y(i) for i in −1..=n.
    let mut y = vec![0i16; n + 2];
    // Step 1: Y(2k+1) for −1 ≤ 2k+1 < n+1 (Annex F, eq. F-9).
    for i in (-1..=n as isize).filter(|i| i.rem_euclid(2) == 1) {
        y[(i + 1) as usize] = xe(i).wrapping_sub(floor_half(xe(i - 1), xe(i + 1)));
    }
    // Step 2: Y(2k) for 0 ≤ 2k < n (eq. F-10).
    for i in (0..n as isize).step_by(2) {
        let t = floor_quarter_rounded(y[i as usize], y[(i + 2) as usize]);
        y[(i + 1) as usize] = xe(i).wrapping_add(t);
    }
    x.copy_from_slice(&y[1..=n]);
}

/// `1D_SR`: inverse of [`sd_1d`].
fn sr_1d(y: &mut [i16]) {
    let n = y.len();
    if n < 2 {
        return;
    }
    let src = y.to_vec();
    let ye = |i: isize| src[pse(i, n)];
    // x[i] holds X(i) for i in 0..=n; X(n) is the extension sample step 2 needs when n is even.
    let mut x = vec![0i16; n + 1];
    // Step 1: even samples (eq. F-11).
    for i in (0..=n as isize).step_by(2) {
        x[i as usize] = ye(i).wrapping_sub(floor_quarter_rounded(ye(i - 1), ye(i + 1)));
    }
    // Step 2: odd samples (eq. F-12).
    for i in (1..n).step_by(2) {
        x[i] = ye(i as isize).wrapping_add(floor_half(x[i - 1], x[i + 1]));
    }
    y.copy_from_slice(&x[..n]);
}

/// Low-pass (even) samples first, then high-pass (odd) ones.
fn deinterleave(x: &[i16]) -> Vec<i16> {
    x.iter()
        .step_by(2)
        .chain(x.iter().skip(1).step_by(2))
        .copied()
        .collect()
}

fn interleave(x: &[i16]) -> Vec<i16> {
    let ns = x.len().div_ceil(2);
    (0..x.len())
        .map(|i| if i % 2 == 0 { x[i / 2] } else { x[ns + i / 2] })
        .collect()
}

/// Forward transform of a row-major `width × height` plane into the Mallat layout.
pub fn forward(data: &[i16], width: usize, height: usize, levels: usize) -> Vec<i16> {
    assert_eq!(data.len(), width * height);
    let geo = Geometry::new(width, height, levels);
    let mut a = data.to_vec();
    for l in 1..=levels {
        let (rw, rh) = geo.ll_size(l - 1);
        for x in 0..rw {
            let mut col: Vec<i16> = (0..rh).map(|y| a[y * width + x]).collect();
            sd_1d(&mut col);
            for (y, v) in deinterleave(&col).into_iter().enumerate() {
                a[y * width + x] = v;
            }
        }
        for y in 0..rh {
            let row = &mut a[y * width..y * width + rw];
            sd_1d(row);
            let out = deinterleave(row);
            row.copy_from_slice(&out);
        }
    }
    a
}

/// Inverse of [`forward`].
pub fn inverse(coeffs: &[i16], width: usize, height: usize, levels: usize) -> Vec<i16> {
    assert_eq!(coeffs.len(), width * height);
    let geo = Geometry::new(width, height, levels);
    let mut a = coeffs.to_vec();
    for l in (1..=levels).rev() {
        let (rw, rh) = geo.ll_size(l - 1);
        for y in 0..rh {
            let row = &mut a[y * width..y * width + rw];
            let mut x = interleave(row);
            sr_1d(&mut x);
            row.copy_from_slice(&x);
        }
        for x in 0..rw {
            let col: Vec<i16> = (0..rh).map(|y| a[y * width + x]).collect();
            let mut col = interleave(&col);
            sr_1d(&mut col);
            for (y, v) in col.into_iter().enumerate() {
                a[y * width + x] = v;
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-row images: only the horizontal transform runs. Values worked out by hand from the
    /// lifting equations, including floor rounding of negative sums.
    #[test]
    fn known_values() {
        let cases: [(&[i16], &[i16]); 5] = [
            (&[5], &[5]),
            (&[3, 7], &[5, 4]),
            (&[10, 20, 30, 40], &[10, 33, 0, 10]),
            (&[-3, -8, 1], &[-6, -2, -7]),
            (&[-3, 0, -2], &[-1, 0, 3]),
        ];
        for (input, expected) in cases {
            let n = input.len();
            assert_eq!(forward(input, n, 1, 1), expected, "forward {input:?}");
            assert_eq!(inverse(expected, n, 1, 1), input, "inverse {expected:?}");
            // The same signal as a column must give the same numbers.
            assert_eq!(forward(input, 1, n, 1), expected, "column {input:?}");
        }
    }

    #[test]
    fn pse_reflects_without_repeating_edges() {
        let idx: Vec<usize> = (-4..8).map(|i| pse(i, 4)).collect();
        assert_eq!(idx, [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]);
        assert_eq!(pse(-1, 2), 1);
        assert_eq!(pse(2, 2), 0);
    }
}
