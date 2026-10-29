//! Reduced-resolution luma for motion estimation (encoder only).

use crate::Plane;
use crate::pool::Pool;

/// Rows per task of [`downsample`].
const BAND_ROWS: usize = 32;

/// 2×2 box downsampling with rounding, in bands of rows on `pool`. An odd last row or column
/// averages with itself.
pub(crate) fn downsample(pool: &Pool, src: &Plane<u8>, dst: &mut Plane<u8>) {
    let (w, h) = (src.width(), src.height());
    assert_eq!(
        (dst.width(), dst.height()),
        (w.div_ceil(2), h.div_ceil(2)),
        "downsampled size"
    );
    let (dw, dh) = (dst.width(), dst.height());
    let rows = dst.shared();
    pool.run(dh.div_ceil(BAND_ROWS), &|band| {
        for y in band * BAND_ROWS..((band + 1) * BAND_ROWS).min(dh) {
            let a = src.row(2 * y);
            let b = src.row((2 * y + 1).min(h - 1));
            // SAFETY: each row is written by one task.
            let row = unsafe { rows.row_mut(y, 0, dw) };
            let at = |x0: usize, x1: usize| {
                let sum = u16::from(a[x0]) + u16::from(a[x1]) + u16::from(b[x0]) + u16::from(b[x1]);
                ((sum + 2) >> 2) as u8
            };
            // Whole pairs in a loop that vectorizes, then an odd last column.
            let pairs = w / 2;
            for ((d, a), b) in row[..pairs]
                .iter_mut()
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let sum = u16::from(a[0]) + u16::from(a[1]) + u16::from(b[0]) + u16::from(b[1]);
                *d = ((sum + 2) >> 2) as u8;
            }
            if w % 2 == 1 {
                row[pairs] = at(w - 1, w - 1);
            }
        }
    });
}

/// Luma at ½ and ¼ resolution.
#[derive(Clone, Debug)]
pub(crate) struct Pyramid {
    pub half: Plane<u8>,
    pub quarter: Plane<u8>,
}

impl Pyramid {
    pub fn new(width: usize, height: usize) -> Self {
        let (hw, hh) = (width.div_ceil(2), height.div_ceil(2));
        Self {
            half: Plane::new(hw, hh),
            quarter: Plane::new(hw.div_ceil(2), hh.div_ceil(2)),
        }
    }

    pub fn build(&mut self, pool: &Pool, luma: &Plane<u8>) {
        downsample(pool, luma, &mut self.half);
        downsample(pool, &self.half, &mut self.quarter);
    }
}
