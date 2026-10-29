//! Subband layout of a multi-level decomposition.

/// Deepest decomposition supported.
pub const MAX_LEVELS: usize = 8;

/// Subband orientation, JPEG 2000 naming: the first letter is the horizontal filter, the second
/// the vertical one. HL holds vertical edges (horizontal high-pass) and sits top-right.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orient {
    LL,
    HL,
    LH,
    HH,
}

/// One subband: a rectangle of the coefficient plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Subband {
    /// Decomposition level that produced the band (1 = finest); for LL, the total level count.
    pub level: usize,
    pub orient: Orient,
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
}

/// Size and depth of a decomposition. Coefficients use the Mallat layout: each level splits the
/// low band of the previous one in place, so the region a level transforms is always at the
/// origin and the coarsest LL ends up top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    width: usize,
    height: usize,
    levels: usize,
}

impl Geometry {
    pub fn new(width: usize, height: usize, levels: usize) -> Self {
        assert!(width > 0 && height > 0, "empty plane");
        assert!(
            levels <= MAX_LEVELS,
            "{levels} levels requested, at most {MAX_LEVELS} supported"
        );
        Self {
            width,
            height,
            levels,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Size of the low band after `l` levels (`l = 0` is the whole plane). Each level keeps
    /// `⌈n/2⌉` low-pass samples, since the signal always starts at an even index.
    pub fn ll_size(&self, l: usize) -> (usize, usize) {
        (self.width.div_ceil(1 << l), self.height.div_ceil(1 << l))
    }

    /// Every subband, coarsest first: `LL_L`, then `HL`, `LH`, `HH` for levels `L` down to 1.
    /// Bands can be empty when a dimension shrinks to 1.
    pub fn subbands(&self) -> Vec<Subband> {
        let (w, h) = self.ll_size(self.levels);
        let mut bands = vec![Subband {
            level: self.levels,
            orient: Orient::LL,
            x0: 0,
            y0: 0,
            width: w,
            height: h,
        }];
        for level in (1..=self.levels).rev() {
            let (pw, ph) = self.ll_size(level - 1);
            let (lw, lh) = self.ll_size(level);
            let band = |orient, x0, y0, width, height| Subband {
                level,
                orient,
                x0,
                y0,
                width,
                height,
            };
            bands.push(band(Orient::HL, lw, 0, pw - lw, lh));
            bands.push(band(Orient::LH, 0, lh, lw, ph - lh));
            bands.push(band(Orient::HH, lw, lh, pw - lw, ph - lh));
        }
        bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subbands_tile_the_plane() {
        for (w, h) in [(1, 1), (1, 9), (7, 1), (1280, 720), (641, 359), (33, 2)] {
            for levels in 0..=MAX_LEVELS {
                let mut hits = vec![0u8; w * h];
                for b in Geometry::new(w, h, levels).subbands() {
                    for y in b.y0..b.y0 + b.height {
                        for x in b.x0..b.x0 + b.width {
                            hits[y * w + x] += 1;
                        }
                    }
                }
                assert!(hits.iter().all(|&c| c == 1), "{w}x{h}, {levels} levels");
            }
        }
    }

    #[test]
    fn hd_luma_sizes() {
        let g = Geometry::new(1280, 720, 5);
        assert_eq!(g.ll_size(5), (40, 23));
        let hh1 = g
            .subbands()
            .into_iter()
            .find(|b| b.level == 1 && b.orient == Orient::HH)
            .unwrap();
        assert_eq!(
            (hh1.x0, hh1.y0, hh1.width, hh1.height),
            (640, 360, 640, 360)
        );
    }
}
