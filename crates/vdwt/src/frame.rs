//! Pictures.

use crate::Plane;

/// An 8-bit 4:2:0 picture. Odd luma dimensions round the chroma size up.
#[derive(Clone, Debug)]
pub struct Frame {
    pub y: Plane<u8>,
    pub u: Plane<u8>,
    pub v: Plane<u8>,
}

impl Frame {
    /// A black-luma, zero-chroma frame (all samples 0).
    pub fn new(width: usize, height: usize) -> Self {
        let (cw, ch) = chroma_size(width, height);
        Self {
            y: Plane::new(width, height),
            u: Plane::new(cw, ch),
            v: Plane::new(cw, ch),
        }
    }

    pub fn width(&self) -> usize {
        self.y.width()
    }

    pub fn height(&self) -> usize {
        self.y.height()
    }

    pub fn planes(&self) -> [&Plane<u8>; 3] {
        [&self.y, &self.u, &self.v]
    }

    pub fn planes_mut(&mut self) -> [&mut Plane<u8>; 3] {
        [&mut self.y, &mut self.u, &mut self.v]
    }
}

/// Chroma plane size for a 4:2:0 frame of the given luma size.
pub fn chroma_size(width: usize, height: usize) -> (usize, usize) {
    (width.div_ceil(2), height.div_ceil(2))
}
