//! Two-dimensional sample planes.

use std::fmt;

use crate::aligned::{ALIGN, AlignedBuf, Pod};

/// A `width × height` grid of samples. Every row starts on a 64-byte boundary; `stride` is the
/// distance between row starts, in samples (≥ `width`).
#[derive(Clone)]
pub struct Plane<T: Pod> {
    width: usize,
    height: usize,
    stride: usize,
    buf: AlignedBuf<T>,
}

impl<T: Pod> Plane<T> {
    /// A zero-filled plane. Both dimensions must be nonzero.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(
            width > 0 && height > 0,
            "plane dimensions must be nonzero, got {width}x{height}"
        );
        let stride = width.next_multiple_of(ALIGN / size_of::<T>());
        let len = stride
            .checked_mul(height)
            .expect("plane size overflows usize");
        Self {
            width,
            height,
            stride,
            buf: AlignedBuf::zeroed(len),
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Every sample, row after row `stride` apart, including the padding at the end of each
    /// row. For kernels that address rows by offset.
    pub fn data(&self) -> &[T] {
        &self.buf
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        self.row_span(y, 0, self.width)
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        self.row_span_mut(y, 0, self.width)
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) outside {}x{} plane",
            self.width,
            self.height
        );
        self.buf[y * self.stride + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) outside {}x{} plane",
            self.width,
            self.height
        );
        self.buf[y * self.stride + x] = value;
    }

    /// Samples `x0 .. x0 + len` of row `y`.
    #[inline]
    pub fn row_span(&self, y: usize, x0: usize, len: usize) -> &[T] {
        let start = self.span_start(y, x0, len);
        &self.buf[start..start + len]
    }

    /// Samples `x0 .. x0 + len` of row `y`, mutably.
    #[inline]
    pub fn row_span_mut(&mut self, y: usize, x0: usize, len: usize) -> &mut [T] {
        let start = self.span_start(y, x0, len);
        &mut self.buf[start..start + len]
    }

    /// The same span of three rows: `m` mutably, `a` and `b` shared. `a` and `b` may be the same
    /// row, but neither may be `m`. This is the access pattern of a vertical lifting step.
    #[inline]
    pub fn row_span_mut3(
        &mut self,
        m: usize,
        a: usize,
        b: usize,
        x0: usize,
        len: usize,
    ) -> (&mut [T], &[T], &[T]) {
        assert!(m != a && m != b, "row {m} would alias itself");
        let (sm, sa, sb) = (
            self.span_start(m, x0, len),
            self.span_start(a, x0, len),
            self.span_start(b, x0, len),
        );
        let base = self.buf.as_mut_ptr();
        // SAFETY: each span lies inside its own row (`span_start` checked `x0 + len <= width <=
        // stride`), and row `m` differs from rows `a` and `b`, so the mutable span overlaps neither
        // shared span. All three are derived from one exclusive borrow of the buffer.
        unsafe {
            (
                std::slice::from_raw_parts_mut(base.add(sm), len),
                std::slice::from_raw_parts(base.add(sa), len),
                std::slice::from_raw_parts(base.add(sb), len),
            )
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.buf.chunks_exact(self.stride).map(|r| &r[..self.width])
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let width = self.width;
        self.buf
            .chunks_exact_mut(self.stride)
            .map(move |r| &mut r[..width])
    }

    pub fn fill(&mut self, value: T) {
        self.buf.fill(value);
    }

    /// The plane as disjoint bands of `rows` rows (the last may be shorter), for threads.
    pub fn bands_mut(&mut self, rows: usize) -> Vec<RowsMut<'_, T>> {
        let (width, stride) = (self.width, self.stride);
        self.buf
            .chunks_mut(stride * rows.max(1))
            .enumerate()
            .map(|(i, data)| RowsMut {
                y0: i * rows.max(1),
                width,
                stride,
                data,
            })
            .collect()
    }

    /// Unsynchronized access to the rows, for parallel tasks that each write their own rows
    /// (see [`SharedRows`]).
    pub(crate) fn shared(&mut self) -> SharedRows<'_, T> {
        SharedRows {
            ptr: self.buf.as_mut_ptr(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            _plane: std::marker::PhantomData,
        }
    }

    /// Copies all samples from a plane of the same size.
    pub fn copy_from(&mut self, src: &Plane<T>) {
        assert_eq!(
            (self.width, self.height),
            (src.width, src.height),
            "plane size mismatch"
        );
        for (d, s) in self.rows_mut().zip(src.rows()) {
            d.copy_from_slice(s);
        }
    }

    #[inline]
    fn span_start(&self, y: usize, x0: usize, len: usize) -> usize {
        assert!(
            y < self.height && len <= self.width && x0 <= self.width - len,
            "span (y={y}, x0={x0}, len={len}) outside {}x{} plane",
            self.width,
            self.height
        );
        y * self.stride + x0
    }
}

/// Consecutive rows of a [`Plane`], mutably: see [`Plane::bands_mut`]. Rows keep their plane's
/// numbering.
pub struct RowsMut<'a, T> {
    y0: usize,
    width: usize,
    stride: usize,
    data: &'a mut [T],
}

impl<T> RowsMut<'_, T> {
    /// The plane rows held.
    pub fn rows(&self) -> std::ops::Range<usize> {
        self.y0..self.y0 + self.data.len().div_ceil(self.stride)
    }

    /// Row `y` of the plane, which must be one of [`Self::rows`].
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let start = (y - self.y0) * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Columns `x0 .. x0 + len` of rows `y .. y + rows` (all within [`Self::rows`]), as one
    /// slice whose rows start [`Self::stride`] apart.
    #[inline]
    pub fn block_mut(&mut self, y: usize, x0: usize, len: usize, rows: usize) -> &mut [T] {
        assert!(rows > 0 && x0 + len <= self.width);
        let start = (y - self.y0) * self.stride + x0;
        &mut self.data[start..start + (rows - 1) * self.stride + len]
    }

    /// Elements from one row to the next.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// A plane's rows for tasks running at once: [`Plane::shared`]. The plane stays mutably borrowed
/// while this lives, so these are the only accesses; the caller keeps them apart.
pub(crate) struct SharedRows<'a, T> {
    ptr: *mut T,
    width: usize,
    height: usize,
    stride: usize,
    _plane: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: access goes through the unsafe methods, whose callers guarantee that no row is written
// while another task reads or writes it.
unsafe impl<T: Send> Send for SharedRows<'_, T> {}
unsafe impl<T: Send> Sync for SharedRows<'_, T> {}

impl<T> SharedRows<'_, T> {
    /// Columns `x0 .. x0 + len` of row `y`.
    ///
    /// # Safety
    /// No other task may write these samples while the slice lives.
    #[inline]
    pub unsafe fn row(&self, y: usize, x0: usize, len: usize) -> &[T] {
        assert!(y < self.height && x0 + len <= self.width);
        // SAFETY: in bounds, and the caller rules out concurrent writes.
        unsafe { std::slice::from_raw_parts(self.ptr.add(y * self.stride + x0), len) }
    }

    /// Columns `x0 .. x0 + len` of row `y`, mutably.
    ///
    /// # Safety
    /// No other task may read or write these samples while the slice lives.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn row_mut(&self, y: usize, x0: usize, len: usize) -> &mut [T] {
        assert!(y < self.height && x0 + len <= self.width);
        // SAFETY: in bounds, and the caller rules out any other access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(y * self.stride + x0), len) }
    }
}

impl<T: Pod> fmt::Debug for Plane<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plane")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish()
    }
}

/// 8-bit samples → the signed, mid-grey-centred form the transform works on (`x − 128`).
pub fn center_u8(src: &Plane<u8>, dst: &mut Plane<i16>) {
    assert_eq!(
        (src.width(), src.height()),
        (dst.width(), dst.height()),
        "plane size mismatch"
    );
    for (d, s) in dst.rows_mut().zip(src.rows()) {
        center_row(s, d);
    }
}

/// [`center_u8`] of one row.
#[inline]
pub(crate) fn center_row(src: &[u8], dst: &mut [i16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = i16::from(s) - 128;
    }
}

/// A P-frame residual row, `cur − pred`.
#[inline]
pub(crate) fn residual_row(cur: &[u8], pred: &[u8], dst: &mut [i16]) {
    for ((d, &c), &p) in dst.iter_mut().zip(cur).zip(pred) {
        *d = i16::from(c) - i16::from(p);
    }
}

/// [`uncenter_to_u8`] of one row.
#[inline]
pub(crate) fn uncenter_row(src: &[i16], dst: &mut [u8]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.saturating_add(128).clamp(0, 255) as u8;
    }
}

/// Adds a residual row to a predicted row, clamping to 8 bits. Saturating: a corrupt stream can
/// carry any residual.
#[inline]
pub(crate) fn add_residual_row(res: &[i16], recon: &mut [u8]) {
    for (r, &d) in recon.iter_mut().zip(res) {
        *r = i16::from(*r).saturating_add(d).clamp(0, 255) as u8;
    }
}

/// Inverse of [`center_u8`], clamping to `0..=255`.
pub fn uncenter_to_u8(src: &Plane<i16>, dst: &mut Plane<u8>) {
    assert_eq!(
        (src.width(), src.height()),
        (dst.width(), dst.height()),
        "plane size mismatch"
    );
    for (d, s) in dst.rows_mut().zip(src.rows()) {
        uncenter_row(s, d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_aligned_and_independent() {
        let mut p = Plane::<i16>::new(37, 5);
        assert_eq!(p.stride() % (ALIGN / 2), 0);
        for y in 0..5 {
            assert_eq!(p.row(y).as_ptr() as usize % ALIGN, 0);
            p.row_mut(y).fill(y as i16);
        }
        let (m, a, b) = p.row_span_mut3(2, 1, 1, 3, 10);
        m.copy_from_slice(a);
        assert_eq!(b, a);
        assert_eq!(&p.row(2)[3..13], &[1; 10]);
        assert_eq!(p.row(2)[2], 2);
    }

    #[test]
    #[should_panic(expected = "alias")]
    fn row_span_mut3_rejects_aliasing() {
        let mut p = Plane::<i16>::new(8, 3);
        let _ = p.row_span_mut3(1, 1, 0, 0, 8);
    }

    #[test]
    fn bands_cover_the_plane() {
        let mut p = Plane::<u8>::new(10, 7);
        for mut band in p.bands_mut(3) {
            for y in band.rows() {
                band.row_mut(y).fill(y as u8);
            }
        }
        assert!((0..7).all(|y| p.row(y).iter().all(|&v| v == y as u8)));
        assert_eq!(p.bands_mut(3).last().unwrap().rows(), 6..7);
    }

    #[test]
    fn center_round_trip() {
        let mut p = Plane::<u8>::new(256, 1);
        for (x, v) in p.row_mut(0).iter_mut().enumerate() {
            *v = x as u8;
        }
        let mut c = Plane::new(256, 1);
        center_u8(&p, &mut c);
        assert_eq!((c.row(0)[0], c.row(0)[255]), (-128, 127));
        let mut back = Plane::new(256, 1);
        uncenter_to_u8(&c, &mut back);
        assert_eq!(back.row(0), p.row(0));
    }
}
