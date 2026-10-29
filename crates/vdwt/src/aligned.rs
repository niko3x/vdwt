//! Cache-line aligned heap buffers.

use std::alloc::{self, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Alignment of every buffer in bytes: one cache line on Cortex-A55 and on x86.
pub const ALIGN: usize = 64;

/// Plain sample types for which an all-zero bit pattern is a valid value.
///
/// # Safety
/// Implementors must be `Copy` types without padding, invalid bit patterns or drop glue.
pub unsafe trait Pod: Copy + Default + Send + Sync + 'static {}

// SAFETY: primitive integers have no padding and accept every bit pattern.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for i16 {}
// SAFETY: as above.
unsafe impl Pod for u16 {}
// SAFETY: as above.
unsafe impl Pod for i32 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}

/// A fixed-length, zero-initialized buffer whose first element is [`ALIGN`]-byte aligned.
pub struct AlignedBuf<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: the buffer owns its elements exclusively, and `T: Pod` is `Send + Sync`.
unsafe impl<T: Pod> Send for AlignedBuf<T> {}
// SAFETY: as above.
unsafe impl<T: Pod> Sync for AlignedBuf<T> {}

impl<T: Pod> AlignedBuf<T> {
    /// Allocates `len` zeroed elements.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len,
            };
        }
        let layout = Self::layout(len);
        // SAFETY: `layout` has a nonzero size because `len > 0` and every `Pod` type is sized > 0.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr =
            NonNull::new(raw.cast::<T>()).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    fn layout(len: usize) -> Layout {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("buffer size overflows usize");
        Layout::from_size_align(bytes, ALIGN).expect("buffer too large")
    }
}

impl<T: Pod> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in `zeroed` with exactly this layout.
            unsafe { alloc::dealloc(self.ptr.as_ptr().cast(), Self::layout(self.len)) }
        }
    }
}

impl<T: Pod> Deref for AlignedBuf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized elements (or is dangling with `len == 0`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Pod> DerefMut for AlignedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Pod> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        let mut copy = Self::zeroed(self.len);
        copy.copy_from_slice(self);
        copy
    }
}

impl<T: Pod> fmt::Debug for AlignedBuf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_and_aligned() {
        for len in [1, 7, 64, 1000] {
            let buf = AlignedBuf::<i16>::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_eq!(buf.as_ptr() as usize % ALIGN, 0);
            assert!(buf.iter().all(|&v| v == 0));
        }
        assert!(AlignedBuf::<u8>::zeroed(0).is_empty());
    }
}
