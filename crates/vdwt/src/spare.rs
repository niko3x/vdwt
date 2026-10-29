//! Per-thread spare buffers (DESIGN.md §10.3.2). Buffers a task or a frame needs again and again
//! (transform rows, band maps, search scratch) are taken from the calling thread's spares and
//! given back when done, instead of being allocated anew: musl's allocator handed such buffers'
//! pages back to the system and faulted them in again every frame, milliseconds a frame on the
//! A55. The pool threads live as long as the codec, so their spares carry from frame to frame.

use std::cell::RefCell;

/// Element types with spares.
pub(crate) trait Spare: Copy + Default + 'static {
    fn with_spares<R>(f: impl FnOnce(&mut Vec<Vec<Self>>) -> R) -> R;
}

macro_rules! spares {
    ($($t:ty),*) => {$(
        impl Spare for $t {
            fn with_spares<R>(f: impl FnOnce(&mut Vec<Vec<Self>>) -> R) -> R {
                thread_local! {
                    static SPARES: RefCell<Vec<Vec<$t>>> = const { RefCell::new(Vec::new()) };
                }
                SPARES.with(|s| f(&mut s.borrow_mut()))
            }
        }
    )*};
}

spares!(u8, u16, i16, i32, u64, bool);

/// Spares kept per thread and type.
const KEEP: usize = 32;

/// A buffer of `n` values from this thread's spares. Its contents are unspecified (whatever the
/// last user left, default where it grew): for buffers written before they are read.
pub(crate) fn take<T: Spare>(n: usize) -> Vec<T> {
    let mut v = T::with_spares(|s| s.pop()).unwrap_or_default();
    v.resize(n, T::default());
    v
}

/// [`take`] with every value `T::default()`.
pub(crate) fn take_zeroed<T: Spare>(n: usize) -> Vec<T> {
    let mut v = T::with_spares(|s| s.pop()).unwrap_or_default();
    v.clear();
    v.resize(n, T::default());
    v
}

/// Returns a buffer to this thread's spares.
pub(crate) fn give<T: Spare>(v: Vec<T>) {
    if v.capacity() == 0 {
        return;
    }
    T::with_spares(|s| {
        if s.len() < KEEP {
            s.push(v);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_buffers_are_zero_and_spares_return() {
        let mut v = take::<i32>(100);
        v.fill(7);
        let p = v.as_ptr();
        give(v);
        let w = take_zeroed::<i32>(50);
        assert_eq!(w.as_ptr(), p);
        assert!(w.iter().all(|&x| x == 0));
        give(w);
    }
}
