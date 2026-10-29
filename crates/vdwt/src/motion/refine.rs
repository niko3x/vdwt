//! Overlapped-prediction-aware refinement of the motion quadtree's leaf offsets (encoder only,
//! DESIGN.md §6.2).
//!
//! The partition search judges a leaf by block matching: its nodes' own pixels, displaced by the
//! leaf's vector. Overlapped prediction blends every pixel from the four nodes around it, so the
//! best vector for a node depends on its neighbours'. This pass re-chooses the offset of every
//! small leaf (up to 2×2 nodes) by the squared error of the blended prediction over every pixel
//! its nodes touch, the other nodes held fixed, plus λ times its rate. Candidates: no offset,
//! the four neighbours' offsets, and steps of a quarter and a whole pixel along each axis
//! ([`STEPS`]); then whole-pixel steps from the best while they win, and quarter-pixel ones
//! ([`WHOLE_ROUNDS`]). Passes repeat over the leaves near a change. On real footage this saves
//! 7–10% of the bits (§10.1). Without `descent` (the light tier) only the first candidates are
//! tried — no offset and the neighbours' — for 5% of the bits at a fifth of the full pass's time.
//!
//! Errors use bilinear interpolation in 1/64; they only rank candidates. The running prediction
//! is kept as integer sums `Σ w·s` (`w = (C − |dx|)(C − |dy|)`, `s` in 1/64 of a sample value)
//! and updated exactly when an offset changes. A candidate's error is summed a few rows at a
//! time and dropped as soon as it cannot win.
//!
//! **Parallelism.** Leaves are visited in raster order of their top-left nodes. A leaf changes
//! only pixels within one node of it and reads only its neighbours, so it interacts only with
//! leaves starting in the two node rows above (or below) and less than [`LEAD`] columns to its
//! right (left): each leaf runs once those rows are that far ahead, a wavefront. Tasks take two
//! node rows each, the lower one `LEAD` columns behind (the upper row of a pair holds the larger
//! leaves, so pairs balance). The result is that of one raster scan, whatever the thread count.

use std::sync::atomic::{AtomicBool, Ordering};

use super::MotionField;
use super::tree;
use crate::Plane;
use crate::pool::{Pool, Progress};

/// Rows between checks whether a candidate can still win.
const CHECK: usize = 4;
/// Leaves of up to `2^MAX_LEVEL × 2^MAX_LEVEL` nodes are refined: in larger ones most pixels are
/// predicted by the leaf alone, where block matching is already exact (refining them too gained
/// nothing).
const MAX_LEVEL: u8 = 1;
/// How many node columns the row above must be ahead of a leaf (its leaves start at most two
/// columns right of the leaf's last node and reach one node further).
const LEAD: usize = 4;
/// After the candidates, rounds of whole-pixel steps around the best while they win, then of
/// quarter-pixel ones. The partition's offsets come from block matching, often pixels from the
/// best for the blended prediction: descending to it saves 3.6% of the bits at 1080p and 2.3%
/// at 720p for a third more encoding time (§10.1); quarter-pixel rounds beyond two save nothing.
const WHOLE_ROUNDS: usize = 32;
const QUARTER_ROUNDS: usize = 2;
/// Offset steps tried around a leaf's current offset (1/8 pel; offsets are whole quarter pels):
/// a quarter and a whole pixel along each axis, and with `half_pel` half a pixel (0.1–0.4%
/// fewer bits for a quarter more time; diagonals changed nothing).
const STEPS: [(i32, i32); 12] = [
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (8, 0),
    (-8, 0),
    (0, 8),
    (0, -8),
    (4, 0),
    (-4, 0),
    (0, 4),
    (0, -4),
];

/// Refines the offsets of `field` (partitioned, with `pred` the global prediction per node) for
/// `cur` against `reference`, in at most `passes` passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine(
    pool: &Pool,
    cur: &Plane<u8>,
    reference: &Plane<u8>,
    field: &mut MotionField,
    pred: &[[i32; 2]],
    lambda: f64,
    passes: usize,
    half_pel: bool,
    descent: bool,
) {
    let grid = field.grid;
    let (nx, ny) = (grid.nx, grid.ny);
    let ctx = Context {
        cur,
        reference,
        c: grid.spacing() as i32,
        nx,
        steps: match (descent, half_pel) {
            (false, _) => &[],
            (true, true) => &STEPS,
            (true, false) => &STEPS[..8],
        },
        descent,
        weight: {
            let c = grid.spacing() as i32;
            let tent: Vec<i32> = (0..2 * c).map(|k| c - (k - c).abs()).collect();
            tent.iter()
                .flat_map(|&wy| tent.iter().map(move |&wx| wy * wx))
                .collect()
        },
    };
    let (w, h) = (cur.width(), cur.height());
    // The prediction sums, one cell row per task (a frame-sized buffer from the spares).
    let mut sum = crate::spare::take_zeroed::<i32>(w * h);
    {
        let c = ctx.c as usize;
        let mut bands: Vec<&mut [i32]> = sum.chunks_mut(c * w).collect();
        let field = &*field;
        pool.for_each_mut(&mut bands, |j, band| {
            let mut sc = Scratch::take();
            let rows = (j * c) as i32..((j + 1) * c).min(h) as i32;
            let out = Area {
                x0: 0,
                y0: rows.start,
                w,
                h: (rows.end - rows.start) as usize,
            };
            for n in (j * nx..(j + 2).min(ny) * nx).filter(|&n| n < grid.nodes()) {
                ctx.contribute(n, field.mv[n], field.intra[n], &out, band, &mut sc);
            }
            sc.give();
        });
    }
    let leaves = leaves(field);
    let mut leaf_of = vec![0; grid.nodes()];
    for (k, &(i0, j0, size)) in leaves.iter().enumerate() {
        for j in j0..(j0 + size).min(ny) {
            for i in i0..(i0 + size).min(nx) {
                leaf_of[j * nx + i] = k;
            }
        }
    }
    // Leaves by row of their top-left node, in raster order.
    let mut by_row: Vec<Vec<usize>> = vec![Vec::new(); ny];
    for (k, &(_, j0, _)) in leaves.iter().enumerate() {
        by_row[j0].push(k);
    }
    let active: Vec<AtomicBool> = leaves.iter().map(|_| AtomicBool::new(true)).collect();
    let next: Vec<AtomicBool> = leaves.iter().map(|_| AtomicBool::new(false)).collect();
    let shared = Shared {
        sum: sum.as_mut_ptr(),
        mv: field.mv.as_mut_ptr(),
    };
    let (intra, depth) = (&field.intra, &field.depth);
    for _ in 0..passes {
        let changed = AtomicBool::new(false);
        // Per row, the column every leaf starting left of which is done.
        let progress = Progress::new(ny);
        pool.run(ny.div_ceil(2), &|t| {
            let mut sc = Scratch::take();
            // Two rows, the lower one `LEAD` columns behind (raster order still).
            let mut order: Vec<(usize, usize, usize)> = Vec::new();
            for (j, row) in by_row.iter().enumerate().skip(2 * t).take(2) {
                let lag = (j - 2 * t) * LEAD;
                order.extend(
                    row.iter()
                        .enumerate()
                        .map(|(q, &k)| (leaves[k].0 + lag, j, q)),
                );
            }
            order.sort_unstable();
            for &(_, j, q) in &order {
                let row = &by_row[j];
                let k = row[q];
                let (i0, j0, size) = leaves[k];
                let n0 = j0 * nx + i0;
                if active[k].load(Ordering::Relaxed) && !intra[n0] && depth[n0] <= MAX_LEVEL {
                    // The leaves it interacts with in the two rows above must be done.
                    let need = (i0 + LEAD).min(nx);
                    for above in j.saturating_sub(2)..j {
                        progress.wait(above, need);
                    }
                    // SAFETY: a leaf runs only while the leaves it interacts with are done
                    // or wait for it (module documentation).
                    let moved =
                        unsafe { ctx.leaf(&shared, pred, lambda, (i0, j0, size), w, h, &mut sc) };
                    if moved {
                        changed.store(true, Ordering::Relaxed);
                        for j in j0.saturating_sub(2)..(j0 + size + 2).min(ny) {
                            for i in i0.saturating_sub(2)..(i0 + size + 2).min(nx) {
                                next[leaf_of[j * nx + i]].store(true, Ordering::Relaxed);
                            }
                        }
                    }
                }
                let done = row.get(q + 1).map_or(nx, |&k| leaves[k].0);
                progress.set(j, done);
            }
            for row in (2 * t..2 * t + 2).filter(|&r| r < ny) {
                progress.set(row, nx);
            }
            sc.give();
        });
        if !changed.load(Ordering::Relaxed) {
            break;
        }
        for (a, n) in active.iter().zip(&next) {
            a.store(n.swap(false, Ordering::Relaxed), Ordering::Relaxed);
        }
    }
    crate::spare::give(sum);
}

/// The prediction sums and the node vectors, written by strip tasks.
struct Shared {
    sum: *mut i32,
    mv: *mut [i32; 2],
}

// SAFETY: tasks write disjoint parts (see `refine`).
unsafe impl Sync for Shared {}

fn median(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

/// Every leaf as its top-left node and size in nodes, in raster order of top-left nodes.
fn leaves(field: &MotionField) -> Vec<(usize, usize, usize)> {
    let grid = field.grid;
    let (nx, ny) = (grid.nx, grid.ny);
    let mut seen = vec![false; grid.nodes()];
    let mut out = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            if seen[j * nx + i] {
                continue;
            }
            let size = 1usize << field.depth[j * nx + i];
            for jj in j..(j + size).min(ny) {
                for ii in i..(i + size).min(nx) {
                    seen[jj * nx + ii] = true;
                }
            }
            out.push((i, j, size));
        }
    }
    out
}

/// A rectangle of pixels.
#[derive(Clone, Copy)]
struct Area {
    x0: usize,
    y0: i32,
    w: usize,
    h: usize,
}

/// Per-task buffers.
#[derive(Default)]
struct Scratch {
    target: Vec<i32>,
    old: Vec<i32>,
    trial: Vec<i32>,
    best: Vec<i32>,
    rows: Vec<u8>,
    nodes: Vec<usize>,
    candidates: Vec<[i32; 2]>,
}

thread_local! {
    /// This thread's scratch between tasks and frames (made per task, its buffers regrew and
    /// faulted pages in every frame).
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

impl Scratch {
    fn take() -> Self {
        SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    fn give(self) {
        SCRATCH.with(|s| *s.borrow_mut() = self);
    }
}

struct Context<'a> {
    /// The offset steps tried.
    steps: &'static [(i32, i32)],
    /// Whether the candidates are followed by whole- and quarter-pixel descent.
    descent: bool,
    cur: &'a Plane<u8>,
    reference: &'a Plane<u8>,
    c: i32,
    nx: usize,
    /// A node's weights over its support, `2C × 2C`: `(C − |dx|)(C − |dy|)` from its zero-weight
    /// first column and row.
    weight: Vec<i32>,
}

/// One row of a node's contribution within an area: `Σ weight·s` with `s` the bilinear
/// interpolation of `a` and `b` (`len + 1` samples each) by `wts`, in 1/64.
struct Row<'a> {
    /// The row within the area and the first column within it.
    y: usize,
    x: usize,
    weight: &'a [i32],
    a: &'a [u8],
    b: &'a [u8],
    wts: [i32; 4],
}

impl Context<'_> {
    /// Node `n`'s support within `area` at `mv` as one block, when every reference sample it
    /// reads is inside the plane and its rows are whole groups of 16 (the usual case).
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    #[inline(always)]
    fn block(&self, n: usize, mv: [i32; 2], area: &Area) -> Option<Block<'_>> {
        let c = self.c;
        let (xn, yn) = ((n % self.nx) as i32 * c, (n / self.nx) as i32 * c);
        let (w, h) = (self.cur.width() as i32, self.cur.height() as i32);
        let xa = (xn - c).max(area.x0 as i32);
        let xb = (xn + c).min((area.x0 + area.w) as i32).min(w);
        let ya = (yn - c).max(area.y0);
        let yb = (yn + c).min(area.y0 + area.h as i32).min(h);
        let len = xb - xa;
        let (ix, iy) = (mv[0] >> 3, mv[1] >> 3);
        let reference = self.reference;
        let (rw, rh) = (reference.width() as i32, reference.height() as i32);
        if len <= 0
            || len % 16 != 0
            || ya >= yb
            || xa + ix < 0
            || xb + ix >= rw
            || ya + iy < 0
            || yb + iy >= rh
        {
            return None;
        }
        let (fx, fy) = (mv[0] & 7, mv[1] & 7);
        let span = 2 * c as usize;
        let stride = reference.stride();
        let first = (ya + iy) as usize * stride + (xa + ix) as usize;
        let rows = (yb - ya) as usize;
        Some(Block {
            y: (ya - area.y0) as usize,
            x: xa as usize - area.x0,
            rows,
            len: len as usize,
            weight: &self.weight[(ya - (yn - c)) as usize * span + (xa - (xn - c)) as usize..],
            weight_stride: span,
            // Rows `first..` and the one below the last, `len + 1` samples each.
            reference: &reference.data()[first..first + rows * stride + len as usize + 1],
            stride,
            wts: [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy],
        })
    }

    /// Calls `f` with every row of node `n`'s support within `area` at `mv`, while it returns
    /// true.
    #[inline(always)]
    fn rows(
        &self,
        n: usize,
        mv: [i32; 2],
        area: &Area,
        sc: &mut Vec<u8>,
        mut f: impl FnMut(Row) -> bool,
    ) {
        let c = self.c;
        let (xn, yn) = ((n % self.nx) as i32 * c, (n / self.nx) as i32 * c);
        let (w, h) = (self.cur.width() as i32, self.cur.height() as i32);
        // The node's support within the area, from its zero-weight first column and row (which
        // keeps rows a multiple of 8 wide).
        let xa = (xn - c).max(area.x0 as i32);
        let xb = (xn + c).min((area.x0 + area.w) as i32).min(w);
        let ya = (yn - c).max(area.y0);
        let yb = (yn + c).min(area.y0 + area.h as i32).min(h);
        if xa >= xb || ya >= yb {
            return;
        }
        let len = (xb - xa) as usize;
        let (ix, iy) = (mv[0] >> 3, mv[1] >> 3);
        let (fx, fy) = (mv[0] & 7, mv[1] & 7);
        let wts = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
        let reference = self.reference;
        let (rw, rh) = (reference.width() as i32, reference.height() as i32);
        let inside_x = xa + ix >= 0 && xb + ix < rw;
        let span = 2 * c as usize;
        let column = (xa - (xn - c)) as usize;
        for y in ya..yb {
            let weight = &self.weight[(y - (yn - c)) as usize * span + column..][..len];
            let row = |yy: i32| reference.row(yy.clamp(0, rh - 1) as usize);
            let (r0, r1) = (row(y + iy), row(y + iy + 1));
            let (a, b): (&[u8], &[u8]) = if inside_x {
                let s = (xa + ix) as usize;
                (&r0[s..s + len + 1], &r1[s..s + len + 1])
            } else {
                sc.clear();
                for r in [r0, r1] {
                    sc.extend((0..=len as i32).map(|k| r[(xa + ix + k).clamp(0, rw - 1) as usize]));
                }
                let (a, b) = sc.split_at(len + 1);
                (a, b)
            };
            let more = f(Row {
                y: (y - area.y0) as usize,
                x: xa as usize - area.x0,
                weight,
                a,
                b,
                wts,
            });
            if !more {
                return;
            }
        }
    }

    /// Adds node `n`'s weighted contribution at `mv` (or as intra) to `out`, which covers
    /// `area` (row stride `area.w`).
    fn contribute(
        &self,
        n: usize,
        mv: [i32; 2],
        intra: bool,
        area: &Area,
        out: &mut [i32],
        sc: &mut Scratch,
    ) {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { self.contribute_avx2(n, mv, intra, area, out, sc) };
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if !intra && let Some(b) = self.block(n, mv, area) {
                // SAFETY: NEON is present.
                return unsafe { neon::contribute(&b, out, area.w) };
            }
            return self.contribute_any(n, mv, intra, area, out, sc, |o, r| {
                // SAFETY: NEON is present.
                let done = unsafe { neon::contribute_row(o, r) };
                for (k, o) in o.iter_mut().enumerate().skip(done) {
                    *o += r.weight[k] * sample(r, k);
                }
            });
        }
        #[allow(unreachable_code)]
        self.contribute_any(n, mv, intra, area, out, sc, |o, r| {
            for (k, o) in o.iter_mut().enumerate() {
                *o += r.weight[k] * sample(r, k);
            }
        })
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn contribute_avx2(
        &self,
        n: usize,
        mv: [i32; 2],
        intra: bool,
        area: &Area,
        out: &mut [i32],
        sc: &mut Scratch,
    ) {
        if !intra && let Some(b) = self.block(n, mv, area) {
            // SAFETY: AVX2 is present.
            return unsafe { x86::contribute(&b, out, area.w) };
        }
        self.contribute_any(n, mv, intra, area, out, sc, |o, r| {
            // SAFETY: AVX2 is present.
            let done = unsafe { x86::contribute_row(o, r) };
            for (k, o) in o.iter_mut().enumerate().skip(done) {
                *o += r.weight[k] * sample(r, k);
            }
        })
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn contribute_any(
        &self,
        n: usize,
        mv: [i32; 2],
        intra: bool,
        area: &Area,
        out: &mut [i32],
        sc: &mut Scratch,
        row: impl Fn(&mut [i32], &Row),
    ) {
        self.rows(n, mv, area, &mut sc.rows, |r| {
            let o = &mut out[r.y * area.w + r.x..][..r.weight.len()];
            if intra {
                for (o, &w) in o.iter_mut().zip(r.weight) {
                    *o += w * (128 * 64);
                }
            } else {
                row(o, &r);
            }
            true
        });
    }

    /// The squared error of `target` (over `area`, which node `n`'s support covers exactly)
    /// less node `n`'s contribution at `mv`, or a partial sum for which `stop` holds.
    fn node_error(
        &self,
        n: usize,
        mv: [i32; 2],
        area: &Area,
        target: &[i32],
        sc: &mut Scratch,
        stop: impl Fn(u64) -> bool,
    ) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { self.node_error_avx2(n, mv, area, target, sc, stop) };
        }
        let shift = 2 * self.c.trailing_zeros();
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if let Some(b) = self.block(n, mv, area) {
                debug_assert!(b.x == 0 && b.len == area.w && b.y == 0);
                // SAFETY: NEON is present.
                return unsafe { neon::node_error(&b, target, shift, stop) };
            }
            let mut total = 0;
            self.rows(n, mv, area, &mut sc.rows, |r| {
                debug_assert!(r.x == 0 && r.weight.len() == area.w);
                let t = &target[r.y * area.w..][..area.w];
                // SAFETY: NEON is present.
                let (sum, done) = unsafe { neon::node_error_row(t, &r, shift) };
                total += sum
                    + (done..t.len())
                        .map(|k| error_at(t, &r, k, shift))
                        .sum::<u64>();
                r.y % CHECK != CHECK - 1 || !stop(total)
            });
            return total;
        }
        #[allow(unreachable_code)]
        let mut total = 0;
        self.rows(n, mv, area, &mut sc.rows, |r| {
            debug_assert!(r.x == 0 && r.weight.len() == area.w);
            let t = &target[r.y * area.w..][..area.w];
            total += (0..t.len()).map(|k| error_at(t, &r, k, shift)).sum::<u64>();
            r.y % CHECK != CHECK - 1 || !stop(total)
        });
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn node_error_avx2(
        &self,
        n: usize,
        mv: [i32; 2],
        area: &Area,
        target: &[i32],
        sc: &mut Scratch,
        stop: impl Fn(u64) -> bool,
    ) -> u64 {
        let shift = 2 * self.c.trailing_zeros();
        if let Some(b) = self.block(n, mv, area) {
            debug_assert!(b.x == 0 && b.len == area.w && b.y == 0);
            // SAFETY: AVX2 is present.
            return unsafe { x86::node_error(&b, target, shift, stop) };
        }
        let mut acc = x86::Sum::default();
        let mut tail = 0;
        self.rows(n, mv, area, &mut sc.rows, |r| {
            debug_assert!(r.x == 0 && r.weight.len() == area.w);
            let t = &target[r.y * area.w..][..area.w];
            // SAFETY: AVX2 is present.
            let done = unsafe { x86::node_error_row(t, &r, shift, &mut acc) };
            tail += (done..t.len())
                .map(|k| error_at(t, &r, k, shift))
                .sum::<u64>();
            r.y % CHECK != CHECK - 1 || !stop(acc.total() + tail)
        });
        acc.total() + tail
    }

    /// Squared error (in 1/64² sample values) of `target − trial`, values in `C²·64` units.
    fn error(&self, target: &[i32], trial: &[i32], area: &Area) -> u64 {
        let shift = 2 * self.c.trailing_zeros();
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present.
            return unsafe { error_avx2(target, trial, area.w, shift) };
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        return error_neon(target, trial, area.w, shift);
        #[allow(unreachable_code)]
        error_any(target, trial, area.w, shift)
    }

    /// Re-chooses the offset of leaf `(i0, j0, size)`; returns whether it changed.
    ///
    /// # Safety
    /// No other thread may touch the pixels within one node of the leaf, nor its nodes and
    /// their neighbours.
    #[allow(clippy::too_many_arguments)]
    unsafe fn leaf(
        &self,
        shared: &Shared,
        pred: &[[i32; 2]],
        lambda: f64,
        at: (usize, usize, usize),
        w: usize,
        h: usize,
        sc: &mut Scratch,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            // SAFETY: AVX2 is present; the rest as the caller promises.
            return unsafe { self.leaf_avx2(shared, pred, lambda, at, w, h, sc) };
        }
        // SAFETY: as the caller promises.
        unsafe { self.leaf_any(shared, pred, lambda, at, w, h, sc) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn leaf_avx2(
        &self,
        shared: &Shared,
        pred: &[[i32; 2]],
        lambda: f64,
        at: (usize, usize, usize),
        w: usize,
        h: usize,
        sc: &mut Scratch,
    ) -> bool {
        // SAFETY: as the caller promises.
        unsafe { self.leaf_any(shared, pred, lambda, at, w, h, sc) }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn leaf_any(
        &self,
        shared: &Shared,
        pred: &[[i32; 2]],
        lambda: f64,
        (i0, j0, size): (usize, usize, usize),
        w: usize,
        h: usize,
        sc: &mut Scratch,
    ) -> bool {
        let (nx, c) = (self.nx, self.c);
        let ny = pred.len() / nx;
        // SAFETY: (all below) as the caller promises.
        let mv = |n: usize| unsafe { *shared.mv.add(n) };
        let delta_of = |n: usize| {
            let v = mv(n);
            [v[0] - pred[n][0], v[1] - pred[n][1]]
        };
        let n0 = j0 * nx + i0;
        sc.nodes.clear();
        for j in j0..(j0 + size).min(ny) {
            sc.nodes
                .extend((i0..(i0 + size).min(nx)).map(|i| j * nx + i));
        }
        let delta = delta_of(n0);
        let left = (i0 > 0).then(|| delta_of(n0 - 1));
        let up = (j0 > 0).then(|| delta_of(n0 - nx));
        let upright = (j0 > 0 && i0 + size < nx).then(|| delta_of(n0 - nx + size));
        let new_pred = match (left, up, upright) {
            (Some(l), Some(u), Some(r)) => std::array::from_fn(|c| median(l[c], u[c], r[c])),
            (Some(l), _, _) => l,
            (None, Some(u), _) => u,
            _ => [0; 2],
        };
        // The pixels the leaf's nodes touch.
        let px = |v: i32, len: usize| v.clamp(0, len as i32);
        let (x0, x1) = (px((i0 as i32 - 1) * c, w), px((i0 + size) as i32 * c, w));
        let (y0, y1) = (px((j0 as i32 - 1) * c, h), px((j0 + size) as i32 * c, h));
        if x0 >= x1 || y0 >= y1 {
            return false;
        }
        let area = Area {
            x0: x0 as usize,
            y0,
            w: (x1 - x0) as usize,
            h: (y1 - y0) as usize,
        };
        let cells = area.w * area.h;
        // The leaf's contribution now, and the target without it.
        let nodes = std::mem::take(&mut sc.nodes);
        let mut old = std::mem::take(&mut sc.old);
        old.clear();
        old.resize(cells, 0);
        for &n in &nodes {
            self.contribute(n, mv(n), false, &area, &mut old, sc);
        }
        let k = 64 * c * c;
        sc.target.clear();
        for y in 0..area.h {
            let row = &self.cur.row(y0 as usize + y)[area.x0..area.x0 + area.w];
            // SAFETY: the area is inside the plane.
            let sums = unsafe {
                std::slice::from_raw_parts(shared.sum.add((y0 as usize + y) * w + area.x0), area.w)
            };
            let o = &old[y * area.w..(y + 1) * area.w];
            sc.target.extend(
                row.iter()
                    .zip(sums)
                    .zip(o)
                    .map(|((&v, &s), &o)| k * i32::from(v) - s + o),
            );
        }
        let bits = |d: [i32; 2]| tree::leaf_bits(d, left, up, new_pred);
        let scale = 1.0 / 4096.0;
        let mut best = (
            delta,
            self.error(&sc.target, &old, &area) as f64 * scale + lambda * bits(delta),
        );
        let mut candidates = std::mem::take(&mut sc.candidates);
        candidates.clear();
        candidates.push([0, 0]);
        candidates.extend(left);
        candidates.extend(up);
        if i0 + size < nx {
            candidates.push(delta_of(n0 + size));
        }
        if j0 + size < ny {
            candidates.push(delta_of(n0 + size * nx));
        }
        candidates.extend(
            self.steps
                .iter()
                .map(|(dx, dy)| [delta[0] + dx, delta[1] + dy]),
        );
        candidates.sort_unstable();
        candidates.dedup();
        let mut trial = std::mem::take(&mut sc.trial);
        let mut kept = std::mem::take(&mut sc.best);
        // The candidates, then descent from the best: whole-pixel steps while they win, then
        // quarter-pixel ones.
        let (mut whole, mut quarter) = if self.descent {
            (WHOLE_ROUNDS, QUARTER_ROUNDS)
        } else {
            (0, 0)
        };
        let mut fine = false;
        let mut centre = delta;
        let mut next = 0;
        loop {
            if next == candidates.len() {
                let moved = best.0 != centre;
                centre = best.0;
                let steps = if !fine && moved && whole > 0 {
                    whole -= 1;
                    &STEPS[4..8]
                } else if quarter > 0 && (moved || !fine) {
                    fine = true;
                    quarter -= 1;
                    &STEPS[..4]
                } else {
                    break;
                };
                for &(dx, dy) in steps {
                    let d = [centre[0] + dx, centre[1] + dy];
                    if !candidates.contains(&d) {
                        candidates.push(d);
                    }
                }
                if next == candidates.len() {
                    break;
                }
            }
            let d = candidates[next];
            next += 1;
            if d == delta {
                continue;
            }
            // Partial errors stop a candidate as soon as it cannot win.
            let rate = lambda * bits(d);
            if rate >= best.1 {
                continue;
            }
            let stop = |e: u64| e as f64 * scale + rate >= best.1;
            let target = std::mem::take(&mut sc.target);
            let error = if let [n] = nodes[..] {
                // A single node covers its area alone: no sum of contributions needed.
                let v = [pred[n][0] + d[0], pred[n][1] + d[1]];
                self.node_error(n, v, &area, &target, sc, stop)
            } else {
                // Bands of C rows.
                let mut error = 0;
                for y in (0..area.h).step_by(c as usize) {
                    let band = Area {
                        y0: area.y0 + y as i32,
                        h: (c as usize).min(area.h - y),
                        ..area
                    };
                    let cells = band.w * band.h;
                    trial.clear();
                    trial.resize(cells, 0);
                    for &n in &nodes {
                        let v = [pred[n][0] + d[0], pred[n][1] + d[1]];
                        self.contribute(n, v, false, &band, &mut trial, sc);
                    }
                    error += self.error(&target[y * area.w..][..cells], &trial, &band);
                    if stop(error) {
                        break;
                    }
                }
                error
            };
            sc.target = target;
            let cost = error as f64 * scale + lambda * bits(d);
            if cost < best.1 {
                best = (d, cost);
            }
        }
        let moved = best.0 != delta;
        if moved {
            // The winner's contribution on every row.
            kept.clear();
            kept.resize(cells, 0);
            for &n in &nodes {
                let v = [pred[n][0] + best.0[0], pred[n][1] + best.0[1]];
                self.contribute(n, v, false, &area, &mut kept, sc);
            }
            for y in 0..area.h {
                // SAFETY: as above.
                let sums = unsafe {
                    std::slice::from_raw_parts_mut(
                        shared.sum.add((y0 as usize + y) * w + area.x0),
                        area.w,
                    )
                };
                let r = y * area.w..(y + 1) * area.w;
                for ((s, &new), &o) in sums.iter_mut().zip(&kept[r.clone()]).zip(&old[r]) {
                    *s += new - o;
                }
            }
            for &n in &nodes {
                // SAFETY: as above.
                unsafe { *shared.mv.add(n) = [pred[n][0] + best.0[0], pred[n][1] + best.0[1]] };
            }
        }
        sc.nodes = nodes;
        sc.old = old;
        sc.trial = trial;
        sc.best = kept;
        sc.candidates = candidates;
        moved
    }
}

/// `((t[k] − weight[k]·s[k]) >> shift)²` at column `k` of a row ([`Row`]).
#[inline(always)]
fn error_at(t: &[i32], r: &Row, k: usize, shift: u32) -> u64 {
    let e = (t[k] - r.weight[k] * sample(r, k)) >> shift;
    u64::from((e * e) as u32)
}

/// The bilinear interpolation at column `k` of a row, in 1/64.
#[inline(always)]
fn sample(r: &Row, k: usize) -> i32 {
    let w = r.wts;
    w[0] * i32::from(r.a[k])
        + w[1] * i32::from(r.a[k + 1])
        + w[2] * i32::from(r.b[k])
        + w[3] * i32::from(r.b[k + 1])
}

/// The squared error of rows `w` wide.
#[inline(always)]
fn error_any(target: &[i32], trial: &[i32], w: usize, shift: u32) -> u64 {
    let mut total = 0u64;
    for (t, v) in target.chunks_exact(w).zip(trial.chunks_exact(w)) {
        for k in 0..w {
            let e = (t[k] - v[k]) >> shift;
            total += u64::from((e * e) as u32);
        }
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn error_avx2(target: &[i32], trial: &[i32], _w: usize, shift: u32) -> u64 {
    // Rows are contiguous: one run.
    // SAFETY: AVX2 is present.
    let (sum, done) = unsafe { x86::error_row(target, trial, shift) };
    let tail = target[done..]
        .iter()
        .zip(&trial[done..])
        .map(|(&t, &v)| {
            let e = (t - v) >> shift;
            u64::from((e * e) as u32)
        })
        .sum::<u64>();
    sum + tail
}

/// A node's support within an area, all inside the reference ([`Context::block`]).
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
struct Block<'a> {
    /// First row and column within the area.
    y: usize,
    x: usize,
    rows: usize,
    /// Columns, a multiple of 16.
    len: usize,
    weight: &'a [i32],
    weight_stride: usize,
    /// The displaced first row onwards, rows `stride` apart.
    reference: &'a [u8],
    stride: usize,
    wts: [i32; 4],
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn error_neon(target: &[i32], trial: &[i32], _w: usize, shift: u32) -> u64 {
    // Rows are contiguous: one run.
    // SAFETY: NEON is present.
    let (sum, done) = unsafe { neon::error_row(target, trial, shift) };
    let tail = target[done..]
        .iter()
        .zip(&trial[done..])
        .map(|(&t, &v)| {
            let e = (t - v) >> shift;
            u64::from((e * e) as u32)
        })
        .sum::<u64>();
    sum + tail
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{Block, Row};

    /// The bilinear samples `s[k]` (in 1/64, below 2^14) of columns `k..k + 16`, as 16-bit
    /// values.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn samples(r: &Row, wv: &[__m256i; 4], k: usize) -> __m256i {
        // SAFETY: the caller keeps 16 samples from `k + 1` inside the rows.
        unsafe {
            let ld = |s: &[u8], at: usize| {
                _mm256_cvtepu8_epi16(_mm_loadu_si128(s.as_ptr().add(at).cast()))
            };
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(wv[0], ld(r.a, k)),
                    _mm256_mullo_epi16(wv[1], ld(r.a, k + 1)),
                ),
                _mm256_add_epi16(
                    _mm256_mullo_epi16(wv[2], ld(r.b, k)),
                    _mm256_mullo_epi16(wv[3], ld(r.b, k + 1)),
                ),
            )
        }
    }

    /// `weight[k]·s[k]` for columns `k..k + 16`, as two vectors of eight. Both factors fit 16
    /// bits (weights are at most `C² ≤ 2^12`), so one multiply-add of each with a zero does it.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn weighted(r: &Row, s: __m256i, k: usize) -> (__m256i, __m256i) {
        // SAFETY: the caller keeps 16 weights from `k` inside the row.
        unsafe {
            let w = |at: usize| _mm256_loadu_si256(r.weight.as_ptr().add(at).cast());
            let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s));
            let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(s));
            (_mm256_madd_epi16(lo, w(k)), _mm256_madd_epi16(hi, w(k + 8)))
        }
    }

    /// [`super::contribute_row`] over whole groups of 16; returns how many were done.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn contribute_row(o: &mut [i32], r: &Row) -> usize {
        let len = o.len();
        let wv = r.wts.map(|v| _mm256_set1_epi16(v as i16));
        let mut k = 0;
        while k + 16 <= len {
            // SAFETY: the rows hold `len + 1` samples, `weight` and `o` `len` values.
            unsafe {
                let (lo, hi) = weighted(r, samples(r, &wv, k), k);
                let at = o.as_mut_ptr().add(k);
                _mm256_storeu_si256(
                    at.cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(at.cast()), lo),
                );
                let at = at.add(8);
                _mm256_storeu_si256(
                    at.cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(at.cast()), hi),
                );
            }
            k += 16;
        }
        k
    }

    /// [`super::error_row`] over whole groups of 16; returns the sum and how many were done.
    /// Errors fit 16 bits (at most 255·64 in magnitude).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn node_error_row(t: &[i32], r: &Row, shift: u32, sum: &mut Sum) -> usize {
        let len = t.len();
        let wv = r.wts.map(|v| _mm256_set1_epi16(v as i16));
        let count = _mm_cvtsi32_si128(shift as i32);
        let mut k = 0;
        while k + 16 <= len {
            // SAFETY: the rows hold `len + 1` samples, `weight` and `t` `len` values.
            unsafe {
                let (lo, hi) = weighted(r, samples(r, &wv, k), k);
                let e = |p: __m256i, at: usize| {
                    let t = _mm256_loadu_si256(t.as_ptr().add(at).cast());
                    _mm256_sra_epi32(_mm256_sub_epi32(t, p), count)
                };
                let e16 = _mm256_packs_epi32(e(lo, k), e(hi, k + 8));
                sum.add(_mm256_madd_epi16(e16, e16));
            }
            k += 16;
        }
        k
    }

    /// The bilinear samples (in 1/64, below 2^14) of 16 columns from `a` and the row `stride`
    /// below, as 16-bit values.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn samples_at(a: *const u8, stride: usize, wv: &[__m256i; 4]) -> __m256i {
        // SAFETY: the caller keeps 17 samples from `a` and from `a + stride` inside the plane.
        unsafe {
            let ld = |p: *const u8| _mm256_cvtepu8_epi16(_mm_loadu_si128(p.cast()));
            let b = a.add(stride);
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(wv[0], ld(a)),
                    _mm256_mullo_epi16(wv[1], ld(a.add(1))),
                ),
                _mm256_add_epi16(
                    _mm256_mullo_epi16(wv[2], ld(b)),
                    _mm256_mullo_epi16(wv[3], ld(b.add(1))),
                ),
            )
        }
    }

    /// `weight·s` for 16 columns, as two vectors of eight ([`weighted`]).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn weighted_at(w: *const i32, s: __m256i) -> (__m256i, __m256i) {
        // SAFETY: the caller keeps 16 weights from `w` inside the table.
        unsafe {
            let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s));
            let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(s));
            (
                _mm256_madd_epi16(lo, _mm256_loadu_si256(w.cast())),
                _mm256_madd_epi16(hi, _mm256_loadu_si256(w.add(8).cast())),
            )
        }
    }

    /// Adds the block's contribution to `out` (rows `w` wide).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn contribute(b: &Block, out: &mut [i32], w: usize) {
        assert!(
            (b.y + b.rows - 1) * w + b.x + b.len <= out.len()
                && (b.rows - 1) * b.weight_stride + b.len <= b.weight.len()
        );
        let wv = b.wts.map(|v| _mm256_set1_epi16(v as i16));
        for r in 0..b.rows {
            // SAFETY: (all below) checked above, and the block's reference rows hold `len + 1`
            // samples each, the last one's lower neighbours included.
            unsafe {
                let a = b.reference.as_ptr().add(r * b.stride);
                let wp = b.weight.as_ptr().add(r * b.weight_stride);
                let o = out.as_mut_ptr().add((b.y + r) * w + b.x);
                for k in (0..b.len).step_by(16) {
                    let (lo, hi) = weighted_at(wp.add(k), samples_at(a.add(k), b.stride, &wv));
                    let at = o.add(k).cast();
                    _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), lo));
                    let at = o.add(k + 8).cast();
                    _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), hi));
                }
            }
        }
    }

    /// The squared error of `target` (rows `len` wide, starting at the block) less the block's
    /// contribution, or a partial sum for which `stop` holds.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn node_error(
        b: &Block,
        target: &[i32],
        shift: u32,
        stop: impl Fn(u64) -> bool,
    ) -> u64 {
        assert!(
            b.rows * b.len <= target.len()
                && (b.rows - 1) * b.weight_stride + b.len <= b.weight.len()
        );
        let wv = b.wts.map(|v| _mm256_set1_epi16(v as i16));
        let count = _mm_cvtsi32_si128(shift as i32);
        let mut sum = Sum::default();
        for r in 0..b.rows {
            // SAFETY: as in `contribute`.
            unsafe {
                let a = b.reference.as_ptr().add(r * b.stride);
                let wp = b.weight.as_ptr().add(r * b.weight_stride);
                let t = target.as_ptr().add(r * b.len);
                for k in (0..b.len).step_by(16) {
                    let (lo, hi) = weighted_at(wp.add(k), samples_at(a.add(k), b.stride, &wv));
                    let e = |p: __m256i, at: usize| {
                        let t = _mm256_loadu_si256(t.add(at).cast());
                        _mm256_sra_epi32(_mm256_sub_epi32(t, p), count)
                    };
                    let e16 = _mm256_packs_epi32(e(lo, k), e(hi, k + 8));
                    sum.add(_mm256_madd_epi16(e16, e16));
                }
            }
            if r % super::CHECK == super::CHECK - 1 && stop(sum.total()) {
                break;
            }
        }
        sum.total()
    }

    /// Sums of squares: pairs (each below 2^29, from errors below 2^14 in magnitude) added
    /// in eight 32-bit lanes, at most [`Sum::PAIRS`] times before moving into four 64-bit ones.
    pub(super) struct Sum {
        wide: __m256i,
        narrow: __m256i,
        pending: u32,
    }

    impl Default for Sum {
        fn default() -> Self {
            // SAFETY: AVX is part of every CPU with AVX2, which every caller has.
            let zero = unsafe { _mm256_setzero_si256() };
            Self {
                wide: zero,
                narrow: zero,
                pending: 0,
            }
        }
    }

    impl Sum {
        /// 8·2^29 < 2^32.
        const PAIRS: u32 = 8;

        #[inline]
        #[target_feature(enable = "avx2")]
        fn add(&mut self, pairs: __m256i) {
            self.narrow = _mm256_add_epi32(self.narrow, pairs);
            self.pending += 1;
            if self.pending == Self::PAIRS {
                self.flush();
            }
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        fn flush(&mut self) {
            let n = self.narrow;
            let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(n));
            let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(n));
            self.wide = _mm256_add_epi64(self.wide, _mm256_add_epi64(lo, hi));
            self.narrow = _mm256_setzero_si256();
            self.pending = 0;
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn total(&mut self) -> u64 {
            self.flush();
            let mut lanes = [0u64; 4];
            // SAFETY: four 64-bit lanes.
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), self.wide) };
            lanes.iter().sum()
        }
    }

    /// `Σ ((t[k] − v[k]) >> shift)²` over whole groups of 16; returns the sum and how many
    /// were done. Errors fit 16 bits (at most 255·64 in magnitude).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn error_row(t: &[i32], v: &[i32], shift: u32) -> (u64, usize) {
        let len = t.len().min(v.len());
        let count = _mm_cvtsi32_si128(shift as i32);
        let mut sum = Sum::default();
        let mut k = 0;
        while k + 16 <= len {
            // SAFETY: both rows hold `len` values.
            unsafe {
                let e = |at: usize| {
                    let a = _mm256_loadu_si256(t.as_ptr().add(at).cast());
                    let b = _mm256_loadu_si256(v.as_ptr().add(at).cast());
                    _mm256_sra_epi32(_mm256_sub_epi32(a, b), count)
                };
                let e16 = _mm256_packs_epi32(e(k), e(k + 8));
                sum.add(_mm256_madd_epi16(e16, e16));
            }
            k += 16;
        }
        (sum.total(), k)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;

    use super::{Block, Row};

    /// The bilinear samples (in 1/64, below 2^14) of 16 columns from `a` and `b`, as two
    /// vectors of eight. The weights are at most 64, so byte multiplies widen exactly.
    #[inline(always)]
    unsafe fn samples(a: *const u8, b: *const u8, wv: &[uint8x16_t; 4]) -> [uint16x8_t; 2] {
        // SAFETY: the caller keeps 17 samples from `a` and from `b` inside the rows.
        unsafe {
            let (a0, a1) = (vld1q_u8(a), vld1q_u8(a.add(1)));
            let (b0, b1) = (vld1q_u8(b), vld1q_u8(b.add(1)));
            let lo = vmull_u8(vget_low_u8(a0), vget_low_u8(wv[0]));
            let lo = vmlal_u8(lo, vget_low_u8(a1), vget_low_u8(wv[1]));
            let lo = vmlal_u8(lo, vget_low_u8(b0), vget_low_u8(wv[2]));
            let lo = vmlal_u8(lo, vget_low_u8(b1), vget_low_u8(wv[3]));
            let hi = vmull_high_u8(a0, wv[0]);
            let hi = vmlal_high_u8(hi, a1, wv[1]);
            let hi = vmlal_high_u8(hi, b0, wv[2]);
            let hi = vmlal_high_u8(hi, b1, wv[3]);
            [lo, hi]
        }
    }

    /// The 16 weights from `w` (at most `C² ≤ 2^12`, never negative) as 16-bit values.
    #[inline(always)]
    unsafe fn weights(w: *const i32) -> [uint16x8_t; 2] {
        // SAFETY: the caller keeps 16 weights from `w` inside the table.
        unsafe {
            let n = |at: usize| vmovn_u32(vreinterpretq_u32_s32(vld1q_s32(w.add(at))));
            [vcombine_u16(n(0), n(4)), vcombine_u16(n(8), n(12))]
        }
    }

    /// Adds `weight·s` of 16 columns to `o`.
    #[inline(always)]
    unsafe fn add16(o: *mut i32, s: [uint16x8_t; 2], w: [uint16x8_t; 2]) {
        // SAFETY: the caller keeps 16 values from `o` inside the output.
        unsafe {
            for h in 0..2 {
                let (lo, hi) = (o.add(8 * h), o.add(8 * h + 4));
                let acc = vreinterpretq_u32_s32(vld1q_s32(lo));
                let acc = vmlal_u16(acc, vget_low_u16(s[h]), vget_low_u16(w[h]));
                vst1q_s32(lo, vreinterpretq_s32_u32(acc));
                let acc = vreinterpretq_u32_s32(vld1q_s32(hi));
                let acc = vmlal_high_u16(acc, s[h], w[h]);
                vst1q_s32(hi, vreinterpretq_s32_u32(acc));
            }
        }
    }

    /// `Σ ((t[k] − weight[k]·s[k]) >> shift)²` over 16 columns, in four lanes. Errors fit 16
    /// bits (at most 255·64 in magnitude), so each lane stays below 2^30.
    #[inline(always)]
    unsafe fn error16(
        t: *const i32,
        s: [uint16x8_t; 2],
        w: [uint16x8_t; 2],
        neg: int32x4_t,
    ) -> uint32x4_t {
        // SAFETY: the caller keeps 16 values from `t` inside the target.
        unsafe {
            let e = |at: usize, p: uint32x4_t| {
                let d = vsubq_s32(vld1q_s32(t.add(at)), vreinterpretq_s32_u32(p));
                vmovn_s32(vshlq_s32(d, neg))
            };
            let e0 = e(0, vmull_u16(vget_low_u16(s[0]), vget_low_u16(w[0])));
            let e1 = e(4, vmull_high_u16(s[0], w[0]));
            let e2 = e(8, vmull_u16(vget_low_u16(s[1]), vget_low_u16(w[1])));
            let e3 = e(12, vmull_high_u16(s[1], w[1]));
            let q = vmull_s16(e0, e0);
            let q = vmlal_s16(q, e1, e1);
            let q = vmlal_s16(q, e2, e2);
            vreinterpretq_u32_s32(vmlal_s16(q, e3, e3))
        }
    }

    fn splat(wts: [i32; 4]) -> [uint8x16_t; 4] {
        // SAFETY: NEON is present.
        wts.map(|v| unsafe { vdupq_n_u8(v as u8) })
    }

    /// [`super::contribute_row`] over whole groups of 16; returns how many were done.
    pub(super) unsafe fn contribute_row(o: &mut [i32], r: &Row) -> usize {
        let wv = splat(r.wts);
        let mut k = 0;
        while k + 16 <= o.len() {
            // SAFETY: the rows hold `len + 1` samples, `weight` and `o` `len` values.
            unsafe {
                let s = samples(r.a.as_ptr().add(k), r.b.as_ptr().add(k), &wv);
                add16(o.as_mut_ptr().add(k), s, weights(r.weight.as_ptr().add(k)));
            }
            k += 16;
        }
        k
    }

    /// The node error of one row over whole groups of 16; returns the sum and how many were
    /// done.
    pub(super) unsafe fn node_error_row(t: &[i32], r: &Row, shift: u32) -> (u64, usize) {
        let wv = splat(r.wts);
        // SAFETY: NEON is present.
        let (neg, mut wide) = unsafe { (vdupq_n_s32(-(shift as i32)), vdupq_n_u64(0)) };
        let mut k = 0;
        while k + 16 <= t.len() {
            // SAFETY: the rows hold `len + 1` samples, `weight` and `t` `len` values.
            unsafe {
                let s = samples(r.a.as_ptr().add(k), r.b.as_ptr().add(k), &wv);
                let w = weights(r.weight.as_ptr().add(k));
                wide = vpadalq_u32(wide, error16(t.as_ptr().add(k), s, w, neg));
            }
            k += 16;
        }
        // SAFETY: NEON is present.
        (unsafe { vaddvq_u64(wide) }, k)
    }

    /// Adds the block's contribution to `out` (rows `w` wide).
    pub(super) unsafe fn contribute(b: &Block, out: &mut [i32], w: usize) {
        assert!(
            (b.y + b.rows - 1) * w + b.x + b.len <= out.len()
                && (b.rows - 1) * b.weight_stride + b.len <= b.weight.len()
        );
        let wv = splat(b.wts);
        for r in 0..b.rows {
            // SAFETY: (all below) checked above, and the block's reference rows hold `len + 1`
            // samples each, the last one's lower neighbours included.
            unsafe {
                let a = b.reference.as_ptr().add(r * b.stride);
                let wp = b.weight.as_ptr().add(r * b.weight_stride);
                let o = out.as_mut_ptr().add((b.y + r) * w + b.x);
                for k in (0..b.len).step_by(16) {
                    let s = samples(a.add(k), a.add(k + b.stride), &wv);
                    add16(o.add(k), s, weights(wp.add(k)));
                }
            }
        }
    }

    /// The squared error of `target` (rows `len` wide, starting at the block) less the block's
    /// contribution, or a partial sum for which `stop` holds.
    pub(super) unsafe fn node_error(
        b: &Block,
        target: &[i32],
        shift: u32,
        stop: impl Fn(u64) -> bool,
    ) -> u64 {
        assert!(
            b.rows * b.len <= target.len()
                && (b.rows - 1) * b.weight_stride + b.len <= b.weight.len()
        );
        let wv = splat(b.wts);
        // SAFETY: NEON is present.
        let (neg, mut wide) = unsafe { (vdupq_n_s32(-(shift as i32)), vdupq_n_u64(0)) };
        for r in 0..b.rows {
            // SAFETY: as in `contribute`.
            unsafe {
                let a = b.reference.as_ptr().add(r * b.stride);
                let wp = b.weight.as_ptr().add(r * b.weight_stride);
                let t = target.as_ptr().add(r * b.len);
                for k in (0..b.len).step_by(16) {
                    let s = samples(a.add(k), a.add(k + b.stride), &wv);
                    let e = error16(t.add(k), s, weights(wp.add(k)), neg);
                    wide = vpadalq_u32(wide, e);
                }
            }
            // SAFETY: NEON is present.
            if r % super::CHECK == super::CHECK - 1 && stop(unsafe { vaddvq_u64(wide) }) {
                break;
            }
        }
        // SAFETY: NEON is present.
        unsafe { vaddvq_u64(wide) }
    }

    /// `Σ ((t[k] − v[k]) >> shift)²` over whole groups of 16; returns the sum and how many
    /// were done. Errors fit 16 bits (at most 255·64 in magnitude).
    pub(super) unsafe fn error_row(t: &[i32], v: &[i32], shift: u32) -> (u64, usize) {
        let len = t.len().min(v.len());
        // SAFETY: NEON is present.
        let (neg, mut wide) = unsafe { (vdupq_n_s32(-(shift as i32)), vdupq_n_u64(0)) };
        let mut k = 0;
        while k + 16 <= len {
            // SAFETY: both rows hold `len` values.
            unsafe {
                let e = |at: usize| {
                    let d = vsubq_s32(vld1q_s32(t.as_ptr().add(at)), vld1q_s32(v.as_ptr().add(at)));
                    vmovn_s32(vshlq_s32(d, neg))
                };
                let (e0, e1, e2, e3) = (e(k), e(k + 4), e(k + 8), e(k + 12));
                let q = vmull_s16(e0, e0);
                let q = vmlal_s16(q, e1, e1);
                let q = vmlal_s16(q, e2, e2);
                let q = vmlal_s16(q, e3, e3);
                wide = vpadalq_u32(wide, vreinterpretq_u32_s32(q));
            }
            k += 16;
        }
        // SAFETY: NEON is present.
        (unsafe { vaddvq_u64(wide) }, k)
    }
}

/// Whether the CPU has AVX2 (checked once).
#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}
