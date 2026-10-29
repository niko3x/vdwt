//! The motion quadtree's encoder side: which leaves, which offsets, which intra nodes
//! (DESIGN.md §6.2, §6.4).
//!
//! Every node stands for the `C × C` pixels around it (`C` the grid spacing). A leaf's distortion
//! is the squared error of those pixels displaced by each node's vector (global prediction plus
//! the leaf's offset); its rate is estimated from the mode and the difference it codes
//! ([`tree::leaf_bits`]). Roots are decided in raster order and quadrants in Z order, as they are
//! coded, so the neighbours a leaf may copy or predict from are known when it is decided.
//!
//! For each region the search starts from candidate offsets (none, the left, upper and
//! upper-right neighbours', the enclosing region's best, and the node's offset in the previous
//! frame), widens the best with an integer diamond search and refines it with Lucas–Kanade, on
//! one window over the whole region. The exact per-node cost then picks the leaf's offset among
//! the finalists. A region splits when its four quadrants, decided the same way, cost less;
//! regions whose whole error could not pay for a split are not split at all.
//!
//! Root rows run as a wavefront: a root waits for the row above to finish the root above and to
//! its right. The result is that of a raster scan, whatever the thread count.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use super::MotionField;
use super::mesh::{ITERATIONS, MeshParams, MeshStats, Window, lucas_kanade};
use super::tree::{self, root_log2};
use crate::Plane;
use crate::pool::{Pool, Progress};

/// Furthest an offset may reach from the global prediction, in pixels: an eighth of the picture
/// width, and at least this. Low over the ground, the foreground moves up to a fifth of the
/// picture per frame beyond the global model; 48 pixels at any size lost 3.6% of the bits at
/// 720p and 5% at 1080p (§10.1).
const MIN_DELTA_PX: i32 = 48;
/// Integer diamond search steps, in pixels, each repeated while it improves.
const DIAMOND: [i32; 7] = [64, 32, 16, 8, 4, 2, 1];
const DIAMOND_MOVES: usize = 4;
/// Squared errors are in 1/64 of a squared sample value.
const SSD_UNIT: f64 = 64.0;
/// The least a split adds in bits: its flag and four leaf modes, less the leaf's own mode.
const SPLIT_MIN_BITS: f64 = 4.0;
/// Regions whose best candidate costs at most this many bits' worth are not searched: a new
/// vector costs at least 6.5 bits, and above that searching rarely finds one that pays (16
/// halves the searches at no cost in bits, §10.1).
const SEARCH_MIN_BITS: f64 = 16.0;
/// [`SEARCH_MIN_BITS`] in realtime mode.
const REALTIME_SEARCH_MIN_BITS: f64 = 32.0;
/// [`REALTIME_SEARCH_MIN_BITS`] at high rate (λ < 10, qp ≲ 22): there the residual can absorb
/// a coarser prediction, and skipping refinements below 128 bits of RD cost with one
/// Lucas–Kanade step costs 0.01 dB and 0.3% of the bits at qp 18. At lower rates the finer
/// threshold stays; forcing it everywhere cost 3–5 BD points on the charts for 7% of the time.
const REALTIME_HIGH_RATE_MIN_BITS: f64 = 128.0;
const REALTIME_HIGH_RATE_LAMBDA: f64 = 10.0;
/// Lucas–Kanade steps in realtime mode: one reaches the sub-pixel minimum on this search's
/// windows; a second costs 10% of the encode time and changes almost no decision.
const REALTIME_LK_ITERATIONS: usize = 1;

/// Decides `field`'s partition, offsets and intra nodes for `cur` against `reference`. `field`
/// holds the global prediction on entry. `previous` is the last frame's field.
pub(crate) fn search(
    pool: &Pool,
    cur: &Plane<u8>,
    reference: &Plane<u8>,
    field: &mut MotionField,
    previous: Option<&MotionField>,
    params: &MeshParams,
) -> MeshStats {
    let grid = field.grid;
    let r = root_log2(grid.log2);
    let root = 1usize << r;
    field.depth.fill(r as u8);
    if !params.refine {
        return MeshStats::default();
    }
    let previous = previous.filter(|p| p.grid == grid);
    let pred = field.mv.clone();
    let (rx, ry) = (grid.nx.div_ceil(root), grid.ny.div_ceil(root));
    let shared = Shared {
        delta: (0..grid.nodes()).map(|_| AtomicU64::new(0)).collect(),
        meta: (0..grid.nodes()).map(|_| AtomicU8::new(0)).collect(),
        progress: Progress::new(ry),
    };
    let windows = node_windows(pool, cur, field, if params.realtime { 2 } else { 1 });
    // Stripes of root rows search independently: a stripe's first row does not wait for (or
    // see) the row above, so the stripes' wavefronts run side by side. Rows are handed out
    // across the stripes in turn, so every stripe starts at once.
    let stripe = if params.stripe_rows == 0 {
        ry
    } else {
        params.stripe_rows.min(ry)
    };
    let stripe_start = |rj: usize| rj.is_multiple_of(stripe);
    let order: Vec<usize> = (0..stripe)
        .flat_map(|k| (0..ry.div_ceil(stripe)).map(move |s| s * stripe + k))
        .filter(|&rj| rj < ry)
        .collect();
    pool.run(ry, &|t| {
        let rj = order[t];
        let bufs = TASK_BUFS.with(|b| std::mem::take(&mut *b.borrow_mut()));
        let mut task = Task {
            cur,
            reference,
            params,
            pred: &pred,
            previous,
            windows: &windows,
            nx: grid.nx,
            ny: grid.ny,
            spacing: grid.spacing() as i32,
            max_delta: 8 * (cur.width() as i32 / 8).max(MIN_DELTA_PX),
            delta: refill(bufs.delta, grid.nodes(), [0; 2]),
            known: refill(bufs.known, grid.nodes(), false),
            intra: refill(bufs.intra, grid.nodes(), false),
            depth: refill(bufs.depth, grid.nodes(), r as u8),
            cache: Cache::reuse(bufs.cache, rj * root * grid.nx, root * grid.nx),
            win: bufs.win,
            tried: bufs.tried,
        };
        for ri in 0..rx {
            let (i0, j0) = (ri * root, rj * root);
            if !stripe_start(rj) {
                // The root above and to the right must be final (upper-right predictions).
                shared.progress.wait(rj - 1, (ri + 2).min(rx));
                // The row of nodes just above this root, and the corners beside it.
                let j = j0 - 1;
                for i in i0.saturating_sub(1)..(i0 + 2 * root).min(grid.nx) {
                    let n = j * grid.nx + i;
                    task.delta[n] = unpack(shared.delta[n].load(Ordering::Relaxed));
                    let m = shared.meta[n].load(Ordering::Relaxed);
                    task.known[n] = true;
                    task.intra[n] = m & 1 == 1;
                    task.depth[n] = m >> 1;
                }
            }
            task.region(i0, j0, r, None);
            for j in j0..(j0 + root).min(grid.ny) {
                for i in i0..(i0 + root).min(grid.nx) {
                    let n = j * grid.nx + i;
                    shared.delta[n].store(pack(task.delta[n]), Ordering::Relaxed);
                    let m = u8::from(task.intra[n]) | task.depth[n] << 1;
                    shared.meta[n].store(m, Ordering::Relaxed);
                }
            }
            shared.progress.set(rj, ri + 1);
        }
        TASK_BUFS.with(|b| {
            *b.borrow_mut() = TaskBufs {
                delta: task.delta,
                known: task.known,
                intra: task.intra,
                depth: task.depth,
                cache: task.cache,
                win: task.win,
                tried: task.tried,
            }
        });
    });
    WINDOWS.with(|w| *w.borrow_mut() = windows);
    let mut stats = MeshStats::default();
    for (n, p) in pred.iter().enumerate() {
        let d = unpack(shared.delta[n].load(Ordering::Relaxed));
        let m = shared.meta[n].load(Ordering::Relaxed);
        field.intra[n] = m & 1 == 1;
        field.depth[n] = m >> 1;
        field.mv[n] = [p[0] + d[0], p[1] + d[1]];
        stats.refined += usize::from(d != [0; 2] && !field.intra[n]);
        stats.intra += usize::from(field.intra[n]);
    }
    stats
}

struct Shared {
    delta: Vec<AtomicU64>,
    /// Intra flag (bit 0) and leaf depth (above).
    meta: Vec<AtomicU8>,
    /// Roots finished per root row.
    progress: Progress,
}

fn pack(d: [i32; 2]) -> u64 {
    u64::from(d[0] as u32) | u64::from(d[1] as u32) << 32
}

fn unpack(v: u64) -> [i32; 2] {
    [v as u32 as i32, (v >> 32) as u32 as i32]
}

/// Every node's own pixels: `C × C` around it, inside the frame.
/// Every node's window: its pixels, every `row_step`-th row.
/// The windows are refilled in the last search's (see [`TaskBufs`]).
fn node_windows(pool: &Pool, cur: &Plane<u8>, field: &MotionField, row_step: usize) -> Vec<Window> {
    let grid = field.grid;
    let s = grid.spacing() as i32;
    let (w, h) = (cur.width() as i32, cur.height() as i32);
    let mut all = WINDOWS.with(|w| std::mem::take(&mut *w.borrow_mut()));
    all.truncate(grid.nodes());
    all.resize_with(grid.nodes(), Window::default);
    let mut rows: Vec<&mut [Window]> = all.chunks_mut(grid.nx).collect();
    pool.for_each_mut(&mut rows, |j, row| {
        let j = j as i32;
        for (i, win) in row.iter_mut().enumerate() {
            let (i, r) = (i as i32, s / 2);
            let xs = ((i * s - r).max(0), (i * s + r).min(w));
            let ys = ((j * s - r).max(0), (j * s + r).min(h));
            if xs.0 < xs.1 && ys.0 < ys.1 {
                win.fill(cur, xs, ys, row_step);
            } else {
                *win = Window::default();
            }
        }
    });
    all
}

/// A task's buffers, kept by its thread from search to search (made per task and frame, they
/// faulted pages in every frame), and the node windows likewise.
#[derive(Default)]
struct TaskBufs {
    delta: Vec<[i32; 2]>,
    known: Vec<bool>,
    intra: Vec<bool>,
    depth: Vec<u8>,
    cache: Cache,
    win: Window,
    tried: Vec<[i32; 2]>,
}

thread_local! {
    static TASK_BUFS: std::cell::RefCell<TaskBufs> = std::cell::RefCell::new(TaskBufs::default());
    static WINDOWS: std::cell::RefCell<Vec<Window>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `v` as `n` copies of `x`, in its buffer.
fn refill<T: Clone>(mut v: Vec<T>, n: usize, x: T) -> Vec<T> {
    v.clear();
    v.resize(n, x);
    v
}

/// Vectors remembered per node.
const CACHED: usize = 24;

/// Exact squared errors already computed for the nodes of one root row, by vector: the last
/// [`CACHED`] per node.
#[derive(Default)]
struct Cache {
    first: usize,
    entries: Vec<[([i32; 2], u64); CACHED]>,
    len: Vec<u8>,
    next: Vec<u8>,
}

impl Cache {
    /// For the `nodes` nodes from node `first`, in `old`'s buffers.
    fn reuse(old: Self, first: usize, nodes: usize) -> Self {
        Self {
            first,
            entries: refill(old.entries, nodes, [([0; 2], 0); CACHED]),
            len: refill(old.len, nodes, 0),
            next: refill(old.next, nodes, 0),
        }
    }

    /// Node `n`'s squared error at `mv`, remembered or computed by `f` (replacing the oldest).
    fn get(&mut self, n: usize, mv: [i32; 2], f: impl FnOnce() -> u64) -> u64 {
        let k = n - self.first;
        let entries = &mut self.entries[k];
        if let Some(&(_, e)) = entries[..usize::from(self.len[k])]
            .iter()
            .find(|(v, _)| *v == mv)
        {
            return e;
        }
        let e = f();
        if usize::from(self.len[k]) < CACHED {
            entries[usize::from(self.len[k])] = (mv, e);
            self.len[k] += 1;
        } else {
            entries[usize::from(self.next[k])] = (mv, e);
            self.next[k] = ((usize::from(self.next[k]) + 1) % CACHED) as u8;
        }
        e
    }
}

/// One root row's search state. Node arrays cover the whole grid; only this row's roots and
/// the nodes just above them are ever read.
struct Task<'a> {
    cur: &'a Plane<u8>,
    reference: &'a Plane<u8>,
    params: &'a MeshParams,
    /// Global prediction per node.
    pred: &'a [[i32; 2]],
    previous: Option<&'a MotionField>,
    windows: &'a [Window],
    nx: usize,
    ny: usize,
    spacing: i32,
    /// Largest offset component, in 1/8 pel.
    max_delta: i32,
    delta: Vec<[i32; 2]>,
    known: Vec<bool>,
    intra: Vec<bool>,
    depth: Vec<u8>,
    /// Exact squared errors already computed per node, by vector.
    cache: Cache,
    win: Window,
    tried: Vec<[i32; 2]>,
}

/// A leaf's decision.
#[derive(Clone, Copy)]
struct Leaf {
    delta: [i32; 2],
    intra: bool,
    /// Rate-distortion cost: squared error plus λ·bits.
    cost: f64,
    /// Squared error alone.
    error: f64,
}

impl Task<'_> {
    /// The nodes of region `(i0, j0)` of `size` nodes.
    fn nodes(&self, i0: usize, j0: usize, size: usize) -> impl Iterator<Item = usize> + use<> {
        let (nx, ny) = (self.nx, self.ny);
        (j0..(j0 + size).min(ny))
            .flat_map(move |j| (i0..(i0 + size).min(nx)).map(move |i| j * nx + i))
    }

    fn known_delta(&self, i: Option<usize>, j: Option<usize>) -> Option<[i32; 2]> {
        let (i, j) = (i?, j?);
        (i < self.nx && j < self.ny && self.known[j * self.nx + i])
            .then(|| self.delta[j * self.nx + i])
    }

    /// Exact squared error (in 1/64) of node `n` at offset `delta`.
    fn node_ssd(&mut self, n: usize, delta: [i32; 2]) -> u64 {
        let mv = [self.pred[n][0] + delta[0], self.pred[n][1] + delta[1]];
        let win = &self.windows[n];
        let reference = self.reference;
        self.cache.get(n, mv, || {
            if win.len() == 0 {
                0
            } else {
                win.ssd(reference, mv) * win.weight() as u64
            }
        })
    }

    fn region_ssd(&mut self, i0: usize, j0: usize, size: usize, delta: [i32; 2]) -> u64 {
        self.nodes(i0, j0, size)
            .map(|n| self.node_ssd(n, delta))
            .sum()
    }

    /// Decides region `(i0, j0)` of `2^level` nodes, writes the decision into the node arrays and
    /// returns its cost. `hint` is the enclosing region's best offset.
    fn region(&mut self, i0: usize, j0: usize, level: u32, hint: Option<[i32; 2]>) -> f64 {
        if i0 >= self.nx || j0 >= self.ny {
            return 0.0;
        }
        let size = 1usize << level;
        let lambda = self.params.lambda;
        let leaf = self.leaf(i0, j0, size, hint);
        let split = level > 0 && leaf.error > lambda * SPLIT_MIN_BITS;
        let mut best = leaf.cost + if level > 0 { lambda } else { 0.0 };
        let mut children = None;
        if split {
            let h = size / 2;
            let mut cost = lambda;
            for (di, dj) in [(0, 0), (h, 0), (0, h), (h, h)] {
                cost += self.region(i0 + di, j0 + dj, level - 1, Some(leaf.delta));
                if cost >= best {
                    break;
                }
            }
            if cost < best {
                best = cost;
                children = Some(());
            }
        }
        if children.is_none() {
            for n in self.nodes(i0, j0, size) {
                self.delta[n] = if leaf.intra { [0; 2] } else { leaf.delta };
                self.intra[n] = leaf.intra;
                self.depth[n] = level as u8;
                self.known[n] = true;
            }
        }
        best
    }

    /// The best single offset for region `(i0, j0)` of `size` nodes.
    fn leaf(&mut self, i0: usize, j0: usize, size: usize, hint: Option<[i32; 2]>) -> Leaf {
        let lambda = self.params.lambda;
        let left = self.known_delta(i0.checked_sub(1), Some(j0));
        let up = self.known_delta(Some(i0), j0.checked_sub(1));
        let new_pred = tree::predict_new(&self.delta, &self.known, self.nx, self.ny, i0, j0, size);
        let bits = |d: [i32; 2]| tree::leaf_bits(d, left, up, new_pred);
        // The node at the region's centre stands for its global motion.
        let (ci, cj) = (
            (i0 + size / 2).min(self.nx - 1),
            (j0 + size / 2).min(self.ny - 1),
        );
        let centre = cj * self.nx + ci;
        let base = self.pred[centre];
        let mut finalists: Vec<[i32; 2]> = vec![[0, 0]];
        for d in [left, up, hint, Some(new_pred)].into_iter().flatten() {
            finalists.push(d);
        }
        if let Some(p) = self.previous {
            finalists.push([p.mv[centre][0] - base[0], p.mv[centre][1] - base[1]]);
        }
        // Offsets are coded in quarter pels.
        let round = |d: [i32; 2]| d.map(|v| (v + 1).div_euclid(2) * 2);
        let error0 = self.region_ssd(i0, j0, size, [0, 0]) as f64 / SSD_UNIT;
        let mut best = Leaf {
            delta: [0; 2],
            intra: false,
            cost: error0 + lambda * bits([0; 2]),
            error: error0,
        };
        let consider = |task: &mut Self, d: [i32; 2], best: &mut Leaf| {
            if d[0].abs() > task.max_delta || d[1].abs() > task.max_delta {
                return;
            }
            let error = task.region_ssd(i0, j0, size, d) as f64 / SSD_UNIT;
            let cost = error + lambda * bits(d);
            if cost < best.cost {
                *best = Leaf {
                    delta: d,
                    intra: false,
                    cost,
                    error,
                };
            }
        };
        let mut tried: Vec<[i32; 2]> = finalists.iter().map(|&d| round(d)).collect();
        tried.sort_unstable();
        tried.dedup();
        for &d in &tried {
            consider(self, d, &mut best);
        }
        // A vector the search finds is new and costs at least 6.5 bits (`tree::leaf_bits`):
        // searching cannot pay where a candidate already costs little more than that.
        let min_bits = if self.params.realtime {
            if lambda < REALTIME_HIGH_RATE_LAMBDA {
                REALTIME_HIGH_RATE_MIN_BITS
            } else {
                REALTIME_SEARCH_MIN_BITS
            }
        } else {
            SEARCH_MIN_BITS
        };
        if best.cost > lambda * min_bits {
            let wide = hint.is_none() || self.params.wide;
            let found = self.window_search(i0, j0, size, base, &finalists, &bits, wide);
            if let Some(d) = found {
                for d in [d, [d[0], 0], [0, d[1]]].map(round) {
                    if !tried.contains(&d) {
                        tried.push(d);
                        consider(self, d, &mut best);
                    }
                }
            }
        }
        // Intra when the prediction error clearly exceeds the region's own activity and plain
        // mid-grey would do better (DESIGN.md §6.4).
        let (mut len, mut sum, mut sum_sq) = (0f64, 0f64, 0f64);
        for n in self.nodes(i0, j0, size) {
            let w = &self.windows[n];
            let k = w.weight() as f64;
            len += k * w.len() as f64;
            sum += k * w.sum as f64;
            sum_sq += k * w.sum_sq as f64;
        }
        if len > 0.0 {
            let activity = sum_sq - sum * sum / len;
            let grey = sum_sq - 256.0 * sum + 16384.0 * len;
            if best.error > (1.5 * activity + len * 16.0).max(grey) {
                best.intra = true;
            }
        }
        best
    }

    /// Diamond search and Lucas–Kanade over one window covering the region, every node moving
    /// by `base + δ`. Returns the best offset found.
    #[allow(clippy::too_many_arguments)]
    fn window_search(
        &mut self,
        i0: usize,
        j0: usize,
        size: usize,
        base: [i32; 2],
        starts: &[[i32; 2]],
        bits: &dyn Fn([i32; 2]) -> f64,
        wide: bool,
    ) -> Option<[i32; 2]> {
        let (cur, reference, lambda) = (self.cur, self.reference, self.params.lambda);
        let max_delta = self.max_delta;
        let s = self.spacing;
        let (w, h) = (cur.width() as i32, cur.height() as i32);
        let span = size as i32 * s;
        // About 256 to 512 samples: every row up to 16 pixels, then fewer rows.
        let step = (span as usize / 16).max(1);
        let x0 = i0 as i32 * s - s / 2;
        let y0 = j0 as i32 * s - s / 2;
        let xs = (x0.max(1), (x0 + span).min(w - 1));
        let ys = (y0.max(1), (y0 + span).min(h - 1));
        if xs.0 >= xs.1 || ys.0 >= ys.1 {
            return None;
        }
        let win = &mut self.win;
        win.fill(cur, xs, ys, step);
        if win.len() < 16 {
            return None;
        }
        win.fill_gradients(cur);
        let win = &self.win;
        let scale = win.weight() as f64 / SSD_UNIT;
        let tried = &mut self.tried;
        tried.clear();
        let mut best = ([0i32; 2], f64::INFINITY);
        let mut consider = |d: [i32; 2], best: &mut ([i32; 2], f64)| {
            if d[0].abs() > max_delta || d[1].abs() > max_delta || tried.contains(&d) {
                return;
            }
            tried.push(d);
            let mv = [base[0] + d[0], base[1] + d[1]];
            let c = win.ssd(reference, mv) as f64 * scale + lambda * bits(d);
            if c < best.1 {
                *best = (d, c);
            }
        };
        for &d in starts {
            consider(d, &mut best);
        }
        // Whole-pixel positions of the absolute vector.
        let snap = |d: [i32; 2]| -> [i32; 2] {
            std::array::from_fn(|c| (base[c] + d[c] + 4).div_euclid(8) * 8 - base[c])
        };
        let mut centre = snap(best.0);
        consider(centre, &mut best);
        // A quadrant starts from its region's best: smaller steps suffice, unless searching wide.
        let steps: &[i32] = if wide {
            &DIAMOND
        } else {
            &DIAMOND[DIAMOND.len() - 2..]
        };
        for d in steps.iter().map(|d| 8 * d) {
            for _ in 0..DIAMOND_MOVES {
                let before = best.0;
                for (dx, dy) in [(d, 0), (-d, 0), (0, d), (0, -d)] {
                    consider([centre[0] + dx, centre[1] + dy], &mut best);
                }
                if best.0 == before || snap(best.0) != best.0 {
                    break;
                }
                centre = best.0;
            }
        }
        let start = [0, 1].map(|c| f64::from(base[c] + best.0[c]) / 8.0);
        let iterations = if self.params.realtime {
            if lambda < REALTIME_HIGH_RATE_LAMBDA {
                REALTIME_LK_ITERATIONS
            } else {
                2
            }
        } else {
            ITERATIONS
        };
        if let Some(mv) = lucas_kanade(win, reference, start, iterations) {
            let d = [0, 1].map(|c| (mv[c] * 8.0).round() as i32 - base[c]);
            consider(d, &mut best);
        }
        Some(best.0)
    }
}
