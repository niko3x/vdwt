//! The rt encoder's motion search: global motion and node vectors, every window on its own, all
//! in parallel (DESIGN.md §10.3).
//!
//! The main encoder estimates global motion from feature patches first, then walks the nodes in
//! a wavefront (each node starts from its decided neighbours, scored by the exact overlapped
//! prediction error). Here nothing waits for anything decided this frame:
//!
//! 1. Every root (64 pixels) searches one vector over its window: from the last frame's global
//!    model and vectors, a diamond at half resolution, then a cross to a half pel at full
//!    resolution, by squared error on every fourth row.
//! 2. An affine model fitted robustly to the textured roots' vectors is the frame's global
//!    motion (and the next frame's start).
//! 3. Every node picks among its own and its neighbouring roots' offsets from that model and its
//!    last vector, refined to a quarter pel. Roots predicted very well or very badly skip this.
//!
//! Leaves are single nodes, merged into whole roots where their nodes kept the same offset.

use crate::Plane;
use crate::motion::gme::Affine;
use crate::motion::tree::root_log2;
use crate::motion::{MAX_MV, MotionField};
use crate::pool::Pool;

/// Diamond steps at half resolution, at most.
const DIAMOND_STEPS: usize = 8;
/// Rows of a node's (and a root's) window sampled at full resolution: every this many.
const ROW_STEP: usize = 4;
const ROOT_ROW_STEP: usize = 4;
/// A window whose best candidate costs under this many bits' worth plus this squared error per
/// pixel is not searched.
const SKIP_BITS: f64 = 32.0;
const SKIP_SSE: f64 = 4.0;
/// A root whose best vector's cost is under this squared error per pixel leaves its nodes
/// unsearched.
const ROOT_GOOD_SSE: f64 = 16.0;
/// Nor one above this: where the error is mostly texture and noise, finer vectors gain little.
const ROOT_BAD_SSE: f64 = 200.0;
/// Roots flatter than this (mean absolute gradient at half resolution) are left out of the
/// global fit.
const MIN_TEXTURE: f64 = 8.0;
/// The roots' rate term relative to the nodes'.
const ROOT_LAMBDA_SCALE: f64 = 0.1;
/// Diamond steps of the root search at half resolution, in 1/8 pel.
const ROOT_DIAMOND: [i32; 3] = [64, 32, 16];

/// A node's cell in luma pixels: `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy)]
struct Cell {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

impl Cell {
    fn halved(self) -> Self {
        Self {
            x0: self.x0 / 2,
            x1: self.x1.div_ceil(2),
            y0: self.y0 / 2,
            y1: self.y1.div_ceil(2),
        }
    }
}

#[inline]
fn sq(a: u8, b: u8) -> u32 {
    let d = u32::from(a.abs_diff(b));
    d * d
}

/// Squared error of `cell` in `cur` against `reference` displaced by whole pixels `(dx, dy)`, on
/// every `step`th row; the reference is clamped at its edges.
fn sse_int(cur: &Plane<u8>, reference: &Plane<u8>, c: Cell, dx: i32, dy: i32, step: usize) -> u32 {
    let (w, h) = (reference.width() as i32, reference.height() as i32);
    let (rx0, rx1) = (c.x0 as i32 + dx, c.x1 as i32 + dx);
    let inside_x = rx0 >= 0 && rx1 <= w;
    let mut sum = 0u32;
    for y in (c.y0..c.y1).step_by(step) {
        let ry = (y as i32 + dy).clamp(0, h - 1) as usize;
        let a = cur.row_span(y, c.x0, c.x1 - c.x0);
        if inside_x {
            sum += sse_row(a, reference.row_span(ry, rx0 as usize, c.x1 - c.x0));
        } else {
            let b = reference.row(ry);
            for (x, &a) in (rx0..rx1).zip(a) {
                sum += sq(a, b[x.clamp(0, w - 1) as usize]);
            }
        }
    }
    sum
}

/// [`sse_int`] at a vector `v` in 1/8 pel, sampled bilinearly. Cells are at most 128 wide.
fn sse_frac(cur: &Plane<u8>, reference: &Plane<u8>, c: Cell, v: [i32; 2], step: usize) -> u32 {
    let (fx, fy) = ((v[0] & 7) as u16, (v[1] & 7) as u16);
    let (dx, dy) = (v[0] >> 3, v[1] >> 3);
    if fx == 0 && fy == 0 {
        return sse_int(cur, reference, c, dx, dy, step);
    }
    let (w, h) = (reference.width() as i32, reference.height() as i32);
    let wts = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
    let n = c.x1 - c.x0;
    let rx0 = c.x0 as i32 + dx;
    let inside = rx0 >= 0 && rx0 + (n as i32) < w;
    let mut sum = 0u32;
    let (mut et, mut eb) = ([0u8; 129], [0u8; 129]);
    for y in (c.y0..c.y1).step_by(step) {
        let ry0 = (y as i32 + dy).clamp(0, h - 1) as usize;
        let ry1 = (y as i32 + dy + 1).clamp(0, h - 1) as usize;
        let (t, b): (&[u8], &[u8]) = if inside {
            (
                reference.row_span(ry0, rx0 as usize, n + 1),
                reference.row_span(ry1, rx0 as usize, n + 1),
            )
        } else {
            let (rt, rb) = (reference.row(ry0), reference.row(ry1));
            for k in 0..=n {
                let x = (rx0 + k as i32).clamp(0, w - 1) as usize;
                (et[k], eb[k]) = (rt[x], rb[x]);
            }
            (&et[..=n], &eb[..=n])
        };
        sum += sse_row_bilinear(cur.row_span(y, c.x0, n), t, b, wts);
    }
    sum
}

/// Σ(a − b)² over a row.
#[inline]
fn sse_row(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(&a, &b)| sq(a, b)).sum()
}

/// Σ(a − p)² over a row, `p` the bilinear interpolation (weights `wts` summing to 64) of the
/// rows `t` and `b` (one longer than `a`). The compiler vectorizes both loops well (better than
/// hand-written NEON on the A55).
#[inline]
fn sse_row_bilinear(a: &[u8], t: &[u8], b: &[u8], wts: [u16; 4]) -> u32 {
    a.iter()
        .zip(t.iter().zip(&t[1..]))
        .zip(b.iter().zip(&b[1..]))
        .map(|((&a, (&t0, &t1)), (&b0, &b1))| {
            let p = wts[0] * u16::from(t0)
                + wts[1] * u16::from(t1)
                + wts[2] * u16::from(b0)
                + wts[3] * u16::from(b1);
            sq(a, ((p + 32) >> 6) as u8)
        })
        .sum()
}

/// Approximate bits of an offset from the prediction.
fn offset_bits(d: [i32; 2]) -> f64 {
    d.iter()
        .map(|&d| {
            if d == 0 {
                1.0
            } else {
                3.0 + 2.0 * f64::from(d.unsigned_abs()).log2()
            }
        })
        .sum()
}

/// The search of one window: from the best of `cands` (vectors), a diamond at half resolution,
/// then a cross at full resolution down to a quarter pel. `bits` prices a vector.
struct Searcher<'a> {
    /// Full-resolution rows sampled: every this many.
    row_step: usize,
    /// Windows whose best candidate costs under `skip` are not searched.
    skip_lambda: f64,
    cur: &'a Plane<u8>,
    cur_half: &'a Plane<u8>,
    reference: &'a Plane<u8>,
    ref_half: &'a Plane<u8>,
    lambda: f64,
}

impl Searcher<'_> {
    fn full(&self, cell: Cell, v: [i32; 2]) -> f64 {
        let step = self.row_step;
        f64::from(step as u32 * sse_frac(self.cur, self.reference, cell, v, step))
    }

    /// The best vector and its cost.
    fn search(
        &self,
        cell: Cell,
        base: [i32; 2],
        cands: &[[i32; 2]],
        bits: impl Fn([i32; 2]) -> f64,
        diamond: &[i32],
        cross: &[i32],
    ) -> ([i32; 2], f64) {
        let cost = |v: [i32; 2]| self.full(cell, v) + self.lambda * bits(v);
        // Offsets from `base` are whole quarter pels.
        let fit = |v: [i32; 2]| -> [i32; 2] {
            std::array::from_fn(|c| base[c] + ((v[c] - base[c]) & !1))
        };
        let (mut best, mut best_cost) = (fit(cands[0]), f64::INFINITY);
        for v in cands.iter().map(|&v| fit(v)) {
            let c = cost(v);
            if c < best_cost {
                (best, best_cost) = (v, c);
            }
        }
        let area = ((cell.x1 - cell.x0) * (cell.y1 - cell.y0)) as f64;
        if best_cost < self.skip_lambda * SKIP_BITS + SKIP_SSE * area {
            return (best, best_cost);
        }
        if !diamond.is_empty() {
            // Whole half pixels are 16 eighths; a half-resolution error counts a quarter of
            // the pixels.
            let half = cell.halved();
            let half_cost = |v: [i32; 2]| {
                let e = sse_int(self.cur_half, self.ref_half, half, v[0] >> 4, v[1] >> 4, 1);
                4.0 * f64::from(e) + self.lambda * bits(v)
            };
            let start = best.map(|c| (c + 8) & !15);
            let (mut hbest, mut hcost) = (start, half_cost(start));
            for &step in diamond {
                for _ in 0..DIAMOND_STEPS {
                    let centre = hbest;
                    for d in [[step, 0], [-step, 0], [0, step], [0, -step]] {
                        let v = [centre[0] + d[0], centre[1] + d[1]];
                        let c = half_cost(v);
                        if c < hcost {
                            (hbest, hcost) = (v, c);
                        }
                    }
                    if hbest == centre {
                        break;
                    }
                }
            }
            if hbest != start {
                let hbest = fit(hbest);
                let c = cost(hbest);
                if c < best_cost {
                    (best, best_cost) = (hbest, c);
                }
            }
        }
        for &step in cross {
            let centre = best;
            for d in [[step, 0], [-step, 0], [0, step], [0, -step]] {
                let v = [centre[0] + d[0], centre[1] + d[1]];
                let c = cost(v);
                if c < best_cost {
                    (best, best_cost) = (v, c);
                }
            }
        }
        (best, best_cost)
    }
}

/// What a search reads.
pub(crate) struct Inputs<'a> {
    pub pool: &'a Pool,
    pub cur: &'a Plane<u8>,
    pub cur_half: &'a Plane<u8>,
    pub reference: &'a Plane<u8>,
    pub ref_half: &'a Plane<u8>,
    /// The last frame's field.
    pub previous: Option<&'a MotionField>,
    /// Squared error per bit.
    pub lambda: f64,
}

impl Inputs<'_> {
    fn searcher(&self, row_step: usize) -> Searcher<'_> {
        Searcher {
            row_step,
            skip_lambda: self.lambda,
            cur: self.cur,
            cur_half: self.cur_half,
            reference: self.reference,
            ref_half: self.ref_half,
            lambda: self.lambda,
        }
    }
}

/// One root's search result.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Root {
    /// Its vector (for the window over its nodes' cells).
    pub v: [i32; 2],
    /// The window's centre in luma pixels.
    pub centre: [f64; 2],
    /// The window lies (partly) inside the picture.
    pub valid: bool,
    /// Its nodes keep its vector's offset unsearched.
    pub settled: bool,
    /// Mean absolute gradient of the window at half resolution: flat windows say little about
    /// the motion.
    pub texture: f64,
}

/// Root size in nodes, and roots across and down.
fn roots_layout(grid: &crate::motion::Grid) -> (usize, usize, usize) {
    let size = 1usize << root_log2(grid.log2);
    (size, grid.nx.div_ceil(size), grid.ny.div_ceil(size))
}

/// Nodes `[i0, i1) × [j0, j1)`'s cells, clipped to the picture.
fn cells(s: usize, w: usize, h: usize, [i0, i1, j0, j1]: [usize; 4]) -> Cell {
    Cell {
        x0: (i0 * s).saturating_sub(s / 2),
        x1: (i1 * s - s / 2).min(w),
        y0: (j0 * s).saturating_sub(s / 2),
        y1: (j1 * s - s / 2).min(h),
    }
}

fn sub(a: [i32; 2], b: [i32; 2]) -> [i32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [i32; 2], b: [i32; 2]) -> [i32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

/// The first pass: one vector per root, every root on its own, all in parallel. `field` holds a
/// global prediction (the search's start and the prior of its rate term).
pub(crate) fn search_roots(inp: &Inputs, field: &MotionField) -> Vec<Root> {
    let grid = field.grid;
    let (s, (w, h)) = (grid.spacing(), (inp.cur.width(), inp.cur.height()));
    let (size, rx, ry) = roots_layout(&grid);
    let previous = inp.previous.filter(|p| p.grid == grid);
    let prev_at = |n: usize| previous.and_then(|p| (!p.intra[n]).then_some(p.mv[n]));
    let mut searcher = inp.searcher(ROOT_ROW_STEP);
    // A weak rate term: a strong one keeps flat-ish roots at the prediction, and the global fit
    // (whose prediction it is) then drifts.
    searcher.lambda *= ROOT_LAMBDA_SCALE;
    inp.pool.map(rx * ry, |k| {
        let (i0, j0) = ((k % rx) * size, (k / rx) * size);
        let (i1, j1) = ((i0 + size).min(grid.nx), (j0 + size).min(grid.ny));
        let cell = cells(s, w, h, [i0, i1, j0, j1]);
        let centre = [
            (cell.x0 + cell.x1) as f64 / 2.0,
            (cell.y0 + cell.y1) as f64 / 2.0,
        ];
        let p = field.mv[j0 * grid.nx + i0];
        if cell.x0 >= cell.x1 || cell.y0 >= cell.y1 {
            return Root {
                v: p,
                centre,
                valid: false,
                settled: true,
                texture: 0.0,
            };
        }
        let mut cands = vec![p];
        for j in j0..j1 {
            cands.extend((i0..i1).filter_map(|i| prev_at(j * grid.nx + i)));
        }
        let (v, cost) = searcher.search(
            cell,
            p,
            &cands,
            |v| offset_bits(sub(v, p)),
            &ROOT_DIAMOND,
            &[8, 4],
        );
        // Predicted well enough as a whole, or so badly that the error is mostly texture and
        // noise: its nodes keep its offset.
        let per_px = cost / ((cell.x1 - cell.x0) * (cell.y1 - cell.y0)) as f64;
        Root {
            v,
            centre,
            valid: true,
            settled: !(ROOT_GOOD_SSE..=ROOT_BAD_SSE).contains(&per_px),
            texture: texture(inp.cur_half, cell.halved()),
        }
    })
}

/// The affine motion that fits the roots' vectors best, robustly: least squares, refitted twice
/// without the roots far off it (moving objects, parallax, flat areas). `None` with too few
/// roots.
pub(crate) fn fit_global(roots: &[Root]) -> Option<Affine> {
    let usable = |r: &Root| r.valid && r.texture >= MIN_TEXTURE;
    let mut used: Vec<bool> = roots.iter().map(usable).collect();
    let mut model = None;
    for _ in 0..3 {
        let picked: Vec<&Root> = roots
            .iter()
            .zip(&used)
            .filter(|(_, u)| **u)
            .map(|(r, _)| r)
            .collect();
        if picked.len() < 6 {
            break;
        }
        // Normal equations of d = a + b·x + c·y, shared by both components.
        let mut m = [[0f64; 3]; 3];
        let mut rhs = [[0f64; 3]; 2];
        for r in &picked {
            let f = [1.0, r.centre[0], r.centre[1]];
            for (row, &fa) in m.iter_mut().zip(&f) {
                for (e, &fb) in row.iter_mut().zip(&f) {
                    *e += fa * fb;
                }
            }
            for (rhs, &v) in rhs.iter_mut().zip(&r.v) {
                for (e, &fa) in rhs.iter_mut().zip(&f) {
                    *e += fa * f64::from(v) / 8.0;
                }
            }
        }
        let [x, y] = [solve3(m, rhs[0])?, solve3(m, rhs[1])?];
        let fit = Affine([x[0], x[1], x[2], y[0], y[1], y[2]]);
        let residual = |r: &Root| {
            let (dx, dy) = fit.at(r.centre[0], r.centre[1]);
            (dx - f64::from(r.v[0]) / 8.0).hypot(dy - f64::from(r.v[1]) / 8.0)
        };
        let mut res: Vec<f64> = picked.iter().map(|r| residual(r)).collect();
        res.sort_by(f64::total_cmp);
        let limit = (3.0 * res[res.len() / 2]).max(1.0);
        for (u, r) in used.iter_mut().zip(roots) {
            *u = usable(r) && residual(r) <= limit;
        }
        model = Some(fit);
    }
    model
}

/// Mean absolute horizontal plus vertical difference over `c` of `plane`.
fn texture(plane: &Plane<u8>, c: Cell) -> f64 {
    let (mut sum, mut n) = (0u32, 0u32);
    for y in c.y0..c.y1.min(plane.height() - 1) {
        let (a, b) = (
            plane.row_span(y, c.x0, c.x1 - c.x0),
            plane.row_span(y + 1, c.x0, c.x1 - c.x0),
        );
        sum += a
            .windows(2)
            .map(|p| u32::from(p[0].abs_diff(p[1])))
            .sum::<u32>();
        sum += a
            .iter()
            .zip(b)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum::<u32>();
        n += 2 * (c.x1 - c.x0) as u32;
    }
    f64::from(sum) / f64::from(n.max(1))
}

/// `m · x = b` by Cramer's rule; `None` when singular.
fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d.abs() < 1e-9 {
        return None;
    }
    Some(std::array::from_fn(|k| {
        let mut mk = m;
        for (row, &v) in mk.iter_mut().zip(&b) {
            row[k] = v;
        }
        det(&mk) / d
    }))
}

/// The second pass: every node's vector, all in parallel. `field` holds this frame's global
/// prediction. Each node chooses among its own root's offset from the global motion, its
/// neighbouring roots' and its last vector, refined by a quarter pel; neighbouring nodes thus
/// share offsets (cheap to code, and coherent) without waiting for each other. Leaves are single
/// nodes, merged into whole roots where all their nodes kept the same offset.
pub(crate) fn search_nodes(inp: &Inputs, field: &mut MotionField, roots: &[Root]) {
    let grid = field.grid;
    let (s, (w, h)) = (grid.spacing(), (inp.cur.width(), inp.cur.height()));
    let (size, rx, ry) = roots_layout(&grid);
    let previous = inp.previous.filter(|p| p.grid == grid);
    let pred = field.mv.clone();
    // Every root's offset from the global prediction at its top-left node.
    let offsets: Vec<[i32; 2]> = (0..rx * ry)
        .map(|k| {
            let n = (k / rx) * size * grid.nx + (k % rx) * size;
            let d = sub(roots[k].v, pred[n]);
            // Offsets are whole quarter pels.
            d.map(|c| c & !1)
        })
        .collect();
    let searcher = inp.searcher(ROW_STEP);
    let rows = inp.pool.map(grid.ny, |j| {
        (0..grid.nx)
            .map(|i| {
                let n = j * grid.nx + i;
                let p = pred[n];
                let (ri, rj) = (i / size, j / size);
                let (own, settled) = (offsets[rj * rx + ri], roots[rj * rx + ri].settled);
                let cell = cells(s, w, h, [i, i + 1, j, j + 1]);
                if settled || cell.x0 >= cell.x1 || cell.y0 >= cell.y1 {
                    return add(p, own);
                }
                let mut cands = vec![add(p, own), p];
                let mut near = |ri: Option<usize>, rj: Option<usize>| {
                    if let (Some(ri), Some(rj)) = (ri, rj)
                        && ri < rx
                        && rj < ry
                    {
                        cands.push(add(p, offsets[rj * rx + ri]));
                    }
                };
                near(ri.checked_sub(1), Some(rj));
                near(Some(ri + 1), Some(rj));
                near(Some(ri), rj.checked_sub(1));
                near(Some(ri), Some(rj + 1));
                cands.extend(previous.and_then(|p| (!p.intra[n]).then_some(p.mv[n])));
                // The root's own offset is coded once for all its nodes if they all keep it.
                let bits = |v: [i32; 2]| {
                    let d = sub(v, p);
                    if d == own {
                        offset_bits(d) / (size * size) as f64
                    } else {
                        offset_bits(d) + 2.0
                    }
                };
                let (v, _) = searcher.search(cell, p, &cands, bits, &[], &[4, 2]);
                v.map(|c| c.clamp(-MAX_MV, MAX_MV))
            })
            .collect::<Vec<_>>()
    });
    for (j, row) in rows.into_iter().enumerate() {
        field.mv[j * grid.nx..(j + 1) * grid.nx].copy_from_slice(&row);
    }
    field.depth.fill(0);
    let r = root_log2(grid.log2) as u8;
    for j0 in (0..grid.ny).step_by(size) {
        for i0 in (0..grid.nx).step_by(size) {
            let nodes: Vec<usize> = (j0..(j0 + size).min(grid.ny))
                .flat_map(|j| (i0..(i0 + size).min(grid.nx)).map(move |i| j * grid.nx + i))
                .collect();
            let offset = |n: usize| sub(field.mv[n], pred[n]);
            let first = offset(nodes[0]);
            if nodes.iter().all(|&n| offset(n) == first) {
                for &n in &nodes {
                    field.depth[n] = r;
                }
            }
        }
    }
}
