//! Global motion estimation (encoder only, DESIGN.md §6.1).
//!
//! The reference is the *source* of the frame the prediction will come from, not its
//! reconstruction: the camera motion between the two frames is the same, and the source is free
//! of coding artifacts, so estimation works as well at the lowest qualities as at the highest.
//!
//! 1. Textured 8×8 patches of the ¼-resolution frame are block-matched against the reference
//!    at ¼ resolution. Each search starts from the best of three guesses: the
//!    motion the previous frame predicts, a whole-frame translation found by matching row and
//!    column profiles, and no motion. A parabola through the SSD gives sub-pixel vectors.
//! 2. RANSAC fits an affine model to the patch vectors, rejecting independently moving objects
//!    and mismatches; least squares on the inliers refines it.
//! 3. A robust Gauss–Newton (Lucas–Kanade) refinement of the six affine parameters on the
//!    ½-resolution frames brings the whole model to sub-pixel accuracy.
//!
//! These are estimation steps only: the prediction itself never uses blocks.

use super::pyramid::Pyramid;
use super::{GlobalMotion, Grid, MAX_GLOBAL_MV};
use crate::Plane;
use crate::pool::Pool;
use crate::rng::SplitMix64;

/// Patch size at ¼ resolution (32×32 pixels at full resolution).
const PATCH: usize = 8;
/// Patches matched per frame, the most textured first.
const MAX_PATCHES: usize = 320;
/// Smallest structure-tensor eigenvalue of a usable patch (gradients as central differences).
const MIN_TEXTURE: f64 = 150.0;
/// Search radius around the best starting guess, at ¼ resolution.
const SEARCH: i32 = 3;
/// Profile-matching range at ¼ resolution (±128 pixels at full resolution).
const PROFILE_RANGE: i32 = 32;
/// RANSAC inlier distance, full-resolution pixels.
const INLIER_PX: f64 = 2.0;
const RANSAC_ROUNDS: usize = 100;
/// RANSAC rounds in realtime mode (§10.2): 30 lose nothing measurable.
const REALTIME_RANSAC_ROUNDS: usize = 30;
/// Gauss–Newton iterations at ½ resolution.
const REFINE_ROUNDS: usize = 4;
/// The refinement sums its samples in this many fixed chunks, one task each, and adds the chunk
/// sums in order: the same floating-point result for any thread count.
const CHUNKS: usize = 16;
/// Samples for the refinement, the most textured first. Of the 57,600 a 720p frame offers, about
/// 36,000 have some texture; the strongest 16,000 pin the model down almost as well (0.07 px rms
/// against 0.06 on rigid motion) in under half the time.
const MAX_SAMPLES: usize = 16_000;

/// An affine displacement field in full-resolution pixels:
/// `d(x, y) = (a0 + a1·x + a2·y, a3 + a4·x + a5·y)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Affine(pub [f64; 6]);

impl Affine {
    pub fn translation(dx: f64, dy: f64) -> Self {
        Self([dx, 0.0, 0.0, dy, 0.0, 0.0])
    }

    pub fn at(&self, x: f64, y: f64) -> (f64, f64) {
        let a = &self.0;
        (a[0] + a[1] * x + a[2] * y, a[3] + a[4] * x + a[5] * y)
    }

    /// The model at the grid corners, in 1/8 pel.
    pub fn to_global(self, grid: &Grid) -> GlobalMotion {
        GlobalMotion {
            corners: grid.corners().map(|(x, y)| {
                let (dx, dy) = self.at(x as f64, y as f64);
                [dx, dy].map(|v| ((v * 8.0).round() as i32).clamp(-MAX_GLOBAL_MV, MAX_GLOBAL_MV))
            }),
        }
    }

    /// Largest displacement difference from `other` over a `w × h` frame.
    fn distance(&self, other: &Affine, w: f64, h: f64) -> f64 {
        [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
            .iter()
            .map(|&(x, y)| {
                let (a, b) = (self.at(x, y), other.at(x, y));
                (a.0 - b.0).hypot(a.1 - b.1)
            })
            .fold(0.0, f64::max)
    }
}

/// What global motion estimation found.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlobalEstimate {
    pub model: Affine,
    /// Patches matched, and how many agreed with the fitted model.
    pub matches: usize,
    pub inliers: usize,
}

#[derive(Clone, Copy, Debug)]
struct Patch {
    /// Top-left corner at ¼ resolution.
    x: usize,
    y: usize,
    texture: f64,
}

/// A patch vector: centre and displacement, in full-resolution pixels.
#[derive(Clone, Copy, Debug)]
struct Match {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

/// A sample for the refinement: half-resolution position, centred and scaled coordinates, value
/// and gradient.
#[derive(Clone, Copy, Debug)]
struct Sample {
    x: f32,
    y: f32,
    u: f32,
    v: f32,
    value: f32,
    gx: f32,
    gy: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct GlobalEstimator {
    width: usize,
    height: usize,
    cur: Pyramid,
    /// Whether `cur` holds the frame being encoded (built by [`Self::estimate`]).
    cur_built: bool,
    /// The reference frame's source.
    reference: Pyramid,
    rng: SplitMix64,
    patches: Vec<Patch>,
    matches: Vec<Match>,
    samples: Vec<Sample>,
    /// Realtime mode (§10.2): fewer RANSAC rounds, and the RANSAC model stands unrefined.
    pub realtime: bool,
}

impl GlobalEstimator {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cur: Pyramid::new(width, height),
            cur_built: false,
            reference: Pyramid::new(width, height),
            rng: SplitMix64::new(0x5EED),
            patches: Vec::new(),
            matches: Vec::new(),
            samples: Vec::new(),
            realtime: false,
        }
    }

    /// Takes the source of the frame just sent: the next frame's reference.
    /// When [`Self::estimate`] already built this frame's pyramid, it simply becomes the
    /// reference.
    pub fn set_reference(&mut self, pool: &Pool, source: &Plane<u8>) {
        if std::mem::take(&mut self.cur_built) {
            std::mem::swap(&mut self.reference, &mut self.cur);
        } else {
            self.reference.build(pool, source);
        }
    }

    /// The share of 16×16 blocks that `model` predicts from the reference worse than the block's
    /// own activity (plus a noise margin), at ¼ resolution. Call after [`Self::estimate`]. Near
    /// 1 for a scene cut.
    ///
    /// Each block moves by the model's displacement at its centre, in 1/16 of a quarter pixel:
    /// across 16 full-resolution pixels an affine model varies by far less than the margin.
    pub fn unpredictable_share(&self, pool: &Pool, model: &Affine) -> f64 {
        const B: usize = 4;
        const MARGIN: u32 = 3;
        let (cur, reference) = (&self.cur.quarter, &self.reference.quarter);
        let (w, h) = (cur.width(), cur.height());
        let (rw, rh) = (reference.width() as i32, reference.height() as i32);
        // Badly predicted blocks per row of blocks.
        let worse_per_row = pool.map(h / B, |row| {
            let by = row * B;
            let mut worse = 0usize;
            let mut block = [0u32; B * B];
            for bx in (0..w / B).map(|b| b * B) {
                // Quarter pixel (x, y) is centred on full-resolution (4x + 1.5, 4y + 1.5).
                let centre = |b: usize| 4.0 * (b as f64 + (B as f64 - 1.0) / 2.0) + 1.5;
                let (dx, dy) = model.at(centre(bx), centre(by));
                let (qx, qy) = ((dx * 4.0).round() as i32, (dy * 4.0).round() as i32);
                let (ix, iy) = (bx as i32 + (qx >> 4), by as i32 + (qy >> 4));
                let (fx, fy) = ((qx & 15) as u32, (qy & 15) as u32);
                let wts = [
                    (16 - fx) * (16 - fy),
                    fx * (16 - fy),
                    (16 - fx) * fy,
                    fx * fy,
                ];
                // The reference at 256 times its value, coordinates clamped to the plane.
                let at = |x: i32, y: i32| {
                    u32::from(
                        reference.get(x.clamp(0, rw - 1) as usize, y.clamp(0, rh - 1) as usize),
                    )
                };
                for (k, r) in block.iter_mut().enumerate() {
                    let (x, y) = (ix + (k % B) as i32, iy + (k / B) as i32);
                    *r = wts[0] * at(x, y)
                        + wts[1] * at(x + 1, y)
                        + wts[2] * at(x, y + 1)
                        + wts[3] * at(x + 1, y + 1);
                }
                let values = (by..by + B).flat_map(|y| cur.row_span(y, bx, B));
                let sum: u32 = values.clone().map(|&v| u32::from(v)).sum();
                // Errors at 256 times the sample scale; the mean at 16 times (B² = 16).
                let (mut inter, mut intra) = (0u32, 0u32);
                for (&v, &r) in values.zip(&block) {
                    inter += (256 * u32::from(v)).abs_diff(r);
                    intra += (16 * u32::from(v)).abs_diff(sum);
                }
                worse += usize::from(inter > 16 * intra + 256 * MARGIN * (B * B) as u32);
            }
            worse
        });
        let worse: usize = worse_per_row.iter().sum();
        let blocks = (h / B) * (w / B);
        if blocks == 0 {
            0.0
        } else {
            worse as f64 / blocks as f64
        }
    }

    /// Global motion of `luma` against the reference. `hint` is the expected motion (the
    /// previous frame's).
    pub fn estimate(&mut self, pool: &Pool, luma: &Plane<u8>, hint: &Affine) -> GlobalEstimate {
        self.cur.build(pool, luma);
        self.cur_built = true;
        self.select_patches(pool);
        let profile = self.profile_translation();
        self.match_patches(pool, hint, profile);
        let (mut model, inliers) = self.fit();
        if inliers < 6 && inliers * 3 < self.matches.len().max(1) {
            // Too little agreement for an affine model: fall back to a translation.
            model = profile.map_or(*hint, |(dx, dy)| Affine::translation(4.0 * dx, 4.0 * dy));
        }
        if !self.realtime
            && let Some(refined) = self.refine(pool, model)
        {
            model = refined;
        }
        GlobalEstimate {
            model,
            matches: self.matches.len(),
            inliers,
        }
    }

    /// The most textured patches: largest smaller eigenvalue of the structure tensor, so edges
    /// (motion known in one direction only) rank below corners and texture.
    fn select_patches(&mut self, pool: &Pool) {
        let q = &self.cur.quarter;
        let (w, h) = (q.width(), q.height());
        self.patches.clear();
        if w < PATCH + 2 || h < PATCH + 2 {
            return;
        }
        // One row of patches per task, in order. Sums of 64 products below 2^16 fit 32 bits.
        let rows: Vec<usize> = (1..h - PATCH).step_by(PATCH).collect();
        let found = pool.map(rows.len(), |r| {
            let py = rows[r];
            let mut patches = Vec::new();
            for px in (1..w - PATCH).step_by(PATCH) {
                let (mut a, mut b, mut c) = (0i32, 0i32, 0i32);
                for y in py..py + PATCH {
                    let (up, row, down) = (q.row(y - 1), q.row(y), q.row(y + 1));
                    for x in px..px + PATCH {
                        let gx = i32::from(row[x + 1]) - i32::from(row[x - 1]);
                        let gy = i32::from(down[x]) - i32::from(up[x]);
                        a += gx * gx;
                        b += gx * gy;
                        c += gy * gy;
                    }
                }
                let (a, b, c) = (f64::from(a), f64::from(b), f64::from(c));
                let texture = (a + c) / 2.0 - (((a - c) / 2.0).powi(2) + b * b).sqrt();
                if texture >= MIN_TEXTURE {
                    patches.push(Patch {
                        x: px,
                        y: py,
                        texture,
                    });
                }
            }
            patches
        });
        self.patches.extend(found.into_iter().flatten());
        self.patches
            .sort_unstable_by(|p, q| q.texture.total_cmp(&p.texture));
        self.patches.truncate(MAX_PATCHES);
    }

    /// Whole-frame translation at ¼ resolution from mean-removed column and row profiles.
    fn profile_translation(&self) -> Option<(f64, f64)> {
        let (a, b) = (&self.cur.quarter, &self.reference.quarter);
        let (w, h) = (a.width(), a.height());
        let columns = |p: &Plane<u8>| -> Vec<f64> {
            let mut s = vec![0.0; w];
            for row in p.rows() {
                for (s, &v) in s.iter_mut().zip(row) {
                    *s += f64::from(v);
                }
            }
            s.iter().map(|v| v / h as f64).collect()
        };
        let rows = |p: &Plane<u8>| -> Vec<f64> {
            p.rows()
                .map(|r| r.iter().map(|&v| f64::from(v)).sum::<f64>() / w as f64)
                .collect()
        };
        let shift = |cur: &[f64], reference: &[f64]| -> Option<f64> {
            let n = cur.len() as i32;
            let range = PROFILE_RANGE.min(n / 4);
            if range < 1 {
                return None;
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (mc, mr) = (mean(cur), mean(reference));
            let cost = |s: i32| {
                let (mut sum, mut count) = (0.0, 0);
                for x in 0.max(-s)..n.min(n - s) {
                    sum += ((cur[x as usize] - mc) - (reference[(x + s) as usize] - mr)).abs();
                    count += 1;
                }
                sum / f64::from(count.max(1))
            };
            let costs: Vec<f64> = (-range..=range).map(cost).collect();
            let best = (0..costs.len()).min_by(|&i, &j| costs[i].total_cmp(&costs[j]))?;
            let frac = if best > 0 && best + 1 < costs.len() {
                parabola(costs[best - 1], costs[best], costs[best + 1])
            } else {
                0.0
            };
            Some(best as f64 - f64::from(range) + frac)
        };
        Some((shift(&columns(a), &columns(b))?, shift(&rows(a), &rows(b))?))
    }

    /// Block-matches every patch, one task each; patches that cannot be matched drop out.
    fn match_patches(&mut self, pool: &Pool, hint: &Affine, profile: Option<(f64, f64)>) {
        let (cur, reference) = (&self.cur.quarter, &self.reference.quarter);
        let (w, h) = (cur.width() as i32, cur.height() as i32);
        let patches = &self.patches;
        let matches = pool.map(patches.len(), |k| {
            let p = &patches[k];
            let (px, py) = (p.x as i32, p.y as i32);
            let valid = |dx: i32, dy: i32| {
                let (x, y) = (px + dx, py + dy);
                x >= 0 && y >= 0 && x + PATCH as i32 <= w && y + PATCH as i32 <= h
            };
            let cost = |dx: i32, dy: i32| {
                block_cost(
                    cur,
                    p.x,
                    p.y,
                    reference,
                    (px + dx) as usize,
                    (py + dy) as usize,
                    false,
                )
            };
            let centre = (
                4.0 * (p.x as f64 + (PATCH as f64 - 1.0) / 2.0) + 1.5,
                4.0 * (p.y as f64 + (PATCH as f64 - 1.0) / 2.0) + 1.5,
            );
            let predicted = hint.at(centre.0, centre.1);
            let mut guesses = vec![
                (
                    (predicted.0 / 4.0).round() as i32,
                    (predicted.1 / 4.0).round() as i32,
                ),
                (0, 0),
            ];
            if let Some((dx, dy)) = profile {
                guesses.push((dx.round() as i32, dy.round() as i32));
            }
            let (mut bx, mut by, mut best) = guesses
                .into_iter()
                .filter(|&(dx, dy)| valid(dx, dy))
                .map(|(dx, dy)| (dx, dy, cost(dx, dy)))
                .min_by_key(|&(_, _, c)| c)?;
            // Full search around the best guess; recentre while the minimum sits on the edge.
            for _ in 0..3 {
                let (cx, cy) = (bx, by);
                for dy in cy - SEARCH..=cy + SEARCH {
                    for dx in cx - SEARCH..=cx + SEARCH {
                        if valid(dx, dy) {
                            let c = cost(dx, dy);
                            if c < best {
                                (bx, by, best) = (dx, dy, c);
                            }
                        }
                    }
                }
                if (bx - cx).abs() < SEARCH && (by - cy).abs() < SEARCH {
                    break;
                }
            }
            let ssd = |dx: i32, dy: i32| {
                block_cost(
                    cur,
                    p.x,
                    p.y,
                    reference,
                    (px + dx) as usize,
                    (py + dy) as usize,
                    true,
                ) as f64
            };
            let centre_ssd = ssd(bx, by);
            let fx = if valid(bx - 1, by) && valid(bx + 1, by) {
                parabola(ssd(bx - 1, by), centre_ssd, ssd(bx + 1, by))
            } else {
                0.0
            };
            let fy = if valid(bx, by - 1) && valid(bx, by + 1) {
                parabola(ssd(bx, by - 1), centre_ssd, ssd(bx, by + 1))
            } else {
                0.0
            };
            Some(Match {
                x: centre.0,
                y: centre.1,
                dx: 4.0 * (f64::from(bx) + fx),
                dy: 4.0 * (f64::from(by) + fy),
            })
        });
        self.matches.clear();
        self.matches.extend(matches.into_iter().flatten());
    }

    /// RANSAC over 3-point affine models, then least squares on the inliers. Returns the model
    /// and its inlier count (0 when there are fewer than three matches).
    fn fit(&mut self) -> (Affine, usize) {
        let n = self.matches.len();
        if n < 3 {
            return (Affine::default(), 0);
        }
        // Coordinates are normalized to about ±1 for conditioning.
        let norm = Normalizer::new(self.width, self.height);
        let mut best: Option<(usize, f64, Affine)> = None;
        let rounds = if self.realtime {
            REALTIME_RANSAC_ROUNDS
        } else {
            RANSAC_ROUNDS
        };
        for _ in 0..rounds {
            let i = self.rng.below(n as u64) as usize;
            let j = self.rng.below(n as u64) as usize;
            let k = self.rng.below(n as u64) as usize;
            if i == j || j == k || i == k {
                continue;
            }
            let Some(model) = norm.fit(&[self.matches[i], self.matches[j], self.matches[k]]) else {
                continue;
            };
            let (count, err) = self.score(&model);
            if best.is_none_or(|(c, e, _)| count > c || (count == c && err < e)) {
                best = Some((count, err, model));
            }
        }
        let Some((_, _, mut model)) = best else {
            return (Affine::default(), 0);
        };
        let mut inliers = 0;
        for _ in 0..2 {
            let set: Vec<Match> = self
                .matches
                .iter()
                .copied()
                .filter(|m| residual(&model, m) < INLIER_PX)
                .collect();
            inliers = set.len();
            match norm.fit(&set) {
                Some(m) if set.len() >= 3 => model = m,
                _ => break,
            }
        }
        (model, inliers)
    }

    /// Inliers of `model` and their summed squared residuals.
    fn score(&self, model: &Affine) -> (usize, f64) {
        let mut count = 0;
        let mut err = 0.0;
        for m in &self.matches {
            let r = residual(model, m);
            if r < INLIER_PX {
                count += 1;
                err += r * r;
            }
        }
        (count, err)
    }

    /// Robust Gauss–Newton refinement of all six parameters at ½ resolution. Every other
    /// pixel of every other row is a sample; Huber weights discount outliers such as moving
    /// objects and occlusions. Returns `None` when there is too little to go on or the result
    /// drifts implausibly far from the start.
    fn refine(&mut self, pool: &Pool, start: Affine) -> Option<Affine> {
        let (cur, reference) = (&self.cur.half, &self.reference.half);
        let (w, h) = (cur.width(), cur.height());
        if w < 8 || h < 8 {
            return None;
        }
        // Half-resolution pixel (x, y) sits at full-resolution (2x + 0.5, 2y + 0.5). In half
        // pixels the model is p0 + p1·x + p2·y with p0 = (a0 + (a1 + a2)/2)/2.
        let (sx, sy) = (w as f64 / 2.0, h as f64 / 2.0);
        let a = start.0;
        // Parameters in centred, scaled half-res coordinates: d = q0 + q1·u + q2·v, u = (x − sx)/sx.
        let to_q = |a: [f64; 6]| -> [f64; 6] {
            let mut q = [0.0; 6];
            for c in 0..2 {
                let (a0, a1, a2) = (a[3 * c], a[3 * c + 1], a[3 * c + 2]);
                let p0 = (a0 + (a1 + a2) / 2.0) / 2.0;
                q[3 * c] = p0 + a1 * sx + a2 * sy;
                q[3 * c + 1] = a1 * sx;
                q[3 * c + 2] = a2 * sy;
            }
            q
        };
        let from_q = |q: [f64; 6]| -> [f64; 6] {
            let mut a = [0.0; 6];
            for c in 0..2 {
                let (a1, a2) = (q[3 * c + 1] / sx, q[3 * c + 2] / sy);
                let p0 = q[3 * c] - a1 * sx - a2 * sy;
                a[3 * c] = 2.0 * p0 - (a1 + a2) / 2.0;
                a[3 * c + 1] = a1;
                a[3 * c + 2] = a2;
            }
            a
        };
        // Every other pixel of every other row, the most textured only: flat ones carry no
        // motion. The threshold on |gx| + |gy| (central differences, so twice the gradient)
        // keeps at most MAX_SAMPLES, and at least 2.
        let strength = |up: &[u8], row: &[u8], down: &[u8], x: usize| {
            (i32::from(row[x + 1]) - i32::from(row[x - 1])).unsigned_abs()
                + (i32::from(down[x]) - i32::from(up[x])).unsigned_abs()
        };
        let mut hist = [0usize; 512];
        for y in (2..h - 2).step_by(2) {
            let (up, row, down) = (cur.row(y - 1), cur.row(y), cur.row(y + 1));
            for x in (2..w - 2).step_by(2) {
                hist[strength(up, row, down, x) as usize] += 1;
            }
        }
        let mut kept = 0;
        let mut threshold = hist.len();
        while threshold > 2 && kept + hist[threshold - 1] <= MAX_SAMPLES {
            threshold -= 1;
            kept += hist[threshold];
        }
        self.samples.clear();
        for y in (2..h - 2).step_by(2) {
            let (up, row, down) = (cur.row(y - 1), cur.row(y), cur.row(y + 1));
            for x in (2..w - 2).step_by(2) {
                if strength(up, row, down, x) as usize >= threshold {
                    let gx = (f32::from(row[x + 1]) - f32::from(row[x - 1])) / 2.0;
                    let gy = (f32::from(down[x]) - f32::from(up[x])) / 2.0;
                    self.samples.push(Sample {
                        x: x as f32,
                        y: y as f32,
                        u: ((x as f64 - sx) / sx) as f32,
                        v: ((y as f64 - sy) / sy) as f32,
                        value: f32::from(row[x]),
                        gx,
                        gy,
                    });
                }
            }
        }
        let mut q = to_q(a);
        let mut huber = 20.0;
        for _ in 0..REFINE_ROUNDS {
            let samples_all = &self.samples;
            let chunk = samples_all.len().div_ceil(CHUNKS).max(1);
            let partial = pool.map(CHUNKS, |c| {
                let mut sums = Sums::default();
                let end = ((c + 1) * chunk).min(samples_all.len());
                for s in samples_all.get(c * chunk..end).unwrap_or(&[]) {
                    let (u, v) = (f64::from(s.u), f64::from(s.v));
                    let (dx, dy) = (q[0] + q[1] * u + q[2] * v, q[3] + q[4] * u + q[5] * v);
                    let Some(r) = bilinear(reference, f64::from(s.x) + dx, f64::from(s.y) + dy)
                    else {
                        continue;
                    };
                    let e = r - f64::from(s.value);
                    sums.hist[(e.abs() as usize).min(255)] += 1;
                    sums.samples += 1;
                    let (gx, gy) = (f64::from(s.gx), f64::from(s.gy));
                    let wgt = if e.abs() <= huber {
                        1.0
                    } else {
                        huber / e.abs()
                    };
                    let j = [gx, gx * u, gx * v, gy, gy * u, gy * v];
                    // The upper triangle only: the matrix is symmetric.
                    for (r, &jr) in j.iter().enumerate() {
                        let wj = wgt * jr;
                        sums.b[r] += wj * e;
                        for (h, &jc) in sums.hm[r][r..].iter_mut().zip(&j[r..]) {
                            *h += wj * jc;
                        }
                    }
                }
                sums
            });
            let mut total = Sums::default();
            for p in &partial {
                total.add(p);
            }
            let Sums {
                mut hm,
                b,
                hist,
                samples,
            } = total;
            // Mirror the upper triangle.
            for r in 1..6 {
                let (upper, lower) = hm.split_at_mut(r);
                for (c, row) in upper.iter().enumerate() {
                    lower[0][c] = row[r];
                }
            }
            if samples < 64 {
                return None;
            }
            let step = solve(hm, b.map(|v| -v))?;
            for (q, s) in q.iter_mut().zip(step) {
                *q += s;
            }
            // Huber threshold from the median absolute residual.
            let mut acc = 0;
            let median = hist
                .iter()
                .position(|&c| {
                    acc += c;
                    acc * 2 >= samples
                })
                .unwrap_or(0);
            huber = (2.5 * 1.4826 * (median as f64 + 0.5)).max(3.0);
            if step.iter().all(|s| s.abs() < 0.005) {
                break;
            }
        }
        let refined = Affine(from_q(q));
        (refined.distance(&start, self.width as f64, self.height as f64) < 8.0).then_some(refined)
    }
}

/// One chunk's share of a Gauss–Newton step: the upper triangle of `Σ w·J·Jᵀ`, `Σ w·J·e`, and
/// the residuals' magnitude histogram.
#[derive(Clone)]
struct Sums {
    hm: [[f64; 6]; 6],
    b: [f64; 6],
    hist: [u32; 256],
    samples: u32,
}

impl Default for Sums {
    fn default() -> Self {
        Self {
            hm: [[0.0; 6]; 6],
            b: [0.0; 6],
            hist: [0; 256],
            samples: 0,
        }
    }
}

impl Sums {
    fn add(&mut self, other: &Sums) {
        for (row, o) in self.hm.iter_mut().zip(&other.hm) {
            for (h, o) in row.iter_mut().zip(o) {
                *h += o;
            }
        }
        for (b, o) in self.b.iter_mut().zip(&other.b) {
            *b += o;
        }
        for (h, o) in self.hist.iter_mut().zip(&other.hist) {
            *h += o;
        }
        self.samples += other.samples;
    }
}

/// Least-squares affine fitting in normalized coordinates.
struct Normalizer {
    cx: f64,
    cy: f64,
    s: f64,
}

impl Normalizer {
    fn new(w: usize, h: usize) -> Self {
        Self {
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
            s: (w.max(h) as f64 / 2.0).max(1.0),
        }
    }

    /// Least-squares affine model through the matches (exact for three).
    fn fit(&self, matches: &[Match]) -> Option<Affine> {
        let mut m = [[0.0f64; 3]; 3];
        let (mut bx, mut by) = ([0.0f64; 3], [0.0f64; 3]);
        for t in matches {
            let v = [1.0, (t.x - self.cx) / self.s, (t.y - self.cy) / self.s];
            for r in 0..3 {
                bx[r] += v[r] * t.dx;
                by[r] += v[r] * t.dy;
                for c in 0..3 {
                    m[r][c] += v[r] * v[c];
                }
            }
        }
        let ax = solve(m, bx)?;
        let ay = solve(m, by)?;
        // Back to pixel coordinates.
        let un = |a: [f64; 3]| {
            let (a1, a2) = (a[1] / self.s, a[2] / self.s);
            (a[0] - a1 * self.cx - a2 * self.cy, a1, a2)
        };
        let ((x0, x1, x2), (y0, y1, y2)) = (un(ax), un(ay));
        Some(Affine([x0, x1, x2, y0, y1, y2]))
    }
}

fn residual(model: &Affine, m: &Match) -> f64 {
    let (dx, dy) = model.at(m.x, m.y);
    (dx - m.dx).hypot(dy - m.dy)
}

/// Vertex offset of the parabola through `(−1, l), (0, c), (1, r)`, within ±½.
fn parabola(l: f64, c: f64, r: f64) -> f64 {
    let den = l + r - 2.0 * c;
    if den <= 0.0 {
        0.0
    } else {
        ((l - r) / (2.0 * den)).clamp(-0.5, 0.5)
    }
}

/// SAD (or SSD) between `PATCH × PATCH` blocks of two planes.
fn block_cost(
    a: &Plane<u8>,
    ax: usize,
    ay: usize,
    b: &Plane<u8>,
    bx: usize,
    by: usize,
    squared: bool,
) -> u32 {
    let mut sum = 0u32;
    for r in 0..PATCH {
        let (ra, rb) = (a.row_span(ay + r, ax, PATCH), b.row_span(by + r, bx, PATCH));
        for (&p, &q) in ra.iter().zip(rb) {
            let d = u32::from(p.abs_diff(q));
            sum += if squared { d * d } else { d };
        }
    }
    sum
}

/// Bilinear sample; `None` outside the plane.
fn bilinear(p: &Plane<u8>, x: f64, y: f64) -> Option<f64> {
    if !(x >= 0.0 && y >= 0.0 && x < (p.width() - 1) as f64 && y < (p.height() - 1) as f64) {
        return None;
    }
    let (x0, y0) = (x as usize, y as usize);
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let (r0, r1) = (p.row_span(y0, x0, 2), p.row_span(y0 + 1, x0, 2));
    let top = f64::from(r0[0]) + fx * (f64::from(r0[1]) - f64::from(r0[0]));
    let bottom = f64::from(r1[0]) + fx * (f64::from(r1[1]) - f64::from(r1[0]));
    Some(top + fy * (bottom - top))
}

/// Solves `a·x = b` by Gaussian elimination with partial pivoting; `None` if singular.
pub(crate) fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    let scale = a
        .iter()
        .flatten()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 * scale {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let (upper, lower) = a.split_at_mut(col + 1);
        let pivot_row = &upper[col];
        for (row, r) in lower.iter_mut().zip(col + 1..) {
            let f = row[col] / pivot_row[col];
            for (x, &p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= f * p;
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for r in (0..N).rev() {
        let s: f64 = (r + 1..N).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth random texture with detail at several scales.
    pub(crate) fn texture(seed: u64) -> impl Fn(f64, f64) -> f64 {
        let mut rng = SplitMix64::new(seed);
        let waves: Vec<(f64, f64, f64, f64)> = (0..24)
            .map(|i| {
                let f = 0.02 + 0.25 * (i as f64 / 24.0).powi(2);
                let angle = rng.below(6283) as f64 / 1000.0;
                let phase = rng.below(6283) as f64 / 1000.0;
                (
                    f * angle.cos(),
                    f * angle.sin(),
                    phase,
                    30.0 / (1.0 + 4.0 * f),
                )
            })
            .collect();
        move |x, y| {
            128.0
                + waves
                    .iter()
                    .map(|&(fx, fy, ph, amp)| amp * (fx * x + fy * y + ph).sin())
                    .sum::<f64>()
                    .clamp(-127.0, 127.0)
        }
    }

    /// Renders the texture seen through `model`: pixel (x, y) shows texture point (x, y) + d.
    pub(crate) fn render(
        tex: &impl Fn(f64, f64) -> f64,
        w: usize,
        h: usize,
        model: &Affine,
    ) -> Plane<u8> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = model.at(x as f64, y as f64);
                let v = tex(x as f64 + dx, y as f64 + dy);
                p.set(x, y, v.round().clamp(0.0, 255.0) as u8);
            }
        }
        p
    }

    #[test]
    fn solves_linear_systems() {
        let a = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]];
        let x = solve(a, [7.0, 4.0, 7.0]).unwrap();
        for (v, e) in x.iter().zip([1.0, 1.0, 1.0]) {
            assert!((v - e).abs() < 1e-9);
        }
        assert!(solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]).is_none());
    }

    #[test]
    fn finds_affine_camera_motion() {
        let (w, h) = (320, 192);
        let tex = texture(7);
        // The reference shows the texture as is. The current frame shows, at pixel p, the
        // texture point p + d(p): its content sits in the reference displaced by d.
        let reference = render(&tex, w, h, &Affine::default());
        for truth in [
            Affine::translation(5.3, -2.7),
            Affine([3.0, 0.01, -0.02, -1.5, 0.02, 0.01]),
            Affine([-11.0, -0.015, 0.0, 6.0, 0.0, -0.015]),
        ] {
            let cur = render(&tex, w, h, &truth);
            let (mut est, pool) = (GlobalEstimator::new(w, h), Pool::new(3));
            est.set_reference(&pool, &reference);
            let found = est.estimate(&pool, &cur, &Affine::default());
            let err = found.model.distance(&truth, w as f64, h as f64);
            assert!(
                err < 0.15,
                "{truth:?}: found {:?} ({} of {} patches), corner error {err:.3} px",
                found.model,
                found.inliers,
                found.matches
            );
        }
    }
}
