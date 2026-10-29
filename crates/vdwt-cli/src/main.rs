//! `vdwt`: command-line tools for the VDWT codec.
//!
//! Built statically for `aarch64-unknown-linux-musl`, the one binary runs the self-test and the
//! benchmarks directly on a Cortex-A55 board.

mod rt;
mod y4m;

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::{Duration, Instant};

use vdwt::bitstream;
use vdwt::dwt::{self, Dwt53, Geometry, MAX_LEVELS, Orient, Wavelet};
use vdwt::encoder::{
    DEFAULT_CHROMA_STEP_SCALE, DEFAULT_EFFORT, DEFAULT_INTER_RECON_OFFSET, DEFAULT_INTER_ROUNDING,
    DEFAULT_INTRA_QP_OFFSET, DEFAULT_ISOLATED_THRESHOLD, DEFAULT_RECON_OFFSET, DEFAULT_ROUNDING,
    DEFAULT_SCENE_CUT, MAX_THREADS,
};
use vdwt::metrics::YuvPsnr;
use vdwt::motion::warp::predict_frame;
use vdwt::motion::{Filter, GlobalMotion, Grid, MotionField, Prediction};
use vdwt::plane::{center_u8, uncenter_to_u8};
use vdwt::postfilter;
use vdwt::profile::{Stage, StageTimes};
use vdwt::rng::SplitMix64;
use vdwt::{Decoder, Encoder, EncoderConfig, Frame, Isa, Plane, RateControl};
use y4m::{Y4mInfo, Y4mReader, Y4mWriter};

const USAGE: &str = "\
vdwt: VDWT codec tools

USAGE:
    vdwt <command> [options]

COMMANDS:
    encode IN OUT.vdwt [--qp 32 | --bitrate RATE[@FRAME],...] [--max-delay 1.5]
           [--quality-cap 48] [--frames N] [--recon RECON.y4m] [--sharpen 0.6]
           [--frame-log FRAMES.csv] [CODEC OPTIONS]
        Encode at a fixed qp, or at a link rate that may change at given frames (e.g.
        4M@0,1M@60,6M@120). --max-delay bounds queueing delay, in frame durations.
        --quality-cap stops coding a frame finer than this estimated luma PSNR and banks the
        leftover link capacity for later frames (0: no ceiling).
        --frame-log writes per-frame type, size, qp, motion and rate-control state.
        IN is a .y4m file (4:2:0, 4:2:2, 4:4:4 or mono, 8-16 bits; coded as 8-bit 4:2:0),
        `-` for Y4M on standard input, or any other video file ffmpeg can read.
    decode IN.vdwt OUT.y4m [--threads N] [--postfilter auto]
        Decode a stream to 8-bit 4:2:0 Y4M; OUT `-` writes it to standard output. --sharpen applies the optional display post-filter (luma unsharp mask;
        it does not change the bitstream or any reference picture).
    rd IN.y4m [--qp 16,24,32,40,48] [--frames 10] [CODEC OPTIONS]
        Rate-distortion table over several qp values, checking that decoding matches.
    band-stats IN.y4m [--qp 16,24,...] [--frames 10] [CODEC OPTIONS]
        Developer: per-band CSV of coded sizes and rate estimates, for fitting the rate model.
    pred-dump IN.y4m OUT.y4m [--qp 32] [--frames N] [CODEC OPTIONS]
        Developer: encode P-frames and write each frame's motion-compensated prediction in
        display order (mid-grey for intra frames), for residual/orientation analysis.
    train-contexts OUT.rs IN.y4m... [--qp 24,32,40,48,56] [--frames 60] [--min-symbols 64]
                   [CODEC OPTIONS]
        Developer: trains the coder's default context tables on clips (P-frame runs, and the
        first frames intra-only) and writes them as crates/vdwt/src/entropy/trained.rs.
    rt-encode IN.y4m OUT.vdrt --bitrate RATE[@FRAME],... [--max-delay 1.5] [--frames N]
              [--tile-roots 0 (auto)] [--refresh 30] [--mtu 1200] [--recon R.y4m]
              [--loop-filter] [--mesh-search] [--global-only]
        Realtime codec for lossy links: independent 64-pixel tiles in self-contained packets of
        at most --mtu bytes, every tile refreshed once per --refresh frames. Slower and better:
        --loop-filter, the main encoder's --mesh-search; --global-only skips
        local motion.
    rt-decode IN.vdrt OUT.y4m [--loss 0.05] [--burst 1] [--seed 1] [--ref SOURCE.y4m]
        Decode an rt stream through a lossy channel: packets dropped at --loss, in bursts of
        --burst packets on average; --ref reports PSNR against the source.
    info
        CPU and SIMD kernel support.
    selftest [--isa all|scalar|avx2|neon] [--cases 300] [--seed 1]
        Bit-exactness of every kernel family: the DWT against the JPEG 2000 reference
        transform, motion compensation against the portable code.
    bench [--width 1280] [--height 720] [--levels 5] [--chroma-levels 4] [--frames 300]
          [--isa all|NAME] [--ghz 1.8] [--input FILE.y4m]
        Single-core speed of the 5/3 DWT and of motion compensation on one 4:2:0 frame.
    dwt-roundtrip FILE.y4m [--levels 5] [--chroma-levels 4] [--frames N]
        Forward and inverse transform of every frame; fails unless it is lossless.
    dwt-dump FILE.y4m OUT.pgm [--frame 0] [--levels 3] [--gain 4]
        The luma subbands of one frame as a greyscale image.
    psnr A.y4m B.y4m
        Per-frame and average PSNR between two clips.
    postfilter IN.y4m OUT.y4m [--postfilter MODE] [--postfilter-strength S]
        Apply a display post-filter to every frame of a Y4M clip (band2 is the wide mid-frequency
        boost; unsharp is the generic 5×5 mask). auto on encode chooses one strength per scene and
        writes it in the frame headers; auto on decode uses that signalled strength.

CODEC OPTIONS:
    --levels 5 --chroma-levels 4    wavelet depth
    --block 64                      band coder quadtree root (4..128, power of two)
    --rounding 0.25                 quantizer θ: smaller widens the dead zone
    --sharpen A                      display unsharp strength (0: off; 0.6 typical)
    --postfilter MODE                none, auto, unsharp, band or band2 (wider boost); auto is
                                     the encoder-chosen per-scene strength carried in the header
    --postfilter-strength S          fixed strength for unsharp/band/band2 (0.6 / 2.0 / 2.0)
    --recon-offset 0.2              reconstruction δ
    --inter-rounding 0.12           θ for P-frames
    --inter-recon-offset 0.1        δ for P-frames
    --chroma-scale 0.4              chroma step multiplier (larger favours luma)
    --no-parent                     no parent-band contexts
    --no-ll-pred                    no LL prediction
    --lossless                      exact reconstruction
    --isolated T                    zero lone ±1 detail values below T steps (0: off)
    --isa NAME                      transform kernels (output is identical for all)
    --threads N                     threads (default: up to 4; output is identical for all)

MOTION OPTIONS:
    --intra-only                    no P-frames
    --intra-period 0                frames between intra frames (0: only the first and cuts)
    --intra-qp-offset -8            added to --qp for intra frames (ultraheavy: -7)
    --mesh auto                     motion node spacing in pixels (8, 16, 32 or 64); auto: 32 when
                                    the shorter side has 480 pixels or more, otherwise 16
    --filter cubic                  interpolation: cubic, bilinear, sharp or smooth
    --prediction auto               motion compensation: auto (per frame), warp, overlapped
    --global-only                   global motion only, no per-node refinement
    --motion-lambda 1               rate-distortion weight of motion decisions (ultraheavy: 0.8)
    --frame-aq S                    causal frame qp adaptation strength (0: off; default 2)
    --frame-aq-clamp N              largest qp step frame AQ may take (default 6)
    --tune psnr                     psnr: squared-error fits (best PSNR); perceptual: trades PSNR
                                    for detail (VMAF): --loop-sharpen 0.4 --finest-tilt auto
    --loop-sharpen A                fit the in-loop filter to a sharpened source (0: MSE, the default)
    --finest-tilt M|auto            multiplier on the finest luma level's steps (default 1.0); auto
                                    picks 1.6 at qp<=46, else 1.0 (fixed qp only)
    --hh-tilt M                     extra multiplier on the finest HH band (experimental)
    --mid-tilt M                    multiplier on the second-finest level's weights (experimental)
    --b-qp-offset N                 B-frame qp offset from its anchor (default 9)
    --b-static-qp-offset N          B-frame qp offset on a static camera (default 6)
    --sparse-zero                   zero lone ±1 block values even beside nonzero blocks
    --temporal-motion               trial the previous frame's offsets as a motion predictor
    --tier medium                   encoder tier: light (fastest), medium (default), heavy,
                                    ultraheavy (general-purpose, smallest stream: a wider
                                    search, frames in pairs coded as P-frames or a B-frame group,
                                    the motion grid probed on the first frames; one frame more
                                    latency, a 3-frame delay limit, about 1.7x heavy's encode
                                    time); --effort 0, 1, 2, 3 name the same tiers
    --realtime                      cheapest search; no loop filter under rate control, for
                                    small CPUs (under 1% more bits, 0.05–0.1 dB at fixed qp)
    --no-loop-filter                no in-loop Wiener filter
    --no-guide                      flat texture guide, skip the prediction DWT (static and
                                    low-motion content; up to 0.9 dB on detailed motion)
    --wavelet 97                    97 (lifted 9/7) or 53 (5/3; lossless always uses it)
    --scene-cut 0.5                 share of badly predicted blocks that makes a frame intra
";

/// Options that take no value.
const FLAGS: [&str; 14] = [
    "lossless",
    "no-parent",
    "no-ll-pred",
    "intra-only",
    "global-only",
    "no-loop-filter",
    "no-guide",
    "realtime",
    "verbose",
    "cubic",
    "loop-filter",
    "mesh-search",
    "sparse-zero",
    "temporal-motion",
];

fn main() -> ExitCode {
    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }
    let command = raw.remove(0);
    let result = Args::parse(raw).and_then(|args| match command.as_str() {
        "encode" => encode(args),
        "decode" => decode(args),
        "rt-encode" => rt::encode(args),
        "rt-decode" => rt::decode(args),
        "rd" => rd(args),
        "band-stats" => band_stats(args),
        "pred-dump" => pred_dump(args),
        "train-contexts" => train_contexts(args),
        "info" => info(args),
        "selftest" => selftest(args),
        "bench" => bench(args),
        "dwt-roundtrip" => dwt_roundtrip(args),
        "dwt-dump" => dwt_dump(args),
        "psnr" => psnr(args),
        "postfilter" => postfilter_cmd(args),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}` (try `vdwt help`)")),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `--name value` options and positional arguments.
struct Args {
    positional: Vec<String>,
    named: HashMap<String, String>,
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self, String> {
        let mut args = Self {
            positional: Vec::new(),
            named: HashMap::new(),
        };
        let mut it = raw.into_iter();
        while let Some(arg) = it.next() {
            match arg.strip_prefix("--") {
                Some(name) if FLAGS.contains(&name) => {
                    args.named.insert(name.to_owned(), String::new());
                }
                Some(name) => {
                    let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                    args.named.insert(name.to_owned(), value);
                }
                None => args.positional.push(arg),
            }
        }
        Ok(args)
    }

    fn get<T: FromStr>(&mut self, name: &str, default: T) -> Result<T, String> {
        match self.named.remove(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value `{v}` for --{name}")),
        }
    }

    fn string(&mut self, name: &str) -> Option<String> {
        self.named.remove(name)
    }

    fn flag(&mut self, name: &str) -> bool {
        self.named.remove(name).is_some()
    }

    /// Fails on leftover options or a wrong number of positional arguments.
    fn finish(&self, positional: usize) -> Result<(), String> {
        if let Some(name) = self.named.keys().next() {
            return Err(format!("unknown option --{name}"));
        }
        if self.positional.len() != positional {
            return Err(format!(
                "expected {positional} file argument(s), got {}",
                self.positional.len()
            ));
        }
        Ok(())
    }
}

fn isa_list(spec: &str) -> Result<Vec<Isa>, String> {
    if spec == "all" {
        return Ok(Isa::supported().collect());
    }
    let isa = Isa::from_name(spec).ok_or_else(|| format!("unknown ISA `{spec}`"))?;
    if !isa.is_supported() {
        return Err(format!("{isa} kernels are not available on this CPU/build"));
    }
    Ok(vec![isa])
}

fn check_levels(levels: usize) -> Result<usize, String> {
    if levels > MAX_LEVELS {
        Err(format!("at most {MAX_LEVELS} levels are supported"))
    } else {
        Ok(levels)
    }
}

fn open_y4m(path: &str) -> Result<Y4mReader, String> {
    Y4mReader::open(Path::new(path)).map_err(|e| format!("{path}: {e}"))
}

/// Whether two source frames are identical in every plane: a static video's frames can then be
/// coded as 7-byte Repeat frames, keeping the decoder's picture and spending no residual bits.
/// Sparse luma difference, for detecting an opening cut before the first frame is encoded.
fn frame_difference(a: &Frame, b: &Frame) -> f64 {
    let h = a.y.height();
    let step = (h / 64).max(1);
    let (mut sum, mut n) = (0u64, 0u64);
    for y in (0..h).step_by(step) {
        let (p, q) = (a.y.row(y), b.y.row(y));
        for x in (0..p.len()).step_by(8) {
            sum += u64::from(p[x].abs_diff(q[x]));
            n += 1;
        }
    }
    sum as f64 / n.max(1) as f64
}

fn same_frame(a: &Frame, b: &Frame) -> bool {
    a.planes().len() == b.planes().len()
        && a.planes().iter().zip(b.planes()).all(|(p, q)| {
            p.width() == q.width() && p.height() == q.height() && p.rows().eq(q.rows())
        })
}

fn info(args: Args) -> Result<(), String> {
    args.finish(0)?;
    println!("vdwt {}", env!("CARGO_PKG_VERSION"));
    println!(
        "target   {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    if let Some(cpu) = cpu_description() {
        println!("cpu      {cpu}");
    }
    println!(
        "threads  {}",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    let isas: Vec<&str> = Isa::supported().map(Isa::name).collect();
    println!("kernels  {} (default: {})", isas.join(", "), Isa::best());
    Ok(())
}

/// The CPU model from `/proc/cpuinfo`: the model name on x86, core types on Arm.
fn cpu_description() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    let field = |line: &str, key: &str| {
        line.split_once(':')
            .filter(|(k, _)| k.trim() == key)
            .map(|(_, v)| v.trim().to_owned())
    };
    if let Some(name) = text.lines().find_map(|l| field(l, "model name")) {
        return Some(name);
    }
    let mut parts: Vec<(String, usize)> = Vec::new();
    for part in text.lines().filter_map(|l| field(l, "CPU part")) {
        match parts.iter_mut().find(|(p, _)| *p == part) {
            Some((_, n)) => *n += 1,
            None => parts.push((part, 1)),
        }
    }
    let describe = |part: &str| match part {
        "0xd03" => "Cortex-A53".to_owned(),
        "0xd05" => "Cortex-A55".to_owned(),
        "0xd08" => "Cortex-A72".to_owned(),
        "0xd09" => "Cortex-A73".to_owned(),
        "0xd0a" => "Cortex-A75".to_owned(),
        "0xd0b" => "Cortex-A76".to_owned(),
        "0xd0d" => "Cortex-A77".to_owned(),
        "0xd41" => "Cortex-A78".to_owned(),
        other => format!("part {other}"),
    };
    let cores: Vec<String> = parts
        .iter()
        .map(|(p, n)| format!("{n}x {}", describe(p)))
        .collect();
    (!cores.is_empty()).then(|| cores.join(" + "))
}

fn selftest(mut args: Args) -> Result<(), String> {
    let isas = isa_list(&args.string("isa").unwrap_or_else(|| "all".into()))?;
    let cases = args.get("cases", 300usize)?;
    let seed = args.get("seed", 1u64)?;
    args.finish(0)?;
    let mut failed = false;
    for isa in isas {
        let start = Instant::now();
        let result = dwt::verify(isa, cases, seed)
            .and_then(|()| dwt::verify_size(isa, 1280, 720, 5, seed))
            .and_then(|()| dwt::verify_size(isa, 640, 360, 4, seed))
            .and_then(|()| dwt::d97::verify(isa, cases, seed))
            .and_then(|()| vdwt::motion::warp::verify(isa, cases.div_ceil(3), seed))
            .and_then(|()| vdwt::loopfilter::verify(cases, seed));
        match result {
            Ok(()) => println!(
                "{isa:>6}  pass  (DWT: {cases} random planes + 1280x720 and 640x360 frames; \
                 warp: {} random fields; loop filter: {cases} random rows; {:.1} s)",
                cases.div_ceil(3),
                start.elapsed().as_secs_f64()
            ),
            Err(e) => {
                println!("{isa:>6}  FAIL  {e}");
                failed = true;
            }
        }
    }
    if failed {
        Err("self-test failed".into())
    } else {
        Ok(())
    }
}

/// Transform buffers for one plane of a frame.
struct Component {
    levels: usize,
    work: Plane<i16>,
    coeffs: Plane<i16>,
    out: Plane<i16>,
}

fn components(frame: &Frame, luma_levels: usize, chroma_levels: usize) -> [Component; 3] {
    let levels = [luma_levels, chroma_levels, chroma_levels];
    let planes = frame.planes();
    std::array::from_fn(|c| {
        let (w, h) = (planes[c].width(), planes[c].height());
        Component {
            levels: levels[c],
            work: Plane::new(w, h),
            coeffs: Plane::new(w, h),
            out: Plane::new(w, h),
        }
    })
}

struct Timing {
    median: Duration,
    min: Duration,
}

/// Runs `f` a few times to warm up, then `iters` timed times.
fn time_iters(iters: usize, mut f: impl FnMut()) -> Timing {
    for _ in 0..iters.clamp(1, 10) {
        f();
    }
    let mut times: Vec<Duration> = (0..iters)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    Timing {
        median: times[times.len() / 2],
        min: times[0],
    }
}

/// Noise; the transform's speed does not depend on content.
fn synthetic_frame(width: usize, height: usize) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut rng = SplitMix64::new(42);
    for plane in frame.planes_mut() {
        for row in plane.rows_mut() {
            row.fill_with(|| rng.next_u64() as u8);
        }
    }
    frame
}

fn bench(mut args: Args) -> Result<(), String> {
    let input = args.string("input");
    let width = args.get("width", 1280usize)?;
    let height = args.get("height", 720usize)?;
    let levels = check_levels(args.get("levels", 5usize)?)?;
    let chroma_levels = check_levels(args.get("chroma-levels", 4usize)?)?;
    let frames = args.get("frames", 300usize)?.max(1);
    let ghz = args.get("ghz", 1.8f64)?;
    let isas = isa_list(&args.string("isa").unwrap_or_else(|| "all".into()))?;
    args.finish(0)?;
    if width == 0 || height == 0 {
        return Err("frame size must be nonzero".into());
    }

    let frame = match input {
        Some(path) => {
            let mut reader = open_y4m(&path)?;
            let mut frame = reader.new_frame();
            if !reader.read_frame(&mut frame).map_err(|e| e.to_string())? {
                return Err(format!("{path}: no frames"));
            }
            frame
        }
        None => synthetic_frame(width, height),
    };
    let (w, h) = (frame.width(), frame.height());
    let samples: usize = frame.planes().iter().map(|p| p.width() * p.height()).sum();

    println!(
        "5/3 DWT, {w}x{h} 4:2:0 ({samples} samples), luma {levels} levels, chroma {chroma_levels} levels"
    );
    println!(
        "median of {frames} frames on one core; cycles/sample assume the core runs at {ghz} GHz"
    );
    for &isa in &isas {
        let mut dwt = Dwt53::with_isa(w, h, isa).expect("isa_list only returns supported ISAs");
        let mut comps = components(&frame, levels, chroma_levels);

        for (comp, src) in comps.iter_mut().zip(frame.planes()) {
            center_u8(src, &mut comp.work);
            let original = comp.work.clone();
            dwt.forward(&mut comp.work, comp.levels, &mut comp.coeffs);
            dwt.inverse(&comp.coeffs, comp.levels, &mut comp.out);
            if !comp.out.rows().eq(original.rows()) {
                return Err(format!("{isa}: round trip is not lossless"));
            }
        }

        // `forward` uses its input as scratch, so later iterations transform leftovers. That is
        // fine for timing: the work does not depend on the values.
        let fwd = time_iters(frames, || {
            for c in comps.iter_mut() {
                dwt.forward(&mut c.work, c.levels, &mut c.coeffs);
            }
        });
        let inv = time_iters(frames, || {
            for c in comps.iter_mut() {
                dwt.inverse(&c.coeffs, c.levels, &mut c.out);
            }
        });
        for (name, t) in [("forward", &fwd), ("inverse", &inv)] {
            let ns = t.median.as_secs_f64() * 1e9 / samples as f64;
            println!(
                "{isa:>6}  {name}  {:7.3} ms/frame (min {:.3})  {ns:6.3} ns/sample  {:6.2} cycles/sample",
                ms(t.median),
                ms(t.min),
                ns * ghz
            );
        }
        let both = fwd.median + inv.median;
        println!(
            "{isa:>6}  fwd+inv  {:7.3} ms/frame = {:.1}% of one core at 60 fps",
            ms(both),
            both.as_secs_f64() * 60.0 * 100.0
        );
    }

    // Motion compensation of the same frame by a drone-like field: zoom, yaw and a sub-pixel
    // pan from the global model, plus small node-to-node variation.
    let global_for = |grid: &Grid| GlobalMotion {
        corners: grid.corners().map(|(x, y)| {
            let (x, y) = (x as f64 - w as f64 / 2.0, y as f64 - h as f64 / 2.0);
            let (dx, dy) = (3.3 + 0.01 * x - 0.008 * y, -1.7 + 0.008 * x + 0.01 * y);
            [(dx * 8.0).round() as i32, (dy * 8.0).round() as i32]
        }),
    };
    let mut rng = SplitMix64::new(5);
    let mut pred = Frame::new(w, h);
    for log2 in [5, 4] {
        let grid = Grid::new(w, h, log2);
        let mut field = MotionField::new(grid);
        field.set_global(&global_for(&grid));
        for mv in &mut field.mv {
            mv[0] += rng.range_i32(-3, 3);
            mv[1] += rng.range_i32(-3, 3);
        }
        println!("motion compensation, {}-pixel grid", 1 << log2);
        for &isa in &isas {
            for prediction in Prediction::ALL {
                for filter in Filter::ALL {
                    let t = time_iters(frames.min(60), || {
                        predict_frame(&frame, &field, prediction, filter, isa, &mut pred);
                    });
                    let ns = t.median.as_secs_f64() * 1e9 / samples as f64;
                    println!(
                        "{isa:>6}  {:<10} {:<8}  {:7.3} ms/frame (min {:.3})  {ns:6.3} ns/sample  {:6.2} cycles/sample",
                        prediction.name(),
                        filter.name(),
                        ms(t.median),
                        ms(t.min),
                        ns * ghz
                    );
                }
            }
        }
    }
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn dwt_roundtrip(mut args: Args) -> Result<(), String> {
    let levels = check_levels(args.get("levels", 5usize)?)?;
    let chroma_levels = check_levels(args.get("chroma-levels", 4usize)?)?;
    let max_frames = args.get("frames", usize::MAX)?;
    args.finish(1)?;
    let path = &args.positional[0];
    let mut reader = open_y4m(path)?;
    let (mut frame, mut back) = (reader.new_frame(), reader.new_frame());
    let mut dwt = Dwt53::new(frame.width(), frame.height());
    let mut comps = components(&frame, levels, chroma_levels);
    let (mut n, mut fwd, mut inv) = (0usize, Duration::ZERO, Duration::ZERO);
    while n < max_frames
        && reader
            .read_frame(&mut frame)
            .map_err(|e| format!("{path}: {e}"))?
    {
        for ((comp, src), dst) in comps.iter_mut().zip(frame.planes()).zip(back.planes_mut()) {
            center_u8(src, &mut comp.work);
            let start = Instant::now();
            dwt.forward(&mut comp.work, comp.levels, &mut comp.coeffs);
            let mid = Instant::now();
            dwt.inverse(&comp.coeffs, comp.levels, &mut comp.out);
            inv += mid.elapsed();
            fwd += mid - start;
            uncenter_to_u8(&comp.out, dst);
        }
        if !frame
            .planes()
            .iter()
            .zip(back.planes())
            .all(|(a, b)| a.rows().eq(b.rows()))
        {
            return Err(format!("frame {n} did not survive the round trip"));
        }
        n += 1;
    }
    if n == 0 {
        return Err(format!("{path}: no frames"));
    }
    let info = &reader.info;
    println!(
        "{n} frames {}x{} @ {}/{} fps, luma {levels} levels, chroma {chroma_levels}: lossless ({})",
        info.width,
        info.height,
        info.fps_num,
        info.fps_den,
        dwt.isa()
    );
    println!(
        "forward {:.3} ms/frame, inverse {:.3} ms/frame",
        ms(fwd) / n as f64,
        ms(inv) / n as f64
    );
    Ok(())
}

fn dwt_dump(mut args: Args) -> Result<(), String> {
    let index = args.get("frame", 0usize)?;
    let levels = check_levels(args.get("levels", 3usize)?)?;
    let gain = args.get("gain", 4.0f64)?;
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let mut reader = open_y4m(input)?;
    let mut frame = reader.new_frame();
    for i in 0..=index {
        if !reader
            .read_frame(&mut frame)
            .map_err(|e| format!("{input}: {e}"))?
        {
            return Err(format!("{input} has only {i} frames"));
        }
    }
    let (w, h) = (frame.width(), frame.height());
    let mut work = Plane::new(w, h);
    let mut coeffs = Plane::new(w, h);
    center_u8(&frame.y, &mut work);
    Dwt53::new(w, h).forward(&mut work, levels, &mut coeffs);

    // LL keeps pixel scale (the 5/3 low-pass has unit DC gain); detail bands are amplified
    // around mid-grey.
    let mut image = vec![0u8; w * h];
    for band in Geometry::new(w, h, levels).subbands() {
        let (offset, scale) = if band.orient == Orient::LL {
            (128.0, 1.0)
        } else {
            (128.0, gain)
        };
        for y in band.y0..band.y0 + band.height {
            for (x, &c) in coeffs.row_span(y, band.x0, band.width).iter().enumerate() {
                image[y * w + band.x0 + x] =
                    (offset + f64::from(c) * scale).clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
    pgm.extend_from_slice(&image);
    fs::write(output, pgm).map_err(|e| format!("{output}: {e}"))?;
    println!("wrote {output}: frame {index}, {levels} levels, detail gain {gain}");
    Ok(())
}

fn postfilter_cmd(mut args: Args) -> Result<(), String> {
    let post = post_filter(&mut args)?;
    if matches!(post, PostFilter::Auto) {
        return Err("--postfilter auto needs a VDWT stream: use `decode`".into());
    }
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let (info, frames) = read_frames(input, usize::MAX)?;
    let mut writer =
        Y4mWriter::create(Path::new(output), &info).map_err(|e| format!("{output}: {e}"))?;
    let mut display = Frame::new(info.width, info.height);
    for frame in &frames {
        let filtered = post.apply(frame, &mut display, 0);
        let shown: &Frame = if filtered { &display } else { frame };
        writer.write_frame(shown).map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    println!(
        "post-filtered {} frames {}x{} -> {output}",
        frames.len(),
        info.width,
        info.height
    );
    Ok(())
}

fn psnr(args: Args) -> Result<(), String> {
    args.finish(2)?;
    let (pa, pb) = (&args.positional[0], &args.positional[1]);
    let (mut ra, mut rb) = (open_y4m(pa)?, open_y4m(pb)?);
    if (ra.info.width, ra.info.height) != (rb.info.width, rb.info.height) {
        return Err("the clips differ in frame size".into());
    }
    let (mut fa, mut fb) = (ra.new_frame(), rb.new_frame());
    let mut sum = [0.0f64; 4];
    let mut n = 0usize;
    loop {
        let more_a = ra.read_frame(&mut fa).map_err(|e| format!("{pa}: {e}"))?;
        let more_b = rb.read_frame(&mut fb).map_err(|e| format!("{pb}: {e}"))?;
        if !(more_a && more_b) {
            if more_a != more_b {
                eprintln!("warning: the clips differ in length; compared the first {n} frames");
            }
            break;
        }
        let p = YuvPsnr::between(&fa, &fb);
        println!(
            "frame {n:5}  Y {:6.2}  U {:6.2}  V {:6.2}  YUV {:6.2} dB",
            p.y,
            p.u,
            p.v,
            p.weighted()
        );
        for (s, v) in sum.iter_mut().zip([p.y, p.u, p.v, p.weighted()]) {
            *s += v;
        }
        n += 1;
    }
    if n == 0 {
        return Err("no frames to compare".into());
    }
    let avg = sum.map(|s| s / n as f64);
    println!(
        "average      Y {:6.2}  U {:6.2}  V {:6.2}  YUV {:6.2} dB  ({n} frames)",
        avg[0], avg[1], avg[2], avg[3]
    );
    Ok(())
}

/// Encoder settings shared by `encode` and `rd`.
struct CodecOptions {
    luma_levels: usize,
    chroma_levels: usize,
    block_log2: u32,
    rounding: f64,
    recon_offset: f64,
    inter_rounding: f64,
    inter_recon_offset: f64,
    chroma_scale: f64,
    parent_contexts: bool,
    ll_prediction: bool,
    lossless: bool,
    isa: Isa,
    p_frames: bool,
    intra_period: u32,
    intra_qp_offset: i8,
    /// `None`: the encoder's default for the frame size.
    mesh_log2: Option<u32>,
    filter: Option<Filter>,
    mesh_refinement: bool,
    motion_lambda: f64,
    frame_aq: f64,
    frame_aq_clamp: u8,
    loop_sharpen: f64,
    finest_tilt: f64,
    finest_tilt_auto: bool,
    hh_tilt: f64,
    mid_tilt: f64,
    b_qp_offset: u8,
    b_static_qp_offset: u8,
    sparse_zero: bool,
    temporal_motion: bool,
    scene_cut: f64,
    isolated: f64,
    prediction: Option<Prediction>,
    threads: usize,
    effort: u8,
    /// The ultraheavy tier: frames in pairs, each coded as P-frames or a B-frame group, and the
    /// motion grid probed on the first frames.
    adaptive: bool,
    realtime: bool,
    loop_filter: bool,
    wavelet: Wavelet,
    no_guide: bool,
}

impl CodecOptions {
    fn parse(args: &mut Args) -> Result<Self, String> {
        let tier = effort(args)?;
        let ultra = tier == 3;
        let block = args.get("block", 64usize)?;
        if !block.is_power_of_two() || !(4..=128).contains(&block) {
            return Err("--block must be a power of two from 4 to 128".into());
        }
        let isa = match args.string("isa") {
            Some(name) => isa_list(&name)?[0],
            None => Isa::best(),
        };
        let mesh = match args.string("mesh").as_deref() {
            None | Some("auto") => None,
            Some(m) => match m.parse::<usize>() {
                Ok(m @ (8 | 16 | 32 | 64)) => Some(m),
                _ => return Err("--mesh must be auto, 8, 16, 32 or 64".into()),
            },
        };
        // `--tune psnr` (default) keeps the squared-error fits; `--tune perceptual` trades PSNR
        // for detail (VMAF): the sharpened loop-filter target and the qp-adaptive finest tilt.
        let perceptual = match args.string("tune").as_deref() {
            None | Some("psnr") => false,
            Some("perceptual") => true,
            Some(t) => return Err(format!("unknown --tune `{t}` (psnr, perceptual)")),
        };
        let (finest_tilt, finest_tilt_auto) = match args.string("finest-tilt").as_deref() {
            None => (1.0, perceptual),
            Some("auto") => (1.0, true),
            Some(v) => (
                v.parse::<f64>()
                    .map_err(|_| "--finest-tilt must be a number in 0.5..=2 or auto".to_string())?,
                false,
            ),
        };
        let filter = match args.string("filter").as_deref() {
            None | Some("auto") => None,
            Some(name) => Some(Filter::from_name(name).ok_or_else(|| {
                format!("unknown filter `{name}` (auto, bilinear, cubic, sharp, smooth)")
            })?),
        };
        let loop_sharpen = args.get("loop-sharpen", if perceptual { 0.4f64 } else { 0.0 })?;
        Ok(Self {
            luma_levels: check_levels(args.get("levels", 5usize)?)?,
            chroma_levels: check_levels(args.get("chroma-levels", 4usize)?)?,
            block_log2: block.trailing_zeros(),
            rounding: args.get("rounding", DEFAULT_ROUNDING)?,
            recon_offset: args.get("recon-offset", DEFAULT_RECON_OFFSET)?,
            inter_rounding: args.get("inter-rounding", DEFAULT_INTER_ROUNDING)?,
            inter_recon_offset: args.get("inter-recon-offset", DEFAULT_INTER_RECON_OFFSET)?,
            chroma_scale: args.get("chroma-scale", DEFAULT_CHROMA_STEP_SCALE)?,
            parent_contexts: !args.flag("no-parent"),
            ll_prediction: !args.flag("no-ll-pred"),
            lossless: args.flag("lossless"),
            isa,
            p_frames: !args.flag("intra-only"),
            intra_period: args.get("intra-period", 0u32)?,
            intra_qp_offset: args.get(
                "intra-qp-offset",
                if ultra {
                    ULTRA_INTRA_QP_OFFSET
                } else {
                    DEFAULT_INTRA_QP_OFFSET
                },
            )?,
            mesh_log2: mesh.map(|m| m.trailing_zeros()),
            filter,
            mesh_refinement: !args.flag("global-only"),
            motion_lambda: args.get(
                "motion-lambda",
                if ultra { ULTRA_MOTION_LAMBDA } else { 1.0 },
            )?,
            frame_aq: args.get("frame-aq", 2.0f64)?,
            frame_aq_clamp: args.get("frame-aq-clamp", 6u8)?,
            loop_sharpen,
            finest_tilt,
            finest_tilt_auto,
            hh_tilt: args.get("hh-tilt", 1.0f64)?,
            mid_tilt: args.get("mid-tilt", 1.0f64)?,
            b_qp_offset: args.get("b-qp-offset", 9u8)?,
            b_static_qp_offset: args.get("b-static-qp-offset", 6u8)?,
            sparse_zero: args.flag("sparse-zero"),
            temporal_motion: args.flag("temporal-motion"),
            scene_cut: args.get("scene-cut", DEFAULT_SCENE_CUT)?,
            isolated: args.get("isolated", DEFAULT_ISOLATED_THRESHOLD)?,
            prediction: match args.string("prediction").as_deref() {
                None | Some("auto") => None,
                Some(name) => Some(Prediction::from_name(name).ok_or_else(|| {
                    format!("unknown prediction `{name}` (auto, warp, overlapped)")
                })?),
            },
            threads: threads(args)?,
            effort: tier,
            adaptive: ultra,
            realtime: args.flag("realtime"),
            loop_filter: !args.flag("no-loop-filter"),
            no_guide: args.flag("no-guide"),
            wavelet: match args.string("wavelet").as_deref() {
                None | Some("97") => Wavelet::Lifted97,
                Some("53") => Wavelet::Reversible53,
                Some(name) => return Err(format!("unknown wavelet `{name}` (97, 53)")),
            },
        })
    }

    /// [`Self::config`] for `frames`: the ultraheavy tier probes its motion grid on the first
    /// frames unless one is given.
    fn config_for(
        &self,
        info: &Y4mInfo,
        qp: u8,
        frames: &[Frame],
    ) -> Result<EncoderConfig, String> {
        let mut cfg = self.config(info, qp);
        cfg.total_frames = Some(frames.len().min(u16::MAX as usize) as u16);
        // A hard cut into the opening I-frame leaves that reference carrying the whole shot;
        // spend the extra bits on it. Softer changes (a moderate scene shift) do not pay.
        let default_intra = if cfg.effort >= 3 {
            ULTRA_INTRA_QP_OFFSET
        } else {
            DEFAULT_INTRA_QP_OFFSET
        };
        if cfg.intra_qp_offset == default_intra
            && frames.len() > 1
            && frame_difference(&frames[0], &frames[1]) > 50.0
        {
            cfg.intra_qp_offset = -9;
        }
        if self.mesh_log2.is_none() && frames.len() > 1 {
            let head: Vec<&Frame> = frames.iter().take(TRIAL_FRAMES).collect();
            let static_mesh =
                vdwt::encoder::probe_static_mesh(&cfg, &head).map_err(|e| e.to_string())?;
            match static_mesh {
                // A static camera below 50 fps already has the finer grid.
                Some(m) if m < cfg.mesh_log2 => cfg.mesh_log2 = m,
                // Otherwise the adaptive probe decides from medium up (forest: 16, 3.6–5.2% fewer
                // bits). Light and realtime keep their grid: on forest the finer one costs them
                // 36–46% more encode time.
                _ if cfg.effort >= 1 && !cfg.realtime => {
                    cfg.mesh_log2 =
                        vdwt::encoder::probe_mesh(&cfg, &head).map_err(|e| e.to_string())?
                }
                Some(m) => cfg.mesh_log2 = m,
                None => {}
            }
        }
        Ok(cfg)
    }

    fn config(&self, info: &Y4mInfo, qp: u8) -> EncoderConfig {
        let mut cfg = EncoderConfig::new(info.width, info.height);
        cfg.fps_num = info.fps_num;
        cfg.fps_den = info.fps_den;
        cfg.luma_levels = self.luma_levels;
        cfg.chroma_levels = self.chroma_levels;
        cfg.block_log2 = self.block_log2;
        cfg.qp = qp;
        cfg.lossless = self.lossless;
        cfg.rounding = self.rounding;
        cfg.inter_rounding = self.inter_rounding;
        cfg.inter_recon_offset = self.inter_recon_offset;
        cfg.recon_offset = self.recon_offset;
        cfg.chroma_step_scale = self.chroma_scale;
        cfg.parent_contexts = self.parent_contexts;
        cfg.ll_prediction = self.ll_prediction;
        cfg.isa = self.isa;
        cfg.p_frames = self.p_frames;
        cfg.intra_period = self.intra_period;
        cfg.intra_qp_offset = self.intra_qp_offset;
        if let Some(mesh_log2) = self.mesh_log2 {
            cfg.mesh_log2 = mesh_log2;
        }
        cfg.filter = self.filter;
        cfg.mesh_refinement = self.mesh_refinement;
        cfg.motion_lambda = self.motion_lambda;
        cfg.frame_aq = self.frame_aq;
        cfg.frame_aq_clamp = self.frame_aq_clamp;
        cfg.loop_sharpen = self.loop_sharpen;
        cfg.finest_tilt = self.finest_tilt;
        cfg.finest_tilt_auto = self.finest_tilt_auto;
        cfg.hh_tilt = self.hh_tilt;
        cfg.mid_tilt = self.mid_tilt;
        cfg.b_qp_offset = self.b_qp_offset;
        cfg.b_static_qp_offset = self.b_static_qp_offset;
        cfg.sparse_zero = self.sparse_zero;
        cfg.temporal_motion = self.temporal_motion;
        cfg.scene_cut = self.scene_cut;
        cfg.threads = self.threads;
        cfg.isolated_threshold = self.isolated;
        cfg.prediction = self.prediction;
        cfg.effort = self.effort;
        // P-frames this much coarser than `--qp` (fixed qp only). Realtime as light (0 before:
        // 1 is 0.4% better on YUV-PSNR 6:1:1 with the cubic filter).
        cfg.inter_qp_offset = if self.realtime {
            1
        } else {
            match cfg.effort {
                1 => 2,
                2 => 2,
                3 => 1,
                _ => 1,
            }
        };
        cfg.realtime = self.realtime;
        // Realtime skips the in-loop Wiener filter under rate control (its bits do not pay
        // there) and keeps it at fixed qp. The flat texture guide stays opt-in (`--no-guide`):
        // it saves time but costs 0.2 dB on simple content and up to 0.9 dB on detailed or
        // moving content, so it is a per-sequence choice rather than a tier default.
        cfg.loop_filter = self.loop_filter;
        cfg.guide = !self.no_guide;
        cfg.wavelet = self.wavelet;
        cfg
    }
}

fn read_frames(path: &str, max: usize) -> Result<(Y4mInfo, Vec<Frame>), String> {
    let mut reader = open_y4m(path)?;
    let mut frames = Vec::new();
    while frames.len() < max {
        let mut frame = reader.new_frame();
        if !reader
            .read_frame(&mut frame)
            .map_err(|e| format!("{path}: {e}"))?
        {
            break;
        }
        frames.push(frame);
    }
    if frames.is_empty() {
        return Err(format!("{path}: no frames"));
    }
    Ok((reader.info.clone(), frames))
}

fn fps(info: &Y4mInfo) -> f64 {
    f64::from(info.fps_num) / f64::from(info.fps_den)
}

/// A link-rate schedule: `RATE[@FRAME],...`, rates in bits/s with an optional `k` or `M`
/// suffix, each holding from FRAME on. `3M` alone means 3 Mbit/s throughout.
fn parse_schedule(spec: &str) -> Result<Vec<(usize, u64)>, String> {
    let rate = |r: &str| -> Result<u64, String> {
        let (num, mul) = match r.strip_suffix(['k', 'K']) {
            Some(n) => (n, 1e3),
            None => match r.strip_suffix('M') {
                Some(n) => (n, 1e6),
                None => (r, 1.0),
            },
        };
        let v: f64 = num
            .trim()
            .parse()
            .map_err(|_| format!("invalid bitrate `{r}`"))?;
        if v * mul < 1.0 {
            return Err(format!("bitrate `{r}` must be positive"));
        }
        Ok((v * mul).round() as u64)
    };
    let mut schedule = Vec::new();
    for (i, part) in spec.split(',').enumerate() {
        let (r, frame) = match part.split_once('@') {
            Some((r, f)) => (
                r,
                f.trim()
                    .parse()
                    .map_err(|_| format!("invalid frame index in `{part}`"))?,
            ),
            None if i == 0 => (part, 0),
            None => return Err(format!("`{part}`: later rates need @FRAME")),
        };
        schedule.push((frame, rate(r)?));
    }
    schedule.sort_by_key(|&(f, _)| f);
    if schedule[0].0 != 0 {
        return Err("the first rate must start at frame 0".into());
    }
    Ok(schedule)
}

fn encode(mut args: Args) -> Result<(), String> {
    let qp = args.get("qp", 32u8)?;
    let max_frames = args.get("frames", usize::MAX)?;
    let recon_path = args.string("recon");
    let schedule = args
        .string("bitrate")
        .map(|s| parse_schedule(&s))
        .transpose()?;
    let max_delay = args.get::<f64>("max-delay", f64::NAN)?;
    let quality_cap = args.get("quality-cap", vdwt::ratectl::DEFAULT_QUALITY_CAP)?;
    let log_path = args.string("frame-log");
    let mut post = post_filter(&mut args)?;
    // The header carries 1/8 strength units; quantise a fixed band2 so the encoder's recon, the
    // header and `decode --postfilter auto` all use the same value.
    if let PostFilter::BandWide(a) = post {
        let q = (f64::from(a) / 32.0).round().clamp(0.0, 40.0) as u16;
        post = PostFilter::BandWide(q * 32);
    }
    let opts = CodecOptions::parse(&mut args)?;
    // A B-frame group's anchor carries most of its frames' bits.
    let max_delay = if max_delay.is_nan() {
        if opts.adaptive {
            1.5 * MAX_GROUP as f64
        } else {
            1.5
        }
    } else {
        max_delay
    };
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let (info, frames) = read_frames(input, max_frames)?;
    let frame_rate = fps(&info);

    let mut cfg = opts.config_for(&info, qp, &frames)?;
    cfg.postfilter = match post {
        PostFilter::Auto => postfilter::PostFilterMode::Auto,
        PostFilter::BandWide(a) => postfilter::PostFilterMode::Fixed(
            ((f64::from(a) / 32.0).round().clamp(0.0, 40.0)) as u8,
        ),
        _ => postfilter::PostFilterMode::Off,
    };
    if let Some(schedule) = &schedule {
        let mut rc = RateControl::new(schedule[0].1);
        rc.max_delay_frames = max_delay;
        rc.quality_cap = quality_cap;
        cfg.rate_control = Some(rc);
    }
    cfg.short_headers = cfg.rate_control.is_none();
    let mut encoder = Encoder::new(cfg).map_err(|e| e.to_string())?;
    if frames.len() > 1 {
        let cut = frame_difference(&frames[0], &frames[1]);
        if cut > 20.0 {
            // Frame 0 is never referenced when frame 1 starts a new scene. A hard cut makes
            // frame 0 a single displayed shot, like the final intra frame (A12): code it at qp
            // instead of spending a keyframe boost nothing predicts from.
            encoder.set_next_intra_offset(Some(if cut > 50.0 { 0 } else { -2 }));
        }
    }
    let mut stream = Vec::new();
    encoder.sequence_header().write(&mut stream);
    let mut display = Frame::new(info.width, info.height);
    let mut recon_out = match &recon_path {
        Some(p) => Some(Y4mWriter::create(Path::new(p), &info).map_err(|e| format!("{p}: {e}"))?),
        None => None,
    };
    let mut log = match &log_path {
        Some(p) => {
            let mut f = String::from(
                "frame,type,bytes,qp,psnr_y,pred_psnr_y,motion_bytes,filter,refined_nodes,intra_nodes,patches,inliers,\
                 gm0x,gm0y,gm1x,gm1y,gm2x,gm2y,gm3x,gm3y,\
                 link_kbps,target_bits,cap_bits,estimate_bits,queue_bits,delay_frames,dropped_bands,postfilter,\
                 header_bytes,layout_bytes,loop_bytes\n",
            );
            f.reserve(frames.len() * 120);
            Some((p.clone(), f))
        }
        None => None,
    };

    let (mut psnr_sum, mut comp_bytes, mut elapsed) = ([0.0f64; 4], [0usize; 3], Duration::ZERO);
    let mut times = StageTimes::default();
    let (mut link, mut link_sum, mut queue, mut worst_delay) = (0u64, 0.0f64, 0.0f64, 0.0f64);
    let (mut min_psnr, mut qp_sum, mut qp_min, mut qp_max) = (f64::INFINITY, 0u64, u8::MAX, 0u8);
    let (mut dropped_frames, mut dropped_bands, mut repeats, mut skips) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut p_count, mut b_count, mut motion_bytes) = (0usize, 0usize, 0usize);
    let mut next = 0;
    while next < frames.len() {
        // A pair of frames at once (ultraheavy), else one frame: in display order, each with its
        // reconstruction. A frame identical to the previous source is a Repeat: the picture has
        // not changed.
        let repeat = next > 0
            && same_frame(&frames[next], &frames[next - 1])
            && encoder.last_psnr()
                >= if frame_rate >= 50.0 {
                    REPEAT_MIN_PSNR_HIGH_FPS
                } else {
                    REPEAT_MIN_PSNR
                };
        let group = if repeat {
            1
        } else {
            group_size(opts.adaptive, schedule.is_none(), next, frames.len())
        };
        let step = next..next + group;
        next = step.end;
        if let Some(rate) = schedule
            .as_ref()
            .and_then(|s| s.iter().find(|&&(f, _)| step.contains(&f)))
            .map(|&(_, r)| r)
        {
            encoder.set_bitrate(rate).map_err(|e| e.to_string())?;
            link = rate;
        }
        let start = Instant::now();
        // Keep only the stats: the reconstruction is read before the next encode, so the
        // 1.4 MB clone per frame (and per repeat) disappears.
        let coded: Vec<(usize, vdwt::encoder::FrameStats)> = if repeat {
            let stats = encoder
                .encode_repeat(&mut stream)
                .map_err(|e| e.to_string())?;
            vec![(0, stats)]
        } else if opts.adaptive {
            let refs: Vec<&Frame> = frames[step.clone()].iter().collect();
            let stats = encoder
                .encode_pair(&refs, &mut stream)
                .map_err(|e| e.to_string())?;
            stats.into_iter().enumerate().collect()
        } else {
            let stats = encoder
                .encode_frame(&frames[step.start], &mut stream)
                .map_err(|e| e.to_string())?;
            vec![(0, stats)]
        };
        elapsed += start.elapsed();
        for (local, stats) in coded {
            let i = step.start + local;
            let frame = &frames[i];
            let stored: &Frame = if opts.adaptive && !repeat {
                encoder.group_recon(local)
            } else {
                encoder.recon()
            };
            let filtered = post.apply(stored, &mut display, stats.postfilter_strength);
            let recon: &Frame = if filtered { &display } else { stored };
            link_sum += link as f64;
            times += stats.times;
            let p = YuvPsnr::between(frame, recon);
            for (s, v) in psnr_sum.iter_mut().zip([p.y, p.u, p.v, p.weighted()]) {
                *s += v;
            }
            for (s, b) in comp_bytes.iter_mut().zip(stats.component_bytes) {
                *s += b;
            }
            min_psnr = min_psnr.min(p.y);
            qp_sum += u64::from(stats.qp);
            (qp_min, qp_max) = (qp_min.min(stats.qp), qp_max.max(stats.qp));
            dropped_bands += stats.dropped_bands;
            dropped_frames += usize::from(stats.dropped_bands > 0);
            repeats += usize::from(stats.repeated && !stats.skipped);
            skips += usize::from(stats.skipped);
            p_count += usize::from(stats.inter);
            b_count += usize::from(stats.bi);
            motion_bytes += stats.motion.map_or(0, |m| m.bytes);
            let mut rate_cols = String::from(",,,,,");
            if let Some(r) = stats.rate {
                // A sent frame finishes sending once the queue ahead of it and the frame itself
                // drain. Skipped frames send nothing.
                let delay = (queue + 8.0 * stats.bytes as f64) / (link as f64 / frame_rate);
                if stats.bytes > 0 {
                    worst_delay = worst_delay.max(delay);
                }
                queue = r.queue_bits;
                rate_cols = format!(
                    "{:.1},{:.0},{:.0},{:.0},{:.0},{delay:.3}",
                    link as f64 / 1000.0,
                    r.target_bits,
                    r.cap_bits,
                    r.estimated_bits,
                    r.queue_bits,
                );
            }
            if let Some((_, f)) = &mut log {
                let kind = match (stats.skipped, stats.repeated, stats.inter) {
                    (true, _, _) => "S",
                    (_, true, _) => "R",
                    (_, _, true) if stats.bi => "B",
                    (_, _, true) => "P",
                    _ => "I",
                };
                let m = stats.motion.unwrap_or_default();
                let corners: Vec<String> = m
                    .global
                    .corners
                    .iter()
                    .flat_map(|c| c.iter().map(|v| v.to_string()))
                    .collect();
                f.push_str(&format!(
                    "{i},{kind},{},{},{:.3},{},{},{},{},{},{},{},{},{rate_cols},{},{},{},{},{}\n",
                    stats.bytes,
                    stats.qp,
                    p.y,
                    if stats.inter {
                        format!("{:.3}", m.prediction_psnr)
                    } else {
                        String::new()
                    },
                    m.bytes,
                    if stats.inter {
                        format!("{}/{}", m.filter.name(), m.prediction.name())
                    } else {
                        String::new()
                    },
                    m.refined_nodes,
                    m.intra_nodes,
                    m.patches,
                    m.inlier_patches,
                    corners.join(","),
                    stats.dropped_bands,
                    stats.postfilter_strength,
                    stats.header_bytes,
                    stats.layout_bytes,
                    stats.loop_bytes,
                ));
            }
            // The recon file mirrors what a decoder outputs: skipped frames produce nothing.
            if let Some(w) = &mut recon_out
                && !stats.skipped
            {
                w.write_frame(recon).map_err(|e| e.to_string())?;
            }
        }
    }
    fs::write(output, &stream).map_err(|e| format!("{output}: {e}"))?;
    if let Some(w) = recon_out {
        w.finish().map_err(|e| e.to_string())?;
    }
    if let Some((path, text)) = log {
        fs::write(&path, text).map_err(|e| format!("{path}: {e}"))?;
    }

    let n = frames.len() as f64;
    let frame_bytes = stream.len() as f64 / n;
    let kbps = frame_bytes * 8.0 * frame_rate / 1000.0;
    let total: usize = comp_bytes.iter().sum::<usize>().max(1);
    let share = comp_bytes.map(|b| 100.0 * b as f64 / total as f64);
    let avg = psnr_sum.map(|s| s / n);
    let mode = match (&schedule, opts.lossless) {
        (Some(_), _) => "CBR".to_owned(),
        (None, true) => "lossless".to_owned(),
        (None, false) => format!("qp {qp}"),
    };
    println!(
        "{} frames {}x{} @ {:.2} fps, {mode}: {kbps:.1} kbps, {frame_bytes:.0} bytes/frame, {:.4} bpp",
        frames.len(),
        info.width,
        info.height,
        frame_rate,
        frame_bytes * 8.0 / (info.width * info.height) as f64
    );
    if schedule.is_some() {
        let target = link_sum / n / 1000.0;
        println!(
            "link   average {target:.1} kbps, delivered {:.1}%, worst queueing delay {worst_delay:.2} frames (limit {max_delay})",
            100.0 * kbps / target
        );
        println!(
            "qp     {qp_min}..{qp_max}, mean {:.1};  bands dropped {dropped_bands} in {dropped_frames} frames;  repeat frames {repeats};  skipped {skips}",
            qp_sum as f64 / n
        );
    }
    if p_count > 0 {
        let bi = if b_count > 0 {
            format!(" ({b_count} of them B)")
        } else {
            String::new()
        };
        println!(
            "frames {} intra, {p_count} P{bi};  motion {:.1}% of the stream ({:.0} bytes per P-frame)",
            frames.len() - p_count - repeats - skips,
            100.0 * motion_bytes as f64 / stream.len() as f64,
            motion_bytes as f64 / p_count as f64
        );
    }
    println!(
        "PSNR   Y {:.2} (min {min_psnr:.2})  U {:.2}  V {:.2}  YUV {:.2} dB",
        avg[0], avg[1], avg[2], avg[3]
    );
    println!(
        "bytes  Y {:.0}%  U {:.0}%  V {:.0}%   encode {:.2} ms/frame ({}, {} thread{})",
        share[0],
        share[1],
        share[2],
        ms(elapsed) / n,
        opts.isa,
        opts.threads,
        if opts.threads == 1 { "" } else { "s" }
    );
    println!("{}", stage_line(&times, frames.len()));
    Ok(())
}

/// Mean time per frame of every stage that took any.
fn stage_line(times: &StageTimes, frames: usize) -> String {
    let parts: Vec<String> = Stage::ALL
        .iter()
        .filter(|&&s| !times.get(s).is_zero())
        .map(|&s| {
            format!(
                "{} {:.2}",
                s.name(),
                ms(times.get(s)) / frames.max(1) as f64
            )
        })
        .collect();
    format!("stages ms/frame: {}", parts.join(", "))
}

/// Frames per step of the ultraheavy tier: a pair (P-frames or a B-frame group).
const MAX_GROUP: usize = 2;
/// Frames per step of the ultraheavy tier at a fixed qp: a group of 4 on a still camera, else
/// two pairs ([`Encoder::encode_pair`]). Rate-controlled streams keep pairs for their delay.
const STILL_GROUP: usize = 4;
/// Frames the ultraheavy tier probes its motion grid on.
const TRIAL_FRAMES: usize = 3;
/// The ultraheavy tier's intra qp offset and motion λ (§10.4).
const ULTRA_INTRA_QP_OFFSET: i8 = -7;
const ULTRA_MOTION_LAMBDA: f64 = 0.8;

/// An identical source frame is repeated instead of coded once the picture is at least this
/// good: below it, a P-frame is worth its bits for the refinement.
const REPEAT_MIN_PSNR: f64 = 45.0;
/// At 50 fps and above an identical frame usually comes from a lower-rate master duplicated on
/// ingest; repeating it at this floor frees bits for the unique frames (0.3 dB at 600 kbit/s
/// 60 fps) without touching true 60 fps video, which has no identical frames.
const REPEAT_MIN_PSNR_HIGH_FPS: f64 = 35.0;

/// Frames the next step codes at once: with adaptive structures [`STILL_GROUP`] at a fixed qp
/// (when that many are left) or up to [`MAX_GROUP`] after the first frame, else one.
fn group_size(adaptive: bool, fixed_qp: bool, next: usize, total: usize) -> usize {
    if adaptive && next > 0 {
        let left = total - next;
        if fixed_qp && left >= STILL_GROUP {
            STILL_GROUP
        } else {
            MAX_GROUP.min(left)
        }
    } else {
        1
    }
}

/// Encoder tier names, in effort order (the last one is the heavy search with B-frames).
const TIERS: [&str; 4] = ["light", "medium", "heavy", "ultraheavy"];

/// The motion search effort from `--tier NAME` or its alias `--effort N`.
fn effort(args: &mut Args) -> Result<u8, String> {
    match (args.string("tier"), args.string("effort")) {
        (Some(_), Some(_)) => Err("--tier and --effort name the same setting: give one".into()),
        (Some(name), None) => TIERS
            .iter()
            .position(|t| t.eq_ignore_ascii_case(&name))
            .map(|e| e as u8)
            .ok_or_else(|| format!("unknown tier `{name}` ({})", TIERS.join(", "))),
        (None, Some(n)) => n
            .parse::<u8>()
            .ok()
            .filter(|&e| usize::from(e) < TIERS.len())
            .ok_or_else(|| format!("invalid value `{n}` for --effort (0, 1, 2, 3)")),
        (None, None) => Ok(DEFAULT_EFFORT),
    }
}

/// `--threads`: by default as many as the target has cores (4), or fewer if this machine has
/// fewer.
fn threads(args: &mut Args) -> Result<usize, String> {
    let default = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
    let n = args.get("threads", default)?;
    if (1..=MAX_THREADS).contains(&n) {
        Ok(n)
    } else {
        Err(format!("--threads must be 1..={MAX_THREADS}"))
    }
}

#[derive(Clone, Copy)]
enum PostFilter {
    None,
    /// Use the strength carried in the frame header (0 disables); the encoder's per-scene mode.
    Auto,
    Unsharp(u16),
    Band(u16),
    BandWide(u16),
}

impl PostFilter {
    /// Applies the filter and returns whether `dst` now holds a filtered picture. `signalled` is
    /// the header's strength in 1/8 units.
    fn apply(self, src: &Frame, dst: &mut Frame, signalled: u8) -> bool {
        match self {
            Self::None => false,
            Self::Auto => {
                if signalled == 0 {
                    false
                } else {
                    postfilter::band_wide(src, dst, u16::from(signalled) * 32);
                    true
                }
            }
            Self::Unsharp(a) => {
                postfilter::sharpen(src, dst, a);
                true
            }
            Self::Band(a) => {
                postfilter::band(src, dst, a);
                true
            }
            Self::BandWide(a) => {
                postfilter::band_wide(src, dst, a);
                true
            }
        }
    }
}

/// `--postfilter MODE`, `--postfilter-strength S`, with `--sharpen A` kept as the unsharp alias.
fn post_filter(args: &mut Args) -> Result<PostFilter, String> {
    let legacy = (args.get("sharpen", 0.0f64)? * 256.0)
        .round()
        .clamp(0.0, 2048.0) as u16;
    let mode = args.string("postfilter").unwrap_or_else(|| {
        if legacy > 0 {
            "unsharp".into()
        } else {
            "none".into()
        }
    });
    let default = match mode.as_str() {
        "unsharp" if legacy > 0 => legacy as f64 / 256.0,
        "unsharp" => 0.6,
        "band" | "band2" => 2.0,
        "auto" => 0.0,
        _ => 0.0,
    };
    let strength = args.get("postfilter-strength", default)?;
    if !(0.0..=8.0).contains(&strength) {
        return Err("--postfilter-strength must be in 0..=8".into());
    }
    let a = (strength * 256.0).round().clamp(0.0, 2048.0) as u16;
    match mode.as_str() {
        "none" => Ok(PostFilter::None),
        "auto" => Ok(PostFilter::Auto),
        "unsharp" => Ok(PostFilter::Unsharp(a)),
        "band" => Ok(PostFilter::Band(a)),
        "band2" => Ok(PostFilter::BandWide(a)),
        other => Err(format!(
            "unknown --postfilter `{other}` (none, auto, unsharp, band, band2)"
        )),
    }
}

fn decode(mut args: Args) -> Result<(), String> {
    let threads = threads(&mut args)?;
    let post = post_filter(&mut args)?;
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let data = fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let (seq, payloads) = bitstream::split_stream(&data).map_err(|e| format!("{input}: {e}"))?;
    let info = Y4mInfo {
        width: seq.width,
        height: seq.height,
        fps_num: seq.fps_num,
        fps_den: seq.fps_den,
    };
    let mut decoder =
        Decoder::with_threads(seq, Isa::best(), threads).map_err(|e| e.to_string())?;
    let mut writer =
        Y4mWriter::create(Path::new(output), &info).map_err(|e| format!("{output}: {e}"))?;
    let mut elapsed = Duration::ZERO;
    let mut times = StageTimes::default();
    // Frames come in coding order (B-frame groups anchor first): shown by frame number. A
    // number never sent (a skipped frame) is passed over once a group's worth is waiting.
    let mut waiting: std::collections::BTreeMap<u16, Frame> = Default::default();
    let mut want: Option<u16> = None;
    // Frames written, whose buffers hold the next ones waiting (a copy per frame, allocated anew,
    // made the allocator fault pages in every frame).
    let mut spare: Vec<Frame> = Vec::new();
    let mut display = Frame::new(info.width, info.height);
    let flush = |writer: &mut Y4mWriter,
                 waiting: &mut std::collections::BTreeMap<u16, Frame>,
                 want: &mut Option<u16>,
                 spare: &mut Vec<Frame>,
                 all: bool|
     -> Result<(), String> {
        loop {
            let Some(w) = *want else { return Ok(()) };
            let key = match waiting.contains_key(&w) {
                true => w,
                false if all || waiting.len() > 4 => {
                    // The nearest number after the wanted one, wrapping.
                    match waiting.keys().min_by_key(|&&k| k.wrapping_sub(w)) {
                        Some(&k) => k,
                        None => return Ok(()),
                    }
                }
                false => return Ok(()),
            };
            let f = waiting.remove(&key).expect("present");
            writer.write_frame(&f).map_err(|e| e.to_string())?;
            spare.push(f);
            *want = Some(key.wrapping_add(1));
        }
    };
    let mut header_state: Option<bitstream::FrameHeaderState> = None;
    for (i, payload) in payloads.iter().enumerate() {
        let header = bitstream::FrameHeader::parse_with(
            &mut bitstream::ByteReader::new(payload),
            header_state.as_ref(),
        )
        .ok();
        if let Some(h) = &header {
            if h.frame_type != bitstream::FrameType::Repeat {
                header_state = Some(bitstream::FrameHeaderState {
                    qp: h.qp,
                    recon_offset: h.recon_offset,
                    postfilter_strength: h.postfilter_strength,
                });
            }
        }
        let num = header.map_or(0, |h| h.frame_num);
        let signalled = header.map_or(0, |h| h.postfilter_strength);
        let start = Instant::now();
        let frame = decoder
            .decode_frame(payload)
            .map_err(|e| format!("frame {i}: {e}"))?;
        elapsed += start.elapsed();
        want.get_or_insert(num);
        let filtered = post.apply(frame, &mut display, signalled);
        let shown: &Frame = if filtered { &display } else { frame };
        if waiting.is_empty() && want == Some(num) {
            // In order (no B-frames waiting): straight out.
            writer.write_frame(shown).map_err(|e| e.to_string())?;
            want = Some(num.wrapping_add(1));
        } else {
            let mut f = spare.pop().unwrap_or_else(|| shown.clone());
            for (d, s) in f.planes_mut().into_iter().zip(shown.planes()) {
                for y in 0..s.height() {
                    d.row_mut(y).copy_from_slice(s.row(y));
                }
            }
            waiting.insert(num, f);
            flush(&mut writer, &mut waiting, &mut want, &mut spare, false)?;
        }
        times += *decoder.times();
    }
    flush(&mut writer, &mut waiting, &mut want, &mut spare, true)?;
    writer.finish().map_err(|e| e.to_string())?;
    let summary = format!(
        "decoded {} frames {}x{}: {:.2} ms/frame ({threads} thread{})\n{}",
        payloads.len(),
        info.width,
        info.height,
        ms(elapsed) / payloads.len().max(1) as f64,
        if threads == 1 { "" } else { "s" },
        stage_line(&times, payloads.len())
    );
    // With the pictures on standard output, the report goes to standard error.
    if output == "-" {
        eprintln!("{summary}");
    } else {
        println!("{summary}");
    }
    Ok(())
}

fn train_contexts(mut args: Args) -> Result<(), String> {
    let qps: Vec<u8> = args
        .string("qp")
        .unwrap_or_else(|| "24,32,40,48,56".into())
        .split(',')
        .map(|q| q.trim().parse().map_err(|_| format!("invalid qp `{q}`")))
        .collect::<Result<_, _>>()?;
    let max_frames = args.get("frames", 60usize)?;
    let min_symbols = args.get("min-symbols", 64u64)?;
    let opts = CodecOptions::parse(&mut args)?;
    if args.positional.len() < 2 {
        return Err("expected OUT.rs and at least one clip".into());
    }
    let (output, clips) = (args.positional[0].clone(), args.positional[1..].to_vec());
    args.positional.truncate(0);
    args.finish(0)?;
    let mut total = vdwt::training::ContextStats::default();
    for clip in &clips {
        let (info, frames) = read_frames(clip, max_frames)?;
        for &qp in &qps {
            // A P-frame run, and the first frames intra-only for the intra classes.
            for intra_only in [false, true] {
                let mut cfg = opts.config(&info, qp);
                cfg.p_frames = !intra_only;
                let n = if intra_only {
                    frames.len().min(8)
                } else {
                    frames.len()
                };
                let mut encoder = Encoder::new(cfg).map_err(|e| e.to_string())?;
                encoder.record_contexts();
                let mut sink = Vec::new();
                for frame in &frames[..n] {
                    sink.clear();
                    encoder
                        .encode_frame(frame, &mut sink)
                        .map_err(|e| e.to_string())?;
                }
                total.merge(&encoder.take_context_stats().expect("recording"));
            }
        }
        println!("{clip}: {} symbols so far", total.symbols());
    }
    fs::write(&output, total.to_rust(min_symbols)).map_err(|e| format!("{output}: {e}"))?;
    println!("wrote {output}");
    Ok(())
}

fn rd(mut args: Args) -> Result<(), String> {
    let qps: Vec<u8> = args
        .string("qp")
        .unwrap_or_else(|| "16,24,32,40,48".into())
        .split(',')
        .map(|q| q.trim().parse().map_err(|_| format!("invalid qp `{q}`")))
        .collect::<Result<_, _>>()?;
    let max_frames = args.get("frames", 10usize)?;
    let opts = CodecOptions::parse(&mut args)?;
    args.finish(1)?;
    let (info, frames) = read_frames(&args.positional[0], max_frames)?;
    let n = frames.len() as f64;
    let pixels = (info.width * info.height) as f64;
    println!(
        "{} frames {}x{} @ {:.2} fps",
        frames.len(),
        info.width,
        info.height,
        fps(&info)
    );
    println!("  qp      kbps     bpp   PSNR-Y  PSNR-U  PSNR-V  PSNR-YUV   enc ms  dec ms");
    for qp in qps {
        let mut encoder =
            Encoder::new(opts.config_for(&info, qp, &frames)?).map_err(|e| e.to_string())?;
        let mut decoder =
            Decoder::with_threads(encoder.sequence_header().clone(), opts.isa, opts.threads)
                .map_err(|e| e.to_string())?;
        let mut header_state: Option<bitstream::FrameHeaderState> = None;
        let (mut bytes, mut psnr_sum, mut enc_t, mut dec_t) =
            (0usize, [0.0f64; 4], Duration::ZERO, Duration::ZERO);
        let mut next = 0;
        while next < frames.len() {
            // Ultraheavy: a pair of frames at once, in coding order.
            let group = group_size(opts.adaptive, true, next, frames.len());
            let mut packet = Vec::new();
            let start = Instant::now();
            if opts.adaptive {
                let refs: Vec<&Frame> = frames[next..next + group].iter().collect();
                encoder
                    .encode_pair(&refs, &mut packet)
                    .map_err(|e| e.to_string())?;
            } else {
                encoder
                    .encode_frame(&frames[next], &mut packet)
                    .map_err(|e| e.to_string())?;
            }
            enc_t += start.elapsed();
            bytes += packet.len();
            let mut r = bitstream::ByteReader::new(&packet);
            for _ in 0..group {
                let payload = r.chunk().map_err(|e| e.to_string())?;
                // Coding order: the frame number says which frame of the group it is.
                let header = bitstream::FrameHeader::parse_with(
                    &mut bitstream::ByteReader::new(payload),
                    header_state.as_ref(),
                )
                .map_err(|e| e.to_string())?;
                if header.frame_type != bitstream::FrameType::Repeat {
                    header_state = Some(bitstream::FrameHeaderState {
                        qp: header.qp,
                        recon_offset: header.recon_offset,
                        postfilter_strength: header.postfilter_strength,
                    });
                }
                let num = header.frame_num;
                let i = next + usize::from(num.wrapping_sub(next as u16));
                let start = Instant::now();
                let decoded = decoder
                    .decode_frame(payload)
                    .map_err(|e| format!("frame {i}: {e}"))?;
                dec_t += start.elapsed();
                let expected = if opts.adaptive {
                    encoder.group_recon(i - next)
                } else {
                    encoder.recon()
                };
                if !decoded
                    .planes()
                    .iter()
                    .zip(expected.planes())
                    .all(|(a, b)| a.rows().eq(b.rows()))
                {
                    return Err(format!(
                        "qp {qp}, frame {i}: decoder output differs from the encoder's reconstruction"
                    ));
                }
                let p = YuvPsnr::between(&frames[i], decoded);
                for (s, v) in psnr_sum.iter_mut().zip([p.y, p.u, p.v, p.weighted()]) {
                    *s += v;
                }
            }
            next += group;
        }
        let avg = psnr_sum.map(|s| s / n);
        let frame_bits = bytes as f64 * 8.0 / n;
        println!(
            "{:>4} {:>9.1} {:>7.4} {:>8.2} {:>7.2} {:>7.2} {:>9.2} {:>8.2} {:>7.2}",
            qp,
            frame_bits * fps(&info) / 1000.0,
            frame_bits / pixels,
            avg[0],
            avg[1],
            avg[2],
            avg[3],
            ms(enc_t) / n,
            ms(dec_t) / n
        );
        if opts.lossless {
            break;
        }
    }
    Ok(())
}

fn pred_dump(mut args: Args) -> Result<(), String> {
    let qp = args.get("qp", 32u8)?;
    let max_frames = args.get("frames", usize::MAX)?;
    let opts = CodecOptions::parse(&mut args)?;
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let (info, frames) = read_frames(input, max_frames)?;
    let cfg = opts.config_for(&info, qp, &frames)?;
    let mut encoder = Encoder::new(cfg).map_err(|e| e.to_string())?;
    let mut writer =
        Y4mWriter::create(Path::new(output), &info).map_err(|e| format!("{output}: {e}"))?;
    let mut sink = Vec::new();
    let mut flat = Frame::new(info.width, info.height);
    flat.y.fill(128);
    flat.u.fill(128);
    flat.v.fill(128);
    let mut inter = 0;
    for frame in &frames {
        sink.clear();
        let stats = encoder
            .encode_frame(frame, &mut sink)
            .map_err(|e| e.to_string())?;
        let shown = if stats.inter {
            inter += 1;
            encoder.prediction()
        } else {
            &flat
        };
        writer.write_frame(shown).map_err(|e| e.to_string())?;
    }
    println!(
        "wrote {output}: {} frames ({} motion-compensated predictions, rest mid-grey), qp {qp}",
        frames.len(),
        inter
    );
    Ok(())
}

fn band_stats(mut args: Args) -> Result<(), String> {
    let qps: Vec<u8> = args
        .string("qp")
        .unwrap_or_else(|| "16,24,32,40,48,56,64".into())
        .split(',')
        .map(|q| q.trim().parse().map_err(|_| format!("invalid qp `{q}`")))
        .collect::<Result<_, _>>()?;
    let max_frames = args.get("frames", 10usize)?;
    let opts = CodecOptions::parse(&mut args)?;
    args.finish(1)?;
    let (info, frames) = read_frames(&args.positional[0], max_frames)?;
    println!(
        "frame,type,qp,comp,level,orient,coefficients,nonzero,magnitude_bits,bytes,est_nonzero,est_bits"
    );
    for qp in qps {
        let mut cfg = opts.config(&info, qp);
        // Rate control pinned to one qp, with a cap that never binds: estimates without decisions.
        cfg.rate_control = Some(RateControl {
            bitrate: u64::MAX / 4,
            max_delay_frames: 60.0,
            min_qp: qp,
            max_qp: qp,
            quality_cap: 0.0,
        });
        let mut encoder = Encoder::new(cfg).map_err(|e| e.to_string())?;
        for (i, frame) in frames.iter().enumerate() {
            let stats = encoder
                .encode_frame(frame, &mut Vec::new())
                .map_err(|e| e.to_string())?;
            let kind = if stats.inter { "P" } else { "I" };
            for b in encoder.band_stats() {
                let (en, eb) = b.estimate.unwrap_or((f64::NAN, f64::NAN));
                println!(
                    "{i},{kind},{qp},{},{},{:?},{},{},{},{},{en:.1},{eb:.1}",
                    b.comp, b.level, b.orient, b.coefficients, b.nonzero, b.magnitude_bits, b.bytes
                );
            }
        }
    }
    Ok(())
}
