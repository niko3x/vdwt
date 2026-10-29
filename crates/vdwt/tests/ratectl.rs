//! Rate control end to end: delay limit, rate tracking, band dropping, repeat frames, and
//! decoder agreement through all of them.

use vdwt::bitstream::{self, ByteReader, FrameHeader, FrameType};
use vdwt::rng::SplitMix64;
use vdwt::{Decoder, Encoder, EncoderConfig, Frame, RateControl};

const W: usize = 320;
const H: usize = 180;
const FPS: f64 = 60.0;

/// A scene: gradients, a moving edge pattern and noise. Changing `scene` changes the content
/// abruptly, like a cut.
fn frame(scene: u64, t: usize) -> Frame {
    let mut rng = SplitMix64::new(scene * 1000 + t as u64);
    let detail = 4 + (scene % 4) as i32 * 10;
    let mut f = Frame::new(W, H);
    for (c, plane) in f.planes_mut().into_iter().enumerate() {
        let (w, h) = (plane.width(), plane.height());
        for y in 0..h {
            for (x, v) in plane.row_mut(y).iter_mut().enumerate() {
                let gradient =
                    ((x + t) * 150 / w + y * 50 / h + 30 * c + scene as usize * 17) as i32;
                let edge = if ((x + 2 * t) / (9 + scene as usize) + y / 13).is_multiple_of(2) {
                    detail
                } else {
                    -detail
                };
                *v = (gradient + edge + rng.range_i32(-detail / 2, detail / 2)).clamp(0, 255) as u8;
            }
        }
    }
    f
}

struct Run {
    /// Per frame: bytes written, link rate in force, queue after, repeated, dropped bands.
    frames: Vec<(usize, u64, f64, bool, usize)>,
    stream: Vec<u8>,
    /// The reconstruction of every frame actually sent (skipped frames send nothing).
    recons: Vec<Frame>,
    skipped: usize,
}

fn run(rc: RateControl, schedule: &[(usize, u64)], scenes: &[(usize, u64)], n: usize) -> Run {
    let mut cfg = EncoderConfig::new(W, H);
    cfg.rate_control = Some(rc);
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let (mut frames, mut recons, mut skipped) = (Vec::new(), Vec::new(), 0);
    let mut link = schedule[0].1;
    for t in 0..n {
        if let Some(&(_, r)) = schedule.iter().find(|&&(f, _)| f == t) {
            enc.set_bitrate(r).unwrap();
            link = r;
        }
        let scene = scenes
            .iter()
            .rev()
            .find(|&&(f, _)| f <= t)
            .map_or(0, |&(_, s)| s);
        let stats = enc.encode_frame(&frame(scene, t), &mut stream).unwrap();
        let rate = stats.rate.expect("rate control reports its state");
        frames.push((
            stats.bytes,
            link,
            rate.queue_bits,
            stats.repeated,
            stats.dropped_bands,
        ));
        if stats.skipped {
            assert_eq!(stats.bytes, 0);
            skipped += 1;
        } else {
            recons.push(enc.recon().clone());
        }
    }
    Run {
        frames,
        stream,
        recons,
        skipped,
    }
}

/// Worst queueing delay in frame durations over the frames actually sent, except the first
/// (which has nothing to repeat, so it is always sent whole). Skipped frames have no delay: after
/// a sudden link drop the encoder sends nothing until the backlog, now draining at the lower rate,
/// fits the limit again.
fn worst_delay(run: &Run) -> f64 {
    let mut queue = 0.0;
    let mut worst: f64 = 0.0;
    for (i, &(bytes, link, after, _, _)) in run.frames.iter().enumerate() {
        if i > 0 && bytes > 0 {
            worst = worst.max((queue + 8.0 * bytes as f64) / (link as f64 / FPS));
        }
        queue = after;
    }
    worst
}

fn same(a: &Frame, b: &Frame) -> bool {
    a.planes()
        .iter()
        .zip(b.planes())
        .all(|(p, q)| p.rows().eq(q.rows()))
}

fn assert_decodes_to_recon(run: &Run) {
    let (seq, payloads) = bitstream::split_stream(&run.stream).unwrap();
    let mut dec = Decoder::new(seq).unwrap();
    assert_eq!(payloads.len(), run.recons.len());
    for (i, (p, recon)) in payloads.iter().zip(&run.recons).enumerate() {
        assert!(same(dec.decode_frame(p).unwrap(), recon), "frame {i}");
    }
}

#[test]
fn follows_a_changing_link_within_the_delay_limit() {
    let schedule = [
        (0, 800_000),
        (40, 250_000),
        (80, 2_000_000),
        (120, 60_000),
        (150, 600_000),
    ];
    let scenes = [(0, 1), (55, 2), (100, 3), (135, 0)];
    let rc = RateControl::new(800_000);
    let limit = rc.max_delay_frames;
    let run = run(rc, &schedule, &scenes, 180);
    let delay = worst_delay(&run);
    assert!(
        delay <= limit + 1e-9,
        "worst delay {delay:.3} frames, limit {limit}"
    );
    // Steady stretches use most of the link: average over each segment's last 25 frames.
    for w in schedule.windows(2) {
        let (start, end, link) = (w[0].0, w[1].0, w[0].1);
        let seg = &run.frames[end - 25..end];
        let rate = seg.iter().map(|f| 8.0 * f.0 as f64).sum::<f64>() / seg.len() as f64 * FPS;
        let used = rate / link as f64;
        assert!(
            (0.85..=1.05).contains(&used),
            "frames {start}..{end} at {link}: used {:.1}%",
            100.0 * used
        );
    }
    assert_decodes_to_recon(&run);
}

#[test]
fn tight_cap_drops_bands_and_repeats_but_stays_decodable() {
    // A qp pinned far too fine for the link: every frame must shed bands to fit. At 40 kbit/s
    // only Repeat frames fit, and at 3 kbit/s not even those.
    let mut rc = RateControl::new(400_000);
    rc.min_qp = 8;
    rc.max_qp = 8;
    let schedule = [(0, 400_000), (10, 40_000), (20, 3_000), (30, 400_000)];
    let run = run(rc.clone(), &schedule, &[(0, 1)], 40);
    let dropped: usize = run.frames.iter().map(|f| f.4).sum();
    let repeats = run.frames.iter().filter(|f| f.3 && f.0 > 0).count();
    assert!(dropped > 0, "no bands were dropped");
    assert!(repeats > 0, "no repeat frames were sent");
    // 3 kbit/s carries 50 bits per frame: less than even a Repeat frame, so some are skipped.
    assert!(run.skipped > 0, "no frames were skipped");
    let delay = worst_delay(&run);
    assert!(
        delay <= rc.max_delay_frames + 1e-9,
        "worst delay {delay:.3} frames"
    );
    assert_decodes_to_recon(&run);

    // Repeat frames are just a header.
    let (_, payloads) = bitstream::split_stream(&run.stream).unwrap();
    let sent: Vec<_> = run.frames.iter().filter(|f| f.0 > 0).collect();
    let repeat = payloads.iter().zip(sent).find(|(_, f)| f.3).unwrap().0;
    let header = FrameHeader::parse(&mut ByteReader::new(repeat)).unwrap();
    assert_eq!(header.frame_type, FrameType::Repeat);
}

/// A smooth texture under a panning camera: content P-frames predict well.
fn panning(t: usize) -> Frame {
    let mut rng = SplitMix64::new(t as u64);
    let mut f = Frame::new(W, H);
    let (ox, oy) = (1.3 * t as f64, -0.4 * t as f64);
    for (c, plane) in f.planes_mut().into_iter().enumerate() {
        let s = if c == 0 { 1.0 } else { 2.0 };
        for y in 0..plane.height() {
            for (x, v) in plane.row_mut(y).iter_mut().enumerate() {
                let (u, w) = (x as f64 * s + ox, y as f64 * s + oy);
                let tex = 128.0
                    + 40.0 * (0.07 * u + 0.03 * w).sin()
                    + 25.0 * (0.21 * u - 0.17 * w + c as f64).sin()
                    + 12.0 * (0.53 * u + 0.41 * w).cos();
                *v = (tex + f64::from(rng.range_i32(-2, 2))).clamp(0.0, 255.0) as u8;
            }
        }
    }
    f
}

/// Per-frame outcome of a P-frame run: (inter, bands coded, bands dropped, repeated, skipped).
type Outcome = (bool, usize, usize, bool, bool);

/// Encodes the panning scene over a link schedule and checks the delay limit and decoding.
fn run_panning(rc: RateControl, schedule: &[(usize, u64)], n: usize) -> Vec<Outcome> {
    let limit = rc.max_delay_frames;
    let mut cfg = EncoderConfig::new(W, H);
    cfg.rate_control = Some(rc);
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let (mut recons, mut outcomes) = (Vec::new(), Vec::new());
    let (mut queue, mut worst, mut link) = (0.0f64, 0.0f64, schedule[0].1);
    for t in 0..n {
        if let Some(&(_, r)) = schedule.iter().find(|&&(f, _)| f == t) {
            enc.set_bitrate(r).unwrap();
            link = r;
        }
        let stats = enc.encode_frame(&panning(t), &mut stream).unwrap();
        if t > 0 && stats.bytes > 0 {
            worst = worst.max((queue + 8.0 * stats.bytes as f64) / (link as f64 / FPS));
        }
        queue = stats.rate.unwrap().queue_bits;
        let coded = enc.band_stats().iter().filter(|b| b.bytes > 0).count();
        outcomes.push((
            stats.inter,
            coded,
            stats.dropped_bands,
            stats.repeated,
            stats.skipped,
        ));
        if !stats.skipped {
            recons.push(enc.recon().clone());
        }
    }
    assert!(worst <= limit + 1e-9, "worst delay {worst:.3} frames");
    assert_decodes_to_recon(&Run {
        frames: Vec::new(),
        stream,
        recons,
        skipped: 0,
    });
    outcomes
}

#[test]
fn p_frames_keep_the_picture_moving_on_a_starved_link() {
    // 10 kbit/s gives a frame 167 bits: room for the motion and little else. qp rises until the
    // residual all but vanishes, and every frame is still a P-frame that moves the picture: no
    // Repeat frames, which would freeze it. The step through 60 kbit/s keeps the backlog from
    // the fast link small.
    let schedule = [(0, 600_000), (20, 60_000), (30, 10_000), (55, 600_000)];
    let out = run_panning(RateControl::new(600_000), &schedule, 70);
    let starved = &out[32..55];
    assert!(starved.iter().all(|o| o.0 && !o.3 && !o.4), "{starved:?}");
    assert!(
        starved.iter().any(|o| o.1 == 0),
        "no motion-only P-frames: {starved:?}"
    );
}

#[test]
fn p_frames_shed_every_band_when_qp_is_pinned() {
    // A qp pinned far too fine for the link: P-frames must drop bands, the LL bands included,
    // down to their motion alone.
    let mut rc = RateControl::new(600_000);
    rc.min_qp = 16;
    rc.max_qp = 16;
    let schedule = [(0, 600_000), (10, 60_000), (20, 12_000), (40, 600_000)];
    let out = run_panning(rc, &schedule, 50);
    assert!(
        out.iter().any(|o| o.0 && o.1 == 0 && o.2 > 0),
        "no P-frame shed all its bands: {out:?}"
    );
}

#[test]
fn repeat_needs_a_previous_picture() {
    let cfg = EncoderConfig::new(W, H);
    let enc = Encoder::new(cfg).unwrap();
    let mut dec = Decoder::new(enc.sequence_header().clone()).unwrap();
    let mut payload = Vec::new();
    FrameHeader {
        frame_type: FrameType::Repeat,
        frame_num: 0,
        qp: 0,
        recon_offset: 0,
        parent_contexts: false,
        ll_prediction: false,
        filter: Default::default(),
        prediction: Default::default(),
        loop_filter: false,
        keep_reference: false,
        postfilter_strength: 0,
    }
    .write(&mut payload);
    assert!(dec.decode_frame(&payload).is_err());
}

#[test]
fn bitrate_changes_need_rate_control() {
    let mut enc = Encoder::new(EncoderConfig::new(W, H)).unwrap();
    assert!(enc.set_bitrate(1_000_000).is_err());
    let mut cfg = EncoderConfig::new(W, H);
    cfg.rate_control = Some(RateControl::new(1_000_000));
    let mut enc = Encoder::new(cfg).unwrap();
    assert!(enc.set_bitrate(0).is_err());
    assert!(enc.set_bitrate(500_000).is_ok());
}
