//! P-frames end to end: motion compensation, the reconstruction loop and decoder agreement over
//! long runs, with every filter and grid spacing.

use vdwt::bitstream::{self, FrameHeader, FrameType};
use vdwt::motion::{Filter, Prediction};
use vdwt::postfilter::PostFilterMode;
use vdwt::rng::SplitMix64;
use vdwt::{Decoder, Encoder, EncoderConfig, Frame, Isa};

/// A textured scene seen by a drifting, rotating, zooming camera, with a small object moving on
/// its own and a little sensor noise. `scene` changes the content entirely (a cut).
fn frame(w: usize, h: usize, scene: u64, t: usize) -> Frame {
    let mut rng = SplitMix64::new(scene * 7919 + t as u64);
    let waves: Vec<(f64, f64, f64, f64)> = {
        let mut r = SplitMix64::new(scene);
        (0..16)
            .map(|i| {
                let f = 0.03 + 0.3 * (f64::from(i) / 16.0).powi(2);
                let a = r.below(6283) as f64 / 1000.0;
                (
                    f * a.cos(),
                    f * a.sin(),
                    r.below(6283) as f64 / 1000.0,
                    28.0 / (1.0 + 3.0 * f),
                )
            })
            .collect()
    };
    let t = t as f64;
    let (angle, zoom) = (0.002 * t, 1.0 + 0.003 * t);
    let (cx, cy) = (w as f64 / 2.0 + 1.7 * t, h as f64 / 2.0 - 0.6 * t);
    let (ox, oy) = (w as f64 * 0.3 + 2.5 * t, h as f64 * 0.4 + 0.8 * t);
    let mut f = Frame::new(w, h);
    for (c, plane) in f.planes_mut().into_iter().enumerate() {
        let s = if c == 0 { 1.0 } else { 2.0 };
        for y in 0..plane.height() {
            for (x, v) in plane.row_mut(y).iter_mut().enumerate() {
                let (px, py) = (x as f64 * s - w as f64 / 2.0, y as f64 * s - h as f64 / 2.0);
                let (u, vv) = (
                    cx + zoom * (angle.cos() * px - angle.sin() * py),
                    cy + zoom * (angle.sin() * px + angle.cos() * py),
                );
                let inside_object =
                    (x as f64 * s - ox).abs() < 12.0 && (y as f64 * s - oy).abs() < 8.0;
                let base = if inside_object {
                    200.0 - 3.0 * (x as f64 * s - ox)
                } else {
                    128.0
                        + waves
                            .iter()
                            .map(|&(fx, fy, ph, amp)| {
                                amp * (fx * u + fy * vv + ph + c as f64).sin()
                            })
                            .sum::<f64>()
                };
                let noise = rng.range_i32(-2, 2) as f64;
                *v = (base + noise).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    f
}

fn clip(w: usize, h: usize, n: usize, cut_at: Option<usize>) -> Vec<Frame> {
    (0..n)
        .map(|t| {
            let scene = u64::from(cut_at.is_some_and(|c| t >= c));
            frame(w, h, scene, t)
        })
        .collect()
}

struct Encoded {
    stream: Vec<u8>,
    recons: Vec<Frame>,
    inter: Vec<bool>,
}

fn encode(cfg: EncoderConfig, frames: &[Frame]) -> Encoded {
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let (mut recons, mut inter) = (Vec::new(), Vec::new());
    for f in frames {
        let stats = enc.encode_frame(f, &mut stream).unwrap();
        recons.push(enc.recon().clone());
        inter.push(stats.inter);
    }
    Encoded {
        stream,
        recons,
        inter,
    }
}

fn same(a: &Frame, b: &Frame) -> bool {
    a.planes()
        .iter()
        .zip(b.planes())
        .all(|(p, q)| p.rows().eq(q.rows()))
}

fn assert_decodes(e: &Encoded, what: &str) {
    let (seq, payloads) = bitstream::split_stream(&e.stream).unwrap();
    let mut dec = Decoder::new(seq).unwrap();
    assert_eq!(payloads.len(), e.recons.len());
    for (i, (p, r)) in payloads.iter().zip(&e.recons).enumerate() {
        assert!(same(dec.decode_frame(p).unwrap(), r), "{what}: frame {i}");
    }
}

#[test]
fn p_frames_decode_exactly_over_long_runs() {
    let frames = clip(176, 104, 90, None);
    for (qp, filter, mesh) in [
        (12, Filter::Cubic, 5),
        (40, Filter::Bilinear, 4),
        (64, Filter::Sharp, 6),
        (90, Filter::Cubic, 5),
    ] {
        let mut cfg = EncoderConfig::new(176, 104);
        cfg.qp = qp;
        cfg.filter = Some(filter);
        cfg.mesh_log2 = mesh;
        let e = encode(cfg, &frames);
        let intra: Vec<usize> = (1..e.inter.len()).filter(|&i| !e.inter[i]).collect();
        assert!(intra.is_empty(), "qp {qp}: intra frames {intra:?}");
        assert_decodes(&e, &format!("qp {qp}, {filter:?}, mesh 2^{mesh}"));
    }
}

#[test]
fn overlapped_prediction_decodes_exactly() {
    let frames = clip(176, 104, 30, None);
    for (qp, filter, mesh) in [
        (12, Filter::Smooth, 5),
        (40, Filter::Bilinear, 4),
        (64, Filter::Sharp, 6),
        (90, Filter::Cubic, 5),
    ] {
        let mut cfg = EncoderConfig::new(176, 104);
        cfg.qp = qp;
        cfg.filter = Some(filter);
        cfg.prediction = Some(Prediction::Overlapped);
        cfg.mesh_log2 = mesh;
        let e = encode(cfg, &frames);
        assert_decodes(
            &e,
            &format!("overlapped, qp {qp}, {filter:?}, mesh 2^{mesh}"),
        );
    }
    for (w, h) in [(1, 1), (7, 5), (33, 17)] {
        let mut cfg = EncoderConfig::new(w, h);
        cfg.prediction = Some(Prediction::Overlapped);
        assert_decodes(
            &encode(cfg, &clip(w, h, 4, None)),
            &format!("overlapped {w}x{h}"),
        );
    }
}

#[test]
fn odd_sizes_and_tiny_frames_work() {
    for (w, h) in [(1, 1), (7, 5), (33, 17), (65, 129)] {
        let frames = clip(w, h, 6, None);
        let mut cfg = EncoderConfig::new(w, h);
        cfg.luma_levels = 3;
        cfg.chroma_levels = 2;
        cfg.qp = 20;
        assert_decodes(&encode(cfg, &frames), &format!("{w}x{h}"));
    }
}

#[test]
fn lossless_p_frames_are_exact() {
    let frames = clip(96, 64, 8, None);
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.lossless = true;
    let e = encode(cfg, &frames);
    assert!(e.inter[1..].iter().all(|&p| p));
    for (f, r) in frames.iter().zip(&e.recons) {
        assert!(same(f, r));
    }
    assert_decodes(&e, "lossless");
}

#[test]
fn p_frames_beat_intra_on_moving_content() {
    let frames = clip(256, 144, 30, None);
    let size = |p_frames: bool| {
        let mut cfg = EncoderConfig::new(256, 144);
        cfg.qp = 40;
        cfg.p_frames = p_frames;
        let e = encode(cfg, &frames);
        let psnr: f64 = frames
            .iter()
            .zip(&e.recons)
            .map(|(f, r)| vdwt::metrics::psnr_u8(&f.y, &r.y))
            .sum::<f64>()
            / frames.len() as f64;
        (e.stream.len(), psnr)
    };
    let (intra, intra_psnr) = size(false);
    let (inter, inter_psnr) = size(true);
    assert!(
        inter * 2 < intra && inter_psnr > intra_psnr - 0.5,
        "P: {inter} bytes at {inter_psnr:.2} dB; intra: {intra} bytes at {intra_psnr:.2} dB"
    );
}

#[test]
fn scene_cuts_are_coded_intra() {
    let frames = clip(160, 96, 12, Some(6));
    let mut cfg = EncoderConfig::new(160, 96);
    cfg.qp = 30;
    let e = encode(cfg, &frames);
    assert_eq!(
        e.inter,
        [
            false, true, true, true, true, true, false, true, true, true, true, true
        ]
    );
    assert_decodes(&e, "scene cut");
}

#[test]
fn intra_period_inserts_intra_frames() {
    let frames = clip(96, 64, 10, None);
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.intra_period = 4;
    let e = encode(cfg, &frames);
    let intra: Vec<usize> = (0..10).filter(|&i| !e.inter[i]).collect();
    assert_eq!(intra, [0, 4, 8]);
    assert_decodes(&e, "intra period");
}

#[test]
fn every_isa_produces_the_same_p_stream() {
    let frames = clip(128, 80, 10, None);
    let streams: Vec<Vec<u8>> = Isa::supported()
        .map(|isa| {
            let mut cfg = EncoderConfig::new(128, 80);
            cfg.isa = isa;
            encode(cfg, &frames).stream
        })
        .collect();
    assert!(streams.windows(2).all(|s| s[0] == s[1]));
}

#[test]
fn threads_change_nothing() {
    // A scene cut, CBR band dropping and every stage's parallel path.
    let frames = clip(320, 192, 12, Some(6));
    let encodes: Vec<Encoded> = [1, 2, 4]
        .into_iter()
        .map(|threads| {
            let mut cfg = EncoderConfig::new(320, 192);
            cfg.threads = threads;
            cfg.rate_control = Some(vdwt::RateControl::new(400_000));
            encode(cfg, &frames)
        })
        .collect();
    assert!(encodes.windows(2).all(|e| e[0].stream == e[1].stream));
    let (seq, payloads) = bitstream::split_stream(&encodes[0].stream).unwrap();
    for threads in [1, 3, 4] {
        let mut dec = Decoder::with_threads(seq.clone(), Isa::best(), threads).unwrap();
        for (i, (p, r)) in payloads.iter().zip(&encodes[0].recons).enumerate() {
            assert!(
                same(dec.decode_frame(p).unwrap(), r),
                "{threads} threads: frame {i}"
            );
        }
    }
}

#[test]
fn corrupt_frames_never_panic_a_threaded_decoder() {
    let frames = clip(160, 96, 3, None);
    let mut cfg = EncoderConfig::new(160, 96);
    cfg.qp = 20;
    let e = encode(cfg, &frames);
    let (seq, payloads) = bitstream::split_stream(&e.stream).unwrap();
    let mut rng = SplitMix64::new(7);
    let mut dec = Decoder::with_threads(seq, Isa::best(), 4).unwrap();
    dec.decode_frame(payloads[0]).unwrap();
    for i in 0..400 {
        let mut p = payloads[i % 3].to_vec();
        for _ in 0..rng.range_usize(1, 8) {
            let at = rng.below(p.len() as u64) as usize;
            p[at] ^= 1 << rng.below(8);
        }
        let _ = dec.decode_frame(&p);
    }
}

#[test]
fn short_headers_decode_exactly() {
    let frames = clip(96, 64, 12, None);
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.qp = 42;
    cfg.short_headers = true;
    let e = encode(cfg, &frames);
    assert_decodes(&e, "short headers");
    let (_, payloads) = bitstream::split_stream(&e.stream).unwrap();
    let mut state: Option<bitstream::FrameHeaderState> = None;
    let mut short = 0;
    for p in &payloads {
        let prev = state.as_ref().map(|h| bitstream::FrameHeaderState {
            qp: h.qp,
            recon_offset: h.recon_offset,
            postfilter_strength: h.postfilter_strength,
        });
        let h = FrameHeader::parse_with(&mut bitstream::ByteReader::new(p), prev.as_ref()).unwrap();
        if p[0] & 4 != 0 {
            short += 1;
        }
        if h.frame_type != FrameType::Repeat {
            state = Some(bitstream::FrameHeaderState {
                qp: h.qp,
                recon_offset: h.recon_offset,
                postfilter_strength: h.postfilter_strength,
            });
        }
    }
    assert!(short > 0, "the encoder never chose a short header");
}

#[test]
fn p_frames_need_a_previous_picture() {
    let frames = clip(64, 48, 2, None);
    let e = encode(EncoderConfig::new(64, 48), &frames);
    let (seq, payloads) = bitstream::split_stream(&e.stream).unwrap();
    let header = FrameHeader::parse(&mut bitstream::ByteReader::new(payloads[1])).unwrap();
    assert_eq!(header.frame_type, FrameType::Inter);
    let mut dec = Decoder::new(seq).unwrap();
    assert!(dec.decode_frame(payloads[1]).is_err());
    assert!(dec.decode_frame(payloads[0]).is_ok());
    assert!(dec.decode_frame(payloads[1]).is_ok());
}

#[test]
fn corrupt_p_frames_never_panic() {
    let frames = clip(96, 64, 4, None);
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.qp = 24;
    let e = encode(cfg, &frames);
    let (seq, payloads) = bitstream::split_stream(&e.stream).unwrap();
    let mut rng = SplitMix64::new(99);
    let mut dec = Decoder::new(seq).unwrap();
    dec.decode_frame(payloads[0]).unwrap();
    for i in 0..1500 {
        let mut p = payloads[1 + i % 3].to_vec();
        match i % 3 {
            0 => p.truncate(rng.below(p.len() as u64) as usize),
            1 => {
                for _ in 0..rng.range_usize(1, 8) {
                    let at = rng.below(p.len() as u64) as usize;
                    p[at] ^= 1 << rng.below(8);
                }
            }
            _ => {
                // Garbage after a valid P-frame header.
                p.truncate(6);
                p.extend((0..rng.below(120)).map(|_| rng.next_u64() as u8));
            }
        }
        let _ = dec.decode_frame(&p);
    }
    // An intra frame still decodes exactly afterwards.
    let mut fresh = Decoder::new(bitstream::split_stream(&e.stream).unwrap().0).unwrap();
    assert!(same(
        dec.decode_frame(payloads[0]).unwrap(),
        fresh.decode_frame(payloads[0]).unwrap()
    ));
}

#[test]
fn recording_contexts_counts_band_and_motion_symbols() {
    let frames = clip(128, 80, 4, None);
    let mut enc = Encoder::new(EncoderConfig::new(128, 80)).unwrap();
    enc.record_contexts();
    let mut out = Vec::new();
    for f in &frames {
        enc.encode_frame(f, &mut out).unwrap();
    }
    let stats = enc.take_context_stats().unwrap();
    assert!(stats.symbols() > 1000, "{}", stats.symbols());
    let rust = stats.to_rust(1);
    // One list per context set: 28 band classes and 3 motion planes. Sets with no symbols are
    // emitted as a single-line `&[],`, so count the openers, not the multi-line form.
    assert_eq!(rust.lines().filter(|l| l.starts_with("    &[")).count(), 31);
    assert!(enc.take_context_stats().is_none());
}

/// Encodes `frames` as the ultraheavy tier does: the first alone, then groups of `group` (a
/// B-frame hierarchy and its anchor) by `encode_group`, a tail of 2 or 1 as it fits. Returns
/// the stream and every payload's expected decode, in coding order.
fn encode_groups(cfg: EncoderConfig, frames: &[Frame], group: usize) -> (Vec<u8>, Vec<Frame>) {
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let mut expected = Vec::new();
    enc.encode_frame(&frames[0], &mut stream).unwrap();
    expected.push(enc.recon().clone());
    let mut i = 1;
    while i < frames.len() {
        let n = match frames.len() - i {
            left if left >= group => group,
            left if left >= 2 => 2,
            _ => 1,
        };
        if n == 1 {
            enc.encode_frame(&frames[i], &mut stream).unwrap();
            expected.push(enc.recon().clone());
            break;
        }
        let refs: Vec<&Frame> = frames[i..i + n].iter().collect();
        let mut packet = Vec::new();
        enc.encode_group(&refs, &mut packet).unwrap();
        // Coding order: each payload's frame number says which frame of the group it is.
        let mut r = bitstream::ByteReader::new(&packet);
        for _ in 0..n {
            let p = r.chunk().unwrap();
            let num = FrameHeader::parse(&mut bitstream::ByteReader::new(p))
                .unwrap()
                .frame_num;
            expected.push(enc.group_recon(usize::from(num) - i).clone());
        }
        stream.extend_from_slice(&packet);
        i += n;
    }
    (stream, expected)
}

#[test]
fn b_frames_decode_exactly_in_display_order() {
    let frames = clip(176, 104, 41, None);
    for (qp, effort, group) in [(40, 1, 2), (64, 2, 4), (24, 0, 4)] {
        let mut cfg = EncoderConfig::new(176, 104);
        cfg.qp = qp;
        cfg.effort = effort;
        let (stream, expected) = encode_groups(cfg, &frames, group);
        let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
        assert_eq!(payloads.len(), frames.len());
        let mut dec = Decoder::new(seq).unwrap();
        let (mut b_frames, mut anchor) = (0, 0);
        for (i, (p, r)) in payloads.iter().zip(&expected).enumerate() {
            let header = FrameHeader::parse(&mut bitstream::ByteReader::new(p)).unwrap();
            if header.frame_type == FrameType::Bi {
                b_frames += 1;
                // Shown before its group's anchor, which decoded first.
                assert!(header.frame_num < anchor, "qp {qp}: payload {i}");
            } else {
                anchor = header.frame_num;
            }
            assert!(
                same(dec.decode_frame(p).unwrap(), r),
                "qp {qp}: payload {i}"
            );
        }
        // Every group but the anchor: 40 frames after the first.
        assert_eq!(b_frames, 40 / group * (group - 1), "qp {qp}");
    }
}

/// Encodes `frames` as the ultraheavy tier does: by `encode_pair`, the first frame alone and
/// then 2 at a time. Returns the stream and every payload's expected decode, in coding
/// order.
fn encode_pairs(cfg: EncoderConfig, frames: &[Frame]) -> (Vec<u8>, Vec<Frame>) {
    encode_steps(cfg, frames, 2)
}

/// Encodes `frames` through [`Encoder::encode_pair`] in steps of up to `step` frames (after a
/// first single frame); returns the stream and the reconstructions in coding order.
fn encode_steps(cfg: EncoderConfig, frames: &[Frame], step: usize) -> (Vec<u8>, Vec<Frame>) {
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let mut expected = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let left = frames.len() - i;
        let n = match i {
            0 => 1,
            _ if left >= step => step,
            _ => 2.min(left),
        };
        let refs: Vec<&Frame> = frames[i..i + n].iter().collect();
        let mut packet = Vec::new();
        enc.encode_pair(&refs, &mut packet).unwrap();
        let mut r = bitstream::ByteReader::new(&packet);
        for _ in 0..n {
            let p = r.chunk().unwrap();
            let num = FrameHeader::parse(&mut bitstream::ByteReader::new(p))
                .unwrap()
                .frame_num;
            expected.push(
                enc.group_recon(usize::from(num.wrapping_sub(i as u16)))
                    .clone(),
            );
        }
        stream.extend_from_slice(&packet);
        i += n;
    }
    (stream, expected)
}

#[test]
fn ultraheavy_pairs_decode_exactly() {
    let frames = clip(176, 104, 23, None);
    let mut kinds = [0usize; 2];
    for (qp, rate) in [(30, None), (66, None), (48, Some(150_000))] {
        let mut cfg = EncoderConfig::new(176, 104);
        cfg.qp = qp;
        cfg.effort = 3;
        cfg.rate_control = rate.map(vdwt::RateControl::new);
        // Coarser chroma than the default keeps motion over the B-pair share on this small clip,
        // so both kinds of pair occur.
        cfg.chroma_step_scale = 0.95;
        let (stream, expected) = encode_pairs(cfg.clone(), &frames);
        let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
        assert_eq!(payloads.len(), frames.len());
        let mut dec = Decoder::new(seq).unwrap();
        for (i, (p, r)) in payloads.iter().zip(&expected).enumerate() {
            let header = FrameHeader::parse(&mut bitstream::ByteReader::new(p)).unwrap();
            match header.frame_type {
                FrameType::Bi => kinds[1] += 1,
                FrameType::Inter => kinds[0] += 1,
                _ => {}
            }
            assert!(
                same(dec.decode_frame(p).unwrap(), r),
                "qp {qp}: payload {i}"
            );
        }
    }
    // Both kinds of pair are chosen somewhere.
    assert!(kinds[0] > 0 && kinds[1] > 0, "{kinds:?}");
}

#[test]
fn ultraheavy_still_groups_of_4_decode_exactly() {
    // A still camera: a fixed background with a square moving over it.
    let base = frame(176, 104, 0, 0);
    let frames: Vec<Frame> = (0..13)
        .map(|t| {
            let mut f = base.clone();
            for y in 40..56 {
                for v in &mut f.y.row_mut(y)[20 + 3 * t..36 + 3 * t] {
                    *v = 230;
                }
            }
            f
        })
        .collect();
    let mut cfg = EncoderConfig::new(176, 104);
    cfg.qp = 36;
    cfg.effort = 3;
    let (stream, expected) = encode_steps(cfg, &frames, 4);
    let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
    assert_eq!(payloads.len(), frames.len());
    let mut dec = Decoder::new(seq).unwrap();
    let mut bi = 0;
    for (i, (p, r)) in payloads.iter().zip(&expected).enumerate() {
        let header = FrameHeader::parse(&mut bitstream::ByteReader::new(p)).unwrap();
        bi += usize::from(header.frame_type == FrameType::Bi);
        assert!(same(dec.decode_frame(p).unwrap(), r), "payload {i}");
    }
    // Groups of 4: three B-frames per group.
    assert!(bi >= 6, "{bi} B-frames");
}

#[test]
fn probe_mesh_picks_the_grid_or_half_of_it() {
    let frames = clip(176, 104, 9, None);
    let mut cfg = EncoderConfig::new(176, 104);
    cfg.qp = 40;
    let refs: Vec<&Frame> = frames.iter().collect();
    let log2 = vdwt::encoder::probe_mesh(&cfg, &refs).unwrap();
    assert!(log2 == cfg.mesh_log2 || log2 + 1 == cfg.mesh_log2, "{log2}");
}

#[test]
fn b_frames_need_their_kept_reference_and_never_panic() {
    let frames = clip(96, 64, 7, None);
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.qp = 50;
    let (stream, _) = encode_groups(cfg, &frames, 4);
    let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
    let is_b = |p: &[u8]| {
        FrameHeader::parse(&mut bitstream::ByteReader::new(p))
            .unwrap()
            .frame_type
            == FrameType::Bi
    };
    let b = payloads.iter().position(|p| is_b(p)).expect("a B-frame");
    // Without the anchor that kept its reference, a B-frame is refused.
    let mut dec = Decoder::new(seq.clone()).unwrap();
    dec.decode_frame(payloads[0]).unwrap();
    assert!(dec.decode_frame(payloads[b]).is_err());
    let mut rng = SplitMix64::new(7);
    let mut dec = Decoder::new(seq).unwrap();
    for p in &payloads[..b] {
        dec.decode_frame(p).unwrap();
    }
    for i in 0..1500 {
        let mut p = payloads[b].to_vec();
        match i % 3 {
            0 => p.truncate(rng.below(p.len() as u64) as usize),
            1 => {
                for _ in 0..rng.range_usize(1, 8) {
                    let at = rng.below(p.len() as u64) as usize;
                    p[at] ^= 1 << rng.below(8);
                }
            }
            _ => {
                p.truncate(11);
                p.extend((0..rng.below(120)).map(|_| rng.next_u64() as u8));
            }
        }
        let _ = dec.decode_frame(&p);
    }
}

#[test]
fn a_repeat_frame_shows_the_last_picture() {
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.loop_filter = false;
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    enc.encode_frame(&frame(96, 64, 1, 0), &mut stream).unwrap();
    let recon = enc.recon().clone();
    let stats = enc.encode_repeat(&mut stream).unwrap();
    assert!(stats.repeated);
    assert_eq!(stats.bytes, 2, "a Repeat is {} bytes", stats.bytes);
    let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
    let mut dec = Decoder::new(seq).unwrap();
    let first = dec.decode_frame(payloads[0]).unwrap().clone();
    let second = dec.decode_frame(payloads[1]).unwrap().clone();
    assert!(same(&first, &recon));
    assert!(same(&second, &recon));
    let header = FrameHeader::parse(&mut bitstream::ByteReader::new(payloads[1])).unwrap();
    assert_eq!(header.frame_type, FrameType::Repeat);
}

#[test]
fn auto_postfilter_is_per_scene_and_signalled() {
    let (w, h) = (128, 96);
    let mut cfg = EncoderConfig::new(w, h);
    cfg.qp = 50;
    cfg.p_frames = true;
    cfg.postfilter = PostFilterMode::Auto;
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let mut strengths = Vec::new();
    for t in 0..8 {
        enc.encode_frame(&frame(w, h, 1, t), &mut stream).unwrap();
        let (_, payloads) = bitstream::split_stream(&stream).unwrap();
        let header = FrameHeader::parse(&mut bitstream::ByteReader::new(payloads[t])).unwrap();
        strengths.push(header.postfilter_strength);
    }
    assert!(strengths[0] > 0, "auto should pick a filter for this scene");
    assert!(strengths.iter().all(|&s| s <= 40));
    assert!(
        strengths.windows(2).all(|w| w[0] == w[1]),
        "one strength per scene, got {strengths:?}"
    );
}

#[test]
fn auto_postfilter_decode_matches_the_filtered_recon() {
    let (w, h) = (128, 96);
    let mut cfg = EncoderConfig::new(w, h);
    cfg.qp = 50;
    cfg.postfilter = PostFilterMode::Auto;
    let mut enc = Encoder::new(cfg).unwrap();
    let seq = enc.sequence_header().clone();
    let mut dec = Decoder::new(seq).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    for t in 0..6 {
        let stats = enc.encode_frame(&frame(w, h, 1, t), &mut stream).unwrap();
        let (_, payloads) = bitstream::split_stream(&stream).unwrap();
        let decoded = dec.decode_frame(payloads[t]).unwrap();
        assert!(stats.postfilter_strength > 0);
        let strength = u16::from(stats.postfilter_strength) * 32;
        let mut a = Frame::new(w, h);
        let mut b = Frame::new(w, h);
        vdwt::postfilter::band_wide(enc.recon(), &mut a, strength);
        vdwt::postfilter::band_wide(&decoded, &mut b, strength);
        for (pa, pb) in a.planes().iter().zip(b.planes()) {
            assert!(
                pa.rows().eq(pb.rows()),
                "frame {t} differs after the signalled filter"
            );
        }
    }
}
