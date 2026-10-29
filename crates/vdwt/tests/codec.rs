//! End-to-end checks through the public API: encode, split the stream, decode.

use vdwt::bitstream::{self, SequenceHeader};
use vdwt::rng::SplitMix64;
use vdwt::{Decoder, Encoder, EncoderConfig, Frame, Isa};

/// Smooth gradients, hard edges and noise, so every band gets some data.
fn test_frame(width: usize, height: usize, seed: u64) -> Frame {
    let mut rng = SplitMix64::new(seed);
    let mut frame = Frame::new(width, height);
    for (c, plane) in frame.planes_mut().into_iter().enumerate() {
        let (w, h) = (plane.width(), plane.height());
        for y in 0..h {
            for (x, v) in plane.row_mut(y).iter_mut().enumerate() {
                let gradient = (x * 160 / w + y * 60 / h + 20 * c) as i32;
                let edge = if (x / 37 + y / 23) % 2 == 0 { 30 } else { -30 };
                let noise = rng.range_i32(-6, 6);
                *v = (gradient + edge + noise).clamp(0, 255) as u8;
            }
        }
    }
    frame
}

fn encode(cfg: EncoderConfig, frames: &[Frame]) -> (Vec<u8>, Vec<Frame>) {
    let mut enc = Encoder::new(cfg).unwrap();
    let mut stream = Vec::new();
    enc.sequence_header().write(&mut stream);
    let mut recons = Vec::new();
    for f in frames {
        enc.encode_frame(f, &mut stream).unwrap();
        recons.push(enc.recon().clone());
    }
    (stream, recons)
}

fn decode(stream: &[u8]) -> Vec<Frame> {
    let (seq, payloads) = bitstream::split_stream(stream).unwrap();
    let mut dec = Decoder::new(seq).unwrap();
    payloads
        .iter()
        .map(|p| dec.decode_frame(p).unwrap().clone())
        .collect()
}

fn same(a: &Frame, b: &Frame) -> bool {
    a.planes()
        .iter()
        .zip(b.planes())
        .all(|(p, q)| p.rows().eq(q.rows()))
}

#[test]
fn decoder_matches_encoder_reconstruction() {
    let frames: Vec<Frame> = (0..2).map(|i| test_frame(161, 97, i)).collect();
    for qp in [0, 12, 24, 40, 60, 90, 200] {
        for (parent, ll) in [(true, true), (false, false), (true, false)] {
            let mut cfg = EncoderConfig::new(161, 97);
            cfg.qp = qp;
            cfg.parent_contexts = parent;
            cfg.ll_prediction = ll;
            let (stream, recons) = encode(cfg, &frames);
            let decoded = decode(&stream);
            assert_eq!(decoded.len(), frames.len());
            for (d, r) in decoded.iter().zip(&recons) {
                assert!(same(d, r), "qp {qp}, parent {parent}, ll {ll}");
            }
        }
    }
}

#[test]
fn lossless_mode_is_exact() {
    for (w, h, levels) in [
        (64, 48, 3),
        (161, 97, 5),
        (1, 1, 0),
        (7, 300, 6),
        (300, 2, 5),
    ] {
        let frames = [test_frame(w, h, 3)];
        let mut cfg = EncoderConfig::new(w, h);
        cfg.lossless = true;
        cfg.luma_levels = levels;
        cfg.chroma_levels = levels.saturating_sub(1);
        let (stream, _) = encode(cfg, &frames);
        assert!(same(&decode(&stream)[0], &frames[0]), "{w}x{h}");
    }
}

#[test]
fn quality_and_size_move_with_qp() {
    let frame = test_frame(256, 144, 5);
    let mut last_bytes = usize::MAX;
    let mut last_psnr = f64::INFINITY;
    for qp in [8, 24, 40, 56] {
        let mut cfg = EncoderConfig::new(256, 144);
        cfg.qp = qp;
        // This checks the quantizer's monotonicity; the perceptual loop filter trades PSNR
        // for VMAF and is off here, and the adaptive finest tilt changes the step weights
        // with qp.
        cfg.loop_sharpen = 0.0;
        cfg.finest_tilt_auto = false;
        let (stream, recons) = encode(cfg, std::slice::from_ref(&frame));
        let psnr = vdwt::metrics::psnr_u8(&frame.y, &recons[0].y);
        assert!(
            stream.len() < last_bytes && psnr < last_psnr,
            "qp {qp}: {} bytes, {psnr:.2} dB",
            stream.len()
        );
        (last_bytes, last_psnr) = (stream.len(), psnr);
    }
}

#[test]
fn every_isa_produces_the_same_stream() {
    let frames = [test_frame(200, 120, 9)];
    let streams: Vec<Vec<u8>> = Isa::supported()
        .map(|isa| {
            let mut cfg = EncoderConfig::new(200, 120);
            cfg.isa = isa;
            encode(cfg, &frames).0
        })
        .collect();
    assert!(streams.windows(2).all(|s| s[0] == s[1]));
}

#[test]
fn corrupt_streams_never_panic() {
    let frames = [test_frame(96, 64, 1)];
    let mut cfg = EncoderConfig::new(96, 64);
    cfg.qp = 20;
    let (stream, _) = encode(cfg, &frames);
    let (seq, payloads) = bitstream::split_stream(&stream).unwrap();
    let payload = payloads[0].to_vec();
    let mut dec = Decoder::new(seq.clone()).unwrap();
    let mut rng = SplitMix64::new(77);
    for i in 0..2000 {
        let mut p = payload.clone();
        match i % 3 {
            0 => p.truncate(rng.below(p.len() as u64) as usize),
            1 => {
                for _ in 0..rng.range_usize(1, 8) {
                    let at = rng.below(p.len() as u64) as usize;
                    p[at] ^= 1 << rng.below(8);
                }
            }
            _ => p = (0..rng.below(200)).map(|_| rng.next_u64() as u8).collect(),
        }
        let _ = dec.decode_frame(&p);
    }
    // The decoder still works afterwards.
    assert!(dec.decode_frame(&payload).is_ok());
}

#[test]
fn bad_sequence_headers_are_rejected() {
    let cfg = EncoderConfig::new(32, 32);
    let enc = Encoder::new(cfg).unwrap();
    let mut buf = Vec::new();
    enc.sequence_header().write(&mut buf);
    assert!(SequenceHeader::parse(&buf).is_ok());
    let mut rng = SplitMix64::new(5);
    for _ in 0..500 {
        let mut b = buf.clone();
        let at = rng.below(b.len() as u64) as usize;
        b[at] = rng.next_u64() as u8;
        if let Ok((seq, _)) = SequenceHeader::parse(&b) {
            // Whatever parses must also be decodable without panicking.
            if let Ok(mut dec) = Decoder::new(seq) {
                let _ = dec.decode_frame(&[0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
            }
        }
    }
}
