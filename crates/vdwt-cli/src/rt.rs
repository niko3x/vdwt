//! `rt-encode` and `rt-decode`: the realtime codec for lossy links (`vdwt::rt`).

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use vdwt::Isa;
use vdwt::metrics::YuvPsnr;
use vdwt::profile::StageTimes;
use vdwt::rt::{self, LossChannel, RtConfig, RtDecoder, RtEncoder};

use super::y4m::{Y4mInfo, Y4mWriter};
use super::{Args, fps, ms, open_y4m, parse_schedule, read_frames, stage_line, threads};

pub(super) fn encode(mut args: Args) -> Result<(), String> {
    let max_frames = args.get("frames", usize::MAX)?;
    let schedule = parse_schedule(&args.string("bitrate").ok_or("rt-encode needs --bitrate")?)?;
    let max_delay = args.get("max-delay", 1.5f64)?;
    let tile_roots = args.get("tile-roots", 0usize)?;
    let refresh = args.get("refresh", 30usize)?;
    let mtu = args.get("mtu", 1200usize)?;
    let loop_filter = args.flag("loop-filter");
    let global_only = args.flag("global-only");
    let mesh_search = args.flag("mesh-search");
    let recon_path = args.string("recon");
    let verbose = args.flag("verbose");
    let threads = threads(&mut args)?;
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let (info, frames) = read_frames(input, max_frames)?;
    let frame_rate = fps(&info);
    let mut cfg = RtConfig::new(info.width, info.height, schedule[0].1);
    cfg.fps_num = info.fps_num;
    cfg.fps_den = info.fps_den;
    cfg.max_delay_frames = max_delay;
    cfg.tile_roots = tile_roots;
    cfg.refresh = refresh;
    cfg.mtu = mtu;
    cfg.loop_filter = loop_filter;
    cfg.global_only = global_only;
    cfg.mesh_search = mesh_search;
    cfg.threads = threads;
    let mut encoder = RtEncoder::new(cfg).map_err(|e| e.to_string())?;
    let mut stream = Vec::new();
    encoder.header().write(&mut stream);
    let mut recon = match &recon_path {
        Some(p) => Some(Y4mWriter::create(Path::new(p), &info).map_err(|e| format!("{p}: {e}"))?),
        None => None,
    };
    let (mut psnr, mut min_y, mut elapsed) = ([0.0f64; 4], f64::INFINITY, Duration::ZERO);
    let mut times = StageTimes::default();
    let (mut packets, mut motion, mut retries, mut qp_sum) = (0usize, 0usize, 0usize, 0u64);
    let (mut link, mut queue, mut worst_delay, mut largest) = (0u64, 0.0f64, 0.0f64, 0usize);
    for (i, frame) in frames.iter().enumerate() {
        if let Some(&(_, rate)) = schedule.iter().find(|&&(f, _)| f == i) {
            encoder.set_bitrate(rate);
            link = rate;
        }
        let start = Instant::now();
        let out = encoder.encode_frame(frame).map_err(|e| e.to_string())?;
        elapsed += start.elapsed();
        times += out.stats.times;
        rt::write_frame(&mut stream, &out.packets);
        let p = YuvPsnr::between(frame, encoder.reconstruction());
        for (s, v) in psnr.iter_mut().zip([p.y, p.u, p.v, p.weighted()]) {
            *s += v;
        }
        min_y = min_y.min(p.y);
        if verbose {
            eprintln!(
                "frame {i} {} qp {} bytes {} packets {} refresh {:?} retries {} psnr {:.2}",
                if out.stats.inter { "P" } else { "I" },
                out.stats.qp,
                out.stats.bytes,
                out.packets.len(),
                out.stats.refreshed,
                out.stats.retries,
                p.y
            );
        }
        packets += out.packets.len();
        largest = largest.max(out.packets.iter().map(Vec::len).max().unwrap_or(0));
        motion += out.stats.motion_bytes;
        retries += out.stats.retries;
        qp_sum += u64::from(out.stats.qp);
        let delay = (queue + 8.0 * out.stats.bytes as f64) / (link as f64 / frame_rate);
        worst_delay = worst_delay.max(delay);
        queue = out.stats.queue_bits;
        if let Some(w) = &mut recon {
            w.write_frame(encoder.reconstruction())
                .map_err(|e| e.to_string())?;
        }
    }
    if let Some(w) = recon {
        w.finish().map_err(|e| e.to_string())?;
    }
    fs::write(output, &stream).map_err(|e| format!("{output}: {e}"))?;
    let n = frames.len().max(1) as f64;
    let seconds = n / frame_rate;
    println!(
        "{} frames {}x{}: {:.1} kbps, {:.2} packets/frame (largest {largest} bytes), motion {:.1}%",
        frames.len(),
        info.width,
        info.height,
        8.0 * stream.len() as f64 / seconds / 1000.0,
        packets as f64 / n,
        100.0 * motion as f64 / stream.len().max(1) as f64,
    );
    println!(
        "PSNR   Y {:.2} (min {min_y:.2})  U {:.2}  V {:.2}  YUV {:.2} dB;  qp mean {:.1};  \
         re-codes {retries};  worst queueing delay {worst_delay:.2} frames (limit {max_delay})",
        psnr[0] / n,
        psnr[1] / n,
        psnr[2] / n,
        psnr[3] / n,
        qp_sum as f64 / n,
    );
    println!(
        "encode {:.2} ms/frame ({} thread{})",
        ms(elapsed) / n,
        threads,
        if threads == 1 { "" } else { "s" }
    );
    println!("{}", stage_line(&times, frames.len()));
    Ok(())
}

pub(super) fn decode(mut args: Args) -> Result<(), String> {
    let threads = threads(&mut args)?;
    let loss = args.get("loss", 0.0f64)?;
    let burst = args.get("burst", 1.0f64)?;
    let seed = args.get("seed", 1u64)?;
    let reference = args.string("ref");
    let verbose = args.flag("verbose");
    // Packets to drop besides the channel's: FRAME:PACKET,...
    let drop: Vec<(usize, usize)> = args
        .string("drop")
        .map(|s| {
            s.split(',')
                .map(|fp| {
                    let (f, p) = fp
                        .split_once(':')
                        .ok_or(format!("--drop {fp}: FRAME:PACKET"))?;
                    let n = |v: &str| v.parse::<usize>().map_err(|_| format!("--drop {fp}"));
                    Ok((n(f)?, n(p)?))
                })
                .collect::<Result<_, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    args.finish(2)?;
    let (input, output) = (&args.positional[0], &args.positional[1]);
    let data = fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let (header, frames) = rt::split_stream(&data).map_err(|e| format!("{input}: {e}"))?;
    let seq = &header.seq;
    let info = Y4mInfo {
        width: seq.width,
        height: seq.height,
        fps_num: seq.fps_num,
        fps_den: seq.fps_den,
    };
    let mut source = reference.as_deref().map(open_y4m).transpose()?;
    let mut src_frame = source.as_ref().map(|s| s.new_frame());
    let mut decoder =
        RtDecoder::new(header.clone(), Isa::best(), threads).map_err(|e| e.to_string())?;
    let mut writer =
        Y4mWriter::create(Path::new(output), &info).map_err(|e| format!("{output}: {e}"))?;
    let mut channel = LossChannel::new(loss, burst, seed);
    let (mut elapsed, mut times) = (Duration::ZERO, StageTimes::default());
    let (mut sent, mut arrived, mut tiles, mut tiles_got, mut repeated) = (0, 0, 0, 0, 0);
    let (mut psnr, mut min_y, mut scored) = ([0.0f64; 4], f64::INFINITY, 0usize);
    for (i, packets) in frames.iter().enumerate() {
        let received: Vec<Option<&[u8]>> = packets
            .iter()
            .enumerate()
            .map(|(k, &p)| (channel.arrives() && !drop.contains(&(i, k))).then_some(p))
            .collect();
        sent += packets.len();
        arrived += received.iter().filter(|p| p.is_some()).count();
        let start = Instant::now();
        // An error means no picture yet (the start was lost): the blank one stands.
        let got = decoder.decode_frame(&received).unwrap_or_default();
        elapsed += start.elapsed();
        times += *decoder.times();
        tiles += got.tiles;
        tiles_got += got.tiles_received;
        repeated += usize::from(got.repeated);
        writer
            .write_frame(decoder.frame())
            .map_err(|e| e.to_string())?;
        if let (Some(src), Some(f)) = (&mut source, &mut src_frame)
            && src.read_frame(f).map_err(|e| e.to_string())?
        {
            let p = YuvPsnr::between(f, decoder.frame());
            for (s, v) in psnr.iter_mut().zip([p.y, p.u, p.v, p.weighted()]) {
                *s += v;
            }
            min_y = min_y.min(p.y);
            scored += 1;
            if verbose {
                eprintln!(
                    "frame {i} packets {}/{} tiles {}/{} psnr {:.2}",
                    received.iter().filter(|p| p.is_some()).count(),
                    packets.len(),
                    got.tiles_received,
                    got.tiles,
                    p.y
                );
            }
        }
    }
    writer.finish().map_err(|e| e.to_string())?;
    let n = frames.len().max(1) as f64;
    println!(
        "decoded {} frames: packets {arrived}/{sent} arrived ({:.1}% lost), tiles {tiles_got}/{tiles}, \
         repeated frames {repeated}",
        frames.len(),
        100.0 * (sent - arrived) as f64 / sent.max(1) as f64,
    );
    if scored > 0 {
        let k = scored as f64;
        println!(
            "PSNR vs source   Y {:.2} (min {min_y:.2})  U {:.2}  V {:.2}  YUV {:.2} dB",
            psnr[0] / k,
            psnr[1] / k,
            psnr[2] / k,
            psnr[3] / k,
        );
    }
    println!(
        "decode {:.2} ms/frame ({threads} thread{})",
        ms(elapsed) / n,
        if threads == 1 { "" } else { "s" }
    );
    println!("{}", stage_line(&times, frames.len()));
    Ok(())
}
