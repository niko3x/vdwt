//! YUV4MPEG2 (`.y4m`) reader and writer.
//!
//! The codec works on 8-bit 4:2:0. The reader also takes 4:2:2, 4:4:4 and monochrome Y4M at 8 to
//! 16 bits and converts them on the fly (chroma averaged down to 4:2:0, samples rounded to 8 bits,
//! a monochrome source gets neutral chroma). `-` reads standard input; any other file that is not
//! `.y4m` is decoded through `ffmpeg` when it is installed. The writer always writes 8-bit 4:2:0,
//! to a file or, for `-`, to standard output.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use vdwt::Frame;

#[derive(Clone, Debug)]
pub struct Y4mInfo {
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
}

/// Source chroma layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chroma {
    C420,
    C422,
    C444,
    Mono,
}

pub struct Y4mReader {
    r: Box<dyn BufRead>,
    child: Option<Child>,
    pub info: Y4mInfo,
    chroma: Chroma,
    depth: u32,
    raw: Vec<u8>,
}

impl Y4mReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let is_y4m = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("y4m"));
        let (r, child): (Box<dyn BufRead>, _) = if path == Path::new("-") {
            (Box::new(BufReader::new(io::stdin())), None)
        } else if is_y4m {
            (Box::new(BufReader::new(File::open(path)?)), None)
        } else {
            // Any other container or codec: let ffmpeg decode it to Y4M in its native format.
            if !path.exists() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "no such file"));
            }
            let mut child = Command::new("ffmpeg")
                .args(["-nostdin", "-loglevel", "error", "-i"])
                .arg(path)
                .args(["-f", "yuv4mpegpipe", "-strict", "-1", "-"])
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    io::Error::new(
                        e.kind(),
                        format!("not a .y4m file and ffmpeg could not be run to decode it ({e})"),
                    )
                })?;
            let out = child.stdout.take().expect("piped stdout");
            (Box::new(BufReader::new(out)), Some(child))
        };
        Self::from_reader(r, child)
    }

    fn from_reader(mut r: Box<dyn BufRead>, child: Option<Child>) -> io::Result<Self> {
        let header = read_line(&mut r)?.ok_or_else(|| invalid("empty file"))?;
        let mut tokens = header.split(' ').filter(|t| !t.is_empty());
        if tokens.next() != Some("YUV4MPEG2") {
            return Err(invalid("not a YUV4MPEG2 file"));
        }
        let (mut width, mut height, mut fps) = (None, None, (30, 1));
        let (mut chroma, mut depth) = (Chroma::C420, 8);
        for token in tokens {
            let (tag, value) = token.split_at(1);
            match tag {
                "W" => width = value.parse().ok(),
                "H" => height = value.parse().ok(),
                "F" => {
                    let (n, d) = value
                        .split_once(':')
                        .ok_or_else(|| invalid("bad frame rate"))?;
                    fps = (
                        n.parse().map_err(|_| invalid("bad frame rate"))?,
                        d.parse().map_err(|_| invalid("bad frame rate"))?,
                    );
                }
                "C" => (chroma, depth) = colour_space(value)?,
                _ => {}
            }
        }
        let (Some(width), Some(height)) = (width, height) else {
            return Err(invalid("missing width or height"));
        };
        if width == 0 || height == 0 {
            return Err(invalid("zero frame size"));
        }
        if fps.0 == 0 || fps.1 == 0 {
            return Err(invalid("bad frame rate"));
        }
        if chroma != Chroma::C420 || depth != 8 {
            let name = match chroma {
                Chroma::C420 => "4:2:0",
                Chroma::C422 => "4:2:2",
                Chroma::C444 => "4:4:4",
                Chroma::Mono => "monochrome",
            };
            eprintln!("note: converting {depth}-bit {name} input to 8-bit 4:2:0");
        }
        Ok(Self {
            r,
            child,
            info: Y4mInfo {
                width,
                height,
                fps_num: fps.0,
                fps_den: fps.1,
            },
            chroma,
            depth,
            raw: Vec::new(),
        })
    }

    pub fn new_frame(&self) -> Frame {
        Frame::new(self.info.width, self.info.height)
    }

    /// Reads the next frame into `frame` (which must match the stream size). `Ok(false)` at end
    /// of stream.
    pub fn read_frame(&mut self, frame: &mut Frame) -> io::Result<bool> {
        let Some(marker) = read_line(&mut self.r)? else {
            return Ok(false);
        };
        if !marker.starts_with("FRAME") {
            return Err(invalid("missing FRAME marker"));
        }
        if self.chroma == Chroma::C420 && self.depth == 8 {
            for plane in frame.planes_mut() {
                for row in plane.rows_mut() {
                    self.r.read_exact(row)?;
                }
            }
            return Ok(true);
        }
        let (w, h) = (self.info.width, self.info.height);
        let (sw, sh) = match self.chroma {
            Chroma::C420 => (w.div_ceil(2), h.div_ceil(2)),
            Chroma::C422 => (w.div_ceil(2), h),
            Chroma::C444 => (w, h),
            Chroma::Mono => (0, 0),
        };
        let bytes = if self.depth > 8 { 2 } else { 1 };
        let n = (w * h + 2 * sw * sh) * bytes;
        self.raw.resize(n, 0);
        self.r.read_exact(&mut self.raw)?;
        let depth = self.depth;
        let sample = |raw: &[u8], i: usize| -> u32 {
            if bytes == 2 {
                u32::from(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
            } else {
                u32::from(raw[i])
            }
        };
        // Rounds a sum of `count` samples at `depth` bits to one 8-bit sample.
        let to8 = |sum: u32, count: u32| -> u8 {
            let scale = count << (depth - 8);
            ((sum + scale / 2) / scale).min(255) as u8
        };
        let [y, u, v] = frame.planes_mut();
        for (r, row) in y.rows_mut().enumerate() {
            for (c, px) in row.iter_mut().enumerate() {
                *px = to8(sample(&self.raw, r * w + c), 1);
            }
        }
        for (k, plane) in [u, v].into_iter().enumerate() {
            if self.chroma == Chroma::Mono {
                for row in plane.rows_mut() {
                    row.fill(128);
                }
                continue;
            }
            let base = w * h + k * sw * sh;
            // Source rows and columns that fold into one 4:2:0 sample.
            let (fy, fx) = match self.chroma {
                Chroma::C420 => (1, 1),
                Chroma::C422 => (2, 1),
                _ => (2, 2),
            };
            for (r, row) in plane.rows_mut().enumerate() {
                for (c, px) in row.iter_mut().enumerate() {
                    let (mut sum, mut count) = (0, 0);
                    for dy in 0..fy {
                        for dx in 0..fx {
                            let (sr, sc) = (r * fy + dy, c * fx + dx);
                            if sr < sh && sc < sw {
                                sum += sample(&self.raw, base + sr * sw + sc);
                                count += 1;
                            }
                        }
                    }
                    *px = to8(sum, count);
                }
            }
        }
        Ok(true)
    }
}

impl Drop for Y4mReader {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            // Stop ffmpeg if the encoder stopped early (--frames).
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The chroma layout and bit depth of a Y4M `C` tag value.
fn colour_space(value: &str) -> io::Result<(Chroma, u32)> {
    let unsupported = || {
        invalid(&format!(
            "unsupported colour space C{value} (4:2:0, 4:2:2, 4:4:4 and mono at 8-16 bits are read)"
        ))
    };
    if let Some(bits) = value.strip_prefix("mono") {
        let depth = if bits.is_empty() {
            8
        } else {
            bits.parse().map_err(|_| unsupported())?
        };
        return check_depth(Chroma::Mono, depth).ok_or_else(unsupported);
    }
    let (layout, rest) = value.split_at(value.len().min(3));
    let chroma = match layout {
        "420" => Chroma::C420,
        "422" => Chroma::C422,
        "444" => Chroma::C444,
        _ => return Err(unsupported()),
    };
    let depth = match rest {
        "" | "jpeg" | "paldv" | "mpeg2" => 8,
        _ => rest
            .strip_prefix('p')
            .and_then(|b| b.parse().ok())
            .ok_or_else(unsupported)?,
    };
    check_depth(chroma, depth).ok_or_else(unsupported)
}

fn check_depth(chroma: Chroma, depth: u32) -> Option<(Chroma, u32)> {
    (8..=16).contains(&depth).then_some((chroma, depth))
}

pub struct Y4mWriter {
    w: Box<dyn Write>,
}

impl Y4mWriter {
    /// Creates `path`, or writes to standard output for `-`.
    pub fn create(path: &Path, info: &Y4mInfo) -> io::Result<Self> {
        let mut w: Box<dyn Write> = if path == Path::new("-") {
            Box::new(BufWriter::new(io::stdout()))
        } else {
            Box::new(BufWriter::new(File::create(path)?))
        };
        writeln!(
            w,
            "YUV4MPEG2 W{} H{} F{}:{} Ip A1:1 C420jpeg",
            info.width, info.height, info.fps_num, info.fps_den
        )?;
        Ok(Self { w })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> io::Result<()> {
        self.w.write_all(b"FRAME\n")?;
        for plane in frame.planes() {
            for row in plane.rows() {
                self.w.write_all(row)?;
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.w.flush()
    }
}

/// One `\n`-terminated header line, or `None` at a clean end of file.
fn read_line(r: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut line = Vec::new();
    r.by_ref().take(4096).read_until(b'\n', &mut line)?;
    if line.is_empty() {
        return Ok(None);
    }
    if line.pop() != Some(b'\n') {
        return Err(invalid("header line too long or truncated"));
    }
    String::from_utf8(line)
        .map(Some)
        .map_err(|_| invalid("header is not ASCII"))
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("y4m: {msg}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(header: &str, body: Vec<u8>) -> Y4mReader {
        let mut data = format!("{header}\nFRAME\n").into_bytes();
        data.extend(body);
        Y4mReader::from_reader(Box::new(io::Cursor::new(data)), None).unwrap()
    }

    #[test]
    fn converts_444_10bit_to_420_8bit() {
        // 2x2 frame: luma 4·(0, 100, 200, 300), U all 512 (= 128), V a 2x2 block averaging 256.
        let mut body = Vec::new();
        for v in [0u16, 400, 800, 1023, 512, 512, 512, 512, 0, 0, 512, 512] {
            body.extend(v.to_le_bytes());
        }
        let mut r = reader("YUV4MPEG2 W2 H2 F30:1 C444p10", body);
        let mut f = r.new_frame();
        assert!(r.read_frame(&mut f).unwrap());
        let [y, u, v] = f.planes();
        assert_eq!(
            y.rows().flatten().copied().collect::<Vec<_>>(),
            [0, 100, 200, 255]
        );
        assert_eq!(u.rows().flatten().copied().collect::<Vec<_>>(), [128]);
        assert_eq!(v.rows().flatten().copied().collect::<Vec<_>>(), [64]);
    }

    #[test]
    fn converts_422_and_mono() {
        // 2x2 4:2:2: chroma is 1 wide, 2 high; the two rows average.
        let body = vec![1, 2, 3, 4, 10, 20, 30, 50];
        let mut r = reader("YUV4MPEG2 W2 H2 F30:1 C422", body);
        let mut f = r.new_frame();
        r.read_frame(&mut f).unwrap();
        let [_, u, v] = f.planes();
        assert_eq!(
            (u.rows().next().unwrap()[0], v.rows().next().unwrap()[0]),
            (15, 40)
        );

        let mut r = reader("YUV4MPEG2 W2 H2 F30:1 Cmono", vec![9, 8, 7, 6]);
        let mut f = r.new_frame();
        r.read_frame(&mut f).unwrap();
        assert_eq!(f.planes()[1].rows().next().unwrap()[0], 128);
        assert!(colour_space("444alpha").is_err());
    }
}
