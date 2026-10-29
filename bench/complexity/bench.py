#!/usr/bin/env python3
"""Wall-clock RD sweep in the chart's format: clip encoder qp kbps psnr_y ms/frame.

    bench.py BIN PREFIX OUT [CLIP...]

Runs 20 frames at qp 42/50/58, 4 threads, all five tiers. The wall time includes process
startup, frame reading and the encoder's startup probes, exactly as the published charts.
"""
import subprocess
import sys
import time

CLIPS = [
    ("a720", "videos/samtime/a720.y4m"),
    ("b720", "videos/samtime/b720.y4m"),
    ("canyon", "videos/nr/clips/canyon.y4m"),
    ("forest", "videos/nr/clips/forest.y4m"),
    ("tasmania", "videos/nr/clips/tasmania.y4m"),
]
TIERS = [
    ("realtime", ["--realtime"]),
    ("light", ["--tier", "light"]),
    ("medium", ["--tier", "medium"]),
    ("heavy", ["--tier", "heavy"]),
    ("ultraheavy", ["--tier", "ultraheavy"]),
]
QPS = [42, 50, 58]


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    binary, prefix, out = sys.argv[1:4]
    clips = [(n, p) for n, p in CLIPS if len(sys.argv) <= 4 or n in sys.argv[4:]]
    rows = []
    for name, path in clips:
        for tier, opts in TIERS:
            for qp in QPS:
                start = time.perf_counter()
                p = subprocess.run(
                    [binary, "encode", path, "/tmp/bench.vdwt", "--qp", str(qp),
                     "--frames", "20", "--threads", "4", *opts],
                    capture_output=True, text=True, check=True,
                )
                ms = (time.perf_counter() - start) * 1000.0 / 20.0
                kbps = psnr = 0.0
                for line in p.stdout.splitlines():
                    if " kbps" in line and ":" in line and "frames" in line:
                        kbps = float(line.split(": ")[1].split()[0])
                    if line.startswith("PSNR") and " Y " in line:
                        psnr = float(line.split()[2])
                rows.append((name, prefix + tier, qp, kbps, psnr, ms))
                print(f"{name} {tier} q{qp}: {kbps:.1f} kbps {psnr:.2f} dB {ms:.2f} ms", flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]:.1f} {r[4]:.3f} {r[5]:.2f}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
