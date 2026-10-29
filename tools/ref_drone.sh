#!/usr/bin/env bash
# x264 at full strength within drone-link constraints, for comparison with `vdwt rd`:
# I and P frames only (no B-frames), no lookahead of any kind (rc-lookahead 0, sync-lookahead 0,
# so no mbtree either), one intra frame at the start, single thread. Everything else is the
# preset's default: multiple reference frames, trellis, sub-pixel RDO, all partitions.
# Prints: codec, setting, kbps (at the clip's frame rate), PSNR-Y, PSNR-YUV(6:1:1), encode fps.
#
#   tools/ref_drone.sh CLIP.y4m [FRAMES] [WORKDIR]
#
# PRESETS (default "ultrafast veryfast medium slow veryslow") and X264_QPS select the points.
set -euo pipefail
clip=$1
frames=${2:-60}
work=${3:-$(mktemp -d)}
mkdir -p "$work"
vdwt=${VDWT:-target/release/vdwt}
fps=$(head -c 200 "$clip" | grep -ao ' F[0-9]*:[0-9]*' | head -1 | tr -d ' F')
src="$work/src.y4m"
ffmpeg -loglevel error -y -i "$clip" -frames:v "$frames" -strict -1 "$src"

printf "%-15s %-8s %10s %8s %9s %8s\n" codec setting kbps PSNR-Y PSNR-YUV enc-fps
for preset in ${PRESETS:-ultrafast veryfast medium slow veryslow}; do
    for qp in ${X264_QPS:-20 26 32 38 44}; do
        start=$(date +%s.%N)
        x264 --quiet --preset "$preset" --tune psnr --bframes 0 --rc-lookahead 0 \
            --sync-lookahead 0 --keyint infinite --threads 1 --qp "$qp" \
            -o "$work/x264.264" "$src" 2>/dev/null
        end=$(date +%s.%N)
        ffmpeg -loglevel error -y -i "$work/x264.264" -pix_fmt yuv420p "$work/dec.y4m"
        bytes=$(stat -c %s "$work/x264.264")
        avg=$("$vdwt" psnr "$src" "$work/dec.y4m" | tail -1)
        y=$(awk '{print $3}' <<<"$avg"); yuv=$(awk '{print $9}' <<<"$avg")
        awk -v c="x264-$preset" -v s="qp$qp" -v b="$bytes" -v n="$frames" -v f="$fps" -v y="$y" \
            -v yuv="$yuv" -v t0="$start" -v t1="$end" \
            'BEGIN { split(f, r, ":"); printf "%-15s %-8s %10.1f %8s %9s %8.1f\n", c, s, b*8/n*r[1]/r[2]/1000, y, yuv, n/(t1-t0) }'
    done
done
