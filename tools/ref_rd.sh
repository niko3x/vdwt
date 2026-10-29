#!/usr/bin/env bash
# Rate-distortion points from reference codecs via ffmpeg, for comparison with `vdwt rd`.
# Prints: codec, setting, kbps (at the clip's frame rate), PSNR-Y, PSNR-YUV(6:1:1).
#
#   tools/ref_rd.sh CLIP.y4m [FRAMES] [WORKDIR]
#
# MODE=intra (default): every frame intra; x264, x265, JPEG 2000, and SVT-AV1 when AV1_CRFS is
# set.
# MODE=inter: low-delay P-frames only, no B-frames, no lookahead, one intra
# frame at the start: x264 ultrafast, veryfast and medium, and x265 medium.
set -euo pipefail
clip=$1
frames=${2:-10}
work=${3:-$(mktemp -d)}
mkdir -p "$work"
vdwt=${VDWT:-target/release/vdwt}
fps=$(head -c 200 "$clip" | grep -ao ' F[0-9]*:[0-9]*' | head -1 | tr -d ' F')
src="$work/src.y4m"
ffmpeg -loglevel error -y -i "$clip" -frames:v "$frames" "$src"

point() { # codec setting encoded-file
    local codec=$1 setting=$2 file=$3
    local bytes; bytes=$(stat -c %s "$file")
    ffmpeg -loglevel error -y -i "$file" -pix_fmt yuv420p "$work/dec.y4m"
    local avg; avg=$("$vdwt" psnr "$src" "$work/dec.y4m" | tail -1)
    local y yuv
    y=$(awk '{print $3}' <<<"$avg"); yuv=$(awk '{print $9}' <<<"$avg")
    awk -v c="$codec" -v s="$setting" -v b="$bytes" -v n="$frames" -v f="$fps" -v y="$y" -v yuv="$yuv" \
        'BEGIN { split(f, r, ":"); printf "%-10s %-8s %10.1f %8s %9s\n", c, s, b*8/n*r[1]/r[2]/1000, y, yuv }'
}

printf "%-10s %-8s %10s %8s %9s\n" codec setting kbps PSNR-Y PSNR-YUV
if [ "${MODE:-intra}" = inter ]; then
    for preset in ultrafast veryfast medium; do
        for qp in ${X264_QPS:-24 30 36 42 48}; do
            ffmpeg -loglevel error -y -i "$src" -c:v libx264 -preset "$preset" -tune psnr,zerolatency \
                -x264-params "keyint=infinite:bframes=0:qp=$qp" "$work/x264.mkv"
            point "x264-$preset" "qp$qp" "$work/x264.mkv"
        done
    done
    for qp in ${X265_QPS:-24 30 36 42 48}; do
        ffmpeg -loglevel error -y -i "$src" -c:v libx265 -preset medium -tune psnr \
            -x265-params "keyint=-1:bframes=0:rc-lookahead=0:qp=$qp:log-level=error" "$work/x265.mkv"
        point x265-medium "qp$qp" "$work/x265.mkv"
    done
    exit 0
fi
for qp in ${X264_QPS:-22 28 34 40 46}; do
    ffmpeg -loglevel error -y -i "$src" -c:v libx264 -preset medium -tune psnr \
        -x264-params "keyint=1:qp=$qp" "$work/x264.mkv"
    point x264 "qp$qp" "$work/x264.mkv"
done
for qp in ${X265_QPS:-22 28 34 40 46}; do
    ffmpeg -loglevel error -y -i "$src" -c:v libx265 -preset medium -tune psnr \
        -x265-params "keyint=1:qp=$qp:log-level=error" "$work/x265.mkv"
    point x265 "qp$qp" "$work/x265.mkv"
done
for q in ${J2K_Q:-2 5 10 20 40}; do
    ffmpeg -loglevel error -y -i "$src" -c:v jpeg2000 -pred 1 -q:v "$q" "$work/j2k.mkv"
    point j2k-5/3 "q$q" "$work/j2k.mkv"
done
if [ -n "${AV1_CRFS:-}" ]; then
    for crf in $AV1_CRFS; do
        SVT_LOG=0 ffmpeg -loglevel error -y -i "$src" -c:v libsvtav1 -preset 6 -g 1 -crf "$crf" "$work/av1.mkv" 2>/dev/null
        point av1-svt "crf$crf" "$work/av1.mkv"
    done
fi
