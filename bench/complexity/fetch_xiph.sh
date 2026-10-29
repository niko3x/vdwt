#!/usr/bin/env bash
# Fetches the public benchmark clips: the first 20 frames of each Xiph test sequence
# (https://media.xiph.org/video/derf/), scaled to 720p height and centre-cropped to 1280x720, as
# 8-bit 4:2:0 Y4M in videos/nr/clips/. Only the frames needed are downloaded.
#
#     bash bench/complexity/fetch_xiph.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=videos/nr/clips
mkdir -p "$OUT"
BASE=https://media.xiph.org/video/derf/y4m
declare -A CLIPS=(
  [x_aerial]=Netflix_Aerial_4096x2160_60fps_8bit_420
  [x_oldtown]=old_town_cross_1080p50
  [x_stockholm]=720p5994_stockholm_ter
  [x_vidyo1]=vidyo1_720p_60fps
  [x_vidyo3]=vidyo3_720p_60fps
  [x_vidyo4]=vidyo4_720p_60fps
  [x_parkrun]=720p50_parkrun_ter
  [x_shields]=720p50_shields_ter
  [x_crowdrun]=crowd_run_1080p50
  [x_parkjoy]=park_joy_1080p50
)
for name in "${!CLIPS[@]}"; do
  [ -s "$OUT/$name.y4m" ] && continue
  echo "fetching $name (${CLIPS[$name]})"
  ffmpeg -nostdin -loglevel error -y -i "$BASE/${CLIPS[$name]}.y4m" -frames:v 20 \
    -vf "scale=-2:720:flags=lanczos,crop=1280:720" -pix_fmt yuv420p -strict -1 "$OUT/$name.y4m"
done
