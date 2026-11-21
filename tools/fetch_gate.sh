#!/usr/bin/env bash
# Fetches the benchmark gate's clips (tools/gate.py): the first 120 frames of standard test
# sequences from https://media.xiph.org/video/derf/ (Xiph, Netflix, JVET class E), scaled with
# Lanczos to 1280x720 (centre crop for 4K's 17:9), 8-bit 4:2:0 Y4M, in videos/gate/. ffmpeg stops
# reading after the frames it needs, so only those are downloaded (~9 GB in all).
#
#     bash tools/fetch_gate.sh [-j 3]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=videos/gate
mkdir -p "$OUT"
BASE=https://media.xiph.org/video/derf/y4m
JOBS=3
[ "${1:-}" = "-j" ] && JOBS=$2
CLIPS=(
  "aerial Netflix_Aerial_4096x2160_60fps_8bit_420"            # drone
  "driving Netflix_DrivingPOV_4096x2160_60fps_8bit_420"       # vehicle FPV
  "coaster Netflix_RollerCoaster_4096x2160_60fps_8bit_420"    # fast FPV
  "fourpeople FourPeople_1280x720_60"                         # videoconference
  "kristen KristenAndSara_1280x720_60"                        # talking heads
  "crowdrun crowd_run_1080p50"                                # dense sports motion
  "boxing Netflix_BoxingPractice_4096x2160_60fps_8bit_420"    # indoor sports
  "ducks ducks_take_off_1080p50"                              # water texture
  "oldtown old_town_cross_1080p50"                            # slow pan, detail
  "crosswalk Netflix_Crosswalk_4096x2160_60fps_8bit_420"      # static camera, people
)
fetch() {
  local name=$1 src=$2
  [ -s "$OUT/$name.y4m" ] && return
  echo "fetching $name ($src)"
  ffmpeg -nostdin -loglevel error -y -i "$BASE/$src.y4m" -frames:v 120 \
    -vf "scale=-2:720:flags=lanczos,crop=1280:720" -pix_fmt yuv420p -strict -1 \
    -f yuv4mpegpipe "$OUT/$name.y4m.part" && mv "$OUT/$name.y4m.part" "$OUT/$name.y4m"
}
export -f fetch; export OUT BASE
printf '%s\n' "${CLIPS[@]}" | xargs -P "$JOBS" -L 1 bash -c 'fetch "$0" "$1"'
ls -la "$OUT"
