#!/bin/bash
# Rebuild the disjoint A21 training corpus. None of these frames is in the evaluation clips
# (a720, b720, canyon, forest, tasmania); gen_a/gen_b and work60 are deliberately excluded
# because they are copies of samtime/tasmania frames.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=bench/contexts/corpus
mkdir -p "$OUT"

# Madeira drone source, three one-second segments (60 fps).
for spec in a:0 b:3 c:6; do
    tag=${spec%%:*}
    start=${spec##*:}
    ffmpeg -hide_banner -loglevel error -y -ss "$start" -i videos/madeira_720p60.y4m \
        -frames:v 60 -pix_fmt yuv420p -f yuv4mpegpipe "$OUT/madeira_$tag.y4m"
done

# POV bike/event footage, 4K30 source downscaled (different motion and texture).
test -s target/yt_demo/src.mp4 || {
    echo "missing target/yt_demo/src.mp4 (the POV clip); see EXPERIMENTS.md A21" >&2
    exit 1
}
ffmpeg -hide_banner -loglevel error -y -ss 2 -i target/yt_demo/src.mp4 -vf scale=1280:720 \
    -frames:v 60 -pix_fmt yuv420p -f yuv4mpegpipe "$OUT/bike.y4m"

# Aerial coastline, 20 frames at 29.97 fps (not canyon or tasmania).
cp videos/nr/clips/coast.y4m "$OUT/coast.y4m"

ls -l "$OUT"
