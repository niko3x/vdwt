#!/bin/bash
# nr.sh CLIP ENC QP: unrestricted encoders (B-frames, lookahead), single thread; appends
# "CLIP ENC QP kbps psnrY cpu_ms_per_frame" to rd.txt next to this script
W=$(dirname $(readlink -f $0)); B=$W/../../target/release/vdwt
clip=$1; enc=$2; p=$3; src=$W/clips/$clip.y4m; t=$W/tmp/${clip}_${enc}_$p; mkdir -p $W/tmp
export LC_ALL=C; TIMEFORMAT=%U+%S
case $enc in
  x265-*) cpu=$( { time x265 --input $src --preset ${enc#x265-} --tune psnr --qp $p --pools 1 --frame-threads 1 -o $t.hevc >/dev/null 2>&1; } 2>&1 ); out=$t.hevc;;
  vdwt-realtime) cpu=$( { time $B encode $src $t.vdwt --qp $p --realtime --threads 1 >/dev/null 2>&1; } 2>&1 ); out=$t.vdwt;;
  vdwt-*) cpu=$( { time $B encode $src $t.vdwt --qp $p --tier ${enc#vdwt-} --threads 1 >/dev/null 2>&1; } 2>&1 ); out=$t.vdwt;;
esac
case ${out##*.} in
  vdwt) $B decode $out $t.y4m >/dev/null 2>&1;;
  *) ffmpeg -loglevel error -y -i $out -pix_fmt yuv420p -f yuv4mpegpipe -strict -1 $t.y4m;;
esac
read fn fd <<<$(head -c 100 $src | grep -oE "F[0-9]+:[0-9]+" | tr -d F | tr : ' ')
kbps=$(python3 -c "import os;print(f'{os.path.getsize(\"$out\")*8*$fn/$fd/20/1000:.1f}')")
y=$($B psnr $src $t.y4m 2>/dev/null | awk '/average/{print $3}')
echo "$clip $enc $p $kbps $y $(python3 -c "print(f'{($cpu)*1000/20:.1f}')")" >> $W/rd.txt
rm -f $out $t.y4m
