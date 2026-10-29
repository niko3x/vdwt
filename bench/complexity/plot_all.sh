#!/bin/bash
# Regenerate every published Pareto chart (encode and decode) from the current hybrid tables.
# Decode timings live in bench/complexity/decode_samtime.txt and decode_drone.txt; refresh them
# with `python3 bench/complexity/decode_bench.py` after re-running the hybrid benchmark.
set -euo pipefail
cd "$(dirname "$0")/../.."

for kind in samtime drone; do
    test -s "bench/complexity/decode_${kind}.txt" || {
        echo "missing bench/complexity/decode_${kind}.txt" >&2
        exit 1
    }
    python3 bench/complexity/plot_hybrid.py "$kind" "videos/pareto_${kind}_encode.svg"
    python3 bench/complexity/plot_hybrid.py "$kind" "videos/pareto_${kind}_clean.svg" clean
    python3 bench/complexity/plot_hybrid.py "$kind" "videos/pareto_${kind}_decode.svg" decode
    python3 bench/complexity/plot_hybrid.py "$kind" "videos/pareto_${kind}_decode_clean.svg" clean decode
    cp "videos/pareto_${kind}_encode.svg" "videos/pareto_${kind}_hybrid.svg"
    cp "videos/pareto_${kind}_clean.svg" "videos/pareto_${kind}_clean_hybrid.svg"
done

python3 bench/complexity/plot_postfilter.py >/dev/null
if [[ -s bench/complexity/tassie_pf_encode.txt && -s bench/complexity/tassie_pf_decode.txt ]]; then
    python3 bench/complexity/plot_tasmania_pf.py >/dev/null
    python3 bench/complexity/plot_tasmania_vmaf.py >/dev/null
fi
if [[ -s bench/complexity/tassie_pf_encode_psnr.txt ]]; then
    python3 bench/complexity/plot_tasmania_psnr.py >/dev/null
fi
if [[ -s bench/complexity/auto_vs_fixed.txt ]]; then
    python3 bench/complexity/plot_auto_vs_fixed.py >/dev/null
    rsvg-convert -w 1960 "videos/auto_vs_fixed_hybrid.svg" -o "videos/auto_vs_fixed_hybrid.png"
fi
if [[ -s bench/complexity/postfilter_off.txt && -s bench/complexity/auto_vs_fixed.txt ]]; then
    python3 bench/complexity/plot_postfilter_onoff.py >/dev/null
    rsvg-convert -w 2000 "videos/postfilter_onoff_pareto.svg" -o "videos/postfilter_onoff_pareto.png"
fi
if [[ -s bench/complexity/samtime_pf_encode_hybrid.txt && -s bench/complexity/samtime_pf_encode_psnr.txt ]]; then
    python3 bench/complexity/plot_samtime_pf.py >/dev/null
fi
if [[ -s bench/complexity/filter_prev_now.csv ]]; then
    python3 bench/complexity/plot_filter_prev_now.py >/dev/null
fi
if [[ -s bench/complexity/drone_filter_sweep.csv && -s target/bench_tmp/filter_sweep/crop_forest_q50_none.png ]]; then
    python3 bench/complexity/plot_forest_filter_sheet.py >/dev/null
fi
if [[ -s bench/complexity/drone_filter_sweep.csv ]]; then
    python3 bench/complexity/plot_drone_filter_curve.py >/dev/null
fi
if [[ -s bench/complexity/filter_strength_sweep.csv && -s target/bench_tmp/filter_sweep/samcrop_a720_none.png ]]; then
    python3 bench/complexity/plot_samtime_filter_sheet.py >/dev/null
fi
if [[ -s bench/complexity/filter_strength_sweep.csv ]]; then
    python3 bench/complexity/plot_samtime_filter_curve.py >/dev/null
fi

for f in pareto_samtime_encode pareto_drone_encode pareto_samtime_clean pareto_drone_clean \
         pareto_samtime_decode pareto_drone_decode pareto_samtime_decode_clean pareto_drone_decode_clean \
         postfilter_vs_x265_stock postfilter_vs_x265_unsharp; do
    rsvg-convert -w 2000 "videos/${f}.svg" -o "videos/${f}.png"
done
if [[ -s videos/tasmania_pf_encode.svg ]]; then
    for f in tasmania_pf_encode tasmania_pf_decode tasmania_pf_encode_psnr tasmania_pf_decode_psnr \
             tasmania_pf_vmaf_encode tasmania_pf_vmaf_decode \
             samtime_pf_hybrid_encode samtime_pf_hybrid_decode samtime_pf_psnr_encode samtime_pf_psnr_decode; do
        [[ -s "videos/${f}.svg" ]] && rsvg-convert -w 2000 "videos/${f}.svg" -o "videos/${f}.png"
    done
fi
[[ -s videos/samtime_filter_prev_vs_now.svg ]] && rsvg-convert -w 2000 videos/samtime_filter_prev_vs_now.svg -o videos/samtime_filter_prev_vs_now.png
[[ -s videos/forest_filter_strength_sheet.svg ]] && rsvg-convert -w 3540 videos/forest_filter_strength_sheet.svg -o videos/forest_filter_strength_sheet.png
[[ -s videos/drone_filter_strength_curve.svg ]] && rsvg-convert -w 2000 videos/drone_filter_strength_curve.svg -o videos/drone_filter_strength_curve.png
[[ -s videos/samtime_filter_strength_sheet.svg ]] && rsvg-convert -w 3540 videos/samtime_filter_strength_sheet.svg -o videos/samtime_filter_strength_sheet.png
[[ -s videos/samtime_filter_strength_curve.svg ]] && rsvg-convert -w 2000 videos/samtime_filter_strength_curve.svg -o videos/samtime_filter_strength_curve.png
echo "wrote the standard chart set (plus Tasmania post-filter charts when their data exists)"
