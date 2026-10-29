#!/usr/bin/env python3
"""Tasmania-only Pareto with the custom band-pass applied to both VDWT and x265."""
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,"bench/complexity")
from plot import bd_rate
ROOT=Path(__file__).resolve().parents[2]; BIN=ROOT/"target/release/vdwt"
SRC=ROOT/"videos/nr/clips/tasmania.y4m"; TMP=ROOT/"target/bench_tmp/tassie_pf"; TMP.mkdir(parents=True,exist_ok=True)
QPS={"vdwt":[42,50,58],"x265":[36,42,48]}
TIERS={"realtime":["--realtime"],"light":["--tier","light"],"medium":["--tier","medium"],
       "heavy":["--tier","heavy"],"ultraheavy":["--tier","ultraheavy"]}
PRESETS=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
def run(c): return subprocess.run(c,check=True,capture_output=True,text=True)
def metrics(ref,dist,tag):
    vm=TMP/f"{tag}.json"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(dist),"-i",str(ref),
         "-lavfi",f"libvmaf=log_fmt=json:log_path={vm}","-frames:v","20","-f","null","-"])
    v=json.load(open(vm))["pooled_metrics"]["vmaf"]["mean"]
    p=run([str(BIN),"psnr",str(ref),str(dist)]).stdout
    m=re.search(r"average\s+Y\s+[\d.]+\s+U\s+[\d.]+\s+V\s+[\d.]+\s+YUV\s+([\d.]+)",p)
    os.remove(vm); return v,float(m.group(1))
def post(inp,out):
    run([str(BIN),"postfilter",str(inp),str(out),"--postfilter","band","--postfilter-strength","2.0"])
def dec_time_vdwt(stream):
    ts=[]
    for _ in range(2):
        t=time.perf_counter(); run([str(BIN),"decode",str(stream),"/dev/null","--threads","4",
                                    "--postfilter","band","--postfilter-strength","2.0"]); ts.append(time.perf_counter()-t)
    return min(ts)*1000/60
def dec_time_x265(hevc):
    ts=[]
    for _ in range(2):
        dec=TMP/"dec60.y4m"; filt=TMP/"fil60.y4m"
        t=time.perf_counter()
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(hevc),"-frames:v","60",
             "-pix_fmt","yuv420p","-strict","-1",str(dec)])
        t1=time.perf_counter()
        run([str(BIN),"postfilter",str(dec),"/dev/null","--postfilter","band","--postfilter-strength","2.0"])
        t2=time.perf_counter()
        ts.append((t1-t)+(t2-t1))
        for p in (dec,filt): p.unlink(missing_ok=True)
    return min(ts)*1000/60
def main():
    ref=TMP/"ref20.y4m"; ref60=TMP/"ref60.y4m"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(SRC),"-frames:v","20","-pix_fmt","yuv420p","-strict","-1",str(ref)])
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(SRC),"-frames:v","60","-pix_fmt","yuv420p","-strict","-1",str(ref60)])
    enc_rows=[]; dec_rows=[]
    for tier,opts in TIERS.items():
        for qp in QPS["vdwt"]:
            # quality, 20 frames
            stream=TMP/f"v_{tier}_{qp}.vdwt"; rec=TMP/f"v_{tier}_{qp}.y4m"
            t=time.perf_counter()
            run([str(BIN),"encode",str(SRC),str(stream),"--qp",str(qp),"--frames","20","--threads","4",*opts,
                 "--recon",str(rec),"--postfilter","band","--postfilter-strength","2.0"])
            # Encode-only timing without the recon/post-filter write.
            t=time.perf_counter()
            run([str(BIN),"encode",str(SRC),TMP/"t.vdwt","--qp",str(qp),"--frames","20","--threads","4",*opts])
            enc_ms=(time.perf_counter()-t)*1000/20
            kbps=stream.stat().st_size*8*60/20/1000
            v,y=metrics(ref,rec,f"v_{tier}_{qp}")
            enc_rows.append(f"tasmania vdwt-{tier} {qp} {kbps:.1f} {v:.4f} {y:.4f} {enc_ms:.2f}\n")
            print("enc vdwt",tier,qp,round(kbps,1),round(v,2),round(y,2),round(enc_ms,1),flush=True)
            for p in (stream,rec,TMP/"t.vdwt"): p.unlink(missing_ok=True)
            # 60-frame timing
            stream60=TMP/f"v60_{tier}_{qp}.vdwt"
            run([str(BIN),"encode",str(SRC),str(stream60),"--qp",str(qp),"--frames","60","--threads","4",*opts])
            dec_rows.append(f"tasmania vdwt-{tier} {qp} {dec_time_vdwt(stream60):.3f}\n")
            print("dec vdwt",tier,qp,dec_rows[-1].split()[-1],flush=True)
            stream60.unlink(missing_ok=True)
    for preset in PRESETS:
        for qp in QPS["x265"]:
            hevc=TMP/f"x_{preset}_{qp}.hevc"; dec=TMP/f"x_{preset}_{qp}.y4m"; filt=TMP/f"x_{preset}_{qp}_f.y4m"
            t=time.perf_counter()
            run(["x265","--input",str(ref),"--preset",preset,"--tune","psnr","--qp",str(qp),
                 "--pools","4","--frame-threads","1","-o",str(hevc)])
            enc_ms=(time.perf_counter()-t)*1000/20
            run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(hevc),"-frames:v","20",
                 "-pix_fmt","yuv420p","-strict","-1",str(dec)])
            post(dec,filt)
            kbps=hevc.stat().st_size*8*60/20/1000
            v,y=metrics(ref,filt,f"x_{preset}_{qp}")
            enc_rows.append(f"tasmania x265-{preset} {qp} {kbps:.1f} {v:.4f} {y:.4f} {enc_ms:.2f}\n")
            print("enc x265",preset,qp,round(kbps,1),round(v,2),round(y,2),round(enc_ms,1),flush=True)
            # 60-frame timing
            hevc60=TMP/f"x60_{preset}_{qp}.hevc"
            t=time.perf_counter()
            run(["x265","--input",str(ref60),"--preset",preset,"--tune","psnr","--qp",str(qp),
                 "--pools","4","--frame-threads","1","-o",str(hevc60)])
            dec_rows.append(f"tasmania x265-{preset} {qp} {dec_time_x265(hevc60):.3f}\n")
            print("dec x265",preset,qp,dec_rows[-1].split()[-1],flush=True)
            for p in (hevc,dec,filt,hevc60): p.unlink(missing_ok=True)
    Path("bench/complexity/tassie_pf_encode.txt").write_text("".join(enc_rows))
    Path("bench/complexity/tassie_pf_decode.txt").write_text("".join(dec_rows))
if __name__=="__main__": main()
