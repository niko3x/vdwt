#!/usr/bin/env python3
"""Samtime-only post-filtered encode/decode Pareto, hybrid and PSNR-Y."""
import math, sys
from collections import defaultdict
import numpy as np
ROOT="bench/complexity"
CLIPS=["a720","b720"]
VDWT=["realtime","light","medium","heavy","ultraheavy"]
X265=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]

def load_hybrid():
    d=defaultdict(lambda: defaultdict(list))
    for line in open(f"{ROOT}/samtime_pf_encode_hybrid.txt"):
        c,e,q,k,v,y,m=line.split(); d[e][c].append((float(k),float(v),float(y),float(m)))
    return d

def load_psnr():
    d=defaultdict(lambda: defaultdict(list))
    for line in open(f"{ROOT}/samtime_pf_encode_psnr.txt"):
        c,e,q,k,y,m=line.split(); d[e][c].append((float(k),float(y),float(m)))
    return d

def load_dec():
    d=defaultdict(lambda: defaultdict(list))
    for line in open(f"{ROOT}/samtime_pf_decode.txt"):
        c,e,q,ms=line.split(); d[e][c].append(float(ms))
    return d

def bd(ref,test,idx):
    a=np.array(sorted((row[idx],math.log(row[0])) for row in ref if row[0]>0))
    b=np.array(sorted((row[idx],math.log(row[0])) for row in test if row[0]>0))
    lo=max(a[0,0],b[0,0]); hi=min(a[-1,0],b[-1,0])
    if hi-lo<0.5: return None
    xs=np.linspace(lo,hi,201)
    ya=np.interp(xs,a[:,0],a[:,1]); yb=np.interp(xs,b[:,0],b[:,1])
    return 100*(math.exp(float(np.mean(yb-ya)))-1)

def scores(d,kind):
    out={}
    ref=d["x265-medium"]
    for e,per in d.items():
        vals=[]
        for c in CLIPS:
            if c not in per or c not in ref: continue
            if kind=="hybrid":
                a=bd(ref[c],per[c],1); b=bd(ref[c],per[c],2)
                if a is None or b is None: continue
                vals.append(0.7*(-a)+0.3*(-b))
            else:
                a=bd(ref[c],per[c],1)
                if a is None: continue
                vals.append(-a)
        if vals: out[e]=sum(vals)/len(vals)
    return out

def times(d):
    return {e:sum(m for c in per.values() for *_,m in c)/sum(len(c) for c in per.values()) for e,per in d.items()}

def draw(path,kind,mode,sc,tm,dec):
    W,H=2000,1400; left,right,top,bottom=250,1915,390,1215
    if mode=="encode": xmin,xmax=8.0,600.0; xticks=(10,20,50,100,200,500); xlabel="Encode time per frame (ms, log; 720p, 4 threads, x86)"
    else: xmin,xmax=3.5,10.0; xticks=(5,10); xlabel="Decode time per frame (ms, log; 720p, 4 threads, x86; filter included)"
    ys=list(sc.values()); ymin=math.floor((min(ys)-5)/10)*10; ymax=math.ceil((max(ys)+5)/10)*10
    px=lambda x:left+(right-left)*(math.log10(x)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))
    py=lambda y:bottom-(bottom-top)*(y-ymin)/(ymax-ymin)
    font="font-family='Inter, Helvetica, Arial, sans-serif'"
    metric="Hybrid" if kind=="hybrid" else "PSNR-Y"
    title=f"Samtime: post-filtered {metric} {mode} Pareto"
    sub=("custom band-pass 2.0 applied to both decoders; BD vs x265 medium+filter; "
         "20-frame quality, 60-frame timing")
    s=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
       f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
       f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
       f"<text x='70' y='92' {font} font-size='44' font-weight='700' fill='#222'>{title}</text>",
       f"<text x='70' y='137' {font} font-size='23' fill='#555'>{sub}</text>",
       f"<text x='70' y='171' {font} font-size='21' fill='#777'>Positive = fewer bits than x265 medium+filter at equal {metric} quality</text>",
       f"<circle cx='262' cy='270' r='16' fill='#1A237E' stroke='#222' stroke-width='2.5'/>",
       f"<text x='287' y='280' {font} font-size='27' fill='#222'>VDWT + custom band</text>",
       f"<circle cx='585' cy='270' r='16' fill='#2E8B57' stroke='#222' stroke-width='2.5'/>",
       f"<text x='610' y='280' {font} font-size='27' fill='#222'>x265 + custom band</text>"]
    s += [f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>",
          f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>"]
    for y in range(int(ymin),int(ymax)+1,10):
        yy=py(y); s.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD' stroke-width='1'/>")
        s.append(f"<text x='{left-22}' y='{yy+8:.1f}' {font} font-size='24' text-anchor='end' fill='#333'>{y:+d}%</text>")
    for x in xticks:
        xx=px(x); s.append(f"<line x1='{xx:.1f}' y1='{top}' x2='{xx:.1f}' y2='{bottom}' stroke='#EEE' stroke-width='1'/>")
        s.append(f"<text x='{xx:.1f}' y='{bottom+42}' {font} font-size='24' text-anchor='middle' fill='#333'>{x}</text>")
    s.append(f"<text x='{(left+right)/2:.0f}' y='{bottom+100}' {font} font-size='26' text-anchor='middle' fill='#333'>{xlabel}</text>")
    s.append(f"<text x='55' y='{(top+bottom)/2:.0f}' {font} font-size='26' text-anchor='middle' fill='#333' transform='rotate(-90 55 {(top+bottom)/2:.0f})'>{metric} bitrate saving vs x265 medium+filter (%)</text>")
    s.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{right}' y2='{py(0):.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")
    def family(names,colour):
        pts=[]
        for n in names:
            if n not in sc: continue
            x=tm[n] if mode=="encode" else dec[n]
            pts.append((n,sc[n],x))
        pts.sort(key=lambda q:q[2]); coords=[(px(x),py(y),n) for n,y,x in pts]
        s.append("<polyline points='"+" ".join(f"{x:.1f},{y:.1f}" for x,y,_ in coords)+f"' fill='none' stroke='{colour}' stroke-width='5' stroke-linejoin='round'/>")
        for x,y,n in coords:
            s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='{colour}' stroke='#222' stroke-width='2.5'/>")
    family([f"x265-{p}" for p in X265],"#2E8B57"); family([f"vdwt-{t}" for t in VDWT],"#1A237E")
    halo="paint-order='stroke' stroke='#FAF9F5' stroke-width='7' stroke-linejoin='round'"
    def label(n,dx,dy,colour):
        if n not in sc: return
        x=px(tm[n] if mode=="encode" else dec[n]); y=py(sc[n])
        s.append(f"<text x='{x+dx:.1f}' y='{y+dy:.1f}' {font} font-size='26' fill='{colour}' text-anchor='middle' {halo}>{n.replace('vdwt-','').replace('x265-','')}</text>")
    if mode=="encode":
        pos={"realtime":(-50,34),"light":(40,-28),"medium":(40,30),"heavy":(45,-30),"ultraheavy":(-50,-30)}
        for t in VDWT: label(f"vdwt-{t}",*pos[t],"#1A237E")
        xpos={"ultrafast":(-40,-30),"medium":(36,34),"slow":(38,-28),"veryslow":(-35,34)}
        for p in xpos: label(f"x265-{p}",*xpos[p],"#2E8B57")
    else:
        pos={"realtime":(40,34),"light":(-50,34),"medium":(-55,-30),"heavy":(45,34),"ultraheavy":(-50,-30)}
        for t in VDWT: label(f"vdwt-{t}",*pos[t],"#1A237E")
        xpos={"ultrafast":(40,34),"medium":(40,-30),"slow":(40,-30),"veryslow":(50,-30)}
        for p in xpos: label(f"x265-{p}",*xpos[p],"#2E8B57")
    s.append("</svg>"); open(path,"w").write("\n".join(s))

if __name__=="__main__":
    hyb=load_hybrid(); psy=load_psnr(); dec=load_dec(); tm=times(hyb)
    for kind,d in [("hybrid",hyb),("psnr",psy)]:
        sc=scores(d,kind)
        print(kind,{k:round(v,2) for k,v in sc.items()})
        dt={k:sum(sum(x) for x in v.values())/sum(len(x) for x in v.values()) for k,v in dec.items()}
        draw(f"videos/samtime_pf_{kind}_encode.svg",kind,"encode",sc,tm,dt)
        draw(f"videos/samtime_pf_{kind}_decode.svg",kind,"decode",sc,tm,dt)
