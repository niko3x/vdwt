#!/usr/bin/env python3
"""Grouped bars: VDWT custom post-filter vs x265 stock and x265+normal unsharp."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "bench/complexity")
from plot import bd_rate
ROOT=Path(__file__).resolve().parents[2]
CLIPS=["a720","b720","canyon","forest","tasmania"]
def load(*paths):
    d=defaultdict(lambda: defaultdict(list))
    for p in paths:
        for line in open(p):
            c,e,q,k,v,y,m=line.split(); d[e][c].append((float(k),float(v),float(y),float(m)))
    return d
def score(test,ref):
    a=bd_rate(sorted((b,v) for b,v,y,m in ref),sorted((b,v) for b,v,y,m in test))
    b=bd_rate(sorted((b,y) for b,v,y,m in ref),sorted((b,y) for b,v,y,m in test))
    return 0.7*(-a)+0.3*(-b)
d=load(ROOT/"bench/complexity/hybrid_samtime.txt",ROOT/"bench/complexity/hybrid_drone.txt",
       ROOT/"bench/complexity/hybrid_postfilter_custom.txt",ROOT/"bench/complexity/hybrid_x265pf060.txt")
stock={c:score(d["vdwt-ultraheavy"][c],d["x265-medium"][c]) for c in CLIPS}
custom={c:score(d["vdwt-ultraheavy-band"][c],d["x265-medium"][c]) for c in CLIPS}
xpf={c:score(d["x265-medium-pf"][c],d["x265-medium"][c]) for c in CLIPS}
custom_vs_pf={c:score(d["vdwt-ultraheavy-band"][c],d["x265-medium-pf"][c]) for c in CLIPS}
stock_vs_pf={c:score(d["vdwt-ultraheavy"][c],d["x265-medium-pf"][c]) for c in CLIPS}

def draw(path,title,sub,series,ymin,ymax):
    # series: list of (name, colour, [value per clip], mean)
    W,H=2000,1200; left,right,top,bottom=230,1920,300,1000
    px=lambda x:left+(right-left)*x
    py=lambda y:bottom-(bottom-top)*(y-ymin)/(ymax-ymin)
    font="font-family='Inter, Helvetica, Arial, sans-serif'"
    s=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
       f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
       f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
       f"<text x='62' y='92' {font} font-size='42' font-weight='700' fill='#222'>{title}</text>",
       f"<text x='62' y='137' {font} font-size='23' fill='#555'>{sub}</text>",
       f"<text x='62' y='170' {font} font-size='21' fill='#777'>Positive = VDWT needs fewer bits than the reference; bars are 5-clip means at the right.</text>"]
    # axes
    s.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    s.append(f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>")
    step=10; start=(int(ymin)//step)*step
    for y in range(start,int(ymax)+1,step):
        yy=py(y)
        s.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD' stroke-width='1'/>")
        s.append(f"<text x='{left-18}' y='{yy+7:.1f}' {font} font-size='22' text-anchor='end' fill='#333'>{y:+d}%</text>")
    s.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{right}' y2='{py(0):.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='7 6'/>")
    n=len(series); group=1.0/(len(CLIPS)+1)
    for i,c in enumerate(CLIPS+["mean"]):
        cx=px((i+0.5)/ (len(CLIPS)+1))
        s.append(f"<text x='{cx:.1f}' y='{bottom+40}' {font} font-size='24' text-anchor='middle' fill='#333'>{c}</text>")
    bw=group*0.72/n
    for j,(name,colour,vals,mean) in enumerate(series):
        for i,c in enumerate(CLIPS+["mean"]):
            v=mean if c=="mean" else vals[i]
            base=(i+ (1-(n*bw/group))/2 + j*bw/group + bw/group/2) / (len(CLIPS)+1)
            x=px(base)*0+px((i+0.5)/(len(CLIPS)+1)) + (j-(n-1)/2)*bw*W*0.82
            # simpler: centre bars around group centre in pixels
            x=px((i+0.5)/(len(CLIPS)+1)) + (j-(n-1)/2)*bw*W*0.82
            w=bw*W*0.78
            y0=py(0); y1=py(v)
            s.append(f"<rect x='{x-w/2:.1f}' y='{min(y0,y1):.1f}' width='{w:.1f}' height='{abs(y1-y0):.1f}' fill='{colour}' stroke='#222' stroke-width='1.5'/>")
            s.append(f"<text x='{x:.1f}' y='{(y1-8) if v>=0 else (y1+22):.1f}' {font} font-size='18' text-anchor='middle' fill='#222'>{v:+.1f}</text>")
    # legend
    lx=left; ly=235
    for name,colour,_,_ in series:
        s.append(f"<rect x='{lx}' y='{ly-16}' width='20' height='20' fill='{colour}' stroke='#222'/>")
        s.append(f"<text x='{lx+30}' y='{ly}' {font} font-size='22' fill='#222'>{name}</text>")
        lx+=360
    s.append("</svg>")
    open(path,"w").write("\n".join(s))

means=lambda v:sum(v)/len(v)
draw(ROOT/"videos/postfilter_vs_x265_stock.svg","Post-filter comparison: vs x265 stock",
     "VDWT ultraheavy (no filter / custom band-pass) and x265 medium + normal 5×5 unsharp (0.6); hybrid BD vs x265 medium stock.",
     [("VDWT stock","#1A237E",[stock[c] for c in CLIPS],means([stock[c] for c in CLIPS])),
      ("VDWT + custom band 2.0","#6A1B9A",[custom[c] for c in CLIPS],means([custom[c] for c in CLIPS])),
      ("x265 + normal unsharp 0.6","#2E8B57",[xpf[c] for c in CLIPS],means([xpf[c] for c in CLIPS]))],
     -40,32)
draw(ROOT/"videos/postfilter_vs_x265_unsharp.svg","Post-filter comparison: vs x265 + normal unsharp",
     "Same filters, now with x265+unsharp as the reference; positive means VDWT needs fewer bits than x265's normal post-filter.",
     [("VDWT stock","#1A237E",[stock_vs_pf[c] for c in CLIPS],means([stock_vs_pf[c] for c in CLIPS])),
      ("VDWT + custom band 2.0","#6A1B9A",[custom_vs_pf[c] for c in CLIPS],means([custom_vs_pf[c] for c in CLIPS]))],
     -65,26)
for name,dct in [("stock",stock),("custom",custom),("x265+unsharp",xpf),("custom vs unsharp",custom_vs_pf)]:
    print(name, {c:round(dct[c],2) for c in CLIPS}, "mean",round(means(list(dct.values())),2))
