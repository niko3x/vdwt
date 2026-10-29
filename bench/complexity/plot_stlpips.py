#!/usr/bin/env python3
"""ST-LPIPS-only Pareto: bitrate saving at equal ST-LPIPS distance vs x265 medium."""
import math, sys
from collections import defaultdict

def load(path):
    d = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        c,e,q,k,s,m = line.split()
        d[e][c].append((float(k), -1000.0*float(s), float(m)))  # higher quality = -distance
    return d

def bd_rate(ref, test):
    def curve(pts):
        p=sorted((q,math.log(k)) for k,q,m in pts)
        return [x for x,y in p], [y for x,y in p]
    xa,ya=curve(ref); xb,yb=curve(test)
    lo=max(xa[0],xb[0]); hi=min(xa[-1],xb[-1])
    if hi<=lo: return None
    n=200; acc=0.0
    for i in range(n+1):
        x=lo+(hi-lo)*i/n
        def at(xs,ys):
            for j in range(len(xs)-1):
                if xs[j] <= x <= xs[j+1]:
                    return ys[j]+(ys[j+1]-ys[j])*(x-xs[j])/(xs[j+1]-xs[j])
            return ys[-1]
        acc += at(xb,yb)-at(xa,ya)
    return 100.0*(math.exp(acc/(n+1))-1.0)

def main():
    path,out=sys.argv[1:3]
    d=load(path); clips=sorted({c for e in d.values() for c in e}); ref=d["x265-medium"]
    res={}
    for e,p in d.items():
        bs=[]; ms=[]
        for c in clips:
            b=bd_rate(ref[c],p[c])
            if b is not None: bs.append(b)
            ms += [m for k,q,m in p[c]]
        if bs: res[e]=(-sum(bs)/len(bs), sum(ms)/len(ms))
    W,H=2000,1400; left,right,top,bottom=230,1920,330,1185
    xmin,xmax=8.0,400.0; ymin,ymax=-125.0,15.0
    px=lambda x:left+(right-left)*(math.log10(x)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))
    py=lambda y:bottom-(bottom-top)*(y-ymin)/(ymax-ymin)
    f="font-family='Inter, Helvetica, Arial, sans-serif'"
    s=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
       f"<rect width='{W}' height='{H}' fill='#FAF9F5'/>",
       f"<rect x='4' y='4' width='{W-8}' height='{H-8}' fill='none' stroke='#222' stroke-width='3'/>",
       f"<text x='62' y='92' {f} font-size='44' font-weight='700' fill='#222'>Samtime: ST-LPIPS-only Pareto</text>",
       f"<text x='62' y='137' {f} font-size='24' fill='#555'>2 clips (a720, b720), 20 frames, ST-LPIPS on frames 0/10/19, VMAF vmaf_v0.6.1 not used</text>",
       f"<text x='62' y='170' {f} font-size='24' fill='#555'>Shift-tolerant AlexNet ST-LPIPS; lower distance is better; bitrate saving at equal ST-LPIPS vs x265 medium</text>",
       f"<text x='62' y='203' {f} font-size='22' fill='#777'>Positive = VDWT needs fewer bits; negative = more bits. Encodes are 20 frames, 720p, 4 threads.</text>"]
    s += [f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#222' stroke-width='2'/>",
          f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#222' stroke-width='2'/>"]
    for y in range(-120,16,20):
        yy=py(y); s.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='#DDD' stroke-width='1'/>")
        s.append(f"<text x='{left-20}' y='{yy+8:.1f}' {f} font-size='24' text-anchor='end' fill='#333'>{y:+d}%</text>")
    for x in (10,20,50,100,200,400):
        xx=px(x); s.append(f"<line x1='{xx:.1f}' y1='{top}' x2='{xx:.1f}' y2='{bottom}' stroke='#EEE' stroke-width='1'/>")
        s.append(f"<text x='{xx:.1f}' y='{bottom+38}' {f} font-size='24' text-anchor='middle' fill='#333'>{x}</text>")
    s.append(f"<text x='{(left+right)/2:.0f}' y='{bottom+95}' {f} font-size='26' text-anchor='middle' fill='#333'>Encode time per frame (ms, log; 720p, 4 threads, x86)</text>")
    s.append(f"<text x='55' y='{(top+bottom)/2:.0f}' {f} font-size='26' text-anchor='middle' fill='#333' transform='rotate(-90 55 {(top+bottom)/2:.0f})'>Bitrate saving at equal ST-LPIPS (%)</text>")
    s.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{right}' y2='{py(0):.1f}' stroke='#222' stroke-width='1.5' stroke-dasharray='8 7'/>")
    vd=["realtime","light","medium","heavy","ultraheavy"]
    pts=[(e,res[e]) for e in res if e.startswith("vdwt-")]
    pts.sort(key=lambda p:p[1][1])
    p=[(px(t),py(v)) for _, (v,t) in pts]
    s.append("<polyline points='"+' '.join(f"{x:.1f},{y:.1f}" for x,y in p)+"' fill='none' stroke='#880E4F' stroke-width='5'/>")
    for (e,(v,t)),(x,y) in zip(pts,p):
        s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='#880E4F' stroke='#222' stroke-width='2.5'/>")
        name=e.replace('vdwt-',''); dy=-24 if name=='ultraheavy' else 34
        s.append(f"<text x='{x:.1f}' y='{y+dy:.1f}' {f} font-size='27' text-anchor='middle' fill='#880E4F'>{name}</text>")
    xp=[(e,res[e]) for e in res if e.startswith('x265-')]
    xp.sort(key=lambda p:p[1][1]); q=[(px(t),py(v)) for _, (v,t) in xp]
    s.append("<polyline points='"+' '.join(f"{x:.1f},{y:.1f}" for x,y in q)+"' fill='none' stroke='#2E8B57' stroke-width='4'/>")
    for (e,(v,t)),(x,y) in zip(xp,q):
        s.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='13' fill='#2E8B57' stroke='#222' stroke-width='2'/>")
        name=e.replace('x265-','')
        if name in ('medium','fast','slow','veryslow'):
            s.append(f"<text x='{x+20:.1f}' y='{y+28:.1f}' {f} font-size='24' fill='#2E8B57'>{name}</text>")
    s += [f"<circle cx='250' cy='270' r='16' fill='#880E4F' stroke='#222' stroke-width='2.5'/>",
          f"<text x='275' y='280' {f} font-size='28' fill='#222'>VDWT v12</text>",
          f"<circle cx='500' cy='270' r='14' fill='#2E8B57' stroke='#222' stroke-width='2'/>",
          f"<text x='525' y='280' {f} font-size='28' fill='#222'>x265</text>", "</svg>"]
    open(out,'w').write('\n'.join(s))
    for e,(v,t) in sorted(res.items(),key=lambda q:-q[1][0]): print(f"{e:20s} {v:+7.2f}% {t:7.2f} ms")
main()
