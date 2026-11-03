#!/usr/bin/env python3
"""Generate 3 held-out CAPTCHA environments (E1 new fonts, E2 six characters, E3 inverted colours)
at the N3 difficulty parameters. Usage: python3 make_env.py OUTDIR"""
import sys, os, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
N3 = dict(rot=20, kern=-2, lines=4, dots=120, warp=3.5, blur=0.4)
DJ = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]
NEW = ["/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf","/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",
       "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf","/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"]
ENVS = {"E1": dict(W=310,H=80,LEN=5,fonts=NEW,inv=False,seed=2718),
        "E2": dict(W=370,H=80,LEN=6,fonts=DJ,inv=False,seed=1618),
        "E3": dict(W=310,H=80,LEN=5,fonts=DJ,inv=True,seed=1414)}
def col(rng,lo,hi): return tuple(rng.randint(lo,hi) for _ in range(3))
def make(text,e,rng,p=N3):
    W,H=e["W"],e["H"]; inv=e["inv"]
    img=Image.new("RGB",(W,H),col(rng,20,70) if inv else col(rng,215,255)); x=12
    fg=(lambda:col(rng,170,255)) if inv else (lambda:col(rng,0,110))
    nz=(lambda:col(rng,90,200)) if inv else (lambda:col(rng,0,150))
    for ch in text:
        s=rng.randint(40,52); f=ImageFont.truetype(rng.choice(e["fonts"]),s)
        t=Image.new("RGBA",(s+20,s+20),(0,0,0,0)); ImageDraw.Draw(t).text((8,2),ch,font=f,fill=fg()+(255,))
        t=t.rotate(rng.uniform(-p["rot"],p["rot"]),expand=True,resample=Image.BICUBIC)
        img.paste(t,(int(x),rng.randint(0,14)),t); x+=t.width*0.62+p["kern"]
    d=ImageDraw.Draw(img)
    for _ in range(p["lines"]): d.line([(rng.randint(0,W),rng.randint(0,H)),(rng.randint(0,W),rng.randint(0,H))],fill=nz(),width=rng.randint(1,3))
    for _ in range(p["dots"]): d.point((rng.randint(0,W-1),rng.randint(0,H-1)),fill=nz())
    a=np.array(img); ph=rng.uniform(0,6.28); o=np.empty_like(a)
    for c in range(W): o[:,c]=np.roll(a[:,c],int(round(p["warp"]*math.sin(c/17.0+ph))),axis=0)
    img=Image.fromarray(o)
    return img.filter(ImageFilter.GaussianBlur(p["blur"]))
if __name__=="__main__":
    out=sys.argv[1]; os.makedirs(out+"/imgs",exist_ok=True); key=[]
    for name,e in ENVS.items():
        rng=random.Random(e["seed"])
        for i in range(40):
            t="".join(rng.choice(CHARS) for _ in range(e["LEN"])); id_=f"{name}_{i:02d}"
            make(t,e,rng).save(f"{out}/imgs/{id_}.png"); key.append(dict(id=id_,nivel=name,verdad=t))
    json.dump(key,open(out+"/key.json","w"),indent=1)
