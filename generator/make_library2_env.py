"""Builds the second real-library environment: 40 CAPTCHAs rendered by multicolorcaptcha 1.2.0 (GPLv3,
Jose Miguel Rios Rubio) with CaptchaGenerator(2).gen_captcha_image() at default settings (4 digits, 640x360).

Usage: python3 make_library2_env.py OUTDIR   (multicolorcaptcha must be importable: pip install multicolorcaptcha)
"""
import sys,os,json,random
B=os.path.dirname(os.path.abspath(__file__)); pass
from multicolorcaptcha import CaptchaGenerator
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
random.seed(20261104); g=CaptchaGenerator(2); key=[]
for i in range(40):
    c=g.gen_captcha_image(); id_=f"M_{i:02d}"; c["image"].convert("RGB").save(f"{out}/{id_}.png"); key.append(dict(id=id_,nivel="M",verdad=c["characters"]))
json.dump(key,open(out+"/key.json","w"),indent=1); print("saved",c["image"].size,sorted({len(k["verdad"]) for k in key}))
