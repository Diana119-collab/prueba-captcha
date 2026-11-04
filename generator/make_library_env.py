"""Builds the 'real library' environment: 40 CAPTCHAs rendered by django-simple-captcha 0.7.1
with its default settings (4 uppercase letters, Vera font, noise dots + noise lines).

Usage: python3 make_library_env.py OUTDIR
Requires django-simple-captcha, Django and Pillow importable (pip install django-simple-captcha).
The same seed gives the same challenges; pixel output depends on the Pillow version.
"""
import sys, os, io, json, random
from django.conf import settings as S
S.configure(INSTALLED_APPS=["django.contrib.contenttypes", "captcha"],
            DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
            SECRET_KEY="x", USE_TZ=True)
import django; django.setup()
from PIL import Image
from captcha import views
from captcha.helpers import random_char_challenge

class State: pass

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
random.seed(7010203); key = []
for i in range(40):
    chal, _ = random_char_challenge()
    st = State(); st.challenge = chal
    resp = views._captcha_image(st, 1)
    im = Image.open(io.BytesIO(resp.content)).convert("RGB")
    id_ = f"D_{i:02d}"; im.save(f"{out}/{id_}.png"); key.append(dict(id=id_, nivel="D", verdad=chal))
json.dump(key, open(f"{out}/key.json", "w"), indent=1)
print("saved", len(key), "images", im.size)
