"""Piloto ampliado: 5 niveles x 40 imágenes. Reutiliza el generador y los solucionadores de pilot.py."""
import csv, json, os, random, sys
sys.argv = [sys.argv[0], sys.argv[1]]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402

OUT = sys.argv[1]
SEED = 31415
N = 40
LEVELS = {
    "N1": dict(rot=5,  kern=4,  lines=0, dots=0,   warp=0.0, blur=0.0),
    "N2": dict(rot=10, kern=2,  lines=2, dots=40,  warp=2.0, blur=0.0),
    "N3": dict(rot=20, kern=-2, lines=4, dots=120, warp=3.5, blur=0.4),
    "N4": dict(rot=28, kern=-5, lines=6, dots=200, warp=4.5, blur=0.6),
    "N5": dict(rot=35, kern=-8, lines=7, dots=250, warp=5.0, blur=0.8),
}

os.makedirs(f"{OUT}/imgs", exist_ok=True)
rng = random.Random(SEED)
items = []
for lvl, p in LEVELS.items():
    for i in range(N):
        text = "".join(rng.choice(pilot.CHARS) for _ in range(pilot.LEN))
        img = pilot.make_captcha(text, p, rng)
        iid = f"{lvl}_{i:02d}"
        img.save(f"{OUT}/imgs/{iid}.png")
        items.append(dict(id=iid, nivel=lvl, verdad=text))

# Tesseract
rows = []
for it in items:
    from PIL import Image
    img = Image.open(f"{OUT}/imgs/{it['id']}.png").convert("RGB")
    for name, fn in pilot.SOLVERS.items():
        pred = fn(img)
        rows.append(dict(id=it["id"], nivel=it["nivel"], solver=name, verdad=it["verdad"], pred=pred,
                         exacto=int(pred == it["verdad"]), car=round(pilot.char_acc(pred, it["verdad"]), 3)))
with open(f"{OUT}/resultados_tesseract.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
json.dump(items, open(f"{OUT}/key.json", "w"))

print(f"{'Solucionador':26s}" + "".join(f"{l:>8s}" for l in LEVELS))
for name in pilot.SOLVERS:
    line = f"{name:26s}"
    for lvl in LEVELS:
        r = [x for x in rows if x["solver"] == name and x["nivel"] == lvl]
        line += f"{100*sum(x['exacto'] for x in r)/len(r):>7.0f}%"
    print(line)
