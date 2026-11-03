"""Pilot: synthetic text CAPTCHAs (own generator, PIL) vs classic OCR (Tesseract).
Everything is generated locally; no real site is touched."""
import csv, math, os, random, sys
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = sys.argv[1]
N_PER_LEVEL = 50
SEED = 2026
CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sin caracteres ambiguos (O/0, I/1)
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
W, H, LEN = 310, 80, 5

LEVELS = {
    # rot: rotación máx. (grados); kern: separación entre letras (px); lines/dots: ruido; warp: amplitud de ondulación
    "leve":   dict(rot=10, kern=2,   lines=2, dots=40,  warp=2.0, blur=0.0),
    "fuerte": dict(rot=35, kern=-8,  lines=7, dots=250, warp=5.0, blur=0.8),
}

def rand_color(rng, lo, hi):
    return tuple(rng.randint(lo, hi) for _ in range(3))

def make_captcha(text, p, rng):
    img = Image.new("RGB", (W, H), rand_color(rng, 215, 255))
    x = 12
    for ch in text:
        size = rng.randint(40, 52)
        font = ImageFont.truetype(rng.choice(FONTS), size)
        tile = Image.new("RGBA", (size + 20, size + 20), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((8, 2), ch, font=font, fill=rand_color(rng, 0, 110) + (255,))
        tile = tile.rotate(rng.uniform(-p["rot"], p["rot"]), expand=True, resample=Image.BICUBIC)
        img.paste(tile, (int(x), rng.randint(0, 14)), tile)
        x += tile.width * 0.62 + p["kern"]
    d = ImageDraw.Draw(img)
    for _ in range(p["lines"]):
        d.line([(rng.randint(0, W), rng.randint(0, H)), (rng.randint(0, W), rng.randint(0, H))],
               fill=rand_color(rng, 0, 150), width=rng.randint(1, 3))
    for _ in range(p["dots"]):
        px, py = rng.randint(0, W - 1), rng.randint(0, H - 1)
        d.point((px, py), fill=rand_color(rng, 0, 160))
    # ondulación sinusoidal
    a = np.array(img)
    ph = rng.uniform(0, 6.28)
    out = np.empty_like(a)
    for col in range(W):
        shift = int(round(p["warp"] * math.sin(col / 17.0 + ph)))
        out[:, col] = np.roll(a[:, col], shift, axis=0)
    img = Image.fromarray(out)
    if p["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    return img

# ---------- solucionadores ----------
CFG = f"--psm 7 -c tessedit_char_whitelist={CHARS}"

def solve_raw(img):
    return pytesseract.image_to_string(img, config=CFG).strip().replace(" ", "")

def solve_pre(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.medianBlur(g, 3)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    b = cv2.copyMakeBorder(b, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return pytesseract.image_to_string(Image.fromarray(b), config=CFG).strip().replace(" ", "")

SOLVERS = {"Tesseract (crudo)": solve_raw, "Tesseract (preprocesado)": solve_pre}

def char_acc(pred, truth):
    return sum(1 for a, b in zip(pred, truth) if a == b) / len(truth)

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def main():
    os.makedirs(f"{OUT}/imgs", exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    for lvl, p in LEVELS.items():
        for i in range(N_PER_LEVEL):
            text = "".join(rng.choice(CHARS) for _ in range(LEN))
            img = make_captcha(text, p, rng)
            fn = f"{lvl}_{i:02d}_{text}.png"
            img.save(f"{OUT}/imgs/{fn}")
            for name, fn_solve in SOLVERS.items():
                pred = fn_solve(img)
                rows.append(dict(nivel=lvl, solver=name, verdad=text, pred=pred,
                                 exacto=int(pred == text), car=round(char_acc(pred, text), 3)))
    with open(f"{OUT}/resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"{'Solucionador':26s}{'Nivel':8s}{'Acierto exacto':>16s}{'IC95%':>16s}{'Acierto/caracter':>18s}")
    summary = []
    for name in SOLVERS:
        for lvl in LEVELS:
            r = [x for x in rows if x["solver"] == name and x["nivel"] == lvl]
            k, n = sum(x["exacto"] for x in r), len(r)
            lo, hi = wilson(k, n)
            ca = sum(x["car"] for x in r) / n
            summary.append((name, lvl, k, n, lo, hi, ca))
            print(f"{name:26s}{lvl:8s}{f'{k}/{n} ({100*k/n:.0f}%)':>16s}{f'[{100*lo:.0f}-{100*hi:.0f}%]':>16s}{100*ca:>17.0f}%")
    return summary

if __name__ == "__main__":
    summary = main()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    names = list(SOLVERS)
    xs = np.arange(len(LEVELS)); bw = 0.36
    for j, name in enumerate(names):
        vals, errs = [], [[], []]
        for lvl in LEVELS:
            s = next(s for s in summary if s[0] == name and s[1] == lvl)
            v = 100 * s[2] / s[3]
            vals.append(v); errs[0].append(v - 100 * s[4]); errs[1].append(100 * s[5] - v)
        ax.bar(xs + (j - 0.5) * bw, vals, bw, yerr=errs, capsize=3, label=name)
    ax.set_xticks(xs); ax.set_xticklabels([f"Distorsión {l}" for l in LEVELS])
    ax.set_ylabel("Acierto exacto (%)"); ax.set_ylim(0, 105)
    ax.set_title("CAPTCHAs sintéticos de texto (n=50 por nivel)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica_piloto.png")
