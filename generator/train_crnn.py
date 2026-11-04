#!/usr/bin/env python3
"""CRNN + CTC para CAPTCHAs de texto (solucionador de extremo a extremo).

Dos modos:
  1) Sintético (por defecto): entrena con CAPTCHAs generados aquí y evalúa en --test_dir
     (carpeta con imgs/*.png y key.json con [{id, nivel, verdad}, ...]).
  2) Real (--real_dir): entrena y evalúa con imágenes reales etiquetadas por nombre de archivo
     (p. ej. 2b827.png -> "2b827"). Divide 80/20 y guarda la lista de prueba.

Uso en Colab (con GPU):
  !python train_crnn.py --test_dir prueba_200
  !python train_crnn.py --real_dir mis_captchas_reales --epochs 60
  !python train_crnn.py --selftest        # solo genera datos, sin torch
"""
import argparse, csv, glob, json, math, os, random, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CHARS_SYN = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
W, H, LEN = 310, 80, 5
IMG_H, IMG_W = 64, 200
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
FONT_PATHS = [p for p in FONT_PATHS if os.path.exists(p)]
LEVELS = {
    "N1": dict(rot=5,  kern=4,  lines=0, dots=0,   warp=0.0, blur=0.0),
    "N2": dict(rot=10, kern=2,  lines=2, dots=40,  warp=2.0, blur=0.0),
    "N3": dict(rot=20, kern=-2, lines=4, dots=120, warp=3.5, blur=0.4),
    "N4": dict(rot=28, kern=-5, lines=6, dots=200, warp=4.5, blur=0.6),
    "N5": dict(rot=35, kern=-8, lines=7, dots=250, warp=5.0, blur=0.8),
}


# ---------------------------------------------------------------- generador
def get_font(size, rng):
    if FONT_PATHS:
        return ImageFont.truetype(rng.choice(FONT_PATHS), size)
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


def rand_color(rng, lo, hi):
    return tuple(rng.randint(lo, hi) for _ in range(3))


def make_captcha(text, p, rng):
    img = Image.new("RGB", (W, H), rand_color(rng, 215, 255))
    x = 12
    for ch in text:
        size = rng.randint(40, 52)
        font = get_font(size, rng)
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
        d.point((rng.randint(0, W - 1), rng.randint(0, H - 1)), fill=rand_color(rng, 0, 160))
    a = np.array(img)
    ph = rng.uniform(0, 6.28)
    out = np.empty_like(a)
    for col in range(W):
        out[:, col] = np.roll(a[:, col], int(round(p["warp"] * math.sin(col / 17.0 + ph))), axis=0)
    img = Image.fromarray(out)
    if p["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    return img


def to_array(img):
    """PIL -> uint8 (IMG_H, IMG_W) en escala de grises."""
    return np.array(img.convert("L").resize((IMG_W, IMG_H), Image.BICUBIC), dtype=np.uint8)


def gen_one(seed):
    rng = random.Random(seed)
    lvl = rng.choice(list(LEVELS))
    text = "".join(rng.choice(CHARS_SYN) for _ in range(LEN))
    return to_array(make_captcha(text, LEVELS[lvl], rng)), text


def gen_dataset(n, seed0):
    from multiprocessing import Pool
    with Pool() as pool:
        res = pool.map(gen_one, range(seed0, seed0 + n), chunksize=200)
    return np.stack([r[0] for r in res]), [r[1] for r in res]


# ---------------------------------------------------------------- modelo y entrenamiento
def build_model(n_classes):
    import torch.nn as nn

    def block(i, o, pool):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(pool)]

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            # entrada (N,1,64,200) -> (N,128,4,50)
            self.cnn = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2),
                                     *block(64, 128, (2, 1)), *block(128, 128, (2, 1)))
            self.proj = nn.Linear(128 * 4, 128)
            self.rnn = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
            self.out = nn.Linear(256, n_classes)

        def forward(self, x):
            f = self.cnn(x)                           # (N,C,4,50)
            n, c, h, w = f.shape
            f = f.permute(0, 3, 1, 2).reshape(n, w, c * h)  # (N,50,512)
            f = self.proj(f).relu()
            f, _ = self.rnn(f)
            return self.out(f)                        # (N,50,classes)

    return CRNN()


def decode(logits, charset):
    """Decodificación voraz CTC: argmax, colapsa repetidos, quita blanco (0)."""
    best = logits.argmax(-1).cpu().numpy()
    outs = []
    for seq in best:
        prev, s = 0, []
        for k in seq:
            if k != prev and k != 0:
                s.append(charset[k - 1])
            prev = k
        outs.append("".join(s))
    return outs


def predict(model, X, charset, device, bs=256):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().div(255).sub(0.5).div(0.5).unsqueeze(1).to(device)
            preds += decode(model(xb), charset)
    return preds


def train(Xtr, ytr, charset, epochs, device, bs=128, lr=2e-3):
    import torch, torch.nn as nn
    c2i = {c: i + 1 for i, c in enumerate(charset)}
    model = build_model(len(charset) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    idx = np.arange(len(Xtr))
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx); tot = 0.0; t0 = time.time()
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(Xtr[b]).float().div(255).sub(0.5).div(0.5).unsqueeze(1).to(device)
            tg = [torch.tensor([c2i[c] for c in ytr[j]]) for j in b]
            tl = torch.tensor([len(t) for t in tg]); tg = torch.cat(tg)
            lp = model(xb).log_softmax(-1).permute(1, 0, 2)  # (T,N,C)
            il = torch.full((lp.size(1),), lp.size(0), dtype=torch.long)
            loss = ctc(lp, tg, il, tl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item() * len(b)
        print(f"época {ep + 1}/{epochs}  pérdida {tot / len(idx):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return model


def report(rows):
    levels = sorted({r["nivel"] for r in rows})
    print(f"\n{'Nivel':8s}{'n':>5s}{'Acierto exacto':>18s}")
    for lv in levels + ["TOTAL"]:
        sel = rows if lv == "TOTAL" else [r for r in rows if r["nivel"] == lv]
        k = sum(r["exacto"] for r in sel)
        print(f"{lv:8s}{len(sel):>5d}{f'{k}/{len(sel)} ({100 * k / len(sel):.0f}%)':>18s}")


# ---------------------------------------------------------------- modos
def eval_dir(model, d, device):
    key = json.load(open(os.path.join(d, "key.json")))
    sub = "imgs" if os.path.isdir(os.path.join(d, "imgs")) else ""  # dataset/library has the PNGs next to key.json
    Xte = np.stack([to_array(Image.open(os.path.join(d, sub, it["id"] + ".png")).convert("RGB")) for it in key])
    preds = predict(model, Xte, CHARS_SYN, device)
    return [dict(id=it["id"], nivel=it["nivel"], verdad=it["verdad"], pred=p, exacto=int(p == it["verdad"]))
            for it, p in zip(key, preds)]


def mode_synthetic(a, device):
    import torch
    print(f"Generando {a.n_train} CAPTCHAs de entrenamiento...", flush=True)
    Xtr, ytr = gen_dataset(a.n_train, seed0=10_000_000)  # semillas distintas a las de prueba
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [None]
    targets = [(a.test_dir, "crnn_levels")] + ([(a.env_dir, "crnn_environments")] if a.env_dir else []) \
              + ([(a.lib_dir, "crnn_library")] if a.lib_dir else []) + ([(a.lib2_dir, "crnn_library2")] if a.lib2_dir else [])
    summary = []
    for sd in seeds:
        if sd is not None:  # seed initialisation and data order; the training images are the same in every run
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(sd)
        tag = "" if sd is None else f"_seed{sd}"
        print(f"\n=== Entrenamiento (semilla {sd}) ===", flush=True)
        model = train(Xtr, ytr, CHARS_SYN, a.epochs, device)
        line = {"seed": sd}
        for d, base in targets:
            rows = eval_dir(model, d, device)
            out = f"{base}{tag}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            report(rows)
            line[base] = sum(r["exacto"] for r in rows)
            print("Guardado:", out)
        summary.append(line)
        with open("crnn_summary.csv", "w", newline="") as f:  # rewritten after every run, so nothing is lost if Colab stops
            w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
        print("Resumen hasta ahora:", summary, flush=True)


def mode_real(a, device):
    files = sorted(sum([glob.glob(os.path.join(a.real_dir, "**", e), recursive=True) for e in ("*.png", "*.jpg", "*.jpeg")], []))
    labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
    keep = [(f, l) for f, l in zip(files, labels) if 3 <= len(l) <= 10]
    random.Random(0).shuffle(keep)
    ntest = min(200, max(50, len(keep) // 5))
    test, train_ = keep[:ntest], keep[ntest:]
    charset = "".join(sorted({c for _, l in keep for c in l}))
    print(f"{len(train_)} para entrenar, {len(test)} para probar. Alfabeto ({len(charset)}): {charset}", flush=True)
    load = lambda fs: np.stack([to_array(Image.open(f).convert("RGB")) for f, _ in fs])
    model = train(load(train_), [l for _, l in train_], charset, a.epochs, device, bs=64, lr=1e-3)
    preds = predict(model, load(test), charset, device)
    rows = [dict(id=os.path.basename(f), nivel="real", verdad=l, pred=p, exacto=int(p == l)) for (f, l), p in zip(test, preds)]
    with open("resultados_crnn_real.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    open("real_test_files.txt", "w").write("\n".join(os.path.basename(f) for f, _ in test))
    report(rows)
    print("\nGuardado: resultados_crnn_real.csv y real_test_files.txt (estas imágenes de prueba sirven para comparar Tesseract y Claude)")


def selftest():
    X, y = gen_dataset(24, seed0=1)
    print("X", X.shape, X.dtype, "| etiquetas:", y[:6])
    strip = np.concatenate(list(X[:6]), axis=0)
    Image.fromarray(strip).save("selftest_muestra.png")
    print("fuentes:", FONT_PATHS or "(por defecto de Pillow)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="prueba_200")
    ap.add_argument("--real_dir", default=None)
    ap.add_argument("--env_dir", default=None, help="carpeta con imgs/ y key.json de entornos nuevos (E1,E2,E3)")
    ap.add_argument("--lib_dir", default=None, help="folder with the django-simple-captcha images + key.json (dataset/library)")
    ap.add_argument("--lib2_dir", default=None, help="second real-library set (dataset/library2)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 1,2,3,4,5: one training run per seed")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device, "(si dice cpu: Entorno de ejecución > Cambiar tipo > GPU)")
    (mode_real if a.real_dir else mode_synthetic)(a, device)
