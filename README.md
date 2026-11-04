# prueba-captcha

Generator, dataset and results for the poster/extended abstract *"Can Current AI Solve CAPTCHAs? What It Says About Machine Intelligence"* (WAIMLAp 2026).

Online human test: https://diana119-collab.github.io/prueba-captcha/ (`index.html`).

## Contents
- `generator/` – Python (Pillow) CAPTCHA generator and solvers.
  - `pilot.py`, `pilot2.py`: generator and Tesseract baselines; `python3 pilot2.py OUTDIR` rebuilds the 5-level set (seed 31415).
  - `make_environments.py`: builds the three held-out environments (`python3 make_environments.py OUTDIR`).
  - `train_crnn.py`: CRNN+CTC trained on generated images (`--n_train 50000 --epochs 60`; `--test_dir`, `--env_dir`).
- `dataset/levels/`: 200 images (levels N1–N5, 40 each) + `key.json` (ground truth).
- `dataset/environments/`: 120 images (E1 new fonts, E2 six characters, E3 inverted colours; 40 each, N3 difficulty) + `key.json`.
- `results/`: per-image predictions of every solver (Tesseract raw/preprocessed, multimodal LLM, CRNN, humans). Human data are anonymised (P1–P6, adults, no personal data).

## Level parameters
| Level | max rotation (°) | letter spacing (px) | lines | noise dots | warp amp. (px) | blur σ |
|---|---|---|---|---|---|---|
| N1 | 5 | +4 | 0 | 0 | 0 | 0 |
| N2 | 10 | +2 | 2 | 40 | 2.0 | 0 |
| N3 | 20 | −2 | 4 | 120 | 3.5 | 0.4 |
| N4 | 28 | −5 | 6 | 200 | 4.5 | 0.6 |
| N5 | 35 | −8 | 7 | 250 | 5.0 | 0.8 |

All images: 32-symbol alphabet (no I, O, 0, 1), 5 characters (6 in E2), 310×80 px (370×80 in E2).

## Notes
- CRNN training is not seeded; two 60-epoch runs gave different accuracies (see `crnn_levels_run1.csv`, `crnn_levels_run2.csv`).
- Synthetic CAPTCHAs only; no real site was accessed.

## Real-data check (exploratory)
`results/real_sample/` holds results on 40 random images (seed 2026) from the public dataset [roncuevas/Captchas](https://github.com/roncuevas/Captchas) (MIT license; real 3-character text CAPTCHAs; provenance: collected from a web application, per its authors). `sample_ids.csv` lists the image hashes and labels; the images are not redistributed here. Only Tesseract and the multimodal model were tested on this sample (CRNN and humans were not).
