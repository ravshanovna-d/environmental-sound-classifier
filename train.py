"""Train an environmental sound classifier on ESC-10, compare three training strategies,
and export the best model + the exact log-mel preprocessing for the browser demo.

Usage:  python train.py
Outputs: web/model.onnx, web/mel_filters.json, web/meta.json, web/examples/*.wav, web/assets/*.png
"""
import copy
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
SEED = 0
# Plain PyTorch convolutions: the oneDNN kernels need large scratch buffers and failed on a low-memory machine.
torch.backends.mkldnn.enabled = False

# Audio / spectrogram settings (shared with web/mel.js through web/mel_filters.json)
SR = 22050            # every clip is resampled to this rate
DURATION = 5.0        # ESC-50 clips are exactly 5 s
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP, N_MELS = 1024, 512, 128
N_FRAMES = 1 + N_SAMPLES // HOP          # 216 (librosa centres frames)
TOP_DB = 80.0
NORM_OFFSET, NORM_SCALE = 40.0, 20.0     # x = (dB + 40) / 20  ->  roughly [-2, 2]
POOL = 2              # the network sees a 2x average-pooled spectrogram (64 x 108) to keep CPU training fast

PRETTY = {"dog": "Dog bark", "rooster": "Rooster", "rain": "Rain", "sea_waves": "Sea waves",
          "crackling_fire": "Crackling fire", "crying_baby": "Crying baby", "sneezing": "Sneezing",
          "clock_tick": "Clock tick", "helicopter": "Helicopter", "chainsaw": "Chainsaw"}

STRATEGIES = {
    # name: (ImageNet weights?, train only the classifier head?, learning rate, epochs)
    "From scratch":       dict(pretrained=False, freeze=False, lr=1e-3, epochs=15),
    "Frozen features":    dict(pretrained=True,  freeze=True,  lr=1e-2, epochs=15),
    "Full fine-tuning":   dict(pretrained=True,  freeze=False, lr=1e-3, epochs=15),
}


# ----------------------------------------------------------------------------- data
def _fetch(url, dst, retries=5):
    for attempt in range(retries):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_esc10() -> pd.DataFrame:
    """Download only the ESC-10 subset (400 clips, ~180 MB) from the official ESC-50 repository.

    The full repository zip is ~650 MB; fetching the 400 needed files individually is faster
    and can resume if the connection drops. Files are cached in data/esc10/.
    """
    out = DATA / "esc10"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out / "esc10.csv"
    if not meta_path.exists():
        meta = pd.read_csv(f"{ESC50_RAW}/meta/esc50.csv")
        meta[meta.esc10].reset_index(drop=True).to_csv(meta_path, index=False)
    meta = pd.read_csv(meta_path)
    missing = [f for f in meta.filename if not (out / "audio" / f).exists()]
    if missing:
        print(f"Downloading {len(missing)} ESC-10 clips...")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda f: _fetch(f"{ESC50_RAW}/audio/{f}", out / "audio" / f), missing))
    return meta


def load_audio(path) -> np.ndarray:
    """Mono float32 at SR, padded / cut to exactly 5 s."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return librosa.util.fix_length(y, size=N_SAMPLES)


def logmel(y: np.ndarray) -> np.ndarray:
    """Normalised log-mel spectrogram, shape (N_MELS, N_FRAMES). The browser reproduces this exactly."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    db = librosa.power_to_db(m, ref=np.max, top_db=TOP_DB)
    return ((db + NORM_OFFSET) / NORM_SCALE).astype(np.float32)


def build_dataset(df: pd.DataFrame):
    classes = sorted(df.category.unique())
    cache = DATA / "esc10_logmel.npz"
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        X = np.stack([logmel(load_audio(DATA / "esc10" / "audio" / f)) for f in df.filename])
        np.savez_compressed(cache, X=X)
    y = np.array([classes.index(c) for c in df.category])
    return torch.tensor(X)[:, None], torch.tensor(y), classes


def split_by_fold(df, X, y):
    """Official ESC folds: 1-3 train, 4 validation, 5 test."""
    fold = df.fold.values
    masks = {"train": fold <= 3, "val": fold == 4, "test": fold == 5}
    return {k: (X[torch.tensor(m)], y[torch.tensor(m)]) for k, m in masks.items()}


# ----------------------------------------------------------------------------- model
class SoundNet(nn.Module):
    """ResNet18 on a 1-channel log-mel image (the channel is repeated to fit ImageNet weights)."""

    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=n_classes)

    def forward(self, x):                      # x: (B, 1, N_MELS, N_FRAMES)
        if POOL > 1:                           # CPU budget: 2x2 average-pool to 64 x 108 before the CNN
            x = F.avg_pool2d(x, POOL)
        return self.backbone(x.expand(-1, 3, -1, -1))


class ProbsWrapper(nn.Module):
    """What gets exported: log-mel in, class probabilities out."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, mel):
        return self.net(mel).softmax(1)


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Light augmentation: random circular time shift + one frequency and one time mask (SpecAugment)."""
    xb = xb.clone()
    B, _, Fq, T = xb.shape
    for i in range(B):
        xb[i] = torch.roll(xb[i], int(torch.randint(0, T, (1,))), dims=-1)
        f = int(torch.randint(0, 17, (1,))); f0 = int(torch.randint(0, Fq - f + 1, (1,)))
        t = int(torch.randint(0, 31, (1,))); t0 = int(torch.randint(0, T - t + 1, (1,)))
        xb[i, :, f0:f0 + f, :] = -2.0          # -2 == silence after normalisation
        xb[i, :, :, t0:t0 + t] = -2.0
    return xb


@torch.no_grad()
def predict_proba(model, X, bs=64):
    model.eval()
    return torch.cat([model(X[i:i + bs]).softmax(1) for i in range(0, len(X), bs)])


def accuracy(model, X, y):
    return (predict_proba(model, X).argmax(1) == y).float().mean().item()


def make_model(pretrained, freeze, n_classes=10):
    model = SoundNet(n_classes, pretrained=pretrained)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.backbone.get_classifier().parameters():
            p.requires_grad = True
    return model


def fit(model, splits, lr, epochs, freeze=False, bs=16, verbose=True):
    """Adam + cosine schedule; returns the best-on-validation model and the per-epoch history."""
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ((len(Xtr) + bs - 1) // bs))
    hist, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze:                     # keep frozen BatchNorm statistics untouched
            model.backbone.eval()
        perm, losses, t0 = torch.randperm(len(Xtr)), [], time.time()
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(augment(Xtr[idx])), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        va = accuracy(model, Xva, yva)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": va})
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch:2d}  loss {hist[-1]['train_loss']:.3f}  val acc {va:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, hist


def run_strategies(splits, strategies=STRATEGIES, verbose=True):
    results = {}
    for name, cfg in strategies.items():
        if verbose:
            print(f"\n== {name} ==")
        torch.manual_seed(SEED)
        model = make_model(cfg["pretrained"], cfg["freeze"])
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        model, hist = fit(model, splits, cfg["lr"], cfg["epochs"], freeze=cfg["freeze"], verbose=verbose)
        results[name] = {"model": model, "history": hist, "cfg": cfg, "trainable_params": n_train,
                         "best_val_acc": max(h["val_acc"] for h in hist),
                         "test_acc": accuracy(model, *splits["test"]),
                         "train_minutes": (time.time() - t0) / 60}
        if verbose:
            print(f"  best val {results[name]['best_val_acc']:.3f}  test {results[name]['test_acc']:.3f}")
    return results


def predict(model, path, classes, k=3):
    """Top-k (class, probability) for any audio file."""
    x = torch.tensor(logmel(load_audio(path)))[None, None]
    p = predict_proba(model, x)[0]
    top = p.topk(k)
    return [(classes[i], round(float(v), 4)) for v, i in zip(top.values, top.indices)]


# ----------------------------------------------------------------------------- browser parity
def export_mel_filters(path):
    """librosa's mel filterbank as sparse rows, so the browser uses the identical matrix."""
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)          # (128, 513)
    rows = []
    for r in fb:
        nz = np.nonzero(r)[0]
        start, end = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
        rows.append({"start": start, "w": [float(f"{v:.9g}") for v in r[start:end]]})
    cfg = {"sr": SR, "duration": DURATION, "n_samples": N_SAMPLES, "n_fft": N_FFT, "hop": HOP,
           "n_mels": N_MELS, "n_frames": N_FRAMES, "top_db": TOP_DB, "amin": 1e-10,
           "norm_offset": NORM_OFFSET, "norm_scale": NORM_SCALE, "filters": rows}
    path.write_text(json.dumps(cfg, separators=(",", ":")))
    return cfg


def js_logmel_numpy(y, cfg):
    """Line-by-line numpy port of web/mel.js (float64), used to check it against librosa."""
    n_fft, hop = cfg["n_fft"], cfg["hop"]
    pad = n_fft // 2
    yp = np.concatenate([np.zeros(pad), y.astype(np.float64), np.zeros(pad)])   # center=True, zero padding
    n = np.arange(n_fft)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)                            # periodic Hann
    out = np.zeros((cfg["n_mels"], cfg["n_frames"]))
    for t in range(cfg["n_frames"]):
        spec = np.fft.rfft(yp[t * hop:t * hop + n_fft] * win)
        power = spec.real ** 2 + spec.imag ** 2
        for m, row in enumerate(cfg["filters"]):
            w = np.asarray(row["w"])
            out[m, t] = power[row["start"]:row["start"] + len(w)] @ w
    db = 10 * np.log10(np.maximum(cfg["amin"], out))
    db -= 10 * np.log10(max(cfg["amin"], out.max()))                           # ref = max
    db = np.maximum(db, db.max() - cfg["top_db"])
    return ((db + cfg["norm_offset"]) / cfg["norm_scale"]).astype(np.float32)


def quantize_weights_int8(src, dst):
    """Weight-only int8: Conv/Gemm weights are stored as int8 with one scale per output channel and
    turned back into float by DequantizeLinear at load time. Activations stay float32, so the
    model is 4x smaller with almost no change in its outputs (unlike dynamic activation quantisation)."""
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(src))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    targets = {n.input[1] for n in g.node if n.op_type in ("Conv", "Gemm") and n.input[1] in inits}
    for name in targets:
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        flat = w.reshape(w.shape[0], -1)
        scale = np.maximum(np.abs(flat).max(1), 1e-12) / 127.0
        q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        g.initializer.remove(inits[name])
        g.initializer.extend([numpy_helper.from_array(q, name + "_q"),
                              numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
                              numpy_helper.from_array(np.zeros(len(scale), np.int8), name + "_zp")])
        g.node.insert(0, helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"],
                                          [name], axis=0, name=name + "_dq"))
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


# ----------------------------------------------------------------------------- figures
def save_figures(df, X, classes, results, best_name, test_pred, yte):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, c in zip(axes.flat, classes):
        idx = df.index[df.category == c][0]
        ax.imshow(X[idx, 0].numpy(), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, DURATION, 0, N_MELS])
        ax.set_title(PRETTY[c], fontsize=12); ax.set_xlabel("seconds"); ax.set_yticks([])
    fig.suptitle("Log-mel spectrograms: one clip per class", fontsize=14)
    fig.tight_layout(); fig.savefig(ASSETS / "spectrograms.png", dpi=110); plt.close(fig)

    colors = {"From scratch": "#9aa2b5", "Frozen features": "#d98a00", "Full fine-tuning": "#4f46e5"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in results.items():
        h = pd.DataFrame(r["history"])
        ax.plot(h.epoch, h.val_acc, marker="o", ms=3, lw=2, color=colors[name], label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (fold 4)"); ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend(); ax.set_title("Three ways to train ResNet18")
    fig.tight_layout(); fig.savefig(ASSETS / "val_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yte, test_pred, display_labels=[PRETTY[c] for c in classes],
                                            cmap="Blues", colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title(f"Confusion matrix — {best_name} (test fold)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    torch.set_num_threads(os.cpu_count() or 4)
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = download_esc10()
    X, y, classes = build_dataset(df)
    splits = split_by_fold(df, X, y)
    print("Clips:", len(df), "| train/val/test:", [len(splits[k][1]) for k in ("train", "val", "test")],
          "| input:", tuple(X.shape[1:]))

    # 1) Three training strategies, same data, same budget
    results = run_strategies(splits)
    best_name = max(results, key=lambda k: results[k]["best_val_acc"])   # chosen on validation, not test
    best = results[best_name]["model"].eval()
    Xte, yte = splits["test"]
    proba = predict_proba(best, Xte).numpy()
    test_pred = proba.argmax(1)
    test_metrics = {"Accuracy": accuracy_score(yte, test_pred),
                    "Macro F1": f1_score(yte, test_pred, average="macro"),
                    "Top-3 accuracy": float(np.mean([t in np.argsort(-p)[:3] for p, t in zip(proba, yte.numpy())]))}
    print(f"\nBest on validation: {best_name} | test:", {k: round(v, 3) for k, v in test_metrics.items()})
    torch.save(best.state_dict(), DATA / "best_model.pt")

    save_figures(df, X, classes, results, best_name, test_pred, yte)

    # 2) ONNX export (fp32), checked against PyTorch, then int8 weight storage for the web (45 MB -> 11 MB)
    wrapper = ProbsWrapper(best).eval()
    fp32_path = DATA / "model_fp32.onnx"
    torch.onnx.export(wrapper, torch.zeros(1, 1, N_MELS, N_FRAMES), fp32_path, opset_version=17,
                      input_names=["mel"], output_names=["probs"],
                      dynamic_axes={"mel": {0: "batch"}, "probs": {0: "batch"}}, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_fp32 = sess.run(None, {"mel": Xte.numpy()})[0]
    fp32_diff = float(np.abs(onnx_fp32 - proba).max())
    print(f"ONNX fp32 vs PyTorch: max prob diff {fp32_diff:.2e}")
    assert fp32_diff < 1e-4

    quantize_weights_int8(fp32_path, WEB / "model.onnx")
    sess_q = ort.InferenceSession(str(WEB / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_q = sess_q.run(None, {"mel": Xte.numpy()})[0]
    q_agree = float(np.mean(onnx_q.argmax(1) == test_pred))
    q_acc = float(np.mean(onnx_q.argmax(1) == yte.numpy()))
    print(f"int8 model: {os.path.getsize(WEB / 'model.onnx') / 1e6:.1f} MB | top-1 agreement with PyTorch "
          f"{q_agree:.3f} | test acc {q_acc:.3f} | max prob diff {np.abs(onnx_q - proba).max():.3f}")

    # 3) Browser preprocessing parity: numpy port of web/mel.js vs librosa on every test clip
    cfg = export_mel_filters(WEB / "mel_filters.json")
    te_files = df.filename[df.fold == 5].tolist()
    js_feats = np.stack([js_logmel_numpy(load_audio(DATA / "esc10" / "audio" / f), cfg) for f in te_files])[:, None]
    mel_diff = float(np.abs(js_feats - Xte.numpy()).max())
    js_pred = sess_q.run(None, {"mel": js_feats})[0].argmax(1)
    js_agree = float(np.mean(js_pred == onnx_q.argmax(1)))
    print(f"JS-style log-mel vs librosa: max abs diff {mel_diff:.2e} (normalised units = {mel_diff * NORM_SCALE:.2e} dB)"
          f" | prediction agreement {js_agree:.3f}")
    assert mel_diff < 1e-3 and js_agree == 1.0

    # 4) Example clips for the demo: one test-fold clip per class that the model classifies correctly
    ex_dir = WEB / "examples"
    if ex_dir.exists():
        shutil.rmtree(ex_dir)
    ex_dir.mkdir()
    te_df = df[df.fold == 5].reset_index(drop=True)
    examples = []
    for ci, c in enumerate(classes):
        rows = te_df.index[(te_df.category == c) & (onnx_q.argmax(1) == ci)]
        r = te_df.loc[rows[0] if len(rows) else te_df.index[te_df.category == c][0]]
        shutil.copy(DATA / "esc10" / "audio" / r.filename, ex_dir / f"{c}.wav")
        examples.append({"file": f"examples/{c}.wav", "label": c, "name": PRETTY[c],
                         "source": f"https://freesound.org/s/{r.src_file}/"})

    comparison = [{"strategy": n, "pretrained": r["cfg"]["pretrained"], "frozen": r["cfg"]["freeze"],
                   "lr": r["cfg"]["lr"], "epochs": r["cfg"]["epochs"], "trainable_params": r["trainable_params"],
                   "best_val_acc": r["best_val_acc"], "test_acc": r["test_acc"],
                   "train_minutes": round(r["train_minutes"], 1)} for n, r in results.items()]
    meta = {
        "title": "Environmental Sound Classifier",
        "subtitle": ("Recognises 10 everyday sounds — dog bark, rain, crying baby, chainsaw and more — "
                     "from a 5-second clip, entirely in your browser."),
        "tags": ["Audio", "Log-mel spectrogram", "ResNet18", "Transfer learning", "PyTorch", "ONNX"],
        "classes": classes, "class_names": [PRETTY[c] for c in classes],
        "model_name": f"ResNet18 — {best_name.lower()}",
        "best_strategy": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "web_model": {"format": "ONNX, int8 weights", "size_mb": round(os.path.getsize(WEB / "model.onnx") / 1e6, 1),
                      "test_acc": q_acc, "agreement_with_pytorch": q_agree},
        "parity": {"max_abs_diff_normalised": mel_diff, "max_abs_diff_db": mel_diff * NORM_SCALE,
                   "prediction_agreement": js_agree, "clips": len(te_files)},
        "examples": examples,
        "figures": [
            {"src": "assets/val_curves.png", "caption": "Validation accuracy per epoch for the three strategies."},
            {"src": "assets/confusion_matrix.png", "caption": "Where the best model is right and wrong on the test fold."},
            {"src": "assets/spectrograms.png", "caption": "What the network actually sees: one log-mel spectrogram per class."},
        ],
        "steps": [
            "Took the ESC-10 subset of ESC-50: 400 five-second clips, 10 classes, using the official folds (1–3 train, 4 validation, 5 test).",
            f"Turned each clip into a {N_MELS}×{N_FRAMES} log-mel spectrogram with librosa ({SR} Hz, n_fft {N_FFT}, hop {HOP}).",
            "Trained ResNet18 three ways with the same budget: from scratch, frozen ImageNet features, full fine-tuning.",
            "Used light augmentation (time shift + SpecAugment masks) and picked the best epoch on the validation fold.",
            f"Chose {best_name.lower()} by validation accuracy and evaluated it once on the test fold.",
            "Exported it to ONNX (int8 weights) and re-implemented the log-mel pipeline in JavaScript, verified against librosa.",
        ],
        "dataset": {"name": "ESC-50 (ESC-10 subset)", "url": "https://github.com/karolpiczak/ESC-50",
                    "license": "CC BY-NC 3.0", "clips": len(df)},
        "repo_url": "https://github.com/ravshanovna-d/environmental-sound-classifier",
        "notebook_url": "https://github.com/ravshanovna-d/environmental-sound-classifier/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    }
    (WEB / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote web/model.onnx, web/mel_filters.json, web/meta.json, web/examples/, web/assets/")


if __name__ == "__main__":
    main()
