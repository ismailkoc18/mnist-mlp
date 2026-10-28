"""
Eğitilmiş MLP modelini kendi el yazınız üzerinde test eder.

Kullanım:
    python predict_own.py --true 2021123456
        --true  : gerçek öğrenci numaranız (her fotoğraftaki rakam dizisiyle karşılaştırılır)
        --model : kullanılacak model (varsayılan models/mlp_baseline.pt)

Çıktılar:
    results/own_handwriting_predictions.png  -> her rakam, tahmin ve olasılık
    results/mnist_vs_own.png                 -> MNIST örnekleri ile kendi rakamlarınızın karşılaştırması
    results/own_handwriting_results.md       -> README'ye kopyalanabilir tablo
"""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from common import DATA_DIR, MLP, MNIST_MEAN, MNIST_STD, MODELS_DIR, RESULTS_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(ROOT, "handwritten", "processed")


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = MLP(hidden_sizes=cfg["hidden_sizes"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def load_digit(path):
    """28x28 PNG -> eğitimdekiyle AYNI normalizasyon: (x/255 - mean) / std"""
    arr = np.asarray(Image.open(path).convert("L").resize((28, 28)), dtype=np.float32) / 255.0
    x = (arr - MNIST_MEAN) / MNIST_STD
    return arr, torch.from_numpy(x).unsqueeze(0).unsqueeze(0)     # (1,1,28,28)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "mlp_baseline.pt"))
    parser.add_argument("--true", default=None, help="gerçek öğrenci numarası, ör. 2021123456")
    args = parser.parse_args()

    model, cfg = load_model(args.model)
    paths = sorted(glob.glob(os.path.join(PROC_DIR, "*.png")))
    if not paths:
        print("İşlenmiş rakam yok. Önce: python prepare_own_digits.py")
        return

    # Aynı fotoğraftan gelen rakamları grupla (dosya adı: <foto>_<sıra>.png)
    groups = defaultdict(list)
    for p in paths:
        src = os.path.basename(p).rsplit("_", 1)[0]
        groups[src].append(p)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md = ["| Görüntü | Gerçek | Tahmin | Doğru rakam | Rakam doğruluğu |", "|---|---|---|---|---|"]
    all_items, total_correct, total_digits = [], 0, 0

    with torch.no_grad():
        for src, plist in groups.items():
            preds, items = [], []
            for p in plist:
                arr, x = load_digit(p)
                prob = torch.softmax(model(x), dim=1)[0].numpy()
                pred = int(prob.argmax())
                preds.append(pred)
                items.append((src, arr, pred, prob))
            pred_str = "".join(map(str, preds))

            line = f"{src}: tahmin = {pred_str}"
            if args.true:
                true = args.true
                n_ok = sum(a == b for a, b in zip(pred_str, true))
                total_correct += n_ok
                total_digits += len(true)
                acc = n_ok / len(true)
                line += f" | gerçek = {true} | {n_ok}/{len(true)} doğru ({acc*100:.0f}%)"
                if len(pred_str) != len(true):
                    line += "  [UYARI: bulunan rakam sayısı farklı, segmentasyonu kontrol edin]"
                md.append(f"| {src} | {true} | {pred_str} | {n_ok}/{len(true)} | {acc*100:.0f}% |")
                items = [(s, a, p, pr, (true[i] if i < len(true) else "?")) for i, (s, a, p, pr) in enumerate(items)]
            else:
                md.append(f"| {src} | - | {pred_str} | - | - |")
                items = [(s, a, p, pr, "?") for (s, a, p, pr) in items]
            print(line)
            all_items.extend(items)

    if args.true and total_digits:
        summary = f"\n**Toplam rakam doğruluğu: {total_correct}/{total_digits} = {total_correct/total_digits*100:.1f}%**"
        print(summary.replace("**", ""))
        md.append(summary)
    with open(os.path.join(RESULTS_DIR, "own_handwriting_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- Tahmin görseli: rakam + en olası 3 sınıf ----
    n = len(all_items)
    cols = min(n, 10)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(1.6 * cols, 3.4 * rows), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] * rows})
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax_img, ax_bar = axes[2 * r, c], axes[2 * r + 1, c]
        if idx >= n:
            ax_img.axis("off"); ax_bar.axis("off"); continue
        src, arr, pred, prob, true = all_items[idx]
        ok = str(pred) == str(true)
        ax_img.imshow(arr, cmap="gray", vmin=0, vmax=1)
        color = "green" if ok else ("red" if true != "?" else "black")
        ax_img.set_title(f"Tahmin: {pred} ({prob[pred]*100:.0f}%)\nGerçek: {true}", fontsize=8, color=color)
        ax_img.axis("off")
        ax_bar.bar(range(10), prob, color=["tab:green" if k == pred else "tab:gray" for k in range(10)])
        ax_bar.set_xticks(range(10)); ax_bar.tick_params(labelsize=6); ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
    plt.suptitle("Kendi el yazım üzerinde MLP tahminleri", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "own_handwriting_predictions.png"), dpi=150)
    plt.close()

    # ---- MNIST ile karşılaştırma (distribution shift'i görmek için) ----
    try:
        from torchvision import datasets
        mnist = datasets.MNIST(DATA_DIR, train=False, download=True)
        k = min(10, n)
        fig, axes = plt.subplots(2, k, figsize=(1.3 * k, 3), squeeze=False)
        for i in range(k):
            axes[0, i].imshow(np.asarray(mnist[i][0]), cmap="gray"); axes[0, i].axis("off")
            axes[1, i].imshow(all_items[i][1], cmap="gray", vmin=0, vmax=1); axes[1, i].axis("off")
        axes[0, 0].set_title("MNIST test", fontsize=8, loc="left")
        axes[1, 0].set_title("Benim el yazım", fontsize=8, loc="left")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "mnist_vs_own.png"), dpi=150)
        plt.close()
    except Exception as e:  # karşılaştırma görseli opsiyonel
        print(f"MNIST karşılaştırma görseli oluşturulamadı: {e}")

    print(f"\nKullanılan model: {args.model}  (hidden={cfg['hidden_sizes']})")
    print(f"Görseller ve tablo: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
