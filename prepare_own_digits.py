"""
Kendi el yazısı görüntülerinizi MNIST formatına dönüştürür.

Kullanım:
    1) Kâğıda öğrenci numaranızı yazın, fotoğrafını çekin / tarayın.
    2) Fotoğrafı handwritten/raw/ klasörüne koyun (jpg / png).
       - Tüm numara tek fotoğrafta olabilir (rakamlar otomatik ayrılır),
       - ya da her rakam ayrı bir fotoğraf olabilir.
    3) python prepare_own_digits.py

MNIST formatına uyum için yapılan işlemler:
    * Gri tonlamaya çevirme
    * Aydınlatma düzeltme (gölge / kâğıt rengi farkını giderme)
    * Renk ters çevirme -> MNIST'teki gibi SİYAH arka plan, BEYAZ rakam
    * Otsu eşikleme ile rakamı arka plandan ayırma
    * Bağlı bileşenlerle rakamları tek tek bulma (soldan sağa)
    * Kalem çizgisini MNIST kalınlığına yaklaştırma
    * En-boy oranını koruyarak 20x20 kutuya sığdırma
    * 28x28 tuvale yerleştirip kütle merkezine göre ortalama (MNIST'in yöntemi)
    * [0,255] aralığına ölçekleme  (mean/std normalizasyonu predict_own.py'de,
      eğitimdekiyle AYNI değerlerle uygulanır)

Çıktılar:
    handwritten/processed/<dosya>_<sıra>.png   -> 28x28 MNIST benzeri rakamlar
    handwritten/preprocessing_steps.png        -> ön işleme adımlarının görseli
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "handwritten", "raw")
OUT_DIR = os.path.join(ROOT, "handwritten", "processed")


def otsu_threshold(arr):
    """Otsu yöntemiyle otomatik eşik değeri (0-255)."""
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, best_t, best_var = 0.0, 0, 0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def load_and_clean(path, max_side=1000):
    """Gri tonlama + aydınlatma düzeltme + ters çevirme. Sonuç: siyah zemin, beyaz yazı."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    gray = np.asarray(img, dtype=np.float32)

    # Arka planı büyük bir bulanıklaştırma ile tahmin et ve böl -> gölgeler kaybolur
    background = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=max(img.size) / 20)),
                            dtype=np.float32)
    flat = np.clip(gray / (background + 1e-6) * 255.0, 0, 255)

    # Arka plan açıksa (beyaz kâğıt) ters çevir: MNIST'te zemin siyah, rakam beyaz
    if np.median(flat) > 127:
        flat = 255.0 - flat
    return gray, flat


def find_digits(flat, min_area_ratio=0.02, border=0.02):
    """Rakamları bağlı bileşenlerle bulur, soldan sağa sıralı kutular döndürür."""
    t = otsu_threshold(flat.astype(np.uint8))
    mask = flat > t
    # Kenarlardaki gölge / kâğıt kenarı gürültüsünü at
    h, w = mask.shape
    bh, bw = int(h * border), int(w * border)
    mask[:bh, :] = mask[-bh or h:, :] = False
    mask[:, :bw] = mask[:, -bw or w:] = False
    mask = ndimage.binary_opening(mask, iterations=1)       # tek piksel gürültüleri sil

    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask, []
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    slices = ndimage.find_objects(labels)
    boxes = []
    for i, sl in enumerate(slices):
        if areas[i] < min_area_ratio * areas.max():
            continue                                         # küçük leke
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])  # x0, y0, x1, y1

    # Yatayda büyük ölçüde çakışan parçaları birleştir (ör. kopuk yazılmış 5'in şapkası)
    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - max(m[0], b[0])
            if overlap >= 0.5 * min(m[2] - m[0], b[2] - b[0]):
                merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                continue
        merged.append(b)

    # Rakam boyunun çok altında kalan küçük parçaları (kalem izi, leke) at
    if merged:
        med_h = np.median([b[3] - b[1] for b in merged])
        merged = [b for b in merged if (b[3] - b[1]) >= 0.4 * med_h]
    return mask, merged


def stroke_width(mask):
    """Ortalama kalem kalınlığı tahmini ≈ 2 * alan / çevre."""
    area = mask.sum()
    edge = mask & ~ndimage.binary_erosion(mask)
    return 2.0 * area / max(edge.sum(), 1)


def to_mnist(flat, mask, box, target_stroke=2.2):
    """Tek bir rakam kutusunu 28x28 MNIST formatına dönüştürür."""
    x0, y0, x1, y1 = box
    digit_mask = mask[y0:y1, x0:x1]
    crop = np.where(digit_mask, flat[y0:y1, x0:x1], 0)       # sadece rakam pikselleri
    crop = crop / max(crop.max(), 1) * 255.0

    # Kalem kalınlığını MNIST'e yaklaştır: 20 px'lik kutuda ~2-2.5 px çizgi
    side = max(crop.shape)
    target_hi = target_stroke * side / 20.0
    sw = stroke_width(digit_mask)
    if sw < target_hi:
        k = int(round(target_hi - sw))
        if k >= 1:
            size = 2 * ((k + 1) // 2) + 1                    # tek sayı filtre boyutu
            crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).filter(ImageFilter.MaxFilter(size)),
                              dtype=np.float32)

    # Kare tuvale koy (en-boy oranı korunur) + küçük kenar boşluğu
    h, w = crop.shape
    side = max(h, w)
    pad = int(side * 0.05)
    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.float32)
    square[pad + (side - h) // 2: pad + (side - h) // 2 + h,
           pad + (side - w) // 2: pad + (side - w) // 2 + w] = crop

    # 20x20'ye küçült (MNIST: rakam 20x20 kutuda, 28x28 içinde)
    small = np.asarray(Image.fromarray(square.astype(np.uint8)).resize((20, 20), Image.LANCZOS),
                       dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = small

    # Kütle merkezini (14,14)'e kaydır
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant", cval=0)
    canvas = np.clip(canvas, 0, 255)
    canvas = canvas / max(canvas.max(), 1) * 255.0
    return canvas.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
                   for p in glob.glob(os.path.join(args.raw, ext)))
    if not paths:
        print(f"'{args.raw}' klasöründe görüntü bulunamadı. El yazısı fotoğraflarınızı oraya koyun.")
        return

    # Eski çıktıları temizle
    for old in glob.glob(os.path.join(args.out, "*.png")):
        os.remove(old)

    figs = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        gray, flat = load_and_clean(path)
        mask, boxes = find_digits(flat)
        digits = [to_mnist(flat, mask, b) for b in boxes]
        for i, d in enumerate(digits):
            Image.fromarray(d).save(os.path.join(args.out, f"{name}_{i:02d}.png"))
        print(f"{os.path.basename(path)}: {len(digits)} rakam bulundu")
        figs.append((name, gray, flat, mask, boxes, digits))

    # Ön işleme adımlarını gösteren görsel
    n_rows = len(figs)
    max_d = max(1, max(len(f[5]) for f in figs))
    fig = plt.figure(figsize=(3 * 3 + 1.1 * max_d, 2.6 * n_rows))
    gs = fig.add_gridspec(n_rows, 3 + max_d, width_ratios=[3, 3, 3] + [1.1] * max_d)
    for r, (name, gray, flat, mask, boxes, digits) in enumerate(figs):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(gray, cmap="gray"); ax.set_title(f"{name}\n1) Orijinal (gri)", fontsize=8); ax.axis("off")
        ax = fig.add_subplot(gs[r, 1]); ax.imshow(flat, cmap="gray"); ax.set_title("2) Aydınlatma düzeltme + ters çevirme", fontsize=8); ax.axis("off")
        ax = fig.add_subplot(gs[r, 2]); ax.imshow(mask, cmap="gray"); ax.set_title("3) Eşikleme + rakam kutuları", fontsize=8); ax.axis("off")
        for (x0, y0, x1, y1) in boxes:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", lw=1))
        for i, d in enumerate(digits):
            ax = fig.add_subplot(gs[r, 3 + i]); ax.imshow(d, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"28x28 #{i}", fontsize=7); ax.axis("off")
    plt.tight_layout()
    out_fig = os.path.join(os.path.dirname(args.out), "preprocessing_steps.png")
    plt.savefig(out_fig, dpi=130); plt.close()
    print(f"\nİşlenmiş rakamlar: {args.out}\nÖn işleme görseli: {out_fig}")
    print("Rakam sayısı yanlışsa: fotoğrafı numaranın etrafından kırpın, rakamları birbirinden ayrık yazın.")


if __name__ == "__main__":
    main()
