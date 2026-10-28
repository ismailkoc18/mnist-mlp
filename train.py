"""
DENEY 1 - Temel MLP modeli

Çalıştırma:
    python train.py

Çıktılar:
    models/mlp_baseline.pt              -> eğitilmiş model (kendi el yazınızı test etmek için)
    results/baseline_results.json       -> tüm metrikler + epoch bazlı geçmiş
    results/baseline_loss.png           -> train / validation loss grafiği
    results/baseline_accuracy.png       -> train / validation accuracy grafiği
    results/baseline_confusion.png      -> test seti karışıklık matrisi
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import MODELS_DIR, RESULTS_DIR, predict_all, get_device, run_experiment, save_json

# ---- HİPERPARAMETRELER (raporda belirtilmesi gerekenler) ----
CONFIG = {
    "name": "baseline",
    "hidden_sizes": [256, 128],   # 2 hidden layer
    "activation": "ReLU",
    "dropout": 0.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 15,
    "loss": "CrossEntropyLoss",
    "seed": 42,
}


def plot_curves(history, out_prefix, title):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], "o-", label="Training loss")
    plt.plot(epochs, history["val_loss"], "s-", label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (Cross-Entropy)")
    plt.title(f"{title} - Loss"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_prefix}_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Training accuracy")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title} - Accuracy"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_prefix}_accuracy.png", dpi=150); plt.close()


def plot_confusion(preds, labels, path):
    cm = np.zeros((10, 10), dtype=int)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek"); plt.title("Test seti - Karışıklık matrisi")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    model, result, test_loader = run_experiment(CONFIG)

    torch.save({"state_dict": model.state_dict(), "config": CONFIG},
               os.path.join(MODELS_DIR, "mlp_baseline.pt"))
    save_json(result, os.path.join(RESULTS_DIR, "baseline_results.json"))

    prefix = os.path.join(RESULTS_DIR, "baseline")
    plot_curves(result["history"], prefix, "Deney 1 - Temel MLP")
    preds, labels = predict_all(model, test_loader, get_device())
    plot_confusion(preds, labels, f"{prefix}_confusion.png")

    print("\n===== DENEY 1 ÖZET =====")
    print(f"Training loss      : {result['final_train_loss']:.4f}")
    print(f"Validation loss    : {result['final_val_loss']:.4f}")
    print(f"Training accuracy  : {result['final_train_acc']*100:.2f}%")
    print(f"Validation accuracy: {result['final_val_acc']*100:.2f}%")
    print(f"Test accuracy      : {result['test_acc']*100:.2f}%  (epoch {result['best_epoch']} ağırlıkları)")
    print(f"Eğitim süresi      : {result['train_time_sec']:.1f} sn")
    print(f"Grafikler ve sonuçlar: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
