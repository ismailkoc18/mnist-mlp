"""
DENEY 2 - Hiperparametre karşılaştırması

Temel modelden yalnızca BİR hiperparametre değiştirilerek etkisi ölçülür.
Model seçimi validation sonuçlarına göre yapılır; test accuracy sadece bilgi
amaçlı en sonda raporlanır.

Çalıştırma:
    python experiments.py            # tüm deneyler (CPU'da ~10-20 dk)
    python experiments.py --quick    # her deney 5 epoch (hızlı deneme)

Çıktılar:
    results/experiments.md           -> karşılaştırma tablosu (README'ye kopyalayın)
    results/experiments.json
    results/experiments_val_acc.png  -> tüm deneylerin validation accuracy eğrileri
    results/experiments_val_loss.png
"""
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, run_experiment, save_json

BASE = {
    "name": "baseline",
    "hidden_sizes": [256, 128],
    "dropout": 0.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 15,
    "seed": 42,
}

# (deney adı, temel yapılandırmaya göre değişiklik, açıklama)
VARIANTS = [
    ("baseline",        {},                                   "Temel model (Deney 1)"),
    ("lr_0.01",         {"lr": 1e-2},                         "Yüksek learning rate"),
    ("lr_0.0001",       {"lr": 1e-4},                         "Düşük learning rate"),
    ("sgd_0.01",        {"optimizer": "sgd", "lr": 1e-2},     "Optimizer: SGD (momentum=0.9)"),
    ("small_64x32",     {"hidden_sizes": [64, 32]},           "Küçük hidden layer'lar"),
    ("wide_512x256",    {"hidden_sizes": [512, 256]},         "Geniş hidden layer'lar"),
    ("deep_3layer",     {"hidden_sizes": [512, 256, 128]},    "3 hidden layer"),
    ("dropout_0.3",     {"dropout": 0.3},                     "Dropout = 0.3"),
    ("batch_32",        {"batch_size": 32},                   "Batch size = 32"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="her deney için 5 epoch")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for name, changes, desc in VARIANTS:
        cfg = copy.deepcopy(BASE)
        cfg.update(changes)
        cfg["name"] = name
        if args.quick:
            cfg["epochs"] = 5
        print("\n" + "=" * 70 + f"\nDENEY: {name}  ({desc})\n" + "=" * 70)
        _, res, _ = run_experiment(cfg, verbose=True)
        res["description"] = desc
        results.append(res)

    save_json(results, os.path.join(RESULTS_DIR, "experiments.json"))

    # ---- Markdown tablo ----
    lines = [
        "| Deney | Açıklama | Hidden | Opt. | LR | Batch | Dropout | Param. | Train loss | Val loss | Train acc | Val acc | Test acc | Süre (sn) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["config"]
        lines.append(
            f"| {c['name']} | {r['description']} | {'-'.join(map(str, c['hidden_sizes']))} | "
            f"{c['optimizer']} | {c['lr']} | {c['batch_size']} | {c['dropout']} | {r['num_params']:,} | "
            f"{r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
            f"{r['final_train_acc']*100:.2f}% | {r['final_val_acc']*100:.2f}% | "
            f"{r['test_acc']*100:.2f}% | {r['train_time_sec']:.1f} |")
    best = max(results, key=lambda r: r["best_val_acc"])
    lines.append("")
    lines.append(f"Validation accuracy'ye göre en iyi yapılandırma: **{best['config']['name']}** "
                 f"(val acc {best['best_val_acc']*100:.2f}%, test acc {best['test_acc']*100:.2f}%)")
    table = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "experiments.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)

    # ---- Karşılaştırma grafikleri ----
    for key, ylabel, fname in [("val_acc", "Validation accuracy (%)", "experiments_val_acc.png"),
                               ("val_loss", "Validation loss", "experiments_val_loss.png")]:
        plt.figure(figsize=(9, 5.5))
        for r in results:
            vals = r["history"][key]
            if key == "val_acc":
                vals = [v * 100 for v in vals]
            plt.plot(range(1, len(vals) + 1), vals, marker=".", label=r["config"]["name"])
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.grid(alpha=0.3)
        plt.title("Hiperparametre karşılaştırması")
        if key == "val_acc":
            plt.ylim(90, 100)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150); plt.close()

    print(f"\nSonuçlar kaydedildi: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
