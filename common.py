"""
Ortak yardımcılar: veri yükleme (train / validation / test ayrımı), MLP modeli,
eğitim ve değerlendirme döngüleri.

CNN KULLANILMAZ. Model yalnızca tam bağlı (Linear) katmanlardan oluşur.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST eğitim setinin piksel ortalaması ve standart sapması (normalizasyon için)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def set_seed(seed: int = 42):
    """Sonuçların tekrarlanabilir olması için tüm rastgelelik kaynaklarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size=128, val_size=10_000, seed=42):
    """
    MNIST'i indirir ve 3 parçaya ayırır:
      * Train      : 50.000 görüntü  (resmi train setinin 50.000'i)  -> ağırlıkları öğrenmek için
      * Validation : 10.000 görüntü  (resmi train setinin kalan 10.000'i) -> model seçimi / hiperparametre
      * Test       : 10.000 görüntü  (resmi test seti) -> SADECE en sonda, bir kez raporlamak için
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0,255] -> [0,1], şekil (1,28,28)
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)    # ayrım her çalıştırmada aynı olsun
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Multi-Layer Perceptron.
    Giriş: 28x28 görüntü -> 784 boyutlu vektöre düzleştirilir (Flatten).
    hidden_sizes=[256, 128] ise mimari: 784 -> 256 -> ReLU -> 128 -> ReLU -> 10
    Son katmanda softmax YOK; CrossEntropyLoss softmax'ı kendi içinde uygular.
    """

    def __init__(self, hidden_sizes=(256, 128), dropout=0.0, input_size=784, num_classes=10):
        super().__init__()
        layers = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


# ---------------------------------------------------------------------------
# EĞİTİM / DEĞERLENDİRME
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def run_experiment(config, verbose=True):
    """
    Bir hiperparametre yapılandırmasıyla modeli SIFIRDAN eğitir.
    Her epoch sonunda validation ölçülür; en düşük validation loss'a sahip
    ağırlıklar saklanır. Test seti yalnızca eğitim bittikten sonra, seçilen
    model üzerinde bir kez değerlendirilir.
    """
    set_seed(config.get("seed", 42))
    device = get_device()

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"], seed=config.get("seed", 42))

    model = MLP(hidden_sizes=config["hidden_sizes"], dropout=config.get("dropout", 0.0)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["lr"])

    if verbose:
        print(model)
        print(f"Eğitilebilir parametre sayısı: {count_params(model):,}")
        print(f"Cihaz: {device}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, best_epoch = float("inf"), None, 0

    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:02d}/{config['epochs']} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
                  f"val loss {va_loss:.4f} acc {va_acc*100:.2f}%")
    train_time = time.time() - start

    # En iyi (validation'a göre) ağırlıkları geri yükle, sonra TEK SEFER test et
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    result = {
        "config": config,
        "num_params": count_params(model),
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_loss": best_val_loss,
        "best_val_acc": history["val_acc"][best_epoch - 1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_sec": train_time,
        "history": history,
    }
    if verbose:
        print(f"\nEn iyi epoch (val loss'a göre): {best_epoch}")
        print(f"Test accuracy: {test_acc*100:.2f}%  |  Test loss: {test_loss:.4f}")
        print(f"Eğitim süresi: {train_time:.1f} sn")
    return model, result, test_loader


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
