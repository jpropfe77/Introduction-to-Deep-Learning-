"""
GenAI acknowledgment: Claude (Anthropic, Claude Opus 5.5) was used to scaffold code for the
models, training, evaluation, and plotting; all code was reviewed, tested,
and edited by me.

Assignment 5: regular CNN, ResNet-18, and small U-Net on Oxford-IIIT Pet.
Requires the provided, unmodified data.py in the same folder.

Run everything (writes requirements.txt from the current environment, explores,
trains all three models for 50 epochs, evaluates):
    python data.py
    python assignment5.py

Run one stage at a time:
    python assignment5.py --stage requirements
    python assignment5.py --stage explore
    python assignment5.py --stage train --model cnn
    python assignment5.py --stage train --model unet --resume
    python assignment5.py --stage evaluate

Outputs:
    requirements.txt  (exact package versions of the environment that ran this script)
    results/data_info.txt, results/losses_<model>.json, results/metrics.json, results/metrics.md
    figures/sample_images.png, figures/classifier_loss.png, figures/unet_loss.png,
    figures/segmentation_examples.png
    <ckpt-dir>/<model>.pt  (checkpoints; do not commit)
"""
import argparse
import importlib
import json
import os
import random
import time
import matplotlib
matplotlib.use("Agg")            # save figures to files only; no display needed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
from tqdm import tqdm

from data import CLASS_NAMES, get_loaders, get_raw_datasets

SEED = 0
EPOCHS = 50
MODEL_NAMES = ["cnn", "resnet", "unet"]
MODEL_LABELS = {"cnn": "Regular CNN", "resnet": "ResNet-18", "unet": "Small U-Net"}

# One fixed 3-color map for every mask plot: pet, background, boundary.
# Processed masks use 0,1,2 (vmin=0, vmax=2); raw masks use 1,2,3 (vmin=1, vmax=3).
MASK_CMAP = ListedColormap(["#e69f00", "#56b4e9", "#000000"])


# ===========================================================================
# Utilities
# ===========================================================================
def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use a CUDA GPU when available, otherwise the CPU."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"CUDA available: {use_cuda} | device: {device} ({name})")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# (pip package name, import name) for every package this project imports.
REQUIRED_PACKAGES = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
    ("matplotlib", "matplotlib"), ("scikit-learn", "sklearn"), ("tqdm", "tqdm"),
    ("pillow", "PIL"),
]


def write_requirements(path="requirements.txt"):
    """Record the exact versions installed in the environment running this script.

    Local build tags such as '+cpu' or '+cu126' are removed because pip cannot
    install them from PyPI by that name.
    """
    lines = []
    for pip_name, module_name in REQUIRED_PACKAGES:
        version = importlib.import_module(module_name).__version__.split("+")[0]
        lines.append(f"{pip_name}=={version}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}:\n  " + "\n  ".join(lines))


# ===========================================================================
# Models
# ===========================================================================
class SimpleCNN(nn.Module):
    """Model 1: regular CNN with global average pooling and two output scores."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)       # raw scores; no softmax before CrossEntropyLoss


def build_resnet18(num_classes=2):
    """Model 2: torchvision ResNet-18, random initialization, 2-class output layer."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)    # replaced before optimizer
    return model


class DoubleConv(nn.Module):
    """(3x3 conv -> ReLU) x 2; padding=1 keeps height and width unchanged."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """Model 3: small U-Net with encoder-to-decoder skip connections."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                           # [B,16,128,128]
        e2 = self.enc2(self.pool(e1))               # [B,32,64,64]
        b = self.bottleneck(self.pool(e2))          # [B,64,32,32]
        d1 = self.up1(b)                            # [B,32,64,64]
        d1 = self.dec1(torch.cat([d1, e2], dim=1))  # cat -> [B,64,64,64] -> [B,32,64,64]
        d2 = self.up2(d1)                           # [B,16,128,128]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # cat -> [B,32,128,128] -> [B,16,128,128]
        return self.out(d2)                         # [B,3,128,128] raw scores


MODEL_BUILDERS = {"cnn": SimpleCNN, "resnet": build_resnet18, "unet": SmallUNet}


# ===========================================================================
# Part 1: data exploration
# ===========================================================================
def explore(args):
    print("\n===== Part 1: Data exploration =====")
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    set_seed(args.seed)
    train_dataset, test_dataset = get_raw_datasets()
    log(f"Training samples: {len(train_dataset)}")
    log(f"Testing samples:  {len(test_dataset)}")

    image, (label, mask) = train_dataset[0]
    log(f"First image size (width, height): {image.size}")
    log(f"First image label: {label} ({CLASS_NAMES[label]})")
    log(f"First mask size (width, height):  {mask.size}")
    log(f"First mask unique values: {np.unique(np.array(mask)).tolist()} "
        f"(1=pet, 2=background, 3=boundary)")

    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for i in range(3):
        img, (lbl, msk) = train_dataset[i]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}: {CLASS_NAMES[lbl]} {img.size}")
        axes[i, 1].imshow(np.array(msk), cmap=MASK_CMAP, vmin=1, vmax=3)
        axes[i, 1].set_title(f"Mask {i} (orange=pet, blue=bg, black=boundary)", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/sample_images.png", dpi=120)
    plt.close(fig)
    log("Saved figures/sample_images.png")

    train_loader, test_loader = get_loaders(batch_size=16)
    images, (labels, masks) = next(iter(train_loader))
    log(f"images shape: {tuple(images.shape)}  dtype={images.dtype}  "
        f"range=[{images.min():.3f}, {images.max():.3f}]")
    log(f"labels shape: {tuple(labels.shape)}  dtype={labels.dtype}")
    log(f"masks shape:  {tuple(masks.shape)}  dtype={masks.dtype}  "
        f"unique={masks.unique().tolist()}")

    with open(os.path.join(args.results_dir, "data_info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ===========================================================================
# Part 2: training
# ===========================================================================
def train_model(name, args, device):
    print(f"\n===== Part 2: Training {MODEL_LABELS[name]} =====")
    set_seed(args.seed)                 # same seed and shuffle order for every model
    train_loader, _ = get_loaders(batch_size=args.batch_size)

    model = MODEL_BUILDERS[name]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Trainable parameters: {count_parameters(model):,}")

    images, (labels, masks) = next(iter(train_loader))
    with torch.no_grad():
        print(f"Output shape check: {tuple(model(images.to(device)).shape)}")

    ckpt_path = os.path.join(args.ckpt_dir, f"{name}.pt")
    losses, epoch_times, start_epoch = [], [], 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        losses, epoch_times, start_epoch = ckpt["losses"], ckpt["epoch_times"], ckpt["epoch"]
        torch.set_rng_state(ckpt["cpu_rng"])
        if torch.cuda.is_available() and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, n_seen, t0 = 0.0, 0, time.time()
        for images, (labels, masks) in tqdm(train_loader, desc=f"{name} epoch {epoch + 1}",
                                            leave=False):
            images = images.to(device)
            targets = (masks if name == "unet" else labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        losses.append(running_loss / n_seen)
        epoch_times.append(time.time() - t0)
        print(f"{name} | epoch {epoch + 1}/{args.epochs} | train loss {losses[-1]:.4f} "
              f"| {epoch_times[-1]:.0f}s")

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "losses": losses, "epoch_times": epoch_times,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, ckpt_path)
        record = {"model": name, "losses": losses, "epoch_times_s": epoch_times,
                  "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                  "optimizer": "Adam", "seed": args.seed, "device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        with open(os.path.join(args.results_dir, f"losses_{name}.json"), "w") as f:
            json.dump(record, f, indent=2)


# ===========================================================================
# Parts 3-4: evaluation and figures
# ===========================================================================
def load_model(name, ckpt_dir, device):
    ckpt = torch.load(os.path.join(ckpt_dir, f"{name}.pt"), map_location="cpu", weights_only=True)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {name} after {ckpt['epoch']} epochs")
    return model.to(device).eval()


def evaluate_classifier(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, (labels, masks) in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
    return accuracy_score(y_true, y_pred), y_true


def evaluate_unet(model, loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, (labels, masks) in loader:
            predicted_masks = model(images.to(device)).argmax(dim=1)
            masks = masks.to(device)
            correct += (predicted_masks == masks).sum().item()
            total += masks.numel()
    return correct / total


def plot_losses(results_dir):
    rec = {}
    for name in MODEL_NAMES:
        with open(os.path.join(results_dir, f"losses_{name}.json")) as f:
            rec[name] = json.load(f)["losses"]

    plt.figure(figsize=(6, 4))
    for name in ["cnn", "resnet"]:
        plt.plot(range(1, len(rec[name]) + 1), rec[name], label=MODEL_LABELS[name])
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss (cross-entropy)")
    plt.title("Classifier training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/classifier_loss.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rec["unet"]) + 1), rec["unet"], label=MODEL_LABELS["unet"])
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss (per-pixel cross-entropy)")
    plt.title("U-Net training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/unet_loss.png", dpi=120)
    plt.close()


def plot_segmentation(model, loader, device):
    images, (labels, masks) = next(iter(loader))     # test loader is not shuffled
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(3):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Test image {i} ({CLASS_NAMES[int(labels[i])]})")
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap=MASK_CMAP, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(preds[i].cpu().numpy(), cmap=MASK_CMAP, vmin=0, vmax=2)
        axes[i, 2].set_title("U-Net prediction")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("orange = pet, blue = background, black = boundary")
    plt.tight_layout()
    plt.savefig("figures/segmentation_examples.png", dpi=120)
    plt.close(fig)


def evaluate(args, device):
    print("\n===== Parts 3-4: Evaluation =====")
    set_seed(args.seed)
    _, test_loader = get_loaders(batch_size=args.batch_size)

    metrics = {}
    for name in ["cnn", "resnet"]:
        acc, y_true = evaluate_classifier(load_model(name, args.ckpt_dir, device),
                                          test_loader, device)
        metrics[f"{name}_accuracy"] = acc
    metrics["majority_dog_baseline"] = sum(1 for y in y_true if y == 1) / len(y_true)

    unet = load_model("unet", args.ckpt_dir, device)
    metrics["unet_pixel_accuracy"] = evaluate_unet(unet, test_loader, device)
    plot_segmentation(unet, test_loader, device)
    plot_losses(args.results_dir)

    table = ("| Model | Test accuracy |\n|---|---|\n"
             f"| Regular CNN | {metrics['cnn_accuracy']:.4f} |\n"
             f"| ResNet-18 | {metrics['resnet_accuracy']:.4f} |\n"
             f"| Majority-class baseline (always Dog) | {metrics['majority_dog_baseline']:.4f} |\n"
             f"\nU-Net pixel accuracy: {metrics['unet_pixel_accuracy']:.4f}\n")
    print("\n" + table)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.results_dir, "metrics.md"), "w") as f:
        f.write(table)


# ===========================================================================
# Main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CNN / ResNet-18 / U-Net assignment")
    p.add_argument("--stage", choices=["requirements", "explore", "train", "evaluate", "all"], default="all")
    p.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all",
                   help="which model(s) to train in the train stage")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--resume", action="store_true", help="continue from <ckpt-dir>/<model>.pt")
    args, _ = p.parse_known_args()      # ignores extra arguments Jupyter/Colab may pass
    return args


def main():
    args = parse_args()
    for folder in ["figures", args.results_dir, args.ckpt_dir]:
        os.makedirs(folder, exist_ok=True)
    print(f"Random seed: {args.seed} | epochs: {args.epochs}")

    if args.stage in ("requirements", "all"):
        write_requirements()
        if args.stage == "requirements":
            return

    if args.stage in ("explore", "all"):
        explore(args)

    device = get_device()
    if args.stage in ("train", "all"):
        names = MODEL_NAMES if args.model == "all" else [args.model]
        for name in names:
            train_model(name, args, device)

    if args.stage in ("evaluate", "all"):
        evaluate(args, device)


if __name__ == "__main__":
    main()
