"""
Provided data code for the CNN / ResNet-18 / U-Net assignment.
You do NOT need to change this file. Put it in your homework_cnn folder and import it.

Step 1 - download the data once (about 800 MB, usually 2-10 minutes, 1.6 GB on disk):

    python data.py

It downloads and extracts the dataset into ./data/oxford-iiit-pet/, then loads every
sample once to check it. Running it again skips the download and only re-checks (~20 s).

Step 2 - use it from your own scripts:

    from data import get_raw_datasets, get_loaders, CLASS_NAMES, MASK_NAMES

    # Original PIL images and masks (for Part 1: printing and plotting)
    raw_train, raw_test = get_raw_datasets()
    image, (label, mask) = raw_train[0]

    # Preprocessed mini-batches (for training and evaluation)
    train_loader, test_loader = get_loaders(batch_size=16)
    for images, (labels, masks) in train_loader:
        ...

What one batch from get_loaders() contains:

    images : float tensor [batch_size, 3, 128, 128], values in [0, 1]
    labels : long tensor  [batch_size]             0 = Cat, 1 = Dog
    masks  : long tensor  [batch_size, 128, 128]   0 = pet, 1 = background, 2 = boundary

Use `labels` as targets for the CNN and ResNet-18, and `masks` as targets for the U-Net.
"""
import os
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets.utils import download_url
from tqdm import tqdm

DATA_ROOT = "./data"   # dataset goes to ./data/oxford-iiit-pet/ (do not commit it to GitHub)

# The two archives of the Oxford-IIIT Pet dataset and their MD5 checksums.
ARCHIVES = [
    ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
]
IMAGE_SIZE = 128

CLASS_NAMES = ["Cat", "Dog"]                        # classification label -> name
MASK_NAMES = ["pet", "background", "boundary"]      # processed mask value -> name
# The ORIGINAL masks (from get_raw_datasets) use 1, 2, 3 instead of 0, 1, 2.

TARGET_TYPES = ["binary-category", "segmentation"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Images hold colors: resize, then ToTensor() gives a float tensor [3, H, W] in [0, 1].
# Note: Resize needs (128, 128). Resize(128) would keep the aspect ratio, and images of
# different shapes cannot be stacked into one batch.
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Masks hold category numbers, not colors, so they are handled differently:
#   - NEAREST interpolation keeps the values 1, 2, 3 (no blended values like 1.5).
#   - No ToTensor(), because it would divide the category numbers by 255.
_mask_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=transforms.InterpolationMode.NEAREST)


def mask_transform(mask):
    """PIL mask with values {1, 2, 3} -> long tensor [128, 128] with values {0, 1, 2}."""
    mask = _mask_resize(mask)
    mask = torch.from_numpy(np.array(mask)).long()
    return mask - 1     # CrossEntropyLoss expects class indices starting at 0


def target_transform(target):
    """The dataset gives (label, mask). Keep the label, preprocess the mask."""
    label, mask = target
    return label, mask_transform(mask)


# ---------------------------------------------------------------------------
# Download (with progress bars)
# ---------------------------------------------------------------------------

def _extract(archive_path, folder):
    """Extract a .tar.gz archive, showing progress by compressed bytes read."""
    name = os.path.basename(archive_path)
    with open(archive_path, "rb") as f, \
            tqdm.wrapattr(f, "read", total=os.path.getsize(archive_path),
                          desc=f"Extracting {name}", unit="B", unit_scale=True) as fobj, \
            tarfile.open(fileobj=fobj, mode="r|gz") as tar:
        if hasattr(tarfile, "data_filter"):     # safe extraction on newer Python versions
            tar.extractall(folder, filter="data")
        else:
            tar.extractall(folder)


def download_data(root=DATA_ROOT):
    """Download and extract the dataset once. Later calls return immediately."""
    folder = os.path.join(root, "oxford-iiit-pet")
    for url, md5 in ARCHIVES:
        archive_name = os.path.basename(url)
        done_marker = os.path.join(folder, archive_name + ".extracted")
        if os.path.exists(done_marker):
            continue
        print(f"Downloading {archive_name} (skipped if already downloaded)")
        download_url(url, folder, md5=md5)          # shows a download progress bar
        _extract(os.path.join(folder, archive_name), folder)
        open(done_marker, "w").close()              # written only after a complete extraction


# ---------------------------------------------------------------------------
# Datasets and loaders
# ---------------------------------------------------------------------------

def get_raw_datasets(root=DATA_ROOT):
    """Original, unprocessed data: PIL images of different sizes and PIL masks with values 1-3."""
    download_data(root)
    train = OxfordIIITPet(root=root, split="trainval", target_types=TARGET_TYPES)
    test = OxfordIIITPet(root=root, split="test", target_types=TARGET_TYPES)
    return train, test


def get_datasets(root=DATA_ROOT):
    """Preprocessed data: every sample is (image tensor, (label, mask tensor))."""
    download_data(root)
    kwargs = dict(root=root, target_types=TARGET_TYPES,
                  transform=image_transform, target_transform=target_transform)
    train = OxfordIIITPet(split="trainval", **kwargs)
    test = OxfordIIITPet(split="test", **kwargs)
    return train, test


def get_loaders(batch_size=16, root=DATA_ROOT):
    """Training loader is shuffled; testing loader keeps a fixed order.

    Lower batch_size (for example 8) if you run out of memory.
    """
    train, test = get_datasets(root)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# `python data.py` downloads the data and checks that everything works.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision
    print("torch", torch.__version__, "| torchvision", torchvision.__version__)
    download_data()

    # Preprocess every sample once to check that all images and masks load correctly.
    train_loader, test_loader = get_loaders(batch_size=64)
    for split, loader in [("train", train_loader), ("test", test_loader)]:
        label_counts = torch.zeros(2, dtype=torch.long)
        for images, (labels, masks) in tqdm(loader, desc=f"Checking {split} data", unit="batch"):
            assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
            assert masks.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
            assert masks.min() >= 0 and masks.max() <= 2
            label_counts += torch.bincount(labels, minlength=2)
        print(f"{split}: {len(loader.dataset)} samples "
              f"({label_counts[0]} {CLASS_NAMES[0]}, {label_counts[1]} {CLASS_NAMES[1]})")
    print("Data is ready.")
