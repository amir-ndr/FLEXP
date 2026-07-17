"""
data/loaders/ham10000.py: HAM10000 ("Human Against Machine with 10000
training images") skin-lesion dataset loader.

Unlike MNIST/CIFAR-10/CIFAR-100, HAM10000 is NOT distributed via torchvision
and cannot be auto-downloaded — it requires registering with the ISIC
archive / Harvard Dataverse, or downloading the Kaggle mirror ("Skin Cancer
MNIST: HAM10000"). This loader reads it from a LOCAL folder you point it at
via `root` (config: data.ham10000_root).

Expected layout of `root` (exact subfolder names/depth do not matter — image
files are found by a recursive search, keyed by filename stem):
    <root>/
      HAM10000_metadata.csv        # columns: lesion_id, image_id, dx, dx_type, age, sex, localization
      <any subfolder(s)>/*.jpg     # e.g. HAM10000_images_part_1/, HAM10000_images_part_2/
                                    # (however Kaggle/ISIC split them up — both
                                    # parts, or one merged folder, all work;
                                    # this loader just globs recursively)

Labels: the `dx` column has exactly 7 possible values (diagnostic
categories), mapped to a FIXED canonical index (alphabetical, so label
indices are reproducible across machines/downloads regardless of which rows
happen to be present):
    akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6

No official train/test split ships with HAM10000. Splitting per-IMAGE would
leak information: many lesions (lesion_id) were photographed more than once,
and if two photos of the SAME lesion land on opposite sides of the split, the
test set is contaminated by near-duplicates the model has already seen. This
loader instead splits by LESION_ID (a grouped split) — every image of a given
lesion goes to the same side — using a fixed seed for reproducibility. Because
lesions carry different numbers of photos, the resulting image-level split
ratio is only approximately test_frac, not exact.

Image resolution: HAM10000's native images are ~600x450. Per this project's
convention (matching how CIFAR-10/CIFAR-100 already fit the framework's
32x32-native CIFAR-adapted models — see flsim/models/{alexnet,vgg,resnet,
cifar_cnn}.py), images are resized down to 32x32 RGB so HAM10000 drops into
every existing 3-channel model with ZERO model changes.

Normalisation constants below are commonly-cited approximate HAM10000/ISIC
statistics from the literature — this loader has not been run against a real
local copy of the dataset yet (none was available at implementation time).
Recompute from your own downloaded copy for exact per-channel statistics if
precision matters for your experiment.
"""

import csv
import os
from typing import List, Tuple

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

_DEFAULT_DATA_ROOT = os.path.expanduser("~/.flsim/data/ham10000")

# Fixed, alphabetical canonical label order — reproducible regardless of the
# order classes happen to appear in a given metadata CSV.
_HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
_CLASS_TO_IDX = {c: i for i, c in enumerate(_HAM10000_CLASSES)}

# Commonly-cited approximate HAM10000/ISIC per-channel statistics (see module
# docstring) — recompute from your local copy for exact values.
_HAM10000_MEAN = (0.7630, 0.5456, 0.5700)
_HAM10000_STD  = (0.1409, 0.1523, 0.1694)

_IMAGE_EXTS = (".jpg", ".jpeg")


def _index_images(root: str) -> dict:
    """Recursively map {filename stem: full path} for every .jpg/.jpeg under root."""
    index = {}
    dupes = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in _IMAGE_EXTS:
                continue
            if stem in index:
                dupes += 1
                continue
            index[stem] = os.path.join(dirpath, fname)
    if dupes:
        print(
            f"[load_ham10000] Warning: {dupes} duplicate image_id filename(s) "
            f"found under {root}; kept the first occurrence of each."
        )
    return index


def _read_metadata(csv_path: str) -> List[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


class HAM10000Dataset(Dataset):
    """
    One split (train or test) of HAM10000: a fixed list of (image_path, label)
    pairs, resolved once up front by load_ham10000().

    Exposes .targets (list[int]) for flsim.data.{shard,dirichlet} partitioners,
    matching the convention of torchvision's MNIST/CIFAR datasets.

    This class does NOT:
    - Parse the metadata CSV or resolve image paths (load_ham10000() does that
      once, up front, and hands this class the resolved record list).
    - Decide the train/test split.
    """

    def __init__(self, records: List[Tuple[str, int]], transform):
        self._records = records
        self.transform = transform
        self.targets = [label for _, label in records]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        path, label = self._records[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_ham10000(
    root: str = _DEFAULT_DATA_ROOT,
    image_size: int = 32,
    test_frac: float = 0.2,
    seed: int = 42,
    metadata_filename: str = "HAM10000_metadata.csv",
):
    """
    Load HAM10000 train/test datasets from a local folder.

    Args:
        root (str): directory containing metadata_filename and the image
            subfolder(s) (searched recursively — see module docstring).
        image_size (int): images are resized to (image_size, image_size).
            Default 32, matching this framework's CIFAR-adapted models.
        test_frac (float): fraction of LESIONS (not images) held out for test.
        seed (int): RNG seed for the lesion-level train/test split.
        metadata_filename (str): name of the metadata CSV inside root.

    Returns:
        tuple[HAM10000Dataset, HAM10000Dataset]: (train_dataset, test_dataset),
        each exposing .targets (7-class, see module docstring) for the
        non-IID partitioners.

    Raises:
        FileNotFoundError: if the metadata CSV is missing, or no images
            anywhere under root, or none of the metadata's image_ids can be
            matched to a found image (wrong path/layout).
        ValueError: if the CSV contains a dx value outside the expected 7.
    """
    csv_path = os.path.join(root, metadata_filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"HAM10000 metadata CSV not found at {csv_path}. Point `root` at "
            f"the folder containing {metadata_filename} and the image "
            f"subfolder(s) (see flsim.data.loaders.ham10000 module docstring)."
        )
    rows = _read_metadata(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty.")

    image_index = _index_images(root)
    if not image_index:
        raise FileNotFoundError(
            f"No .jpg/.jpeg images found anywhere under {root}. Point `root` "
            f"at the folder containing the HAM10000 image subfolder(s)."
        )

    unknown_dx = {r["dx"] for r in rows} - set(_HAM10000_CLASSES)
    if unknown_dx:
        raise ValueError(
            f"Unexpected dx label(s) {sorted(unknown_dx)} in {csv_path} — "
            f"expected only {_HAM10000_CLASSES}."
        )

    # Resolve each metadata row to an image path; skip rows whose image is
    # missing (partial local copy) rather than failing outright.
    resolved = []   # list of (path, label, lesion_id)
    missing = 0
    for r in rows:
        path = image_index.get(r["image_id"])
        if path is None:
            missing += 1
            continue
        resolved.append((path, _CLASS_TO_IDX[r["dx"]], r["lesion_id"]))
    if not resolved:
        raise FileNotFoundError(
            f"Found {len(image_index)} image file(s) under {root}, but none "
            f"match any image_id in {csv_path}. Check that root points at the "
            f"correct HAM10000 download."
        )
    if missing:
        print(
            f"[load_ham10000] Warning: {missing}/{len(rows)} metadata rows "
            f"have no matching image file under {root} — proceeding with "
            f"the {len(resolved)} that do."
        )

    # Grouped (by lesion_id) train/test split — avoids leaking multiple
    # photos of the same lesion across the split (see module docstring).
    lesion_ids = sorted(set(lid for _, _, lid in resolved))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(lesion_ids)
    n_test_lesions = max(1, int(round(test_frac * len(shuffled))))
    test_lesions = set(shuffled[:n_test_lesions])

    train_records = [(p, y) for p, y, lid in resolved if lid not in test_lesions]
    test_records  = [(p, y) for p, y, lid in resolved if lid in test_lesions]

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(_HAM10000_MEAN, _HAM10000_STD),
    ])

    train = HAM10000Dataset(train_records, transform=transform)
    test = HAM10000Dataset(test_records, transform=transform)
    return train, test


def list_classes() -> list:
    """Canonical class order (index -> dx label), e.g. for labeling plots/confusion matrices."""
    return list(_HAM10000_CLASSES)
