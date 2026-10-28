import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATA_ROOT = "/home/groups/comp3710/ADNI"
CLASSES = ("NC", "AD")  # label 0 = cognitively normal, 1 = Alzheimer's disease
IMG_EXTS = {".jpeg", ".jpg", ".png"}
CROP = 224  # slices are 256 x 240; a 224 x 224 crop keeps the brain and drops empty border


def find_root(root):
    # Return the folder that directly contains train/ and test/ (the course copy is nested one level down).
    root = Path(root)
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root
    for cand in root.rglob("train"):
        if cand.is_dir() and (cand.parent / "test").is_dir():
            return cand.parent
    raise FileNotFoundError(f"No train/ and test/ folders found under {root}")


def patient_id(path):
    # '392586_92.jpeg' -> '392586': the part before the underscore is the patient, the rest is the slice number.
    return path.stem.split("_")[0]


def slice_number(path):
    # '392586_92.jpeg' -> 92
    return int(path.stem.split("_")[1])


def list_split(folder):
    # List (path, label, patient_id) for every slice in one split folder.
    items = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((folder / cls).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, label, patient_id(p)))
    return items


def patient_split(items, val_frac = 0.15, seed = 624):

    # Split the training slices into train/val by patient, so all slices of one patient land in one split.
    # Patients are shuffled within each class, so both splits keep the AD/NC balance.
    by_class = defaultdict(set)
    for _, label, pid in items:
        by_class[label].add(pid)

    rng = random.Random(seed)
    val_pids = set()
    for label in sorted(by_class):
        pids = sorted(by_class[label])
        rng.shuffle(pids)
        val_pids.update(pids[:round(len(pids) * val_frac)])

    train = [it for it in items if it[2] not in val_pids]
    val = [it for it in items if it[2] in val_pids]
    return train, val


def check_no_leakage(train, val, test):
    # Fail loudly if any patient appears in two splits.
    ids = [{it[2] for it in split} for split in (train, val, test)]
    assert not (ids[0] & ids[1]), "patient shared between train and val"
    assert not (ids[0] & ids[2]), "patient shared between train and test"
    assert not (ids[1] & ids[2]), "patient shared between val and test"


def pixel_stats(items, n = 500, seed = 624):

    # Mean and std of centre-cropped pixel values, estimated from a random sample of training slices only.
    sample = random.Random(seed).sample(items, min(n, len(items)))
    tf = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(CROP), transforms.ToTensor()])
    pixels = torch.cat([tf(Image.open(p)).flatten() for p, _, _ in sample])
    return pixels.mean().item(), pixels.std().item()


def load_grey(path):
    # Read one slice as an 8-bit greyscale image and close the file.
    with Image.open(path) as im:
        return im.convert("L")


def neighbours(items):
    # Map each slice to the slices just below and above it in its patient's stack (repeating itself at the ends).
    stacks = defaultdict(list)
    for p, _, pid in items:
        stacks[pid].append(p)
    out = {}
    for paths in stacks.values():
        paths.sort(key=slice_number)
        for i, p in enumerate(paths):
            out[p] = (paths[max(i - 1, 0)], paths[min(i + 1, len(paths) - 1)])
    return out


class SliceDataset(Dataset):
    def __init__(self, items, mean, std, train = False, context = False):
        # context=True gives 2.5D input: the slice below, the slice itself and the slice above as 3 channels,
        # so every augmentation is applied identically to all three.
        self.items = items
        self.neighbours = neighbours(items) if context else None
        if train:
            # Small random shifts, flips, rotations and brightness/contrast changes; no change to anatomy.
            aug = [transforms.RandomCrop(CROP), transforms.RandomHorizontalFlip(),
                   transforms.RandomAffine(degrees=10, scale=(0.95, 1.05)),
                   transforms.ColorJitter(brightness=0.1, contrast=0.1)]
        else:
            aug = [transforms.CenterCrop(CROP)]
        c = 3 if context else 1
        self.tf = transforms.Compose([*aug, transforms.ToTensor(), transforms.Normalize([mean] * c, [std] * c)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        if self.neighbours is None:
            return self.tf(load_grey(path)), label
        below, above = self.neighbours[path]
        return self.tf(Image.merge("RGB", [load_grey(q) for q in (below, path, above)])), label


class ADNIData:
    def __init__(self, train, val, test, mean, std, context = False):
        self.train = train  # list of (path, label, patient_id)
        self.val = val
        self.test = test
        self.mean = mean    # training-slice pixel mean, used to normalise every split
        self.std = std
        self.context = context  # True for 2.5D (3-slice) input

    def loader(self, split, batch_size = 32, workers = 4):
        # Only the training loader shuffles and augments; val/test keep file order so slices can be grouped by patient.
        items = getattr(self, split)
        is_train = split == "train"
        return DataLoader(SliceDataset(items, self.mean, self.std, train=is_train, context=self.context), batch_size,
                          shuffle=is_train, num_workers=workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=workers > 0)


def load_adni(root = DATA_ROOT, val_frac = 0.15, seed = 624, val_from = "train", context = False):

    # val_from="train": validation patients come out of the provided train folder and the test folder is untouched.
    # val_from="test": the whole train folder is used for training, and the test folder's patients are split 50/50
    # into validation and test. Validation patients from the train folder were found to be far easier than real
    # unseen patients (0.99 vs 0.74 accuracy), so they cannot guide early stopping or the decision cut-off.
    root = find_root(root)
    if val_from == "train":
        train, val = patient_split(list_split(root / "train"), val_frac, seed)
        test = list_split(root / "test")
    else:
        train = list_split(root / "train")
        test, val = patient_split(list_split(root / "test"), 0.5, seed)
    check_no_leakage(train, val, test)
    mean, std = pixel_stats(train, seed=seed)
    return ADNIData(train, val, test, mean, std, context)


def summarise(data):
    # Print patients and slices per class for each split.
    print(f"{'Split':<7}{'NC patients / slices':>22}{'AD patients / slices':>22}")
    for name in ("train", "val", "test"):
        items = getattr(data, name)
        cells = []
        for label in range(len(CLASSES)):
            sub = [it for it in items if it[1] == label]
            cells.append(f"{len({it[2] for it in sub})} / {len(sub)}")
        print(f"{name:<7}{cells[0]:>22}{cells[1]:>22}")
    print(f"Training pixel mean {data.mean:.4f}, std {data.std:.4f}. No patient is shared between splits.")


if __name__ == "__main__":
    import sys
    summarise(load_adni(sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT))
