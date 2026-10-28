import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


DATA_ROOT = "/home/groups/comp3710/ADNI"
CLASSES = ("NC", "AD")  # label 0 = cognitively normal, 1 = Alzheimer's disease
IMG_EXTS = {".jpeg", ".jpg", ".png"}
CROP = 224  # slices are 256 x 240; a 224 x 224 crop keeps the brain and drops empty border
SMALL = (30, 32)  # PIL (width, height) of a shrunken slice, used to compare whole scans
SAME_PERSON = 0.8  # scans more alike than this (cosine similarity) are treated as one person


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


class BrainScale:
    # Divide a slice by the 99th percentile of its brain pixels (the background is ~0) and cap at 1.5, so scans
    # from different scanner generations share one intensity scale. In train/ every AD scan is from an older id
    # range than every NC scan, so absolute brightness and contrast could otherwise act as a class cue.

    def __call__(self, x):
        brain = x[x > 0.02]
        if brain.numel() < 100:  # (almost) empty slice: leave it alone
            return x
        return (x / torch.quantile(brain, 0.99)).clamp(max=1.5)


def shared_slices(items, min_frac = 0.02):
    # Slice numbers that at least `min_frac` of the patients of every class have in their 20-slice window.
    # In train/ the window starts at different slice numbers for AD and NC, so slices outside this range would
    # tell the model the class by position alone.
    counts = Counter((slice_number(p), label) for p, label, _ in items)
    patients = Counter(label for label, _ in {(label, pid) for _, label, pid in items})
    need = {label: max(1, min_frac * n) for label, n in patients.items()}
    nums = {s for s, _ in counts}
    return sorted(s for s in nums if all(counts[(s, label)] >= need[label] for label in need))


def balance_weights(items):
    # Sampling weight per slice so that every (slice number, class) cell is drawn equally often:
    # the slice position then carries no information about the class.
    counts = Counter((slice_number(p), label) for p, label, _ in items)
    return [1.0 / counts[(slice_number(p), label)] for p, label, _ in items]


def pixel_stats(items, n = 500, seed = 624, brain_scale = False):

    # Mean and std of centre-cropped pixel values, estimated from a random sample of training slices only.
    sample = random.Random(seed).sample(items, min(n, len(items)))
    tf = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(CROP), transforms.ToTensor(),
                             *([BrainScale()] if brain_scale else [])])
    pixels = torch.cat([tf(Image.open(p)).flatten() for p, _, _ in sample])
    return pixels.mean().item(), pixels.std().item()


def load_grey(path):
    # Read one slice as an 8-bit greyscale image and close the file.
    with Image.open(path) as im:
        return im.convert("L")


def scan_vectors(stacks):
    # One vector per scan from its slice paths (in slice order, the same number for every scan): the slices shrunk
    # to 30 x 32 and stacked, minus the mean over all scans (the brain shape every scan shares), scaled to unit
    # length. The dot product of two vectors is then the cosine similarity of the scans beyond that shared shape.
    vecs = np.stack([np.concatenate([np.asarray(load_grey(p).resize(SMALL, Image.BILINEAR), np.float32).ravel()
                                     for p in paths]) for paths in stacks])
    vecs -= vecs.mean(0)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs


def person_groups(items, threshold = SAME_PERSON):
    # Group the scan ids of one folder into probable people. The file name names a scan, not a person: explore.py
    # found pairs with consecutive ids that look identical. Ids whose scans are more alike than `threshold` are
    # joined, together with everything they are joined to. Ids with an unusual number of slices cannot be compared
    # and stay on their own.
    stacks = defaultdict(list)
    for p, _, pid in items:
        stacks[pid].append(p)
    n = Counter(len(v) for v in stacks.values()).most_common(1)[0][0]
    pids = sorted(pid for pid, v in stacks.items() if len(v) == n)
    vecs = scan_vectors([sorted(stacks[pid], key=slice_number) for pid in pids])
    _, person = connected_components(vecs @ vecs.T > threshold, directed=False)
    groups = defaultdict(list)
    for pid, g in zip(pids, person):
        groups[g].append(pid)
    return list(groups.values()) + [[pid] for pid in sorted(stacks) if len(stacks[pid]) != n]


def person_folds(items, k = 3, seed = 624):
    # Split one folder's patients into k folds of whole people with the same AD/NC balance. Within each class the
    # people are shuffled, then placed largest first, each into the fold with the fewest patients of that class.
    labels = {pid: label for _, label, pid in items}
    by_class = defaultdict(list)
    groups = person_groups(items)
    for g in groups:
        by_class[Counter(labels[pid] for pid in g).most_common(1)[0][0]].append(g)
    rng = random.Random(seed)
    fold_of = {}
    for label in sorted(by_class):
        people = by_class[label]
        rng.shuffle(people)
        people.sort(key=len, reverse=True)
        sizes = [0] * k
        for g in people:
            f = sizes.index(min(sizes))
            sizes[f] += len(g)
            fold_of.update({pid: f for pid in g})
    print(f"{len(labels)} ids in {len(groups)} people (largest {max(map(len, groups))} ids) -> {k} folds")
    return [[it for it in items if fold_of[it[2]] == f] for f in range(k)]


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
    def __init__(self, items, mean, std, train = False, context = False, img_size = CROP, brain_scale = False):
        # context=True gives 2.5D input: the slice below, the slice itself and the slice above as 3 channels,
        # so every augmentation is applied identically to all three.
        # img_size > CROP enlarges the 224 x 224 crop, so the network's feature maps cover the brain in finer steps.
        self.items = items
        self.neighbours = neighbours(items) if context else None
        if train:
            # Small random shifts, flips, rotations and brightness/contrast changes; no change to anatomy.
            aug = [transforms.RandomCrop(CROP), transforms.RandomHorizontalFlip(),
                   transforms.RandomAffine(degrees=10, scale=(0.95, 1.05)),
                   transforms.ColorJitter(brightness=0.1, contrast=0.1)]
        else:
            aug = [transforms.CenterCrop(CROP)]
        if img_size != CROP:
            aug.append(transforms.Resize(img_size, antialias=True))
        c = 3 if context else 1
        self.tf = transforms.Compose([*aug, transforms.ToTensor(), *([BrainScale()] if brain_scale else []),
                                      transforms.Normalize([mean] * c, [std] * c)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        if self.neighbours is None:
            return self.tf(load_grey(path)), label
        below, above = self.neighbours[path]
        return self.tf(Image.merge("RGB", [load_grey(q) for q in (below, path, above)])), label


class ADNIData:
    def __init__(self, train, val, test, mean, std, context = False, img_size = CROP, brain_scale = False,
                 weights = None, slices = None):
        self.train = train  # list of (path, label, patient_id)
        self.val = val
        self.test = test
        self.mean = mean    # training-slice pixel mean, used to normalise every split
        self.std = std
        self.context = context        # True for 2.5D (3-slice) input
        self.img_size = img_size      # side length fed to the network
        self.brain_scale = brain_scale  # True to rescale each slice by its own brain intensity
        self.weights = weights        # per-training-slice sampling weights, or None for plain shuffling
        self.slices = slices          # slice numbers kept (shared by both classes), or None for all

    def loader(self, split, batch_size = 32, workers = 4):
        # Only the training loader shuffles and augments; val/test keep file order so slices can be grouped by patient.
        # With weights, training slices are drawn with replacement so each (slice number, class) cell is equally likely.
        items = getattr(self, split)
        is_train = split == "train"
        ds = SliceDataset(items, self.mean, self.std, train=is_train, context=self.context, img_size=self.img_size,
                          brain_scale=self.brain_scale)
        sampler = WeightedRandomSampler(self.weights, len(items)) if is_train and self.weights else None
        return DataLoader(ds, batch_size, shuffle=is_train and sampler is None, sampler=sampler,
                          num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def load_adni(root = DATA_ROOT, val_frac = 0.15, seed = 624, val_from = "train", context = False, img_size = CROP,
              balance = False, brain_scale = False, folds = 0, fold = 0):

    # val_from="train": validation patients come out of the provided train folder and the test folder is untouched.
    # val_from="test": the whole train folder is used for training, and the test folder's patients are split 50/50
    # into validation and test. Validation patients from the train folder were found to be far easier than real
    # unseen patients (0.99 vs 0.74 accuracy), so they cannot guide early stopping or the decision cut-off.
    # folds=k (cross-fitting, overrides val_from): the test folder is split into k folds of whole people. Fold `fold`
    # is the test set, the next fold is validation, and the other folds join the train folder for training, so the
    # model also sees newer-era AD scans, which train/ lacks. Running every fold scores each test-folder patient
    # once, with a model that never saw that person.
    root = find_root(root)
    if folds:
        assert folds >= 3 and 0 <= fold < folds, "cross-fitting needs at least 3 folds (test, validation, training)"
        parts = person_folds(list_split(root / "test"), folds, seed)
        test, val = parts[fold], parts[(fold + 1) % folds]
        extra = [it for f, part in enumerate(parts) if f not in (fold, (fold + 1) % folds) for it in part]
        train = list_split(root / "train") + extra
    elif val_from == "train":
        train, val = patient_split(list_split(root / "train"), val_frac, seed)
        test = list_split(root / "test")
    else:
        train = list_split(root / "train")
        test, val = patient_split(list_split(root / "test"), 0.5, seed)
    check_no_leakage(train, val, test)

    # balance=True: keep only slice numbers both classes have in train/ (in every split, so each patient's decision
    # uses the same brain region), and sample training slices evenly over (slice number, class).
    slices = weights = None
    if balance:
        slices = shared_slices(train)
        keep = set(slices)
        train, val, test = ([it for it in s if slice_number(it[0]) in keep] for s in (train, val, test))
        weights = balance_weights(train)
    mean, std = pixel_stats(train, seed=seed, brain_scale=brain_scale)
    return ADNIData(train, val, test, mean, std, context, img_size, brain_scale, weights, slices)


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
