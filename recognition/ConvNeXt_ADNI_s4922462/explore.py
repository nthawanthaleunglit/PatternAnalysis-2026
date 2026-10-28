"""
Explore the ADNI folders and look for leakage that a split by file-name id cannot catch.

1. Counts, image sizes, slice numbers and id ranges per split and class.
2. Exact duplicate slices (byte-identical files) anywhere in the data.
3. Near-duplicate scans: each id's 20 slices are shrunk to 30 x 32, stacked into one vector, the average over all
   ids is subtracted (removing the brain shape every scan shares) and the vector is scaled to unit length. The
   cosine similarity between two ids then measures how alike the two scans are beyond that shared shape. If one
   person appears under several ids, those pairs stand out as unusually close nearest neighbours.
4. A nearest-neighbour check: label each id with the label of its most similar *other* id. If train-folder ids
   are matched far better within the train folder than test-folder ids are matched to the train folder, train
   ids have "twins" and any validation split carved out of train/ is optimistic.

Usage:
    python explore.py                 # course copy on Rangpur
    python explore.py <data_root> [out_dir]
"""
import hashlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import CLASSES, DATA_ROOT, find_root, list_split, slice_number

SMALL = (30, 32)  # PIL (width, height) of a shrunken slice


def load_ids(root):
    # One record per (folder, id): its labels and slice paths in slice order.
    ids = {}
    for split in ("train", "test"):
        for p, label, pid in list_split(root / split):
            rec = ids.setdefault((split, pid), {"labels": set(), "paths": []})
            rec["labels"].add(label)
            rec["paths"].append(p)
    for rec in ids.values():
        rec["paths"].sort(key=slice_number)
    return ids


def basic_stats(ids):
    print("== 1. Counts, sizes, slice numbers, id ranges")
    mixed = [k for k, r in ids.items() if len(r["labels"]) > 1]
    both = {pid for s, pid in ids if s == "train"} & {pid for s, pid in ids if s == "test"}
    print(f"ids with slices in both NC and AD: {len(mixed)};  ids in both train/ and test/: {len(both)}")
    for split in ("train", "test"):
        for label, cls in enumerate(CLASSES):
            recs = {pid: r for (s, pid), r in ids.items() if s == split and label in r["labels"]}
            n_slices = Counter(len(r["paths"]) for r in recs.values())
            sizes = Counter(Image.open(r["paths"][0]).size for r in recs.values())
            first = Counter(slice_number(r["paths"][0]) for r in recs.values())
            span = Counter(slice_number(r["paths"][-1]) - slice_number(r["paths"][0]) for r in recs.values())
            nums = sorted(int(pid) for pid in recs if pid.isdigit())
            print(f"{split:5s} {cls}: {len(recs)} ids, slices/id {dict(n_slices)}, image sizes {dict(sizes)}")
            print(f"           first slice number (top 5) {first.most_common(5)}, last - first {dict(span)}")
            if nums:
                print(f"           numeric id min / median / max: {nums[0]} / {nums[len(nums) // 2]} / {nums[-1]}")


def exact_duplicates(ids):
    print("\n== 2. Byte-identical slices")
    by_hash = {}
    for (split, pid), r in ids.items():
        for p in r["paths"]:
            by_hash.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((split, pid, p.name))
    groups = [g for g in by_hash.values() if len(g) > 1]
    cross_id = [g for g in groups if len({(s, pid) for s, pid, _ in g}) > 1]
    cross_split = [g for g in groups if len({s for s, _, _ in g}) > 1]
    print(f"{sum(len(g) for g in by_hash.values())} slices, {len(groups)} duplicate groups, "
          f"{len(cross_id)} spanning different ids, {len(cross_split)} spanning train/ and test/")
    for g in cross_id[:5]:
        print("   e.g.", g)


def volume_vectors(ids, keys):
    # Shrunken, centred, unit-length vector per id (all ids have the same number of slices).
    vecs = np.stack([np.concatenate([np.asarray(Image.open(p).convert("L").resize(SMALL, Image.BILINEAR),
                                                np.float32).ravel() for p in ids[k]["paths"]]) for k in keys])
    vecs -= vecs.mean(0)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs


def near_duplicates(ids, out_dir):
    print("\n== 3. Near-duplicate scans (cosine similarity of centred, shrunken volumes)")
    n = Counter(len(r["paths"]) for r in ids.values()).most_common(1)[0][0]
    keys = sorted(k for k, r in ids.items() if len(r["paths"]) == n)
    print(f"using the {len(keys)} ids with {n} slices ({len(ids) - len(keys)} skipped)")
    split = np.array([s for s, _ in keys])
    label = np.array([min(ids[k]["labels"]) for k in keys])
    vecs = volume_vectors(ids, keys)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -np.inf)
    tr, te = split == "train", split == "test"

    groups = {"train -> nearest train": (tr, tr), "test -> nearest train": (te, tr), "test -> nearest test": (te, te)}
    nn_sims = {}
    print(f"{'query -> neighbour':26s}{'NN sim p10 / p50 / p90':>26s}{'NN label agreement':>22s}")
    for name, (q, ref) in groups.items():
        sub = sim[np.ix_(q, ref)]
        nn = sub.argmax(1)
        nn_sim = sub[np.arange(len(nn)), nn]
        agree = (label[q] == label[ref][nn]).mean()
        nn_sims[name] = nn_sim
        p10, p50, p90 = np.percentile(nn_sim, [10, 50, 90])
        print(f"{name:26s}{p10:>12.3f} / {p50:.3f} / {p90:.3f}{agree:>16.3f}")

    print("pairs above a similarity threshold (each pair counted once):")
    iu = np.triu_indices(len(keys), 1)
    pair_sim = sim[iu]
    a, b = iu
    kinds = {"train-train": tr[a] & tr[b], "train-test": (tr[a] & te[b]) | (te[a] & tr[b]), "test-test": te[a] & te[b]}
    for t in (0.9, 0.8, 0.7, 0.6, 0.5):
        hit = pair_sim > t
        parts = ", ".join(f"{k} {int((hit & m).sum())} (same label {int((hit & m & (label[a] == label[b])).sum())})"
                          for k, m in kinds.items())
        print(f"   > {t}: {parts}")

    top = np.argsort(-pair_sim)[:10]
    print("10 most similar pairs:")
    for i in top:
        ka, kb = keys[a[i]], keys[b[i]]
        print(f"   {pair_sim[i]:.3f}  {ka[0]}/{CLASSES[label[a[i]]]}/{ka[1]}  vs  {kb[0]}/{CLASSES[label[b[i]]]}/{kb[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, v in nn_sims.items():
        ax.hist(v, bins=60, alpha=0.5, label=name)
    ax.set(xlabel="cosine similarity to nearest other id", ylabel="ids", title="Nearest-neighbour similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nn_similarity.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for col, i in enumerate(top[:5]):
        for row, k in enumerate((keys[a[i]], keys[b[i]])):
            mid = ids[k]["paths"][len(ids[k]["paths"]) // 2]
            axes[row, col].imshow(Image.open(mid).convert("L"), cmap="gray")
            axes[row, col].set_title(f"{k[0]}/{CLASSES[min(ids[k]['labels'])]}/{k[1]}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Most similar pairs of different ids (middle slice of each)")
    fig.tight_layout()
    fig.savefig(out_dir / "closest_pairs.png", dpi=120)
    plt.close(fig)
    print(f"figures saved to {out_dir}/nn_similarity.png and {out_dir}/closest_pairs.png")


if __name__ == "__main__":
    root = find_root(sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT)
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs_explore")
    ids = load_ids(root)
    basic_stats(ids)
    exact_duplicates(ids)
    near_duplicates(ids, out)
