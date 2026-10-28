"""
Show example predictions and failure cases of the ConvNeXt ensemble on test patients.

Reads the per-patient table written by analyse.py and draws, for each chosen patient, five slices spread over the
20-slice window with the calibrated AD probability, the scan id's era and the window's first slice number:
    examples.png       the two most confident correct AD and NC patients
    failure_cases.png  the two most confident missed AD patients, the two most confident false alarms, and the
                       patient closest to the decision boundary

Usage:
    python predict.py [--table outputs_analysis/test_patients_convnext.csv]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import CLASSES, DATA_ROOT, find_root, list_split, slice_number


def read_table(path):
    with open(path) as f:
        return [{"patient": r["patient"], "label": int(r["label"]), "p": float(r["p_ad_calibrated"]),
                 "first": int(r["first_slice"])} for r in csv.DictReader(f)]


def choose(rows):
    # Confident correct examples, confident errors of each kind, and the most uncertain patient.
    by_p = sorted(rows, key=lambda r: r["p"])
    tp = [r for r in reversed(by_p) if r["label"] == 1 and r["p"] >= 0.5][:2]
    tn = [r for r in by_p if r["label"] == 0 and r["p"] < 0.5][:2]
    fn = [r for r in by_p if r["label"] == 1 and r["p"] < 0.5][:2]            # AD called NC, lowest p(AD) first
    fp = [r for r in reversed(by_p) if r["label"] == 0 and r["p"] >= 0.5][:2]  # NC called AD, highest p(AD) first
    edge = min(rows, key=lambda r: abs(r["p"] - 0.5))
    return {"examples": tp + tn, "failure_cases": fn + fp + [edge]}


def describe(r):
    pred = CLASSES[int(r["p"] >= 0.5)]
    era = "< 400k" if int(r["patient"]) < 400_000 else ">= 400k"
    return (f"{r['patient']}  true {CLASSES[r['label']]} -> {pred}  p(AD) {r['p']:.2f}\n"
            f"id {era}, window starts at slice {r['first']}")


def draw(rows, slices, path, title):
    fig, axes = plt.subplots(len(rows), 5, figsize=(11, 2.4 * len(rows)), squeeze=False)
    for row, r in zip(axes, rows):
        paths = slices[r["patient"]]
        for ax, k in zip(row, (0, 5, 10, 14, 19)):
            p = paths[min(k, len(paths) - 1)]
            ax.imshow(Image.open(p).convert("L"), cmap="gray")
            ax.set_title(f"slice {slice_number(p)}", fontsize=7)
            ax.axis("off")
        row[0].text(-0.1, 0.5, describe(r), transform=row[0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="outputs_analysis/test_patients_convnext.csv")
    p.add_argument("--data-root", default=DATA_ROOT)
    args = p.parse_args()

    slices = {}
    for path, _, pid in list_split(find_root(args.data_root) / "test"):
        slices.setdefault(pid, []).append(path)
    for v in slices.values():
        v.sort(key=slice_number)

    out = Path(args.table).parent
    picks = choose(read_table(args.table))
    draw(picks["examples"], slices, out / "examples.png", "Confident correct test patients (ConvNeXt ensemble)")
    draw(picks["failure_cases"], slices, out / "failure_cases.png",
         "Failure cases: confident misses, confident false alarms, most uncertain patient")
    for name, rows in picks.items():
        print(f"== {name}")
        for r in rows:
            print("  " + describe(r).replace("\n", "  |  "))
    print(f"saved {out}/examples.png and {out}/failure_cases.png")


if __name__ == "__main__":
    main()
