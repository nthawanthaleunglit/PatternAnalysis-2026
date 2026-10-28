"""
Evaluate a saved checkpoint on the validation and test sets, at slice level and patient level.

Reports accuracy, AUROC, the confusion matrix (rows = true NC/AD, columns = predicted NC/AD), and test accuracy by
slice position, i.e. the slice's rank (0-19) within its patient's stack, to show which part of the brain is informative.
Also picks the patient-level cut-off that maximises validation accuracy and reports test accuracy at that cut-off.

Usage:
    python evaluate.py --model convnext_pt
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from dataset import DATA_ROOT, load_adni, slice_number
from modules import build_model
from train import predict_logits


def per_patient(probs, labels, items):
    # Mean AD probability and label for each patient, in sorted patient order.
    groups = {}
    for p, y, it in zip(probs.tolist(), labels.tolist(), items):
        groups.setdefault(it[2], [[], y])[0].append(p)
    pids = sorted(groups)
    return (torch.tensor([sum(groups[k][0]) / len(groups[k][0]) for k in pids]),
            torch.tensor([groups[k][1] for k in pids]))


def best_cutoff(probs, labels):
    # The cut-off on the AD probability that gives the highest accuracy (0.5 wins ties, as it is listed first).
    cands = torch.cat([torch.tensor([0.5]), probs.unique()])
    accs = torch.stack([((probs >= c).long() == labels).float().mean() for c in cands])
    return cands[accs.argmax()].item()


def scores(probs, labels, cutoff=0.5):
    # Accuracy at a given cut-off, AUROC, and confusion matrix for one set of AD probabilities.
    pred = (probs >= cutoff).long()
    return {"acc": (pred == labels).float().mean().item(), "auroc": roc_auc_score(labels, probs),
            "confusion": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def position_accuracy(probs, labels, items):
    # Test accuracy for each slice rank within a patient (0 = lowest slice number).
    ranks = {}
    for pid in {it[2] for it in items}:
        nums = sorted(slice_number(it[0]) for it in items if it[2] == pid)
        ranks.update({(pid, n): r for r, n in enumerate(nums)})
    correct = ((probs >= 0.5).long() == labels).tolist()
    by_rank = {}
    for ok, it in zip(correct, items):
        by_rank.setdefault(ranks[(it[2], slice_number(it[0]))], []).append(ok)
    return {r: round(sum(v) / len(v), 4) for r, v in sorted(by_rank.items())}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.out_dir) / f"{args.model}.pt", map_location=device)
    context = ckpt["args"].get("context", False)
    model = build_model(args.model, in_chans=3 if context else 1, pretrained=False).to(device)  # weights from ckpt
    model.load_state_dict(ckpt["state_dict"])
    val_from = ckpt["args"].get("val_from", "train")
    data = load_adni(args.data_root, seed=ckpt["args"]["seed"], val_from=val_from, context=context)

    results = {"model": args.model, "val_from": val_from}
    patient_probs = {}
    for split in ("val", "test"):
        items = getattr(data, split)
        logits, labels = predict_logits(model, data.loader(split, args.batch_size, args.workers), device)
        probs = logits.softmax(1)[:, 1]
        patient_probs[split] = per_patient(probs, labels, items)
        results[split] = {"slice": scores(probs, labels), "patient": scores(*patient_probs[split])}
        if split == "test":
            results["test_acc_by_slice_position"] = position_accuracy(probs, labels, items)

    # Patient cut-off chosen on validation patients only, then applied unchanged to test patients.
    cutoff = best_cutoff(*patient_probs["val"])
    results["val_cutoff"] = round(cutoff, 4)
    results["test"]["patient_at_val_cutoff"] = scores(*patient_probs["test"], cutoff)

    out = Path(args.out_dir) / f"{args.model}_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("cnn", "convnext", "convnext_pt"), default="convnext_pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
