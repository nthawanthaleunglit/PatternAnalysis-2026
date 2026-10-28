"""
Evaluate a saved checkpoint on the validation and test sets, at slice level and patient level.

Reports accuracy, AUROC, the confusion matrix (rows = true NC/AD, columns = predicted NC/AD), and test accuracy by
slice position, i.e. the slice's rank (0-19) within its patient's stack, to show which part of the brain is informative.
Also picks the patient-level cut-off that maximises validation accuracy and reports test accuracy at that cut-off.

Usage:
    python evaluate.py --model convnext_pt --runs outputs_ft2
    python evaluate.py --model convnext_pt --tta --runs outputs_ft2 outputs_ft3 outputs_ens/s1   # ensemble
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


def shortcut_breakdown(probs, labels, items, cutoff):
    # Patient accuracy and mean AD probability per class, split by two cues that separate the classes in train/:
    # the scan id's era (all train AD ids are below 400,000, all train NC ids above 800,000) and the slice number
    # the patient's 20-slice window starts at. probs/labels are per patient, in sorted patient order.
    first = {}
    for p, _, pid in items:
        first[pid] = min(first.get(pid, slice_number(p)), slice_number(p))
    pids = sorted(first)
    out = {}
    for label, cls in enumerate(("NC", "AD")):
        cells = {}
        for i, pid in enumerate(pids):
            if labels[i] != label:
                continue
            for key in (f"id {'< 400k' if int(pid) < 400_000 else '>= 400k'}", f"first slice {first[pid]}"):
                cells.setdefault(key, []).append((probs[i].item(), (probs[i] >= cutoff).item() == bool(label)))
        out[cls] = {k: {"patients": len(v), "acc": round(sum(c for _, c in v) / len(v), 3),
                        "mean_p_ad": round(sum(p for p, _ in v) / len(v), 3)} for k, v in sorted(cells.items())}
    return out


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


def run_probs(run_dir, args, device):
    # Slice-level AD probabilities on val and test for the checkpoint in one run folder.
    ckpt = torch.load(Path(run_dir) / f"{args.model}.pt", map_location=device)
    a = ckpt["args"]
    context = a.get("context", False)
    model = build_model(args.model, in_chans=3 if context else 1, pretrained=False,  # weights come from the ckpt
                        arch=a.get("arch", "tiny")).to(device)
    model.load_state_dict(ckpt["state_dict"])
    split = (a.get("val_from", "train"), a.get("split_seed") or a["seed"])
    data = load_adni(args.data_root, seed=split[1], val_from=split[0], context=context,
                     img_size=a.get("img_size", 224), balance=a.get("balance_slices", False),
                     brain_scale=a.get("brain_scale", False))
    probs = {}
    for s in ("val", "test"):
        logits, labels = predict_logits(model, data.loader(s, args.batch_size, args.workers), device, tta=args.tta)
        probs[s] = (logits.softmax(1)[:, 1], labels)
    return split, data, probs


def evaluate(args):
    # With several run folders, the members' slice probabilities are averaged (an ensemble). All members must
    # share the same patient split, so the averaged slices line up.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [run_probs(r, args, device) for r in args.runs]
    split, data, _ = members[0]
    assert all(m[0] == split for m in members), "ensemble members use different patient splits"
    assert all(m[1].test == data.test and m[1].val == data.val for m in members), \
        "ensemble members use different slices (mix only runs with the same --balance-slices setting)"

    results = {"model": args.model, "runs": args.runs, "tta": args.tta, "val_from": split[0]}
    patient_probs = {}
    for s in ("val", "test"):
        items = getattr(data, s)
        labels = members[0][2][s][1]
        probs = torch.stack([m[2][s][0] for m in members]).mean(0)
        patient_probs[s] = per_patient(probs, labels, items)
        results[s] = {"slice": scores(probs, labels), "patient": scores(*patient_probs[s])}
        if s == "test":
            results["test_acc_by_slice_position"] = position_accuracy(probs, labels, items)

    # Patient cut-off chosen on validation patients only, then applied unchanged to test patients.
    cutoff = best_cutoff(*patient_probs["val"])
    results["val_cutoff"] = round(cutoff, 4)
    results["test"]["patient_at_val_cutoff"] = scores(*patient_probs["test"], cutoff)
    results["test_shortcut_breakdown"] = shortcut_breakdown(*patient_probs["test"], data.test, cutoff)

    tag = "_tta" if args.tta else ""
    out = (Path(args.runs[0]) / f"{args.model}_eval{tag}.json" if len(args.runs) == 1
           else Path(f"ensemble_eval{tag}.json"))
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("cnn", "convnext", "convnext_pt"), default="convnext_pt")
    p.add_argument("--runs", nargs="+", default=["outputs"], help="one run folder, or several to ensemble")
    p.add_argument("--tta", action="store_true", help="also average each slice with its left-right mirror")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-root", default=DATA_ROOT)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
