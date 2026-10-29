import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from dataset import CLASSES, DATA_ROOT, slice_number
from evaluate import run_probs


def system_probs(model, runs, args, device):
    # Patient AD probability, label and id on val and test, averaging slice probabilities over the runs.
    members = [run_probs(r, Namespace(model=model, tta=False, batch_size=args.batch_size, workers=args.workers,
                                      data_root=args.data_root), device) for r in runs]
    data = members[0][1]
    out = {}
    for s in ("val", "test"):
        items = getattr(data, s)
        probs = torch.stack([m[2][s][0] for m in members]).mean(0).tolist()
        groups = {}
        for p, (path, label, pid) in zip(probs, items):
            g = groups.setdefault(pid, {"p": [], "y": label, "first": slice_number(path)})
            g["p"].append(p)
            g["first"] = min(g["first"], slice_number(path))
        pids = sorted(groups)
        out[s] = {"pid": pids, "p": np.array([np.mean(groups[k]["p"]) for k in pids]),
                  "y": np.array([groups[k]["y"] for k in pids]), "first": [groups[k]["first"] for k in pids]}
    return out


def summary_cost(model, runs):
    # Parameters, inference time and peak GPU memory, summed (time, parameters) or maxed (memory) over the runs.
    s = [json.loads((Path(r) / f"{model}_summary.json").read_text()) for r in runs]
    return {"models": len(s), "parameters": sum(x["parameters"] for x in s),
            "inference_ms_per_slice": round(sum(x["inference_ms_per_slice"] for x in s), 3),
            "peak_gpu_mb": max(x["peak_gpu_mb"] or 0 for x in s), "train_time_min": round(sum(x["train_time_s"] for x in s) / 60, 1)}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    # a, b minimising validation log-loss of sigmoid(a * logit(p) + b); C is large, so there is (almost) no penalty.
    lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
    return lr.coef_[0, 0], lr.intercept_[0]


def apply_platt(p, a, b):
    return 1 / (1 + np.exp(-(a * logit(p) + b)))


def ece(p, y, n_bins=10):
    # Expected calibration error on the confidence of the predicted class, plus the points of a reliability diagram.
    conf, correct = np.maximum(p, 1 - p), ((p >= 0.5) == y)
    edges, total, bins = np.linspace(0.5, 1, n_bins + 1), 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
            bins.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return total, bins


def risk_coverage(p, y):
    # Patients sorted by confidence: confidence, share automated, and accuracy on the automated share.
    conf, correct = np.maximum(p, 1 - p), ((p >= 0.5) == y)
    order = np.argsort(-conf, kind="stable")
    n = np.arange(1, len(p) + 1)
    return conf[order], n / len(p), np.cumsum(correct[order]) / n


def choose_threshold(p, y, target):
    # Lowest confidence threshold whose automated validation patients still reach `target` accuracy.
    conf, _, acc = risk_coverage(p, y)
    ok = np.nonzero(acc >= target)[0]
    return float(conf[ok.max()]) if len(ok) else 1.0


def reject_option(p, y, threshold):
    conf, pred = np.maximum(p, 1 - p), (p >= 0.5).astype(int)
    auto = conf >= threshold
    missed_ad = int(((pred == 0) & (y == 1) & auto).sum())
    return {"threshold": round(threshold, 4), "coverage": auto.mean(), "automated_accuracy": (pred == y)[auto].mean(),
            "referred_share": 1 - auto.mean(), "referred_accuracy": (pred == y)[~auto].mean() if (~auto).any() else None,
            "automated_missed_ad": missed_ad, "automated_false_alarms": int(((pred == 1) & (y == 0) & auto).sum())}


def classification(p, y):
    pred = (p >= 0.5).astype(int)
    r = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": r["accuracy"], "macro_f1": r["macro avg"]["f1-score"], "auroc": roc_auc_score(y, p),
            "per_class": {c: {k: round(r[c][k], 3) for k in ("precision", "recall", "f1-score", "support")}
                          for c in CLASSES},
            "confident_errors": int((((p >= 0.5) != y) & (np.maximum(p, 1 - p) >= 0.9)).sum())}


def analyse(name, sp, target):
    a, b = fit_platt(sp["val"]["p"], sp["val"]["y"])
    raw, cal = sp["test"]["p"], apply_platt(sp["test"]["p"], a, b)
    y = sp["test"]["y"]
    ece_raw, bins_raw = ece(raw, y)
    ece_cal, bins_cal = ece(cal, y)
    threshold = choose_threshold(apply_platt(sp["val"]["p"], a, b), sp["val"]["y"], target)
    result = {"platt_a": a, "platt_b": b, "equivalent_cutoff_on_raw_p": float(1 / (1 + np.exp(b / a))),
              "test_raw": classification(raw, y), "test_calibrated": classification(cal, y),
              "ece_raw": ece_raw, "ece_calibrated": ece_cal, "reject_option_test": reject_option(cal, y, threshold)}
    curves = {"bins_raw": bins_raw, "bins_cal": bins_cal, "risk_coverage": risk_coverage(cal, y)}
    return result, curves, cal


def plot_reliability(curves, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 4), squeeze=False)
    for ax, (name, c) in zip(axes[0], curves.items()):
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.8, label="perfect calibration")
        for key, label in (("bins_raw", "raw"), ("bins_cal", "after Platt scaling")):
            conf, acc, _ = zip(*c[key])
            ax.plot(conf, acc, "o-", ms=4, label=label)
        ax.set(title=f"{name} reliability (test patients)", xlabel="confidence", ylabel="accuracy",
               xlim=(0.5, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_risk_coverage(curves, target, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, c in curves.items():
        _, coverage, acc = c["risk_coverage"]
        ax.plot(coverage, acc, label=name)
    ax.axhline(target, color="grey", ls="--", lw=0.8, label=f"target accuracy {target}")
    ax.set(title="Accuracy of automated patients vs coverage", xlabel="coverage (share automated)",
           ylabel="accuracy on automated patients", ylim=(0.5, 1.01))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_shortcut(tables, path):
    # Test accuracy of AD and NC patients by id era: the cue that separates the classes in train/.
    groups = [("NC", ">= 400k"), ("AD", "< 400k"), ("AD", ">= 400k")]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(tables)
    for i, (name, t) in enumerate(tables.items()):
        vals = [t[c][e]["acc"] if e in t[c] else 0 for c, e in groups]
        ax.bar(np.arange(len(groups)) + i * width, vals, width, label=name)
    ax.set_xticks(np.arange(len(groups)) + width * (len(tables) - 1) / 2,
                  [f"{c}, id {e}\n(n={next(iter(tables.values()))[c].get(e, {}).get('n', 0)})" for c, e in groups])
    ax.set(ylabel="test accuracy (calibrated)", ylim=(0, 1.05), title="Test accuracy by scan-id era")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def era_table(sp, cal):
    # Accuracy and count per class and id era for test patients.
    t = sp["test"]
    out = {}
    for c, label in ((CLASSES[0], 0), (CLASSES[1], 1)):
        out[c] = {}
        for era, test in (("< 400k", lambda pid: int(pid) < 400_000), (">= 400k", lambda pid: int(pid) >= 400_000)):
            m = np.array([y == label and test(pid) for pid, y in zip(t["pid"], t["y"])])
            if m.any():
                out[c][era] = {"n": int(m.sum()), "acc": round(float(((cal[m] >= 0.5) == label).mean()), 3)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn", nargs="+", required=True, help="run folder(s) of the baseline CNN")
    p.add_argument("--convnext", nargs="+", required=True, help="run folder(s) of the ConvNeXt ensemble")
    p.add_argument("--target-acc", type=float, default=0.90)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="outputs_analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    systems = {"CNN": ("cnn", args.cnn), "ConvNeXt": ("convnext_pt", args.convnext)}
    results, curves, eras = {}, {}, {}
    pids = None
    for name, (model, runs) in systems.items():
        sp = system_probs(model, runs, args, device)
        assert pids is None or sp["test"]["pid"] == pids, "systems were evaluated on different test patients"
        pids = sp["test"]["pid"]
        results[name], curves[name], cal = analyse(name, sp, args.target_acc)
        results[name]["cost"] = summary_cost(model, runs)
        eras[name] = results[name]["test_by_id_era"] = era_table(sp, cal)
        with open(out / f"test_patients_{name.lower()}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient", "label", "p_ad_raw", "p_ad_calibrated", "first_slice"])
            t = sp["test"]
            for row in zip(t["pid"], t["y"], t["p"].round(4), cal.round(4), t["first"]):
                w.writerow(row)

    (out / "analysis.json").write_text(json.dumps(results, indent=2, default=float))
    plot_reliability(curves, out / "reliability.png")
    plot_risk_coverage(curves, args.target_acc, out / "risk_coverage.png")
    plot_shortcut(eras, out / "shortcut_by_era.png")

    print(f"{len(pids)} test patients; calibration and thresholds fitted on validation patients only")
    print(f"{'':10}{'acc raw':>8}{'acc cal':>8}{'macroF1':>8}{'AUROC':>7}{'ECE raw':>8}{'ECE cal':>8}"
          f"{'conf.err':>9}{'cover':>7}{'auto acc':>9}{'missed AD':>10}{'params':>12}{'ms/slice':>9}")
    for name, r in results.items():
        ro, tc = r["reject_option_test"], r["test_calibrated"]
        print(f"{name:10}{r['test_raw']['accuracy']:>8.3f}{tc['accuracy']:>8.3f}{tc['macro_f1']:>8.3f}"
              f"{tc['auroc']:>7.3f}{r['ece_raw']:>8.3f}{r['ece_calibrated']:>8.3f}{r['test_raw']['confident_errors']:>9d}"
              f"{ro['coverage']:>7.3f}{ro['automated_accuracy']:>9.3f}{ro['automated_missed_ad']:>10d}"
              f"{r['cost']['parameters']:>12,d}{r['cost']['inference_ms_per_slice']:>9.2f}")
        print(f"{'':10}per class (calibrated): " + ", ".join(
            f"{c} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1-score']:.2f}" for c, v in tc["per_class"].items())
            + f" | by id era: {r['test_by_id_era']}")
    print(f"saved {out}/analysis.json, reliability.png, risk_coverage.png, shortcut_by_era.png, test_patients_*.csv")


if __name__ == "__main__":
    main()
