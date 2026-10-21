"""
Evaluation for the confidence and ambiguity dilemma.

All functions take logits and labels, so the same code can evaluate any classifier:
  * per-class precision / recall / F1, macro-F1 and one-vs-rest AUROC
  * calibration: expected calibration error (ECE) and reliability diagrams
  * temperature scaling fitted on validation logits (Guo et al., 2017)
  * reject option: automate predictions above a confidence threshold, refer the rest to a human

Usage (after training both models):
    python evaluate.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

from dataset import CLASS_NAMES, load_facebook
from modules import build_model


def load_model(path, data, device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = build_model(a["model"], data.x.size(1), data.num_classes, a["hidden"], a["layers"], a["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def classification_metrics(logits, y):
    probs = logits.softmax(1)
    report = classification_report(y.cpu(), probs.argmax(1).cpu(), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    report["auroc_ovr_macro"] = roc_auc_score(y.cpu(), probs.cpu(), multi_class="ovr", average="macro")
    return report


def expected_calibration_error(logits, y, n_bins=15):
    """ECE = sum over confidence bins of (bin share) * |bin accuracy - bin mean confidence|."""
    conf, pred = logits.softmax(1).max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece, bins = torch.zeros((), device=logits.device), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            acc, avg_conf = correct[in_bin].mean(), conf[in_bin].mean()
            ece += in_bin.float().mean() * (acc - avg_conf).abs()
            bins.append((avg_conf.item(), acc.item(), int(in_bin.sum())))
    return ece.item(), bins


def fit_temperature(val_logits, val_y):
    """Find T > 0 minimising validation negative log-likelihood of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True, device=val_logits.device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / log_t.exp(), val_y)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def risk_coverage(logits, y):
    """Sort predictions by confidence; return thresholds, coverage and accuracy on the automated part."""
    conf, pred = logits.softmax(1).max(1)
    order = conf.argsort(descending=True)
    correct = (pred[order] == y[order]).float()
    n = torch.arange(1, len(y) + 1, device=logits.device)
    return conf[order], n / len(y), correct.cumsum(0) / n


def choose_threshold(val_logits, val_y, target_acc=0.95):
    """Lowest confidence threshold whose automated validation predictions still reach `target_acc`."""
    thresholds, _, sel_acc = risk_coverage(val_logits, val_y)
    ok = (sel_acc >= target_acc).nonzero()
    return thresholds[ok.max()].item() if len(ok) else 1.0


def apply_threshold(logits, y, threshold):
    conf, pred = logits.softmax(1).max(1)
    auto = conf >= threshold
    return {
        "threshold": threshold,
        "coverage": auto.float().mean().item(),
        "automated_accuracy": (pred[auto] == y[auto]).float().mean().item() if auto.any() else float("nan"),
        "referred_accuracy": (pred[~auto] == y[~auto]).float().mean().item() if (~auto).any() else float("nan"),
        "referred_share": (~auto).float().mean().item(),
    }


def evaluate_model(name, model, data, target_acc):
    with torch.no_grad():
        logits = model(data.x, data.adj)
    val_logits, val_y = logits[data.val_mask], data.y[data.val_mask]
    test_logits, test_y = logits[data.test_mask], data.y[data.test_mask]

    t = fit_temperature(val_logits, val_y)
    ece_raw, bins_raw = expected_calibration_error(test_logits, test_y)
    ece_cal, bins_cal = expected_calibration_error(test_logits / t, test_y)
    threshold = choose_threshold(val_logits / t, val_y, target_acc)
    report = classification_metrics(test_logits, test_y)
    result = {
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "auroc_ovr_macro": report["auroc_ovr_macro"],
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASS_NAMES},
        "temperature": t,
        "ece_before": ece_raw,
        "ece_after": ece_cal,
        "reject_option_test": apply_threshold(test_logits / t, test_y, threshold),
    }
    curves = {"bins_raw": bins_raw, "bins_cal": bins_cal,
              "risk_coverage": [v.cpu() for v in risk_coverage(test_logits / t, test_y)]}
    return result, curves


def plot_reliability(curves, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 4), squeeze=False)
    for ax, (name, c) in zip(axes[0], curves.items()):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        for key, label in (("bins_raw", "before scaling"), ("bins_cal", "after temperature scaling")):
            conf, acc, _ = zip(*c[key])
            ax.plot(conf, acc, "o-", ms=4, label=label)
        ax.set(title=f"{name.upper()} reliability (test)", xlabel="confidence", ylabel="accuracy",
               xlim=(0.2, 1), ylim=(0.2, 1))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_risk_coverage(curves, target_acc, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, c in curves.items():
        _, coverage, sel_acc = c["risk_coverage"]
        ax.plot(coverage, sel_acc, label=name.upper())
    ax.axhline(target_acc, color="grey", ls="--", lw=0.8, label=f"target accuracy {target_acc}")
    ax.set(title="Accuracy of automated predictions vs coverage (test)", xlabel="coverage (share automated)",
           ylabel="accuracy on automated nodes", ylim=(0.6, 1.005))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["mlp", "gcn"])
    p.add_argument("--target-acc", type=float, default=0.95)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    results, curves = {}, {}
    for name in args.models:
        ckpt = out / f"{name}.pt"
        data = load_facebook(args.data_root, seed=torch.load(ckpt, map_location="cpu")["args"]["seed"]).to(device)
        results[name], curves[name] = evaluate_model(name, load_model(ckpt, data, device), data, args.target_acc)

    (out / "evaluation.json").write_text(json.dumps(results, indent=2))
    plot_reliability(curves, out / "reliability.png")
    plot_risk_coverage(curves, args.target_acc, out / "risk_coverage.png")

    print(f"{'':<5}{'acc':>7}{'macroF1':>9}{'AUROC':>8}{'T':>7}{'ECE before':>12}{'ECE after':>11}"
          f"{'coverage':>10}{'auto acc':>10}")
    for name, r in results.items():
        ro = r["reject_option_test"]
        print(f"{name.upper():<5}{r['test_accuracy']:>7.3f}{r['macro_f1']:>9.3f}{r['auroc_ovr_macro']:>8.3f}"
              f"{r['temperature']:>7.2f}{r['ece_before']:>12.3f}{r['ece_after']:>11.3f}"
              f"{ro['coverage']:>10.3f}{ro['automated_accuracy']:>10.3f}")


if __name__ == "__main__":
    main()
