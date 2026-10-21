"""
Train, validate, test and save a model on the Facebook Page-Page graph.

Full-batch training: the whole graph is one forward pass, and the loss is computed on training nodes only.
The checkpoint with the best validation accuracy is kept (early stopping).

Usage:
    python train.py --model gcn
    python train.py --model mlp
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import load_facebook
from modules import build_model


@torch.no_grad()
def accuracy(logits, y, mask):
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_facebook(args.data_root, seed=args.seed).to(device)
    model = build_model(args.model, data.x.size(1), data.num_classes, args.hidden, args.layers, args.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}  parameters: {params}  device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val, best_epoch, best_state = -1.0, 0, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimiser.zero_grad()
        logits = model(data.x, data.adj)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.adj)
            val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
        train_acc, val_acc = accuracy(logits, data.y, data.train_mask), accuracy(logits, data.y, data.val_mask)
        for k, v in zip(history, (loss.item(), val_loss, train_acc, val_acc)):
            history[k].append(v)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  val_loss {val_loss:.4f}  "
                  f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best val_acc {best_val:.4f} at epoch {best_epoch})")
            break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits = model(data.x, data.adj)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) * 1000
    test_acc = accuracy(logits, data.y, data.test_mask)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out / f"{args.model}.pt")
    summary = {
        "model": args.model, "parameters": params, "best_epoch": best_epoch, "val_acc": best_val,
        "test_acc": test_acc, "train_time_s": round(train_time, 2), "full_graph_inference_ms": round(infer_ms, 2),
        "peak_gpu_mb": peak_mb, "device": device.type,
    }
    (out / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    plot_history(history, best_epoch, args.model, out / f"{args.model}_curves.png")
    print(json.dumps(summary, indent=2))


def plot_history(history, best_epoch, name, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(title=f"{name.upper()} cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(title=f"{name.upper()} accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="grey", ls="--", lw=0.8, label="best epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("gcn", "mlp"), default="gcn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
