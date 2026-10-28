"""
Train, validate, test and save a model on ADNI AD vs NC slices.

AdamW with linear warm-up then cosine decay, bf16 mixed precision on GPU, and early stopping on validation accuracy.
The checkpoint with the best validation accuracy is kept and evaluated once on the untouched test set.

Usage:
    python train.py --model cnn
    python train.py --model convnext
    python train.py --model convnext_pt --epochs 30 --warmup 2 --lr 1e-4 --drop-path 0.2
    python train.py --model convnext_pt --val-from test --label-smoothing 0.1 --epochs 15 --warmup 1 --lr 5e-5 \
        --drop-path 0.3 --patience 5
    (add --context for 2.5D input: the slices below and above as two extra channels)
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from dataset import DATA_ROOT, load_adni
from modules import build_model


@torch.no_grad()
def predict_logits(model, loader, device, tta=False):
    # Logits and labels for every slice of a loader, in loader order.
    # tta=True averages the logits of each slice and its left-right mirror image.
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x).float()
            if tta:
                out = (out + model(x.flip(3)).float()) / 2
            logits.append(out.cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def patient_accuracy(logits, labels, items):
    # Average the AD probability over all slices of a patient, then classify the patient at 0.5.
    probs = logits.softmax(1)[:, 1]
    pids = [it[2] for it in items]
    per_patient = {}
    for p, y, pid in zip(probs.tolist(), labels.tolist(), pids):
        per_patient.setdefault(pid, [[], y])[0].append(p)
    correct = [(sum(ps) / len(ps) >= 0.5) == bool(y) for ps, y in per_patient.values()]
    return sum(correct) / len(correct)


def random_erase(x, p):
    # In a fraction p of the slices, blank one random rectangle covering 2-33 % of the image. After normalisation
    # 0 is the mean intensity, so the blanked area carries no information and the model cannot rely on any one region.
    n, _, h, w = x.shape
    for i in torch.nonzero(torch.rand(n) < p).flatten().tolist():
        area = h * w * (0.02 + 0.31 * torch.rand(1).item())
        ratio = math.exp(math.log(0.3) + (math.log(3.3) - math.log(0.3)) * torch.rand(1).item())
        eh, ew = min(h, round(math.sqrt(area * ratio))), min(w, round(math.sqrt(area / ratio)))
        top, left = torch.randint(0, h - eh + 1, (1,)).item(), torch.randint(0, w - ew + 1, (1,)).item()
        x[i, :, top:top + eh, left:left + ew] = 0
    return x


def param_groups(model, args):
    # Layer-wise learning-rate decay for the pretrained ConvNeXt: the new head gets the full learning rate and each
    # earlier part of the backbone gets `layer_decay` times the rate of the part above it, so the general-purpose
    # ImageNet features near the input change least. Each group's multiplier is stored under "scale".
    if args.model != "convnext_pt" or args.layer_decay >= 1:
        return [{"params": model.parameters(), "scale": 1.0}]
    feats = model.net.features  # stem, stage 1, down, stage 2, down, stage 3, down, stage 4
    groups = [{"params": feats[i].parameters(), "scale": args.layer_decay ** (len(feats) - i)}
              for i in range(len(feats))]
    return groups + [{"params": model.net.classifier.parameters(), "scale": 1.0}]


def lr_at(epoch, step, steps_per_epoch, args):
    # Linear warm-up for `warmup` epochs, then cosine decay to zero.
    t = epoch + step / steps_per_epoch
    if t < args.warmup:
        return args.lr * t / args.warmup
    return 0.5 * args.lr * (1 + math.cos(math.pi * (t - args.warmup) / (args.epochs - args.warmup)))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_seed = args.seed if args.split_seed is None else args.split_seed
    data = load_adni(args.data_root, seed=split_seed, val_from=args.val_from, context=args.context,
                     img_size=args.img_size, balance=args.balance_slices, brain_scale=args.brain_scale)
    if data.slices:
        print(f"Slices kept: numbers {data.slices[0]}-{data.slices[-1]} ({len(data.slices)} positions)")
    train_loader = data.loader("train", args.batch_size, args.workers)
    val_loader = data.loader("val", args.batch_size, args.workers)
    model = build_model(args.model, in_chans=3 if args.context else 1, drop_path=args.drop_path,
                        arch=args.arch).to(device)
    optimiser = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    # Exponential moving average of the weights: a smoother model that is used for validation and kept as the result.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema)) if args.ema > 0 else None
    eval_model = ema.module if ema else model
    mix = torch.distributions.Beta(args.mixup, args.mixup) if args.mixup > 0 else None
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  parameters: {params}  device: {device}  "
          f"train/val/test slices: {len(data.train)}/{len(data.val)}/{len(data.test)}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val, best_epoch, best_state = -1.0, 0, None
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = seen = 0
        for step, (x, y) in enumerate(train_loader):
            for g in optimiser.param_groups:
                g["lr"] = lr_at(epoch, step, len(train_loader), args) * g["scale"]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.erase > 0:
                x = random_erase(x, args.erase)
            if mix:
                # Mixup: blend each slice with another slice from the batch and blend their targets the same way.
                lam = mix.sample().item()
                perm = torch.randperm(x.shape[0], device=device)
                x = lam * x + (1 - lam) * x[perm]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                if mix:
                    loss = lam * loss + (1 - lam) * F.cross_entropy(logits, y[perm],
                                                                    label_smoothing=args.label_smoothing)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if ema:
                ema.update_parameters(model)
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()

        val_logits, val_y = predict_logits(eval_model, val_loader, device)
        val_loss = F.cross_entropy(val_logits, val_y).item()
        val_acc = (val_logits.argmax(1) == val_y).float().mean().item()
        for k, v in zip(history, (total_loss / seen, val_loss, correct / seen, val_acc)):
            history[k].append(v)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
        print(f"epoch {epoch + 1:3d}  loss {total_loss / seen:.4f}  val_loss {val_loss:.4f}  "
              f"train_acc {correct / seen:.4f}  val_acc {val_acc:.4f}  ({time.time() - start:.0f} s)", flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_acc {best_val:.4f} at epoch {best_epoch})")
            break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    del train_loader, val_loader  # shut down their worker processes before starting the test loader's
    test_loader = data.loader("test", args.batch_size, args.workers)
    t0 = time.time()
    test_logits, test_y = predict_logits(model, test_loader, device)
    infer_ms = (time.time() - t0) * 1000 / len(test_y)  # includes image loading, so an upper bound
    test_acc = (test_logits.argmax(1) == test_y).float().mean().item()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out / f"{args.model}.pt")
    summary = {
        "model": args.model, "parameters": params, "best_epoch": best_epoch, "val_acc": best_val,
        "test_acc": test_acc, "test_patient_acc": patient_accuracy(test_logits, test_y, data.test),
        "train_time_s": round(train_time, 1), "inference_ms_per_slice": round(infer_ms, 3),
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
    ax1.set(title=f"{name} cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(title=f"{name} slice accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="grey", ls="--", lw=0.8, label="best epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("cnn", "convnext", "convnext_pt"), default="convnext")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--val-from", choices=("train", "test"), default="train")
    p.add_argument("--context", action="store_true", help="2.5D input: neighbouring slices as extra channels")
    p.add_argument("--arch", choices=("tiny", "small", "base"), default="tiny", help="size of the convnext_pt backbone")
    p.add_argument("--img-size", type=int, default=224, help="the 224 x 224 crop is resized to this side length")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a, a) parameter; 0 turns it off")
    p.add_argument("--erase", type=float, default=0.0, help="fraction of training slices given a random blank patch")
    p.add_argument("--ema", type=float, default=0.0, help="weight EMA decay, e.g. 0.999; 0 turns it off")
    p.add_argument("--layer-decay", type=float, default=1.0, help="layer-wise LR decay for convnext_pt; 1 = off")
    p.add_argument("--balance-slices", action="store_true",
                   help="use only slice numbers both classes share and sample (slice number, class) evenly")
    p.add_argument("--brain-scale", action="store_true", help="rescale each slice by its own brain intensity")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=624)
    p.add_argument("--split-seed", type=int, default=None, help="seed for the patient split (default: --seed); "
                   "keep it fixed when training several seeds for an ensemble so all members share one split")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
