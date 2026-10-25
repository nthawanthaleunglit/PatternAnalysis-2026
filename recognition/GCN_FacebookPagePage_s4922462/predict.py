import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from dataset import CLASS_NAMES, load_facebook
from evaluate import load_model


def neighbour_stats(data, node):
    
    # Degree and share of neighbours (all labels, as the graph is transductive) in each class.
    nbrs = data.edge_index[1][data.edge_index[0] == node]
    counts = torch.bincount(data.y[nbrs], minlength=data.num_classes).float()
    return len(nbrs), (counts / counts.sum().clamp_min(1)).tolist()


def plot_tsne(emb, labels, path, seed=42, max_points=4000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), size=min(max_points, len(emb)), replace=False)
    pts = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30).fit_transform(emb[idx])
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(CLASS_NAMES):
        sel = labels[idx] == c
        ax.scatter(pts[sel, 0], pts[sel, 1], s=4, alpha=0.6, label=name)
    ax.set(title="t-SNE of test-node embeddings (true category colours)", xticks=[], yticks=[])
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("gcn", "mlp"), default="gcn")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--failures", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.out_dir) / f"{args.model}.pt"
    data = load_facebook(args.data_root, seed=torch.load(ckpt, map_location="cpu")["args"]["seed"]).to(device)
    model = load_model(ckpt, data, device)

    with torch.no_grad():
        logits = model(data.x, data.adj)
        emb = model.embed(data.x, data.adj)
    conf, pred = logits.softmax(1).max(1)
    test = data.test_mask.nonzero(as_tuple=True)[0]
    acc = (pred[test] == data.y[test]).float().mean().item()
    print(f"{args.model.upper()} test accuracy: {acc:.4f} on {len(test)} pages\n")

    print("Example predictions:")
    for i in test[: args.examples].tolist():
        mark = "ok " if pred[i] == data.y[i] else "ERR"
        print(f"  [{mark}] {data.page_names[i][:40]:<40} true={CLASS_NAMES[data.y[i]]:<11} "
              f"pred={CLASS_NAMES[pred[i]]:<11} conf={conf[i]:.2f}")

    wrong = test[pred[test] != data.y[test]]
    wrong = wrong[conf[wrong].argsort(descending=True)][: args.failures]
    print(f"\nMost confident errors ({len(test[pred[test] != data.y[test]])} errors in total):")
    for i in wrong.tolist():
        degree, shares = neighbour_stats(data, i)
        mix = ", ".join(f"{n[:4]} {s:.0%}" for n, s in zip(CLASS_NAMES, shares) if s > 0)
        print(f"  {data.page_names[i][:38]:<38} true={CLASS_NAMES[data.y[i]]:<11} pred={CLASS_NAMES[pred[i]]:<11} "
              f"conf={conf[i]:.2f} degree={degree:<4} neighbours: {mix}")

    path = Path(args.out_dir) / f"{args.model}_tsne.png"
    plot_tsne(emb[test].cpu().numpy(), data.y[test].cpu().numpy(), path)
    print(f"\nSaved t-SNE plot to {path}")


if __name__ == "__main__":
    main()
