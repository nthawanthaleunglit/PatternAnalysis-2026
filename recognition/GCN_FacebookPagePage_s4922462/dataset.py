"""
Data loading for the Facebook Large Page-Page Network.

The graph comes from the course-linked `facebook.npz` (gayanku/facebook_page-page_dataset), the same file
PyTorch Geometric's FacebookPagePage loader uses:
    X        (22470, 128) float32  node features (SVD of page-description tokens)
    edges_x  (342004,)    int64    source node of each directed edge
    edges_y  (342004,)    int64    target node (every undirected like appears in both directions)
    y        (22470,)     int      0 tvshow, 1 government, 2 company, 3 politician
    page_name(22470,)     str      human-readable page names (used for failure analysis only)

Setting: transductive semi-supervised node classification. The whole graph (all features and edges) is
visible during message passing, but only training-node labels are used in the loss. Validation and test
labels are never seen during training.
"""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_URL = "https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/main/facebook.npz"
CLASS_NAMES = ("tvshow", "government", "company", "politician")


@dataclass
class GraphData:
    x: torch.Tensor           # (N, F) standardised node features
    y: torch.Tensor           # (N,) class labels
    adj: torch.Tensor         # (N, N) sparse GCN-normalised adjacency with self loops
    edge_index: torch.Tensor  # (2, E) directed edge list, no self loops
    train_mask: torch.Tensor  # (N,) bool
    val_mask: torch.Tensor    # (N,) bool
    test_mask: torch.Tensor   # (N,) bool
    page_names: list

    @property
    def num_classes(self):
        return int(self.y.max()) + 1

    def to(self, device):
        for name in ("x", "y", "adj", "edge_index", "train_mask", "val_mask", "test_mask"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def download(root="data"):
    """Download facebook.npz into `root` if it is not already there; return its path."""
    path = Path(root) / "facebook.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def stratified_split(y, train_frac=0.7, val_frac=0.15, seed=42):
    """Split node indices into train/val/test masks, keeping class proportions in every split."""
    gen = torch.Generator().manual_seed(seed)
    n = y.numel()
    masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        n_train = round(train_frac * idx.numel())
        n_val = round(val_frac * idx.numel())
        masks[0][idx[:n_train]] = True
        masks[1][idx[n_train:n_train + n_val]] = True
        masks[2][idx[n_train + n_val:]] = True
    return masks


def normalised_adjacency(edge_index, n):
    """Build the GCN propagation matrix D^-1/2 (A + I) D^-1/2 as a sparse tensor (Kipf & Welling, 2017)."""
    loops = torch.arange(n).repeat(2, 1)
    ei = torch.cat([edge_index, loops], dim=1)
    deg = torch.zeros(n).scatter_add_(0, ei[0], torch.ones(ei.size(1)))
    inv_sqrt = deg.pow(-0.5)
    values = inv_sqrt[ei[0]] * inv_sqrt[ei[1]]
    return torch.sparse_coo_tensor(ei, values, (n, n)).coalesce()


def load_facebook(root="data", train_frac=0.7, val_frac=0.15, seed=42):
    """Load the graph, split the nodes and standardise features with training-node statistics only."""
    raw = np.load(download(root), allow_pickle=True)
    x = torch.from_numpy(raw["X"]).float()
    y = torch.from_numpy(raw["y"].astype(np.int64))
    edge_index = torch.from_numpy(np.stack([raw["edges_x"], raw["edges_y"]])).long()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # drop any self loops; added back once below

    train_mask, val_mask, test_mask = stratified_split(y, train_frac, val_frac, seed)
    assert not (train_mask & val_mask).any() and not (train_mask & test_mask).any() and not (val_mask & test_mask).any()

    mean, std = x[train_mask].mean(0), x[train_mask].std(0).clamp_min(1e-8)
    x = (x - mean) / std

    return GraphData(
        x=x, y=y, adj=normalised_adjacency(edge_index, x.size(0)), edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        page_names=[str(s) for s in raw["page_name"]],
    )


def summarise(data):
    """Print dataset statistics: sizes, class balance per split and edge homophily."""
    n, e = data.x.size(0), data.edge_index.size(1)
    print(f"Nodes: {n}  Directed edges: {e} ({e // 2} undirected)  Features: {data.x.size(1)}")
    print(f"{'Split':<7}{'Nodes':>7}  " + "  ".join(f"{c:>10}" for c in CLASS_NAMES))
    for name, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        counts = torch.bincount(data.y[mask], minlength=data.num_classes).tolist()
        print(f"{name:<7}{int(mask.sum()):>7}  " + "  ".join(f"{c:>10}" for c in counts))
    src, dst = data.edge_index
    print(f"Edge homophily (share of edges joining same-class pages): {(data.y[src] == data.y[dst]).float().mean():.3f}")


if __name__ == "__main__":
    summarise(load_facebook())
