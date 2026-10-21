"""
Model components (pure PyTorch, no NumPy).

Both models share the call signature `model(x, adj)` so they can be trained and evaluated by the same code.
`embed(x, adj)` returns the last hidden representation, used for t-SNE plots.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Baseline: classifies each page from its own 128-d features only and ignores the graph."""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, x, adj=None):
        for layer in self.hidden:
            x = layer(self.dropout(x)).relu()
        return x

    def forward(self, x, adj=None):
        return self.out(self.dropout(self.embed(x, adj)))


class GCNLayer(nn.Module):
    """One graph convolution (Kipf & Welling, 2017): H' = A_hat (H W), where A_hat = D^-1/2 (A + I) D^-1/2.

    Each node's new representation is a degree-weighted average of its own and its neighbours'
    transformed features, so information flows one hop along the graph per layer.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.linear(x)) + self.bias


class GCN(nn.Module):
    """Multi-layer GCN: (dropout -> GCNLayer -> ReLU) x (num_layers - 1), then a final GCNLayer to class logits."""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.hidden = nn.ModuleList(GCNLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = GCNLayer(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, x, adj):
        for layer in self.hidden:
            x = layer(self.dropout(x), adj).relu()
        return x

    def forward(self, x, adj):
        return self.out(self.dropout(self.embed(x, adj)), adj)


def build_model(name, in_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.5):
    """Create a model by name ('mlp' or 'gcn')."""
    models = {"mlp": MLP, "gcn": GCN}
    return models[name](in_dim, hidden_dim, num_classes, num_layers, dropout)
