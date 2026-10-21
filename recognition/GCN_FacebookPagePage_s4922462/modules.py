"""
Model components (pure PyTorch, no NumPy).

Both models share the call signature `model(x, adj)` so they can be trained and evaluated by the same code.
`embed(x, adj)` returns the last hidden representation, used for t-SNE plots.
"""
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
