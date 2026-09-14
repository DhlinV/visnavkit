import torch.nn as nn


def build_mlp(in_dim: int, hidden: int, out_dim: int, *, layers: int = 2, dropout: float = 0.0, activation=nn.ReLU):
    """``layers`` hidden layers of ``hidden`` units with ``activation``; no activation on the output."""
    if layers < 1:
        raise ValueError("layers must be positive")
    modules: list[nn.Module] = []
    width = in_dim
    for _ in range(layers):
        modules += [nn.Linear(width, hidden), activation(), nn.Dropout(dropout)]
        width = hidden
    modules.append(nn.Linear(width, out_dim))
    return nn.Sequential(*modules)
