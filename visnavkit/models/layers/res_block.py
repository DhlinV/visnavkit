import torch.nn as nn


class FusableResBlock(nn.Module):
    """Linear-BN-ReLU-Linear-BN residual block; Linear/BN pairs fold at export time."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.relu(out + x)
