import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.25, norm=nn.BatchNorm1d):
        # def __init__(self, input_dim, hidden_dim, dropout=0.25, norm=None):
        super(ResBlock, self).__init__()

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)  # Dropout after first activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = norm(hidden_dim) if norm is not None else nn.Identity()

        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)  # Dropout after second activation
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm2 = norm(input_dim) if norm is not None else nn.Identity()

    def forward(self, x):
        out = self.relu1(x)
        out = self.dropout1(out)
        out = self.norm1(self.fc1(out))

        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.norm2(self.fc2(out))

        # Add residual connection
        out = out + x

        return out


# TODO: make it fuseable (something like this)
# def fuse_resblock_linear_bn(module):
#     for child in module.children():
#         if isinstance(child, FusableResBlock):
#             # Assuming FusableResBlock has fc1, bn1, fc2, bn2
#             child.fc1 = utils.fuse_linear_bn_eval(child.fc1, child.bn1)
#             child.bn1 = nn.Identity()
#             child.fc2 = utils.fuse_linear_bn_eval(child.fc2, child.bn2)
#             child.bn2 = nn.Identity()
#         elif isinstance(child, nn.Sequential) or isinstance(child, nn.Module):
#             fuse_resblock_linear_bn(child)
class FusableResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FusableResBlock, self).__init__()
        # Both linear layers maintain the same input/output dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # BN right after fc1
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)  # BN right after fc2
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # Residual connection

        # First branch: Linear -> BN -> ReLU
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second branch: Linear -> BN
        out = self.fc2(out)
        out = self.bn2(out)

        # Residual addition and final activation
        out = out + identity
        out = self.relu(out)
        return out
