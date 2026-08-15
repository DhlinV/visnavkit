import torch
import torch.nn as nn

# def generate_causal_mask(seq_len: int, device=None) -> torch.Tensor:
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Create an upper-triangular matrix filled with -inf (masking future tokens)
#     mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
#     return mask


# TODO: do per sample instead of across the batch
def generate_causal_mask(seq_len: int, mask_p: float = 0.0, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start with standard causal mask (mask future tokens)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)

    if mask_p > 0.0:
        rand_mask = torch.rand(seq_len, seq_len, device=device) < mask_p
        lower_triangle = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
        random_mask = rand_mask * lower_triangle
        causal_mask = causal_mask.masked_fill(random_mask.bool(), float("-inf"))

    return causal_mask


# TODO: random mask
class Summarizer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        seq_len=10,
        reduction="last",
        mask_p=0.5,
        route_embed_dim=0,
        num_layers=1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.reduction = reduction
        self.mask_p = mask_p

        model_dim = embed_dim + route_embed_dim

        # num_layers=0 -> no temporal mixing at all (per-frame passthrough + reduction)
        if num_layers == 0:
            self.tformer = None
            return

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(seq_len, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",  # Activation function
            batch_first=True,  # Input shape: (batch_size, seq_len, embed_dim)
            norm_first=True,  # Layer normalization before other operations
        )
        # single layer stays unwrapped to keep existing checkpoint state_dict keys
        self.tformer = layer if num_layers == 1 else nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, x):
        b, s, e = x.shape

        if self.tformer is not None:
            # pos embed
            positions = self.pos_embedding(torch.arange(s, device=x.device))[None, :, :].expand(b, s, e)
            x = x + positions

            # tformer (positional mask arg: src_mask for a single layer, mask for a stack)
            mask = generate_causal_mask(self.seq_len, mask_p=self.mask_p if self.training else 0, device=x.device)
            x = self.tformer(x, mask if self.reduction == "none" else None)

        # reduction over time dimension
        if self.reduction == "last":
            return x[:, -1, :]
        elif self.reduction == "avg":
            return x.mean(dim=1)
        elif self.reduction == "sum":
            return x.sum(dim=1)
        elif self.reduction == "none":
            return x
        else:
            raise ValueError(f"Unknown reduction value {self.reduction=}")
