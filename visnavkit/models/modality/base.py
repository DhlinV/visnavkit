"""Modality encoders: any non-image input -> per-frame tokens.

Vision is the one stage the policy knows by name. Everything else a policy observes — ego
state, camera calibration, depth, LiDAR, a BEV raster, an IMU window — is a *modality*: it
declares the batch keys it consumes and returns ``(B, F, G, D)`` tokens that join the vision
tokens of the same frame. Adding one is a new subclass plus a config entry; the policy, the
export contract and the feature buffer follow automatically.
"""

import torch
import torch.nn as nn

__all__ = ["BaseModalityEncoder"]


class BaseModalityEncoder(nn.Module):
    """Shared token contract, learned null token, and per-sample dropout.

    Subclasses set :attr:`input_names` to the batch keys they read, in the order
    :meth:`encode` receives them, and implement :meth:`encode`. Those names become dataset
    keys, ``forward`` keywords, and ONNX input names, so they are the whole interface.

    Args:
        feat_size: token width ``D``.
        num_tokens: ``G`` tokens emitted per frame (0 disables the modality).
        p_drop: training-time probability of replacing a sample's tokens with the null token,
            so the policy also runs when the modality is missing at deployment.
    """

    input_names: tuple[str, ...] = ()

    def __init__(self, feat_size: int, num_tokens: int = 1, p_drop: float = 0.0):
        super().__init__()
        if num_tokens < 0 or feat_size < 1:
            raise ValueError("num_tokens must be nonnegative and feat_size positive")
        if not 0 <= p_drop <= 1:
            raise ValueError("p_drop must be between 0 and 1")
        self.feat_size = feat_size
        self.num_tokens = num_tokens
        self.p_drop = p_drop
        if num_tokens:
            self.null_token = nn.Parameter(torch.zeros(1, 1, num_tokens, feat_size))
            nn.init.normal_(self.null_token, std=0.02)

    def encode(self, *inputs: torch.Tensor, image_hw) -> torch.Tensor:
        """Inputs in :attr:`input_names` order -> ``(B, F, G, D)`` tokens.

        ``image_hw`` is the size of the frames the policy is running on, for modalities whose
        values are tied to the image resolution.
        """
        raise NotImplementedError

    def example_inputs(self, batch_size: int, frames: int = 1, image_hw=(64, 64), device=None) -> tuple:
        """Synthetic batch in :attr:`input_names` order, for shape checks and export tracing."""
        return ()

    def forward(self, *inputs, batch_size: int | None = None, frames: int = 1, image_hw=(1, 1)) -> torch.Tensor:
        if self.num_tokens == 0:
            present = next((value for value in inputs if value is not None), None)
            n = batch_size if present is None else present.shape[0]
            device = None if present is None else present.device
            return torch.zeros(n, frames, 0, self.feat_size, device=device)
        if len(inputs) != len(self.input_names):
            raise ValueError(f"{type(self).__name__} expects {self.input_names}, got {len(inputs)} inputs")
        if any(value is None for value in inputs):
            if batch_size is None:
                raise ValueError("batch_size is required when the modality is absent")
            return self.null_token.repeat(batch_size, frames, 1, 1)  # a copy, not a parameter view
        for name, value in zip(self.input_names, inputs):
            if value.ndim < 3:
                raise ValueError(f"{name} must be (B, F, ...), got {tuple(value.shape)}")
        tokens = self.encode(*inputs, image_hw=image_hw)
        if self.training and self.p_drop > 0:
            keep = torch.rand(tokens.shape[0], 1, 1, 1, device=tokens.device) >= self.p_drop
            tokens = torch.where(keep, tokens, self.null_token.to(tokens.dtype))
        return tokens
