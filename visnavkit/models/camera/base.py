"""Camera encoders turn the calibration of each frame into tokens."""

import torch
import torch.nn as nn

__all__ = ["BaseCameraEncoder"]


class BaseCameraEncoder(nn.Module):
    """Shared camera-token contract, learned null token, and per-sample dropout.

    Calibration is passed as two tensors, both per frame so a moving or switching camera is
    expressible and a static rig simply repeats:

    - ``intrinsics`` ``(B, F, 3, 3)`` pinhole ``K`` for the frames as the policy sees them, i.e.
      already adjusted for the dataset's crop and downscale.
    - ``extrinsics`` ``(B, F, 4, 4)`` camera-to-ego rigid transform.

    Tokens are per frame, so they join the vision and ego tokens of the same frame.

    Args:
        feat_size: token width ``D``.
        num_tokens: ``G`` tokens emitted per frame (0 for policies that ignore calibration).
        p_drop: training-time probability of replacing a sample's camera tokens with the null
            token, so the policy also runs on clips with unknown calibration.
    """

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

    def encode(self, intrinsics: torch.Tensor, extrinsics: torch.Tensor, image_hw) -> torch.Tensor:
        """``(B, F, 3, 3)`` and ``(B, F, 4, 4)`` -> ``(B, F, G, D)`` tokens.

        ``image_hw`` is the size of the frames the policy is running on, so intrinsics can be
        normalized without the encoder having to duplicate the dataset's resolution.
        """
        raise NotImplementedError

    def example_input(self, batch_size: int, frames: int = 1, image_hw=(64, 64), device=None):
        """Synthetic ``(intrinsics, extrinsics)`` for shape checks and export tracing."""
        return None, None

    def forward(
        self,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        batch_size: int | None = None,
        frames: int = 1,
        image_hw=(1, 1),
    ) -> torch.Tensor:
        if self.num_tokens == 0:
            n = batch_size if intrinsics is None else intrinsics.shape[0]
            device = None if intrinsics is None else intrinsics.device
            return torch.zeros(n, frames, 0, self.feat_size, device=device)
        if intrinsics is None or extrinsics is None:
            if batch_size is None:
                raise ValueError("batch_size is required when no calibration is supplied")
            return self.null_token.repeat(batch_size, frames, 1, 1)  # a copy, not a parameter view
        if intrinsics.shape[-2:] != (3, 3) or intrinsics.ndim != 4:
            raise ValueError(f"Intrinsics must be (B, F, 3, 3), got {tuple(intrinsics.shape)}")
        if extrinsics.shape[-2:] != (4, 4) or extrinsics.ndim != 4:
            raise ValueError(f"Extrinsics must be (B, F, 4, 4), got {tuple(extrinsics.shape)}")
        tokens = self.encode(intrinsics, extrinsics, image_hw)
        if self.training and self.p_drop > 0:
            keep = torch.rand(tokens.shape[0], 1, 1, 1, device=tokens.device) >= self.p_drop
            tokens = torch.where(keep, tokens, self.null_token.to(tokens.dtype))
        return tokens
