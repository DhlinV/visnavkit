import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp

from .base import BaseModalityEncoder

__all__ = ["PinholeCameraEncoder"]

FEATURE_DIM = 16


class PinholeCameraEncoder(BaseModalityEncoder):
    """Per-frame pinhole calibration: ``intrinsics (B, F, 3, 3)`` and ``extrinsics (B, F, 4, 4)``.

    The intrinsics are normalized by the size of the frames the policy is actually running on
    (``fx / W``, ``fy / H``, ``cx / W``, ``cy / H``), so the same weights transfer across crops
    and downscales; the camera-to-ego extrinsics contribute their rotation matrix and
    translation. That is 16 numbers per frame, fed to an MLP.
    """

    input_names = ("intrinsics", "extrinsics")

    def __init__(self, feat_size: int, hidden: int = 128, num_tokens: int = 1, p_drop: float = 0.0, **kwargs):
        super().__init__(feat_size, num_tokens=num_tokens, p_drop=p_drop, **kwargs)
        self.mlp = build_mlp(FEATURE_DIM, hidden, num_tokens * feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    @staticmethod
    def features(intrinsics: torch.Tensor, extrinsics: torch.Tensor, image_hw) -> torch.Tensor:
        """``(B, F, 3, 3)`` and ``(B, F, 4, 4)`` -> ``(B, F, 16)`` calibration features."""
        height, width = float(image_hw[0]), float(image_hw[1])
        focal = torch.stack([intrinsics[..., 0, 0] / width, intrinsics[..., 1, 1] / height], dim=-1)
        principal = torch.stack([intrinsics[..., 0, 2] / width, intrinsics[..., 1, 2] / height], dim=-1)
        rotation = extrinsics[..., :3, :3].flatten(-2)
        translation = extrinsics[..., :3, 3]
        return torch.cat([focal, principal, rotation, translation], dim=-1)

    def encode(self, intrinsics, extrinsics, *, image_hw):
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Intrinsics must be (B, F, 3, 3), got {tuple(intrinsics.shape)}")
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (4, 4):
            raise ValueError(f"Extrinsics must be (B, F, 4, 4), got {tuple(extrinsics.shape)}")
        b, f = intrinsics.shape[:2]
        features = self.features(intrinsics.float(), extrinsics.float(), image_hw)
        return self.norm(self.mlp(features).reshape(b, f, self.num_tokens, self.feat_size))

    def example_inputs(self, batch_size, frames=1, image_hw=(64, 64), device=None):
        height, width = image_hw
        intrinsics = torch.eye(3, device=device).repeat(batch_size, frames, 1, 1)
        intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = float(max(height, width))
        intrinsics[..., 0, 2], intrinsics[..., 1, 2] = width / 2.0, height / 2.0
        extrinsics = torch.eye(4, device=device).repeat(batch_size, frames, 1, 1)
        return intrinsics, extrinsics
