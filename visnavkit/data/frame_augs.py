"""Photometric augmentations of a window's frames: pixels change, geometry does not, so targets stay put.

Runs on the train batch after it reaches the GPU, like the reference's DALI augs: hue jitter alone
costs ~140 ms per 20-frame window on a CPU worker."""

import math

import torch
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF


def _uniform(low, high):
    return low + (high - low) * torch.rand(()).item()


class FrameAugment:
    """Color jitter, then camera degradations, on the (N, 3, H, W) uint8 frames of one window.

    Color jitter draws one brightness / contrast / saturation / hue factor per window (the reference's
    ``color_jitter_*``: uniform in [1 - d, 1 + d], hue in [-d, d]). Each degradation then fires with its own
    probability, in optical order (occluder -> exposure -> optics -> sensor -> codec), as the reference's
    camera-degradation augs do:

    - ``p_erase``: lens dirt, one black rectangle with sides in ``erase_size`` (fraction of H, W), fixed
      across frames.
    - ``p_frame_brightness``: auto-exposure flicker, an independent gain in [1 - d, 1 + d] per frame, so
      consecutive frames are no longer exposure-matched.
    - ``p_grayscale``: full desaturation.
    - ``p_blur``: defocus / wet lens, one Gaussian sigma in ``blur_sigma`` px per window.
    - ``p_noise``: sensor noise, iid Gaussian per pixel and frame, stddev in ``noise_stddev`` (uint8 scale).
    - ``p_jpeg``: codec blockiness, one JPEG quality in ``jpeg_quality`` per window.
    """

    def __init__(
        self,
        color_jitter=(0.0, 0.0, 0.0, 0.0),
        p_erase=0.0,
        erase_size=(0.05, 0.2),
        p_frame_brightness=0.0,
        frame_brightness_delta=0.05,
        p_grayscale=0.0,
        p_blur=0.0,
        blur_sigma=(0.3, 1.5),
        p_noise=0.0,
        noise_stddev=(2.0, 12.0),
        p_jpeg=0.0,
        jpeg_quality=(50, 92),
    ):
        if len(color_jitter) != 4 or any(v < 0 for v in color_jitter):
            raise ValueError(
                f"color_jitter must be 4 non-negative [brightness, contrast, saturation, hue], got {color_jitter}"
            )
        probabilities = (p_erase, p_frame_brightness, p_grayscale, p_blur, p_noise, p_jpeg)
        if any(not 0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError(f"augmentation probabilities must lie in [0, 1], got {probabilities}")
        self.color_jitter = v2.ColorJitter(*color_jitter) if any(v > 0 for v in color_jitter) else None
        self.p_erase, self.erase_size = p_erase, tuple(erase_size)
        self.p_frame_brightness, self.frame_brightness_delta = p_frame_brightness, frame_brightness_delta
        self.p_grayscale = p_grayscale
        self.p_blur, self.blur_sigma = p_blur, tuple(blur_sigma)
        self.p_noise, self.noise_stddev = p_noise, tuple(noise_stddev)
        self.p_jpeg, self.jpeg_quality = p_jpeg, tuple(jpeg_quality)
        self.enabled = self.color_jitter is not None or any(p > 0 for p in probabilities)

    @staticmethod
    def _fires(p):
        return p > 0 and bool(torch.rand(()) < p)

    def __call__(self, frames):
        if not self.enabled or frames.numel() == 0:
            return frames
        if self.color_jitter is not None:
            frames = self.color_jitter(frames)
        if self._fires(self.p_erase):
            h, w = frames.shape[-2:]
            eh, ew = round(_uniform(*self.erase_size) * h), round(_uniform(*self.erase_size) * w)
            y0, x0 = int(torch.randint(0, h, ())), int(torch.randint(0, w, ()))
            frames = frames.clone()
            frames[..., y0 : y0 + eh, x0 : x0 + ew] = 0
        if self._fires(self.p_frame_brightness):
            d = self.frame_brightness_delta
            gain = 1.0 + d * (2.0 * torch.rand(frames.shape[0], 1, 1, 1, device=frames.device) - 1.0)
            frames = (frames.float() * gain).round_().clamp_(0, 255).to(torch.uint8)
        if self._fires(self.p_grayscale):
            frames = TF.rgb_to_grayscale(frames, num_output_channels=3)
        if self._fires(self.p_blur):
            sigma = _uniform(*self.blur_sigma)
            k = 2 * math.ceil(3 * sigma) + 1
            frames = TF.gaussian_blur(frames, [k, k], [sigma, sigma])
        if self._fires(self.p_noise):
            noise = torch.randn(frames.shape, device=frames.device) * _uniform(*self.noise_stddev)
            frames = (frames.float() + noise).round_().clamp_(0, 255).to(torch.uint8)
        if self._fires(self.p_jpeg):
            encoded = encode_jpeg(list(frames), quality=round(_uniform(*self.jpeg_quality)))  # nvjpeg on a GPU
            frames = torch.stack(decode_jpeg([e.cpu() for e in encoded], device=frames.device))
        return frames

    def batch(self, vision, frame_mask=None):
        """Each window of ``vision`` (B, S, 3, H, W) uint8 on its own draw, its real frames only; in place."""
        for b in range(vision.shape[0]):
            real = frame_mask[b] if frame_mask is not None else slice(None)
            vision[b, real] = self(vision[b, real])
        return vision
