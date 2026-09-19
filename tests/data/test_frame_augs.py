import pytest
import torch

from visnavkit.data.frame_augs import FrameAugment

EVERY = dict(color_jitter=[0.5, 0.5, 0.5, 0.2], p_erase=1, p_frame_brightness=1, p_grayscale=1, p_blur=1, p_noise=1)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]
)
def test_every_aug_changes_real_frames_only(device):
    torch.manual_seed(0)
    vision = torch.randint(30, 220, (2, 6, 3, 24, 32), dtype=torch.uint8, device=device)
    mask = torch.tensor([[1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.bool, device=device)
    vision[~mask] = 0
    before = vision.clone()
    out = FrameAugment(**EVERY, p_jpeg=1).batch(vision, mask)
    assert out.dtype == torch.uint8 and out.shape == before.shape and out.device == before.device
    assert torch.all(out[~mask] == 0)  # empty slots stay empty
    assert not torch.equal(out[mask], before[mask])
    assert torch.equal(FrameAugment().batch(before.clone(), mask), before)  # nothing configured: identity


def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="color_jitter"):
        FrameAugment(color_jitter=[0.2, 0.2])
    with pytest.raises(ValueError, match="probabilities"):
        FrameAugment(p_blur=1.5)
