"""Supervision and input signals normalize independently, with their own methods."""

import numpy as np
import pytest
import torch

from visnavkit.models.action import ActionNormalizer
from visnavkit.models.goal import GpsGoalEncoder
from visnavkit.models.modality import VectorEncoder
from visnavkit.models.normalization import Normalizer


@pytest.mark.parametrize("mode", ["meanstd", "minmax"])
def test_fit_save_load_round_trip(tmp_path, mode):
    torch.manual_seed(0)
    values = torch.randn(64, 4, 2) * 3 + 1
    fitted = Normalizer(mode=mode).fit(values)
    normalized = fitted.normalize(values)
    assert normalized.abs().max() <= (1.0 + 1e-5 if mode == "minmax" else 10.0)
    torch.testing.assert_close(fitted.unnormalize(normalized), values, atol=1e-4, rtol=1e-4)

    path = tmp_path / "stats.npz"
    fitted.save_stats(path)
    torch.testing.assert_close(Normalizer(mode=mode, stats_path=str(path)).normalize(values), normalized)


def test_scale_mode_needs_no_corpus():
    """A known range is a hand-set divisor, not statistics to fit."""
    scaled = Normalizer(mode="scale", scale=[20.0, 4.0])
    torch.testing.assert_close(scaled.normalize(torch.tensor([[10.0, 2.0]])), torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(
        scaled.unnormalize(scaled.normalize(torch.tensor([[3.0, 1.0]]))), torch.tensor([[3.0, 1.0]])
    )
    with pytest.raises(ValueError, match="nothing to fit"):
        scaled.fit(torch.randn(4, 2, 2))
    with pytest.raises(ValueError, match="mode=scale requires scale"):
        Normalizer(mode="scale")
    with pytest.raises(ValueError, match="only used with mode=scale"):
        Normalizer(mode="meanstd", scale=2.0)
    assert Normalizer().identity and not scaled.identity


def test_supervision_and_input_normalizers_are_independent(tmp_path):
    """The action targets and an ego vector carry different statistics in the same policy."""
    stats = tmp_path / "ego.npz"
    np.savez(stats, mean=np.array([1.0, 0.0], dtype=np.float32), std=np.array([2.0, 0.5], dtype=np.float32))
    ego = VectorEncoder(8, in_dim=2, hidden=16, normalizer=Normalizer(mode="meanstd", stats_path=str(stats)))
    actions = ActionNormalizer(mode="scale", scale=10.0)

    torch.testing.assert_close(ego.normalizer.normalize(torch.tensor([[3.0, 0.5]])), torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(actions.normalize(torch.tensor([[5.0, 5.0]])), torch.tensor([[0.5, 0.5]]))
    assert ego(torch.randn(2, 3, 2), image_hw=(32, 32)).shape == (2, 3, 1, 8)


def test_gps_goal_scales_metres_and_defaults_to_the_horizon():
    encoder = GpsGoalEncoder(8, hidden=16)
    assert encoder.goal_type == "gps" and encoder.per_frame
    torch.testing.assert_close(encoder.normalizer.normalize(torch.tensor([[20.0, -10.0]])), torch.tensor([[1.0, -0.5]]))
    assert encoder(encoder.example_input(3)).shape == (3, 1, 8)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        encoder(torch.zeros(3, 3))
