import numpy as np
import pytest

from visnavkit.benchmark.quality import trajectory_metrics


def test_ranked_and_oracle_metrics_have_different_selection():
    targets = np.zeros((1, 3, 2))
    predictions = np.zeros((1, 2, 3, 2))
    predictions[0, 0, :, 0] = 1
    predictions[0, 1, :, 0] = 3
    result = trajectory_metrics(predictions, targets, [0, 1, 3], [0, 1, 3], scores=[[0.1, 0.9]])
    for row in result["horizons"].values():
        assert row["min_ade"] == 1
        assert row["top1_ade"] == 3
        assert row["mean_sample_ade"] == 2
    unranked = trajectory_metrics(predictions, targets, [0, 1, 3], [0, 1, 3])
    assert "top1_ade" not in unranked["horizons"]["1s"]


def test_min_ade_selects_independently_at_each_horizon():
    predictions = np.zeros((1, 2, 3, 2))
    predictions[0, 0, :, 0] = [0, 0, 8]
    predictions[0, 1, :, 0] = [0, 2, 0]
    result = trajectory_metrics(predictions, np.zeros((1, 3, 2)), [0, 1, 2], [0, 1, 2], horizons=[1, 2])
    assert result["horizons"]["1s"]["min_ade"] == 0
    assert result["horizons"]["2s"]["min_ade"] == 1
    assert result["horizons"]["2s"]["min_fde"] == 0


def test_canonical_target_grid_is_independent_of_model_anchors():
    target_times = [0, 0.5, 1, 3]
    targets = np.zeros((1, len(target_times), 2))
    results = []
    for times in ([0, 3], [0, 0.2, 0.5, 1, 2, 3]):
        pred = np.zeros((1, 1, len(times), 2))
        pred[0, 0, :, 0] = times
        results.append(trajectory_metrics(pred, targets, times, target_times, horizons=[3]))
    assert results[0]["horizons"] == results[1]["horizons"]
    assert results[0]["horizons"]["3s"]["top1_ade"] == 1.5


def test_fde_interpolates_to_requested_horizon():
    pred = np.zeros((1, 1, 2, 2))
    pred[0, 0, :, 0] = [0, 2]
    result = trajectory_metrics(pred, np.zeros((1, 3, 2)), [0, 1], [0, 0.5, 1], horizons=[0.75, 3])
    assert result["horizons"]["0.75s"]["top1_ade"] == 1.25
    assert result["horizons"]["0.75s"]["top1_fde"] == 1.5
    assert result["horizons"]["3s"] == {"count": 0}


def test_valid_prefix_masks_control_counts_and_ignore_padding():
    targets = np.zeros((2, 3, 2))
    targets[0, :, 0] = 1
    targets[1, :, 0] = 3
    targets[1, 2] = np.nan
    result = trajectory_metrics(
        np.zeros((2, 1, 3, 2)),
        targets,
        [0, 1, 2],
        [0, 1, 2],
        horizons=[1, 2],
        valid_mask=[[True, True, True], [True, True, False]],
    )
    assert result["horizons"]["1s"]["count"] == 2
    assert result["horizons"]["1s"]["top1_ade"] == 2
    assert result["horizons"]["2s"]["count"] == 1
    assert result["horizons"]["2s"]["top1_ade"] == 1
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        trajectory_metrics(
            np.zeros((1, 1, 3, 2)), np.zeros((1, 3, 2)), [0, 1, 2], [0, 1, 2], valid_mask=[[True, False, True]]
        )


def test_speed_metric_and_invalid_timestamps():
    pred = np.zeros((1, 1, 2, 3))
    pred[..., 2] = 2
    result = trajectory_metrics(pred, np.zeros((1, 2, 3)), [0, 1], [0, 1], horizons=[1])
    assert result["horizons"]["1s"]["top1_speed_mae"] == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_metrics(pred, np.zeros((1, 2, 3)), [1, 1], [0, 1])


def test_missing_early_prediction_anchors_do_not_change_evaluation_grid():
    result = trajectory_metrics(np.zeros((1, 1, 2, 2)), np.zeros((1, 4, 2)), [1, 3], [0, 0.5, 1, 3], horizons=[3])
    assert result["horizons"]["3s"] == {"count": 0}


@pytest.mark.parametrize("horizon", [float("nan"), float("inf"), 0, -1])
def test_nonfinite_or_nonpositive_horizons_are_rejected(horizon):
    with pytest.raises(ValueError, match="horizons must be positive"):
        trajectory_metrics(np.zeros((1, 1, 2, 2)), np.zeros((1, 2, 2)), [0, 1], [0, 1], horizons=[horizon])
