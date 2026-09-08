import pytest
import torch

from visnavkit.evaluation.calculators.ade_calculator import MinADEMetricsCalculator, Top1ADEMetricsCalculator
from visnavkit.evaluation.metrics.ade import compute_min_ade, compute_top1_ade


def predictions(errors):
    plans = torch.zeros(1, len(errors), len(errors[0]), 2)
    plans[0, :, :, 0] = torch.tensor(errors)
    return {"plans": plans, "best_plan": plans[:, 0]}


def test_minimum_mode_is_chosen_separately_for_each_horizon():
    preds = predictions([[0, 0, 10], [1, 1, 0]])
    values = compute_min_ade(preds, torch.zeros(1, 3, 2), 3.0, [0.5, 3.0], torch.tensor([0.0, 0.5, 3.0]))
    assert values[0.5].item() == 0.0
    assert values[3.0].item() == pytest.approx(2 / 3)


def test_explicit_offset_times_change_horizon_membership():
    preds = predictions([[2, 10, 20]])
    values = compute_top1_ade(preds, torch.zeros(1, 3, 2), 3.0, [1.0, 3.0], torch.tensor([0.4, 1.2, 3.0]))
    assert values[1.0].item() == 2.0
    assert values[3.0].item() == pytest.approx(32 / 3)
    legacy = compute_top1_ade(preds, torch.zeros(1, 3, 2), 3.0, [1.0])
    assert legacy[1.0].item() == 6.0


@pytest.mark.parametrize("calculator_cls", [MinADEMetricsCalculator, Top1ADEMetricsCalculator])
def test_calculators_forward_actual_target_times(calculator_cls):
    calculator = calculator_cls(3.0, [1.0])
    preds = predictions([[2, 10, 20]])
    targets = {"action": {"future_poses": torch.zeros(1, 3, 2), "target_times_s": torch.tensor([[0.4, 1.2, 3.0]])}}
    values = calculator.calculate(preds, targets)
    assert list(values.values())[0].item() == 2.0


def test_per_example_timestamps_average_examples_equally():
    preds = predictions([[2, 10, 20]])
    preds = {key: value.expand(2, *value.shape[1:]) for key, value in preds.items()}
    times = torch.tensor([[0.4, 1.2, 3.0], [0.1, 0.2, 3.0]])
    values = compute_top1_ade(preds, torch.zeros(2, 3, 2), 3.0, [1.0], times)
    assert values[1.0].item() == 4.0


def test_malformed_target_timestamps_rejected():
    with pytest.raises(ValueError, match="point_times"):
        compute_min_ade(predictions([[1, 2, 3]]), torch.zeros(1, 3, 2), 3.0, [1.0], torch.tensor([0.0, 3.0]))
