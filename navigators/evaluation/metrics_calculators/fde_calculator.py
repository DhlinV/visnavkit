from navigators.evaluation.metrics.fde import compute_min_fde, compute_top1_fde
from navigators.evaluation.metrics_calculators.base_calculator import MetricsCalculatorBase


class MinFDEMetricsCalculator(MetricsCalculatorBase):
    """Computes min FDE for plan predictions.

    Picks the mode with min FDE at the last timestep of the trajectory, then reports
    that mode's displacement error at the final point.
    """

    NAME = "min_FDE"
    LOWER_IS_IMPROVEMENT = True

    def __init__(self):
        pass

    def calculate(self, preds: dict, targets: dict) -> dict[str, float]:
        gt_poses = targets["action"]["future_poses"]
        return {self.name: compute_min_fde(preds, gt_poses)}


class Top1FDEMetricsCalculator(MetricsCalculatorBase):
    """Computes FDE for the planner's selected trajectory."""

    NAME = "top1_FDE"
    LOWER_IS_IMPROVEMENT = True

    def __init__(self):
        pass

    def calculate(self, preds: dict, targets: dict) -> dict[str, float]:
        gt_poses = targets["action"]["future_poses"]
        return {self.name: compute_top1_fde(preds, gt_poses)}
