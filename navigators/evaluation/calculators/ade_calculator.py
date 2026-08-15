from navigators.evaluation.metrics.ade import compute_min_ade, compute_top1_ade
from navigators.evaluation.metrics.utils import timestep_key
from navigators.evaluation.calculators.base_calculator import MetricsCalculatorBase


class MinADEMetricsCalculator(MetricsCalculatorBase):
    """Computes min ADE at specified timesteps for plan predictions.

    For each timestep, uses trajectory points from start up to that time,
    computes ADE for all modes, returns the minimum over modes (then mean over batch).
    """

    NAME = "min_ADE"
    LOWER_IS_IMPROVEMENT = True

    def __init__(
        self,
        max_val: float,
        timesteps_to_compute: list[float],
    ):
        self.max_val = max_val
        self._timesteps = tuple(timesteps_to_compute)

    @property
    def timesteps(self) -> tuple[float, ...]:
        return self._timesteps

    def calculate(self, preds: dict, targets: dict) -> dict[str, float]:
        gt_poses = targets["action"]["future_poses"]
        values = compute_min_ade(
            preds,
            gt_poses,
            self.max_val,
            self.timesteps,
        )
        return {timestep_key(t, self.name): value for t, value in values.items()}


class Top1ADEMetricsCalculator(MetricsCalculatorBase):
    """Computes ADE for the planner's selected trajectory at specified timesteps.

    For each timestep, uses trajectory points from start up to that time,
    computes ADE for the mode with the highest confidence only and returns it (then mean over batch).
    """

    NAME = "top1_ADE"
    LOWER_IS_IMPROVEMENT = True

    def __init__(
        self,
        max_val: float,
        timesteps_to_compute: list[float],
    ):
        self.max_val = max_val
        self._timesteps = tuple(timesteps_to_compute)

    @property
    def timesteps(self) -> tuple[float, ...]:
        return self._timesteps

    def calculate(self, preds: dict, targets: dict) -> dict[str, float]:
        gt_poses = targets["action"]["future_poses"]
        values = compute_top1_ade(
            preds,
            gt_poses,
            self.max_val,
            self.timesteps,
        )
        return {timestep_key(t, self.name): value for t, value in values.items()}
