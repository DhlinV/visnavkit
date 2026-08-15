from abc import ABC, abstractmethod
from typing import ClassVar

from navigators.evaluation.metrics.utils import timestep_key


class MetricsCalculatorBase(ABC):
    """Abstract base class for validation metrics calculators."""

    NAME: ClassVar[str]
    LOWER_IS_IMPROVEMENT: ClassVar[bool]

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def lower_is_improvement(self) -> bool:
        return self.LOWER_IS_IMPROVEMENT

    @property
    def timesteps(self) -> tuple[float, ...]:
        return ()

    @property
    def metric_keys(self) -> tuple[str, ...]:
        if self.timesteps:
            return tuple(timestep_key(t, self.name) for t in self.timesteps)
        return (self.name,)

    @abstractmethod
    def calculate(self, preds: dict, targets: dict) -> dict[str, float]:
        """
        Compute metrics from model predictions and targets.

        Args:
            preds: Parsed planner predictions
            targets: Ground-truth dict (e.g. future_poses, frame_speeds).
            **kwargs: Optional extra context (e.g. batch size, model ref).

        Returns:
            Dict mapping metric names to scalar values (or tensors for Lightning reduction).
        """
        pass
