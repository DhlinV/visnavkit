"""Supervision-side normalization: dataset action targets in, unit-scale actions out."""

from visnavkit.models.normalization import Normalizer

__all__ = ["ActionNormalizer"]


class ActionNormalizer(Normalizer):
    """:class:`~visnavkit.models.normalization.Normalizer` over ``(..., T, A)`` action targets.

    Generative decoders expect roughly unit-scale targets, so fit ``meanstd`` or ``minmax`` on a
    corpus once it exists. Input signals normalize separately, in their own encoders.
    """
