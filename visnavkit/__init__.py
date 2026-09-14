"""VisNavKit: visual navigation models, training, and benchmarks."""

from omegaconf import ListConfig, OmegaConf


def _goal_types(encoder):
    """Dataset-facing goal type(s) for one goal encoder or a list of them."""
    if isinstance(encoder, (ListConfig, list, tuple)):
        return [entry["goal_type"] for entry in encoder]
    return encoder["goal_type"]


# Lets dataset configs track `model.goal_encoder` whether it is one encoder or a list.
OmegaConf.register_new_resolver("goal_types", _goal_types, replace=True)
