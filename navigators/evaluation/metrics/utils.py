"""Shared helpers for metrics."""


def timestep_key(t: float, prefix: str) -> str:
    """Return a deterministic key for timestep t with the given prefix.

    Decimal is replaced by _; whole numbers get _0 (e.g. 2 -> {prefix}_2_0, 2.5 -> {prefix}_2_5).
    No trailing zeros in fractions.

    Examples:
        timestep_key(2.0, "min_ade") -> "min_ade_2_0"
        timestep_key(0.5, "min_fde") -> "min_fde_0_5"
        timestep_key(1.5, "min_dist_to_undriveable") -> "min_dist_to_undriveable_1_5"
    """
    x = float(t)
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        suffix = f"{int(x)}_0"
    else:
        suffix = s.replace(".", "_")
    return f"{prefix}@{suffix}s"
