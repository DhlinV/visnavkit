"""Predictor registration with explicit model-space to metric-space conversion."""

import numpy as np

from visnavkit.benchmark.registry import PREDICTORS


@PREDICTORS.register("stationary")
class Stationary:
    selection = "single_trajectory"

    def predict(self, data, times):
        n = len(data["targets"])
        return np.zeros((n, 1, len(times), 2), dtype=np.float32), None


@PREDICTORS.register("constant_velocity")
class ConstantVelocity:
    selection = "single_trajectory"

    def predict(self, data, times):
        if "current_speed" not in data:
            raise ValueError("constant_velocity needs current_speed measured at the last observation.")
        speed = np.asarray(data["current_speed"]).reshape(-1)
        if len(speed) != len(data["targets"]) or not np.isfinite(speed).all():
            raise ValueError("current_speed must be finite and match samples.")
        pred = np.zeros((len(speed), 1, len(times), 2), dtype=np.float32)
        pred[:, 0, :, 0] = speed[:, None] * np.asarray(times)
        return pred, None


@PREDICTORS.register("onnx")
class OnnxPredictor:
    """Inputs are prepared per model; this adapter never invents missing goals/depth."""

    def __init__(self, session, *, output_name="trajectories", scores_name=None, xy_scale=1.0, layout="NKTD", speed_index=None):
        if layout not in ("NKTD", "NTD"):
            raise ValueError("layout must be NKTD or NTD; use a custom registered predictor for other outputs.")
        if not np.isfinite(xy_scale) or xy_scale <= 0:
            raise ValueError("xy_scale must be finite and positive.")
        self.session = session
        self.output_name = output_name
        self.scores_name = scores_name
        self.xy_scale = xy_scale
        self.layout = layout
        self.speed_index = speed_index

    def predict(self, data, times):
        inputs = self.session.get_inputs()
        count = len(data["targets"])
        trajectories, scores = [], []
        for i in range(count):
            feeds = {}
            for node in inputs:
                key = node.name if node.name == "vision" else f"input__{node.name}"
                if key not in data:
                    raise ValueError(f"Prepared dataset missing {key!r} for ONNX input {node.name!r}.")
                values = np.asarray(data[key])
                if values.shape[0] != count:
                    raise ValueError(f"{key} must have leading sample dimension {count}.")
                value = values[i:i + 1]
                if node.name == "vision" and value.dtype == np.uint8:
                    value = value.astype(np.float32) / 255.0
                # Diffusion noise input has candidate batch as its leading dimension.
                if node.name == "initial_noise":
                    value = values[i]
                feeds[node.name] = np.ascontiguousarray(value)
            names = [self.output_name] + ([self.scores_name] if self.scores_name else [])
            outputs = self.session.run(names, feeds)
            pred = np.asarray(outputs[0])
            if self.layout == "NTD":
                pred = pred[:, None]
            if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[-2] != len(times):
                raise ValueError(f"Invalid trajectory shape {pred.shape}; expected (1,K,{len(times)},D).")
            channels = [0, 1] + ([self.speed_index] if self.speed_index is not None else [])
            pred = pred[..., channels].copy()
            pred[..., :2] *= self.xy_scale
            trajectories.append(pred)
            if self.scores_name:
                scores.append(outputs[1])
        return np.concatenate(trajectories), np.concatenate(scores) if scores else None
