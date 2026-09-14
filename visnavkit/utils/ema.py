"""Exponential moving average of the policy weights (diffusers ``EMAModel``, diffusion policy).

Denoising decoders are sensitive to the weight noise of the last few steps, so diffusion
policy, NoMaD and openpi all evaluate and ship an averaged copy of the network instead of
the online one.
"""

import lightning as L
import torch

__all__ = ["EMACallback"]


class EMACallback(L.Callback):
    """Track an EMA of every floating-point tensor in the module's ``state_dict``.

    The averaged weights are swapped in for validation and written into the checkpoint's
    ``state_dict``, so monitored metrics, export and the benchmark all see the EMA policy.
    The online weights travel alongside them under ``ema_online_state_dict``, so resuming a
    run continues from the exact weights the optimizer state belongs to.

    Args:
        decay: upper bound on the decay.
        inv_gamma, power: warmup of the diffusers/diffusion-policy schedule
            ``1 - (1 + step / inv_gamma) ** -power``, so early steps are not pinned to the
            initialization.
        every_n_steps: update interval, in optimizer steps.
    """

    def __init__(self, decay: float = 0.9999, inv_gamma: float = 1.0, power: float = 2 / 3, every_n_steps: int = 1):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if inv_gamma <= 0 or power <= 0 or every_n_steps < 1:
            raise ValueError("inv_gamma and power must be positive and every_n_steps at least 1")
        self.decay = decay
        self.inv_gamma = inv_gamma
        self.power = power
        self.every_n_steps = every_n_steps
        self.shadow: dict[str, torch.Tensor] | None = None
        self._online: dict[str, torch.Tensor] | None = None

    def current_decay(self, step: int) -> float:
        return min(max(1.0 - (1.0 + max(step, 0) / self.inv_gamma) ** -self.power, 0.0), self.decay)

    @staticmethod
    def _averaged(pl_module: L.LightningModule) -> dict[str, torch.Tensor]:
        return {name: value for name, value in pl_module.state_dict().items() if value.is_floating_point()}

    def _align(self, state: dict[str, torch.Tensor]) -> None:
        """Create the shadow on first use, or move a checkpoint-restored one onto the device."""
        device = next(iter(state.values())).device
        if self.shadow is None:
            self.shadow = {name: value.detach().to(torch.float32).clone() for name, value in state.items()}
        elif next(iter(self.shadow.values())).device != device:
            self.shadow = {name: value.to(device) for name, value in self.shadow.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        state = self._averaged(pl_module)
        self._align(state)
        if trainer.global_step % self.every_n_steps:
            return
        weight = 1.0 - self.current_decay(trainer.global_step)
        for name, value in state.items():
            self.shadow[name].lerp_(value.detach().to(torch.float32), weight)

    def on_validation_start(self, trainer, pl_module) -> None:
        if self.shadow is None:
            return
        self._online = {name: value.detach().clone() for name, value in self._averaged(pl_module).items()}
        pl_module.load_state_dict(self.shadow, strict=False)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._online is not None:
            pl_module.load_state_dict(self._online, strict=False)
            self._online = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        state = checkpoint.get("state_dict")
        if self.shadow is None or state is None:
            return
        checkpoint["ema_online_state_dict"] = {name: state[name] for name in self.shadow if name in state}
        for name, value in self.shadow.items():
            if name in state:
                state[name] = value.to(state[name].dtype)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        online = checkpoint.get("ema_online_state_dict")
        if online is None:
            return
        state = checkpoint["state_dict"]
        self.shadow = {name: state[name].detach().to(torch.float32).clone() for name in online}
        pl_module.load_state_dict(online, strict=False)  # the optimizer state belongs to these weights
