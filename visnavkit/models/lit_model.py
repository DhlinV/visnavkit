import copy
import time
import warnings

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from visnavkit.evaluation.calculators.base_calculator import MetricsCalculatorBase
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def _to_device(value, device):
    """Batch entries may be a tensor, a list of tensors (multi-goal policies), or missing."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [item.to(device, non_blocking=True) for item in value]
    return value.to(device, non_blocking=True)


# PyTorch bug that raises a false-positive warning
# More info: https://github.com/Lightning-AI/litgpt/issues/1561
warning_message = r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"
warnings.filterwarnings(
    action="ignore", message=warning_message, category=UserWarning, module=r".*torch\.optim\.lr_scheduler.*"
)


def build_targets(batch, action_reduction="none"):
    """Supervise either every action or the final decision, plus optional per-frame vision targets.

    Reduced temporal features describe the complete observed window; their action
    target is the trajectory following its final frame.
    """
    if action_reduction not in {"none", "last", "avg", "sum"}:
        raise ValueError(f"Unknown action reduction: {action_reduction}")
    future_poses = batch["future_poses"]
    targets = dict(
        vision={},
        action=dict(future_poses=future_poses.flatten(0, 1) if action_reduction == "none" else future_poses[:, -1]),
    )
    if "frame_speeds" in batch:
        targets["vision"]["frame_speeds"] = batch["frame_speeds"].flatten(0, 1)

    if "target_times_s" in batch:
        times = batch["target_times_s"]
        targets["action"]["target_times_s"] = (
            times
            if times.ndim == 1 or action_reduction != "none"
            else times.repeat_interleave(future_poses.shape[1], dim=0)
        )
    return targets


def disable_pretrained_downloads(model_cfg: DictConfig) -> DictConfig:
    """Complete checkpoints carry every weight; skip backbone downloads and initialization files."""
    goal_cfg = model_cfg.get("goal_encoder")
    goal_cfgs = list(goal_cfg) if isinstance(goal_cfg, ListConfig) else [goal_cfg]
    for component in [model_cfg.vision_encoder, *goal_cfgs]:
        if component is None:
            continue
        if "pretrained" in component:
            component.pretrained = False
        if "weights" in component:
            component.weights = None
    return model_cfg


def load_pretrained_model_weights(model: torch.nn.Module, pretrained_cfg: DictConfig) -> None:
    if not pretrained_cfg or not pretrained_cfg.get("ckpt_path"):
        return

    ckpt_path = pretrained_cfg.get("ckpt_path")
    strict = pretrained_cfg.get("strict", True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning checkpoints store LitModel keys like "model.vision_encoder..."
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}

    model.load_state_dict(state_dict, strict=strict)


@torch.no_grad()
def compute_and_log_metrics(
    lit_model: L.LightningModule,
    y_hat: dict,
    targets: dict,
    calculators: list[MetricsCalculatorBase],
    batch_size: int,
) -> None:
    """Run all metric calculators on predictions/targets and log results to the configured logger."""
    if not calculators:
        return

    metrics = {}
    for calc in calculators:
        metrics.update(calc.calculate(y_hat, targets))
    for k, v in metrics.items():
        lit_model.log(f"val/metrics/{k}", v, batch_size=batch_size, sync_dist=True)


def make_lr_scheduler(optimizer, *, total_steps, warmup_steps, eta_min):
    if total_steps < 1 or warmup_steps < 0:
        raise ValueError("total_steps must be positive and warmup_steps nonnegative")
    warmup_steps = min(int(warmup_steps), int(total_steps))
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    if warmup_steps == total_steps:
        return warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig, initialize_pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg})
        model_cfg = copy.deepcopy(cfg.model)
        if not initialize_pretrained:
            disable_pretrained_downloads(model_cfg)
        self.model = instantiate(model_cfg)
        if initialize_pretrained and (pretrained_cfg := cfg.get("pretrained")):
            load_pretrained_model_weights(self.model, pretrained_cfg)
        self.automatic_optimization = False
        validation_metrics_cfg = cfg.metrics.get("validation_metrics") or {}
        planner_calculators_cfg = validation_metrics_cfg.get("planner_calculators") or []
        self.planner_calculators = [instantiate(c) for c in planner_calculators_cfg]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # Complete checkpoints contain their own weights; backbone initialization
        # must not trigger downloads or depend on a previous initialization file.
        kwargs["initialize_pretrained"] = False
        kwargs.setdefault("weights_only", False)  # checkpoints store the OmegaConf config
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

    def _step(self, batch, batch_idx, stage: str):
        start_time = time.time()
        x = batch["vision"]
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        x = x.to(self.device, non_blocking=True)
        batch_size, seq_len = x.shape[:2]
        effective_batch_size = batch_size * seq_len
        reduction = self.model.temporal_encoder.reduction
        targets = build_targets(batch, action_reduction=reduction)
        if reduction != "none":
            effective_batch_size = batch_size
        goal = _to_device(batch.get("goal"), self.device)
        modalities = {
            name: _to_device(batch[name], self.device) for name in self.model.modality_input_names if name in batch
        }

        y_hat = self.model(x, goal=goal, **modalities)

        loss_dict, loss_debug = self.model.get_losses(y_hat, targets)

        for k, v in loss_dict.items():
            self.log(f"{stage}/{k}", v, batch_size=effective_batch_size, sync_dist=(stage == "val"))

        loss = loss_dict["loss"]
        if stage == "train":
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            # clip_grad_norm_ returns the pre-clip total norm; a falsy max_grad_norm disables clipping.
            max_grad_norm = self.cfg.optimizer.get("max_grad_norm")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_grad_norm if max_grad_norm else float("inf")
            )
            self.log("train/grad_norm", grad_norm, batch_size=effective_batch_size)
            optimizer.step()
            scheduler.step()

        elapsed = time.time() - start_time
        rps = torch.tensor(x.shape[0] / elapsed, device=self.device, dtype=torch.float32)
        self.log(f"{stage}/rps", rps, prog_bar=True, sync_dist=True, reduce_fx="sum")

        return y_hat, targets, loss_debug, x, effective_batch_size

    def training_step(self, batch, batch_idx):
        self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        y_hat, targets, loss_debug, x, effective_batch_size = self._step(batch, batch_idx, stage="val")

        planner_preds = self.model.action_decoder.parse_output(y_hat.plan.plans)
        compute_and_log_metrics(
            self,
            planner_preds,
            targets,
            self.planner_calculators,
            effective_batch_size,
        )

    def configure_optimizers(self):
        cfg = self.cfg
        steps = int(self.trainer.estimated_stepping_batches)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler = make_lr_scheduler(
            optimizer, total_steps=steps, warmup_steps=cfg.optimizer.warmup_steps, eta_min=cfg.optimizer.eta_min
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
