import copy
import time
import warnings
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torchvision.io import write_png

from visnavkit.data.frame_augs import FrameAugment
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


ROUTE_COLORS = torch.tensor(  # route class id -> RGB: background, sidewalk, crosswalk, then spares
    [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [128, 128, 128], [255, 255, 0]], dtype=torch.uint8
)


def route_image(batch):
    """The batch's first window with a current route patch: [current frame | route patch in colour], (3, H, W + H) uint8."""
    if "route_patch" not in batch or "route_mask" not in batch or not batch["route_mask"][:, -1].any():
        return None
    i = int(batch["route_mask"][:, -1].nonzero()[0])
    frame = batch["vision"][i, -1].cpu()
    if frame.dtype != torch.uint8:
        frame = (frame.float() * 255).round().clamp(0, 255).to(torch.uint8)
    ids = batch["route_patch"][i, -1].cpu().long().clamp(0, len(ROUTE_COLORS) - 1)
    patch = ROUTE_COLORS[ids].permute(2, 0, 1)[None].float()
    patch = F.interpolate(patch, size=(frame.shape[-2],) * 2, mode="nearest")[0].to(torch.uint8)
    return torch.cat([frame, patch], -1)


def disable_pretrained_downloads(model_cfg: DictConfig) -> DictConfig:
    """Complete checkpoints carry every weight; skip backbone downloads and initialization files, however nested."""
    for key, value in model_cfg.items():
        if key == "pretrained":
            model_cfg[key] = False
        elif key == "weights":
            model_cfg[key] = None
        elif isinstance(value, DictConfig):
            disable_pretrained_downloads(value)
        elif isinstance(value, ListConfig):
            for item in value:
                if isinstance(item, DictConfig):
                    disable_pretrained_downloads(item)
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
        self.augment = FrameAugment(**augs) if (augs := cfg.get("augs")) else None

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
        if stage == "train" and self.augment is not None and x.dtype == torch.uint8:
            x = self.augment.batch(x.to(self.device, non_blocking=True), batch.get("frame_mask"))
            batch["vision"] = x  # the recorded route image shows what the policy saw
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
        if missing := [name for name in self.model.modality_input_names if name not in batch]:
            # Silently falling back to the null token would look like training, so say it once.
            self._warned = getattr(self, "_warned", False)
            if not self._warned:
                logger.warning(f"Dataset provides no {missing}; those modalities fall back to their null token")
                self._warned = True

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
        every = (self.cfg.trainer.logging.get("log_images_every_n_batches") or {}).get(stage)
        if every and batch_idx % every == 0 and self.trainer.is_global_zero:
            self.record_route(batch, stage)

        return y_hat, targets, loss_debug, x, effective_batch_size

    def on_fit_start(self):
        self.images_dir = Path(self.trainer.log_dir or ".") / "images"  # on every rank: log_dir is a broadcast

    def record_route(self, batch, stage):
        """The route image to <log_dir>/images/<stage>_route_step<N>.png and to every logger that takes images."""
        image = route_image(batch)
        if image is None:
            return
        path = self.images_dir / f"{stage}_route_step{self.global_step:07d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(image, str(path))
        for experiment_logger in self.loggers:
            if hasattr(experiment_logger, "log_image"):  # e.g. WandbLogger
                image_hwc = image.permute(1, 2, 0).numpy()
                experiment_logger.log_image(key=f"{stage}/route", images=[image_hwc], step=self.global_step)

    def training_step(self, batch, batch_idx):
        self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        y_hat, targets, loss_debug, x, effective_batch_size = self._step(batch, batch_idx, stage="val")

        planner_preds = self.model.action_decoder.parse_output(y_hat.plan.plans)
        valid = getattr(y_hat.plan, "valid", None)  # rows the policy did not decide (e.g. slots without a frame)
        if valid is not None:
            planner_preds = {k: v[valid] for k, v in planner_preds.items()}
            targets["action"] = {
                k: v[valid] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(valid) else v
                for k, v in targets["action"].items()
            }
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
