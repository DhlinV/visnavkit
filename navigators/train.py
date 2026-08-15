import subprocess
import sys
import time
import warnings

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from navigators.evaluation.metrics_calculators.base_calculator import MetricsCalculatorBase
from navigators.utils.logger import get_logger

logger = get_logger(__name__)

# PyTorch bug that raises a false-positive warning
# More info: https://github.com/Lightning-AI/litgpt/issues/1561
warning_message = r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"
warnings.filterwarnings(
    action="ignore", message=warning_message, category=UserWarning, module=r".*torch\.optim\.lr_scheduler.*"
)


torch.set_float32_matmul_precision("medium")


def recursive_flatten_01(d):
    """Recursively flatten along dims 0 and 1"""
    if isinstance(d, torch.Tensor):
        return d.flatten(0, 1)
    elif isinstance(d, dict):
        return {k: recursive_flatten_01(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_flatten_01(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(recursive_flatten_01(v) for v in d)
    else:
        return d  # leave other types unchanged


def check_git_status():
    # Check for uncommitted changes (both staged and unstaged)
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        print("ERROR: Uncommitted changes detected. Please commit or stash your changes before training.")
        sys.exit(1)
    print("Git status is clean. Proceeding with training.")


def build_targets(batch):
    """Build flattened targets from batch (vision, policy). Returns targets."""
    return recursive_flatten_01(
        dict(
            vision=dict(frame_speeds=batch["frame_speeds"]),
            policy=dict(future_poses=batch["future_poses"]),
        )
    )


def load_pretrained_model_weights(model: torch.nn.Module, pretrained_cfg: DictConfig) -> None:
    if not pretrained_cfg or not pretrained_cfg.get("ckpt_path"):
        return

    ckpt_path = pretrained_cfg.get("ckpt_path")
    strict = pretrained_cfg.get("strict", True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning checkpoints store LitModel keys like "model.vision_model..."
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
    """Run all metric calculators on predictions/targets and log results to wandb."""
    if not calculators:
        return

    metrics = {}
    for calc in calculators:
        metrics.update(calc.calculate(y_hat, targets))
    for k, v in metrics.items():
        lit_model.log(f"val/metrics/{k}", v, batch_size=batch_size, sync_dist=True)


class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        if pretrained_cfg := cfg.get("pretrained"):
            load_pretrained_model_weights(self.model, pretrained_cfg)
        self.automatic_optimization = False
        validation_metrics_cfg = cfg.metrics.get("validation_metrics") or {}
        planner_calculators_cfg = validation_metrics_cfg.get("planner_calculators") or []
        self.planner_calculators = [instantiate(c) for c in planner_calculators_cfg]

    def _step(self, batch, batch_idx, stage: str):
        start_time = time.time()
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        x = batch["frames"]
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        x = x.to(self.device, non_blocking=True)
        batch_size, seq_len = x.shape[:2]
        effective_batch_size = batch_size * seq_len
        targets = build_targets(batch)
        route_patch = batch.get("route_patch")
        if route_patch is not None:
            route_patch = route_patch.to(self.device, non_blocking=True)

        y_hat = self.model(x, route_patch=route_patch)

        loss_dict, loss_debug = self.model.get_losses(y_hat, targets)

        for k, v in loss_dict.items():
            self.log(f"{stage}/{k}", v, batch_size=effective_batch_size, sync_dist=(stage == "val"))

        loss = loss_dict["loss"]
        if stage == "train":
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

        plan_head = self.model.policy_model.plan_head
        planner_preds = plan_head.parse_output(y_hat["policy"]["plan"]["plans"])
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
        warmup_steps = cfg.optimizer.warmup_steps
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=cfg.optimizer.eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    check_git_status()
    if not cfg.get("dataset"):
        sys.exit("ERROR: no dataset config yet. Add navigators/configs/dataset/<name>.yaml and pass dataset=<name>.")
    L.seed_everything(cfg.seed, workers=True)

    # data
    datamodule = instantiate(cfg.dataset)

    # logs
    wandb_project = cfg.get("wandb_project", "navigators")
    wandb_logger = WandbLogger(project=wandb_project, name=cfg.exp_name, save_dir=cfg.log_dir)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/policy_reg",
        save_top_k=cfg.trainer.logging.save_top_k_ckpts,
        save_last=True,
        mode="min",
        filename="epoch_{epoch:02d}-step_{step}-val_loss_{val/policy_reg:.4f}",
        auto_insert_metric_name=False,
    )

    # trainer
    trainer = L.Trainer(**cfg.trainer.kwargs, logger=wandb_logger, callbacks=[lr_monitor, checkpoint_callback])
    model = LitModel(cfg)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.trainer.resume.ckpt_path)


if __name__ == "__main__":
    main()
