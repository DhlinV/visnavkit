"""Lightning module behind ``visnavkit-train-route``: a route-patch AE or VAE."""

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from visnavkit.models.lit_model import make_lr_scheduler


class LitRouteAutoencoder(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg})
        self.model = instantiate(cfg.route)

    def _step(self, batch, stage: str):
        x = batch["route_image"]
        x = x.float().div(255.0) if x.dtype == torch.uint8 else x
        losses = self.model.loss(x, self.model(x))
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, batch_size=x.shape[0], sync_dist=stage == "val", prog_bar=name == "loss")
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = self.cfg.optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = make_lr_scheduler(
            optimizer,
            total_steps=int(self.trainer.estimated_stepping_batches),
            warmup_steps=opt.warmup_steps,
            eta_min=opt.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
