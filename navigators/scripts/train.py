import subprocess
import sys

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from navigators.models.lit_model import LitModel

torch.set_float32_matmul_precision("medium")


def check_git_status():
    # Check for uncommitted changes (both staged and unstaged)
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        print("ERROR: Uncommitted changes detected. Please commit or stash your changes before training.")
        sys.exit(1)
    print("Git status is clean. Proceeding with training.")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
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
        monitor="val/action_reg",
        save_top_k=cfg.trainer.logging.save_top_k_ckpts,
        save_last=True,
        mode="min",
        filename="epoch_{epoch:02d}-step_{step}-val_loss_{val/action_reg:.4f}",
        auto_insert_metric_name=False,
    )

    # trainer
    trainer = L.Trainer(**cfg.trainer.kwargs, logger=wandb_logger, callbacks=[lr_monitor, checkpoint_callback])
    model = LitModel(cfg)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.trainer.resume.ckpt_path)


if __name__ == "__main__":
    main()
