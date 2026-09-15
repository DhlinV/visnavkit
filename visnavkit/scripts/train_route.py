"""Train a route-patch autoencoder (route=ae) or VAE (route=vae) on the route_images.npy sidecars."""

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from visnavkit.models.lit_route import LitRouteAutoencoder
from visnavkit.scripts.train import check_git_status, create_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_route")
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.provenance = check_git_status(strict=cfg.get("strict_git", False))
    L.seed_everything(cfg.seed, workers=True)
    trainer = create_trainer(cfg)
    trainer.fit(LitRouteAutoencoder(cfg), datamodule=instantiate(cfg.dataset), ckpt_path=cfg.trainer.resume.ckpt_path)


if __name__ == "__main__":
    main()
