"""Route AE / VAE: losses and gradients, encoder transfer into the goal encoder, the sidecar dataset, a training run."""

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.data.route_dataset import RouteImageDataset
from visnavkit.models.goal.route import RouteImageGoalEncoder
from visnavkit.models.goal.route_ae import RouteAutoencoder
from visnavkit.models.lit_route import LitRouteAutoencoder
from visnavkit.scripts.train import create_trainer

SMALL = dict(channels=(4, 8), latent_dim=6, image_hw=(16, 8))


@pytest.mark.parametrize("variational", [False, True])
def test_autoencoder_reconstructs_and_trains(variational):
    model = RouteAutoencoder(variational=variational, **SMALL).train()
    x = torch.rand(3, 3, 16, 8)
    out = model(x)
    losses = model.loss(x, out)
    assert out["recon"].shape == x.shape and out["z"].shape == (3, 6)
    assert (out["mu"] is None) == (not variational) and (losses["kl"] > 0) == variational
    losses["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    assert torch.equal(model(x)["z"], model(x)["z"])  # the VAE encodes the mean outside training
    with pytest.raises(ValueError, match="route patches"):
        model(torch.rand(1, 3, 8, 8))
    with pytest.raises(ValueError, match="multiples of 4"):
        RouteAutoencoder(channels=(4, 8), image_hw=(6, 8))


def test_trained_encoder_loads_into_the_goal_encoder(tmp_path):
    ae = RouteAutoencoder(**SMALL)
    torch.save({"state_dict": {f"model.{k}": v for k, v in ae.state_dict().items()}}, tmp_path / "route.ckpt")
    goal = RouteImageGoalEncoder(feat_size=8, channels=(4, 8), weights=str(tmp_path / "route.ckpt"))
    assert list(goal.cnn.state_dict()) == list(ae.encoder.state_dict())
    for ours, theirs in zip(goal.cnn.state_dict().values(), ae.encoder.state_dict().values()):
        torch.testing.assert_close(ours, theirs)
    torch.save({"state_dict": {"model.decoder.0.weight": torch.zeros(1)}}, tmp_path / "other.ckpt")
    with pytest.raises(ValueError, match="route autoencoder"):
        RouteImageGoalEncoder(feat_size=8, channels=(4, 8), weights=str(tmp_path / "other.ckpt"))


def _corpus(root, clips=2, frames=6, hw=(16, 8)):
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in range(clips):
        clip = root / f"clip{label}"
        clip.mkdir()
        np.save(clip / "route_images.npy", np.random.randint(0, 255, (frames, *hw, 3), dtype=np.uint8))
        (clip / "video.mp4").touch()
        lines.append(f"clip{label}/video.mp4 {label} 1 {frames}")
    (root / "train.txt").write_text("\n".join(lines) + "\n")
    (root / "val.txt").write_text(lines[0] + "\n")
    return root


def test_route_dataset_indexes_manifest_ranges_and_flips(tmp_path):
    root = _corpus(tmp_path)
    dataset = RouteImageDataset(file_list="train.txt", data_root=root, p_hflip=1.0)
    assert len(dataset) == 2 * 5
    item = dataset[0]["route_image"]
    assert item.shape == (3, 16, 8) and item.dtype == torch.uint8
    raw = torch.from_numpy(np.load(root / "clip0" / "route_images.npy")[1]).permute(2, 0, 1)
    assert torch.equal(item, torch.flip(raw, dims=[-1]))
    (root / "bad.txt").write_text("clip0/video.mp4 0 0 99\n")
    with pytest.raises(ValueError, match="has 6 frames"):
        RouteImageDataset(file_list="bad.txt", data_root=root)


@pytest.mark.parametrize("route", ["ae", "vae"])
def test_train_route_runs_one_step(tmp_path, route):
    root = _corpus(tmp_path / "data")
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(
            config_name="train_route",
            overrides=[
                f"route={route}",
                f"common.data_root={root}",
                "route.channels=[4,8]",
                "route.latent_dim=6",
                "route.image_hw=[16,8]",
                "dataset.batch_size=2",
                "dataset.num_workers=0",
                "logger=null",
                f"log_dir={tmp_path}",
                f"+trainer.kwargs.default_root_dir={tmp_path}",
                "+trainer.kwargs.accelerator=cpu",
                "trainer.kwargs.precision=32",
                "+trainer.kwargs.fast_dev_run=true",
                "+trainer.kwargs.enable_progress_bar=false",
                "+trainer.kwargs.enable_model_summary=false",
            ],
        )
    assert cfg.route.variational == (route == "vae")
    trainer = create_trainer(cfg)
    trainer.fit(LitRouteAutoencoder(cfg), datamodule=instantiate(cfg.dataset))
    metrics = trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"]) and torch.isfinite(metrics["val/loss"])
    assert (metrics["val/kl"] > 0) == (route == "vae")
