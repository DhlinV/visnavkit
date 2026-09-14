import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from visnavkit.utils.ema import EMACallback


class _Tiny(L.LightningModule):
    """Records the online weight at every hook the callback interacts with."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 1, bias=False)
        self.after_each_step: list[torch.Tensor] = []
        self.during_validation: list[torch.Tensor] = []
        self.at_train_start: torch.Tensor | None = None

    def on_train_start(self):
        self.at_train_start = self.model.weight.detach().clone()

    def training_step(self, batch, batch_idx):
        return (self.model(batch[0]) - 1.0).pow(2).mean()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.after_each_step.append(self.model.weight.detach().clone())

    def validation_step(self, batch, batch_idx):
        self.during_validation.append(self.model.weight.detach().clone())
        self.log("val/loss", (self.model(batch[0]) - 1.0).pow(2).mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.2)


def _fit(tmp_path, ema, max_epochs=1, ckpt_path=None):
    module = _Tiny()
    data = DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=2)
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_last=True, save_top_k=0), ema],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, data, data, ckpt_path=ckpt_path)
    return module


def test_decay_warms_up_and_is_capped():
    ema = EMACallback(decay=0.9, inv_gamma=1.0, power=2 / 3)
    assert ema.current_decay(0) == 0.0
    assert 0.0 < ema.current_decay(5) < ema.current_decay(50) <= 0.9
    assert ema.current_decay(10**9) == pytest.approx(0.9)


@pytest.mark.parametrize("bad", [dict(decay=1.5), dict(power=0.0), dict(every_n_steps=0)])
def test_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        EMACallback(**bad)


def test_average_tracks_the_weights_and_is_swapped_in_for_validation(tmp_path):
    ema = EMACallback(decay=0.9)
    module = _fit(tmp_path, ema)

    expected = module.after_each_step[0].clone()
    for step, weight in enumerate(module.after_each_step, start=1):
        expected.lerp_(weight, 1.0 - ema.current_decay(step))
    torch.testing.assert_close(ema.shadow["model.weight"], expected)
    assert not torch.allclose(ema.shadow["model.weight"], module.model.weight)

    # validation ran on the averaged weights, and the online ones were put back afterwards
    torch.testing.assert_close(module.during_validation[0], expected)
    torch.testing.assert_close(module.model.weight.detach(), module.after_each_step[-1])
    assert ema._online is None


def test_checkpoint_carries_the_average_and_resumes_the_online_weights(tmp_path):
    ema = EMACallback(decay=0.9)
    module = _fit(tmp_path, ema)
    checkpoint = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)

    torch.testing.assert_close(checkpoint["state_dict"]["model.weight"], ema.shadow["model.weight"])
    torch.testing.assert_close(checkpoint["ema_online_state_dict"]["model.weight"], module.model.weight.detach())

    # Lightning restores the (averaged) state_dict first; the callback then seeds its average
    # from it and puts the online weights back, which is what the optimizer state belongs to.
    restored, restored_ema = _Tiny(), EMACallback(decay=0.9)
    restored.load_state_dict(checkpoint["state_dict"])
    restored_ema.on_load_checkpoint(None, restored, checkpoint)
    torch.testing.assert_close(restored_ema.shadow["model.weight"], ema.shadow["model.weight"])
    torch.testing.assert_close(restored.model.weight.detach(), module.model.weight.detach())

    resumed = _fit(tmp_path, EMACallback(decay=0.9), max_epochs=2, ckpt_path=str(tmp_path / "last.ckpt"))
    torch.testing.assert_close(resumed.at_train_start, module.model.weight.detach())
