"""Every corpus bundled under assets/datasets/ must load and run a forward pass."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

pytest.importorskip("torchcodec")

from visnavkit.models.lit_model import build_targets, disable_pretrained_downloads
from visnavkit.scripts.dataset.cache import build_dataset
from visnavkit.scripts.dataset.preprocess import find_clips, preprocess

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "datasets"
SMALL = [
    "model/vision_encoder=resnet18",
    "model.feat_size=8",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.ff_dim=16",
    "model.temporal_encoder.num_heads=2",
    "model.action_decoder.hidden=16",
    "common.seq_length=2",
    "common.downscale_factor=1",
    "plan_len_points=4",
]


def _corpora() -> list[Path]:
    if not ASSETS.is_dir():
        return []
    return sorted(path for path in ASSETS.iterdir() if path.is_dir() and find_clips(path))


@pytest.mark.parametrize("corpus", _corpora() or [None], ids=lambda p: p.name if p else "none-bundled")
def test_bundled_corpus_trains_one_step(corpus, tmp_path):
    if corpus is None:
        pytest.skip(f"No corpora under {ASSETS}; drop one in and this covers it")
    preprocess(corpus, output_dir=tmp_path, val_fraction=0.0)
    (tmp_path / "val.txt").write_text((tmp_path / "train.txt").read_text())
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(
            config_name="dataset_tools",
            overrides=["dataset=torch", f"common.data_root={tmp_path}", *SMALL],
        )
    disable_pretrained_downloads(cfg.model)

    dataset = build_dataset(cfg, "train")
    assert len(dataset) > 0, f"{corpus.name} produced no windows; check the target horizon"
    sample = dataset[0]
    assert sample["vision"].dtype == torch.uint8 and sample["vision"].shape[1] == 3
    assert torch.isfinite(sample["future_poses"]).all()

    batch = {key: value[None] for key, value in sample.items() if torch.is_tensor(value)}
    model = instantiate(cfg.model).train()
    output = model(batch["vision"].float().div(255.0))
    losses, _ = model.get_losses(output, build_targets(batch, model.temporal_encoder.reduction))
    assert torch.isfinite(losses["loss"]), f"{corpus.name} produced a nonfinite loss"
    losses["loss"].backward()
