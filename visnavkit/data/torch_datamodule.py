import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from visnavkit.data.mp4_dataset import Mp4WindowDataset


class TorchDataModule(L.LightningDataModule):
    """Plain torch DataLoader counterpart of DaliDataModule (same loader cfg keys + batch keys)."""

    def __init__(
        self,
        *,
        train_loader: DictConfig,
        val_loader: DictConfig,
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        train_drop_last: bool = True,
        val_drop_last: bool = False,
    ):
        super().__init__()

        if not train_loader.get("file_list"):
            raise ValueError("train_loader.file_list is required")
        if not val_loader.get("file_list"):
            raise ValueError("val_loader.file_list is required")

        self.train_cfg = train_loader
        self.val_cfg = val_loader
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.train_drop_last = train_drop_last
        self.val_drop_last = val_drop_last

    def setup(self, stage=None) -> None:
        world_size = int(getattr(self.trainer, "world_size", 1))
        # Same convention as DaliDataModule: per-GPU batch = global_batch / world_size
        # (Lightning injects a DistributedSampler that shards the dataset).
        if self.batch_size % world_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by world_size ({world_size})")
        self._per_gpu_batch_size = self.batch_size // world_size

        if stage in (None, "fit"):
            self.train_dataset = Mp4WindowDataset(**OmegaConf.to_container(self.train_cfg, resolve=True))
            self.val_dataset = Mp4WindowDataset(**OmegaConf.to_container(self.val_cfg, resolve=True))
        elif stage == "validate":
            self.val_dataset = Mp4WindowDataset(**OmegaConf.to_container(self.val_cfg, resolve=True))

    def _dataloader(self, dataset, shuffle, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self._per_gpu_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not initialized. Call setup(stage='fit') first.")
        return self._dataloader(
            self.train_dataset, shuffle=bool(self.train_cfg.get("shuffle", True)), drop_last=self.train_drop_last
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not initialized. Call setup(stage='fit' or 'validate') first.")
        return self._dataloader(self.val_dataset, shuffle=False, drop_last=self.val_drop_last)
