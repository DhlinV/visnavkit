import lightning as L
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from omegaconf import DictConfig, OmegaConf

from visnavkit.data.dali_dataset import DaliDataset


class DaliDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_loader: DictConfig,
        val_loader: DictConfig,
        last_batch_policy: str = "drop",
        val_last_batch_policy: str = "partial",
        batch_size: int = 64,
    ):
        super().__init__()

        if not train_loader.get("file_list"):
            raise ValueError("train_loader.file_list is required")
        if not val_loader.get("file_list"):
            raise ValueError("val_loader.file_list is required")

        self.train_cfg = train_loader
        self.val_cfg = val_loader
        self.train_loader = None
        self.val_loader = None
        policies = {"drop": LastBatchPolicy.DROP, "partial": LastBatchPolicy.PARTIAL}
        if last_batch_policy not in policies or val_last_batch_policy not in policies:
            raise ValueError("last_batch_policy and val_last_batch_policy must be 'drop' or 'partial'")
        self.last_batch_policy = policies[last_batch_policy]
        self.val_last_batch_policy = policies[val_last_batch_policy]
        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        world_size = int(getattr(self.trainer, "world_size", 1))
        rank = int(getattr(self.trainer, "global_rank", 0))
        device_id = int(getattr(self.trainer, "local_rank", 0))

        # Divide batch_size by world_size so global steps stay constant with sharded data:
        # each GPU sees 1/world_size of data, so per-GPU batch = global_batch/world_size.
        batch_size = self.batch_size // world_size
        if self.batch_size % world_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by world_size ({world_size})")

        train_kwargs = {
            **OmegaConf.to_container(self.train_cfg, resolve=True),
            "batch_size": batch_size,
            "world_size": world_size,
            "rank": rank,
            "device_id": device_id,
            "last_batch_policy": self.last_batch_policy,
        }
        val_kwargs = {
            **OmegaConf.to_container(self.val_cfg, resolve=True),
            "batch_size": batch_size,
            "world_size": world_size,
            "rank": rank,
            "device_id": device_id,
            "last_batch_policy": self.val_last_batch_policy,
        }

        if stage in (None, "fit"):
            self.train_loader = DaliDataset(**train_kwargs)
            self.val_loader = DaliDataset(**val_kwargs)
        elif stage == "validate":
            self.val_loader = DaliDataset(**val_kwargs)

    def train_dataloader(self):
        if self.train_loader is None:
            raise RuntimeError("train_loader not initialized. Call setup(stage='fit') first.")
        return self.train_loader

    def val_dataloader(self):
        if self.val_loader is None:
            raise RuntimeError("val_loader not initialized. Call setup(stage='fit' or 'validate') first.")
        return self.val_loader
