"""Route patches straight from the ``route_images.npy`` sidecars, for ``visnavkit-train-route``."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from visnavkit.data.file_list import parse_file_list_frame_ranges
from visnavkit.data.torch_datamodule import TorchDataModule


class RouteImageDataset(Dataset):
    """One ``route_image`` (3, h, w) uint8 per frame inside the manifest's ``[start, end)`` ranges.

    Same ``path label start end`` file_list as the window dataset and the same horizontal flip the
    policy trains with (``p_hflip``); no video is decoded. ``shuffle`` belongs to the loader.
    """

    def __init__(self, file_list, data_root, p_hflip: float = 0.0, shuffle: bool = True):
        self.p_hflip = p_hflip
        self.index: list[tuple[Path, int]] = []
        self._routes: dict[Path, np.ndarray] = {}
        for video, _, start, end in parse_file_list_frame_ranges(file_list, data_root):
            routes = Path(video).parent / "route_images.npy"
            count = self._open(routes).shape[0]
            if end > count:
                raise ValueError(f"{routes} has {count} frames; the manifest asks for [{start}, {end})")
            self.index += [(routes, frame) for frame in range(start, end)]

    def _open(self, routes: Path) -> np.ndarray:
        if routes not in self._routes:
            array = np.load(routes, mmap_mode="r")
            if array.ndim != 4 or array.shape[-1] != 3:
                raise ValueError(f"{routes} must be (N, h, w, 3) uint8, got {array.shape}")
            self._routes[routes] = array
        return self._routes[routes]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        routes, frame = self.index[index]
        image = torch.from_numpy(np.ascontiguousarray(self._open(routes)[frame])).permute(2, 0, 1)
        if self.p_hflip and torch.rand(()) < self.p_hflip:
            image = torch.flip(image, dims=[-1])
        return {"route_image": image}


class RouteDataModule(TorchDataModule):
    dataset_cls = RouteImageDataset
