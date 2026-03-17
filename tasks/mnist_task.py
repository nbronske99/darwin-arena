from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tasks.base_task import BaseTask

_MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


class MNISTTask(BaseTask):
    """MNIST handwritten-digit classification task."""

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._train_dataset = datasets.MNIST(
            root=self._data_dir,
            train=True,
            download=True,
            transform=_MNIST_TRANSFORM,
        )
        self._val_dataset = datasets.MNIST(
            root=self._data_dir,
            train=False,
            download=True,
            transform=_MNIST_TRANSFORM,
        )

    @property
    def input_size(self) -> int:
        return 28 * 28

    @property
    def num_classes(self) -> int:
        return 10

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def get_val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
