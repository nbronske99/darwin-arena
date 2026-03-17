from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseTask(ABC):
    """Abstract interface that every arena task must implement."""

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Flat input dimensionality (e.g. 784 for 28×28 MNIST)."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes for classification tasks."""

    @abstractmethod
    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader over the training split."""

    @abstractmethod
    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader over the validation split."""
