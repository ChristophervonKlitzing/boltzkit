from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal
import numpy as np
from boltzkit.utils.dataloader import DatasetLoader
from boltzkit.utils.dataset import Dataset


@dataclass
class RawDataset:
    samples: np.ndarray | None
    log_probs: np.ndarray | None
    scores: np.ndarray | None


class DatasetProvider(ABC):
    @abstractmethod
    def load_dataset(
        self,
        type: Literal["train", "val", "test"],
        length: int,
        include_samples: bool = True,
        include_log_probs: bool = False,
        include_scores: bool = False,
        **kwargs,
    ):
        raise NotImplementedError


class ExternalDatasetProvider(DatasetProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset_loader = None

    def set_dataset_loader(self, dataset_loader: DatasetLoader):
        self._dataset_loader = dataset_loader

    def load_dataset(
        self,
        type,
        length,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
        **kwargs,
    ):
        return self._dataset_loader.load_dataset(
            type=type,
            length=length,
            include_samples=include_samples,
            include_log_probs=include_log_probs,
            include_scores=include_scores,
            **kwargs,
        )


class ProceduralDatasetProvider(DatasetProvider):
    def __init__(self, *, procedural_seed: int = 0, **kwargs):
        super().__init__(**kwargs)

        # Additional random mangling to prevent accidental collisions
        # between train and test dataset when changing the seed
        self.__seed = np.random.default_rng(procedural_seed).integers(
            0, 2**32, size=1
        )[0]

    def load_dataset(
        self,
        type,
        length,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
        **kwargs,
    ):
        # all three splits will always have different root-seeds (guaranteed by design)
        split_ids = {"train": 0, "val": 1, "test": 2}
        root_seed = self.__seed + split_ids[type]

        dataset = self._generate_procedural_prefix(
            seed=root_seed,
            length=length,
            include_samples=include_samples,
            include_log_probs=include_log_probs,
            include_scores=include_scores,
            **kwargs,
        )

        if include_samples and dataset.get_samples() is None:
            raise ValueError("...")

        if include_log_probs and dataset.get_log_probs() is None:
            raise ValueError("...")

        if include_scores and dataset.get_scores() is None:
            raise ValueError("...")

        return dataset

    @abstractmethod
    def _generate_procedural_prefix(
        self,
        seed: int,
        length: int,
        include_samples: bool,
        include_log_probs: bool,
        include_scores: bool,
        **kwargs,
    ) -> Dataset:
        raise NotImplementedError("Procedural generation of dataset is not implemented")
