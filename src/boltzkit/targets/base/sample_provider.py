from abc import ABC, abstractmethod

import numpy as np


class SampleProvider(ABC):
    @abstractmethod
    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """
        Generate batched samples.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        seed : int | None, optional
            Random seed used to ensure reproducibility. If None, the
            generator's current random state is used.

        Returns
        -------
        np.ndarray
            An array of generated samples with shape (n_samples, ...),
            depending on the underlying data distribution/model.
        """
        raise NotImplementedError
