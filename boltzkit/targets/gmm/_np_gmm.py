import numpy as np


class NumPyGMM:
    def __init__(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        logits: np.ndarray,
    ):
        pass

    def log_prob(self, x: np.ndarray) -> np.ndarray: ...

    def sample(self, n_samples: int) -> np.ndarray: ...
