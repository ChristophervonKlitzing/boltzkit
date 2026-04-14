from abc import ABC, abstractmethod

import numpy as np


class NumpyEval(ABC):
    @abstractmethod
    def get_log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_log_prob_and_score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
