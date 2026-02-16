from abc import ABC, abstractmethod
import numpy as np
from boltzkit.utils.framework import make_agnostic, FrameworkAgnosticFunction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boltzkit.utils.framework import Array
    import jax
    import torch


# TODO: Also add:
"""
can_sample and sample (should use an rng seeded in the constructor)
logZ -> None | float

!!! Not sure about whether the following should even be part of the class:

__init__ should take arguments to load the datasets: subset of dict {"validation": ..., "test": ..., "train": ...}
get/load_raw_dataset(split="validation") -> None | Dataset # or "test" or "train"
Dataset is either only holding samples or also evals (only if already available in the dataset!).

get_evaluation_data(n_samples: int, split="validation", fallback="sample", include_log_prob: bool = True, include_score: bool = True) -> Data
split: "validation" or "test"
fallback: "error" (default), "sample", "train", "valdiation", "test", "sample_warn" 
Data object should provide sample function and other utility and be easily convertible into torch dataloader etc,
but it should also provide functionality to sample like from a torch dataloader.
get_evaluation_data should fail if the dataset is not loaded and sample is not an option.
"""


class BaseTarget(ABC):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def get_log_prob(self, x: "Array") -> "Array":
        """
        Evaluate the potentially unnormalized log density on x.

        :param x: batched or unbatched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the unnormalized log density at x with shape (batch,).
        :rtype: Array
        """
        return self._get_wrapper().get_value(x)

    def get_score(self, x: "Array") -> "Array":
        """
        Evaluate the score of the log-prob.

        :param x: batched or unbatched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the score at x with shape (batch, dim).
        :rtype: Array
        """
        return self._get_wrapper().get_grad(x)

    def get_log_prob_and_score(self, x: "Array") -> tuple["Array", "Array"]:
        """
        Evaluate the log-prob and the score of the log-prob.

        :param x: batched or unbatched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the unnormalized log-prob of shape (batch,) and the score at x with shape (batch, dim).
        :rtype: Array
        """
        return self._get_wrapper().get_value_and_grad(x)

    @abstractmethod
    def _get_wrapper(self) -> FrameworkAgnosticFunction:
        raise NotImplementedError

    @abstractmethod
    def can_sample(self) -> bool:
        raise NotImplementedError

    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def get_logZ(self) -> float | None:
        return None


class NumPyTarget(BaseTarget):
    def __init__(self, dim):
        super().__init__(dim)
        self._wrapper = make_agnostic(
            implementation="numpy",
            value_fn=self._numpy_log_prob,
            grad_fn=self._numpy_score,
            value_and_grad_fn=self._numpy_log_prob_and_score,
        )

    @abstractmethod
    def _numpy_log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Batched log-prob implementation in numpy
        """
        raise NotImplementedError

    @abstractmethod
    def _numpy_score(self, x: np.ndarray) -> np.ndarray:
        """
        Batched score implementation in numpy
        """
        raise NotImplementedError

    def _numpy_log_prob_and_score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._numpy_log_prob(x), self._numpy_score(x)

    def _get_wrapper(self):
        return self._wrapper


class JaxTarget(BaseTarget):
    def __init__(self, dim):
        super().__init__(dim)
        self._wrapper = make_agnostic(implementation="jax", value_fn=self._jax_log_prob)

    @abstractmethod
    def _jax_log_prob(self, x: "jax.Array"):
        """
        Unbatched log-prob implementation in jax.
        """
        raise NotImplementedError

    def _get_wrapper(self):
        return self._wrapper


class PyTorchTarget(BaseTarget):
    def __init__(self, dim):
        super().__init__(dim)
        self._wrapper = make_agnostic(
            implementation="pytorch", value_fn=self._pytorch_log_prob
        )

    @abstractmethod
    def _pytorch_log_prob(self, x: "torch.Tensor"):
        """
        Batched log-prob implementation in pytorch.
        """
        raise NotImplementedError

    def _get_wrapper(self):
        return self._wrapper
