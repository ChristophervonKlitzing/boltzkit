from abc import ABC, abstractmethod
import numpy as np
from boltzkit.utils.framework import make_agnostic, FrameworkAgnosticFunction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boltzkit.utils.framework import Array
    import jax
    import torch


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
            implementation="pytorch", value_fn=self._jax_log_prob
        )

    @abstractmethod
    def _pytorch_log_prob(self, x: "torch.Tensor"):
        """
        Batched log-prob implementation in pytorch.
        """
        raise NotImplementedError

    def _get_wrapper(self):
        return self._wrapper
