from abc import ABC, abstractmethod
import numpy as np
from boltzkit.utils.framework import make_agnostic, FrameworkAgnosticFunction
from typing import TYPE_CHECKING, Callable

from boltzkit.utils.framework import create_dispatch

if TYPE_CHECKING:
    from boltzkit.utils.framework import GenericArrayType
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

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Evaluate the potentially unnormalized log density on x.

        :param x: batched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the unnormalized log density at x with shape (batch,).
        :rtype: Array
        """
        raise NotImplementedError

    def get_score(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Evaluate the score of the log-prob.

        :param x: batched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the score at x with shape (batch, dim).
        :rtype: Array
        """
        raise NotImplementedError

    def get_log_prob_and_score(
        self, x: "GenericArrayType"
    ) -> tuple["GenericArrayType", "GenericArrayType"]:
        """
        Evaluate the log-prob and the score of the log-prob.

        :param x: batched sample(s) of shape (batch, dim)
        :type x: Array

        :return: Returns the unnormalized log-prob of shape (batch,) and the score at x with shape (batch, dim).
        :rtype: Array
        """
        raise NotImplementedError

    @abstractmethod
    def can_sample(self) -> bool:
        raise NotImplementedError

    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def get_logZ(self) -> float | None:
        return None


class FrameworkAgnosticTarget(BaseTarget):
    def __init__(self, dim):
        super().__init__(dim)

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        return self._get_agnostic_impl().get_value(x)

    def get_score(self, x: "GenericArrayType") -> "GenericArrayType":
        return self._get_agnostic_impl().get_grad(x)

    def get_log_prob_and_score(
        self, x: "GenericArrayType"
    ) -> tuple["GenericArrayType", "GenericArrayType"]:
        return self._get_agnostic_impl().get_value_and_grad(x)

    @abstractmethod
    def _get_agnostic_impl(self) -> FrameworkAgnosticFunction:
        raise NotImplementedError


class NumPyTarget(FrameworkAgnosticTarget):
    def __init__(self, dim):
        super().__init__(dim)
        self._agnostic_impl = make_agnostic(
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

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class JaxTarget(FrameworkAgnosticTarget):
    def __init__(
        self, dim, jax_log_prob_fn_single: Callable[["jax.Array"], "jax.Array"]
    ):
        super().__init__(dim)
        self._agnostic_impl = make_agnostic(
            implementation="jax", value_fn=jax_log_prob_fn_single
        )

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class PyTorchTarget(FrameworkAgnosticTarget):
    def __init__(self, dim):
        super().__init__(dim)
        self._agnostic_impl = make_agnostic(
            implementation="pytorch", value_fn=self._pytorch_log_prob
        )

    @abstractmethod
    def _pytorch_log_prob(self, x: "torch.Tensor"):
        """
        Batched log-prob implementation in pytorch.
        """
        raise NotImplementedError

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class DispatchedTarget(FrameworkAgnosticTarget):
    def __init__(
        self,
        dim,
        jax_log_prob_fn_single: Callable[["jax.Array"], "jax.Array"] | None = None,
    ):
        """
        vmap_jax: Whether the jax_log_prob implementation is batched or not.
        If vmap_jax is True, the implementation should not be batched as it will be batched automatically
        """
        super().__init__(dim)

        # TODO: Create FrameworkAgnosticFunction object here by calling `make_agnostic_by_lazy_dispatch`
        # TODO: Overwrite _get_agnostic_impl function to do the rest
        # TODO: Define factory methods for np, jax, and torch implementation

        self._dispatched_log_prob_fn = create_dispatch(
            impl_np=self._numpy_log_prob,
            impl_jax=jax_log_prob_fn_single,
            impl_torch=self._torch_log_prob,
            vmap_jax=True,
            use_jit=True,
        )

    @abstractmethod
    def _numpy_log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _create_torch_log_prob_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _create_jax_log_prob_single_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    # deprecated
    @abstractmethod
    def _torch_log_prob(self, x: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        return self._dispatched_log_prob_fn(x)
