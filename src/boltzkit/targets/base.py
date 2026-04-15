from abc import ABC, abstractmethod
import numpy as np
from boltzkit.targets._np import NumpyEval
from boltzkit.utils.framework import (
    make_agnostic,
    FrameworkAgnosticFunction,
    detect_framework,
)
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from boltzkit.utils.framework import GenericArrayType
    from boltzkit.targets._torch import TorchEval
    from boltzkit.targets._jax import JaxEval
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

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
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
        self, dim: int, jax_log_prob_fn_single: Callable[["jax.Array"], "jax.Array"]
    ):
        super().__init__(dim)
        self._agnostic_impl = make_agnostic(
            implementation="jax", value_fn=jax_log_prob_fn_single
        )

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class PyTorchTarget(FrameworkAgnosticTarget):
    def __init__(self, dim: int):
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


class DispatchedTarget(BaseTarget):
    def __init__(self, dim):
        super().__init__(dim)

        self.__np_eval_cache: NumpyEval | None = None
        self.__torch_eval_cache = None
        self.__jax_eval_cache = None

    @property
    def __np_eval(self):
        if self.__np_eval_cache is None:
            self.__np_eval_cache = self._create_numpy_eval()
        return self.__np_eval_cache

    @property
    def __jax_eval(self):
        if self.__jax_eval_cache is None:
            self.__jax_eval_cache = self._create_jax_eval()
        return self.__jax_eval_cache

    @property
    def __torch_eval(self):
        if self.__torch_eval_cache is None:
            self.__torch_eval_cache = self._create_torch_eval()
        return self.__torch_eval_cache

    def get_log_prob_and_score(self, x):
        framework = detect_framework(x)
        if framework == "numpy":
            return self.__np_eval.get_log_prob_and_score(x)
        elif framework == "pytorch":
            self.__torch_eval.parameters()[0]
            return self.__torch_eval.get_log_prob_and_score(x)
        elif framework == "jax":
            return self.__jax_eval.get_log_prob_and_score(x)
        else:
            raise ValueError(
                f"Framework '{framework}' currently not supported by this class"
            )

    def get_score(self, x):
        framework = detect_framework(x)
        if framework == "numpy":
            return self.__np_eval.get_score(x)
        elif framework == "pytorch":
            return self.__torch_eval.get_score(x)
        elif framework == "jax":
            return self.__jax_eval.get_score(x)
        else:
            raise ValueError(
                f"Framework '{framework}' currently not supported by this class"
            )

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        framework = detect_framework(x)
        if framework == "numpy":
            return self.__np_eval.get_log_prob(x)
        elif framework == "pytorch":
            return self.__torch_eval.get_log_prob(x)
        elif framework == "jax":
            return self.__jax_eval.get_log_prob(x)
        else:
            raise ValueError(
                f"Framework '{framework}' currently not supported by this class"
            )

    @abstractmethod
    def _create_numpy_eval(self) -> NumpyEval: ...

    @abstractmethod
    def _create_torch_eval(self) -> "TorchEval": ...

    @abstractmethod
    def _create_jax_eval(self) -> "JaxEval": ...
