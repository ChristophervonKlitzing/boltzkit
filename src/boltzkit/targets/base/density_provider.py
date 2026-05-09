from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

import numpy as np

from boltzkit.targets.base.dispatched_eval.jax import JaxEval
from boltzkit.targets.base.dispatched_eval.np import NumpyEval
from boltzkit.targets.base.dispatched_eval.torch import TorchEval
from boltzkit.utils.framework import (
    FrameworkAgnosticFunction,
    GenericArrayType,
    detect_framework,
    make_agnostic,
)

if TYPE_CHECKING:
    import torch
    import jax


class DensityProvider(ABC):
    @abstractmethod
    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Evaluate unnormalized log-density.

        .. math::
            \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Batch of inputs of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Log-density of shape (batch,).
        """
        raise NotImplementedError

    @abstractmethod
    def get_score(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Compute score function.

        .. math::
            s(x) = \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Batch of inputs of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Score vectors of shape (batch, dim).
        """
        raise NotImplementedError

    @abstractmethod
    def get_log_prob_and_score(
        self, x: "GenericArrayType"
    ) -> tuple["GenericArrayType", "GenericArrayType"]:
        """
        Compute log-density and score jointly.

        .. math::
            \\log \\tilde{p}(x), \\quad \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Batch of inputs of shape (batch, dim).

        Returns
        -------
        tuple[GenericArrayType, GenericArrayType]
            Log-density and score.
        """
        raise NotImplementedError


class WrappedDensityProvider(DensityProvider):
    """
    Target distribution that dispatches computations by converting the input array
    to the appropriate backend type, doing the log-prob and score evaluation with that backend,
    and converting the result back to the original input type.
    """

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Compute unnormalized log-density.

        Input is converted to the backend format before evaluation and
        the result is converted back to the input type.

        .. math::
            \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Log-density of shape (batch,).
        """
        return self._get_agnostic_impl().get_value(x)

    def get_score(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Compute score function (gradient of log-density).

        Input is converted to the backend format before evaluation, and
        the result is converted back to the input type.

        .. math::
            \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Score of shape (batch, dim).
        """
        return self._get_agnostic_impl().get_grad(x)

    def get_log_prob_and_score(
        self, x: "GenericArrayType"
    ) -> tuple["GenericArrayType", "GenericArrayType"]:
        """
        Compute log-density and score jointly.

        Input is converted to the backend format before evaluation, and
        both outputs are converted back to the input type.

        .. math::
            \\log \\tilde{p}(x), \\quad \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        tuple[GenericArrayType, GenericArrayType]
            Log-density and score.
        """
        return self._get_agnostic_impl().get_value_and_grad(x)

    @abstractmethod
    def _get_agnostic_impl(self) -> FrameworkAgnosticFunction:
        """
        Return backend-agnostic function wrapper.

        Returns
        -------
        FrameworkAgnosticFunction
        """
        raise NotImplementedError


class NumpyDensityProvider(WrappedDensityProvider):
    """
    Base class for NumPy-based target distributions.

    Subclasses must implement NumPy versions of:
    - log-probability
    - score (gradient of log-probability)

    These implementations are automatically wrapped into a framework-agnostic
    interface via `make_agnostic`.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

        self._agnostic_impl = make_agnostic(
            implementation="numpy",
            value_fn=self._numpy_log_prob,
            grad_fn=self._numpy_score,
            value_and_grad_fn=self._numpy_log_prob_and_score,
        )

    @abstractmethod
    def _numpy_log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute unnormalized log-density in NumPy.

        .. math::
            \\log \\tilde{p}(x)

        Parameters
        ----------
        x : np.ndarray
            Input of shape (batch, dim).

        Returns
        -------
        np.ndarray
            Log-density of shape (batch,).
        """
        raise NotImplementedError

    @abstractmethod
    def _numpy_score(self, x: np.ndarray) -> np.ndarray:
        """
        Compute score function (gradient of log-density) in NumPy.

        .. math::
            \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : np.ndarray
            Input of shape (batch, dim).

        Returns
        -------
        np.ndarray
            Score of shape (batch, dim).
        """
        raise NotImplementedError

    def _numpy_log_prob_and_score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute unnormalized log-density and score jointly in NumPy.

        This method evaluates both the log-probability and its gradient
        with respect to the input. By default, both are computed separately.
        If log-density and score can be computed jointly, this method can be overriden for more efficiency.

        .. math::
            \\log \\tilde{p}(x), \\quad \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (batch, dim).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Log-density of shape (batch,)
            - Score of shape (batch, dim)
        """
        return self._numpy_log_prob(x), self._numpy_score(x)

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class JaxDensityProvider(WrappedDensityProvider):
    def __init__(
        self,
        *,
        jax_log_prob_fn_single: Callable[["jax.Array"], "jax.Array"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agnostic_impl = make_agnostic(
            implementation="jax", value_fn=jax_log_prob_fn_single
        )

    def _get_agnostic_impl(self):
        return self._agnostic_impl


class PytorchDensityProvider(WrappedDensityProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class DispatchedDensityProvider(DensityProvider):
    """
    Target distribution that dispatches log-prob and score computations to the appropriate
    backend (NumPy, JAX, or PyTorch) at runtime.

    The backend is selected automatically based on the input type.

    This class expects backend-specific evaluation objects to be created
    lazily via `_create_numpy_eval`, `_create_torch_eval`, and `_create_jax_eval`.
    """

    def __init__(self, **kwargs):
        """
        Initialize a dispatched target distribution.

        Backend evaluators for NumPy, JAX, and PyTorch are created lazily
        on first use via the corresponding `_create_*_eval` methods.

        """
        super().__init__(**kwargs)

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

    def get_log_prob(self, x: "GenericArrayType") -> "GenericArrayType":
        """
        Compute unnormalized log-density via backend dispatch.

        The implementation is selected based on the input type.

        .. math::
            \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Log-density of shape (batch,).
        """
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

    def get_score(self, x):
        """
        Compute score function via backend dispatch.

        The implementation is selected based on the input type.

        .. math::
            \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        GenericArrayType
            Score of shape (batch, dim).
        """
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

    def get_log_prob_and_score(self, x):
        """
        Compute log-density and score via backend dispatch.

        The implementation is selected based on the input type.

        .. math::
            \\log \\tilde{p}(x), \\quad \\nabla_x \\log \\tilde{p}(x)

        Parameters
        ----------
        x : GenericArrayType
            Input of shape (batch, dim).

        Returns
        -------
        tuple[GenericArrayType, GenericArrayType]
            Log-density and score.
        """
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

    @abstractmethod
    def _create_numpy_eval(self) -> NumpyEval:
        """
        Create NumPy backend evaluator.

        Returns
        -------
        NumpyEval
            Object implementing NumPy-based log-probability and score computation.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_torch_eval(self) -> "TorchEval":
        """
        Create PyTorch backend evaluator.

        Returns
        -------
        TorchEval
            Object implementing PyTorch-based log-probability and score computation.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_jax_eval(self) -> "JaxEval":
        """
        Create JAX backend evaluator.

        Returns
        -------
        JaxEval
            Object implementing JAX-based log-probability and score computation.
        """
        raise NotImplementedError
