from abc import ABC


class BaseTarget(ABC):
    """
    Minimal base class for all targets.
    A target represents a distribution-like object over R^dim with a
    potentially unknown normalization constant Z.
    """

    def __init__(
        self, *, dim: int, logZ: float | None = None, kB_T: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)

        self._dim = dim
        self._logZ = logZ
        self._kB_T = kB_T

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def logZ(self) -> float | None:
        return self._logZ

    @property
    def kB_T(self) -> float:
        return self._kB_T
