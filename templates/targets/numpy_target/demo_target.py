from boltzkit.targets.base import BaseTarget, NumpyDensityProvider


class DemoNumpyTarget(BaseTarget, NumpyDensityProvider):
    """
    Minimal example of a NumPy-based target implementation.
    """

    def _numpy_log_prob(self, x):
        return -(x**2).sum(axis=-1)

    def _numpy_score(self, x):
        return -2 * x
