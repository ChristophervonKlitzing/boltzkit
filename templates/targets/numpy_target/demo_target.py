from boltzkit.targets.base import NumPyTarget


class DemoNumpyTarget(NumPyTarget):
    """
    Minimal example of a NumPy-based target implementation.
    """

    def can_sample(self):
        return False

    def load_dataset(
        self,
        type,
        length,
        *,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
    ):
        raise NotImplementedError

    def _numpy_log_prob(self, x):
        return -(x**2).sum(axis=-1)

    def _numpy_score(self, x):
        return -2 * x
