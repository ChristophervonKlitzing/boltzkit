from boltzkit.targets.base import NumPyTarget


class DemoNumpyTarget(NumPyTarget):
    """
    Minimal example of a NumPy-based target distribution.
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
        """
        This demo does not provide a dataset.

        In practice, you would either:

        - load from `CachedRepo`
        Use `Dataset.create_from_cached_repo` to automatically load a dataset from a cached repo.
        The `CachedRepo` abstraction allows easy integration of different data sources along with local caching
        of log-probability and score evaluations.

        - generate samples programmatically
        Programmatically generated samples should ideally behave like the prefix of an imaginary infinite-sized array,
        or at least be seedable for reproducibility.

        - raise `NotImplementedError` if unused
        """
        raise NotImplementedError

    def _numpy_log_prob(self, x):
        return -(x**2).sum(axis=-1)

    def _numpy_score(self, x):
        return -2 * x
