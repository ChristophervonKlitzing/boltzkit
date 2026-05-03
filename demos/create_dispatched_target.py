from boltzkit.targets.base import DispatchedTarget


class DemoDispatchedTarget(DispatchedTarget):
    def can_sample(self):
        raise NotImplementedError

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

    def _create_numpy_eval(self):
        """
        Lazy instantiation of NumPy-based log-prob & score evaluation.
        """

        from boltzkit.targets.base.dispatched_eval.np import NumpyEval

        class CustomNumpyEval(NumpyEval):
            def __init__(self):
                super().__init__()
                """
                Optional constructor to pass arguments that parameterize the target.

                Example: means and covariances in a Gaussian mixture target
                """

            def get_log_prob(self, x):
                return -(x**2).sum(-1)

            def get_score(self, x):
                return -2 * x

            def get_log_prob_and_score(self, x):
                return self.get_log_prob(x), self.get_score(x)

        return CustomNumpyEval()

    def _create_torch_eval(self):
        """
        Lazy initilization of PyTorch-based log-prob & score evaluation.

        Make sure to import any torch-dependent implementations from inside this
        function to not add global dependency on PyTorch.
        """

        from boltzkit.targets.base.dispatched_eval.torch import TorchEval

        class CustomTorchEval(TorchEval):
            def __init__(self):
                """
                Optional constructor to pass arguments that parameterize the target.

                Example: means and covariances in a Gaussian mixture target
                """
                super().__init__()

            def _get_log_prob(self, x):
                return -(x**2).sum(-1)

        return CustomTorchEval()

    def _create_jax_eval(self):
        """
        Lazy initilization of PyTorch-based log-prob & score evaluation.

        Make sure to import any jax-dependent implementations from inside this
        function to not add global dependency on Jax.
        """

        from boltzkit.targets.base.dispatched_eval.jax import JaxEval
        import jax

        def jax_log_prob_single(x: jax.Array):  # x is NOT batched in this case
            return -(x**2)

        return JaxEval.create_from_log_prob_single(jax_log_prob_single)
