from boltzkit.targets.base import DispatchedTarget


class DemoDispatchedTarget(DispatchedTarget):
    def _create_numpy_eval(self):
        """
        Lazy instantiation of NumPy-based log-prob & score evaluation.
        """
        from ._np import CustomNumpyEval

        return CustomNumpyEval()

    def _create_torch_eval(self):
        """
        Lazy initilization of PyTorch-based log-prob & score evaluation.

        Make sure to import any torch-dependent implementations from inside this
        function to not add global dependency on PyTorch.
        """

        from ._torch import CustomTorchEval

        return CustomTorchEval()

    def _create_jax_eval(self):
        """
        Lazy initilization of PyTorch-based log-prob & score evaluation.

        Make sure to import any jax-dependent implementations from inside this
        function to not add global dependency on Jax.
        """
        from ._jax import create_custom_jax_eval

        return create_custom_jax_eval()
