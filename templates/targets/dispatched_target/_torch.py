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
