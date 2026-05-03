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
