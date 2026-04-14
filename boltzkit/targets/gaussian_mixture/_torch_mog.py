import numpy as np
import torch
from boltzkit.targets._torch import TorchEval


class TorchMoG(TorchEval):
    """
    PyTorch implementation of a GMM from pre-defined means, scales and logits (unnormalized log component weights).
    """

    def __init__(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        logits: np.ndarray,
    ):
        """
        Initialize a Gaussian Mixture Model (GMM) distribution with diagonal covariance.

        Args:
            means (np.ndarray): Array of component means with shape
                (num_components, dim), where `num_components` is the number
                of Gaussian components and `dim` is the dimensionality of each component.
            scales (np.ndarray): Array of diagonal standard deviations with shape
                (num_components, dim). Each row corresponds to a component's standard deviations.
            logits (np.ndarray): Array of unnormalized log mixture weights with shape (num_components,).
            n_test_set_samples (int, optional): Number of samples to generate for a test set.
                Defaults to 1000.
        """
        super().__init__()

        self.n_components, self.dim = means.shape

        self.cat_probs: torch.Tensor
        self.locs: torch.Tensor
        self.scale_trils: torch.Tensor
        self.register_buffer("cat_probs", torch.from_numpy(logits))
        self.register_buffer("locs", torch.from_numpy(means))
        self.register_buffer("scale_trils", torch.diag_embed(torch.from_numpy(scales)))

        self._scales = scales

        self._dist = None

    def _get_dist(self):
        if self._dist is None:
            mix = torch.distributions.Categorical(self.cat_probs)
            com = torch.distributions.MultivariateNormal(
                self.locs, scale_tril=self.scale_trils, validate_args=False
            )
            self._dist = torch.distributions.MixtureSameFamily(
                mixture_distribution=mix,
                component_distribution=com,
                validate_args=False,
            )
        return self._dist

    @property
    def device(self):
        return self.locs.device

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            raise RuntimeError("Could not ")

        # Reset _dist object to re-create it on the new device
        self._dist = None

    def _get_log_prob(self, x):
        log_prob = self._get_dist().log_prob(x)
        return log_prob
