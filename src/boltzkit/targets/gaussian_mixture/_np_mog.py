import numpy as np
from scipy.special import logsumexp
from boltzkit.targets.base import NumpyEval


class NumpyMoG(NumpyEval):
    def __init__(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        logits: np.ndarray,
    ):
        self.means = means  # (K, D)
        self.scales = scales  # (K, D)
        self.logits = logits  # (K,)

        self.n_components, self.dim = means.shape

        # Precompute useful terms
        self.log_weights = logits - logsumexp(logits)  # normalized log π_k
        self.vars = scales**2  # (K, D)
        self.log_det = np.sum(np.log(self.vars), axis=1)  # (K,)

        self._log_norm_const = -0.5 * (
            self.dim * np.log(2 * np.pi) + self.log_det
        )  # (K,)

    def _component_log_prob(self, x):
        """
        Compute log N(x | μ_k, Σ_k) for all k.
        x: (B, D)
        returns: (B, K)
        """
        diff = x[:, None, :] - self.means[None, :, :]  # (B, K, D)

        mahal = np.sum((diff**2) / self.vars[None, :, :], axis=-1)  # (B, K)

        return self._log_norm_const[None, :] - 0.5 * mahal  # (B, K)

    def get_log_prob(self, x):
        """
        x: (B, D)
        returns: (B,)
        """
        comp_log_prob = self._component_log_prob(x)  # (B, K)

        return logsumexp(
            self.log_weights[None, :] + comp_log_prob,
            axis=1,
        )

    def get_score(self, x):
        """
        ∇_x log p(x)
        returns: (B, D)
        """
        comp_log_prob = self._component_log_prob(x)  # (B, K)

        # responsibilities (posterior probs)
        log_resp = self.log_weights[None, :] + comp_log_prob  # (B, K)
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)  # (B, 1)
        resp = np.exp(log_resp - log_norm)  # (B, K)

        diff = x[:, None, :] - self.means[None, :, :]  # (B, K, D)

        # score per component: -(x - μ_k) / σ_k^2
        comp_score = -diff / self.vars[None, :, :]  # (B, K, D)

        # weighted sum
        score = np.sum(resp[:, :, None] * comp_score, axis=1)  # (B, D)

        return score

    def get_log_prob_and_score(self, x):
        comp_log_prob = self._component_log_prob(x)  # (B, K)

        log_resp = self.log_weights[None, :] + comp_log_prob
        log_prob = logsumexp(log_resp, axis=1)  # (B,)

        resp = np.exp(log_resp - log_prob[:, None])  # (B, K)

        diff = x[:, None, :] - self.means[None, :, :]
        comp_score = -diff / self.vars[None, :, :]

        score = np.sum(resp[:, :, None] * comp_score, axis=1)

        return log_prob, score
