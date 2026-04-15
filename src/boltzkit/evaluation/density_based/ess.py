import numpy as np
from scipy.special import logsumexp as _logsumexp

from boltzkit.utils.shape_utils import squeeze_last_dim


def get_reverse_ess(log_weights: np.ndarray) -> float:
    """
    Compute the reverse effective sample size (ESS) from log-weights.

    Reverse ESS is used in importance sampling to measure how well
    samples from a proposal distribution `q(x)` cover a target distribution `p(x)`.

    Formula (self-normalized ESS):
        ESS = (sum_i w_i)^2 / (N * sum_i w_i^2)
    where w_i = exp(log_weights[i]).

    Key points about log_weights:

    - log_weights[i] = log p(x_i) - log q(x_i), where x_i ~ q(x) (proposal samples)
    - Both q(x) & p(x) do **not need to be normalized**
    - Using log-weights allows numerically stable computation, especially when p/q varies widely

    Parameters
    ----------
    log_weights : np.ndarray
        Array of log importance weights (log p(x) - log q(x))
        computed from samples x_i drawn from the proposal q(x).

    Returns
    -------
    ess : float
        Reverse ESS, normalized to the range (0, 1], representing the effective fraction of
        samples contributing to the estimate.
    """
    log_weights = squeeze_last_dim(log_weights)

    logN = np.log(log_weights.shape[0])

    log_sum_w = _logsumexp(log_weights, axis=0)
    log_sum_w2 = _logsumexp(2.0 * log_weights, axis=0)

    log_ess = 2.0 * log_sum_w - logN - log_sum_w2
    return float(np.exp(log_ess))


def get_forward_ess(log_weights: np.ndarray) -> float:
    """
    Compute the forward effective sample size (ESS) from log-weights.

    Forward ESS is symmetric in weights and their inverse,
    and is typically used when samples are drawn from the **target distribution** `p(x)`:

    ESS = 1 / (E_p[w] * E_p[1/w]),
    where w = exp(log_weights).

    Key points about log_weights:

    - log_weights[i] = log p(x_i) - log q(x_i), where x_i ~ p(x) (target samples)
    - Both q(x) & p(x) do **not need to be normalized**
    - Forward ESS detects poor overlap between target and proposal:
        - ESS is close to 1 if q and p align well
        - ESS decreases if q under-samples p

    Parameters
    ----------
    log_weights : np.ndarray
        Array of log importance weights ±(log p(x) - log q(x))
        computed from samples x_i drawn from the target distribution p(x).

    Returns
    -------
    ess : float
        Forward ESS, normalized to (0, 1], representing the effective fraction of
        target samples that are well-represented by the proposal distribution.
    """
    log_weights = squeeze_last_dim(log_weights)

    logN = np.log(log_weights.shape[0])
    log_z_inv = _logsumexp(-log_weights, 0) - logN
    log_z_expectation_p_over_q = _logsumexp(log_weights, 0) - logN
    log_forward_ess = -(log_z_inv + log_z_expectation_p_over_q)
    return float(np.exp(log_forward_ess))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Number of samples
    N = 1_000

    # ------------------------------------------------------------------
    # Samples x ~ q(x): model distribution (normalized)
    # ------------------------------------------------------------------
    # q = N(0, 1)
    model_samples = rng.normal(loc=0.0, scale=1.0, size=N)

    # log q(x) -- normalized density
    model_log_prob = -0.5 * model_samples**2 - 0.5 * np.log(2.0 * np.pi)

    # ------------------------------------------------------------------
    # Target distribution p(x): unnormalized
    # ------------------------------------------------------------------
    # p(x) ∝ exp(-0.5 * (x - 2)^2 / 0.5^2)
    # (intentionally NOT normalized)
    target_log_prob = -0.5 * ((model_samples - 1.0) / 0.5) ** 2

    # ------------------------------------------------------------------
    # Importance log-weights
    # ------------------------------------------------------------------
    log_weights = target_log_prob - model_log_prob

    # ------------------------------------------------------------------
    # Compute ESS
    # ------------------------------------------------------------------
    reverse_ess = get_reverse_ess(log_weights)

    target_samples = rng.normal(loc=1.0, scale=0.5, size=N)
    target_log_prob_fwd = -0.5 * ((target_samples - 1.0) / 0.5) ** 2
    model_log_prob_fwd = -0.5 * target_samples**2 - 0.5 * np.log(2.0 * np.pi)
    fwd_log_weights = target_log_prob_fwd - model_log_prob_fwd
    forward_ess = get_forward_ess(fwd_log_weights)

    print("Gaussian importance sampling example")
    print("-----------------------------------")
    print(f"Reverse ESS           : {reverse_ess*100:.6f}%")
    print(f"Forward ESS           : {forward_ess*100:.6f}%")
