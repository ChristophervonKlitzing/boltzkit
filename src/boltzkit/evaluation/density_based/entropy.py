import numpy as np
from scipy.special import logsumexp as _logsumexp

from boltzkit.utils.shape_utils import squeeze_last_dim


def get_shannon_entropy(log_probs: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a normalized distribution.

    Approximates:

    .. math::

        H[p] = -\\mathbb{E}_{p(x)}[\\log p(x)]

    using Monte Carlo samples.

    Parameters
    ----------
    log_probs : np.ndarray
        Log-probabilities of samples under the normalized distribution.

    Returns
    -------
    float
        Estimated Shannon entropy.
    """
    log_probs = squeeze_last_dim(log_probs)
    return float(-log_probs.mean())


def get_tsallis_entropy(log_probs: np.ndarray, q: float) -> float:
    """
    Compute the Tsallis entropy from log-probabilities.

    The Tsallis entropy of order :math:`q \\neq 1` is:

    .. math::

        H_q[p] = \\frac{1 - \\mathbb{E}_p[p(x)^{q-1}]}{q - 1}

    Using log-probabilities for numerical stability:

    .. math::

        p(x)^{q-1} = \\exp((q - 1) \\log p(x))

    Parameters
    ----------
    log_probs : np.ndarray
        Log-probabilities of samples under a normalized distribution.
    q : float
        Entropy parameter. Must satisfy :math:`q \\neq 1`.

    Returns
    -------
    float
        Estimated Tsallis entropy.

    Raises
    ------
    ValueError
        If ``q == 1.0`` (use Shannon entropy instead).
    """
    if q == 1.0:
        raise ValueError(
            "q=1 corresponds to Shannon entropy; use Shannon entropy instead."
        )
    log_probs = squeeze_last_dim(log_probs)

    log_int_estimate = _logsumexp(log_probs * (q - 1)) - np.log(log_probs.shape[0])
    return float((1 - np.exp(log_int_estimate)) / (q - 1))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Number of samples
    N = 10_000

    model_samples = rng.normal(loc=0.0, scale=1.0, size=N)

    # log q(x) -- normalized density
    model_log_prob = -0.5 * model_samples**2 - 0.5 * np.log(2.0 * np.pi)

    shannon_entropy = get_shannon_entropy(model_log_prob)

    q = 1.001
    tsallis_entropy = get_tsallis_entropy(model_log_prob, q=q)
    print(f"Shannon entropy: {shannon_entropy:.4f}")
    print(f"Tsallis entropy: {tsallis_entropy:.4f} (q={q})")
