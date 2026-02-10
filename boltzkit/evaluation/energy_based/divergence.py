import numpy as np
from scipy.special import logsumexp as _logsumexp
from boltzkit.utils.shape_utils import squeeze_last_dim


def compute_reverse_logZ(log_weights: np.ndarray) -> float:
    """
    Estimate the log-normalization constant of p(x) using samples from q(x).

    This is the "reverse" estimate of log Z, appropriate when
    x ~ q(x) and log_weights = log p(x) - log q(x).

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights of shape (batch,) or (batch, 1),
        i.e., log p(x) - log q(x) for samples x ~ q(x).

    Returns
    -------
    float
        Estimated log-normalization constant log Z.
    """
    logZ = _logsumexp(log_weights) - np.log(log_weights.shape[0])
    return float(logZ)


def compute_forward_logZ(log_weights: np.ndarray) -> float:
    """
    Estimate the log-normalization constant of p(x) using samples from p(x).

    This is the "forward" estimate of log Z, appropriate when
    x ~ p(x) and log_weights = log p(x) - log q(x) (p can be unnormalized).

    Parameters
    ----------
    log_weights : np.ndarray
        Log weights of shape (batch,) or (batch, 1),
        i.e., log p(x) - log q(x) for samples x ~ p(x).

    Returns
    -------
    float
        Estimated log-normalization constant log Z for p.
    """
    logZ = -_logsumexp(-log_weights) + np.log(log_weights.shape[0])
    return float(logZ)


def compute_kl_divergence_q(log_weights: np.ndarray, logZ: float | None = None):
    """
    Compute the importance-weighted forward KL using samples from q:
        KL(p || q) = E_q[ w(x) * log w(x) ],  w(x) = p(x)/q(x)

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights of shape (batch, 1) or (batch,): log p(x) - log q(x), x ~ q(x)
    logZ : float | None
        Log normalization constant of weights, if known.
        If None, logZ is estimated using `log_weights` (reverse estimate)

    Returns
    -------
    float
        Importance-weighted forward KL estimate.
    """
    log_weights = squeeze_last_dim(log_weights)

    if logZ is None:
        logZ = compute_reverse_logZ(log_weights)
    log_w = log_weights - logZ

    # KL = E_q[w * log w]
    kl = (np.exp(log_w) * log_w).mean()
    return float(kl)


def compute_kl_divergence_p(log_weights: np.ndarray, logZ: float | None = None):
    """
    Compute the forward KL using samples from p:
        KL(p || q) = E_p[ log w(x) ],  w(x) = p(x)/q(x)

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights of shape (batch, 1) or (batch,): log p(x) - log q(x), x ~ p(x)
    logZ : float | None
        Log normalization constant of weights, if known.
        If None, logZ is estimated using `log_weights` (forward estimate)

    Returns
    -------
    float
        forward KL estimate.
    """
    log_weights = squeeze_last_dim(log_weights)

    if logZ is None:
        logZ = compute_forward_logZ(log_weights)
    log_weights = log_weights - logZ

    kl = np.mean(log_weights)
    return float(kl)


def compute_alpha_divergence_q(
    log_weights: np.ndarray, alpha: float, logZ: float | None = None
):
    """
    Estimate the α-divergence D_alpha(p || q) using samples from q(x).

    This is an **importance-weighted** estimator:
        D_alpha(p || q) = (E_{x~q}[(p(x)/q(x))^alpha] - 1) / (alpha*(alpha-1))

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights of shape (batch, 1) or (batch,): log p(x) - log q(x), x ~ q(x)
    alpha : float
        The α parameter of the divergence (α ≠ 0, 1).
        In the limits α → 0 and α → 1, the α-divergence recovers
        KL(q || p) and KL(p || q), respectively.

    Returns
    -------
    float
        Estimate of the α-divergence using samples from q(x).
    """
    logN: float = np.log(log_weights.shape[0])

    if logZ is None:
        logZ = compute_reverse_logZ(log_weights)
    log_weights = log_weights - logZ

    log_int = _logsumexp(log_weights * alpha) - logN
    alpha_div = (np.exp(log_int) - 1) / (alpha * (alpha - 1))
    return float(alpha_div)


def compute_alpha_divergence_p(
    log_weights: np.ndarray, alpha: float, logZ: float | None = None
):
    """
    Estimate the α-divergence D_alpha(p || q) using samples from p(x).

    This is an estimator with expectation over p:
        D_alpha(p || q) = (E_{x~p}[(p(x)/q(x))^(alpha-1)] - 1) / (alpha*(alpha-1))

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights of shape (batch, 1) or (batch,): log p(x) - log q(x), x ~ p(x).
        Can be computed from unnormalized p(x) and normalized q(x).
    alpha : float
        The α parameter of the divergence (α ≠ 0, 1).
        In the limits α → 0 and α → 1, the α-divergence recovers
        KL(q || p) and KL(p || q), respectively.

    Returns
    -------
    float
        Estimate of the α-divergence using samples from p(x).
    """

    logN: float = np.log(log_weights.shape[0])

    if logZ is None:
        logZ = compute_forward_logZ(log_weights)
    log_weights = log_weights - logZ

    log_int = _logsumexp(log_weights * (alpha - 1)) - logN
    alpha_div = (np.exp(log_int) - 1) / (alpha * (alpha - 1))
    return float(alpha_div)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100_000
    shift = 0.5  # small shift for high overlap

    # ------------------------------------------------------------------
    # Samples x ~ q(x): proposal (normalized)
    # ------------------------------------------------------------------
    model_samples = rng.normal(loc=0.0, scale=1.0, size=N)
    model_log_prob = -0.5 * model_samples**2 - 0.5 * np.log(2 * np.pi)

    # ------------------------------------------------------------------
    # Target distribution p(x): unnormalized
    # ------------------------------------------------------------------
    target_log_prob_q = -0.5 * ((model_samples - shift)) ** 2  # unnormalized log p(x)

    # ------------------------------------------------------------------
    # Importance log-weights for samples x ~ q(x)
    # ------------------------------------------------------------------
    log_weights_q = target_log_prob_q - model_log_prob
    iw_fwd_kl = compute_kl_divergence_q(log_weights_q)

    # ------------------------------------------------------------------
    # Samples x ~ p(x) for forward KL
    # ------------------------------------------------------------------
    target_samples = rng.normal(loc=shift, scale=1, size=N)
    target_log_prob_p = -0.5 * ((target_samples - shift)) ** 2  # unnormalized log p(x)
    model_log_prob_p = -0.5 * target_samples**2 - 0.5 * np.log(2 * np.pi)
    log_weights_p = target_log_prob_p - model_log_prob_p
    fwd_kl = compute_kl_divergence_p(log_weights_p)  # automatically normalizes p

    print("Gaussian importance sampling KL example")
    print("--------------------------------------")
    print(f"fwd kl div (x ~ q): {iw_fwd_kl:.6f}")
    print(f"fwd kl div (x ~ p): {fwd_kl:.6f}")

    alpha = 0.999999
    alpha_div_q = compute_alpha_divergence_q(log_weights_q, alpha=alpha)
    alpha_div_p = compute_alpha_divergence_p(log_weights_p, alpha=alpha)
    print(f"alpha div (x ~ q, {alpha=}): {alpha_div_q:.6f}")
    print(f"alpha div (x ~ p, {alpha=}): {alpha_div_p:.6f}")
