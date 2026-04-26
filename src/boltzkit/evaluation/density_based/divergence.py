import numpy as np
from scipy.special import logsumexp as _logsumexp
from boltzkit.utils.shape_utils import squeeze_last_dim


def get_reverse_logZ(log_weights: np.ndarray) -> float:
    """
    Estimate the log-normalization constant of :math:`\\tilde{p}(x)` using samples from q(x).

    For :math:`x \\sim q(x)`,

    .. math::

        \\log w(x) = \\log \\tilde{p}(x) - \\log q(x), \\quad
        Z_r = \\mathbb{E}_q[w].

    Therefore,

    .. math::

        \\log Z_r = \\log \\mathbb{E}_q[w].

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim q(x)`.

    Returns
    -------
    float
        Estimated :math:`\\log Z_r`.
    """
    logZ = _logsumexp(log_weights) - np.log(log_weights.shape[0])
    return float(logZ)


def get_forward_logZ(log_weights: np.ndarray) -> float:
    """
    Estimate the log-normalization constant of :math:`\\tilde{p}(x)` using samples from p(x).

    For :math:`x \\sim p(x)`,

    .. math::

        \\log w(x) = \\log \\tilde{p}(x) - \\log q(x), \\quad
        Z_f = 1/\\mathbb{E}_p[1/w].

    Therefore,

    .. math::

        \\log Z_f = -\\log \\mathbb{E}_p[1/w].

    Parameters
    ----------
    log_weights : np.ndarray
        Log weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim p(x)`.

    Returns
    -------
    float
        Estimated :math:`\\log Z_f`.
    """
    logZ = -_logsumexp(-log_weights) + np.log(log_weights.shape[0])
    return float(logZ)


def get_kl_divergence_q(log_weights: np.ndarray, logZ: float | None = None):
    """
    Compute the importance-weighted forward KL using samples from q.

    For :math:`x \\sim q(x)`,

    .. math::

        \\log w(x) = \\log p(x) - \\log q(x), \\quad
        \\mathrm{KL}(p \\| q) = \\mathbb{E}_q[w \\log w].

    Parameters
    ----------
    log_weights : np.ndarray
        Unnormalized log importance weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim q(x)`.

    logZ : float | None
        Log normalization constant (or estimate thereof) of :math:`\\tilde{p}`.
        If None, estimated from ``log_weights``.

    Returns
    -------
    float
        Importance-weighted forward KL estimate.
    """
    log_weights = squeeze_last_dim(log_weights)

    if logZ is None:
        logZ = get_reverse_logZ(log_weights)
    log_w = log_weights - logZ

    # KL = E_q[w * log w]
    kl = (np.exp(log_w) * log_w).mean()
    return float(kl)


def get_kl_divergence_p(log_weights: np.ndarray, logZ: float | None = None):
    """
    Compute the forward KL using samples from p.

    For :math:`x \\sim p(x)`,

    .. math::

        \\log w(x) = \\log p(x) - \\log q(x), \\quad
        \\mathrm{KL}(p \\| q) = \\mathbb{E}_p[\\log w].

    Parameters
    ----------
    log_weights : np.ndarray
        Unnormalized log importance weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim p(x)`.

    logZ : float | None
        Log normalization constant (or estimate thereof) of :math:`\\tilde{p}`.
        If None, estimated from ``log_weights``.

    Returns
    -------
    float
        Forward KL estimate.
    """
    log_weights = squeeze_last_dim(log_weights)

    if logZ is None:
        logZ = get_forward_logZ(log_weights)
    log_weights = log_weights - logZ

    kl = np.mean(log_weights)
    return float(kl)


def get_alpha_divergence_q(
    log_weights: np.ndarray, alpha: float, logZ: float | None = None
):
    """
    Estimate the Amari α-divergence using samples from q(x).

    For :math:`x \\sim q(x)`,

    .. math::

        \\log w(x) = \\log p(x) - \\log q(x), \\quad
        D_\\alpha(p \\| q)
        =
        \\frac{\\mathbb{E}_q[w^{\\alpha}] - 1}{\\alpha(\\alpha - 1)}.

    Using the mapping :math:`\\alpha = (1 + \\beta)/2`, this corresponds to the
    Amari α-divergence for :math:`\\beta \\neq -1, 1`. In the limits
    :math:`\\beta \\to -1` and :math:`\\beta \\to 1`
    (equivalently :math:`\\alpha \\to 0` and :math:`\\alpha \\to 1`),
    it recovers :math:`\\mathrm{KL}(q \\| p)` and :math:`\\mathrm{KL}(p \\| q)`,
    respectively.

    Parameters
    ----------
    log_weights : np.ndarray
        Unnormalized log importance weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim q(x)`.

    alpha : float
        The α parameter (:math:`\\alpha \\neq 0, 1`).

    logZ : float | None
        Log normalization constant (or estimate thereof) of :math:`\\tilde{p}`.
        If None, estimated from ``log_weights``.

    Returns
    -------
    float
        Estimate of the α-divergence.
    """

    log_weights = squeeze_last_dim(log_weights)
    logN: float = np.log(log_weights.shape[0])

    if logZ is None:
        logZ = get_reverse_logZ(log_weights)
    log_weights = log_weights - logZ

    log_int = _logsumexp(log_weights * alpha) - logN
    alpha_div = (np.exp(log_int) - 1) / (alpha * (alpha - 1))
    return float(alpha_div)


def get_alpha_divergence_p(
    log_weights: np.ndarray, alpha: float, logZ: float | None = None
):
    """
    Estimate the Amari α-divergence using samples from p(x).

    For :math:`x \\sim p(x)`,

    .. math::

        \\log w(x) = \\log p(x) - \\log q(x), \\quad
        D_\\alpha(p \\| q)
        =
        \\frac{\\mathbb{E}_p[w^{\\alpha - 1}] - 1}{\\alpha(\\alpha - 1)}.

    Using the mapping :math:`\\alpha = (1 + \\beta)/2`, this corresponds to the
    Amari α-divergence for :math:`\\beta \\neq -1, 1`. In the limits
    :math:`\\beta \\to -1` and :math:`\\beta \\to 1`
    (equivalently :math:`\\alpha \\to 0` and :math:`\\alpha \\to 1`),
    it recovers :math:`\\mathrm{KL}(q \\| p)` and :math:`\\mathrm{KL}(p \\| q)`,
    respectively.

    Parameters
    ----------
    log_weights : np.ndarray
        Unnormalized log importance weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim p(x)`.

    alpha : float
        The α parameter (:math:`\\alpha \\neq 0, 1`).

    logZ : float | None
        Log normalization constant (or estimate thereof) of :math:`\\tilde{p}`.
        If None, estimated from ``log_weights``.

    Returns
    -------
    float
        Estimate of the α-divergence.
    """
    log_weights = squeeze_last_dim(log_weights)
    logN: float = np.log(log_weights.shape[0])

    if logZ is None:
        logZ = get_forward_logZ(log_weights)
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
    iw_fwd_kl = get_kl_divergence_q(log_weights_q)

    # ------------------------------------------------------------------
    # Samples x ~ p(x) for forward KL
    # ------------------------------------------------------------------
    target_samples = rng.normal(loc=shift, scale=1, size=N)
    target_log_prob_p = -0.5 * ((target_samples - shift)) ** 2  # unnormalized log p(x)
    model_log_prob_p = -0.5 * target_samples**2 - 0.5 * np.log(2 * np.pi)
    log_weights_p = target_log_prob_p - model_log_prob_p
    fwd_kl = get_kl_divergence_p(log_weights_p)  # automatically normalizes p

    print("Gaussian importance sampling KL example")
    print("--------------------------------------")
    print(f"fwd kl div (x ~ q): {iw_fwd_kl:.6f}")
    print(f"fwd kl div (x ~ p): {fwd_kl:.6f}")

    alpha = 0.999999
    alpha_div_q = get_alpha_divergence_q(log_weights_q, alpha=alpha)
    alpha_div_p = get_alpha_divergence_p(log_weights_p, alpha=alpha)
    print(f"alpha div (x ~ q, {alpha=}): {alpha_div_q:.6f}")
    print(f"alpha div (x ~ p, {alpha=}): {alpha_div_p:.6f}")
