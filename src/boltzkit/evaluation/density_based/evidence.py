import numpy as np

from boltzkit.utils.shape_utils import squeeze_last_dim


def get_elbo(log_weights: np.ndarray) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) using samples from q(x).

    For :math:`x \\sim q(x)`,

    .. math::

        \\log w(x) = \\log \\tilde{p}(x) - \\log q(x), \\quad
        \\mathrm{ELBO} = \\mathbb{E}_q[\\log w].

    Parameters
    ----------
    log_weights : np.ndarray
        Log weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim q(x)`.

    Returns
    -------
    float
        Estimate of the ELBO.
    """
    log_weights = squeeze_last_dim(log_weights)
    return float(log_weights.mean())


def get_eubo(log_weights: np.ndarray) -> float:
    """
    Compute the Evidence Upper Bound (EUBO) using samples from p(x).

    For :math:`x \\sim p(x)`,

    .. math::

        \\log w(x) = \\log \\tilde{p}(x) - \\log q(x), \\quad
        \\mathrm{EUBO} = \\mathbb{E}_p[\\log w].

    Parameters
    ----------
    log_weights : np.ndarray
        Log weights, i.e.,
        :math:`\\log \\tilde{p}(x) - \\log q(x)` for samples :math:`x \\sim p(x)`.

    Returns
    -------
    float
        Estimate of the EUBO.
    """
    log_weights = squeeze_last_dim(log_weights)
    return float(log_weights.mean())


def get_nll(model_log_prob: np.ndarray) -> float:
    """
    Compute the negative log-likelihood under q(x) using samples from p(x).

    For :math:`x \\sim p(x)`,

    .. math::

        \\mathrm{NLL} = -\\mathbb{E}_p[\\log q(x)].

    Parameters
    ----------
    model_log_prob : np.ndarray
        Log-probabilities :math:`\\log q(x)` evaluated at samples :math:`x \\sim p(x)`.

    Returns
    -------
    float
        Estimate of the negative log-likelihood.
    """
    model_log_prob = squeeze_last_dim(model_log_prob)
    return float(-model_log_prob.mean())


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
    elbo = get_elbo(log_weights_q)

    # ------------------------------------------------------------------
    # Samples x ~ p(x) for forward KL
    # ------------------------------------------------------------------
    target_samples = rng.normal(loc=shift, scale=1, size=N)
    target_log_prob_p = -0.5 * ((target_samples - shift)) ** 2  # unnormalized log p(x)
    model_log_prob_p = -0.5 * target_samples**2 - 0.5 * np.log(2 * np.pi)
    log_weights_p = target_log_prob_p - model_log_prob_p
    eubo = get_eubo(log_weights_p)  # automatically normalizes p

    print(f"ELBO: {elbo:.4f}")
    print(f"EUBO: {eubo:.4f}")
