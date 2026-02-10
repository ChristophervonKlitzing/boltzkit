import numpy as np


def compute_shannon_entropy(log_probs: np.ndarray) -> float:
    """
    Compute the Shannon entropy of the **normalized** distribution q(x).

    Approximates H[q] = -E_q[log q(x)] using Monte Carlo samples.

    Parameters
    ----------
    log_probs : np.ndarray
        log-probabilities of samples under the proposal q(x).

    Returns
    -------
    float
        Estimated entropy of q(x).
    """
    return float(-log_probs.mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Number of samples
    N = 10_000

    model_samples = rng.normal(loc=0.0, scale=1.0, size=N)

    # log q(x) -- normalized density
    model_log_prob = -0.5 * model_samples**2 - 0.5 * np.log(2.0 * np.pi)

    shannon_entropy = compute_shannon_entropy(model_log_prob)

    print(f"Shannon Entropy: {shannon_entropy:.2f}")
