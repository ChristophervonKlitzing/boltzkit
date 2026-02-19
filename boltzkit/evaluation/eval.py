# orchestrator for running the sample- and energy-based evaluation

from abc import ABC
from collections.abc import Iterable, Mapping
from typing import Any
import numpy as np

from boltzkit.evaluation.density_based.divergence import (
    get_kl_divergence_q,
    get_reverse_logZ,
)
from boltzkit.evaluation.density_based.entropy import get_shannon_entropy
from boltzkit.evaluation.density_based.evidence import get_eubo, get_nll
from boltzkit.evaluation.sample_based.energy_histogram import (
    get_reduced_energy_hist,
    visualize_energy_hist_dual,
)
from boltzkit.utils.histogram import Histogram1D, Histogram2D
from boltzkit.utils.pdf import PdfBuffer, pdf_to_wandb_image

from boltzkit.utils.shape_utils import squeeze_last_dim


ValueType = float | int | PdfBuffer | Histogram1D | Histogram2D | Any


class CustomEval(ABC):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        samples_true: np.ndarray,
        samples_pred: np.ndarray,
        true_samples_target_log_prob: np.ndarray,
        pred_samples_target_log_prob: np.ndarray,
        true_samples_model_log_prob: np.ndarray | None = None,
        pred_samples_model_log_prob: np.ndarray | None = None,
    ) -> dict[str, ValueType]:
        raise NotImplementedError


def _to_list(
    obj: CustomEval | Iterable[CustomEval] | None,
) -> list[tuple[str, CustomEval]]:
    result = None

    if obj is None:
        result = []

    if isinstance(obj, CustomEval):
        result = ["", obj]

    if isinstance(obj, Mapping):
        result = [(str(k), v) for k, v in obj.items()]

    if isinstance(obj, Iterable):
        result = [("", o) for o in obj]

    for _, val in result:
        if not isinstance(val, CustomEval):
            raise TypeError("Iterable must contain only CustomEval objects")

    if result is None:
        raise TypeError("Expected CustomEval, Iterable[CustomEval], or None")

    return result


def _prefix_dict(metrics: dict[str, ValueType], prefix: str):
    return {(k + prefix): v for k, v in metrics.items()}


def eval_sample_based(
    samples_true: np.ndarray,
    samples_pred: np.ndarray,
    true_samples_target_log_prob: np.ndarray,
    pred_samples_target_log_prob: np.ndarray,
):
    pred_energy_hist = get_reduced_energy_hist(pred_samples_target_log_prob)
    true_energy_hist = get_reduced_energy_hist(true_samples_target_log_prob)
    energy_hist_pdf = visualize_energy_hist_dual(
        pred_energy_hist=pred_energy_hist, true_energy_hist=true_energy_hist
    )
    return {
        "pred_energy_hist": pred_energy_hist,
        "true_energy_hist": true_energy_hist,
        "energy_hist_vis": energy_hist_pdf,
    }


def eval_density_based(
    true_samples_target_log_prob: np.ndarray,
    pred_samples_target_log_prob: np.ndarray,
    true_samples_model_log_prob: np.ndarray | None = None,
    pred_samples_model_log_prob: np.ndarray | None = None,
):
    if pred_samples_model_log_prob is not None:
        rev_log_iw = pred_samples_target_log_prob - pred_samples_model_log_prob
    else:
        rev_log_iw = None

    if true_samples_model_log_prob is not None:
        fwd_log_iw = true_samples_target_log_prob - true_samples_model_log_prob
    else:
        fwd_log_iw = None

    metrics = {}

    if rev_log_iw is not None:
        rev_logZ = get_reverse_logZ(rev_log_iw)
        metrics["rev_logZ"] = rev_logZ
        metrics["iw_fwd_kl"] = get_kl_divergence_q(rev_log_iw, logZ=rev_logZ)

    if pred_samples_model_log_prob is not None:
        metrics["model_shannon_entropy"] = get_shannon_entropy(
            pred_samples_model_log_prob
        )

    if fwd_log_iw is not None:
        metrics["EUBO"] = get_eubo(fwd_log_iw)
        metrics["NLL"] = get_nll(true_samples_model_log_prob)

    return metrics


def eval(
    samples_true: np.ndarray,
    samples_pred: np.ndarray,
    true_samples_target_log_prob: np.ndarray,
    pred_samples_target_log_prob: np.ndarray,
    true_samples_model_log_prob: np.ndarray | None = None,
    pred_samples_model_log_prob: np.ndarray | None = None,
    custom_evals: Iterable[CustomEval] | CustomEval | None = None,
) -> dict[str, ValueType]:
    # ========== Check sample shape =========
    assert len(samples_true.shape) == 2
    sample_shape = samples_true.shape
    assert samples_pred.shape == sample_shape

    # === Check if batch-size is the same ===
    batch_size = sample_shape[0]

    true_samples_target_log_prob = squeeze_last_dim(true_samples_target_log_prob)
    assert true_samples_target_log_prob.shape[0] == batch_size

    pred_samples_target_log_prob = squeeze_last_dim(pred_samples_target_log_prob)
    assert pred_samples_target_log_prob.shape[0] == batch_size

    if true_samples_model_log_prob is not None:
        true_samples_model_log_prob = squeeze_last_dim(true_samples_model_log_prob)
        assert true_samples_model_log_prob.shape[0] == batch_size

    if pred_samples_model_log_prob is not None:
        pred_samples_model_log_prob = squeeze_last_dim(pred_samples_model_log_prob)
        assert pred_samples_model_log_prob.shape[0] == batch_size

    # =======================================

    all_metrics = {}

    # Density-based metrics (e.g., reverse ESS, NLL, ...)
    density_based = (
        true_samples_model_log_prob is not None
        or pred_samples_model_log_prob is not None
    )
    if density_based:
        density_based_metrics = eval_density_based(
            true_samples_target_log_prob=true_samples_target_log_prob,
            pred_samples_target_log_prob=pred_samples_target_log_prob,
            true_samples_model_log_prob=true_samples_model_log_prob,
            pred_samples_model_log_prob=pred_samples_model_log_prob,
        )
    else:
        density_based_metrics = {}
    all_metrics.update(density_based_metrics)

    # Sample-based metrics (energy histogram, ...) that are not target specific
    sample_based_general_metrics = eval_sample_based(
        samples_true=samples_true,
        samples_pred=samples_pred,
        true_samples_target_log_prob=true_samples_target_log_prob,
        pred_samples_target_log_prob=pred_samples_target_log_prob,
    )
    all_metrics.update(sample_based_general_metrics)

    # Custom, e.g., target-specific like TICA plots, torsion marginals, inter-atomic distance histograms, ...
    eval_list = _to_list(custom_evals)
    for prefix, eval in eval_list:
        custom_metrics = eval(
            samples_true=samples_true,
            samples_pred=samples_pred,
            true_samples_target_log_prob=true_samples_target_log_prob,
        )
        custom_metrics = _prefix_dict(custom_metrics, prefix)
        all_metrics.update(custom_metrics)

    return all_metrics


# TODO: Replace by
# transform_wandb_compatible # result can be logged via wandb.log(...) (drops the histogram raw data)
# get_histograms # result can be logged separately for custom visualization
# transform_PDFs # result are high quality visualizations for publications


def transform_wandb_compatible(
    data: dict[str, ValueType],
    dpi: int = 50,
):
    """
    Convert all elements in the dict into wandb-compatible items (e.g., pdf (in the form of a binary buffer) -> wandb.Image).
    This function requires the installation of the pip `wandb` package.
    """

    def transform(v):
        if isinstance(v, PdfBuffer):
            v = pdf_to_wandb_image(v, dpi=dpi)

        return v

    return {k: transform(v) for k, v in data.items()}


def get_histograms(
    data: dict[str, ValueType],
) -> dict[str, Histogram1D | Histogram2D]:
    def is_hist(v):
        return isinstance(v, (Histogram1D, Histogram2D))

    return {k: v for k, v in data.items() if is_hist(v)}


def get_pdfs(data: dict[str, ValueType]) -> dict[str, PdfBuffer]:
    def is_pdf_buffer(v):
        return isinstance(v, PdfBuffer)

    return {k: v for k, v in data.items() if is_pdf_buffer(v)}


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # -------------------------
    # Generate dummy samples
    # -------------------------
    batch_size = 100_000
    dim = 2

    # True samples ~ N(0, I)
    samples_true = rng.normal(loc=0.0, scale=1.0, size=(batch_size, dim))

    # Pred samples ~ N(0.5, 1.2 I)
    pred_scale = 1.5
    pred_mean = 0.5
    samples_pred = rng.normal(loc=pred_mean, scale=pred_scale, size=(batch_size, dim))

    # -------------------------
    # Dummy log-probabilities
    # -------------------------
    # Target log-prob (assume true distribution N(0, I))
    true_samples_target_log_prob = -0.5 * np.sum(samples_true**2, axis=1)
    pred_samples_target_log_prob = -0.5 * np.sum(samples_pred**2, axis=1)

    # Model log-prob (assume model N(0.5, 1.2 I))
    true_samples_model_log_prob = -0.5 * np.sum(
        ((samples_true - pred_mean) / pred_scale) ** 2, axis=1
    )
    pred_samples_model_log_prob = -0.5 * np.sum(
        ((samples_pred - pred_mean) / pred_scale) ** 2, axis=1
    )

    # -------------------------
    # Run evaluation
    # -------------------------
    metrics = eval(
        samples_true=samples_true,
        samples_pred=samples_pred,
        true_samples_target_log_prob=true_samples_target_log_prob,
        pred_samples_target_log_prob=pred_samples_target_log_prob,
        true_samples_model_log_prob=true_samples_model_log_prob,
        pred_samples_model_log_prob=pred_samples_model_log_prob,
    )

    # -------------------------
    # Print results
    # -------------------------
    print("\n=== Evaluation Metrics ===\n")

    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k:30s}: {v:.6f}")
        else:
            print(f"{k:30s}: {v}")

    print("\nDone.")
