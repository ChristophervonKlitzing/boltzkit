# orchestrator for running the sample- and energy-based evaluation

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, asdict
from typing import Any, Literal, Optional, TypeAlias
import warnings
import numpy as np

from boltzkit.evaluation.density_based.divergence import (
    get_kl_divergence_q,
    get_reverse_logZ,
)
from boltzkit.evaluation.density_based.entropy import get_shannon_entropy
from boltzkit.evaluation.density_based.ess import get_reverse_ess
from boltzkit.evaluation.density_based.evidence import get_nll
from boltzkit.evaluation.sample_based.energy_histogram import (
    get_reduced_energy_hist,
    visualize_energy_hist_dual,
)
from boltzkit.utils.histogram import Histogram1D, Histogram2D
from boltzkit.utils.pdf import PdfBuffer, pdf_to_wandb_image

from boltzkit.utils.shape_utils import squeeze_last_dim


ValueType = float | int | PdfBuffer | Histogram1D | Histogram2D | Any


EvalField: TypeAlias = Literal[
    "samples_true",
    "samples_pred",
    "true_samples_target_log_prob",
    "pred_samples_target_log_prob",
    "true_samples_model_log_prob",
    "pred_samples_model_log_prob",
]


@dataclass
class EvalData:
    """Container for all possible evaluation inputs."""

    # This is used internally to provide better error messages
    _restricted_access: bool = False
    _eval_cls: Optional["Evaluation"] = None

    samples_true: Optional[np.ndarray] = None
    samples_pred: Optional[np.ndarray] = None
    true_samples_target_log_prob: Optional[np.ndarray] = None
    pred_samples_target_log_prob: Optional[np.ndarray] = None
    true_samples_model_log_prob: Optional[np.ndarray] = None
    pred_samples_model_log_prob: Optional[np.ndarray] = None

    def fits_requirements(self, requirements: list[EvalField]) -> bool:
        return len(self.get_missing_requirements(requirements)) == 0

    def get_missing_requirements(
        self, requirements: list[EvalField]
    ) -> list[EvalField]:
        return [r for r in requirements if getattr(self, r, None) is None]

    def __post_init__(self):
        populated_fields = self._get_populated_fields()

        # Remove potential single trailing ones in the fields
        for k, v in populated_fields.items():
            if "log_prob" in k:
                setattr(self, k, squeeze_last_dim(v))

        self._check_type(populated_fields)
        self._check_same_batch_size(populated_fields)
        self._check_sample_shapes(populated_fields)

    def _get_populated_fields(self) -> dict[str, np.ndarray]:
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
            if not k.startswith("_")
        }

    def _check_type(self, populated_fields: dict[str, np.ndarray]):
        invalid = [
            f"{k}: {type(v).__name__}"
            for k, v in populated_fields.items()
            if not isinstance(v, np.ndarray)
        ]

        if invalid:
            raise TypeError(
                f"The following fields must be np.ndarray but are not: {invalid}"
            )

    def _check_same_batch_size(self, populated_fields: dict[str, np.ndarray]):
        batch_sizes = {k: v.shape[0] for k, v in populated_fields.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError("... use keys and actual batch-sizes")

    def _check_sample_shapes(self, populated_fields: dict[str, np.ndarray]):
        sample_shapes = {
            k: v.shape for k, v in populated_fields.items() if "log_prob" not in k
        }
        if not all([len(shape) == 2 for shape in sample_shapes.values()]):
            invalid = {k: s for k, s in sample_shapes.items() if len(s) != 2}
            raise ValueError(
                f"All sample arrays must be 2D (batch_size, dim). "
                f"Found invalid shapes: {invalid}"
            )

        cond1 = len(sample_shapes) > 0
        cond2 = len(set([shape[1] for shape in sample_shapes.values()])) != 1
        if cond1 and cond2:
            raise ValueError(
                f"Dimension mismatch: All samples must have the same dimension (index 1). "
                f"Detected dimensions at index 1: { {k: s[1] for k, s in sample_shapes.items()} }"
            )

    def get_required_fields(self, requirements: list[str]):
        populated_fields = self._get_populated_fields()
        required_fields = {k: populated_fields[k] for k in requirements}
        return required_fields

    def copy_required(self, requirements: list[str], eval_cls: type["Evaluation"]):
        required_fields = self.get_required_fields(requirements)
        data = EvalData(**required_fields, _eval_cls=eval_cls)
        data._restricted_access = True
        return data

    def __getattribute__(self, name):
        # Prevent access to attributes not listed in the requirements.
        # This avoids subtle bugs and makes dependencies explicit.

        restricted_access = object.__getattribute__(self, "_restricted_access")

        if restricted_access:
            try:
                value = object.__getattribute__(self, name)
            except AttributeError:
                raise

            if value is None:
                eval_cls: Evaluation = object.__getattribute__(self, "_eval_cls")
                eval_cls_name = eval_cls.__name__
                requirements = eval_cls.requirements

                if name not in requirements:
                    raise AttributeError(
                        f"The attribute '{name}' was requested by '{eval_cls_name}' without being an explicit requirement, which is not allowed in restricted access mode. Consider adding '{name}' to '{eval_cls_name}.requirements'."
                    )
                else:
                    raise AttributeError("Something went fatally wrong")

        # IMPORTANT: delegate to the base implementation
        return super().__getattribute__(name)


class Evaluation(ABC):
    requirements: list[EvalField] = []

    def __init__(self):
        super().__init__()
        # Verify that all requirements are actually valid keys in the dataclass
        valid_fields = {f.name for f in fields(EvalData)}
        if not set(self.requirements).issubset(valid_fields):
            raise TypeError(f"Invalid requirement in {self.__class__.__name__}")

    @abstractmethod
    def _eval(self, data: EvalData) -> dict[str, ValueType]:
        raise NotImplementedError

    def eval(self, data: EvalData, skip_on_missing_data: bool = False):
        missing = data.get_missing_requirements(self.requirements)
        if missing:
            if skip_on_missing_data:
                infill = "this field is" if len(missing) == 1 else "these fields are"
                warnings.warn(
                    f"Evaluation Skipped: '{self.__class__.__name__}' requires [{', '.join(missing)}], "
                    f"but {infill} missing from the provided EvalData.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            else:
                raise ValueError(
                    f"Evaluation failed: {self.__class__.__name__} requires the following "
                    f"fields to be populated, but they are currently None: {', '.join(missing)}."
                )

        # prevent access to attribute that are not explicitely required.
        # This avoids subtle bugs and makes dependencies explicit.
        safe_data = data.copy_required(self.requirements, type(self))
        return self._eval(safe_data)


class EnergyHistEval(Evaluation):
    requirements = ["true_samples_target_log_prob", "pred_samples_target_log_prob"]

    def __init__(
        self,
        include_pdf: bool = True,
        include_pred_histogram: bool = True,
        include_true_histogram: bool = True,
    ):
        super().__init__()
        self.include_pdf = include_pdf
        self.include_pred_histogram = include_pred_histogram
        self.include_true_histogram = include_true_histogram

    def _eval(self, data):
        metrics = {}

        pred_energy_hist = get_reduced_energy_hist(data.pred_samples_target_log_prob)
        true_energy_hist = get_reduced_energy_hist(data.true_samples_target_log_prob)

        if self.include_pred_histogram:
            metrics["energy_hist/pred"] = pred_energy_hist
            metrics["energy_hist/true"] = true_energy_hist

        if self.include_pdf:
            energy_hist_pdf = visualize_energy_hist_dual(
                pred_energy_hist=pred_energy_hist, true_energy_hist=true_energy_hist
            )
            metrics["energy_hist/pdf"] = energy_hist_pdf

        return metrics


class NllEval(Evaluation):
    requirements = ["true_samples_model_log_prob"]

    def _eval(self, data):
        nll = get_nll(data.true_samples_model_log_prob)
        return {"NLL": nll}


class ModelShannonEntropyEval(Evaluation):
    requirements = ["pred_samples_model_log_prob"]

    def _eval(self, data):
        entropy = get_shannon_entropy(data.pred_samples_model_log_prob)
        return {"model_shannon_entropy": entropy}


class ReverseLogWeightsEval(Evaluation):
    requirements = ["pred_samples_target_log_prob", "pred_samples_model_log_prob"]

    def __init__(
        self,
        include_logZ: bool = True,
        include_ess: bool = True,
        include_iw_kl: bool = True,
    ):
        super().__init__()
        self.include_logZ = include_logZ
        self.include_ess = include_ess
        self.include_iw_kl = include_iw_kl

    def _eval(self, data):
        metrics = {}
        rev_log_iw = (
            data.pred_samples_target_log_prob - data.pred_samples_model_log_prob
        )
        rev_logZ = get_reverse_logZ(rev_log_iw)

        if self.include_logZ:
            metrics["rev_logZ"] = rev_logZ

        if self.include_iw_kl:
            metrics["iw_fwd_kl"] = get_kl_divergence_q(rev_log_iw, logZ=rev_logZ)

        if self.include_ess:
            metrics["reverse_ess"] = get_reverse_ess(rev_log_iw)

        return metrics


def _to_list(
    obj: (
        list[Evaluation | tuple[Evaluation, str]] | Evaluation | tuple[Evaluation, str]
    ),
) -> list[tuple[Evaluation, str | None]]:
    if isinstance(obj, Evaluation):
        l = [(obj, None)]
        return l

    if isinstance(obj, tuple):
        l = [obj]
        return l

    l = []
    for o in obj:
        if isinstance(o, Evaluation):
            l.append((o, None))
        elif isinstance(o, tuple):
            l.append(o)
        else:
            raise ValueError(f"Invalid evaluation object of type '{type(o).__name__}'")
    return l


def _prefix_dict(metrics: dict[str, ValueType], prefix: str | None):
    if prefix:  # None or empty string
        prefix = prefix + "/"
    else:
        prefix = ""

    return {(prefix + k): v for k, v in metrics.items()}


def update_dict_with_id(target: dict, new_data: dict, idx: int) -> dict:
    """
    Update `target` with `new_data`.
    If a key already exists in `target`, append `unique_id` to the key.
    """
    for key, value in new_data.items():
        if key in target:
            new_key = f"{key}_eval_idx_{idx}"
            # Ensure uniqueness even if that key also exists
            i = 1
            while new_key in target:
                new_key = f"{key}_eval_idx_{idx}_{i}"
                i += 1
            target[new_key] = value
        else:
            target[key] = value

    return target


def run_eval(
    data: EvalData,
    *,
    evals: list[Evaluation | tuple[Evaluation]] = [],
    skip_on_missing_data: bool = True,
) -> dict[str, ValueType]:
    if len(evals) == 0:
        warnings.warn("Empty evaluation pipeline -> running no evaluations")

    eval_list = _to_list(evals)

    all_metrics = {}

    for i, (eval, prefix) in enumerate(eval_list):
        metrics = eval.eval(data, skip_on_missing_data=skip_on_missing_data)
        if metrics is None:
            continue
        metrics = _prefix_dict(metrics, prefix)

        update_dict_with_id(all_metrics, metrics, i)

    return all_metrics


def make_wandb_compatible(
    data: dict[str, ValueType], dpi: int = 100, update_keys: bool = True
):
    """
    Convert all elements in the dict into wandb-compatible items (e.g., pdf (in the form of a binary buffer) -> wandb.Image).
    This function requires the installation of the pip `wandb` package.
    """

    def transform(k: str, v):
        if isinstance(v, PdfBuffer):
            if update_keys:
                k = k.replace("pdf", "vis")
            v = pdf_to_wandb_image(v, dpi=dpi)

        return k, v

    def is_valid(v):
        return isinstance(v, (float, int, PdfBuffer))

    return dict((transform(k, v) for k, v in data.items() if is_valid(v)))


def get_scalar_metrics(data: dict[str, ValueType]):
    def _is_float_like(v):
        try:
            float(v)
        except:
            return False
        return True

    return {k: float(v) for k, v in data.items() if _is_float_like(v)}


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
    from boltzkit.utils.pdf import plot_pdf

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

    data = EvalData(
        samples_true=samples_true,
        samples_pred=samples_pred,
        true_samples_target_log_prob=true_samples_target_log_prob,
        pred_samples_target_log_prob=pred_samples_target_log_prob,
        true_samples_model_log_prob=true_samples_model_log_prob,
        pred_samples_model_log_prob=pred_samples_model_log_prob,
    )

    pipeline = [EnergyHistEval()]

    # -------------------------
    # Run evaluation
    # -------------------------
    metrics = run_eval(data, evals=pipeline)

    # -------------------------
    # Print results
    # -------------------------
    print("\n=== Evaluation Metrics ===\n")

    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k:30s}: {v:.6f}")
        else:
            print(f"{k:30s}: {v}")

    pdfs = get_pdfs(metrics)
    for pdf in pdfs.values():
        print(pdf)
        plot_pdf(pdf, show=True)

    print("\nDone.")
