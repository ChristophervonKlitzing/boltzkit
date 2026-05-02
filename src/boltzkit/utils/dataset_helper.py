from typing import Callable

import numpy as np

from boltzkit.utils.cached_repo import CachedRepo
from boltzkit.utils.dataloader import cache_load_sample_derived_data
from boltzkit.utils.dataset import Dataset


def get_dataset_config_from_cached_repo(repo: CachedRepo, type: str, T: float = 1.0):
    datasets: dict[str, dict[str, str]] | None = repo.config.get("datasets", None)
    if datasets is None:
        raise ValueError("Missing datasets config")

    temp_cfg = datasets.get(str(T), None)
    if temp_cfg is None:
        available_temps = list(datasets.keys())
        raise RuntimeError(
            f"Missing dataset: "
            f"Searched for temperature {T}K, but only found {available_temps}."
        )

    dset_cfg: dict[str, str] | str | None = temp_cfg.get(type, None)
    if dset_cfg is None:
        available_keys = list(temp_cfg.keys())
        raise RuntimeError(
            f"Missing dataset type for temperature {T}K. "
            f"Searched for type '{type}', but only found {available_keys}."
        )

    if isinstance(dset_cfg, str):
        dset_cfg = {"samples": dset_cfg}

    return dset_cfg


def create_dataset_from_cached_repo(
    repo: CachedRepo,
    type: str,
    length,
    kB_T: float,
    *,
    include_samples=True,
    include_log_probs=False,
    include_scores=False,
    allow_autogen: bool = True,
    cache_log_probs: bool = True,
    cache_scores: bool = False,
    log_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None,
):
    dset_cfg = get_dataset_config_from_cached_repo(repo, type=type)
    remote_samples_fpath = dset_cfg.get("samples")
    samples_fpath = repo.load_file(remote_samples_fpath)
    samples: np.ndarray = np.load(samples_fpath)

    if samples.shape[0] < length:
        raise ValueError(
            f"Requested more samples ({length}) than available ({samples.shape[0]})"
        )
    samples = samples[:length]

    if include_log_probs:
        log_probs_fpath = repo.try_load_file(dset_cfg.get("log_probs"))
        log_probs = cache_load_sample_derived_data(
            samples,
            data_cache_fpath=log_probs_fpath,
            data_eval_fn=log_prob_fn,
            allow_autogen=allow_autogen,
            cache_data=cache_log_probs,
        )
    else:
        log_probs = None

    if include_scores:
        scores_fpath = repo.try_load_file(dset_cfg.get("scores"))
        scores = cache_load_sample_derived_data(
            samples,
            data_cache_fpath=scores_fpath,
            data_eval_fn=score_fn,
            allow_autogen=allow_autogen,
            cache_data=cache_scores,
        )
    else:
        scores = None

    if not include_samples:
        samples = None

    raise Dataset(kB_T=kB_T, samples=samples, log_probs=log_probs, scores=scores)
