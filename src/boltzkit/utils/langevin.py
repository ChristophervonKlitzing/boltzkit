from typing import Callable, Optional
import numpy as np


def integrate_langevin(
    score_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    stepsize: float,
    n_steps: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
    callback_every: int = 1,
) -> np.ndarray:
    """
    Batched Langevin integrator.

    Args:
        log_prob_grad: Function mapping (batch, dim) -> (batch, dim),
            returning gradient of log-probability (score).
        x0: Initial samples of shape (batch, dim).
        stepsize: Integration stepsize.
        n_steps: Number of steps to run (if None, runs indefinitely).
        callback: Optional function called as callback(x, step).
        callback_every: Call callback every k steps.

    Returns:
        Final samples of shape (batch, dim).
    """
    x = x0.copy()
    step = 0

    while n_steps is None or step < n_steps:
        grad = score_fn(x)
        noise = np.random.randn(*x.shape)

        x = x + stepsize * grad + np.sqrt(2 * stepsize) * noise

        if not np.isfinite(x).all():
            raise FloatingPointError(f"Non-finite or NaN values at step {step}")

        if callback is not None and (step % callback_every == 0):
            callback(x, step)

        step += 1

    return x


def integrate_langevin_middle(
    score_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    stepsize: float,
    n_steps: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
    callback_every: int = 1,
) -> np.ndarray:
    """
    Batched Langevin integrator using the "middle" (Strang splitting) scheme.

    Args:
        score_fn: Function mapping (batch, dim) -> (batch, dim),
            returning gradient of log-probability (score).
        x0: Initial samples of shape (batch, dim).
        stepsize: Integration stepsize.
        n_steps: Number of steps to run (if None, runs indefinitely).
        callback: Optional function called as callback(x, step).
        callback_every: Call callback every k steps.

    Returns:
        Final samples of shape (batch, dim).
    """
    x = x0.copy()
    step = 0

    while n_steps is None or step < n_steps:
        # half drift
        x = x + 0.5 * stepsize * score_fn(x)

        # full diffusion
        noise = np.random.randn(*x.shape)
        x = x + np.sqrt(2 * stepsize) * noise

        # half drift (recompute score at new position)
        x = x + 0.5 * stepsize * score_fn(x)

        if not np.isfinite(x).all():
            raise FloatingPointError(f"Non-finite or NaN values at step {step}")

        if callback is not None and (step % callback_every == 0):
            callback(x, step)

        step += 1

    return x


if __name__ == "__main__":
    from boltzkit.utils.pdf import plot_pdf
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.evaluation.sample_based.torsion_marginals import (
        get_torsion_angles,
        visualize_torsion_marginals_all,
        get_torsion_marginal_hists,
    )

    target = MolecularBoltzmann(
        "datasets/chrklitz99/alanine_dipeptide", openmm_platform="CPU"
    )
    topology = target.get_mdtraj_topology()
    score_fn = lambda x: target.get_score(x) / 6  # increase temperature
    batch_size = 1000
    x0 = np.stack([target.get_position_min_energy()] * batch_size, 0)

    def callback(xi: np.ndarray, i: int):
        torsion_angles = get_torsion_angles(xi, topology)
        torsion_hists = get_torsion_marginal_hists(*torsion_angles)
        torsion_pdf = visualize_torsion_marginals_all(torsion_hists)
        plot_pdf(torsion_pdf, dpi=200, show=True)

    integrate_langevin_middle(
        score_fn=score_fn, x0=x0, stepsize=2e-5, callback=callback, callback_every=50
    )
