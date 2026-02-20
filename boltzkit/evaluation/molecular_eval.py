from boltzkit.evaluation.sample_based.torsion_marginals import (
    get_torsion_angles,
    get_torsion_marginal_hists,
    visualize_torsion_marginals_all,
)
from boltzkit.evaluation.sample_based.tica import (
    get_tica_hist,
    get_tica_projections,
    visualize_tica_true_and_pred,
)
from boltzkit.utils.histogram import Histogram1D, Histogram2D

from .eval import CustomEval
import mdtraj as md
import numpy as np


class MolecularEval(CustomEval):
    def __init__(
        self, topology: md.Topology, tica_model=None, plot_as_free_energy=True
    ):
        super().__init__()
        self._topology = topology
        self._tica_model = tica_model
        self._plot_as_free_energy = plot_as_free_energy

    def __call__(
        self,
        samples_true,
        samples_pred,
        **kwargs,
    ):
        metrics = {}

        tica_metrics = self._get_tica_metrics(
            samples_true=samples_true, samples_pred=samples_pred
        )
        metrics.update(tica_metrics)

        torsion_metrics = self._get_torsion_marginal_metrics(
            samples_true=samples_true, samples_pred=samples_pred
        )
        metrics.update(torsion_metrics)

        return metrics

    def _get_tica_metrics(self, samples_true: np.ndarray, samples_pred: np.ndarray):
        # true
        tica_proj_true = get_tica_projections(
            samples_true, self._topology, self._tica_model
        )
        tica_hist_true = get_tica_hist(tica_proj_true)

        # pred
        tica_proj_pred = get_tica_projections(
            samples_pred, self._topology, self._tica_model
        )
        tica_hist_pred = get_tica_hist(tica_proj_pred)

        tica_pdf_buffer = visualize_tica_true_and_pred(
            tica_hist_true=tica_hist_true,
            tica_hist_pred=tica_hist_pred,
            plot_as_free_energy=self._plot_as_free_energy,
        )

        metrics = {
            "tica_hist_true": tica_hist_true,
            "tica_hist_pred": tica_hist_pred,
            "tica_vis": tica_pdf_buffer,
        }
        return metrics

    def _get_torsion_marginal_metrics(
        self, samples_true: np.ndarray, samples_pred: np.ndarray
    ):
        angles_true = get_torsion_angles(samples_true, self._topology)
        angles_pred = get_torsion_angles(samples_pred, self._topology)

        torsion_marginals_true = get_torsion_marginal_hists(*angles_true)
        torsion_marginals_pred = get_torsion_marginal_hists(*angles_pred)

        pdf_buffer = visualize_torsion_marginals_all(
            torsion_marginals_true,
            plot_as_free_energy=self._plot_as_free_energy,
        )

        def flatten_marginals(
            marginals: tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]],
            prefix: str,
        ):
            d: dict[str, Histogram1D | Histogram2D] = {}
            for i, (ram_hist, phi_hist, psi_hist) in enumerate(zip(*marginals)):
                key_part = f"{prefix}_torsion_marginal_{i}"
                d[f"{key_part}_phi_psi"] = ram_hist
                d[f"{key_part}_phi"] = phi_hist
                d[f"{key_part}_psi"] = psi_hist
            return d

        flattened_marginals_true = flatten_marginals(torsion_marginals_true, "true")
        flattened_marginals_pred = flatten_marginals(torsion_marginals_pred, "pred")

        metrics = {"torsion_marginals_vis": pdf_buffer}
        metrics.update(flattened_marginals_true)
        metrics.update(flattened_marginals_pred)

        return metrics


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from . import get_pdfs
    from boltzkit.utils.pdf import plot_pdf

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()

    gt_samples = bm.load_dataset(T=300.0, type="val")

    molecular_eval = MolecularEval(topology, tica_model, plot_as_free_energy=True)
    metrics = molecular_eval(gt_samples, gt_samples)

    print(metrics)

    pdfs = get_pdfs(metrics)
    for pdf in pdfs.values():
        print(pdf)
        plot_pdf(pdf, show=True)
