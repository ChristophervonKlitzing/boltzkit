from boltzkit.evaluation.sample_based.torsion_marginals import (
    get_torsion_angles,
    get_torsion_marginal_hists,
    visualize_torsion_marginals_dual,
)
from boltzkit.evaluation.sample_based.tica import (
    get_tica_hist,
    get_tica_projections,
    visualize_tica_true_and_pred,
)
from boltzkit.utils.histogram import Histogram1D, Histogram2D

from .eval import Evaluation
import mdtraj as md
import numpy as np


class TorsionMarginalEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        plot_as_free_energy: bool = True,
        include_pdf: bool = True,
        include_true_histograms: bool = True,
        include_pred_histograms: bool = True,
    ):
        super().__init__()
        self._topology = topology
        self.plot_as_free_energy = plot_as_free_energy

        self.include_pdf = include_pdf
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

    def _eval(self, data):
        samples_true = self._reshape(data.samples_true)
        samples_pred = self._reshape(data.samples_pred)

        torsion_metrics = self._get_torsion_marginal_metrics(
            samples_true=samples_true, samples_pred=samples_pred
        )
        return torsion_metrics

    @staticmethod
    def _reshape(obj: np.ndarray):
        return obj.reshape(obj.shape[0], -1, 3)

    def _get_torsion_marginal_metrics(
        self, samples_true: np.ndarray, samples_pred: np.ndarray
    ):
        metrics = {}
        angles_true = get_torsion_angles(samples_true, self._topology)
        angles_pred = get_torsion_angles(samples_pred, self._topology)

        torsion_marginals_true = get_torsion_marginal_hists(*angles_true)
        torsion_marginals_pred = get_torsion_marginal_hists(*angles_pred)

        if self.include_pdf:
            pdf_buffer = visualize_torsion_marginals_dual(
                torsion_marginals_true=torsion_marginals_true,
                torsion_marginals_pred=torsion_marginals_pred,
                plot_as_free_energy=self.plot_as_free_energy,
            )

            infill = "free_energy" if self.plot_as_free_energy else "density"
            key = f"torsion_marginals_{infill}_pdf"
            metrics[key] = pdf_buffer

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

        if self.include_true_histograms:
            flattened_marginals_true = flatten_marginals(torsion_marginals_true, "true")
            metrics.update(flattened_marginals_true)

        if self.include_pred_histograms:
            flattened_marginals_pred = flatten_marginals(torsion_marginals_pred, "pred")
            metrics.update(flattened_marginals_pred)

        return metrics


class TicaEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        tica_model,
        plot_as_free_energy: bool = True,
        include_pdf: bool = True,
        include_true_histogram: bool = True,
        include_pred_histogram: bool = True,
    ):
        super().__init__()
        self._topology = topology
        self._tica_model = tica_model
        self.plot_as_free_energy = plot_as_free_energy

        self.include_pdf = include_pdf
        self.include_true_histogram = include_true_histogram
        self.include_pred_histogram = include_pred_histogram

    def _eval(self, data):
        tica_metrics = self._get_tica_metrics(
            samples_true=data.samples_true, samples_pred=data.samples_pred
        )
        return tica_metrics

    def _get_tica_metrics(self, samples_true: np.ndarray, samples_pred: np.ndarray):
        metrics = {}

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

        if self.include_pdf:
            tica_pdf_buffer = visualize_tica_true_and_pred(
                tica_hist_true=tica_hist_true,
                tica_hist_pred=tica_hist_pred,
                plot_as_free_energy=self.plot_as_free_energy,
            )
            metrics["tica_pdf"] = tica_pdf_buffer

        if self.include_pred_histogram:
            metrics["tica_hist_pred"] = tica_hist_pred

        if self.include_true_histogram:
            metrics["tica_hist_true"] = tica_hist_true

        return metrics


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from . import get_pdfs
    from boltzkit.utils.pdf import plot_pdf
    from boltzkit.evaluation.eval import EvalData

    bm = MolecularBoltzmann("datasets/chrklitz99/alanine_tetrapeptide")

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()

    gt_samples = bm.load_dataset(T=300.0, type="val")[:1_000]
    gt_samples = gt_samples.reshape(gt_samples.shape[0], -1)
    pred_samples = gt_samples + 0.1 * np.random.randn(*gt_samples.shape)
    eval_data = EvalData(samples_true=gt_samples, samples_pred=pred_samples)

    molecular_eval = TorsionMarginalEval(topology, plot_as_free_energy=True)
    torsion_metrics = molecular_eval.eval(eval_data)

    tica_eval = TicaEval(topology, tica_model, plot_as_free_energy=True)
    tica_metrics = tica_eval.eval(eval_data)

    metrics = {}
    metrics.update(torsion_metrics)
    metrics.update(tica_metrics)

    print(metrics)

    pdfs = get_pdfs(metrics)
    for pdf in pdfs.values():
        print(pdf)
        plot_pdf(pdf, show=True)
