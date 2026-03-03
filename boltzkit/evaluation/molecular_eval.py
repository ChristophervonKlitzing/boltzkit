from boltzkit.evaluation.sample_based.internal_coordinate_eval import (
    get_bond_angle_hist,
    get_bond_length_hist,
)
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
from boltzkit.utils.histogram import (
    Histogram1D,
    Histogram2D,
    VisualizationMode,
    plot_as_log_density,
    visualize_histogram_1d,
    visualize_histograms,
)
from boltzkit.utils.molecular.marginals import get_bond_angles, get_bond_lengths


from boltzkit.evaluation.eval import Evaluation, EvalData
import mdtraj as md
import numpy as np


class TorsionMarginalEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_true_histograms: bool = True,
        include_pred_histograms: bool = True,
    ):
        super().__init__()
        self._topology = topology
        self.vis_mode = vis_mode

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
                vis_mode=self.vis_mode,
            )

            infill = self.vis_mode.id
            key = f"torsion_marginals_{infill}_pdf"
            metrics[key] = pdf_buffer

        def flatten_marginals(
            marginals: tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]],
            prefix: str,
        ):
            d: dict[str, Histogram1D | Histogram2D] = {}
            for i, (ram_hist, phi_hist, psi_hist) in enumerate(zip(*marginals)):
                key_part = f"{prefix}_torsion_marginal_hist_{i}"
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


class InternalCoordinateEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        z_matrix: list[tuple[int, int, int, int]],
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdfs: bool = True,
        include_true_histograms: bool = False,
        include_pred_histograms: bool = False,
        eval_bond_lengths: bool = True,
        eval_bond_angles: bool = True,
        eval_dihedral_angles: bool = True,
        max_histogram_bond_length: float | None = 2.0,
    ):
        super().__init__()
        assert (
            eval_bond_lengths or eval_bond_angles or eval_dihedral_angles
        ), "At least one of these must be set to True"

        self._topology = topology
        self._z_matrix = z_matrix

        self.vis_mode = vis_mode

        self.include_pdfs = include_pdfs
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

        self.eval_bond_lengths = eval_bond_lengths
        self.eval_bond_angles = eval_bond_angles
        self.eval_dihedral_angles = eval_dihedral_angles

        self.max_histogram_bond_length = max_histogram_bond_length

    def _eval(self, data):
        metrics = {}
        if self.eval_bond_lengths:
            metrics.update(self._eval_bond_lengths(data))

        if self.eval_bond_angles:
            metrics.update(self._eval_bond_angles(data))

        return metrics

    def _eval_bond_lengths(self, data: EvalData):
        metrics = {}

        true = get_bond_lengths(
            data.samples_true,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        pred = get_bond_lengths(
            data.samples_pred,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        assert true.shape == pred.shape
        n_marginals = true.shape[1]

        hists: list[dict[str, Histogram1D]] = []
        for i in range(n_marginals):
            hist_true = get_bond_length_hist(
                true[:, i], max_bond_length=self.max_histogram_bond_length
            )
            hist_pred = get_bond_length_hist(
                pred[:, i], max_bond_length=self.max_histogram_bond_length
            )

            if self.include_true_histograms:
                metrics[f"bond_length_hist_{i}_true"] = hist_true
            if self.include_pred_histograms:
                metrics[f"bond_length_hist_{i}_pred"] = hist_pred

            hists.append({f"true_{i}": hist_true, f"pred_{i}": hist_pred})

        if self.include_pdfs:
            pdf = visualize_histograms(
                hists,
                vis_mode=self.vis_mode,
                progressbar_description="visualize bond lengths",
            )
            metrics[f"bond_length_pdf"] = pdf

        return metrics

    def _eval_bond_angles(self, data: EvalData):
        metrics = {}

        true = get_bond_angles(
            data.samples_true,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        pred = get_bond_angles(
            data.samples_pred,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        assert true.shape == pred.shape
        n_marginals = true.shape[1]

        hists: list[dict[str, Histogram1D]] = []
        for i in range(n_marginals):
            hist_true = get_bond_angle_hist(true[:, i])
            hist_pred = get_bond_angle_hist(pred[:, i])

            if self.include_true_histograms:
                metrics[f"bond_angle_hist_{i}_true"] = hist_true
            if self.include_pred_histograms:
                metrics[f"bond_angle_hist_{i}_pred"] = hist_pred

            hists.append({f"true_{i}": hist_true, f"pred_{i}": hist_pred})

        if self.include_pdfs:
            pdf = visualize_histograms(
                hists,
                vis_mode=self.vis_mode,
                progressbar_description="visualize bond angles",
            )
            metrics[f"bond_angles_pdf"] = pdf

        return metrics


class TicaEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        tica_model,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_true_histogram: bool = True,
        include_pred_histogram: bool = True,
    ):
        super().__init__()
        self._topology = topology
        self._tica_model = tica_model
        self.vis_mode = vis_mode

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
                vis_mode=self.vis_mode,
            )
            metrics["tica_pdf"] = tica_pdf_buffer

        if self.include_pred_histogram:
            metrics["tica_hist_pred"] = tica_hist_pred

        if self.include_true_histogram:
            metrics["tica_hist_true"] = tica_hist_true

        return metrics


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.utils.pdf import plot_pdf
    from boltzkit.evaluation.eval import EvalData
    from boltzkit.evaluation.eval import get_pdfs

    bm = MolecularBoltzmann("datasets/chrklitz99/alanine_tetrapeptide")

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()
    z_matrix = bm.get_z_matrix()

    gt_samples = bm.load_dataset(T=300.0, type="val")[:500]
    gt_samples = gt_samples.reshape(gt_samples.shape[0], -1)
    pred_samples = gt_samples + 0.1 * np.random.randn(*gt_samples.shape)
    eval_data = EvalData(samples_true=gt_samples, samples_pred=pred_samples)

    metrics = {}

    # molecular_eval = TorsionMarginalEval(topology, vis_mode=plot_as_log_density)
    # metrics.update(molecular_eval.eval(eval_data))
    #
    # tica_eval = TicaEval(topology, tica_model, vis_mode=plot_as_log_density)
    # metrics.update(tica_eval.eval(eval_data))

    ic_eval = InternalCoordinateEval(
        topology, z_matrix, vis_mode=plot_as_log_density, eval_bond_lengths=False
    )
    metrics.update(ic_eval.eval(eval_data))

    print(metrics)

    pdfs = get_pdfs(metrics)
    for key, pdf in pdfs.items():
        print(pdf)
        plot_pdf(pdf, show=True, dpi=100, title=key)
