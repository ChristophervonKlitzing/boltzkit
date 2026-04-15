import warnings

from boltzkit.evaluation.sample_based.histogram_comparison import (
    HistogramMetric,
    get_histogram_fwd_kullback_leibler,
    get_histogram_metrics,
)
from boltzkit.evaluation.sample_based.internal_coordinate_eval import (
    get_bond_angle_hist,
    get_bond_length_hist,
    get_dihedral_angle_hist,
)
from boltzkit.evaluation.sample_based.torsion_marginals import (
    get_free_energy_difference,
    get_torsion_angles,
    get_torsion_marginal_hists,
    visualize_torsion_marginals_dual,
)
from boltzkit.evaluation.sample_based.tica import (
    get_tica_hist,
    visualize_tica_true_and_pred,
)
from boltzkit.utils.histogram import (
    Histogram1D,
    Histogram2D,
    VisualizationMode,
    plot_as_density,
    plot_as_log_density,
    visualize_histograms,
)
from boltzkit.utils.molecular.marginals import (
    get_bond_angles,
    get_bond_lengths,
    get_dihedral_angles,
)


from boltzkit.evaluation.eval import Evaluation, EvalData
import mdtraj as md
import numpy as np
import deeptime as dt

from boltzkit.utils.molecular.tica import TicaModelWithLengthScale


class TorsionMarginalEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_true_histograms: bool = True,
        include_pred_histograms: bool = True,
        histogram_metrics: tuple[HistogramMetric] = [
            get_histogram_fwd_kullback_leibler
        ],
    ):
        super().__init__()
        self._topology = topology
        self.vis_mode = vis_mode

        self.include_pdf = include_pdf
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

        self.histogram_metrics = histogram_metrics

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

        # Compute free energy difference on phi angles
        # (neg log weight ratio between high and low energy region)
        phis_true = angles_true[0]
        phis_pred = angles_pred[0]
        free_energy_difference_true = get_free_energy_difference(phis_true)
        free_energy_difference_pred = get_free_energy_difference(phis_pred)
        metrics["torsion_marginals/free_energy_difference_true"] = (
            free_energy_difference_true
        )
        metrics["torsion_marginals/free_energy_difference_pred"] = (
            free_energy_difference_pred
        )

        # Compute histogram metrics
        if len(self.histogram_metrics) > 0:
            # Only get 2D histogram metrics fow now
            marginals_true_2d = torsion_marginals_true[0]
            marginals_pred_2d = torsion_marginals_pred[0]

            metrics.update(
                get_histogram_metrics(
                    self.histogram_metrics,
                    marginals_true_2d,
                    marginals_pred_2d,
                    group="torsion_marginals",
                    h_type="phi_psi",
                ),
            )

        if self.include_pdf:
            pdf_buffer = visualize_torsion_marginals_dual(
                torsion_marginals_true=torsion_marginals_true,
                torsion_marginals_pred=torsion_marginals_pred,
                vis_mode=self.vis_mode,
            )

            infill = self.vis_mode.id
            key = f"torsion_marginals/{infill}_pdf"
            metrics[key] = pdf_buffer

        def flatten_marginals(
            marginals: tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]],
            prefix: str,
        ):
            d: dict[str, Histogram1D | Histogram2D] = {}
            for i, (ram_hist, phi_hist, psi_hist) in enumerate(zip(*marginals)):
                key_part = f"torsion_marginals/{prefix}_hist_{i}"
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


class BondLengthEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        z_matrix: list[tuple[int, int, int, int]],
        vis_mode: VisualizationMode = plot_as_density,
        include_pdfs: bool = True,
        include_true_histograms: bool = False,
        include_pred_histograms: bool = False,
        max_histogram_bond_length: float | None = None,
    ):
        super().__init__()

        self._topology = topology
        self._z_matrix = z_matrix

        self.vis_mode = vis_mode

        self.include_pdfs = include_pdfs
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

        self.max_histogram_bond_length = max_histogram_bond_length

    def _eval(self, data: EvalData):
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
        print(true.shape, pred.shape, true.mean(), pred.mean(), true.std(), pred.std())
        # assert true.shape == pred.shape
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
                metrics[f"bond_lengths/hist_{i}_true"] = hist_true
            if self.include_pred_histograms:
                metrics[f"bond_lengths/hist_{i}_pred"] = hist_pred

            hists.append({f"true_{i}": hist_true, f"pred_{i}": hist_pred})

        if self.include_pdfs:
            pdf = visualize_histograms(
                hists,
                vis_mode=self.vis_mode,
                progressbar_description="visualize bond lengths",
            )
            metrics[f"bond_lengths/pdf"] = pdf

        return metrics


class BondAngleEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        z_matrix: list[tuple[int, int, int, int]],
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdfs: bool = True,
        include_true_histograms: bool = False,
        include_pred_histograms: bool = False,
    ):
        super().__init__()

        self._topology = topology
        self._z_matrix = z_matrix

        self.vis_mode = vis_mode

        self.include_pdfs = include_pdfs
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

    def _eval(self, data: EvalData):
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
        # assert true.shape == pred.shape
        n_marginals = true.shape[1]

        hists: list[dict[str, Histogram1D]] = []
        for i in range(n_marginals):
            hist_true = get_bond_angle_hist(true[:, i])
            hist_pred = get_bond_angle_hist(pred[:, i])

            if self.include_true_histograms:
                metrics[f"bond_angles/hist_{i}_true"] = hist_true
            if self.include_pred_histograms:
                metrics[f"bond_angles/hist_{i}_pred"] = hist_pred

            hists.append({f"true_{i}": hist_true, f"pred_{i}": hist_pred})

        if self.include_pdfs:
            pdf = visualize_histograms(
                hists,
                vis_mode=self.vis_mode,
                progressbar_description="visualize bond angles",
            )
            metrics[f"bond_angles/pdf"] = pdf

        return metrics


class DihedralAngleEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        z_matrix: list[tuple[int, int, int, int]],
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdfs: bool = True,
        include_true_histograms: bool = False,
        include_pred_histograms: bool = False,
        histogram_metrics: tuple[HistogramMetric] = [
            get_histogram_fwd_kullback_leibler
        ],
        include_individual_hist_metrics: bool = False,
        include_aggregated_hist_metrics: bool = True,
    ):
        super().__init__()

        self._topology = topology
        self._z_matrix = z_matrix

        self.vis_mode = vis_mode

        self.include_pdfs = include_pdfs
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms

        self.histogram_metrics = histogram_metrics
        self.include_individual_hist_metrics = include_individual_hist_metrics
        self.include_aggregated_hist_metrics = include_aggregated_hist_metrics

    def _eval(self, data: EvalData):
        metrics = {}

        true = get_dihedral_angles(
            data.samples_true,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        pred = get_dihedral_angles(
            data.samples_pred,
            topology=self._topology,
            z_matrix=self._z_matrix,
        )
        # assert true.shape == pred.shape
        n_marginals = true.shape[1]

        true_hists = []
        pred_hists = []
        hists: list[dict[str, Histogram1D]] = []
        for i in range(n_marginals):
            hist_true = get_dihedral_angle_hist(true[:, i])
            hist_pred = get_dihedral_angle_hist(pred[:, i])

            if self.include_true_histograms:
                metrics[f"dihedral_angles/hist_{i}_true"] = hist_true
            if self.include_pred_histograms:
                metrics[f"dihedral_angles/hist_{i}_pred"] = hist_pred

            # Save for visualization
            hists.append({f"true_{i}": hist_true, f"pred_{i}": hist_pred})

            # Save for histogram metrics
            true_hists.append(hist_true)
            pred_hists.append(hist_pred)

        # Visualize
        if self.include_pdfs:
            pdf = visualize_histograms(
                hists,
                vis_mode=self.vis_mode,
                progressbar_description="visualize dihedral angles",
            )
            metrics[f"dihedral_angles/pdf"] = pdf

        # Compute metrics on histograms
        if len(self.histogram_metrics) > 0:
            metrics.update(
                get_histogram_metrics(
                    self.histogram_metrics,
                    true_hists,
                    pred_hists,
                    group="dihedral_angles",
                    include_individual=self.include_individual_hist_metrics,
                    include_aggregated=self.include_aggregated_hist_metrics,
                ),
            )

        return metrics


class TicaEval(Evaluation):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        tica_model: TicaModelWithLengthScale | dt.decomposition.TransferOperatorModel,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_true_histogram: bool = True,
        include_pred_histogram: bool = True,
        plot_pred_in_true_range: bool = True,
    ):
        super().__init__()
        self._topology = topology

        if isinstance(tica_model, dt.decomposition.TransferOperatorModel):
            warnings.warn(
                "TICA model provided without an explicit length scale. "
                "The code will assume that the TICA model and the sample data "
                f"use the same length units (e.g., nm or Å). Use the wrapper '{TicaModelWithLengthScale.__name__}' instead."
            )
            tica_model = TicaModelWithLengthScale(tica_model)

        self._tica_model = tica_model
        self.vis_mode = vis_mode

        self.include_pdf = include_pdf
        self.include_true_histogram = include_true_histogram
        self.include_pred_histogram = include_pred_histogram
        self.plot_pred_in_true_range = plot_pred_in_true_range

    def _eval(self, data):
        tica_metrics = self._get_tica_metrics(
            samples_true=data.samples_true, samples_pred=data.samples_pred
        )
        return tica_metrics

    def _get_tica_metrics(self, samples_true: np.ndarray, samples_pred: np.ndarray):
        metrics = {}

        # true
        tica_proj_true = self._tica_model.project_from_cartesian(
            samples_true, self._topology
        )
        tica_hist_true = get_tica_hist(tica_proj_true)

        # pred
        tica_proj_pred = self._tica_model.project_from_cartesian(
            samples_pred, self._topology
        )
        tica_hist_pred = get_tica_hist(tica_proj_pred)

        if self.include_pdf:
            tica_pdf_buffer = visualize_tica_true_and_pred(
                tica_hist_true=tica_hist_true,
                tica_hist_pred=tica_hist_pred,
                vis_mode=self.vis_mode,
                clip_pred_to_true_range=self.plot_pred_in_true_range,
            )
            metrics["tica/pdf"] = tica_pdf_buffer

        if self.include_pred_histogram:
            metrics["tica/hist_pred"] = tica_hist_pred

        if self.include_true_histogram:
            metrics["tica/hist_true"] = tica_hist_true

        return metrics


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.utils.pdf import plot_pdf
    from boltzkit.evaluation.eval import EvalData, EnergyHistEval, run_eval
    from boltzkit.evaluation.molecular_eval import BondLengthEval
    from boltzkit.evaluation.eval import get_pdfs, make_wandb_compatible
    from boltzkit.evaluation.sample_based.histogram_comparison import (
        get_histogram_jensen_shannon_divergence,
    )

    from boltzkit.utils.histogram import plot_as_free_energy

    bm = MolecularBoltzmann(
        "datasets/chrklitz99/alanine_hexapeptide", length_unit="nanometer", n_workers=2
    )

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()
    z_matrix = bm.get_z_matrix()

    val_dataset = bm.load_dataset(
        T=300.0, type="val", length=10_000, include_energies=True
    )
    val_samples = val_dataset.get_samples()
    print("loaded dataset size:", val_samples.shape)

    true_samples = val_samples
    true_samples = true_samples.reshape(true_samples.shape[0], -1)

    test_dataset = bm.load_dataset(
        T=300.0, type="test", length=10_000, include_energies=True
    )
    pred_samples = test_dataset.get_samples()
    pred_samples = pred_samples.reshape(pred_samples.shape[0], -1)
    # pred_samples: np.ndarray = np.load("300K_val.npy", mmap_mode="r")[:10_000] / 10.0
    # pred_samples = pred_samples.reshape(pred_samples.shape[0], -1)

    true_samples_log_prob = val_dataset.get_log_probs()
    pred_samples_log_probs = bm.get_log_prob(pred_samples)

    eval_data = EvalData(
        samples_true=true_samples,
        samples_pred=pred_samples,
        true_samples_target_log_prob=true_samples_log_prob,
        pred_samples_target_log_prob=pred_samples_log_probs,
    )

    mol_eval_pipeline = []

    torsion_eval = TorsionMarginalEval(
        topology,
        vis_mode=plot_as_log_density,
        histogram_metrics=[
            get_histogram_fwd_kullback_leibler,
            get_histogram_jensen_shannon_divergence,
        ],
    )
    mol_eval_pipeline.append(torsion_eval)

    tica_eval = TicaEval(topology, tica_model, vis_mode=plot_as_log_density)
    mol_eval_pipeline.append(tica_eval)

    energy_hist_eval = EnergyHistEval()
    mol_eval_pipeline.append(energy_hist_eval)

    # bond_length_eval = BondLengthEval(topology, z_matrix)
    # mol_eval_pipeline.append(bond_length_eval)

    # bond_angle_eval = BondAngleEval(topology, z_matrix)
    # mol_eval_pipeline.append(bond_angle_eval)

    # dihedral_angle_eval = DihedralAngleEval(topology, z_matrix)
    # mol_eval_pipeline.append(dihedral_angle_eval)

    metrics = run_eval(eval_data, evals=mol_eval_pipeline)

    metrics_wandb = make_wandb_compatible(metrics)
    print(metrics_wandb)

    pdfs = get_pdfs(metrics)
    for key, pdf in pdfs.items():
        print(pdf)
        plot_pdf(pdf, show=True, dpi=100, title=key)
