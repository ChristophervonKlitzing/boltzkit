import os

import h5py
import numpy as np
from reform.simu_utils import OMMTReplicas, SimulationHook
import wandb

from boltzkit.evaluation.eval import (
    Evaluation,
    run_eval,
    EvalData,
    make_wandb_compatible,
)
from boltzkit.evaluation.sample_based.torsion_marginals import (
    get_torsion_angles,
    get_torsion_marginal_hists,
)
from boltzkit.targets.boltzmann import MolecularBoltzmann
import mdtraj as md

from boltzkit.utils.histogram import (
    VisualizationMode,
    plot_as_log_density,
    visualize_histogram_2d,
)


class REMDEval(Evaluation):
    requirements = ["samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        vis_mode: VisualizationMode = plot_as_log_density,
    ):
        super().__init__()
        self._topology = topology
        self.vis_mode = vis_mode

    def _eval(self, data):
        samples_true = self._reshape(data.samples_true)
        metrics = {}

        angles_true = get_torsion_angles(samples_true, self._topology)
        torsion_marginals_true, *_ = get_torsion_marginal_hists(*angles_true)

        for i, hist in enumerate(torsion_marginals_true):
            pdf = visualize_histogram_2d(hist)
            metrics[f"torsion_marginal/phi_psi_{i}_pdf"] = pdf

        return metrics

    @staticmethod
    def _reshape(obj: np.ndarray):
        return obj.reshape(obj.shape[0], -1, 3)


class WandbHook(SimulationHook):
    def __init__(
        self,
        args,
        system: MolecularBoltzmann,
        traj_path: str,
        dataset_name: str,
        max_samples: int = 1000,
    ) -> None:
        super().__init__()

        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

        topology = system.get_mdtraj_topology()
        torsion_marginal_eval = REMDEval(topology)

        self._eval_pipeline = [torsion_marginal_eval]

        self._traj_path = traj_path
        self._dset_name = dataset_name
        self._max_samples = max_samples

    def action(self, context: OMMTReplicas) -> None:
        print("Log to wandb...")
        current_step = context.get_state(0).getStepCount()

        if not os.path.exists(self._traj_path):
            return

        with h5py.File(self._traj_path) as f:
            dset = f[self._dset_name]
            traj_length = dset.shape[0]
            samples = np.copy(dset[-self._max_samples :])
            samples = samples[:, 0, :, :]

        result = run_eval(EvalData(samples_true=samples), evals=self._eval_pipeline)
        wandb_metrics = make_wandb_compatible(result, dpi=20)
        wandb_metrics["trajectory_length"] = traj_length
        wandb.log(wandb_metrics, step=current_step)
        print(f"Logged to wandb: {wandb_metrics}")

    def __str__(self) -> str:
        return "Hook for wandb reporting."

    def finish(self):
        wandb.finish()
