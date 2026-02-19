from .eval import CustomEval
import mdtraj as md


class MolecularEval(CustomEval):
    def __init__(self, topology: md.Topology, tica_model=None):
        super().__init__()
        self._topology = topology
        self._tica_model = tica_model

    def __call__(
        self,
        samples_true,
        samples_pred,
        true_samples_target_log_prob,
        pred_samples_target_log_prob,
        true_samples_model_log_prob=None,
        pred_samples_model_log_prob=None,
    ):
        return {}
