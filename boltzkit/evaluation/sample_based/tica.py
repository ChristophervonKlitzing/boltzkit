import numpy as np
import mdtraj as md
import deeptime as dt
from .histogram import get_histogram_2d
from .wasserstein import compute_euclidean_wasserstein_1_2
from ._tica_help import _tica_features


def get_tica_projections(
    data: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
) -> np.ndarray:
    data = data.reshape(data.shape[0], -1, 3)
    ticas_list = []
    for i in range(0, data.shape[0], 50000):
        tica_features_batch = _tica_features(
            md.Trajectory(
                xyz=data[i : i + 50000],
                topology=topology,
            )
        )
        ticas_list.append(tica_model.transform(tica_features_batch))
    return np.concatenate(ticas_list, axis=0)


def get_tica_hist(
    tica_proj: np.ndarray,
    **kwargs,
):
    return get_histogram_2d(tica_proj, **kwargs)


def get_tica_wasserstein_1_2(
    tica_projections_true: np.ndarray,
    tica_projections_pred: np.ndarray,
    include_w1=False,
    include_w2=True,
):
    tica_W1, tica_W2 = compute_euclidean_wasserstein_1_2(
        tica_projections_pred,
        tica_projections_true,
        include_w1=include_w1,
        include_w2=include_w2,
    )
    return tica_W1, tica_W2


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.evaluation.sample_based.visualize import visualize_histogram_2d

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()

    gt_samples_path = bm._repo.load_file("300K_val.npy")
    gt_samples = np.load(gt_samples_path)

    tica_proj = get_tica_projections(gt_samples, topology, tica_model)
    tica_hist = get_tica_hist(tica_proj)
    visualize_histogram_2d(tica_hist, show=True, vmax=12.5)

    tica_proj_true = tica_proj[:1000]
    tica_proj_pred = tica_proj[:1000] + 0.01 * np.random.randn(*tica_proj_true.shape)
    _, tica_w2 = get_tica_wasserstein_1_2(tica_proj_true, tica_proj_pred)
    print("TICA W2:", tica_w2)
