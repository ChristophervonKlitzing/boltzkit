import numpy as np

from boltzkit.targets.base import DispatchedTarget
from boltzkit.targets.gmm._np_gmm import NumPyGMM


class GMM(DispatchedTarget):
    def __init__(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        logits: np.ndarray,
    ):
        dim = means.shape[1]
        self._means = means
        self._scales = scales
        self._logits = logits

        try:
            # Currently, the jax log prob function must be immediately created here
            # TODO: Implement a lazy variant of `create_dispatch` used in the base class.
            # This would allow passing the factory method instead of actual log prob function.

            from boltzkit.targets.gmm._jax_gmm import make_jax_gmm_log_prob

            jax_log_prob_fn_single = make_jax_gmm_log_prob(
                means=self._means, scales=self._scales, logits=self._logits
            )
        except Exception as e:
            print(f"Failed to initialize jax log prob function: '{e}'")
            jax_log_prob_fn_single = None

        super().__init__(dim, jax_log_prob_fn_single=jax_log_prob_fn_single)

        self._np_gmm = NumPyGMM(
            means=self._means, scales=self._scales, logits=self._logits
        )

        self._torch_gmm = None

    def _numpy_log_prob(self, x):
        return self._np_gmm.log_prob(x)

    def _torch_log_prob(self, x):
        if self._torch_gmm is None:
            from boltzkit.targets.gmm._torch_gmm import TorchGMM

            self._torch_gmm = TorchGMM(
                means=self._means, scales=self._scales, logits=self._logits
            )
            # Assumes that most calls will use the same device
            # and that device moves are therefore uncommon.
            if self._torch_gmm.device != x.device:
                self._torch_gmm.to(x.device)

        return self._torch_gmm.log_prob(x)

    def _create_torch_log_prob(self): ...

    def can_sample(self):
        return True

    def sample(self, n_samples):
        return self._np_gmm.sample(n_samples)


if __name__ == "__main__":
    import torch
    import jax.numpy as jnp

    means = np.array([[0.0, 0.0], [1.0, 0.0]])
    scales = np.array([[1.0, 1.0], [1.0, 1.0]])
    logits = np.log(np.array([0.5, 0.5]))
    target = GMM(means, scales, logits)
    print(target.get_log_prob(torch.tensor([[0.5, 0], [0.5, 1]])))
    print(target.get_log_prob(np.array([[0.5, 0], [0.5, 1]])))
    print(target.get_log_prob(jnp.asarray([[0.5, 0], [0.5, 1]])))
    print(target.get_log_prob(jnp.asarray([[0.5, 0], [0.5, 1], [0.5, 1.2]])))
    print(target.get_log_prob(jnp.asarray([[0.5, 0], [0.5, 1], [0.5, 1.2]])))
