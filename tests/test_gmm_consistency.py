import unittest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from boltzkit.targets.gmm._jax_gmm import JaxGMM
from boltzkit.targets.gmm._torch_gmm import TorchGMM


class TestGMMConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ---- Config ----
        cls.seed = 0
        np.random.seed(cls.seed)
        cls.key = jax.random.PRNGKey(cls.seed)

        cls.K = 5
        cls.D = 3
        cls.N = 1000
        cls.N_samples = 10_000

        # ---- Parameters ----
        cls.means = np.random.uniform(low=-5.0, high=5.0, size=(cls.K, cls.D)).astype(
            np.float32
        )

        cls.scales = np.ones((cls.K, cls.D), dtype=np.float32)  # std = 1
        cls.logits: np.ndarray = np.zeros((cls.K,)) - np.log(cls.K)
        cls.logits = cls.logits.astype(np.float32)

        # ---- Models ----
        cls.torch_gmm = TorchGMM(cls.means, cls.scales, cls.logits)

        cls.jax_gmm = JaxGMM(
            jnp.array(cls.means),
            jnp.array(cls.scales),
            jnp.array(cls.logits),
        )

    def test_log_prob_consistency(self):
        x_np = np.random.randn(self.N, self.D).astype(np.float32)

        torch_lp = self.torch_gmm.log_prob(torch.from_numpy(x_np)).detach().numpy()

        jax_lp = np.array(self.jax_gmm.log_prob(jnp.array(x_np)))

        self.assertTrue(
            np.allclose(torch_lp, jax_lp, atol=1e-5, rtol=1e-5),
            msg="Torch and JAX log_prob outputs differ!",
        )

    def test_sampling_mean(self):
        torch_samples = self.torch_gmm.sample(self.N_samples).detach().numpy()

        self.__class__.key, subkey = jax.random.split(self.__class__.key)
        jax_samples = np.array(self.jax_gmm.sample(subkey, self.N_samples))

        torch_mean = torch_samples.mean(axis=0)
        jax_mean = jax_samples.mean(axis=0)

        self.assertTrue(
            np.allclose(torch_mean, jax_mean, atol=0.1),
            msg=f"Means differ! Torch: {torch_mean}, JAX: {jax_mean}",
        )

    def test_sampling_variance(self):
        torch_samples = self.torch_gmm.sample(self.N_samples).detach().numpy()

        self.__class__.key, subkey = jax.random.split(self.__class__.key)
        jax_samples = np.array(self.jax_gmm.sample(subkey, self.N_samples))

        torch_std = torch_samples.std(axis=0)
        jax_std = jax_samples.std(axis=0)

        self.assertTrue(
            np.allclose(torch_std, jax_std, atol=0.1),
            msg=f"Standard deviations differ! Torch: {torch_std}, JAX: {jax_std}",
        )

    def test_2d_marginal_consistency(self):
        x_np = np.random.randn(self.N, 2).astype(np.float32)

        torch_2d = self.torch_gmm.get_distribution_2D(0, 1)
        jax_2d = self.jax_gmm.get_distribution_2D(0, 1)

        torch_lp = torch_2d.log_prob(torch.from_numpy(x_np)).detach().numpy()

        jax_lp = np.array(jax_2d.log_prob(jnp.array(x_np)))

        self.assertTrue(
            np.allclose(torch_lp, jax_lp, atol=1e-5, rtol=1e-5),
            msg="2D marginal log_prob mismatch!",
        )


if __name__ == "__main__":
    unittest.main()
