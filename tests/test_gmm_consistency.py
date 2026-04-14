import unittest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from boltzkit.targets.gaussian_mixture._jax_mog import create_jax_MoG_eval
from boltzkit.targets.gaussian_mixture._np_mog import NumpyMoG
from boltzkit.targets.gaussian_mixture._torch_mog import TorchMoG


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
        cls.torch_gmm = TorchMoG(cls.means, cls.scales, cls.logits)

        cls.jax_gmm = create_jax_MoG_eval(
            jnp.array(cls.means),
            jnp.array(cls.scales),
            jnp.array(cls.logits),
        )

        cls.np_gmm = NumpyMoG(cls.means, cls.scales, cls.logits)

    def test_log_prob_consistency(self):
        x_np = np.random.randn(self.N, self.D).astype(np.float32)

        torch_log_prob = (
            self.torch_gmm.get_log_prob(torch.from_numpy(x_np)).detach().numpy()
        )

        jax_log_prob = np.array(self.jax_gmm.get_log_prob(jnp.array(x_np)))

        np_log_prob = self.np_gmm.get_log_prob(x_np)

        # Torch vs JAX
        self.assertTrue(
            np.allclose(torch_log_prob, jax_log_prob, atol=1e-5, rtol=1e-5),
            msg="Torch and JAX log_prob outputs differ!",
        )

        # Torch vs NumPy
        self.assertTrue(
            np.allclose(torch_log_prob, np_log_prob, atol=1e-5, rtol=1e-5),
            msg="Torch and NumPy log_prob outputs differ!",
        )

    def test_score_consistency(self):
        x_np = np.random.randn(self.N, self.D).astype(np.float32)

        torch_score = self.torch_gmm.get_score(torch.from_numpy(x_np)).detach().numpy()

        jax_score = np.array(self.jax_gmm.get_score(jnp.array(x_np)))

        np_score = self.np_gmm.get_score(x_np)

        # Torch vs JAX
        self.assertTrue(
            np.allclose(torch_score, jax_score, atol=1e-5, rtol=1e-5),
            msg="Torch and JAX score outputs differ!",
        )

        # Torch vs NumPy
        self.assertTrue(
            np.allclose(torch_score, np_score, atol=1e-5, rtol=1e-5),
            msg="Torch and NumPy score outputs differ!",
        )

    def test_log_prob_and_score_consistency(self):
        x_np = np.random.randn(self.N, self.D).astype(np.float32)

        torch_log_prob, torch_score = self.torch_gmm.get_log_prob_and_score(
            torch.from_numpy(x_np)
        )
        torch_log_prob = torch_log_prob.detach().numpy()
        torch_score = torch_score.detach().numpy()

        jax_log_prob, jax_score = self.jax_gmm.get_log_prob_and_score(jnp.array(x_np))

        np_log_prob, np_score = self.np_gmm.get_log_prob_and_score(x_np)

        # Torch vs JAX
        self.assertTrue(
            np.allclose(torch_score, jax_score, atol=1e-5, rtol=1e-5),
            msg="Torch and JAX score outputs differ!",
        )

        self.assertTrue(
            np.allclose(torch_log_prob, jax_log_prob, atol=1e-5, rtol=1e-5),
            msg="Torch and JAX log_prob outputs differ!",
        )

        # Torch vs NumPy
        self.assertTrue(
            np.allclose(torch_score, np_score, atol=1e-5, rtol=1e-5),
            msg="Torch and NumPy score outputs differ!",
        )

        self.assertTrue(
            np.allclose(torch_log_prob, np_log_prob, atol=1e-5, rtol=1e-5),
            msg="Torch and NumPy log_prob outputs differ!",
        )

        # Internal consistency (NumPy)
        np_log_prob_2 = self.np_gmm.get_log_prob(x_np)
        np_score_2 = self.np_gmm.get_score(x_np)

        self.assertTrue(np.allclose(np_log_prob, np_log_prob_2))
        self.assertTrue(np.allclose(np_score, np_score_2))


if __name__ == "__main__":
    unittest.main()
