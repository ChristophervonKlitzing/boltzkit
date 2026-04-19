import unittest
import numpy as np


from boltzkit.targets.gaussian_mixture import DiagonalGaussianMixture


class TestGMMConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ---- Config ----
        cls.seed = 0
        np.random.seed(cls.seed)

        cls.target = DiagonalGaussianMixture.create_isotropic_uniform(
            std=1.0, n_components=4, dim=2, mean_range=(-4, 4)
        )

    def test_load_dataset(self):
        """
        Checks the prefix consistency of load_dataset for the GMM model.
        Sampling from GMMs is cheap, making storing a dataset redundant.
        However, load_dataset should behave like it returns a prefix of length `length`
        of the underlying dataset. This test checks exactly that for the GMM model
        as its implementation uses rngs instead of pre-computed ("offline") samples.
        """
        for length in range(1, 1000, 200):
            for k in range(1, 1000, 200):
                d1 = self.target.load_dataset(type="val", length=length)
                d2 = self.target.load_dataset(type="val", length=length + k)

                self.assertTrue(
                    np.array_equal(d1.get_samples(), d2.get_samples()[:length])
                )


if __name__ == "__main__":
    unittest.main()
