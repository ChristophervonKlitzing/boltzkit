import io
import os
import unittest
import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path

from boltzkit.utils.cached_repo import VirtualRepo
from boltzkit.utils.dataloader import CacheLoadingArgs, CachedRepoDatasetLoader


def npy_to_bytes(arr: np.ndarray):
    """Converts a numpy array to the raw bytes of a .npy file."""
    f = io.BytesIO()
    np.save(f, arr)
    return f.getvalue()
    # We use a helper to serialize numpy arrays to bytes for the VirtualRepo


class TestCachedRepoDatasetLoader(unittest.TestCase):
    def setUp(self):
        # 1. Create a temporary directory for the local cache
        self.test_base_dir = Path(tempfile.mkdtemp())
        self.local_repo_path = self.test_base_dir / "virtual_test_repo"
        self.local_repo_path.mkdir()

        # 2. Prepare toy data
        self.kB_T = 1.0
        self.T = 1.0
        self.samples = np.random.randn(10, 3)
        self.log_prob_fn = lambda x: -0.5 * np.sum(x**2, axis=-1)
        self.log_probs = self.log_prob_fn(self.samples)

        # 3. Define the Virtual File Tree (simulating a remote repo)
        self.info_yaml_content = {
            "datasets": {
                "1.0": {
                    "train": {
                        "samples": "train_samples.npy",
                        "log_probs": "train_lp.npy",
                    },
                    "test": "test_samples.npy",  # Shorthand test
                }
            }
        }

        self.file_tree = {
            "info.yaml": yaml.dump(self.info_yaml_content),
            "train_samples.npy": npy_to_bytes(self.samples),
            "train_lp.npy": npy_to_bytes(self.log_probs),
            "test_samples.npy": npy_to_bytes(self.samples[:5]),
        }

        # 4. Initialize the VirtualRepo
        # Note: VirtualRepo calls post_init in __init__, which requires info.yaml
        self.repo = VirtualRepo(
            remote_uri="virtual://test_repo",
            local_repo_path=self.local_repo_path,
            lazy_load=True,
            file_tree=self.file_tree,
        )

    def tearDown(self):
        shutil.rmtree(self.test_base_dir)

    def test_load_train_with_log_probs(self):
        """Verify that samples and log_probs are loaded correctly from VirtualRepo."""
        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=None,  # Not used since we have files
            score_fn=None,
            caching_args=CacheLoadingArgs(
                allow_autogen=False,
            ),
        )

        dataset = loader.load_dataset(type="train", length=5, include_log_probs=True)

        self.assertEqual(len(dataset.get_samples()), 5)
        np.testing.assert_allclose(dataset.get_samples(), self.samples[:5])
        np.testing.assert_allclose(dataset.get_log_probs(), self.log_probs[:5])

    def test_load_test_shorthand(self):
        """Test just samples"""
        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=None,
            score_fn=None,
        )

        dataset = loader.load_dataset(type="test", length=2)

        self.assertEqual(len(dataset.get_samples()), 2)
        self.assertIsNone(dataset.get_log_probs())
        self.assertIsNone(dataset.get_scores())

    def test_insufficient_samples_error(self):
        """Verify that requesting more samples than exist triggers a ValueError."""
        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=None,
            score_fn=None,
        )

        with self.assertRaisesRegex(ValueError, "Requested more samples"):
            loader.load_dataset(type="train", length=100)

    def test_autogen_fallback(self):
        """Test that log_prob_fn is called when file is missing and allow_autogen is True."""

        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=self.log_prob_fn,
            score_fn=None,
            caching_args=CacheLoadingArgs(
                allow_autogen=True,
            ),
        )

        # =====================
        # Request 'test' split which has NO log_probs in info.yaml
        dataset = loader.load_dataset(type="test", length=3, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 3)
        np.testing.assert_array_equal(dataset.get_log_probs(), self.log_probs[:3])

        # =====================
        # Load with more samples
        dataset = loader.load_dataset(type="test", length=5, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 5)
        np.testing.assert_array_equal(dataset.get_log_probs(), self.log_probs[:5])

    def test_autogen_fallback_scores(self):
        """
        Test that score_fn is called when file is missing and allow_autogen is True.

        Caching is turned OFF in this case!
        """
        # Define a toy score function (e.g., negative of samples)

        def score_fn(x: np.ndarray):
            return -x

        count = 0

        def score_fn_counting(x: np.ndarray):
            nonlocal count
            count += x.shape[0]
            return score_fn(x)

        expected_scores = score_fn(self.samples)

        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=None,
            score_fn=score_fn_counting,
            caching_args=CacheLoadingArgs(
                allow_autogen=True,
                cache_scores=False,  # just to be explicit
            ),
        )

        # =====================
        # Request 'test' split which has NO scores in info.yaml
        dataset = loader.load_dataset(type="test", length=3, include_scores=True)

        self.assertEqual(len(dataset.get_scores()), 3)
        np.testing.assert_array_equal(dataset.get_scores(), expected_scores[:3])
        self.assertEqual(count, 3)

        # =====================
        # Load with more samples to ensure cache is updated/sliced correctly
        dataset = loader.load_dataset(type="test", length=5, include_scores=True)

        self.assertEqual(len(dataset.get_scores()), 5)
        np.testing.assert_array_equal(dataset.get_scores(), expected_scores[:5])
        self.assertEqual(count, 3 + 5)  # no caching -> scores get re-computed

    def test_autogen_fallback_scores_cached(self):
        """
        Test that score_fn is called when file is missing and allow_autogen is True.

        Caching is turned OFF in this case!
        """

        # Define a toy score function (e.g., negative of samples)
        def score_fn(x: np.ndarray):
            return -x

        count = 0

        def score_fn_counting(x: np.ndarray):
            nonlocal count
            count += x.shape[0]
            return score_fn(x)

        expected_scores = score_fn(self.samples)

        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=None,
            score_fn=score_fn_counting,
            caching_args=CacheLoadingArgs(
                allow_autogen=True,
                cache_scores=True,
            ),
        )

        # =====================
        # Request 'test' split which has NO scores in info.yaml
        dataset = loader.load_dataset(type="test", length=3, include_scores=True)

        self.assertEqual(len(dataset.get_scores()), 3)
        np.testing.assert_array_equal(dataset.get_scores(), expected_scores[:3])
        self.assertEqual(count, 3)

        # =====================
        # Load with more samples to ensure cache is updated/sliced correctly
        dataset = loader.load_dataset(type="test", length=5, include_scores=True)

        self.assertEqual(len(dataset.get_scores()), 5)
        np.testing.assert_array_equal(dataset.get_scores(), expected_scores[:5])
        self.assertEqual(count, 3 + 2)  # caching -> reuse of 3 scores

    def test_autogen_fallback_log_probs(self):
        """
        Test that log_prob_fn is called when file is missing and allow_autogen is True.
        Caching is turned OFF in this case.
        """

        log_prob_fn = self.log_prob_fn

        count = 0

        def log_prob_fn_counting(x: np.ndarray):
            nonlocal count
            count += x.shape[0]
            return log_prob_fn(x)

        expected_lp = log_prob_fn(self.samples)

        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=log_prob_fn_counting,
            score_fn=None,
            caching_args=CacheLoadingArgs(
                allow_autogen=True,
                cache_log_probs=False,  # Explicitly OFF
            ),
        )

        # =====================
        # Request 'test' split (no log_probs in VirtualRepo)
        dataset = loader.load_dataset(type="test", length=3, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 3)
        np.testing.assert_array_equal(dataset.get_log_probs(), expected_lp[:3])
        self.assertEqual(count, 3)

        # =====================
        # Load with more samples: no caching -> everything re-computed
        dataset = loader.load_dataset(type="test", length=5, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 5)
        np.testing.assert_array_equal(dataset.get_log_probs(), expected_lp[:5])
        self.assertEqual(count, 3 + 5)

    def test_autogen_fallback_log_probs_cached(self):
        """
        Test that log_prob_fn is called when file is missing and allow_autogen is True.
        Caching is turned ON.
        """

        log_prob_fn = self.log_prob_fn

        count = 0

        def log_prob_fn_counting(x: np.ndarray):
            nonlocal count
            count += x.shape[0]
            return log_prob_fn(x)

        expected_lp = log_prob_fn(self.samples)

        loader = CachedRepoDatasetLoader(
            kB_T=self.kB_T,
            T=self.T,
            cached_repo=self.repo,
            log_prob_fn=log_prob_fn_counting,
            score_fn=None,
            caching_args=CacheLoadingArgs(
                allow_autogen=True,
                cache_log_probs=True,  # Explicitly ON
            ),
        )

        # =====================
        # First call: computes 3 points
        dataset = loader.load_dataset(type="test", length=3, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 3)
        self.assertEqual(count, 3)

        # =====================
        # Second call with more samples:
        # Should reuse 3 from cache and only compute the 2 missing ones.
        dataset = loader.load_dataset(type="test", length=5, include_log_probs=True)

        self.assertEqual(len(dataset.get_log_probs()), 5)
        np.testing.assert_array_equal(dataset.get_log_probs(), expected_lp[:5])
        self.assertEqual(count, 3 + 2)  # Total count 5, verify partial compute


if __name__ == "__main__":
    unittest.main()
