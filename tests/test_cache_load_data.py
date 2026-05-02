import unittest
import numpy as np
from pathlib import Path
import tempfile

from boltzkit.utils.dataloader import cache_load_sample_derived_data


class TestCacheLoadSampleDerivedData(unittest.TestCase):

    def setUp(self):
        self.samples = np.arange(10).reshape(-1, 1)

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return x**2

        self.eval_fn = eval_fn
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.primary_path = self.tmp_path / "data.npy"
        self.cache_path = self.tmp_path / "cache.npy"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_autogen_and_cache(self):
        data = cache_load_sample_derived_data(
            samples=self.samples,
            data_fpath=None,
            data_cache_fpath=self.cache_path,
            data_eval_fn=self.eval_fn,
            allow_autogen=True,
            cache_data=True,
        )
        self.assertTrue(np.allclose(data, self.eval_fn(self.samples)))
        self.assertTrue(self.cache_path.exists())

    def test_load_from_cache(self):
        np.save(self.cache_path, self.eval_fn(self.samples) + 1)

        data = cache_load_sample_derived_data(
            samples=self.samples,
            data_fpath=None,
            data_cache_fpath=self.cache_path,
            allow_autogen=True,
            cache_data=True,
        )
        self.assertTrue(np.allclose(data, self.eval_fn(self.samples)) + 1)

    def test_load_from_primary(self):
        np.save(self.primary_path, self.samples)

        data = cache_load_sample_derived_data(
            samples=self.samples,
            data_fpath=self.primary_path,
            data_cache_fpath=self.cache_path,
            allow_autogen=True,
            cache_data=True,
        )
        self.assertTrue(np.allclose(data, self.samples))

    def test_partial_autogen(self):
        partial = self.eval_fn(self.samples[:5])
        np.save(self.cache_path, partial)

        data = cache_load_sample_derived_data(
            samples=self.samples,
            data_fpath=None,
            data_cache_fpath=self.cache_path,
            data_eval_fn=self.eval_fn,
            allow_autogen=True,
            cache_data=True,
        )
        self.assertTrue(np.allclose(data, self.samples**2))

    def test_autogen_without_fn_raises(self):
        with self.assertRaises(ValueError):
            cache_load_sample_derived_data(
                samples=self.samples,
                data_fpath=None,
                data_cache_fpath=self.cache_path,
                data_eval_fn=None,
                allow_autogen=True,
                cache_data=False,
            )

    def test_no_data_raises(self):
        with self.assertRaises(RuntimeError):
            cache_load_sample_derived_data(
                samples=self.samples,
                data_fpath=None,
                data_cache_fpath=None,
                data_eval_fn=None,
                allow_autogen=False,
                cache_data=False,
            )


if __name__ == "__main__":
    unittest.main()
