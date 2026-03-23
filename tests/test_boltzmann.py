import unittest

import numpy as np
from boltzkit.targets.boltzmann import MolecularBoltzmann


class TestBoltzmann(unittest.TestCase):
    def setUp(self):
        path = "datasets/chrklitz99/alanine_dipeptide"
        bm = MolecularBoltzmann(path)
        self.bm = bm

    def test_log_probs_match_forces(self):
        np_positions = self.bm._pdb.getPositions(asNumpy=True)
        np_positions = np.expand_dims(np_positions, 0)

        # rng = np.random.default_rng(seed=0)

        log_prob, score = self.bm.get_log_prob_and_score(np_positions)
        shift = 0.0000001 * score
        pos_shifted = np_positions + shift
        log_prob_shifted = self.bm.get_log_prob(pos_shifted)

        v1 = log_prob_shifted - log_prob
        v2: np.ndarray = np.dot(score.flatten(), shift.flatten())
        self.assertAlmostEqual(v1.item() / v2.item(), 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
