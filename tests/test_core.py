"""Basic tests for the parts that do not need model checkpoints or network access."""

import unittest

from molecule_tools import inspect_molecule
from scoring import evaluate_candidate


class TestMoleculeTools(unittest.TestCase):
    def test_valid_smiles(self):
        result = inspect_molecule("CCO")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["qed"], 0)
        self.assertLessEqual(result["qed"], 1)

    def test_invalid_smiles(self):
        self.assertIsNone(inspect_molecule("not a smiles"))


class TestScoring(unittest.TestCase):
    def test_filter_result_is_boolean(self):
        properties = {"qed": 0.7}
        predictions = {"predicted_jak2": -8.0, "predicted_lck": -6.0}
        result = evaluate_candidate(properties, predictions)
        self.assertIsInstance(result["passes_filters"], bool)
        self.assertIsInstance(result["combined_score"], float)


if __name__ == "__main__":
    unittest.main()
