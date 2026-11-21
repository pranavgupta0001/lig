import unittest

from lig import SmilesEncoder, represent_smiles
from lig.models import ModelFactory
from lig.preprocessing import MolecularRepresentation


class RepresentationTests(unittest.TestCase):
    def test_fingerprint_is_fixed_length(self):
        first = represent_smiles("CCO", "fingerprint", size=64)
        second = represent_smiles("C=C", MolecularRepresentation.FINGERPRINT, size=64)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(second), 64)
        self.assertEqual(set(first) | set(second), {0, 1})

    def test_empty_smiles_is_rejected(self):
        with self.assertRaises(ValueError):
            represent_smiles(" ")

    def test_encoder_returns_one_row_per_polymer(self):
        encoded = SmilesEncoder("descriptors").fit_transform(["CC", "CCO"])
        self.assertEqual(len(encoded), 2)
        self.assertEqual(len(encoded[0]), len(encoded[1]))

    def test_invalid_representation_is_rejected(self):
        with self.assertRaises(ValueError):
            represent_smiles("CC", "invalid")


class ModelFactoryTests(unittest.TestCase):
    def test_supported_names(self):
        try:
            ModelFactory.create("random_forest")
        except ImportError:
            self.skipTest("the 'ml' extra is not installed")

    def test_random_forest_training(self):
        try:
            from lig import train_and_evaluate
        except ImportError:
            self.skipTest("the 'ml' extra is not installed")
        smiles = ["CC", "CCO", "CCC", "C=C", "C#N", "COC"]
        targets = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
        model, report = train_and_evaluate(
            smiles,
            targets,
            representation="fingerprint",
            model="random_forest",
            test_size=0.25,
        )
        self.assertEqual(report["n_samples"], 6)
        self.assertTrue(hasattr(model, "feature_importances_"))


if __name__ == "__main__":
    unittest.main()
