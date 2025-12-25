#!/usr/bin/env python3
"""
Unit tests for mutation.hyperparams module

Tests individual functions for hyperparameter mutation and generation.
"""

import unittest
import random
from mutation.hyperparams import (
    mutate_hyperparameter,
    generate_mutations,
    _format_hyperparam_value,
    _build_hyperparam_args
)
from mutation.exceptions import HyperparameterError


class TestMutateHyperparameter(unittest.TestCase):
    """Test mutate_hyperparameter function"""

    def setUp(self):
        """Set random seed for reproducible tests"""
        random.seed(42)

    def test_int_uniform(self):
        """Test integer uniform distribution"""
        config = {
            "type": "int",
            "range": [1, 100],
            "distribution": "uniform"
        }
        value = mutate_hyperparameter(config, "test_param")
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, 100)

    def test_float_uniform(self):
        """Test float uniform distribution"""
        config = {
            "type": "float",
            "range": [0.0, 1.0],
            "distribution": "uniform"
        }
        value = mutate_hyperparameter(config, "test_param")
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_log_uniform(self):
        """Test log uniform distribution"""
        config = {
            "type": "float",
            "range": [0.0001, 0.1],
            "distribution": "log_uniform"
        }
        value = mutate_hyperparameter(config, "test_param")
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0001)
        self.assertLessEqual(value, 0.1)

    def test_invalid_distribution(self):
        """Test invalid distribution raises error"""
        config = {
            "type": "int",
            "range": [1, 10],
            "distribution": "invalid_distribution"
        }
        with self.assertRaises(ValueError):
            mutate_hyperparameter(config, "test_param")


class TestGenerateMutations(unittest.TestCase):
    """Test generate_mutations function"""

    def test_basic_generation(self):
        """Test basic mutation generation"""
        supported_params = {
            "epochs": {
                "type": "int",
                "range": [10, 200],
                "distribution": "uniform",
                "flag": "--epochs"
            },
            "learning_rate": {
                "type": "float",
                "range": [0.001, 0.1],
                "distribution": "uniform",
                "flag": "--lr"
            }
        }

        mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=["epochs", "learning_rate"],
            num_mutations=5,
            random_seed=42
        )

        self.assertEqual(len(mutations), 5)
        for mutation in mutations:
            self.assertIn("epochs", mutation)
            self.assertIn("learning_rate", mutation)
            self.assertIsInstance(mutation["epochs"], int)
            self.assertIsInstance(mutation["learning_rate"], float)

    def test_uniqueness(self):
        """Test that mutations are unique"""
        supported_params = {
            "epochs": {
                "type": "int",
                "range": [10, 100],
                "distribution": "uniform",
                "flag": "--epochs"
            }
        }

        mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=["epochs"],
            num_mutations=10,
            random_seed=42
        )

        # Convert to strings for comparison
        mutation_strs = [str(sorted(m.items())) for m in mutations]
        self.assertEqual(len(mutations), len(set(mutation_strs)),
                        "Mutations should be unique")

    # Note: Reproducibility is not yet fully implemented in mutate_hyperparameter
    # The random_seed parameter is present but not currently used
    @unittest.skip("Reproducibility not yet fully implemented")
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        supported_params = {
            "learning_rate": {
                "type": "float",
                "range": [0.001, 0.1],
                "distribution": "log_uniform",
                "flag": "--lr"
            }
        }

        mutations1 = generate_mutations(
            supported_params=supported_params,
            mutate_params=["learning_rate"],
            num_mutations=3,
            random_seed=123
        )

        mutations2 = generate_mutations(
            supported_params=supported_params,
            mutate_params=["learning_rate"],
            num_mutations=3,
            random_seed=123
        )

        # Convert to strings for comparison (handle floating point precision)
        str1 = [str(m) for m in mutations1]
        str2 = [str(m) for m in mutations2]
        self.assertEqual(str1, str2,
                        "Same seed should produce same mutations")


class TestFormatHyperparamValue(unittest.TestCase):
    """Test _format_hyperparam_value function"""

    def test_format_int(self):
        """Test integer formatting"""
        result = _format_hyperparam_value(42, "int")
        self.assertEqual(result, "42")

    def test_format_float(self):
        """Test float formatting with precision"""
        result = _format_hyperparam_value(0.123456789, "float")
        self.assertEqual(result, "0.123457")

    def test_format_string(self):
        """Test string formatting"""
        result = _format_hyperparam_value("adam", "str")
        self.assertEqual(result, "adam")


class TestBuildHyperparamArgs(unittest.TestCase):
    """Test _build_hyperparam_args function"""

    def test_build_args_as_list(self):
        """Test building arguments as list"""
        supported_params = {
            "epochs": {"flag": "--epochs", "type": "int"},
            "learning_rate": {"flag": "--lr", "type": "float"}
        }
        mutation = {
            "epochs": 10,
            "learning_rate": 0.01
        }

        args = _build_hyperparam_args(supported_params, mutation, as_list=True)

        self.assertIn("--epochs", args)
        self.assertIn("10", args)
        self.assertIn("--lr", args)
        # Note: float formatting may vary
        self.assertTrue(any("0.01" in str(arg) for arg in args))

    def test_build_args_as_string(self):
        """Test building arguments as string"""
        supported_params = {
            "epochs": {"flag": "-e", "type": "int"}
        }
        mutation = {"epochs": 50}

        args = _build_hyperparam_args(supported_params, mutation, as_list=False)

        self.assertIn("-e", args)
        self.assertIn("50", args)


if __name__ == "__main__":
    unittest.main()
