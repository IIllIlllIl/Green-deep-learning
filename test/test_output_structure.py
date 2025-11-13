#!/usr/bin/env python3
"""
Unit Tests for Output Structure Implementation

Tests the hierarchical directory structure and CSV generation functionality
implemented in mutation.py, including:
- ExperimentSession class
- Directory structure creation
- CSV generation
- Parallel training background log placement
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation import ExperimentSession, MutationRunner


class TestExperimentSession(unittest.TestCase):
    """Test ExperimentSession class functionality"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_session_creation(self):
        """Test that session directory is created correctly"""
        session = ExperimentSession(self.temp_dir)

        # Check session directory exists
        self.assertTrue(session.session_dir.exists())
        self.assertTrue(session.session_dir.is_dir())

        # Check directory name format
        dir_name = session.session_dir.name
        self.assertTrue(dir_name.startswith("run_"))
        self.assertEqual(len(dir_name), len("run_20251112_150000"))

        # Check session_id format
        self.assertEqual(len(session.session_id), len("20251112_150000"))

    def test_experiment_counter_initialization(self):
        """Test that experiment counter starts at 0"""
        session = ExperimentSession(self.temp_dir)
        self.assertEqual(session.experiment_counter, 0)

    def test_experiments_list_initialization(self):
        """Test that experiments list is initialized empty"""
        session = ExperimentSession(self.temp_dir)
        self.assertEqual(session.experiments, [])

    def test_experiment_dir_generation(self):
        """Test experiment directory generation with auto-incrementing sequence"""
        session = ExperimentSession(self.temp_dir)

        # Generate first experiment directory
        exp_dir1, exp_id1 = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20")

        self.assertEqual(session.experiment_counter, 1)
        self.assertTrue(exp_dir1.exists())
        self.assertEqual(exp_id1, "pytorch_resnet_cifar10_resnet20_001")
        self.assertEqual(exp_dir1.name, "pytorch_resnet_cifar10_resnet20_001")

        # Check energy subdirectory created
        energy_dir1 = exp_dir1 / "energy"
        self.assertTrue(energy_dir1.exists())
        self.assertTrue(energy_dir1.is_dir())

        # Generate second experiment directory
        exp_dir2, exp_id2 = session.get_next_experiment_dir("VulBERTa", "mlp")

        self.assertEqual(session.experiment_counter, 2)
        self.assertTrue(exp_dir2.exists())
        self.assertEqual(exp_id2, "VulBERTa_mlp_002")

        # Generate third experiment directory
        exp_dir3, exp_id3 = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20")

        self.assertEqual(session.experiment_counter, 3)
        self.assertTrue(exp_dir3.exists())
        self.assertEqual(exp_id3, "pytorch_resnet_cifar10_resnet20_003")

    def test_parallel_experiment_naming(self):
        """Test parallel experiment naming with _parallel suffix"""
        session = ExperimentSession(self.temp_dir)

        # Generate normal experiment
        exp_dir1, exp_id1 = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20", mode="train")
        self.assertEqual(exp_id1, "pytorch_resnet_cifar10_resnet20_001")
        self.assertFalse(exp_id1.endswith("_parallel"))

        # Generate parallel experiment
        exp_dir2, exp_id2 = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20", mode="parallel")
        self.assertEqual(exp_id2, "pytorch_resnet_cifar10_resnet20_002_parallel")
        self.assertTrue(exp_id2.endswith("_parallel"))

    def test_add_experiment_result(self):
        """Test adding experiment results to session"""
        session = ExperimentSession(self.temp_dir)

        result1 = {
            "experiment_id": "test_001",
            "repository": "pytorch_resnet_cifar10",
            "model": "resnet20",
            "training_success": True
        }

        result2 = {
            "experiment_id": "test_002",
            "repository": "VulBERTa",
            "model": "mlp",
            "training_success": True
        }

        session.add_experiment_result(result1)
        self.assertEqual(len(session.experiments), 1)
        self.assertEqual(session.experiments[0], result1)

        session.add_experiment_result(result2)
        self.assertEqual(len(session.experiments), 2)
        self.assertEqual(session.experiments[1], result2)


class TestCSVGeneration(unittest.TestCase):
    """Test CSV generation functionality"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_csv_generation_empty(self):
        """Test CSV generation with no experiments"""
        session = ExperimentSession(self.temp_dir)

        csv_file = session.generate_summary_csv()
        self.assertIsNone(csv_file)

    def test_csv_generation_single_experiment(self):
        """Test CSV generation with single experiment"""
        session = ExperimentSession(self.temp_dir)

        result = {
            "experiment_id": "pytorch_resnet_cifar10_resnet20_001",
            "timestamp": "2025-11-12T15:00:00.123456",
            "repository": "pytorch_resnet_cifar10",
            "model": "resnet20",
            "training_success": True,
            "duration_seconds": 1234.56,
            "retries": 0,
            "hyperparameters": {
                "epochs": 100,
                "learning_rate": 0.001,
                "dropout": 0.5
            },
            "performance_metrics": {
                "accuracy": 0.92
            },
            "energy_metrics": {
                "cpu_energy_total_joules": 80095.55,
                "gpu_energy_total_joules": 527217.33,
                "gpu_power_avg_watts": 246.36
            }
        }

        session.add_experiment_result(result)
        csv_file = session.generate_summary_csv()

        self.assertIsNotNone(csv_file)
        self.assertTrue(csv_file.exists())
        self.assertEqual(csv_file.name, "summary.csv")

        # Read and verify CSV content
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]

            # Check base columns
            self.assertEqual(row["experiment_id"], "pytorch_resnet_cifar10_resnet20_001")
            self.assertEqual(row["repository"], "pytorch_resnet_cifar10")
            self.assertEqual(row["model"], "resnet20")
            self.assertEqual(row["training_success"], "True")
            self.assertEqual(row["duration_seconds"], "1234.56")
            self.assertEqual(row["retries"], "0")

            # Check hyperparameter columns
            self.assertEqual(row["hyperparam_dropout"], "0.5")
            self.assertEqual(row["hyperparam_epochs"], "100")
            self.assertEqual(row["hyperparam_learning_rate"], "0.001")

            # Check performance metric columns
            self.assertEqual(row["perf_accuracy"], "0.92")

            # Check energy metric columns
            self.assertEqual(row["energy_cpu_total_joules"], "80095.55")
            self.assertEqual(row["energy_gpu_total_joules"], "527217.33")
            self.assertEqual(row["energy_gpu_avg_watts"], "246.36")

    def test_csv_generation_multiple_experiments(self):
        """Test CSV generation with multiple experiments"""
        session = ExperimentSession(self.temp_dir)

        result1 = {
            "experiment_id": "pytorch_resnet_cifar10_resnet20_001",
            "timestamp": "2025-11-12T15:00:00",
            "repository": "pytorch_resnet_cifar10",
            "model": "resnet20",
            "training_success": True,
            "duration_seconds": 1234.56,
            "retries": 0,
            "hyperparameters": {"epochs": 100, "learning_rate": 0.001},
            "performance_metrics": {"accuracy": 0.92},
            "energy_metrics": {"cpu_energy_total_joules": 80000}
        }

        result2 = {
            "experiment_id": "pytorch_resnet_cifar10_resnet20_002",
            "timestamp": "2025-11-12T16:00:00",
            "repository": "pytorch_resnet_cifar10",
            "model": "resnet20",
            "training_success": True,
            "duration_seconds": 1300.00,
            "retries": 1,
            "hyperparameters": {"epochs": 150, "learning_rate": 0.0005},
            "performance_metrics": {"accuracy": 0.94},
            "energy_metrics": {"cpu_energy_total_joules": 90000}
        }

        session.add_experiment_result(result1)
        session.add_experiment_result(result2)
        csv_file = session.generate_summary_csv()

        self.assertTrue(csv_file.exists())

        # Read and verify CSV content
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 2)

            # Verify first row
            self.assertEqual(rows[0]["experiment_id"], "pytorch_resnet_cifar10_resnet20_001")
            self.assertEqual(rows[0]["hyperparam_epochs"], "100")
            self.assertEqual(rows[0]["hyperparam_learning_rate"], "0.001")

            # Verify second row
            self.assertEqual(rows[1]["experiment_id"], "pytorch_resnet_cifar10_resnet20_002")
            self.assertEqual(rows[1]["hyperparam_epochs"], "150")
            self.assertEqual(rows[1]["hyperparam_learning_rate"], "0.0005")

    def test_csv_dynamic_columns(self):
        """Test that CSV columns adapt to different hyperparameters"""
        session = ExperimentSession(self.temp_dir)

        # First experiment with epochs and learning_rate
        result1 = {
            "experiment_id": "test_001",
            "timestamp": "2025-11-12T15:00:00",
            "repository": "pytorch_resnet_cifar10",
            "model": "resnet20",
            "training_success": True,
            "duration_seconds": 1000,
            "retries": 0,
            "hyperparameters": {"epochs": 100, "learning_rate": 0.001},
            "performance_metrics": {"accuracy": 0.92},
            "energy_metrics": {}
        }

        # Second experiment with dropout and weight_decay (different params!)
        result2 = {
            "experiment_id": "test_002",
            "timestamp": "2025-11-12T16:00:00",
            "repository": "VulBERTa",
            "model": "mlp",
            "training_success": True,
            "duration_seconds": 500,
            "retries": 0,
            "hyperparameters": {"dropout": 0.5, "weight_decay": 0.0001},
            "performance_metrics": {"f1_score": 0.85},
            "energy_metrics": {}
        }

        session.add_experiment_result(result1)
        session.add_experiment_result(result2)
        csv_file = session.generate_summary_csv()

        # Read CSV and check columns
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Should have columns for ALL hyperparameters from both experiments
            self.assertIn("hyperparam_epochs", fieldnames)
            self.assertIn("hyperparam_learning_rate", fieldnames)
            self.assertIn("hyperparam_dropout", fieldnames)
            self.assertIn("hyperparam_weight_decay", fieldnames)

            # Should have columns for ALL performance metrics from both experiments
            self.assertIn("perf_accuracy", fieldnames)
            self.assertIn("perf_f1_score", fieldnames)

            # Read rows
            rows = list(reader)

            # First row should have epochs/learning_rate filled, dropout/weight_decay empty
            self.assertEqual(rows[0]["hyperparam_epochs"], "100")
            self.assertEqual(rows[0]["hyperparam_learning_rate"], "0.001")
            self.assertEqual(rows[0]["hyperparam_dropout"], "")
            self.assertEqual(rows[0]["hyperparam_weight_decay"], "")

            # Second row should have dropout/weight_decay filled, epochs/learning_rate empty
            self.assertEqual(rows[1]["hyperparam_dropout"], "0.5")
            self.assertEqual(rows[1]["hyperparam_weight_decay"], "0.0001")
            self.assertEqual(rows[1]["hyperparam_epochs"], "")
            self.assertEqual(rows[1]["hyperparam_learning_rate"], "")


class TestParallelBackgroundLogs(unittest.TestCase):
    """Test parallel training background log placement"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parallel_experiment_structure(self):
        """Test that parallel experiments create correct directory structure"""
        session = ExperimentSession(self.temp_dir)

        # Create parallel experiment directory
        exp_dir, exp_id = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20", mode="parallel")

        # Verify parallel naming
        self.assertTrue(exp_id.endswith("_parallel"))
        self.assertEqual(exp_id, "pytorch_resnet_cifar10_resnet20_001_parallel")

        # Create background_logs subdirectory (as done in run_parallel_experiment)
        bg_log_dir = exp_dir / "background_logs"
        bg_log_dir.mkdir(exist_ok=True, parents=True)

        # Verify structure
        self.assertTrue(exp_dir.exists())
        self.assertTrue((exp_dir / "energy").exists())
        self.assertTrue(bg_log_dir.exists())

        # Simulate background logs
        bg_log_1 = bg_log_dir / "run_1.log"
        bg_log_2 = bg_log_dir / "run_2.log"
        bg_log_1.write_text("Background training run 1")
        bg_log_2.write_text("Background training run 2")

        # Verify background logs are in foreground experiment directory
        self.assertTrue(bg_log_1.exists())
        self.assertTrue(bg_log_2.exists())
        self.assertEqual(bg_log_1.parent.parent, exp_dir)
        self.assertEqual(bg_log_2.parent.parent, exp_dir)

    def test_background_logs_location(self):
        """Test that background logs are in correct location relative to foreground experiment"""
        session = ExperimentSession(self.temp_dir)

        # Create parallel experiment
        fg_exp_dir, fg_exp_id = session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20", mode="parallel")

        # Background logs directory
        bg_log_dir = fg_exp_dir / "background_logs"
        bg_log_dir.mkdir(exist_ok=True, parents=True)

        # Expected structure:
        # pytorch_resnet_cifar10_resnet20_001_parallel/
        #   ├── energy/
        #   └── background_logs/

        expected_structure = {
            "energy": fg_exp_dir / "energy",
            "background_logs": bg_log_dir
        }

        for name, path in expected_structure.items():
            self.assertTrue(path.exists(), f"{name} should exist at {path}")
            self.assertTrue(path.is_dir(), f"{name} should be a directory")

        # Verify parent relationship
        self.assertEqual(bg_log_dir.parent, fg_exp_dir)


class TestMutationRunnerIntegration(unittest.TestCase):
    """Test MutationRunner integration with new output structure"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = Path(__file__).parent.parent / "config" / "models_config.json"

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_mutation_runner_session_initialization(self):
        """Test that MutationRunner initializes ExperimentSession"""
        # Need to temporarily override results_dir
        runner = MutationRunner(config_path=str(self.config_path))

        # Check session is initialized
        self.assertIsNotNone(runner.session)
        self.assertIsInstance(runner.session, ExperimentSession)
        self.assertEqual(runner.session.experiment_counter, 0)

    def test_experiment_directory_structure(self):
        """Test that experiment directory structure is created correctly"""
        runner = MutationRunner(config_path=str(self.config_path))

        # Get experiment directory
        exp_dir, exp_id = runner.session.get_next_experiment_dir("pytorch_resnet_cifar10", "resnet20")

        # Verify structure
        self.assertTrue(exp_dir.exists())
        self.assertTrue((exp_dir / "energy").exists())

        # Expected paths
        expected_log = exp_dir / "training.log"
        expected_json = exp_dir / "experiment.json"
        expected_energy_dir = exp_dir / "energy"

        # These files don't exist yet, but paths should be correct
        self.assertEqual(expected_log.parent, exp_dir)
        self.assertEqual(expected_json.parent, exp_dir)
        self.assertEqual(expected_energy_dir.parent, exp_dir)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentSession))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelBackgroundLogs))
    suite.addTests(loader.loadTestsFromTestCase(TestMutationRunnerIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
