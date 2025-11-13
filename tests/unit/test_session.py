#!/usr/bin/env python3
"""
Unit tests for mutation.session module

Tests ExperimentSession class for directory management and result tracking.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from mutation.session import ExperimentSession


class TestExperimentSession(unittest.TestCase):
    """Test ExperimentSession class"""

    def setUp(self):
        """Create temporary directory for testing"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_session_creation(self):
        """Test session directory creation"""
        session = ExperimentSession(self.temp_dir)

        self.assertTrue(session.session_dir.exists())
        self.assertEqual(session.experiment_counter, 0)
        self.assertIsNotNone(session.session_id)
        self.assertEqual(len(session.experiments), 0)

    def test_session_id_format(self):
        """Test session ID format (timestamp)"""
        session = ExperimentSession(self.temp_dir)

        # Should match format: YYYYMMDD_HHMMSS
        self.assertEqual(len(session.session_id), 15)
        self.assertTrue(session.session_id[8] == "_")

        # Should be parseable as datetime
        try:
            datetime.strptime(session.session_id, "%Y%m%d_%H%M%S")
        except ValueError:
            self.fail("Session ID format is invalid")

    def test_experiment_directory_train_mode(self):
        """Test experiment directory creation in train mode"""
        session = ExperimentSession(self.temp_dir)

        exp_dir, exp_id = session.get_next_experiment_dir("test_repo", "test_model", mode="train")

        self.assertTrue(exp_dir.exists())
        # train mode doesn't add suffix, just sequence number
        self.assertEqual(exp_id, "test_repo_test_model_001")
        self.assertEqual(session.experiment_counter, 1)

    def test_experiment_directory_parallel_mode(self):
        """Test experiment directory creation in parallel mode"""
        session = ExperimentSession(self.temp_dir)

        exp_dir, exp_id = session.get_next_experiment_dir("repo1", "model1", mode="parallel")

        self.assertTrue(exp_dir.exists())
        # parallel mode adds _parallel suffix
        self.assertEqual(exp_id, "repo1_model1_001_parallel")

    def test_counter_increment(self):
        """Test that counter increments correctly"""
        session = ExperimentSession(self.temp_dir)

        exp_dir1, exp_id1 = session.get_next_experiment_dir("repo", "model")
        exp_dir2, exp_id2 = session.get_next_experiment_dir("repo", "model")
        exp_dir3, exp_id3 = session.get_next_experiment_dir("repo", "model")

        self.assertEqual(session.experiment_counter, 3)
        self.assertIn("_001", exp_id1)
        self.assertIn("_002", exp_id2)
        self.assertIn("_003", exp_id3)

    def test_add_experiment_result(self):
        """Test adding experiment results"""
        session = ExperimentSession(self.temp_dir)

        result = {
            "experiment_id": "test_001",
            "repository": "test_repo",
            "model": "test_model",
            "hyperparameters": {"epochs": 10},
            "training_success": True,
            "duration_seconds": 100,
            "energy_metrics": {},
            "performance_metrics": {},
            "retries": 0
        }

        session.add_experiment_result(result)

        self.assertEqual(len(session.experiments), 1)
        self.assertEqual(session.experiments[0]["experiment_id"], "test_001")

    def test_generate_summary_csv(self):
        """Test CSV summary generation"""
        session = ExperimentSession(self.temp_dir)

        # Add multiple results
        for i in range(3):
            result = {
                "experiment_id": f"test_{i:03d}",
                "timestamp": datetime.now().isoformat(),
                "repository": "test_repo",
                "model": "test_model",
                "hyperparameters": {"epochs": 10 + i, "learning_rate": 0.01},
                "training_success": True,
                "duration_seconds": 100 + i,
                "energy_metrics": {"cpu_energy_total_joules": 1000 + i * 100},
                "performance_metrics": {"test_accuracy": 0.9 + i * 0.01},
                "retries": 0,
                "error_message": ""
            }
            session.add_experiment_result(result)

        csv_file = session.generate_summary_csv()

        self.assertIsNotNone(csv_file)
        self.assertTrue(csv_file.exists())

        # Read CSV and verify content
        with open(csv_file, 'r') as f:
            content = f.read()
            self.assertIn("experiment_id", content)
            self.assertIn("test_000", content)
            self.assertIn("test_001", content)
            self.assertIn("test_002", content)
            self.assertIn("epochs", content)
            self.assertIn("learning_rate", content)

    def test_csv_dynamic_columns(self):
        """Test that CSV includes dynamic columns from metrics"""
        session = ExperimentSession(self.temp_dir)

        result = {
            "experiment_id": "test_001",
            "timestamp": datetime.now().isoformat(),
            "repository": "test_repo",
            "model": "test_model",
            "hyperparameters": {"batch_size": 32},
            "training_success": True,
            "duration_seconds": 100,
            "energy_metrics": {
                "cpu_energy_total_joules": 5000,
                "gpu_energy_total_joules": 10000
            },
            "performance_metrics": {
                "test_accuracy": 0.95,
                "train_loss": 0.05
            },
            "retries": 0,
            "error_message": ""
        }
        session.add_experiment_result(result)

        csv_file = session.generate_summary_csv()

        with open(csv_file, 'r') as f:
            header = f.readline()
            # Check dynamic columns are present (with prefixes)
            self.assertIn("hyperparam_batch_size", header)
            self.assertIn("energy_cpu_total_joules", header)
            self.assertIn("energy_gpu_total_joules", header)
            self.assertIn("perf_test_accuracy", header)
            self.assertIn("perf_train_loss", header)


if __name__ == "__main__":
    unittest.main()
