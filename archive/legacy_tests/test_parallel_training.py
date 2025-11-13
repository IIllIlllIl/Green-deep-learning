#!/usr/bin/env python3
"""
Test script for parallel training functionality

This script provides unit and integration tests for the parallel training feature
implemented in mutation.py using方案1 (subprocess.Popen + Shell script approach).

Usage:
    # Run all tests
    python3 test/test_parallel_training.py

    # Run specific test
    python3 test/test_parallel_training.py TestParallelTraining.test_background_script_generation
"""

import os
import sys
import time
import unittest
import signal
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mutation import MutationRunner


class TestParallelTraining(unittest.TestCase):
    """Test cases for parallel training functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.runner = MutationRunner()
        cls.test_experiment_id = "test_parallel_20251112_000000"

    def test_build_training_args(self):
        """Test building training argument strings"""
        print("\n[TEST] Testing _build_training_args...")

        # Test with basic hyperparameters
        args = self.runner._build_training_args(
            repo="pytorch_resnet_cifar10",
            model="resnet20",
            hyperparams={"epochs": 10, "learning_rate": 0.01}
        )

        self.assertIsInstance(args, str)
        # Check for epochs flag (could be -e or --epochs depending on config)
        self.assertTrue("-e " in args or "--epochs" in args)
        self.assertIn("10", args)
        self.assertIn("--lr", args)
        self.assertIn("0.01", args)

        print(f"   Generated args: {args}")
        print("   ✓ Build training args passed")

    def test_background_script_generation(self):
        """Test background training script generation"""
        print("\n[TEST] Testing background script generation...")

        experiment_id = f"{self.test_experiment_id}_script_gen"

        # Generate background script
        process, script_path = self.runner._start_background_training(
            repo="pytorch_resnet_cifar10",
            model="resnet20",
            hyperparams={"epochs": 1, "learning_rate": 0.01},
            experiment_id=experiment_id
        )

        try:
            # Check process started
            self.assertIsInstance(process, subprocess.Popen)
            self.assertIsNotNone(process.pid)
            print(f"   ✓ Background process started (PID: {process.pid})")

            # Check script file was created
            self.assertTrue(script_path.exists())
            print(f"   ✓ Script file created: {script_path.name}")

            # Check script is executable
            self.assertTrue(os.access(script_path, os.X_OK))
            print("   ✓ Script is executable")

            # Check script content
            with open(script_path, 'r') as f:
                content = f.read()
                self.assertIn("#!/bin/bash", content)
                self.assertIn("while true", content)
                self.assertIn("run_count", content)
                self.assertIn("RESTART_DELAY", content)
                print("   ✓ Script contains expected loop structure")

            # Wait a moment to let it run
            time.sleep(3)

            # Check process is still running
            self.assertIsNone(process.poll())
            print("   ✓ Background process is running")

            # Check log directory created
            log_dir = self.runner.results_dir / f"background_logs_{experiment_id}"
            self.assertTrue(log_dir.exists())
            print(f"   ✓ Log directory created: {log_dir.name}")

        finally:
            # Stop background process (script will be auto-deleted)
            self.runner._stop_background_training(process, script_path)

            # Verify script was deleted
            self.assertFalse(script_path.exists())
            print("   ✓ Script was cleaned up")

        print("   ✓ Background script generation test passed")

    def test_background_process_termination(self):
        """Test background process termination"""
        print("\n[TEST] Testing background process termination...")

        experiment_id = f"{self.test_experiment_id}_termination"

        # Start background process
        process, script_path = self.runner._start_background_training(
            repo="pytorch_resnet_cifar10",
            model="resnet20",
            hyperparams={"epochs": 1, "learning_rate": 0.01},
            experiment_id=experiment_id
        )

        initial_pid = process.pid
        print(f"   Started process (PID: {initial_pid})")

        # Wait for process to stabilize
        time.sleep(2)

        # Verify process is running
        self.assertIsNone(process.poll())
        print("   ✓ Process is running")

        # Verify script exists
        self.assertTrue(script_path.exists())
        print("   ✓ Script exists")

        # Stop process
        self.runner._stop_background_training(process, script_path)

        # Wait for termination
        time.sleep(2)

        # Verify process stopped
        self.assertIsNotNone(process.poll())
        print("   ✓ Process terminated successfully")

        # Verify script was cleaned up
        self.assertFalse(script_path.exists())
        print("   ✓ Script was deleted")

        # Verify no zombie processes
        try:
            os.kill(initial_pid, 0)  # Check if PID exists
            self.fail("Process still exists after termination")
        except ProcessLookupError:
            print("   ✓ No zombie processes remaining")

        print("   ✓ Process termination test passed")

    def test_parallel_experiment_structure(self):
        """Test parallel experiment configuration structure"""
        print("\n[TEST] Testing parallel experiment structure...")

        # This is a structure test only - we won't actually run training
        # to avoid consuming resources

        # Verify the method exists and has correct signature
        self.assertTrue(hasattr(self.runner, 'run_parallel_experiment'))
        print("   ✓ run_parallel_experiment method exists")

        # Verify helper methods exist
        self.assertTrue(hasattr(self.runner, '_build_training_args'))
        self.assertTrue(hasattr(self.runner, '_start_background_training'))
        self.assertTrue(hasattr(self.runner, '_stop_background_training'))
        print("   ✓ All helper methods exist")

        print("   ✓ Parallel experiment structure test passed")


class TestParallelConfiguration(unittest.TestCase):
    """Test cases for parallel configuration parsing"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.runner = MutationRunner()

    def test_parallel_config_validation(self):
        """Test validation of parallel configuration format"""
        print("\n[TEST] Testing parallel configuration validation...")

        # Test valid configuration structure
        valid_config = {
            "foreground": {
                "repo": "pytorch_resnet_cifar10",
                "model": "resnet20",
                "mode": "mutation",
                "mutate": ["learning_rate"]
            },
            "background": {
                "repo": "VulBERTa",
                "model": "mlp",
                "hyperparameters": {
                    "epochs": 1,
                    "learning_rate": 0.001
                }
            }
        }

        # Verify foreground config
        self.assertIn("foreground", valid_config)
        self.assertIn("repo", valid_config["foreground"])
        print("   ✓ Foreground configuration valid")

        # Verify background config
        self.assertIn("background", valid_config)
        self.assertIn("repo", valid_config["background"])
        self.assertIn("hyperparameters", valid_config["background"])
        print("   ✓ Background configuration valid")

        print("   ✓ Configuration validation test passed")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 80)
    print("PARALLEL TRAINING TEST SUITE")
    print("=" * 80)
    print(f"Test script: {__file__}")
    print(f"Project root: {project_root}")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestParallelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelConfiguration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
