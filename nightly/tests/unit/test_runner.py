#!/usr/bin/env python3
"""
Unit tests for mutation.runner module

Tests MutationRunner class for experiment orchestration.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from mutation.runner import MutationRunner


class TestMutationRunner(unittest.TestCase):
    """Test MutationRunner class"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.json"

        # Create minimal test configuration
        test_config = {
            "models": {
                "test_repo": {
                    "path": "repos/test_repo",
                    "train_script": "./train.sh",
                    "models": ["test_model"],
                    "supported_hyperparams": {
                        "epochs": {
                            "flag": "--epochs",
                            "type": "int",
                            "distribution": "uniform",
                            "min": 1,
                            "max": 100,
                            "default": 50
                        },
                        "learning_rate": {
                            "flag": "--lr",
                            "type": "float",
                            "distribution": "log_uniform",
                            "min": 0.0001,
                            "max": 0.1,
                            "default": 0.01
                        }
                    }
                }
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_runner_initialization(self):
        """Test MutationRunner initialization"""
        runner = MutationRunner(config_path=str(self.config_path))

        self.assertIsNotNone(runner.config)
        self.assertIsNotNone(runner.session)
        self.assertIsNotNone(runner.cmd_runner)
        self.assertIsNotNone(runner.logger)
        self.assertTrue(runner.results_dir.exists())

    def test_runner_initialization_with_seed(self):
        """Test MutationRunner initialization with random seed"""
        runner = MutationRunner(config_path=str(self.config_path), random_seed=42)

        self.assertEqual(runner.random_seed, 42)

    @patch('mutation.runner.CommandRunner')
    @patch('mutation.runner.check_training_success')
    @patch('mutation.runner.extract_performance_metrics')
    @patch('mutation.runner.parse_energy_metrics')
    def test_run_experiment_calls_build_command(
        self,
        mock_parse_energy,
        mock_extract_perf,
        mock_check_success,
        mock_cmd_runner_class
    ):
        """Test that run_experiment properly calls build_training_command_from_dir"""

        # Setup mocks
        mock_cmd_runner = MagicMock()
        mock_cmd_runner_class.return_value = mock_cmd_runner

        # Mock build_training_command_from_dir to return a command
        mock_cmd_runner.build_training_command_from_dir.return_value = [
            '/path/to/run.sh',
            'repos/test_repo',
            './train.sh',
            '/tmp/training.log',
            '/tmp/energy',
            '--epochs', '50',
            '--lr', '0.01'
        ]

        # Mock run_training_with_monitoring to return success
        mock_cmd_runner.run_training_with_monitoring.return_value = (0, 100.0, {"cpu_energy_total_joules": 5000})

        # Mock training success check
        mock_check_success.return_value = (True, "")
        mock_extract_perf.return_value = {"accuracy": 0.95}

        # Create runner
        runner = MutationRunner(config_path=str(self.config_path))
        runner.cmd_runner = mock_cmd_runner  # Replace with mock

        # Run experiment
        mutation = {"epochs": 50, "learning_rate": 0.01}
        result = runner.run_experiment("test_repo", "test_model", mutation, max_retries=0)

        # Verify build_training_command_from_dir was called
        mock_cmd_runner.build_training_command_from_dir.assert_called_once()
        call_args = mock_cmd_runner.build_training_command_from_dir.call_args

        self.assertEqual(call_args.kwargs['repo'], 'test_repo')
        self.assertEqual(call_args.kwargs['model'], 'test_model')
        self.assertEqual(call_args.kwargs['mutation'], mutation)

        # Verify run_training_with_monitoring was called with cmd parameter
        mock_cmd_runner.run_training_with_monitoring.assert_called_once()
        run_call_args = mock_cmd_runner.run_training_with_monitoring.call_args

        self.assertIn('cmd', run_call_args.kwargs)
        self.assertIn('log_file', run_call_args.kwargs)
        self.assertIn('exp_dir', run_call_args.kwargs)

        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['duration'], 100.0)

    @patch('mutation.runner.CommandRunner')
    @patch('mutation.runner.check_training_success')
    @patch('mutation.runner.extract_performance_metrics')
    def test_run_experiment_retries_on_failure(
        self,
        mock_extract_perf,
        mock_check_success,
        mock_cmd_runner_class
    ):
        """Test that run_experiment retries on failure"""

        # Setup mocks
        mock_cmd_runner = MagicMock()
        mock_cmd_runner_class.return_value = mock_cmd_runner

        mock_cmd_runner.build_training_command_from_dir.return_value = ['command']
        mock_cmd_runner.run_training_with_monitoring.return_value = (1, 50.0, {})

        # First call fails, second succeeds
        mock_check_success.side_effect = [
            (False, "Training failed"),
            (True, "")
        ]
        mock_extract_perf.return_value = {}

        # Create runner
        runner = MutationRunner(config_path=str(self.config_path))
        runner.cmd_runner = mock_cmd_runner

        # Mock sleep to speed up test
        with patch('time.sleep'):
            mutation = {"epochs": 50, "learning_rate": 0.01}
            result = runner.run_experiment("test_repo", "test_model", mutation, max_retries=1)

        # Should have called build_training_command_from_dir twice (initial + 1 retry)
        self.assertEqual(mock_cmd_runner.build_training_command_from_dir.call_count, 2)
        self.assertEqual(mock_cmd_runner.run_training_with_monitoring.call_count, 2)

        # Final result should be success
        self.assertTrue(result['success'])
        self.assertEqual(result['retries'], 1)

    def test_run_experiment_signature_bug_fix(self):
        """
        Regression test for bug: CommandRunner.run_training_with_monitoring()
        got an unexpected keyword argument 'repo'

        This test ensures that run_experiment correctly calls:
        1. build_training_command_from_dir(repo=..., model=..., mutation=..., ...)
        2. run_training_with_monitoring(cmd=..., log_file=..., exp_dir=..., ...)

        NOT: run_training_with_monitoring(repo=..., model=..., ...) which caused the bug
        """
        with patch('mutation.runner.CommandRunner') as mock_cmd_runner_class:
            mock_cmd_runner = MagicMock()
            mock_cmd_runner_class.return_value = mock_cmd_runner

            # Mock the methods
            mock_cmd_runner.build_training_command_from_dir.return_value = ['cmd']
            mock_cmd_runner.run_training_with_monitoring.return_value = (0, 100.0, {})

            with patch('mutation.runner.check_training_success', return_value=(True, "")):
                with patch('mutation.runner.extract_performance_metrics', return_value={}):
                    runner = MutationRunner(config_path=str(self.config_path))
                    runner.cmd_runner = mock_cmd_runner

                    mutation = {"epochs": 50}
                    runner.run_experiment("test_repo", "test_model", mutation, max_retries=0)

            # The key assertion: run_training_with_monitoring should be called with 'cmd', not 'repo'
            call_kwargs = mock_cmd_runner.run_training_with_monitoring.call_args.kwargs

            # These should be present
            self.assertIn('cmd', call_kwargs,
                         "run_training_with_monitoring should be called with 'cmd' parameter")
            self.assertIn('log_file', call_kwargs)
            self.assertIn('exp_dir', call_kwargs)

            # These should NOT be present (this was the bug)
            self.assertNotIn('repo', call_kwargs,
                           "run_training_with_monitoring should NOT be called with 'repo' parameter")
            self.assertNotIn('model', call_kwargs,
                           "run_training_with_monitoring should NOT be called with 'model' parameter")
            self.assertNotIn('mutation', call_kwargs,
                           "run_training_with_monitoring should NOT be called with 'mutation' parameter")

    def test_paths_are_relative_not_absolute(self):
        """
        Regression test for Bug #3: Path duplication bug

        Ensures that run_experiment generates relative paths, not absolute paths.
        This prevents path duplication in run.sh when it concatenates PROJECT_ROOT.

        Background:
        - Old code: log_file = str(exp_dir / "training.log") → absolute path
        - New code: log_file = f"results/{session_dir}/{exp_id}/training.log" → relative path
        - If absolute paths are passed to run.sh, it creates: /project//absolute/path
        """
        with patch('mutation.runner.CommandRunner') as mock_cmd_runner_class:
            mock_cmd_runner = MagicMock()
            mock_cmd_runner_class.return_value = mock_cmd_runner

            # Mock the methods
            mock_cmd_runner.build_training_command_from_dir.return_value = ['cmd']
            mock_cmd_runner.run_training_with_monitoring.return_value = (0, 100.0, {})

            with patch('mutation.runner.check_training_success', return_value=(True, "")):
                with patch('mutation.runner.extract_performance_metrics', return_value={}):
                    runner = MutationRunner(config_path=str(self.config_path))
                    runner.cmd_runner = mock_cmd_runner

                    mutation = {"epochs": 50}
                    runner.run_experiment("test_repo", "test_model", mutation, max_retries=0)

            # Get the arguments passed to build_training_command_from_dir
            call_kwargs = mock_cmd_runner.build_training_command_from_dir.call_args.kwargs

            log_file = call_kwargs.get('log_file')
            energy_dir = call_kwargs.get('energy_dir')

            # Critical assertions: paths must be relative, not absolute
            self.assertIsNotNone(log_file, "log_file should be provided")
            self.assertIsNotNone(energy_dir, "energy_dir should be provided")

            # Check that paths are relative (don't start with /)
            self.assertFalse(
                log_file.startswith('/'),
                f"log_file should be relative path, not absolute. Got: {log_file}"
            )
            self.assertFalse(
                str(energy_dir).startswith('/'),
                f"energy_dir should be relative path, not absolute. Got: {energy_dir}"
            )

            # Verify paths start with "results/" (expected format)
            self.assertTrue(
                log_file.startswith('results/'),
                f"log_file should start with 'results/'. Got: {log_file}"
            )
            self.assertTrue(
                str(energy_dir).startswith('results/'),
                f"energy_dir should start with 'results/'. Got: {energy_dir}"
            )

            # Verify no path duplication patterns (no double slashes)
            self.assertNotIn('//', log_file, "log_file should not contain '//' (path duplication)")
            self.assertNotIn('//', str(energy_dir), "energy_dir should not contain '//' (path duplication)")


if __name__ == "__main__":
    unittest.main()
