#!/usr/bin/env python3
"""
Unit tests for mutation.energy module

Tests energy and performance metric parsing functions.
"""

import unittest
import tempfile
import shutil
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from mutation.energy import (
    check_training_success,
    extract_performance_metrics,
    parse_energy_metrics,
    _parse_csv_metric_streaming
)


class TestCheckTrainingSuccess(unittest.TestCase):
    """Test check_training_success function"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "training.log"

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_function_signature(self):
        """Test that check_training_success has correct signature"""
        import inspect
        sig = inspect.signature(check_training_success)
        params = list(sig.parameters.keys())

        # Verify parameter names
        self.assertEqual(params[0], 'log_file')
        self.assertEqual(params[1], 'repo')
        self.assertEqual(params[2], 'min_log_file_size_bytes')
        self.assertEqual(params[3], 'logger')

        # Verify parameter count (should be 4)
        self.assertEqual(len(params), 4)

    def test_log_file_not_found(self):
        """Test behavior when log file doesn't exist"""
        success, error_msg = check_training_success(
            log_file=str(self.temp_dir / "nonexistent.log"),
            repo="test_repo",
            min_log_file_size_bytes=1000,
            logger=None
        )

        self.assertFalse(success)
        self.assertEqual(error_msg, "Log file not found")

    def test_success_pattern_detected(self):
        """Test detection of training success patterns"""
        # Write log with success indicator
        with open(self.log_file, 'w') as f:
            f.write("Some training output\n")
            f.write("Training completed successfully\n")
            f.write("More output\n")

        success, error_msg = check_training_success(
            log_file=str(self.log_file),
            repo="test_repo",
            min_log_file_size_bytes=10,
            logger=None
        )

        self.assertTrue(success)
        self.assertIn("successfully", error_msg.lower())

    def test_error_pattern_detected(self):
        """Test detection of error patterns"""
        # Write log with error
        with open(self.log_file, 'w') as f:
            f.write("Some training output\n")
            f.write("RuntimeError: CUDA out of memory\n")
            f.write("More output\n" * 100)  # Make it larger than min size

        success, error_msg = check_training_success(
            log_file=str(self.log_file),
            repo="test_repo",
            min_log_file_size_bytes=10,
            logger=None
        )

        self.assertFalse(success)
        self.assertIn("Error pattern found", error_msg)

    def test_log_file_too_small(self):
        """Test detection of too-small log files"""
        # Write very small log file
        with open(self.log_file, 'w') as f:
            f.write("Small\n")

        success, error_msg = check_training_success(
            log_file=str(self.log_file),
            repo="test_repo",
            min_log_file_size_bytes=1000,
            logger=None
        )

        self.assertFalse(success)
        self.assertIn("too small", error_msg.lower())

    def test_with_custom_logger(self):
        """Test that custom logger is used correctly"""
        mock_logger = Mock()

        # Create log file that will trigger warning
        success, error_msg = check_training_success(
            log_file=str(self.temp_dir / "nonexistent.log"),
            repo="test_repo",
            min_log_file_size_bytes=1000,
            logger=mock_logger
        )

        # Verify logger.warning was called
        mock_logger.warning.assert_called()


class TestExtractPerformanceMetrics(unittest.TestCase):
    """Test extract_performance_metrics function"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "training.log"

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_function_signature(self):
        """Test that extract_performance_metrics has correct signature"""
        import inspect
        sig = inspect.signature(extract_performance_metrics)
        params = list(sig.parameters.keys())

        # Verify parameter names
        self.assertEqual(params[0], 'log_file')
        self.assertEqual(params[1], 'repo')
        self.assertEqual(params[2], 'log_patterns')
        self.assertEqual(params[3], 'logger')

        # Verify parameter count (should be 4)
        self.assertEqual(len(params), 4)

    def test_log_file_not_found(self):
        """Test behavior when log file doesn't exist"""
        log_patterns = {"accuracy": r"Accuracy:\s*([0-9.]+)"}

        metrics = extract_performance_metrics(
            log_file=str(self.temp_dir / "nonexistent.log"),
            repo="test_repo",
            log_patterns=log_patterns,
            logger=None
        )

        self.assertEqual(metrics, {})

    def test_metric_extraction(self):
        """Test successful metric extraction"""
        # Write log with metrics
        with open(self.log_file, 'w') as f:
            f.write("Training epoch 1\n")
            f.write("Accuracy: 0.85\n")
            f.write("Training epoch 2\n")
            f.write("Accuracy: 0.92\n")

        log_patterns = {"accuracy": r"Accuracy:\s*([0-9.]+)"}

        metrics = extract_performance_metrics(
            log_file=str(self.log_file),
            repo="test_repo",
            log_patterns=log_patterns,
            logger=None
        )

        self.assertIn("accuracy", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 0.92)  # Should get last match

    def test_multiple_metrics(self):
        """Test extraction of multiple metrics"""
        # Write log with multiple metrics
        with open(self.log_file, 'w') as f:
            f.write("Results:\n")
            f.write("Accuracy: 0.95\n")
            f.write("Loss: 0.123\n")
            f.write("mAP: 0.88\n")

        log_patterns = {
            "accuracy": r"Accuracy:\s*([0-9.]+)",
            "loss": r"Loss:\s*([0-9.]+)",
            "mAP": r"mAP:\s*([0-9.]+)"
        }

        metrics = extract_performance_metrics(
            log_file=str(self.log_file),
            repo="test_repo",
            log_patterns=log_patterns,
            logger=None
        )

        self.assertEqual(len(metrics), 3)
        self.assertAlmostEqual(metrics["accuracy"], 0.95)
        self.assertAlmostEqual(metrics["loss"], 0.123)
        self.assertAlmostEqual(metrics["mAP"], 0.88)

    def test_no_patterns(self):
        """Test behavior when no patterns provided"""
        with open(self.log_file, 'w') as f:
            f.write("Some log content\n")

        metrics = extract_performance_metrics(
            log_file=str(self.log_file),
            repo="test_repo",
            log_patterns={},
            logger=None
        )

        self.assertEqual(metrics, {})

    def test_pattern_not_found(self):
        """Test behavior when pattern doesn't match"""
        with open(self.log_file, 'w') as f:
            f.write("Some log content without metrics\n")

        log_patterns = {"accuracy": r"Accuracy:\s*([0-9.]+)"}

        metrics = extract_performance_metrics(
            log_file=str(self.log_file),
            repo="test_repo",
            log_patterns=log_patterns,
            logger=None
        )

        self.assertEqual(metrics, {})


class TestParseEnergyMetrics(unittest.TestCase):
    """Test parse_energy_metrics function"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.energy_dir = self.temp_dir / "energy"
        self.energy_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_function_signature(self):
        """Test that parse_energy_metrics has correct signature"""
        import inspect
        sig = inspect.signature(parse_energy_metrics)
        params = list(sig.parameters.keys())

        # Verify parameter names
        self.assertEqual(params[0], 'energy_dir')
        self.assertEqual(params[1], 'logger')

        # Verify parameter count (should be 2)
        self.assertEqual(len(params), 2)

    def test_empty_directory(self):
        """Test behavior with empty energy directory"""
        metrics = parse_energy_metrics(
            energy_dir=self.energy_dir,
            logger=None
        )

        # Should return dictionary with None values
        self.assertIsInstance(metrics, dict)
        self.assertIsNone(metrics["cpu_energy_total_joules"])
        self.assertIsNone(metrics["gpu_power_avg_watts"])

    def test_cpu_energy_parsing(self):
        """Test CPU energy file parsing"""
        cpu_file = self.energy_dir / "cpu_energy.txt"
        with open(cpu_file, 'w') as f:
            f.write("Package Energy: 80000.50 J\n")
            f.write("RAM Energy: 5000.25 J\n")
            f.write("Total CPU Energy: 85000.75 J\n")

        metrics = parse_energy_metrics(
            energy_dir=self.energy_dir,
            logger=None
        )

        self.assertAlmostEqual(metrics["cpu_energy_pkg_joules"], 80000.50)
        self.assertAlmostEqual(metrics["cpu_energy_ram_joules"], 5000.25)
        self.assertAlmostEqual(metrics["cpu_energy_total_joules"], 85000.75)

    def test_gpu_power_csv_parsing(self):
        """Test GPU power CSV parsing"""
        gpu_power_file = self.energy_dir / "gpu_power.csv"
        with open(gpu_power_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'power_draw_w'])
            writer.writerow(['2024-01-01 00:00:00', '250.5'])
            writer.writerow(['2024-01-01 00:00:01', '260.0'])
            writer.writerow(['2024-01-01 00:00:02', '255.2'])

        metrics = parse_energy_metrics(
            energy_dir=self.energy_dir,
            logger=None
        )

        # Check average, max, min
        self.assertIsNotNone(metrics["gpu_power_avg_watts"])
        self.assertAlmostEqual(metrics["gpu_power_max_watts"], 260.0)
        self.assertAlmostEqual(metrics["gpu_power_min_watts"], 250.5)


class TestParseCsvMetricStreaming(unittest.TestCase):
    """Test _parse_csv_metric_streaming function"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_file_not_found(self):
        """Test behavior when CSV file doesn't exist"""
        csv_file = self.temp_dir / "nonexistent.csv"

        stats = _parse_csv_metric_streaming(
            csv_file=csv_file,
            field_name='value',
            logger=None
        )

        # Should return empty stats
        self.assertIsNone(stats['avg'])
        self.assertIsNone(stats['max'])
        self.assertIsNone(stats['min'])
        self.assertIsNone(stats['sum'])

    def test_valid_csv_parsing(self):
        """Test parsing of valid CSV file"""
        csv_file = self.temp_dir / "metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'value'])
            writer.writerow(['t1', '100'])
            writer.writerow(['t2', '200'])
            writer.writerow(['t3', '300'])

        stats = _parse_csv_metric_streaming(
            csv_file=csv_file,
            field_name='value',
            logger=None
        )

        self.assertAlmostEqual(stats['avg'], 200.0)
        self.assertAlmostEqual(stats['max'], 300.0)
        self.assertAlmostEqual(stats['min'], 100.0)
        self.assertAlmostEqual(stats['sum'], 600.0)

    def test_field_not_found(self):
        """Test behavior when field doesn't exist in CSV"""
        csv_file = self.temp_dir / "metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'other_field'])
            writer.writerow(['t1', '100'])

        stats = _parse_csv_metric_streaming(
            csv_file=csv_file,
            field_name='nonexistent_field',
            logger=None
        )

        # Should return empty stats
        self.assertIsNone(stats['avg'])


class TestEnergyFunctionIntegration(unittest.TestCase):
    """Integration tests for energy module functions with runner"""

    def test_check_training_success_integration(self):
        """Test that check_training_success can be called from runner context"""
        # This test simulates how runner.py should call check_training_success
        temp_dir = Path(tempfile.mkdtemp())
        log_file = temp_dir / "training.log"

        try:
            # Create a successful training log
            with open(log_file, 'w') as f:
                f.write("Training output\n" * 100)
                f.write("Training completed successfully\n")

            # Simulate runner.py call
            MIN_LOG_FILE_SIZE_BYTES = 1000
            mock_logger = Mock()

            success, error_msg = check_training_success(
                log_file=str(log_file),
                repo="pytorch_resnet_cifar10",
                min_log_file_size_bytes=MIN_LOG_FILE_SIZE_BYTES,
                logger=mock_logger
            )

            self.assertTrue(success)
        finally:
            shutil.rmtree(temp_dir)

    def test_extract_performance_metrics_integration(self):
        """Test that extract_performance_metrics can be called from runner context"""
        # This test simulates how runner.py should call extract_performance_metrics
        temp_dir = Path(tempfile.mkdtemp())
        log_file = temp_dir / "training.log"

        try:
            # Create a log with performance metrics
            with open(log_file, 'w') as f:
                f.write("Epoch 1\n")
                f.write("Test Accuracy: 0.92\n")
                f.write("Test Error: 8.0\n")

            # Simulate runner.py call
            repo_config = {
                "performance_metrics": {
                    "accuracy": r"Test Accuracy:\s*([0-9.]+)",
                    "error": r"Test Error:\s*([0-9.]+)"
                }
            }
            log_patterns = repo_config.get("performance_metrics", {})
            mock_logger = Mock()

            metrics = extract_performance_metrics(
                log_file=str(log_file),
                repo="pytorch_resnet_cifar10",
                log_patterns=log_patterns,
                logger=mock_logger
            )

            self.assertIn("accuracy", metrics)
            self.assertAlmostEqual(metrics["accuracy"], 0.92)
        finally:
            shutil.rmtree(temp_dir)

    def test_runner_calls_with_wrong_signature_should_fail(self):
        """Regression test: ensure old-style calls with wrong parameters fail"""
        temp_dir = Path(tempfile.mkdtemp())
        log_file = temp_dir / "training.log"

        try:
            with open(log_file, 'w') as f:
                f.write("Training completed successfully\n" * 100)  # Make it large enough

            # This should raise TypeError because we're passing wrong types
            # The 3rd parameter should be min_log_file_size_bytes (int), not config (dict)
            with self.assertRaises((TypeError, AttributeError)):
                # Old wrong way: passing config dict as 3rd arg instead of int
                # This will cause AttributeError when code tries to call logger.warning()
                # because the 4th arg (project_root Path) is being used as logger
                check_training_success(
                    str(log_file),
                    "repo",
                    {"models": {}},  # config dict (wrong! should be int)
                    Path(".")        # project_root Path (wrong! should be logger)
                )
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
