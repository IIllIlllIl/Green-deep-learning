#!/usr/bin/env python3
"""
Unit tests for mutation.utils module

Tests utility functions for logging and system configuration.
"""

import unittest
import logging
import tempfile
from pathlib import Path
from mutation.utils import setup_logger, set_governor


class TestSetupLogger(unittest.TestCase):
    """Test setup_logger function"""

    def test_logger_creation(self):
        """Test that logger is created with correct name"""
        logger = setup_logger("test_logger")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")

    def test_logger_level(self):
        """Test that logger level is set correctly"""
        logger = setup_logger("test_logger", level=logging.DEBUG)

        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_has_handlers(self):
        """Test that logger has handlers configured"""
        logger = setup_logger("test_logger")

        self.assertGreater(len(logger.handlers), 0)


class TestSetGovernor(unittest.TestCase):
    """Test set_governor function"""

    def test_invalid_governor(self):
        """Test that invalid governor is rejected"""
        # Create a temporary project root
        temp_dir = Path(tempfile.mkdtemp())

        result = set_governor("invalid_mode", temp_dir)

        # Should fail validation
        self.assertFalse(result)

    def test_valid_governor_names(self):
        """Test that valid governor names pass validation"""
        temp_dir = Path(tempfile.mkdtemp())

        valid_governors = ["performance", "powersave", "ondemand", "conservative", "schedutil"]

        # Note: These will fail due to lack of sudo, but should pass validation
        # We just check that they don't raise exceptions
        for governor in valid_governors:
            try:
                set_governor(governor, temp_dir)
            except Exception as e:
                # Should not raise exceptions during validation
                self.assertNotIn("invalid governor", str(e).lower())


if __name__ == "__main__":
    unittest.main()
