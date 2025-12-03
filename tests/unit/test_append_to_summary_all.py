#!/usr/bin/env python3
"""
Unit tests for _append_to_summary_all() method

Verifies that the CSV append logic:
1. Correctly appends session data to summary_all.csv
2. Preserves existing data in summary_all.csv
3. Handles first-time file creation (writes header)
4. Handles empty session data gracefully
5. Maintains data integrity across multiple appends
"""

import csv
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestAppendToSummaryAll:
    """Test suite for _append_to_summary_all() method"""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_runner(self, temp_results_dir):
        """Create mock MutationRunner instance"""
        # Import inside fixture to avoid circular imports
        from mutation.runner import MutationRunner

        # Create a mock runner with minimal initialization
        runner = Mock(spec=MutationRunner)
        runner.results_dir = temp_results_dir
        runner.append_to_summary = True
        runner.logger = Mock()

        # Create mock session with session_dir
        runner.session = Mock()
        runner.session.session_dir = temp_results_dir / "run_test_001"
        runner.session.session_dir.mkdir(parents=True, exist_ok=True)

        # Bind the real method to the mock
        from mutation.runner import MutationRunner
        runner._append_to_summary_all = MutationRunner._append_to_summary_all.__get__(runner)

        return runner

    @pytest.fixture
    def sample_session_data(self):
        """Sample session CSV data"""
        return [
            {
                "experiment_id": "test_exp_001",
                "timestamp": "2025-12-02T10:00:00",
                "repository": "examples",
                "model": "mnist",
                "training_success": "True",
                "duration_seconds": "100.5",
                "energy_cpu_total_joules": "5000.0",
                "energy_gpu_total_joules": "10000.0"
            },
            {
                "experiment_id": "test_exp_002",
                "timestamp": "2025-12-02T10:05:00",
                "repository": "examples",
                "model": "mnist_rnn",
                "training_success": "True",
                "duration_seconds": "150.2",
                "energy_cpu_total_joules": "6000.0",
                "energy_gpu_total_joules": "12000.0"
            }
        ]

    def create_session_summary(self, session_dir, data):
        """Helper to create session summary.csv"""
        summary_file = session_dir / "summary.csv"
        with open(summary_file, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        return summary_file

    def read_csv_data(self, csv_path):
        """Helper to read CSV data"""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def test_append_to_new_file(self, mock_runner, sample_session_data):
        """Test appending to a new (non-existent) summary_all.csv"""
        # Create session summary
        self.create_session_summary(mock_runner.session.session_dir, sample_session_data)

        # Run append
        mock_runner._append_to_summary_all()

        # Verify summary_all.csv was created
        summary_all = mock_runner.results_dir / "summary_all.csv"
        assert summary_all.exists(), "summary_all.csv should be created"

        # Verify data
        data = self.read_csv_data(summary_all)
        assert len(data) == 2, "Should have 2 rows"
        assert data[0]["experiment_id"] == "test_exp_001"
        assert data[1]["experiment_id"] == "test_exp_002"

    def test_append_preserves_existing_data(self, mock_runner, sample_session_data):
        """Test that appending preserves existing data in summary_all.csv"""
        summary_all = mock_runner.results_dir / "summary_all.csv"

        # Create existing data in summary_all.csv
        existing_data = [
            {
                "experiment_id": "existing_001",
                "timestamp": "2025-12-01T10:00:00",
                "repository": "examples",
                "model": "siamese",
                "training_success": "True",
                "duration_seconds": "200.0",
                "energy_cpu_total_joules": "7000.0",
                "energy_gpu_total_joules": "14000.0"
            }
        ]

        with open(summary_all, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=existing_data[0].keys())
            writer.writeheader()
            writer.writerows(existing_data)

        # Create session summary with new data
        self.create_session_summary(mock_runner.session.session_dir, sample_session_data)

        # Run append
        mock_runner._append_to_summary_all()

        # Verify data
        data = self.read_csv_data(summary_all)
        assert len(data) == 3, "Should have 3 rows (1 existing + 2 new)"
        assert data[0]["experiment_id"] == "existing_001", "Existing data should be preserved"
        assert data[1]["experiment_id"] == "test_exp_001"
        assert data[2]["experiment_id"] == "test_exp_002"

    def test_append_to_empty_file(self, mock_runner, sample_session_data):
        """Test appending to an empty summary_all.csv (size 0)"""
        summary_all = mock_runner.results_dir / "summary_all.csv"

        # Create empty file
        summary_all.touch()
        assert summary_all.stat().st_size == 0, "File should be empty"

        # Create session summary
        self.create_session_summary(mock_runner.session.session_dir, sample_session_data)

        # Run append
        mock_runner._append_to_summary_all()

        # Verify data
        data = self.read_csv_data(summary_all)
        assert len(data) == 2, "Should have 2 rows"
        assert data[0]["experiment_id"] == "test_exp_001"

    def test_skip_when_session_summary_missing(self, mock_runner, capsys):
        """Test that append is skipped when session summary.csv doesn't exist"""
        # Don't create session summary

        # Run append
        mock_runner._append_to_summary_all()

        # Verify no file was created
        summary_all = mock_runner.results_dir / "summary_all.csv"
        assert not summary_all.exists(), "summary_all.csv should not be created"

        # Verify warning message
        captured = capsys.readouterr()
        assert "Session summary not found" in captured.out

    def test_skip_when_append_to_summary_false(self, mock_runner, sample_session_data):
        """Test that append is skipped when append_to_summary is False"""
        mock_runner.append_to_summary = False

        # Create session summary
        self.create_session_summary(mock_runner.session.session_dir, sample_session_data)

        # Run append
        mock_runner._append_to_summary_all()

        # Verify no file was created
        summary_all = mock_runner.results_dir / "summary_all.csv"
        assert not summary_all.exists(), "summary_all.csv should not be created when append_to_summary=False"

    def test_skip_when_session_data_empty(self, mock_runner, capsys):
        """Test that append is skipped when session has no data"""
        # Create empty session summary (header only)
        summary_file = mock_runner.session.session_dir / "summary.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "timestamp"])
            writer.writeheader()
            # No rows

        # Run append
        mock_runner._append_to_summary_all()

        # Verify warning message
        captured = capsys.readouterr()
        assert "No data in session summary" in captured.out

    def test_multiple_appends_accumulate_data(self, mock_runner):
        """Test that multiple appends correctly accumulate data"""
        summary_all = mock_runner.results_dir / "summary_all.csv"

        # First append
        data1 = [
            {
                "experiment_id": "batch1_001",
                "timestamp": "2025-12-02T10:00:00",
                "repository": "examples",
                "model": "mnist"
            }
        ]
        self.create_session_summary(mock_runner.session.session_dir, data1)
        mock_runner._append_to_summary_all()

        # Simulate new session
        mock_runner.session.session_dir = mock_runner.results_dir / "run_test_002"
        mock_runner.session.session_dir.mkdir(parents=True, exist_ok=True)

        # Second append
        data2 = [
            {
                "experiment_id": "batch2_001",
                "timestamp": "2025-12-02T11:00:00",
                "repository": "examples",
                "model": "mnist_rnn"
            }
        ]
        self.create_session_summary(mock_runner.session.session_dir, data2)
        mock_runner._append_to_summary_all()

        # Third append
        mock_runner.session.session_dir = mock_runner.results_dir / "run_test_003"
        mock_runner.session.session_dir.mkdir(parents=True, exist_ok=True)
        data3 = [
            {
                "experiment_id": "batch3_001",
                "timestamp": "2025-12-02T12:00:00",
                "repository": "examples",
                "model": "siamese"
            }
        ]
        self.create_session_summary(mock_runner.session.session_dir, data3)
        mock_runner._append_to_summary_all()

        # Verify all data accumulated
        data = self.read_csv_data(summary_all)
        assert len(data) == 3, "Should have 3 rows from 3 appends"
        assert data[0]["experiment_id"] == "batch1_001"
        assert data[1]["experiment_id"] == "batch2_001"
        assert data[2]["experiment_id"] == "batch3_001"

    def test_handles_different_column_orders(self, mock_runner):
        """Test that append handles different column orders correctly"""
        summary_all = mock_runner.results_dir / "summary_all.csv"

        # Create existing data with column order A
        existing_data = [
            {
                "experiment_id": "existing_001",
                "timestamp": "2025-12-01T10:00:00",
                "repository": "examples",
                "model": "mnist"
            }
        ]
        with open(summary_all, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "timestamp", "repository", "model"])
            writer.writeheader()
            writer.writerows(existing_data)

        # Create session data with column order B (different order)
        session_data = [
            {
                "model": "mnist_rnn",
                "timestamp": "2025-12-02T10:00:00",
                "experiment_id": "new_001",
                "repository": "examples"
            }
        ]
        self.create_session_summary(mock_runner.session.session_dir, session_data)

        # Run append
        mock_runner._append_to_summary_all()

        # Verify data integrity (should use session's column order)
        data = self.read_csv_data(summary_all)
        assert len(data) == 2
        assert data[1]["experiment_id"] == "new_001"

    def test_error_handling_for_corrupted_session_file(self, mock_runner, capsys):
        """Test error handling when session summary is corrupted"""
        # Create corrupted session summary (invalid CSV)
        summary_file = mock_runner.session.session_dir / "summary.csv"
        with open(summary_file, 'w') as f:
            f.write("corrupted,csv\ndata,that\"has,unmatched\"quotes\n")

        # Run append
        mock_runner._append_to_summary_all()

        # Verify error message
        captured = capsys.readouterr()
        assert "Error appending" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
