#!/usr/bin/env python3
"""
Test script to verify parallel experiment directory structure fix

This test verifies that:
1. Parallel experiments create a directory with '_parallel' suffix
2. Foreground training results are saved in the parallel directory
3. No separate sequential directory is created for the foreground training
4. All expected files (training.log, experiment.json, energy data) are in the parallel directory

Author: Claude Code
Date: 2025-11-18
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from mutation.runner import MutationRunner


def cleanup_test_results(results_dir: Path) -> None:
    """Remove test results directory if it exists"""
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"✓ Cleaned up test results: {results_dir}")


def verify_parallel_directory_structure(session_dir: Path, expected_parallel_count: int) -> Tuple[bool, str]:
    """Verify parallel experiment directory structure

    Args:
        session_dir: Path to the session directory
        expected_parallel_count: Expected number of parallel experiment directories

    Returns:
        (success, message): Success status and description message
    """
    # Find all directories in the session
    all_dirs = [d for d in session_dir.iterdir() if d.is_dir()]

    # Find parallel directories (ending with '_parallel')
    parallel_dirs = [d for d in all_dirs if d.name.endswith('_parallel')]

    print(f"\n{'─' * 60}")
    print(f"Directory Structure Verification")
    print(f"{'─' * 60}")
    print(f"Session directory: {session_dir}")
    print(f"Total directories: {len(all_dirs)}")
    print(f"Parallel directories: {len(parallel_dirs)}")
    print(f"Expected parallel directories: {expected_parallel_count}")

    # Check 1: Correct number of parallel directories
    if len(parallel_dirs) != expected_parallel_count:
        return False, f"Expected {expected_parallel_count} parallel directories, found {len(parallel_dirs)}"

    # Check 2: No extra sequential directories should be created for parallel experiments
    # Count experiment IDs from directory names
    experiment_ids = set()
    for d in all_dirs:
        # Extract base ID (remove _parallel suffix if present)
        base_name = d.name.replace('_parallel', '')
        # Extract experiment ID (e.g., "001" from "repo_model_001")
        parts = base_name.split('_')
        if parts and parts[-1].isdigit():
            experiment_ids.add(int(parts[-1]))

    # If we have N parallel experiments, we should have exactly N experiment IDs
    if len(experiment_ids) != expected_parallel_count:
        return False, f"Expected {expected_parallel_count} unique experiment IDs, found {len(experiment_ids)}: {sorted(experiment_ids)}"

    print(f"✓ Correct number of parallel directories: {len(parallel_dirs)}")
    print(f"✓ No duplicate sequential directories created")

    # Check 3: Verify each parallel directory has the required files
    print(f"\n{'─' * 60}")
    print(f"Verifying parallel directory contents...")
    print(f"{'─' * 60}")

    for parallel_dir in parallel_dirs:
        print(f"\nDirectory: {parallel_dir.name}")

        # Check for required files and directories
        training_log = parallel_dir / "training.log"
        experiment_json = parallel_dir / "experiment.json"
        energy_dir = parallel_dir / "energy"
        background_logs_dir = parallel_dir / "background_logs"

        # Verify training.log exists
        if not training_log.exists():
            return False, f"Missing training.log in {parallel_dir.name}"
        print(f"  ✓ training.log exists ({training_log.stat().st_size} bytes)")

        # Verify experiment.json exists
        if not experiment_json.exists():
            return False, f"Missing experiment.json in {parallel_dir.name}"
        print(f"  ✓ experiment.json exists")

        # Verify experiment.json has correct data
        with open(experiment_json) as f:
            exp_data = json.load(f)
            if not exp_data.get("experiment_id"):
                return False, f"experiment.json missing experiment_id in {parallel_dir.name}"
            print(f"  ✓ experiment.json has valid data (experiment_id: {exp_data['experiment_id']})")

        # Verify energy directory exists
        if not energy_dir.exists():
            return False, f"Missing energy directory in {parallel_dir.name}"
        print(f"  ✓ energy/ directory exists")

        # Verify background_logs directory exists
        if not background_logs_dir.exists():
            return False, f"Missing background_logs directory in {parallel_dir.name}"
        print(f"  ✓ background_logs/ directory exists")

        # Count background log files
        bg_logs = list(background_logs_dir.glob("*.log"))
        print(f"  ✓ {len(bg_logs)} background log files")

    print(f"\n{'─' * 60}")
    print(f"✅ All parallel directories have correct structure")
    print(f"{'─' * 60}")

    return True, "All checks passed"


def create_minimal_parallel_config(config_path: Path) -> None:
    """Create a minimal parallel experiment configuration for testing

    Uses the fastest training examples:
    - Foreground: examples/mnist_ff (very fast, ~7 seconds)
    - Background: examples/mnist_ff (same, for simplicity)
    """
    config = {
        "experiment_name": "parallel_fix_test",
        "description": "Minimal test to verify parallel experiment directory fix",
        "governor": "performance",
        "runs_per_config": 2,  # Test with 2 parallel experiments
        "max_retries": 0,  # No retries for quick testing
        "mode": "parallel",
        "experiments": [
            {
                "mode": "parallel",
                "foreground": {
                    "repo": "examples",
                    "model": "mnist_ff",
                    "mode": "default",
                    "hyperparameters": {
                        "epochs": 1,  # Minimal epochs for speed
                        "batch_size": 32,
                        "learning_rate": 0.01,
                        "seed": 1
                    }
                },
                "background": {
                    "repo": "examples",
                    "model": "mnist_ff",
                    "hyperparameters": {
                        "epochs": 10,  # Background runs longer
                        "batch_size": 32,
                        "learning_rate": 0.01,
                        "seed": 1
                    }
                }
            }
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created test configuration: {config_path}")


def test_parallel_experiment_fix():
    """Test the parallel experiment directory structure fix"""

    print("\n" + "=" * 60)
    print("PARALLEL EXPERIMENT FIX TEST")
    print("=" * 60)

    # Setup
    test_results_dir = project_root / "results_test_parallel_fix"
    test_config_path = project_root / "settings" / "test_parallel_fix.json"

    try:
        # Clean up any previous test results
        cleanup_test_results(test_results_dir)

        # Create test configuration
        test_config_path.parent.mkdir(exist_ok=True)
        create_minimal_parallel_config(test_config_path)

        # Override results directory for testing
        print(f"\n{'─' * 60}")
        print(f"Running parallel experiments...")
        print(f"{'─' * 60}")

        # Initialize runner with test results directory
        runner = MutationRunner()
        runner.results_dir = test_results_dir
        runner.session = runner.session.__class__(test_results_dir)  # Re-create session with test dir

        # Run experiments from config
        runner.run_from_experiment_config(str(test_config_path.relative_to(project_root)))

        # Find the session directory (should be only one)
        session_dirs = [d for d in test_results_dir.iterdir() if d.is_dir()]
        if len(session_dirs) != 1:
            print(f"❌ Expected 1 session directory, found {len(session_dirs)}")
            return False

        session_dir = session_dirs[0]
        print(f"\n✓ Session directory created: {session_dir.name}")

        # Verify directory structure
        success, message = verify_parallel_directory_structure(session_dir, expected_parallel_count=2)

        if success:
            print(f"\n{'=' * 60}")
            print(f"✅ TEST PASSED")
            print(f"{'=' * 60}")
            print(f"Result: {message}")
            print(f"\nThe fix successfully ensures that:")
            print(f"  1. Parallel experiments create directories with '_parallel' suffix")
            print(f"  2. Foreground results are saved in the parallel directory")
            print(f"  3. No duplicate sequential directories are created")
            print(f"  4. All required files (training.log, experiment.json, energy/) exist")
            print(f"{'=' * 60}")
            return True
        else:
            print(f"\n{'=' * 60}")
            print(f"❌ TEST FAILED")
            print(f"{'=' * 60}")
            print(f"Error: {message}")
            print(f"{'=' * 60}")
            return False

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"❌ TEST ERROR")
        print(f"{'=' * 60}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'=' * 60}")
        return False

    finally:
        # Cleanup
        if test_config_path.exists():
            test_config_path.unlink()
            print(f"\n✓ Cleaned up test config: {test_config_path}")

        # Note: Keep test results for inspection
        print(f"\nℹ️  Test results preserved for inspection: {test_results_dir}")
        print(f"   Run 'rm -rf {test_results_dir}' to clean up manually")


if __name__ == "__main__":
    success = test_parallel_experiment_fix()
    sys.exit(0 if success else 1)
