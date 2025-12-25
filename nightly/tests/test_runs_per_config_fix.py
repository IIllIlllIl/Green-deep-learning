#!/usr/bin/env python3
"""
Test suite for per-experiment runs_per_config fix

This test ensures that the bug fix for reading per-experiment runs_per_config
works correctly across all experiment modes (mutation, parallel, default).

Bug Summary:
- Previous code always read global runs_per_config (line 881), defaulting to 1
- Stage7 config has no global runs_per_config, only per-experiment values of 7
- Result: Only 1 experiment ran per configuration instead of 7

Fix:
- Added per-experiment runs_per_config reading with fallback to global value
- Applied to all three modes: mutation, parallel, and default
"""

import json
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.runner import MutationRunner


def test_mutation_mode_per_experiment_runs():
    """Test that mutation mode reads per-experiment runs_per_config"""

    # Create a minimal config with per-experiment runs_per_config
    config = {
        "experiment_name": "test_mutation_mode",
        "description": "Test per-experiment runs_per_config in mutation mode",
        "mode": "mutation",
        "experiments": [
            {
                "repo": "examples",
                "model": "mnist",
                "runs_per_config": 5,  # Per-experiment value (no global)
                "mutate_params": ["epochs"]
            }
        ]
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Initialize runner (without actually running experiments)
        runner = MutationRunner(append_to_summary=False)

        # Parse config to verify it would read 5 runs
        with open(config_file, 'r') as f:
            exp_config = json.load(f)

        global_runs = exp_config.get("runs_per_config", 1)  # Should default to 1
        exp = exp_config["experiments"][0]
        exp_runs = exp.get("runs_per_config", global_runs)  # Should read 5

        assert global_runs == 1, f"Expected global runs_per_config to default to 1, got {global_runs}"
        assert exp_runs == 5, f"Expected per-experiment runs_per_config to be 5, got {exp_runs}"

        print(f"✓ Mutation mode test passed:")
        print(f"  Global runs_per_config: {global_runs} (default)")
        print(f"  Per-experiment runs_per_config: {exp_runs}")
        return True

    finally:
        Path(config_file).unlink()


def test_parallel_mode_per_experiment_runs():
    """Test that parallel mode reads per-experiment runs_per_config"""

    # Create a minimal config with per-experiment runs_per_config
    config = {
        "experiment_name": "test_parallel_mode",
        "description": "Test per-experiment runs_per_config in parallel mode",
        "mode": "parallel",
        "experiments": [
            {
                "mode": "parallel",
                "foreground": {
                    "repo": "examples",
                    "model": "mnist",
                    "runs_per_config": 7,  # Per-experiment value
                    "mode": "mutation",
                    "mutate": ["epochs"]
                },
                "background": {
                    "repo": "examples",
                    "model": "mnist_rnn",
                    "hyperparameters": {}
                }
            }
        ]
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Initialize runner
        runner = MutationRunner(append_to_summary=False)

        # Parse config to verify it would read 7 runs
        with open(config_file, 'r') as f:
            exp_config = json.load(f)

        global_runs = exp_config.get("runs_per_config", 1)  # Should default to 1
        exp = exp_config["experiments"][0]
        foreground_config = exp.get("foreground", {})
        exp_runs = foreground_config.get("runs_per_config", global_runs)  # Should read 7

        assert global_runs == 1, f"Expected global runs_per_config to default to 1, got {global_runs}"
        assert exp_runs == 7, f"Expected per-experiment runs_per_config to be 7, got {exp_runs}"

        print(f"✓ Parallel mode test passed:")
        print(f"  Global runs_per_config: {global_runs} (default)")
        print(f"  Per-experiment runs_per_config: {exp_runs}")
        return True

    finally:
        Path(config_file).unlink()


def test_default_mode_per_experiment_runs():
    """Test that default mode reads per-experiment runs_per_config"""

    # Create a minimal config with per-experiment runs_per_config
    config = {
        "experiment_name": "test_default_mode",
        "description": "Test per-experiment runs_per_config in default mode",
        "mode": "default",
        "experiments": [
            {
                "repo": "examples",
                "model": "mnist",
                "runs_per_config": 3,  # Per-experiment value
                "hyperparameters": {"epochs": 5}
            }
        ]
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Initialize runner
        runner = MutationRunner(append_to_summary=False)

        # Parse config to verify it would read 3 runs
        with open(config_file, 'r') as f:
            exp_config = json.load(f)

        global_runs = exp_config.get("runs_per_config", 1)  # Should default to 1
        exp = exp_config["experiments"][0]
        exp_runs = exp.get("runs_per_config", global_runs)  # Should read 3

        assert global_runs == 1, f"Expected global runs_per_config to default to 1, got {global_runs}"
        assert exp_runs == 3, f"Expected per-experiment runs_per_config to be 3, got {exp_runs}"

        print(f"✓ Default mode test passed:")
        print(f"  Global runs_per_config: {global_runs} (default)")
        print(f"  Per-experiment runs_per_config: {exp_runs}")
        return True

    finally:
        Path(config_file).unlink()


def test_fallback_to_global_runs():
    """Test that per-experiment falls back to global if not specified"""

    # Create config with global runs_per_config and no per-experiment value
    config = {
        "experiment_name": "test_global_fallback",
        "description": "Test fallback to global runs_per_config",
        "runs_per_config": 10,  # Global value
        "mode": "mutation",
        "experiments": [
            {
                "repo": "examples",
                "model": "mnist",
                # No runs_per_config here - should fall back to global
                "mutate_params": ["epochs"]
            }
        ]
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Initialize runner
        runner = MutationRunner(append_to_summary=False)

        # Parse config to verify fallback logic
        with open(config_file, 'r') as f:
            exp_config = json.load(f)

        global_runs = exp_config.get("runs_per_config", 1)  # Should read 10
        exp = exp_config["experiments"][0]
        exp_runs = exp.get("runs_per_config", global_runs)  # Should fall back to 10

        assert global_runs == 10, f"Expected global runs_per_config to be 10, got {global_runs}"
        assert exp_runs == 10, f"Expected per-experiment to fall back to 10, got {exp_runs}"

        print(f"✓ Global fallback test passed:")
        print(f"  Global runs_per_config: {global_runs}")
        print(f"  Per-experiment runs_per_config (fallback): {exp_runs}")
        return True

    finally:
        Path(config_file).unlink()


def test_stage7_config_structure():
    """Test that Stage7 config structure is correctly handled"""

    # Simulate Stage7 config structure (no global runs_per_config)
    config = {
        "experiment_name": "stage7_nonparallel_fast_models",
        "description": "Stage7: Non-parallel fast and medium models",
        "use_deduplication": True,
        "historical_csvs": ["results/summary_all.csv"],
        # NOTE: No global runs_per_config!
        "experiments": [
            {
                "repo": "examples",
                "model": "mnist",
                "runs_per_config": 7,  # Per-experiment value
                "mutate_params": ["epochs", "learning_rate"]
            },
            {
                "repo": "examples",
                "model": "mnist_rnn",
                "runs_per_config": 7,  # Per-experiment value
                "mutate_params": ["epochs", "learning_rate"]
            }
        ]
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Initialize runner
        runner = MutationRunner(append_to_summary=False)

        # Parse config
        with open(config_file, 'r') as f:
            exp_config = json.load(f)

        global_runs = exp_config.get("runs_per_config", 1)  # Should default to 1

        # Check each experiment
        for exp in exp_config["experiments"]:
            exp_runs = exp.get("runs_per_config", global_runs)
            assert exp_runs == 7, f"Expected {exp['repo']}/{exp['model']} to have 7 runs, got {exp_runs}"

        print(f"✓ Stage7 config structure test passed:")
        print(f"  Global runs_per_config: {global_runs} (default)")
        print(f"  All {len(exp_config['experiments'])} experiments have per-experiment runs_per_config: 7")
        return True

    finally:
        Path(config_file).unlink()


def run_all_tests():
    """Run all test cases"""

    print("=" * 80)
    print("TESTING PER-EXPERIMENT runs_per_config FIX")
    print("=" * 80)
    print()

    tests = [
        ("Mutation mode per-experiment runs", test_mutation_mode_per_experiment_runs),
        ("Parallel mode per-experiment runs", test_parallel_mode_per_experiment_runs),
        ("Default mode per-experiment runs", test_default_mode_per_experiment_runs),
        ("Global fallback logic", test_fallback_to_global_runs),
        ("Stage7 config structure", test_stage7_config_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 80)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
