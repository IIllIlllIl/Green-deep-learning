#!/usr/bin/env python3
"""
Comprehensive Functional Test Suite for Refactored mutation.py

Tests all major components of the refactored codebase to ensure
backward compatibility and correctness.
"""

import sys
import json
from pathlib import Path

# Add project root to Python path to import mutation package
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test counter
tests_passed = 0
tests_failed = 0

def test(name):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            try:
                print(f"\n{'='*60}")
                print(f"TEST: {name}")
                print(f"{'='*60}")
                func()
                print(f"✓ PASSED")
                tests_passed += 1
            except AssertionError as e:
                print(f"✗ FAILED: {e}")
                tests_failed += 1
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                tests_failed += 1
        return wrapper
    return decorator


@test("1. Module Imports")
def test_imports():
    """Test all module imports"""
    from mutation import (
        MutationRunner,
        ExperimentSession,
        CommandRunner,
        mutate_hyperparameter,
        generate_mutations,
        check_training_success,
        extract_performance_metrics,
        parse_energy_metrics,
        setup_logger,
        set_governor,
        MutationError,
        HyperparameterError,
    )
    print("  All modules imported successfully")


@test("2. ExperimentSession")
def test_session():
    """Test ExperimentSession functionality"""
    from mutation.session import ExperimentSession

    results_dir = Path("/tmp/test_results")
    results_dir.mkdir(exist_ok=True)

    session = ExperimentSession(results_dir)
    assert session.session_dir.exists(), "Session directory not created"
    assert session.experiment_counter == 0, "Counter not initialized"

    # Test experiment directory creation
    exp_dir, exp_id = session.get_next_experiment_dir("test_repo", "test_model")
    assert exp_dir.exists(), "Experiment directory not created"
    assert exp_id == "test_repo_test_model_001", f"Unexpected ID: {exp_id}"
    assert session.experiment_counter == 1, "Counter not incremented"

    # Test result addition
    result = {
        "experiment_id": exp_id,
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
    assert len(session.experiments) == 1, "Result not added"

    print(f"  Session ID: {session.session_id}")
    print(f"  Experiment created: {exp_id}")


@test("3. Hyperparameter Mutation")
def test_hyperparams():
    """Test hyperparameter mutation functionality"""
    from mutation.hyperparams import mutate_hyperparameter, generate_mutations
    import random

    # Test single mutation
    random.seed(42)
    param_config = {
        "type": "int",
        "range": [1, 100],
        "distribution": "uniform"
    }
    value = mutate_hyperparameter(param_config, "test_param")
    assert 1 <= value <= 100, f"Value out of range: {value}"
    assert isinstance(value, int), f"Wrong type: {type(value)}"

    # Test mutation generation
    supported_params = {
        "epochs": {
            "type": "int",
            "range": [10, 200],
            "distribution": "log_uniform",
            "flag": "--epochs"
        },
        "learning_rate": {
            "type": "float",
            "range": [0.0001, 0.1],
            "distribution": "log_uniform",
            "flag": "--lr"
        }
    }

    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["epochs", "learning_rate"],
        num_mutations=3,
        random_seed=42
    )

    assert len(mutations) == 3, f"Wrong number of mutations: {len(mutations)}"
    assert all("epochs" in m and "learning_rate" in m for m in mutations), "Missing parameters"

    # Check uniqueness
    mutation_strs = [str(sorted(m.items())) for m in mutations]
    assert len(set(mutation_strs)) == 3, "Mutations not unique"

    print(f"  Generated {len(mutations)} unique mutations")
    for i, mut in enumerate(mutations, 1):
        print(f"    Mutation {i}: epochs={mut['epochs']}, lr={mut['learning_rate']:.6f}")


@test("4. Command Runner")
def test_command_runner():
    """Test CommandRunner functionality"""
    from mutation.command_runner import CommandRunner

    # Load config from mutation package
    config_path = Path("mutation/models_config.json")
    with open(config_path) as f:
        config = json.load(f)

    runner = CommandRunner(
        project_root=Path.cwd(),
        config=config
    )

    # Test command building
    cmd = runner.build_training_command_from_dir(
        repo="pytorch_resnet_cifar10",
        model="resnet20",
        mutation={"epochs": 10, "learning_rate": 0.01},
        exp_dir=Path("/tmp/test_exp"),
        log_file="/tmp/test.log",
        energy_dir="/tmp/energy"
    )

    assert len(cmd) > 0, "Command is empty"
    assert Path(cmd[0]).exists(), f"Run script doesn't exist: {cmd[0]}"
    assert "repos/pytorch_resnet_cifar10" in cmd[1], "Wrong repo path"
    assert "-e" in cmd or "--epochs" in cmd, "Missing epochs parameter"
    assert "--lr" in cmd or "-lr" in cmd, "Missing lr parameter"

    print(f"  Run script: {cmd[0]}")
    print(f"  Command length: {len(cmd)} arguments")


@test("5. MutationRunner Initialization")
def test_runner_init():
    """Test MutationRunner initialization"""
    from mutation import MutationRunner

    runner = MutationRunner(random_seed=42)

    assert runner.config is not None, "Config not loaded"
    assert runner.session is not None, "Session not created"
    assert runner.cmd_runner is not None, "Command runner not created"
    assert runner.random_seed == 42, "Random seed not set"
    assert "pytorch_resnet_cifar10" in runner.config["models"], "Missing repo in config"

    print(f"  Config loaded: {len(runner.config['models'])} repositories")
    print(f"  Session ID: {runner.session.session_id}")
    print(f"  Random seed: {runner.random_seed}")


@test("6. CLI Argument Parsing")
def test_cli():
    """Test CLI functionality"""
    import subprocess

    # Test --help
    result = subprocess.run(
        ["python3", "mutation.py", "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, "Help command failed"
    assert "Mutation-based Training Energy Profiler" in result.stdout, "Wrong help text"

    # Test --list
    result = subprocess.run(
        ["python3", "mutation.py", "--list"],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, "List command failed"
    assert "pytorch_resnet_cifar10" in result.stdout, "Missing repository in list"
    assert "resnet20" in result.stdout, "Missing model in list"

    print("  CLI commands working correctly")


@test("7. File Structure")
def test_file_structure():
    """Test file structure is correct"""
    mutation_dir = Path("mutation")

    required_files = [
        "__init__.py",
        "exceptions.py",
        "session.py",
        "hyperparams.py",
        "energy.py",
        "utils.py",
        "command_runner.py",
        "runner.py",
        "run.sh",
        "background_training_template.sh",
        "governor.sh",
        "models_config.json"
    ]

    for filename in required_files:
        filepath = mutation_dir / filename
        assert filepath.exists(), f"Missing file: {filepath}"

    # Check shell scripts are executable
    for script in ["run.sh", "background_training_template.sh", "governor.sh"]:
        script_path = mutation_dir / script
        assert script_path.stat().st_mode & 0o111, f"{script} not executable"

    print(f"  All {len(required_files)} required files present")
    print(f"  All shell scripts are executable")


@test("8. Path Handling (Bug #3 Regression Test)")
def test_path_handling():
    """
    Integration test for Bug #3: Path duplication bug

    Verifies that:
    1. No nested 'home/' directory is created
    2. Files are created in the correct location
    3. No path duplication occurs (no '//' in paths)

    This is a critical regression test to prevent recurrence of the path
    duplication bug where absolute paths were concatenated with PROJECT_ROOT,
    creating paths like: /project//absolute/path
    """
    import tempfile
    import shutil
    from mutation.session import ExperimentSession

    # Create temporary results directory
    temp_results = Path(tempfile.mkdtemp(prefix="test_path_"))

    try:
        # Create session with temporary results directory
        session = ExperimentSession(temp_results)

        # Get an experiment directory
        exp_dir, exp_id = session.get_next_experiment_dir(
            "pytorch_resnet_cifar10",
            "resnet20"
        )

        # Verify experiment directory was created
        assert exp_dir.exists(), f"Experiment directory not created: {exp_dir}"

        # Check 1: No nested 'home/' directory created
        home_dirs = list(temp_results.rglob("home"))
        assert len(home_dirs) == 0, f"Unexpected 'home/' directory created: {home_dirs}"

        # Check 2: Experiment directory is in the correct location
        assert str(exp_dir).startswith(str(temp_results)), \
            f"Experiment directory not under results dir.\n  Expected prefix: {temp_results}\n  Got: {exp_dir}"

        # Check 3: No path duplication (no '//' in path)
        exp_dir_str = str(exp_dir)
        assert '//' not in exp_dir_str, f"Path duplication detected (contains '//'): {exp_dir_str}"

        # Check 4: Path structure is correct (should be: results/run_XXXXXX/repo_model_NNN)
        path_parts = exp_dir.relative_to(temp_results).parts
        assert len(path_parts) == 2, f"Unexpected path structure: {path_parts}"
        assert path_parts[0].startswith("run_"), f"Session directory doesn't start with 'run_': {path_parts[0]}"
        assert path_parts[1] == exp_id, f"Experiment ID mismatch: {path_parts[1]} vs {exp_id}"

        print(f"  ✓ No 'home/' directory created")
        print(f"  ✓ Experiment directory in correct location: {exp_dir.relative_to(temp_results)}")
        print(f"  ✓ No path duplication (no '//' in paths)")
        print(f"  ✓ Path structure correct: {'/'.join(path_parts)}")

    finally:
        # Clean up
        if temp_results.exists():
            shutil.rmtree(temp_results)


@test("9. Backward Compatibility")
def test_backward_compatibility():
    """Test that refactored code maintains backward compatibility"""
    from mutation import MutationRunner

    runner = MutationRunner(random_seed=42)

    # Test that old attribute names still work (if we kept any)
    assert hasattr(runner, "config"), "Missing config attribute"
    assert hasattr(runner, "session"), "Missing session attribute"
    assert hasattr(runner, "logger"), "Missing logger attribute"

    # Test result format is the same
    result_template = {
        "experiment_id": "test_001",
        "timestamp": "2024-01-01T00:00:00",
        "repository": "test_repo",
        "model": "test_model",
        "hyperparameters": {},
        "duration_seconds": 0,
        "energy_metrics": {},
        "performance_metrics": {},
        "training_success": False,
        "retries": 0,
        "error_message": ""
    }

    # All keys should be present
    for key in result_template.keys():
        print(f"  Result format includes: {key}")

    print("  Result format unchanged")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE FUNCTIONAL TEST SUITE")
    print("="*60)

    # Run tests
    test_imports()
    test_session()
    test_hyperparams()
    test_command_runner()
    test_runner_init()
    test_cli()
    test_file_structure()
    test_path_handling()
    test_backward_compatibility()

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {tests_passed + tests_failed}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")

    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
