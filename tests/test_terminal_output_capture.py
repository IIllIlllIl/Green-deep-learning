"""
Test Terminal Output Capture Functionality

Tests the new capture_stdout parameter in CommandRunner.run_training_with_monitoring()
to ensure:
1. New functionality correctly captures and saves terminal output
2. Old functionality is not affected when capture_stdout=False
3. Timeout scenarios correctly save partial output

Created: 2025-12-12
Purpose: Verify data extraction debugging feature
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.command_runner import CommandRunner


def create_test_config() -> Dict:
    """Create minimal test configuration for CommandRunner"""
    return {
        "models": {
            "test_repo": {
                "path": ".",
                "train_script": "test_train.sh",
                "supported_hyperparams": {
                    "epochs": {
                        "flag": "--epochs",
                        "type": "int",
                        "default": 10
                    }
                }
            }
        }
    }


def create_test_training_script(script_path: Path, output_type: str = "normal"):
    """Create a simple test training script

    Args:
        script_path: Path to save the test script
        output_type: Type of output to generate
            - "normal": Print to stdout and stderr
            - "slow": Print gradually (for timeout testing)
            - "silent": No output
    """
    if output_type == "normal":
        script_content = """#!/bin/bash
echo "STDOUT: Training started"
echo "STDOUT: Epoch 1/10 - loss: 0.5, accuracy: 0.85"
echo "STDERR: Warning: some warning message" >&2
echo "STDOUT: Epoch 2/10 - loss: 0.4, accuracy: 0.87"
echo "STDOUT: Training completed"
echo "STDOUT: Test accuracy: 0.89"
exit 0
"""
    elif output_type == "slow":
        script_content = """#!/bin/bash
echo "STDOUT: Training started"
sleep 2
echo "STDOUT: Epoch 1/10"
sleep 2
echo "STDOUT: Epoch 2/10"
sleep 10
echo "STDOUT: This should not appear due to timeout"
exit 0
"""
    elif output_type == "silent":
        script_content = """#!/bin/bash
# Silent script - no output
exit 0
"""
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable


def test_capture_enabled():
    """Test 1: capture_stdout=True correctly saves output"""
    print("\n" + "="*80)
    print("TEST 1: Capture Enabled (capture_stdout=True)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test script
        script_path = tmpdir_path / "test_train.sh"
        create_test_training_script(script_path, output_type="normal")

        # Create CommandRunner
        config = create_test_config()
        runner = CommandRunner(
            project_root=tmpdir_path,
            config=config,
            logger=logging.getLogger(__name__)
        )

        # Create experiment directory
        exp_dir = tmpdir_path / "exp_test"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path
        log_file = exp_dir / "train.log"

        # Build simple command
        cmd = [str(script_path)]

        # Run with capture enabled
        exit_code, duration, energy_metrics = runner.run_training_with_monitoring(
            cmd=cmd,
            log_file=str(log_file),
            exp_dir=exp_dir,
            timeout=10,
            capture_stdout=True  # NEW FEATURE
        )

        # Verify terminal_output.txt was created
        terminal_output_file = exp_dir / "terminal_output.txt"
        assert terminal_output_file.exists(), "❌ terminal_output.txt was not created"
        print("✓ terminal_output.txt created")

        # Read and verify content
        content = terminal_output_file.read_text()

        # Check for expected sections
        assert "STDOUT:" in content, "❌ STDOUT section not found"
        assert "STDERR:" in content, "❌ STDERR section not found"
        print("✓ STDOUT and STDERR sections present")

        # Check for actual output
        assert "Training started" in content, "❌ Training output not captured"
        assert "Test accuracy: 0.89" in content, "❌ Final metrics not captured"
        assert "Warning: some warning message" in content, "❌ Stderr not captured"
        print("✓ Output content correctly captured")

        # Verify exit code
        assert exit_code == 0, f"❌ Unexpected exit code: {exit_code}"
        print("✓ Exit code correct")

        print("\n✅ TEST 1 PASSED: Capture functionality works correctly")
        return True


def test_capture_disabled():
    """Test 2: capture_stdout=False maintains original behavior"""
    print("\n" + "="*80)
    print("TEST 2: Capture Disabled (capture_stdout=False)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test script
        script_path = tmpdir_path / "test_train.sh"
        create_test_training_script(script_path, output_type="normal")

        # Create CommandRunner
        config = create_test_config()
        runner = CommandRunner(
            project_root=tmpdir_path,
            config=config,
            logger=logging.getLogger(__name__)
        )

        # Create experiment directory
        exp_dir = tmpdir_path / "exp_test"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path
        log_file = exp_dir / "train.log"

        # Build simple command
        cmd = [str(script_path)]

        # Run with capture DISABLED (old behavior)
        exit_code, duration, energy_metrics = runner.run_training_with_monitoring(
            cmd=cmd,
            log_file=str(log_file),
            exp_dir=exp_dir,
            timeout=10,
            capture_stdout=False  # OLD BEHAVIOR
        )

        # Verify terminal_output.txt was NOT created
        terminal_output_file = exp_dir / "terminal_output.txt"
        assert not terminal_output_file.exists(), "❌ terminal_output.txt should not be created when capture_stdout=False"
        print("✓ terminal_output.txt correctly not created")

        # Verify exit code still works
        assert exit_code == 0, f"❌ Unexpected exit code: {exit_code}"
        print("✓ Exit code correct")

        print("\n✅ TEST 2 PASSED: Old behavior preserved when capture disabled")
        return True


def test_timeout_capture():
    """Test 3: Timeout scenario correctly saves partial output"""
    print("\n" + "="*80)
    print("TEST 3: Timeout with Capture")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create slow test script that will timeout
        script_path = tmpdir_path / "test_train_slow.sh"
        create_test_training_script(script_path, output_type="slow")

        # Create CommandRunner
        config = create_test_config()
        runner = CommandRunner(
            project_root=tmpdir_path,
            config=config,
            logger=logging.getLogger(__name__)
        )

        # Create experiment directory
        exp_dir = tmpdir_path / "exp_test_timeout"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path
        log_file = exp_dir / "train.log"

        # Build simple command
        cmd = [str(script_path)]

        # Run with SHORT timeout (5 seconds)
        exit_code, duration, energy_metrics = runner.run_training_with_monitoring(
            cmd=cmd,
            log_file=str(log_file),
            exp_dir=exp_dir,
            timeout=5,  # Will timeout after 5 seconds
            capture_stdout=True
        )

        # Verify timeout occurred
        assert exit_code == -1, f"❌ Expected timeout exit code -1, got {exit_code}"
        print("✓ Timeout correctly detected")

        # Verify terminal_output.txt was created even with timeout
        terminal_output_file = exp_dir / "terminal_output.txt"
        assert terminal_output_file.exists(), "❌ terminal_output.txt not created on timeout"
        print("✓ terminal_output.txt created even on timeout")

        # Read and verify partial content
        content = terminal_output_file.read_text()

        # Check for timeout marker
        assert "TIMEOUT" in content or "PARTIAL OUTPUT" in content, "❌ Timeout marker not found"
        print("✓ Timeout marker present")

        # Check for partial output that should have completed before timeout
        assert "Training started" in content, "❌ Initial output not captured"
        assert "Epoch 1/10" in content, "❌ Partial progress not captured"
        print("✓ Partial output correctly captured")

        # Verify the message that appears after long sleep is NOT present
        assert "This should not appear" not in content, "❌ Output after timeout incorrectly captured"
        print("✓ Post-timeout output correctly excluded")

        print("\n✅ TEST 3 PASSED: Timeout scenario handled correctly")
        return True


def test_empty_output():
    """Test 4: Handle silent scripts with no output"""
    print("\n" + "="*80)
    print("TEST 4: Empty Output (Silent Script)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create silent test script
        script_path = tmpdir_path / "test_train_silent.sh"
        create_test_training_script(script_path, output_type="silent")

        # Create CommandRunner
        config = create_test_config()
        runner = CommandRunner(
            project_root=tmpdir_path,
            config=config,
            logger=logging.getLogger(__name__)
        )

        # Create experiment directory
        exp_dir = tmpdir_path / "exp_test_silent"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path
        log_file = exp_dir / "train.log"

        # Build simple command
        cmd = [str(script_path)]

        # Run with capture enabled
        exit_code, duration, energy_metrics = runner.run_training_with_monitoring(
            cmd=cmd,
            log_file=str(log_file),
            exp_dir=exp_dir,
            timeout=10,
            capture_stdout=True
        )

        # Verify terminal_output.txt was created
        terminal_output_file = exp_dir / "terminal_output.txt"
        assert terminal_output_file.exists(), "❌ terminal_output.txt not created for silent script"
        print("✓ terminal_output.txt created")

        # Read and verify content
        content = terminal_output_file.read_text()

        # Check for empty markers
        assert "(empty)" in content, "❌ Empty output not marked correctly"
        print("✓ Empty output correctly marked as '(empty)'")

        # Verify exit code
        assert exit_code == 0, f"❌ Unexpected exit code: {exit_code}"
        print("✓ Exit code correct")

        print("\n✅ TEST 4 PASSED: Empty output handled correctly")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TERMINAL OUTPUT CAPTURE TESTS")
    print("Testing new capture_stdout functionality")
    print("="*80)

    results = []

    try:
        results.append(("Capture Enabled", test_capture_enabled()))
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        results.append(("Capture Enabled", False))

    try:
        results.append(("Capture Disabled", test_capture_disabled()))
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        results.append(("Capture Disabled", False))

    try:
        results.append(("Timeout Capture", test_timeout_capture()))
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        results.append(("Timeout Capture", False))

    try:
        results.append(("Empty Output", test_empty_output()))
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        results.append(("Empty Output", False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - New functionality verified!")
        print("✓ capture_stdout=True correctly captures output")
        print("✓ capture_stdout=False preserves old behavior")
        print("✓ Timeout scenarios handled correctly")
        print("✓ Empty output handled gracefully")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Please review errors above")
        return 1


if __name__ == "__main__":
    exit(main())
