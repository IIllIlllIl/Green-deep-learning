#!/usr/bin/env python3
"""
Test Suite for Script Reuse Implementation

Tests the new background training template script approach:
1. Template script exists and is executable
2. Script accepts correct parameters
3. Background process starts and stops correctly
4. Multiple runs reuse the same template
5. No temporary scripts are created/deleted
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mutation import MutationRunner


class TestScriptReuse:
    """Test suite for script reuse functionality"""

    def __init__(self):
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.results_dir = project_root / "results"
        self.template_path = self.scripts_dir / "background_training_template.sh"
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result_msg = f"{status}: {test_name}"
        if message:
            result_msg += f"\n    {message}"
        print(result_msg)

        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "message": message
        })

        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

    def test_template_exists(self) -> bool:
        """Test 1: Template script exists"""
        print("\n" + "=" * 80)
        print("TEST 1: Template Script Existence")
        print("=" * 80)

        exists = self.template_path.exists()
        is_file = self.template_path.is_file()
        is_executable = os.access(self.template_path, os.X_OK)

        self.log_test(
            "Template script exists",
            exists,
            f"Path: {self.template_path}"
        )
        self.log_test(
            "Template is a file",
            is_file,
            f"Type: {'file' if is_file else 'not a file'}"
        )
        self.log_test(
            "Template is executable",
            is_executable,
            f"Permissions: {oct(self.template_path.stat().st_mode) if exists else 'N/A'}"
        )

        return exists and is_file and is_executable

    def test_template_content(self) -> bool:
        """Test 2: Template script has correct structure"""
        print("\n" + "=" * 80)
        print("TEST 2: Template Script Content")
        print("=" * 80)

        with open(self.template_path, 'r') as f:
            content = f.read()

        # Check for essential components
        checks = [
            ("Shebang line", "#!/bin/bash" in content),
            ("Parameter validation", "if [ $# -lt 4 ]" in content),
            ("REPO_PATH variable", 'REPO_PATH="$1"' in content),
            ("TRAIN_SCRIPT variable", 'TRAIN_SCRIPT="$2"' in content),
            ("TRAIN_ARGS variable", 'TRAIN_ARGS="$3"' in content),
            ("LOG_DIR variable", 'LOG_DIR="$4"' in content),
            ("RESTART_DELAY variable", 'RESTART_DELAY="${5:-2}"' in content),
            ("Infinite loop", "while true; do" in content),
            ("Run counter", "run_count=" in content),
            ("Training execution", "$TRAIN_SCRIPT $TRAIN_ARGS" in content),
        ]

        all_passed = True
        for check_name, check_result in checks:
            self.log_test(f"Content check: {check_name}", check_result)
            all_passed = all_passed and check_result

        return all_passed

    def test_mutation_runner_integration(self) -> bool:
        """Test 3: MutationRunner uses template correctly"""
        print("\n" + "=" * 80)
        print("TEST 3: MutationRunner Integration")
        print("=" * 80)

        try:
            runner = MutationRunner()

            # Check that _start_background_training returns None for script_path
            experiment_id = f"test_{int(time.time())}"

            # Mock parameters
            repo = "VulBERTa"
            model = "mlp"
            hyperparams = {"epochs": 1, "learning_rate": 0.001}

            # Start background training
            print("  Starting background training...")
            process, script_path = runner._start_background_training(
                repo, model, hyperparams, experiment_id
            )

            # Verify process started
            process_started = process is not None and process.poll() is None
            self.log_test(
                "Background process started",
                process_started,
                f"PID: {process.pid if process else 'N/A'}"
            )

            # Verify script_path is None (template reuse)
            script_path_is_none = script_path is None
            self.log_test(
                "script_path is None (template reuse)",
                script_path_is_none,
                f"Value: {script_path}"
            )

            # Verify log directory was created
            log_dir = self.results_dir / f"background_logs_{experiment_id}"
            log_dir_exists = log_dir.exists()
            self.log_test(
                "Log directory created",
                log_dir_exists,
                f"Path: {log_dir}"
            )

            # Stop background training
            print("  Stopping background training...")
            runner._stop_background_training(process, script_path)
            time.sleep(2)  # Wait for cleanup

            # Verify process stopped
            process_stopped = process.poll() is not None
            self.log_test(
                "Background process stopped",
                process_stopped,
                f"Exit code: {process.poll()}"
            )

            # Verify no zombie processes
            try:
                os.kill(process.pid, 0)
                zombie_exists = True
            except ProcessLookupError:
                zombie_exists = False

            self.log_test(
                "No zombie process",
                not zombie_exists,
                f"Process {process.pid} {'still exists' if zombie_exists else 'cleaned up'}"
            )

            # Clean up log directory
            if log_dir.exists():
                import shutil
                shutil.rmtree(log_dir)

            return (process_started and script_path_is_none and
                    log_dir_exists and process_stopped and not zombie_exists)

        except Exception as e:
            self.log_test("Integration test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_multiple_runs_reuse_template(self) -> bool:
        """Test 4: Multiple runs reuse the same template"""
        print("\n" + "=" * 80)
        print("TEST 4: Multiple Runs Reuse Template")
        print("=" * 80)

        try:
            runner = MutationRunner()

            # Count scripts before
            scripts_before = list(self.results_dir.glob("background_training_*.sh"))
            print(f"  Scripts before: {len(scripts_before)}")

            processes = []
            experiment_ids = []

            # Start 3 background trainings
            for i in range(3):
                experiment_id = f"test_multi_{int(time.time())}_{i}"
                experiment_ids.append(experiment_id)

                print(f"\n  Starting background training #{i+1}...")
                process, script_path = runner._start_background_training(
                    "VulBERTa", "mlp",
                    {"epochs": 1, "learning_rate": 0.001},
                    experiment_id
                )

                processes.append((process, script_path, experiment_id))
                time.sleep(1)  # Stagger starts

            # Count scripts after starting
            scripts_after_start = list(self.results_dir.glob("background_training_*.sh"))
            print(f"\n  Scripts after starting 3 runs: {len(scripts_after_start)}")

            no_new_scripts = len(scripts_after_start) == len(scripts_before)
            self.log_test(
                "No new scripts created (template reuse)",
                no_new_scripts,
                f"Scripts before: {len(scripts_before)}, after: {len(scripts_after_start)}"
            )

            # Verify all processes are running
            all_running = all(p[0].poll() is None for p in processes)
            self.log_test(
                "All 3 background processes running",
                all_running,
                f"PIDs: {[p[0].pid for p in processes]}"
            )

            # Stop all processes
            print("\n  Stopping all background trainings...")
            for process, script_path, experiment_id in processes:
                runner._stop_background_training(process, script_path)

            time.sleep(2)  # Wait for cleanup

            # Verify all processes stopped
            all_stopped = all(p[0].poll() is not None for p in processes)
            self.log_test(
                "All 3 background processes stopped",
                all_stopped,
                f"Exit codes: {[p[0].poll() for p in processes]}"
            )

            # Count scripts after stopping
            scripts_after_stop = list(self.results_dir.glob("background_training_*.sh"))
            print(f"  Scripts after stopping: {len(scripts_after_stop)}")

            no_scripts_deleted = len(scripts_after_stop) == len(scripts_after_start)
            self.log_test(
                "Scripts not deleted (template preserved)",
                no_scripts_deleted,
                f"Scripts after stop: {len(scripts_after_stop)}"
            )

            # Clean up log directories
            for _, _, experiment_id in processes:
                log_dir = self.results_dir / f"background_logs_{experiment_id}"
                if log_dir.exists():
                    import shutil
                    shutil.rmtree(log_dir)

            return no_new_scripts and all_running and all_stopped and no_scripts_deleted

        except Exception as e:
            self.log_test("Multiple runs test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_template_parameter_passing(self) -> bool:
        """Test 5: Template correctly receives and uses parameters"""
        print("\n" + "=" * 80)
        print("TEST 5: Template Parameter Passing")
        print("=" * 80)

        try:
            # Create a simple test script to verify parameters
            test_script_path = self.results_dir / "test_params.sh"
            marker_file = self.results_dir / "param_test_marker.txt"

            with open(test_script_path, 'w') as f:
                f.write(f"""#!/bin/bash
# Test script to verify parameters are passed correctly
echo "Test script executed at $(date)"
echo "Arguments received: $@"
echo "Template executed successfully" > {marker_file}
exit 0
""")
            os.chmod(test_script_path, 0o755)

            # Test template with parameters
            log_dir = self.results_dir / "test_param_logs"
            log_dir.mkdir(exist_ok=True)

            print(f"  Launching template with test script: {test_script_path.name}")

            process = subprocess.Popen(
                [
                    str(self.template_path),
                    str(self.results_dir),  # repo_path
                    str(test_script_path.name),  # train_script (relative to repo_path)
                    "--epochs 5 --lr 0.01",  # train_args
                    str(log_dir),  # log_dir
                    "2"  # restart_delay (short for testing)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Wait for first execution
            print("  Waiting 5 seconds for template to execute...")
            time.sleep(5)

            # Check if marker file was created
            marker_exists = marker_file.exists()
            if marker_exists:
                with open(marker_file, 'r') as f:
                    marker_content = f.read().strip()
                    print(f"  Marker content: {marker_content}")

            self.log_test(
                "Template executed training script",
                marker_exists,
                f"Marker file: {marker_file}"
            )

            # Check if log file was created
            log_files = list(log_dir.glob("run_*.log"))
            log_created = len(log_files) > 0

            if log_created:
                # Read first log to verify
                with open(log_files[0], 'r') as f:
                    log_content = f.read()
                    print(f"  Log content preview: {log_content[:200]}...")

            self.log_test(
                "Template created log file",
                log_created,
                f"Log files: {len(log_files)}"
            )

            # Stop process
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                except:
                    pass

            # Clean up
            if test_script_path.exists():
                test_script_path.unlink()
            if marker_file.exists():
                marker_file.unlink()
            if log_dir.exists():
                import shutil
                shutil.rmtree(log_dir)

            return marker_exists and log_created

        except Exception as e:
            self.log_test("Parameter passing test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_template_location(self) -> bool:
        """Test 6: Template is in scripts/ directory"""
        print("\n" + "=" * 80)
        print("TEST 6: Template Location")
        print("=" * 80)

        in_scripts_dir = self.template_path.parent == self.scripts_dir
        self.log_test(
            "Template in scripts/ directory",
            in_scripts_dir,
            f"Location: {self.template_path.parent}"
        )

        # Verify scripts directory exists
        scripts_dir_exists = self.scripts_dir.exists()
        self.log_test(
            "scripts/ directory exists",
            scripts_dir_exists,
            f"Path: {self.scripts_dir}"
        )

        return in_scripts_dir and scripts_dir_exists

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 80)
        print("SCRIPT REUSE FUNCTIONALITY TESTS")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print(f"Template path: {self.template_path}")
        print("=" * 80)

        # Run tests
        self.test_template_location()
        self.test_template_exists()
        self.test_template_content()
        self.test_mutation_runner_integration()
        self.test_multiple_runs_reuse_template()
        self.test_template_parameter_passing()

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Tests run: {self.tests_passed + self.tests_failed}")
        print(f"Successes: {self.tests_passed}")
        print(f"Failures: {self.tests_failed}")

        if self.tests_failed == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ {self.tests_failed} test(s) failed")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['name']}")
                    if result["message"]:
                        print(f"    {result['message']}")

        print("=" * 80)

        return self.tests_failed == 0


if __name__ == "__main__":
    tester = TestScriptReuse()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
