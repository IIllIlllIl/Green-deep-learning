"""
Command Construction and Subprocess Execution Module

This module provides the CommandRunner class for building training commands and
managing subprocess execution with energy monitoring. It handles both foreground
training with monitoring and background training loops for parallel experiments.

Key Features:
- Command construction with hyperparameter arguments
- Subprocess execution with timeout and error handling
- Background process management with platform-specific cleanup
- Energy monitoring integration
- Process group management (POSIX) and Windows compatibility
"""

import logging
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mutation.hyperparams import _build_hyperparam_args as build_hyperparam_args
from mutation.energy import parse_energy_metrics


# Constants
BACKGROUND_TERMINATION_TIMEOUT_SECONDS = 10


class CommandRunner:
    """Handles command construction and subprocess execution for training experiments"""

    def __init__(
        self,
        project_root: Path,
        config: Dict,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize CommandRunner

        Args:
            project_root: Root directory of the project
            config: Configuration dictionary (from models_config.json)
            logger: Logger instance for debug/error messages (optional)
        """
        self.project_root = project_root
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Track active background processes for cleanup
        self._active_background_processes: List[subprocess.Popen] = []

        # Platform detection for process group management
        self._is_posix = platform.system() != 'Windows'
        if not self._is_posix:
            self.logger.warning("Running on Windows - background process cleanup may be limited")

    def build_training_command_from_dir(
        self,
        repo: str,
        model: str,
        mutation: Dict[str, Any],
        exp_dir: Path,
        log_file: str,
        energy_dir: str
    ) -> List[str]:
        """Build training command using experiment directory structure

        Args:
            repo: Repository name
            model: Model name
            mutation: Dictionary of mutated hyperparameters
            exp_dir: Experiment directory path
            log_file: Log file path (already determined)
            energy_dir: Energy directory path (already determined)

        Returns:
            Command list for training

        Raises:
            KeyError: If repository not found in config
        """
        repo_config = self.config["models"][repo]
        repo_path = repo_config["path"]
        train_script = repo_config["train_script"]
        supported_params = repo_config["supported_hyperparams"]

        # Build command - use run.sh from mutation package
        run_script = Path(__file__).parent / "run.sh"
        cmd = [str(run_script), repo_path, train_script, log_file, energy_dir]

        # Add model flag if needed
        if "model_flag" in repo_config and model != "default":
            cmd.extend([repo_config["model_flag"], model])

        # Add required arguments
        if "required_args" in repo_config:
            for arg_name, arg_value in repo_config["required_args"].items():
                cmd.extend(arg_value.split())

        # Add mutated hyperparameters using imported function
        hyperparam_args = build_hyperparam_args(supported_params, mutation, as_list=True)
        cmd.extend(hyperparam_args)

        self.logger.debug(f"Built training command: {' '.join(cmd)}")
        return cmd

    def run_training_with_monitoring(
        self,
        cmd: List[str],
        log_file: str,
        exp_dir: Path,
        timeout: Optional[int] = None
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Run training with energy monitoring

        Note: Energy monitoring is integrated into run.sh.
        This method runs the training command and parses the results.

        Args:
            cmd: Training command (includes energy_dir in run.sh args)
            log_file: Path to log file
            exp_dir: Experiment directory (energy/ subdirectory will be used)
            timeout: Maximum training time in seconds (default: None, no limit)

        Returns:
            Tuple of (exit_code, duration_seconds, energy_metrics)
        """
        # Use canonical energy directory under exp_dir
        energy_dir = exp_dir / "energy"
        energy_dir.mkdir(exist_ok=True, parents=True)

        print(f"🚀 Starting training with integrated energy monitoring...")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Log: {log_file}")
        print(f"   Energy directory: {energy_dir}")
        if timeout is not None:
            print(f"   Timeout: {timeout}s ({timeout/3600:.1f}h)")
        else:
            print(f"   Timeout: None (no limit)")

        start_time = time.time()

        # Run training with integrated energy monitoring
        # run.sh handles both training and energy monitoring
        try:
            train_process = subprocess.run(
                cmd,
                capture_output=False,  # Output is handled by run.sh (tee to log)
                text=True,
                timeout=timeout
            )
            exit_code = train_process.returncode

        except subprocess.TimeoutExpired:
            print(f"⚠️  Warning: Training timed out after {timeout}s")
            exit_code = -1
        except Exception as e:
            print(f"❌ Error running training: {e}")
            self.logger.error(f"Training execution error: {e}", exc_info=True)
            exit_code = -1

        end_time = time.time()
        duration = end_time - start_time

        print(f"✓ Training finished in {duration:.1f}s with exit code {exit_code}")

        # Parse energy metrics from the energy directory
        # run.sh has already written the energy data
        energy_metrics = parse_energy_metrics(energy_dir, logger=self.logger)

        # Print energy summary
        if energy_metrics.get("cpu_energy_total_joules"):
            print(f"   CPU Energy: {energy_metrics['cpu_energy_total_joules']:.2f} J")
        if energy_metrics.get("gpu_energy_total_joules"):
            print(f"   GPU Energy: {energy_metrics['gpu_energy_total_joules']:.2f} J")

        return exit_code, duration, energy_metrics

    def _build_training_args(
        self,
        repo: str,
        model: str,
        hyperparams: Dict[str, Any]
    ) -> str:
        """Build training argument string for background training script

        Args:
            repo: Repository name
            model: Model name
            hyperparams: Hyperparameters dictionary

        Returns:
            String of training arguments

        Raises:
            KeyError: If repository not found in config
        """
        repo_config = self.config["models"][repo]
        supported_params = repo_config["supported_hyperparams"]

        args = []

        # Add model flag if needed
        if "model_flag" in repo_config and model != "default":
            args.extend([repo_config["model_flag"], model])

        # Add required arguments
        if "required_args" in repo_config:
            for arg_name, arg_value in repo_config["required_args"].items():
                args.extend(arg_value.split())

        # Add hyperparameters using imported function
        hyperparam_args = build_hyperparam_args(supported_params, hyperparams, as_list=True)
        args.extend(hyperparam_args)

        args_str = " ".join(args)
        self.logger.debug(f"Built training arguments: {args_str}")
        return args_str

    def start_background_training(
        self,
        repo: str,
        model: str,
        hyperparams: Dict[str, Any],
        log_dir: Path,
        restart_delay_seconds: int = 2
    ) -> Tuple[subprocess.Popen, None]:
        """Start background training loop using reusable template script

        This starts a background training process that continuously runs
        training in a loop, used for parallel experiments to provide GPU load.

        Args:
            repo: Repository name
            model: Model name
            hyperparams: Hyperparameters for background training
            log_dir: Path to directory for background training logs
            restart_delay_seconds: Delay between background training restarts

        Returns:
            Tuple of (subprocess.Popen object, None)
            Note: Returns None for script_path since we use a reusable template

        Raises:
            RuntimeError: If background training template script not found
            KeyError: If repository not found in config

        Platform Support:
            - POSIX (Linux/macOS): Uses os.setsid for process group management
            - Windows: Uses CREATE_NEW_PROCESS_GROUP flag (limited cleanup capability)
        """
        repo_config = self.config["models"][repo]
        repo_path = self.project_root / repo_config["path"]
        train_script = repo_config["train_script"]

        # Build training arguments
        train_args = self._build_training_args(repo, model, hyperparams)

        # Ensure log directory exists
        log_dir.mkdir(exist_ok=True, parents=True)

        # Use reusable template script from mutation/ directory
        template_script_path = Path(__file__).parent / "background_training_template.sh"

        if not template_script_path.exists():
            raise RuntimeError(
                f"Background training template script not found: {template_script_path}"
            )

        # Platform-specific process group management
        popen_kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if self._is_posix:
            # POSIX: Use process group for reliable cleanup
            popen_kwargs["preexec_fn"] = os.setsid
        else:
            # Windows: Use CREATE_NEW_PROCESS_GROUP (limited, but best available)
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        # Launch background process using template script with parameters
        process = subprocess.Popen(
            [
                str(template_script_path),
                str(repo_path),
                train_script,
                train_args,
                str(log_dir),
                str(restart_delay_seconds)
            ],
            **popen_kwargs
        )

        print(f"🔄 Background training started (PID: {process.pid})")
        print(f"   Template: {template_script_path.name}")
        print(f"   Repository: {repo}")
        print(f"   Model: {model}")
        print(f"   Arguments: {train_args}")
        print(f"   Log directory: {log_dir}")
        if not self._is_posix:
            print(f"   Platform: Windows (limited process cleanup)")

        # Track background process for cleanup
        self._active_background_processes.append(process)
        self.logger.info(f"Started background training process {process.pid}")

        return process, None  # Return None since we don't need to delete template

    def stop_background_training(
        self,
        process: subprocess.Popen,
        script_path: Optional[Path] = None
    ) -> None:
        """Stop background training process group

        Gracefully terminates the background training process by sending SIGTERM
        to the entire process group, with fallback to SIGKILL if needed.

        Args:
            process: Background process to terminate
            script_path: Not used (kept for backward compatibility)
                        Template script is reusable and not deleted
        """
        if process.poll() is not None:
            print("✓ Background training already stopped")
            self.logger.debug("Background training process already terminated")
            return

        try:
            # Send SIGTERM to entire process group for graceful shutdown
            print("🛑 Stopping background training...")
            if self._is_posix:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                # Windows: Just terminate the process
                process.terminate()

            # Wait for graceful termination (configurable timeout)
            process.wait(timeout=BACKGROUND_TERMINATION_TIMEOUT_SECONDS)
            print("✓ Background training stopped gracefully")
            self.logger.info(f"Background process {process.pid} stopped gracefully")

        except subprocess.TimeoutExpired:
            # Force kill if timeout
            print("⚠️  Background training did not stop gracefully, forcing termination...")
            try:
                if self._is_posix:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                print("✓ Background training force killed")
                self.logger.warning(f"Background process {process.pid} force killed")
            except ProcessLookupError:
                print("✓ Background training already terminated")
                self.logger.debug(f"Process {process.pid} already terminated during force kill")

        except ProcessLookupError:
            print("✓ Background training already stopped")
            self.logger.debug(f"Process {process.pid} already terminated")

        except Exception as e:
            print(f"⚠️  Warning: Error stopping background training: {e}")
            self.logger.error(f"Error stopping background process {process.pid}: {e}", exc_info=True)

    def cleanup_all_background_processes(self) -> None:
        """Terminate all tracked background processes

        This method is typically called during cleanup to ensure all background
        processes are properly terminated. It iterates through all active processes
        and attempts graceful termination followed by force kill if needed.
        """
        for proc in self._active_background_processes[:]:  # Copy list to avoid modification issues
            if proc.poll() is None:  # Process still running
                try:
                    if self._is_posix:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()

                    proc.wait(timeout=2)
                    self.logger.info(f"Terminated background process {proc.pid}")
                except subprocess.TimeoutExpired:
                    # Force kill after timeout
                    try:
                        if self._is_posix:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                        self.logger.warning(f"Force killed background process {proc.pid}")
                    except ProcessLookupError:
                        self.logger.debug(f"Process {proc.pid} already terminated")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to kill process {proc.pid}: {e}",
                            exc_info=True
                        )
                except ProcessLookupError:
                    self.logger.debug(f"Process {proc.pid} already terminated")
                except Exception as e:
                    self.logger.error(
                        f"Error terminating process {proc.pid}: {e}",
                        exc_info=True
                    )
            self._active_background_processes.remove(proc)

        self.logger.debug("All background processes cleaned up")
