#!/usr/bin/env python3
"""
MutationRunner - Orchestrates mutation-based training experiments

This module provides the main orchestration class that coordinates:
- Experiment lifecycle management
- Signal handling and cleanup
- Training execution with retries
- Energy and performance data collection
- Results aggregation and CSV generation
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .session import ExperimentSession
from .command_runner import CommandRunner
from .hyperparams import generate_mutations
from .energy import check_training_success, extract_performance_metrics, parse_energy_metrics
from .utils import setup_logger, set_governor


class MutationRunner:
    """Main orchestration class for running mutation-based training experiments

    This class coordinates all aspects of the mutation-based profiling workflow:
    - Manages experiment lifecycle (initialization, execution, cleanup)
    - Handles signal interrupts and background process cleanup
    - Orchestrates training runs with retries and energy monitoring
    - Aggregates results and generates CSV summaries

    The runner composes specialized modules rather than implementing their logic:
    - ExperimentSession: Directory structure and experiment tracking
    - CommandRunner: Command execution and process management
    - hyperparams module: Hyperparameter mutation generation
    - energy module: Training validation and metric extraction
    - utils module: Logging and system configuration
    """

    # Security whitelist for governor modes
    ALLOWED_GOVERNORS = {"performance", "powersave", "ondemand", "conservative", "schedutil"}

    # Timing constants (seconds)
    GOVERNOR_TIMEOUT_SECONDS = 10
    RETRY_SLEEP_SECONDS = 30
    RUN_SLEEP_SECONDS = 60
    CONFIG_SLEEP_SECONDS = 60
    # IMPORTANT: No default timeout - allows long-running experiments
    # Each model can specify its own timeout in the config file
    DEFAULT_TRAINING_TIMEOUT_SECONDS = None  # No limit by default
    FAST_TRAINING_TIMEOUT_SECONDS = 3600     # 1 hour for quick testing

    # Parallel training constants
    BACKGROUND_STARTUP_WAIT_SECONDS = 30  # Wait for background training to fully start
    BACKGROUND_RESTART_DELAY_SECONDS = 2  # Delay between background training restarts
    BACKGROUND_TERMINATION_TIMEOUT_SECONDS = 10  # Max wait for graceful termination

    # Validation constants
    MIN_LOG_FILE_SIZE_BYTES = 1000
    DEFAULT_MAX_RETRIES = 3

    # Format constants
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Consistent timestamp format across all methods
    FLOAT_PRECISION = 6  # Decimal places for float hyperparameters

    def __init__(self, config_path: Optional[str] = None, random_seed: Optional[int] = None):
        """Initialize the mutation runner

        Sets up the experiment environment including:
        - Loading model configurations
        - Creating results directory structure
        - Initializing session and command runner
        - Registering cleanup handlers for background processes

        Args:
            config_path: Path to the models configuration file (default: mutation/models_config.json)
            random_seed: Random seed for reproducibility (default: None, uses system time)
        """
        self.project_root = Path(__file__).parent.parent.absolute()

        # Default to models_config.json in the mutation package
        if config_path is None:
            self.config_path = Path(__file__).parent / "models_config.json"
        else:
            self.config_path = self.project_root / config_path
        self.config = self._load_config()
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Set random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            import random
            random.seed(random_seed)
            print(f"Random seed set to: {random_seed}")

        # Initialize logger
        self.logger = setup_logger(__name__)

        # Track active background processes for cleanup
        self._active_background_processes = []
        self._cleanup_registered = False

        # Register cleanup handlers for reliable process termination
        atexit.register(self._cleanup_all_background_processes)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._cleanup_registered = True

        # Create experiment session for hierarchical directory structure
        self.session = ExperimentSession(self.results_dir)

        # Create command runner for executing training commands
        self.cmd_runner = CommandRunner(
            project_root=self.project_root,
            config=self.config,
            logger=self.logger
        )

    def _load_config(self) -> Dict:
        """Load models configuration from JSON file

        Returns:
            Dictionary containing model configurations

        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interrupt signals gracefully

        Ensures all background processes are terminated before exiting.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print(f"\nReceived signal {signum}, cleaning up...")
        self._cleanup_all_background_processes()
        sys.exit(0)

    def close(self) -> None:
        """Explicitly close and cleanup all resources"""
        if hasattr(self, '_active_background_processes'):
            self._cleanup_all_background_processes()
        print("MutationRunner closed (all background processes terminated)")

    def __enter__(self):
        """Context manager entry

        Returns:
            Self for use in with statements
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensure cleanup

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            False to not suppress exceptions
        """
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Cleanup background processes on deletion (last resort)"""
        # Try cleanup, but don't rely on it (atexit is primary)
        try:
            if hasattr(self, '_cleanup_registered') and not self._cleanup_registered:
                # Only cleanup if atexit wasn't registered
                self._cleanup_all_background_processes()
        except:
            pass  # Ignore all errors during finalization

    def _cleanup_all_background_processes(self) -> None:
        """Terminate all tracked background processes

        Attempts graceful termination first (SIGTERM), then forces
        termination (SIGKILL) if processes don't respond.
        """
        for proc in self._active_background_processes[:]:  # Copy list to avoid modification during iteration
            if proc.poll() is None:  # Process still running
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=2)
                    self.logger.info(f"Terminated background process {proc.pid}")
                except subprocess.TimeoutExpired:
                    # Force kill after timeout
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        self.logger.warning(f"Force killed background process {proc.pid}")
                    except ProcessLookupError:
                        self.logger.debug(f"Process {proc.pid} already terminated")
                    except Exception as e:
                        self.logger.error(f"Failed to kill process {proc.pid}: {e}", exc_info=True)
                except ProcessLookupError:
                    self.logger.debug(f"Process {proc.pid} already terminated")
                except Exception as e:
                    self.logger.error(f"Error terminating process {proc.pid}: {e}", exc_info=True)
            self._active_background_processes.remove(proc)

    def save_results(self,
                    experiment_id: str,
                    repo: str,
                    model: str,
                    mutation: Dict[str, Any],
                    duration: float,
                    energy_metrics: Dict[str, Any],
                    performance_metrics: Dict[str, float],
                    success: bool,
                    retries: int,
                    error_message: str = "") -> None:
        """Save experiment results to JSON file in experiment directory and add to session

        Results are saved in two formats:
        1. Individual JSON file in experiment directory (detailed)
        2. Added to session for CSV summary generation (aggregated)

        Args:
            experiment_id: Unique experiment identifier
            repo: Repository name
            model: Model name
            mutation: Mutated hyperparameters
            duration: Training duration in seconds
            energy_metrics: Energy consumption metrics
            performance_metrics: Model performance metrics
            success: Whether training succeeded
            retries: Number of retries attempted
            error_message: Error message if failed
        """
        result = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "repository": repo,
            "model": model,
            "hyperparameters": mutation,
            "duration_seconds": duration,
            "energy_metrics": energy_metrics,
            "performance_metrics": performance_metrics,
            "training_success": success,
            "retries": retries,
            "error_message": error_message
        }

        # Save to experiment directory (new structure)
        exp_dir = self.session.session_dir / experiment_id
        result_file = exp_dir / "experiment.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to: {result_file}")

        # Add to session for CSV generation
        self.session.add_experiment_result(result)

    def run_parallel_experiment(self,
                               fg_repo: str,
                               fg_model: str,
                               fg_mutation: Dict[str, Any],
                               bg_repo: str,
                               bg_model: str,
                               bg_hyperparams: Dict[str, Any],
                               max_retries: int = 2) -> Dict[str, Any]:
        """Run parallel experiment with foreground (monitored) and background (load) training

        This method enables testing training under GPU load by running two processes:
        1. Foreground: Monitored training with energy profiling
        2. Background: Continuous training loop for GPU utilization

        The background training runs in a loop to maintain consistent GPU load,
        automatically restarting when each training completes.

        Args:
            fg_repo: Foreground repository name
            fg_model: Foreground model name
            fg_mutation: Foreground mutated hyperparameters
            bg_repo: Background repository name
            bg_model: Background model name
            bg_hyperparams: Background hyperparameters (default values)
            max_retries: Maximum retries for foreground training

        Returns:
            Dictionary containing experiment results including:
            - experiment_id: Unique identifier
            - mode: "parallel"
            - foreground_result: Complete foreground experiment results
            - background_info: Background training configuration
        """
        # Get experiment directory from session with 'parallel' mode
        exp_dir, experiment_id = self.session.get_next_experiment_dir(fg_repo, fg_model, mode="parallel")

        # Create background logs directory within foreground experiment directory
        bg_log_dir = exp_dir / "background_logs"
        bg_log_dir.mkdir(exist_ok=True, parents=True)

        background_process = None
        script_path = None

        print("\n" + "=" * 80)
        print(f"PARALLEL EXPERIMENT: {experiment_id}")
        print("=" * 80)
        print(f"Foreground (Monitored):")
        print(f"  Repository: {fg_repo}")
        print(f"  Model: {fg_model}")
        print(f"  Hyperparameters: {fg_mutation}")
        print(f"\nBackground (GPU Load):")
        print(f"  Repository: {bg_repo}")
        print(f"  Model: {bg_model}")
        print(f"  Hyperparameters: {bg_hyperparams}")
        print(f"\nExperiment directory: {exp_dir}")
        print(f"Background logs: {bg_log_dir}")
        print("=" * 80)

        try:
            # 1. Start background training loop (logs go to bg_log_dir)
            background_process, script_path = self.cmd_runner.start_background_training(
                bg_repo, bg_model, bg_hyperparams, bg_log_dir
            )

            # Track for cleanup
            self._active_background_processes.append(background_process)

            # 2. Wait for background training to stabilize
            print(f"Waiting {self.BACKGROUND_STARTUP_WAIT_SECONDS} seconds for background training to start...")
            time.sleep(self.BACKGROUND_STARTUP_WAIT_SECONDS)

            # 3. Run foreground training with full monitoring
            print(f"\n{'─' * 80}")
            print(f"Starting foreground training...")
            print(f"{'─' * 80}\n")

            foreground_result = self.run_experiment(
                fg_repo, fg_model, fg_mutation, max_retries
            )

            print(f"\n{'─' * 80}")
            print(f"Foreground training completed")
            print(f"{'─' * 80}")

        finally:
            # 4. Always stop background training, even if foreground failed
            if background_process and background_process.poll() is None:
                self.cmd_runner.stop_background_training(background_process, script_path)
                if background_process in self._active_background_processes:
                    self._active_background_processes.remove(background_process)

        # 5. Return combined result
        return {
            "experiment_id": experiment_id,
            "mode": "parallel",
            "foreground_result": foreground_result,
            "background_info": {
                "repo": bg_repo,
                "model": bg_model,
                "hyperparameters": bg_hyperparams,
                "log_directory": str(bg_log_dir),
                "note": "Background training served as GPU load only (not monitored)"
            }
        }

    def run_experiment(self,
                      repo: str,
                      model: str,
                      mutation: Dict[str, Any],
                      max_retries: int = 2) -> Dict[str, Any]:
        """Run a single training experiment with retries

        Executes a complete training experiment including:
        1. Creating experiment directory structure
        2. Running training with energy monitoring
        3. Validating training success
        4. Extracting performance metrics
        5. Retrying on failure (up to max_retries)
        6. Saving results

        All retry attempts use the same experiment directory to maintain
        a single coherent record of the experiment.

        Args:
            repo: Repository name
            model: Model name
            mutation: Mutated hyperparameters
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary containing:
            - experiment_id: Unique identifier
            - success: Whether training succeeded
            - duration: Training duration in seconds
            - retries: Number of retries attempted
        """
        # Get experiment directory from session (BEFORE retry loop)
        exp_dir, experiment_id = self.session.get_next_experiment_dir(repo, model, mode="train")

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {experiment_id}")
        print(f"   Repository: {repo}")
        print(f"   Model: {model}")
        print(f"   Hyperparameters: {mutation}")
        print(f"   Experiment directory: {exp_dir}")
        print("=" * 80)

        success = False
        retries = 0
        exit_code = -1
        duration = 0
        energy_metrics = {}
        performance_metrics = {}
        error_message = ""

        # Use the same experiment directory for all retry attempts
        # Pass relative paths to run.sh, which will handle path resolution
        log_file = f"results/{self.session.session_dir.name}/{experiment_id}/training.log"
        energy_dir = f"results/{self.session.session_dir.name}/{experiment_id}/energy"

        while not success and retries <= max_retries:
            if retries > 0:
                print(f"\nRetry {retries}/{max_retries}")

            # Build training command
            cmd = self.cmd_runner.build_training_command_from_dir(
                repo=repo,
                model=model,
                mutation=mutation,
                exp_dir=exp_dir,
                log_file=log_file,
                energy_dir=str(energy_dir)
            )

            # Run training with energy monitoring
            exit_code, duration, energy_metrics = self.cmd_runner.run_training_with_monitoring(
                cmd=cmd,
                log_file=log_file,
                exp_dir=exp_dir,
                timeout=self.DEFAULT_TRAINING_TIMEOUT_SECONDS
            )

            # Check training success
            success, error_message = check_training_success(
                log_file=log_file,
                repo=repo,
                min_log_file_size_bytes=self.MIN_LOG_FILE_SIZE_BYTES,
                logger=self.logger
            )

            if success:
                print(f"Training successful!")
                # Extract performance metrics
                repo_config = self.config["models"][repo]
                perf_config = repo_config.get("performance_metrics", {})
                log_patterns = perf_config.get("log_patterns", {})
                performance_metrics = extract_performance_metrics(
                    log_file=log_file,
                    repo=repo,
                    log_patterns=log_patterns,
                    logger=self.logger
                )
                if performance_metrics:
                    print(f"Performance metrics: {performance_metrics}")
            else:
                print(f"Training failed: {error_message}")
                retries += 1

                if retries <= max_retries:
                    print(f"Waiting {self.RETRY_SLEEP_SECONDS} seconds before retry...")
                    time.sleep(self.RETRY_SLEEP_SECONDS)

        # Save results to experiment directory and add to session
        self.save_results(
            experiment_id=experiment_id,
            repo=repo,
            model=model,
            mutation=mutation,
            duration=duration,
            energy_metrics=energy_metrics,
            performance_metrics=performance_metrics,
            success=success,
            retries=retries,
            error_message=error_message
        )

        return {
            "experiment_id": experiment_id,
            "success": success,
            "duration": duration,
            "retries": retries
        }

    def run_mutation_experiments(self,
                                repo: str,
                                model: str,
                                mutate_params: List[str],
                                num_runs: int = 1,
                                governor: Optional[str] = None,
                                max_retries: int = 2) -> None:
        """Run multiple mutation experiments

        Generates and runs multiple experiments with mutated hyperparameters.
        This is the primary interface for command-line usage.

        Workflow:
        1. Set CPU governor (if specified)
        2. Generate unique mutations
        3. Run each experiment with retries
        4. Sleep between runs to prevent energy interference
        5. Generate summary CSV

        Args:
            repo: Repository name
            model: Model name
            mutate_params: List of hyperparameters to mutate
            num_runs: Number of mutation runs
            governor: CPU governor mode to set
            max_retries: Maximum retries per experiment
        """
        print("\n" + "=" * 80)
        print("MUTATION-BASED TRAINING ENERGY PROFILER")
        print("=" * 80)
        print(f"Repository: {repo}")
        print(f"Model: {model}")
        print(f"Parameters to mutate: {mutate_params}")
        print(f"Number of runs: {num_runs}")
        print(f"Max retries: {max_retries}")
        print("=" * 80)

        # Set CPU governor if specified
        if governor:
            set_governor(governor, self.project_root, self.logger)

        # Get repository configuration
        if repo not in self.config["models"]:
            raise ValueError(f"Repository not found in config: {repo}")
        repo_config = self.config["models"][repo]
        supported_params = repo_config["supported_hyperparams"]

        # Generate mutations
        mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=mutate_params,
            num_mutations=num_runs,
            random_seed=self.random_seed,
            logger=self.logger
        )

        # Run experiments
        results_summary = []
        total_start_time = time.time()

        for i, mutation in enumerate(mutations, 1):
            print(f"\n\n{'█' * 80}")
            print(f"█ RUN {i}/{num_runs}")
            print(f"{'█' * 80}")

            result = self.run_experiment(repo, model, mutation, max_retries)
            results_summary.append(result)

            # Sleep between runs to prevent energy interference
            if i < len(mutations):
                print(f"\nSleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                time.sleep(self.RUN_SLEEP_SECONDS)

        total_duration = time.time() - total_start_time

        # Print summary
        print("\n\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total experiments: {len(results_summary)}")
        print(f"Successful: {sum(1 for r in results_summary if r['success'])}")
        print(f"Failed: {sum(1 for r in results_summary if not r['success'])}")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Results directory: {self.results_dir}")
        print("=" * 80)

        # Print individual results
        for i, result in enumerate(results_summary, 1):
            status = "✅" if result['success'] else "❌"
            print(f"{status} Run {i}: {result['experiment_id']} "
                  f"({result['duration']:.1f}s, {result['retries']} retries)")

        # Generate summary CSV
        print("\n" + "=" * 80)
        print("Generating session summary...")
        print("=" * 80)
        csv_file = self.session.generate_summary_csv()
        if csv_file:
            print(f"Summary CSV: {csv_file}")

        # Restore file ownership if running with sudo
        self.session.restore_permissions()

        print("\nAll experiments completed!\n")

    def run_from_experiment_config(self, config_file: str) -> None:
        """Run experiments from configuration file

        This method enables complex experimental workflows defined in JSON,
        including:
        - Multiple repositories and models
        - Different experiment modes (mutation, default, parallel)
        - Custom hyperparameter configurations
        - Batch execution with controlled timing

        Configuration format:
        {
            "experiment_name": "descriptive_name",
            "description": "experiment description",
            "governor": "performance",
            "runs_per_config": 5,
            "max_retries": 2,
            "mode": "mutation",
            "experiments": [
                {
                    "repo": "repo_name",
                    "model": "model_name",
                    "mode": "mutation",
                    "mutate": ["param1", "param2"]
                },
                ...
            ]
        }

        Args:
            config_file: Path to experiment configuration JSON file

        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        config_path = self.project_root / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {config_path}")

        # Load experiment configuration
        with open(config_path, 'r') as f:
            exp_config = json.load(f)

        experiment_name = exp_config.get("experiment_name", "unnamed")
        description = exp_config.get("description", "")
        governor = exp_config.get("governor")
        runs_per_config = exp_config.get("runs_per_config", 1)
        max_retries = exp_config.get("max_retries", 2)
        experiments = exp_config.get("experiments", [])
        mode = exp_config.get("mode", "mutation")

        print("\n" + "=" * 80)
        print(f"EXPERIMENT CONFIGURATION: {experiment_name}")
        print("=" * 80)
        print(f"Description: {description}")
        print(f"Mode: {mode}")
        print(f"Total configurations: {len(experiments)}")
        print(f"Runs per configuration: {runs_per_config}")
        print(f"Max retries: {max_retries}")
        if governor:
            print(f"Governor: {governor}")
        print("=" * 80)

        # Set CPU governor if specified
        if governor:
            set_governor(governor, self.project_root, self.logger)

        # Run each experiment configuration
        all_results = []
        total_start_time = time.time()

        for exp_idx, exp in enumerate(experiments, 1):
            exp_mode = exp.get("mode", mode)  # Allow per-experiment mode override

            print(f"\n\n{'=' * 80}")
            print(f"CONFIGURATION {exp_idx}/{len(experiments)}")
            print(f"Mode: {exp_mode}")
            print("=" * 80)

            try:
                if exp_mode == "parallel":
                    # Parallel mode: Run foreground (monitored) + background (load) training
                    foreground_config = exp.get("foreground", {})
                    background_config = exp.get("background", {})

                    # Validate parallel configuration
                    if not foreground_config or not background_config:
                        print(f"Warning: Parallel mode requires 'foreground' and 'background' configs, skipping")
                        continue

                    # Extract foreground configuration
                    fg_repo = foreground_config["repo"]
                    fg_model = foreground_config["model"]
                    fg_mode = foreground_config.get("mode", "mutation")

                    # Extract background configuration
                    bg_repo = background_config["repo"]
                    bg_model = background_config["model"]
                    bg_hyperparams = background_config.get("hyperparameters", {})

                    # Validate foreground repo/model
                    if fg_repo not in self.config["models"]:
                        print(f"Warning: Foreground repo '{fg_repo}' not found, skipping")
                        continue

                    fg_repo_config = self.config["models"][fg_repo]
                    if fg_model not in fg_repo_config["models"]:
                        print(f"Warning: Foreground model '{fg_model}' not found in {fg_repo}, skipping")
                        continue

                    # Validate background repo/model
                    if bg_repo not in self.config["models"]:
                        print(f"Warning: Background repo '{bg_repo}' not found, skipping")
                        continue

                    bg_repo_config = self.config["models"][bg_repo]
                    if bg_model not in bg_repo_config["models"]:
                        print(f"Warning: Background model '{bg_model}' not found in {bg_repo}, skipping")
                        continue

                    print(f"Parallel mode:")
                    print(f"   Foreground: {fg_repo}/{fg_model} (monitored)")
                    print(f"   Background: {bg_repo}/{bg_model} (GPU load only)")

                    # Generate foreground mutations or use default hyperparameters
                    if fg_mode == "mutation":
                        mutate_params = foreground_config.get("mutate", ["all"])
                        print(f"   Mutating foreground parameters: {mutate_params}")

                        # Get foreground repository configuration
                        fg_repo_config = self.config["models"][fg_repo]
                        fg_supported_params = fg_repo_config["supported_hyperparams"]

                        fg_mutations = generate_mutations(
                            supported_params=fg_supported_params,
                            mutate_params=mutate_params,
                            num_mutations=runs_per_config,
                            random_seed=self.random_seed,
                            logger=self.logger
                        )
                    else:
                        # Use default hyperparameters
                        fg_hyperparams = foreground_config.get("hyperparameters", {})
                        fg_mutations = [fg_hyperparams] * runs_per_config

                    # Run each parallel experiment
                    for run, fg_mutation in enumerate(fg_mutations, 1):
                        print(f"\n{'─' * 80}")
                        print(f"PARALLEL RUN {run}/{runs_per_config}")
                        print(f"{'─' * 80}")

                        result = self.run_parallel_experiment(
                            fg_repo, fg_model, fg_mutation,
                            bg_repo, bg_model, bg_hyperparams,
                            max_retries
                        )
                        all_results.append(result)

                        # CRITICAL: Stop background training and sleep to allow GPU cooling
                        # This ensures 60 seconds of GPU idle time between runs
                        if run < runs_per_config:
                            print(f"\nGPU Cooling Period:")
                            print(f"   All training stopped. GPU will cool down during {self.RUN_SLEEP_SECONDS}s idle time.")
                            print(f"Sleeping {self.RUN_SLEEP_SECONDS} seconds for GPU cooling...")
                            time.sleep(self.RUN_SLEEP_SECONDS)

                elif exp_mode == "default":
                    # Default mode: use specified hyperparameters directly
                    repo = exp["repo"]
                    model = exp["model"]

                    # Validate repository and model
                    if repo not in self.config["models"]:
                        print(f"Warning: Repository '{repo}' not found, skipping")
                        continue

                    repo_config = self.config["models"][repo]
                    if model not in repo_config["models"]:
                        print(f"Warning: Model '{model}' not available in {repo}, skipping")
                        continue

                    print(f"Repository/Model: {repo}/{model}")
                    hyperparams = exp.get("hyperparameters", {})
                    print(f"Using default hyperparameters: {hyperparams}")

                    # Run with specified hyperparameters (no mutation)
                    for run in range(runs_per_config):
                        if runs_per_config > 1:
                            print(f"\n{'─' * 80}")
                            print(f"RUN {run + 1}/{runs_per_config}")
                            print(f"{'─' * 80}")

                        result = self.run_experiment(repo, model, hyperparams, max_retries)
                        all_results.append(result)

                        # Sleep between runs
                        if run < runs_per_config - 1:
                            print(f"\nSleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                            time.sleep(self.RUN_SLEEP_SECONDS)
                else:
                    # Mutation mode: mutate specified parameters
                    repo = exp["repo"]
                    model = exp["model"]

                    # Validate repository and model
                    if repo not in self.config["models"]:
                        print(f"Warning: Repository '{repo}' not found, skipping")
                        continue

                    repo_config = self.config["models"][repo]
                    if model not in repo_config["models"]:
                        print(f"Warning: Model '{model}' not available in {repo}, skipping")
                        continue

                    print(f"Repository/Model: {repo}/{model}")
                    mutate_params = exp.get("mutate", ["all"])
                    print(f"Mutating parameters: {mutate_params}")

                    # Get repository configuration
                    repo_config = self.config["models"][repo]
                    supported_params = repo_config["supported_hyperparams"]

                    # Generate mutations
                    mutations = generate_mutations(
                        supported_params=supported_params,
                        mutate_params=mutate_params,
                        num_mutations=runs_per_config,
                        random_seed=self.random_seed,
                        logger=self.logger
                    )

                    # Run each mutation
                    for run, mutation in enumerate(mutations, 1):
                        print(f"\n{'─' * 80}")
                        print(f"RUN {run}/{runs_per_config}")
                        print(f"{'─' * 80}")

                        result = self.run_experiment(repo, model, mutation, max_retries)
                        all_results.append(result)

                        # Sleep between runs
                        if run < runs_per_config:
                            print(f"\nSleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                            time.sleep(self.RUN_SLEEP_SECONDS)

            except Exception as e:
                print(f"Error running experiment: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Sleep between configurations
            if exp_idx < len(experiments):
                print(f"\nSleeping {self.CONFIG_SLEEP_SECONDS} seconds before next configuration...")
                time.sleep(self.CONFIG_SLEEP_SECONDS)

        total_duration = time.time() - total_start_time

        # Print final summary
        print("\n\n" + "=" * 80)
        print(f"FINAL SUMMARY - {experiment_name}")
        print("=" * 80)
        print(f"Total experiments: {len(all_results)}")
        print(f"Successful: {sum(1 for r in all_results if r.get('success') or r.get('foreground_result', {}).get('success'))}")
        print(f"Failed: {sum(1 for r in all_results if not (r.get('success') or r.get('foreground_result', {}).get('success')))}")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Results directory: {self.results_dir}")
        print("=" * 80)

        # Print all results
        for i, result in enumerate(all_results, 1):
            # Handle both regular and parallel results
            if result.get('mode') == 'parallel':
                fg_result = result.get('foreground_result', {})
                success = fg_result.get('success', False)
                duration = fg_result.get('duration', 0)
                retries = fg_result.get('retries', 0)
            else:
                success = result.get('success', False)
                duration = result.get('duration', 0)
                retries = result.get('retries', 0)

            status = "✅" if success else "❌"
            print(f"{status} Exp {i}: {result.get('experiment_id', 'unknown')} "
                  f"({duration:.1f}s, {retries} retries)")

        # Generate summary CSV
        print("\n" + "=" * 80)
        print("Generating session summary...")
        print("=" * 80)
        csv_file = self.session.generate_summary_csv()
        if csv_file:
            print(f"Summary CSV: {csv_file}")

        # Restore file ownership if running with sudo
        self.session.restore_permissions()

        print("\nAll experiments completed!\n")
