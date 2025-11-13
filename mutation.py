#!/usr/bin/env python3
"""
Mutation-based Training Energy Profiler

This script automates the process of:
1. Mutating hyperparameters for deep learning models
2. Running training with energy monitoring
3. Collecting performance and energy metrics
4. Validating training success and retrying if needed

Usage:
    python mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \\
                       --mutate epochs,learning_rate,seed \\
                       --governor performance

    python mutation.py --repo VulBERTa --model mlp \\
                       --mutate all \\
                       --runs 5
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union


class ExperimentSession:
    """Manages a single mutation.py run session with hierarchical directory structure"""

    def __init__(self, results_dir: Path):
        """Initialize experiment session

        Args:
            results_dir: Base results directory
        """
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = results_dir / f"run_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True, parents=True)

        self.experiment_counter = 0
        self.experiments = []  # List to store all experiment results for CSV generation

        print(f"📁 Session directory created: {self.session_dir}")

    def get_next_experiment_dir(self, repo: str, model: str, mode: str = "train") -> Tuple[Path, str]:
        """Get next experiment directory with auto-incrementing sequence number

        Args:
            repo: Repository name
            model: Model name
            mode: Experiment mode ('train' or 'parallel')

        Returns:
            Tuple of (experiment_directory_path, experiment_id)
        """
        self.experiment_counter += 1

        # Create experiment ID with sequence number
        # Format: {repo}_{model}_{sequence} or {repo}_{model}_{sequence}_parallel
        sequence_str = f"{self.experiment_counter:03d}"
        if mode == "parallel":
            exp_id = f"{repo}_{model}_{sequence_str}_parallel"
        else:
            exp_id = f"{repo}_{model}_{sequence_str}"

        exp_dir = self.session_dir / exp_id
        exp_dir.mkdir(exist_ok=True, parents=True)

        # Create energy subdirectory
        energy_dir = exp_dir / "energy"
        energy_dir.mkdir(exist_ok=True, parents=True)

        return exp_dir, exp_id

    def add_experiment_result(self, result: Dict[str, Any]):
        """Add experiment result to session for CSV generation

        Args:
            result: Experiment result dictionary (from save_results)
        """
        self.experiments.append(result)

    def generate_summary_csv(self) -> Path:
        """Generate CSV summary of all experiments in this session

        Returns:
            Path to generated CSV file
        """
        if not self.experiments:
            print("⚠️  No experiments to summarize")
            return None

        csv_file = self.session_dir / "summary.csv"

        # Collect all unique hyperparameter names, performance metric names
        all_hyperparams = set()
        all_perf_metrics = set()

        for exp in self.experiments:
            all_hyperparams.update(exp.get("hyperparameters", {}).keys())
            all_perf_metrics.update(exp.get("performance_metrics", {}).keys())

        # Sort for consistent column order
        hyperparam_cols = sorted(list(all_hyperparams))
        perf_metric_cols = sorted(list(all_perf_metrics))

        # Define CSV columns
        base_cols = [
            "experiment_id",
            "timestamp",
            "repository",
            "model",
            "training_success",
            "duration_seconds",
            "retries"
        ]

        hyperparam_prefix_cols = [f"hyperparam_{hp}" for hp in hyperparam_cols]
        perf_prefix_cols = [f"perf_{pm}" for pm in perf_metric_cols]

        energy_cols = [
            "energy_cpu_pkg_joules",
            "energy_cpu_ram_joules",
            "energy_cpu_total_joules",
            "energy_gpu_avg_watts",
            "energy_gpu_max_watts",
            "energy_gpu_min_watts",
            "energy_gpu_total_joules",
            "energy_gpu_temp_avg_celsius",
            "energy_gpu_temp_max_celsius",
            "energy_gpu_util_avg_percent",
            "energy_gpu_util_max_percent"
        ]

        all_cols = base_cols + hyperparam_prefix_cols + perf_prefix_cols + energy_cols

        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()

            for exp in self.experiments:
                row = {}

                # Base columns
                row["experiment_id"] = exp.get("experiment_id", "")
                row["timestamp"] = exp.get("timestamp", "")
                row["repository"] = exp.get("repository", "")
                row["model"] = exp.get("model", "")
                row["training_success"] = exp.get("training_success", False)
                row["duration_seconds"] = exp.get("duration_seconds", 0)
                row["retries"] = exp.get("retries", 0)

                # Hyperparameters
                hyperparams = exp.get("hyperparameters", {})
                for hp in hyperparam_cols:
                    row[f"hyperparam_{hp}"] = hyperparams.get(hp, "")

                # Performance metrics
                perf_metrics = exp.get("performance_metrics", {})
                for pm in perf_metric_cols:
                    row[f"perf_{pm}"] = perf_metrics.get(pm, "")

                # Energy metrics
                energy_metrics = exp.get("energy_metrics", {})
                row["energy_cpu_pkg_joules"] = energy_metrics.get("cpu_energy_pkg_joules", "")
                row["energy_cpu_ram_joules"] = energy_metrics.get("cpu_energy_ram_joules", "")
                row["energy_cpu_total_joules"] = energy_metrics.get("cpu_energy_total_joules", "")
                row["energy_gpu_avg_watts"] = energy_metrics.get("gpu_power_avg_watts", "")
                row["energy_gpu_max_watts"] = energy_metrics.get("gpu_power_max_watts", "")
                row["energy_gpu_min_watts"] = energy_metrics.get("gpu_power_min_watts", "")
                row["energy_gpu_total_joules"] = energy_metrics.get("gpu_energy_total_joules", "")
                row["energy_gpu_temp_avg_celsius"] = energy_metrics.get("gpu_temp_avg_celsius", "")
                row["energy_gpu_temp_max_celsius"] = energy_metrics.get("gpu_temp_max_celsius", "")
                row["energy_gpu_util_avg_percent"] = energy_metrics.get("gpu_util_avg_percent", "")
                row["energy_gpu_util_max_percent"] = energy_metrics.get("gpu_util_max_percent", "")

                writer.writerow(row)

        print(f"📊 Summary CSV generated: {csv_file}")
        print(f"   Total experiments: {len(self.experiments)}")
        print(f"   Successful: {sum(1 for e in self.experiments if e.get('training_success'))}")
        print(f"   Failed: {sum(1 for e in self.experiments if not e.get('training_success'))}")

        return csv_file


class MutationRunner:
    """Main class for running mutation-based training experiments"""

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

    # Mutation constants
    SCALE_FACTOR_MIN = 0.5
    SCALE_FACTOR_MAX = 1.5
    MAX_MUTATION_ATTEMPTS = 1000  # Maximum attempts to generate unique mutations

    # Format constants
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Consistent timestamp format across all methods
    FLOAT_PRECISION = 6  # Decimal places for float hyperparameters

    # Result templates (DRY)
    EMPTY_STATS_DICT = {"avg": None, "max": None, "min": None, "sum": None}

    def __init__(self, config_path: str = "config/models_config.json", random_seed: Optional[int] = None):
        """Initialize the mutation runner

        Args:
            config_path: Path to the models configuration file
            random_seed: Random seed for reproducibility (default: None, uses system time)
        """
        self.project_root = Path(__file__).parent.absolute()
        self.config_path = self.project_root / config_path
        self.config = self._load_config()
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Set random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            print(f"🎲 Random seed set to: {random_seed}")

        # Track active background processes for cleanup
        self._active_background_processes = []

        # Create experiment session for hierarchical directory structure
        self.session = ExperimentSession(self.results_dir)

    def _load_config(self) -> Dict:
        """Load models configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def __del__(self):
        """Cleanup background processes on deletion"""
        self._cleanup_all_background_processes()

    def _cleanup_all_background_processes(self):
        """Terminate all tracked background processes"""
        for proc in self._active_background_processes[:]:  # Copy list to avoid modification during iteration
            if proc.poll() is None:  # Process still running
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=2)
                except:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        pass
            self._active_background_processes.remove(proc)

    def _format_hyperparam_value(self, value: Any, param_type: str) -> str:
        """Format hyperparameter value based on type (DRY helper)

        Args:
            value: Parameter value to format
            param_type: Type of the parameter ('int' or 'float')

        Returns:
            Formatted string representation
        """
        if param_type == "int":
            return str(int(value))
        elif param_type == "float":
            return f"{value:.{self.FLOAT_PRECISION}f}"
        else:
            return str(value)

    def _build_hyperparam_args(self,
                              supported_params: Dict,
                              hyperparams: Dict[str, Any],
                              as_list: bool = True) -> Union[List[str], str]:
        """Build hyperparameter arguments (DRY helper to avoid duplication)

        Args:
            supported_params: Dictionary of supported hyperparameters from config
            hyperparams: Dictionary of hyperparameter values
            as_list: If True, return list of args; if False, return space-separated string

        Returns:
            List of argument strings or single space-separated string
        """
        args = []
        for param, value in hyperparams.items():
            if param in supported_params:
                flag = supported_params[param]["flag"]
                param_type = supported_params[param]["type"]
                formatted_value = self._format_hyperparam_value(value, param_type)
                args.extend([flag, formatted_value])

        return args if as_list else " ".join(args)

    def set_governor(self, mode: str) -> bool:
        """Set CPU governor mode using governor.sh

        Args:
            mode: Governor mode (performance, powersave, ondemand, etc.)

        Returns:
            True if successful, False otherwise
        """
        governor_script = self.project_root / "governor.sh"

        if not governor_script.exists():
            print(f"⚠️  WARNING: Governor script not found: {governor_script}")
            return False

        try:
            print(f"🔧 Setting CPU governor to: {mode}")
            result = subprocess.run(
                ["sudo", str(governor_script), mode],
                capture_output=True,
                text=True,
                timeout=self.GOVERNOR_TIMEOUT_SECONDS
            )

            if result.returncode == 0:
                print(f"✓ CPU governor set to: {mode}")
                return True
            else:
                print(f"⚠️  WARNING: Failed to set governor: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("⚠���  WARNING: Governor command timed out")
            return False
        except Exception as e:
            print(f"⚠️  WARNING: Error setting governor: {e}")
            return False

    def mutate_hyperparameter(self, param_config: Dict, param_name: str = "") -> Any:
        """Mutate a single hyperparameter with advanced strategies

        Implements parameter-specific mutation strategies:
        - Epochs: Log-uniform distribution [default×0.5, default×2.0]
        - Learning Rate: Log-uniform distribution [default×0.1, default×10.0]
        - Weight Decay: 30% zero + 70% log-uniform [0.0, default×100]
        - Dropout: Uniform distribution [0.0, 0.7]
        - Seed: Uniform integer [0, 9999]

        Args:
            param_config: Configuration for the hyperparameter (contains type, range, default, etc.)
            param_name: Name of the parameter (used to determine strategy)

        Returns:
            Mutated value
        """
        param_type = param_config["type"]
        param_range = param_config["range"]
        default_value = param_config.get("default")

        # Get mutation distribution strategy (default to "uniform")
        distribution = param_config.get("distribution", "uniform")
        zero_probability = param_config.get("zero_probability", 0.0)

        # Handle zero probability for parameters like weight_decay
        if zero_probability > 0 and random.random() < zero_probability:
            return 0.0 if param_type == "float" else 0

        # Determine the range
        min_val, max_val = param_range[0], param_range[1]

        # Handle different distribution strategies
        if distribution == "log_uniform":
            # Log-uniform distribution (for exponentially-sensitive parameters)
            if min_val <= 0:
                raise ValueError(f"Log-uniform distribution requires min_val > 0, got {min_val}")

            log_min = math.log(min_val)
            log_max = math.log(max_val)
            log_value = random.uniform(log_min, log_max)
            value = math.exp(log_value)

            if param_type == "int":
                return max(min_val, min(max_val, int(round(value))))
            else:
                return max(min_val, min(max_val, value))

        elif distribution == "uniform":
            # Standard uniform distribution
            if param_type == "int":
                return random.randint(min_val, max_val)
            else:
                return random.uniform(min_val, max_val)

        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

    def generate_mutations(self,
                          repo: str,
                          model: str,
                          mutate_params: List[str],
                          num_mutations: int = 1) -> List[Dict[str, Any]]:
        """Generate mutated hyperparameter sets with uniqueness guarantee

        Args:
            repo: Repository name
            model: Model name
            mutate_params: List of hyperparameters to mutate
            num_mutations: Number of unique mutation sets to generate

        Returns:
            List of mutated hyperparameter dictionaries (all unique)
        """
        if repo not in self.config["models"]:
            raise ValueError(f"Repository not found in config: {repo}")

        repo_config = self.config["models"][repo]
        supported_params = repo_config["supported_hyperparams"]

        # Determine which parameters to mutate
        if "all" in mutate_params:
            params_to_mutate = list(supported_params.keys())
        else:
            params_to_mutate = [p for p in mutate_params if p in supported_params]

        if not params_to_mutate:
            raise ValueError(f"No valid hyperparameters to mutate. Available: {list(supported_params.keys())}")

        print(f"📊 Generating {num_mutations} unique mutation(s) for parameters: {params_to_mutate}")

        mutations = []
        seen_mutations = set()  # Track unique mutations using frozenset of items
        attempts = 0

        while len(mutations) < num_mutations and attempts < self.MAX_MUTATION_ATTEMPTS:
            attempts += 1

            # Generate new mutation
            mutation = {}
            for param in params_to_mutate:
                param_config = supported_params[param]
                mutation[param] = self.mutate_hyperparameter(param_config, param)

            # Convert to hashable form for uniqueness check
            mutation_key = frozenset(mutation.items())

            # Check if this mutation is unique
            if mutation_key not in seen_mutations:
                seen_mutations.add(mutation_key)
                mutations.append(mutation)
                print(f"   Mutation {len(mutations)}: {mutation}")

        # Warning if we couldn't generate enough unique mutations
        if len(mutations) < num_mutations:
            print(f"⚠️  Warning: Could only generate {len(mutations)} unique mutations after {attempts} attempts")
            print(f"   Requested: {num_mutations}, Generated: {len(mutations)}")
            print(f"   Consider widening hyperparameter ranges or reducing num_mutations")

        return mutations

    def build_training_command(self,
                              repo: str,
                              model: str,
                              mutation: Dict[str, Any],
                              energy_dir: str) -> Tuple[List[str], str]:
        """Build the training command with mutated hyperparameters (DEPRECATED - use _build_training_command_from_dir)

        NOTE: This method is kept for backward compatibility but should not be used for new code.
        Use _build_training_command_from_dir() instead.

        Args:
            repo: Repository name
            model: Model name
            mutation: Dictionary of mutated hyperparameters
            energy_dir: Directory for energy monitoring outputs

        Returns:
            Tuple of (command list, log file path)
        """
        repo_config = self.config["models"][repo]
        repo_path = repo_config["path"]
        train_script = repo_config["train_script"]
        supported_params = repo_config["supported_hyperparams"]

        # Create timestamp for log file using constant
        timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
        log_file = f"results/training_{repo}_{model}_{timestamp}.log"

        # Build command (now includes energy_dir for integrated monitoring)
        run_script = self.project_root / "scripts" / "run.sh"
        cmd = [str(run_script), repo_path, train_script, log_file, energy_dir]

        # Add model flag if needed
        if "model_flag" in repo_config and model != "default":
            cmd.extend([repo_config["model_flag"], model])

        # Add required arguments
        if "required_args" in repo_config:
            for arg_name, arg_value in repo_config["required_args"].items():
                cmd.extend(arg_value.split())

        # Add mutated hyperparameters using helper method
        hyperparam_args = self._build_hyperparam_args(supported_params, mutation, as_list=True)
        cmd.extend(hyperparam_args)

        return cmd, log_file

    def _build_training_command_from_dir(self,
                                        repo: str,
                                        model: str,
                                        mutation: Dict[str, Any],
                                        exp_dir: Path,
                                        log_file: str,
                                        energy_dir: str) -> List[str]:
        """Build the training command using experiment directory structure

        Args:
            repo: Repository name
            model: Model name
            mutation: Dictionary of mutated hyperparameters
            exp_dir: Experiment directory path
            log_file: Log file path (already determined)
            energy_dir: Energy directory path (already determined)

        Returns:
            Command list for training
        """
        repo_config = self.config["models"][repo]
        repo_path = repo_config["path"]
        train_script = repo_config["train_script"]
        supported_params = repo_config["supported_hyperparams"]

        # Build command
        run_script = self.project_root / "scripts" / "run.sh"
        cmd = [str(run_script), repo_path, train_script, log_file, energy_dir]

        # Add model flag if needed
        if "model_flag" in repo_config and model != "default":
            cmd.extend([repo_config["model_flag"], model])

        # Add required arguments
        if "required_args" in repo_config:
            for arg_name, arg_value in repo_config["required_args"].items():
                cmd.extend(arg_value.split())

        # Add mutated hyperparameters using helper method
        hyperparam_args = self._build_hyperparam_args(supported_params, mutation, as_list=True)
        cmd.extend(hyperparam_args)

        return cmd

    def check_training_success(self, log_file: str, repo: str) -> Tuple[bool, str]:
        """Check if training completed successfully

        Args:
            log_file: Path to training log file
            repo: Repository name

        Returns:
            Tuple of (success, error_message)
        """
        log_path = self.project_root / log_file

        if not log_path.exists():
            return False, "Log file not found"

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        # IMPORTANT: Check for success indicators FIRST
        # Some repos may have warnings/tracebacks but still complete successfully
        success_patterns = [
            r"Training completed successfully",
            r"训练完成",
            r"训练成功完成",
            r"\[SUCCESS\]",
            r"✓.*训练成功",
            r"Evaluation completed",
            r"All.*completed",
            r"Rank@1:",  # Person_reID success indicator
            r"mAP:",     # Person_reID success indicator
        ]

        for pattern in success_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                return True, "Training completed successfully"

        # Only check for error patterns if no success indicators found
        # More specific error patterns to avoid false positives
        error_patterns = [
            r"CUDA out of memory",
            r"RuntimeError:.*(?!DeprecationWarning)",  # Exclude DeprecationWarnings
            r"AssertionError",
            r"FileNotFoundError",
            r"KeyboardInterrupt",
            r"Training.*FAILED",
            r"Fatal error:",
            r"致命错误:",
        ]

        for pattern in error_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                return False, f"Error pattern found: {pattern}"

        # If no clear indicators, check file size
        if log_path.stat().st_size < self.MIN_LOG_FILE_SIZE_BYTES:
            return False, "Log file too small, training likely failed"

        return True, "No errors detected"

    def extract_performance_metrics(self, log_file: str, repo: str) -> Dict[str, float]:
        """Extract performance metrics from training log

        Args:
            log_file: Path to training log file
            repo: Repository name

        Returns:
            Dictionary of performance metrics
        """
        log_path = self.project_root / log_file
        metrics = {}

        if not log_path.exists():
            print(f"⚠️  Warning: Log file not found for performance metric extraction: {log_path}")
            return metrics

        repo_config = self.config["models"][repo]
        log_patterns = repo_config.get("performance_metrics", {}).get("log_patterns", {})

        if not log_patterns:
            print(f"⚠️  Warning: No performance metric patterns defined for repo: {repo}")
            return metrics

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        for metric_name, pattern in log_patterns.items():
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (usually final result)
                    metrics[metric_name] = float(matches[-1])
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Warning: Failed to parse metric '{metric_name}': {e}")
            else:
                print(f"⚠️  Warning: Metric '{metric_name}' pattern not found in log")

        return metrics

    def _parse_csv_metric_streaming(self, csv_file: Path, field_name: str) -> Dict[str, Optional[float]]:
        """Parse CSV file and compute statistics in streaming fashion (memory-efficient)

        Args:
            csv_file: Path to CSV file
            field_name: Name of the field to extract

        Returns:
            Dictionary with avg, max, min, sum statistics (or None if file doesn't exist)
        """
        if not csv_file.exists():
            return self.EMPTY_STATS_DICT.copy()

        try:
            count = 0
            total = 0.0
            max_val = float('-inf')
            min_val = float('inf')

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        value = float(row[field_name])
                        count += 1
                        total += value
                        max_val = max(max_val, value)
                        min_val = min(min_val, value)
                    except (ValueError, KeyError):
                        pass

            if count == 0:
                return self.EMPTY_STATS_DICT.copy()

            return {
                "avg": total / count,
                "max": max_val,
                "min": min_val,
                "sum": total
            }

        except Exception as e:
            print(f"⚠️  Warning: Error parsing {csv_file}: {e}")
            return self.EMPTY_STATS_DICT.copy()

    def parse_energy_metrics(self, energy_dir: Path) -> Dict[str, Any]:
        """Parse energy consumption metrics

        Args:
            energy_dir: Directory containing energy monitoring outputs

        Returns:
            Dictionary of energy metrics
        """
        metrics = {
            "cpu_energy_pkg_joules": None,
            "cpu_energy_ram_joules": None,
            "cpu_energy_total_joules": None,
            "gpu_power_avg_watts": None,
            "gpu_power_max_watts": None,
            "gpu_power_min_watts": None,
            "gpu_energy_total_joules": None,
            "gpu_temp_avg_celsius": None,
            "gpu_temp_max_celsius": None,
            "gpu_util_avg_percent": None,
            "gpu_util_max_percent": None,
        }

        # Parse CPU energy
        cpu_file = energy_dir / "cpu_energy.txt"
        if cpu_file.exists():
            with open(cpu_file, 'r') as f:
                content = f.read()
                pkg_match = re.search(r"Package Energy.*:\s*([0-9.]+)", content)
                ram_match = re.search(r"RAM Energy.*:\s*([0-9.]+)", content)
                total_match = re.search(r"Total CPU Energy.*:\s*([0-9.]+)", content)

                if pkg_match:
                    metrics["cpu_energy_pkg_joules"] = float(pkg_match.group(1))
                if ram_match:
                    metrics["cpu_energy_ram_joules"] = float(ram_match.group(1))
                if total_match:
                    metrics["cpu_energy_total_joules"] = float(total_match.group(1))

        # Parse GPU power (streaming)
        gpu_power_file = energy_dir / "gpu_power.csv"
        power_stats = self._parse_csv_metric_streaming(gpu_power_file, 'power_draw_w')
        metrics["gpu_power_avg_watts"] = power_stats["avg"]
        metrics["gpu_power_max_watts"] = power_stats["max"]
        metrics["gpu_power_min_watts"] = power_stats["min"]
        # Approximate energy (sum of power samples, assuming 1 second intervals)
        metrics["gpu_energy_total_joules"] = power_stats["sum"]

        # Parse GPU temperature (streaming)
        gpu_temp_file = energy_dir / "gpu_temperature.csv"
        temp_stats = self._parse_csv_metric_streaming(gpu_temp_file, 'gpu_temp_c')
        metrics["gpu_temp_avg_celsius"] = temp_stats["avg"]
        metrics["gpu_temp_max_celsius"] = temp_stats["max"]

        # Parse GPU utilization (streaming)
        gpu_util_file = energy_dir / "gpu_utilization.csv"
        util_stats = self._parse_csv_metric_streaming(gpu_util_file, 'gpu_util_percent')
        metrics["gpu_util_avg_percent"] = util_stats["avg"]
        metrics["gpu_util_max_percent"] = util_stats["max"]

        return metrics

    def run_training_with_monitoring(self,
                                    cmd: List[str],
                                    log_file: str,
                                    experiment_id: str,
                                    timeout: Optional[int] = None) -> Tuple[int, float, Dict[str, Any]]:
        """Run training with energy monitoring

        Note: Energy monitoring is now integrated into run.sh
        This method simply runs the training command and parses the results.

        Args:
            cmd: Training command (includes energy_dir in run.sh args)
            log_file: Path to log file
            experiment_id: Unique experiment identifier
            timeout: Maximum training time in seconds (default: use class constant)

        Returns:
            Tuple of (exit_code, duration_seconds, energy_metrics)
        """
        energy_dir = self.results_dir / f"energy_{experiment_id}"
        energy_dir.mkdir(exist_ok=True)

        # Use provided timeout or default
        if timeout is None:
            timeout = self.DEFAULT_TRAINING_TIMEOUT_SECONDS

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
        # run.sh now handles both training and energy monitoring
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
            exit_code = -1

        end_time = time.time()
        duration = end_time - start_time

        print(f"✓ Training finished in {duration:.1f}s with exit code {exit_code}")

        # Parse energy metrics from the energy directory
        # run.sh has already written the energy data
        energy_metrics = self.parse_energy_metrics(energy_dir)

        # Print energy summary
        if energy_metrics.get("cpu_energy_total_joules"):
            print(f"   CPU Energy: {energy_metrics['cpu_energy_total_joules']:.2f} J")
        if energy_metrics.get("gpu_energy_total_joules"):
            print(f"   GPU Energy: {energy_metrics['gpu_energy_total_joules']:.2f} J")

        return exit_code, duration, energy_metrics

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

        print(f"💾 Results saved to: {result_file}")

        # Add to session for CSV generation
        self.session.add_experiment_result(result)

    def _build_training_args(self,
                            repo: str,
                            model: str,
                            hyperparams: Dict[str, Any]) -> str:
        """Build training argument string for background training script

        Args:
            repo: Repository name
            model: Model name
            hyperparams: Hyperparameters dictionary

        Returns:
            String of training arguments
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

        # Add hyperparameters using helper method
        hyperparam_args = self._build_hyperparam_args(supported_params, hyperparams, as_list=True)
        args.extend(hyperparam_args)

        return " ".join(args)

    def _start_background_training(self,
                                   repo: str,
                                   model: str,
                                   hyperparams: Dict[str, Any],
                                   log_dir: Path) -> Tuple[subprocess.Popen, None]:
        """Start background training loop using reusable template script

        Args:
            repo: Repository name
            model: Model name
            hyperparams: Hyperparameters for background training
            log_dir: Path to directory for background training logs

        Returns:
            Tuple of (subprocess.Popen object, None)
            Note: Returns None for script_path since we use a reusable template
        """
        repo_config = self.config["models"][repo]
        repo_path = self.project_root / repo_config["path"]
        train_script = repo_config["train_script"]

        # Build training arguments
        train_args = self._build_training_args(repo, model, hyperparams)

        # Ensure log directory exists
        log_dir.mkdir(exist_ok=True, parents=True)

        # Use reusable template script from scripts/ directory
        template_script_path = self.project_root / "scripts" / "background_training_template.sh"

        if not template_script_path.exists():
            raise RuntimeError(f"Background training template script not found: {template_script_path}")

        # Launch background process using template script with parameters
        # CRITICAL: Use os.setsid to create new process group for clean termination
        process = subprocess.Popen(
            [
                str(template_script_path),
                str(repo_path),
                train_script,
                train_args,
                str(log_dir),
                str(self.BACKGROUND_RESTART_DELAY_SECONDS)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # Create new process group
        )

        print(f"🔄 Background training started (PID: {process.pid})")
        print(f"   Template: {template_script_path.name}")
        print(f"   Repository: {repo}")
        print(f"   Model: {model}")
        print(f"   Arguments: {train_args}")
        print(f"   Log directory: {log_dir}")

        return process, None  # Return None since we don't need to delete template

    def _stop_background_training(self, process: subprocess.Popen, script_path: Optional[Path] = None) -> None:
        """Stop background training process group

        Args:
            process: Background process to terminate
            script_path: Not used (kept for backward compatibility)
                        Template script is reusable and not deleted
        """
        if process.poll() is not None:
            print("✓ Background training already stopped")
            return

        try:
            # Send SIGTERM to entire process group for graceful shutdown
            print("🛑 Stopping background training...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Wait for graceful termination (configurable timeout)
            process.wait(timeout=self.BACKGROUND_TERMINATION_TIMEOUT_SECONDS)
            print("✓ Background training stopped gracefully")

        except subprocess.TimeoutExpired:
            # Force kill if timeout
            print("⚠️  Background training did not stop gracefully, forcing termination...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
                print("✓ Background training force killed")
            except ProcessLookupError:
                print("✓ Background training already terminated")

        except ProcessLookupError:
            print("✓ Background training already stopped")

        except Exception as e:
            print(f"⚠️  Warning: Error stopping background training: {e}")

    def run_parallel_experiment(self,
                               fg_repo: str,
                               fg_model: str,
                               fg_mutation: Dict[str, Any],
                               bg_repo: str,
                               bg_model: str,
                               bg_hyperparams: Dict[str, Any],
                               max_retries: int = 2) -> Dict[str, Any]:
        """Run parallel experiment with foreground (monitored) and background (load) training

        Args:
            fg_repo: Foreground repository name
            fg_model: Foreground model name
            fg_mutation: Foreground mutated hyperparameters
            bg_repo: Background repository name
            bg_model: Background model name
            bg_hyperparams: Background hyperparameters (default values)
            max_retries: Maximum retries for foreground training

        Returns:
            Dictionary containing experiment results (foreground only)
        """
        # Get experiment directory from session with 'parallel' mode
        exp_dir, experiment_id = self.session.get_next_experiment_dir(fg_repo, fg_model, mode="parallel")

        # Create background logs directory within foreground experiment directory
        bg_log_dir = exp_dir / "background_logs"
        bg_log_dir.mkdir(exist_ok=True, parents=True)

        background_process = None
        script_path = None

        print("\n" + "=" * 80)
        print(f"🔬 PARALLEL EXPERIMENT: {experiment_id}")
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
            background_process, script_path = self._start_background_training(
                bg_repo, bg_model, bg_hyperparams, bg_log_dir
            )

            # 2. Wait for background training to stabilize
            print(f"⏳ Waiting {self.BACKGROUND_STARTUP_WAIT_SECONDS} seconds for background training to start...")
            time.sleep(self.BACKGROUND_STARTUP_WAIT_SECONDS)

            # 3. Run foreground training with full monitoring
            print(f"\n{'─' * 80}")
            print(f"🚀 Starting foreground training...")
            print(f"{'─' * 80}\n")

            foreground_result = self.run_experiment(
                fg_repo, fg_model, fg_mutation, max_retries
            )

            print(f"\n{'─' * 80}")
            print(f"✅ Foreground training completed")
            print(f"{'─' * 80}")

        finally:
            # 4. Always stop background training, even if foreground failed
            if background_process and background_process.poll() is None:
                self._stop_background_training(background_process, script_path)

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

        Args:
            repo: Repository name
            model: Model name
            mutation: Mutated hyperparameters
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary containing experiment results
        """
        # Get experiment directory from session (BEFORE retry loop)
        exp_dir, experiment_id = self.session.get_next_experiment_dir(repo, model, mode="train")

        print("\n" + "=" * 80)
        print(f"🔬 EXPERIMENT: {experiment_id}")
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
        log_file = str(exp_dir / "training.log")
        energy_dir = str(exp_dir / "energy")

        while not success and retries <= max_retries:
            if retries > 0:
                print(f"\n🔄 Retry {retries}/{max_retries}")

            # Build training command with experiment directory
            cmd = self._build_training_command_from_dir(repo, model, mutation, exp_dir, log_file, energy_dir)

            # Run training with monitoring
            exit_code, duration, energy_metrics = self.run_training_with_monitoring(
                cmd, log_file, experiment_id
            )

            # Check training success
            success, error_message = self.check_training_success(log_file, repo)

            if success:
                print(f"✅ Training successful!")
                # Extract performance metrics
                performance_metrics = self.extract_performance_metrics(log_file, repo)
                if performance_metrics:
                    print(f"📊 Performance metrics: {performance_metrics}")
            else:
                print(f"❌ Training failed: {error_message}")
                retries += 1

                if retries <= max_retries:
                    print(f"⏳ Waiting {self.RETRY_SLEEP_SECONDS} seconds before retry...")
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

        Args:
            repo: Repository name
            model: Model name
            mutate_params: List of hyperparameters to mutate
            num_runs: Number of mutation runs
            governor: CPU governor mode to set
            max_retries: Maximum retries per experiment
        """
        print("\n" + "=" * 80)
        print("🧬 MUTATION-BASED TRAINING ENERGY PROFILER")
        print("=" * 80)
        print(f"Repository: {repo}")
        print(f"Model: {model}")
        print(f"Parameters to mutate: {mutate_params}")
        print(f"Number of runs: {num_runs}")
        print(f"Max retries: {max_retries}")
        print("=" * 80)

        # Set CPU governor if specified
        if governor:
            self.set_governor(governor)

        # Generate mutations
        mutations = self.generate_mutations(repo, model, mutate_params, num_runs)

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
                print(f"\n⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                time.sleep(self.RUN_SLEEP_SECONDS)

        total_duration = time.time() - total_start_time

        # Print summary
        print("\n\n" + "=" * 80)
        print("📊 EXPERIMENT SUMMARY")
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
        print("📊 Generating session summary...")
        print("=" * 80)
        csv_file = self.session.generate_summary_csv()
        if csv_file:
            print(f"✅ Summary CSV: {csv_file}")

        print("\n✨ All experiments completed!\n")

    def run_from_experiment_config(self, config_file: str) -> None:
        """Run experiments from configuration file

        Args:
            config_file: Path to experiment configuration JSON file
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
        print(f"🧪 EXPERIMENT CONFIGURATION: {experiment_name}")
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
            self.set_governor(governor)

        # Run each experiment configuration
        all_results = []
        total_start_time = time.time()

        for exp_idx, exp in enumerate(experiments, 1):
            repo = exp["repo"]
            model = exp["model"]
            exp_mode = exp.get("mode", mode)  # Allow per-experiment mode override

            print(f"\n\n{'=' * 80}")
            print(f"CONFIGURATION {exp_idx}/{len(experiments)}: {repo}/{model}")
            print(f"Mode: {exp_mode}")
            print("=" * 80)

            # Validate repository and model
            if repo not in self.config["models"]:
                print(f"⚠️  Warning: Repository '{repo}' not found, skipping")
                continue

            repo_config = self.config["models"][repo]
            if model not in repo_config["models"]:
                print(f"⚠️  Warning: Model '{model}' not available in {repo}, skipping")
                continue

            try:
                if exp_mode == "parallel":
                    # Parallel mode: Run foreground (monitored) + background (load) training
                    foreground_config = exp.get("foreground", {})
                    background_config = exp.get("background", {})

                    # Validate parallel configuration
                    if not foreground_config or not background_config:
                        print(f"⚠️  Warning: Parallel mode requires 'foreground' and 'background' configs, skipping")
                        continue

                    # Extract foreground configuration
                    fg_repo = foreground_config.get("repo", repo)
                    fg_model = foreground_config.get("model", model)
                    fg_mode = foreground_config.get("mode", "mutation")

                    # Extract background configuration
                    bg_repo = background_config["repo"]
                    bg_model = background_config["model"]
                    bg_hyperparams = background_config.get("hyperparameters", {})

                    # Validate background repo/model
                    if bg_repo not in self.config["models"]:
                        print(f"⚠️  Warning: Background repo '{bg_repo}' not found, skipping")
                        continue

                    bg_repo_config = self.config["models"][bg_repo]
                    if bg_model not in bg_repo_config["models"]:
                        print(f"⚠️  Warning: Background model '{bg_model}' not found in {bg_repo}, skipping")
                        continue

                    print(f"📊 Parallel mode:")
                    print(f"   Foreground: {fg_repo}/{fg_model} (monitored)")
                    print(f"   Background: {bg_repo}/{bg_model} (GPU load only)")

                    # Generate foreground mutations or use default hyperparameters
                    if fg_mode == "mutation":
                        mutate_params = foreground_config.get("mutate", ["all"])
                        print(f"   Mutating foreground parameters: {mutate_params}")
                        fg_mutations = self.generate_mutations(fg_repo, fg_model, mutate_params, runs_per_config)
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
                            print(f"\n❄️  GPU Cooling Period:")
                            print(f"   All training stopped. GPU will cool down during {self.RUN_SLEEP_SECONDS}s idle time.")
                            print(f"⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds for GPU cooling...")
                            time.sleep(self.RUN_SLEEP_SECONDS)

                elif exp_mode == "default":
                    # Default mode: use specified hyperparameters directly
                    hyperparams = exp.get("hyperparameters", {})
                    print(f"📊 Using default hyperparameters: {hyperparams}")

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
                            print(f"\n⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                            time.sleep(self.RUN_SLEEP_SECONDS)
                else:
                    # Mutation mode: mutate specified parameters
                    mutate_params = exp.get("mutate", ["all"])
                    print(f"📊 Mutating parameters: {mutate_params}")

                    # Generate mutations
                    mutations = self.generate_mutations(repo, model, mutate_params, runs_per_config)

                    # Run each mutation
                    for run, mutation in enumerate(mutations, 1):
                        print(f"\n{'─' * 80}")
                        print(f"RUN {run}/{runs_per_config}")
                        print(f"{'─' * 80}")

                        result = self.run_experiment(repo, model, mutation, max_retries)
                        all_results.append(result)

                        # Sleep between runs
                        if run < runs_per_config:
                            print(f"\n⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
                            time.sleep(self.RUN_SLEEP_SECONDS)

            except Exception as e:
                print(f"❌ Error running experiment {repo}/{model}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Sleep between configurations
            if exp_idx < len(experiments):
                print(f"\n⏳ Sleeping {self.CONFIG_SLEEP_SECONDS} seconds before next configuration...")
                time.sleep(self.CONFIG_SLEEP_SECONDS)

        total_duration = time.time() - total_start_time

        # Print final summary
        print("\n\n" + "=" * 80)
        print(f"📊 FINAL SUMMARY - {experiment_name}")
        print("=" * 80)
        print(f"Total experiments: {len(all_results)}")
        print(f"Successful: {sum(1 for r in all_results if r['success'])}")
        print(f"Failed: {sum(1 for r in all_results if not r['success'])}")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Results directory: {self.results_dir}")
        print("=" * 80)

        # Print all results
        for i, result in enumerate(all_results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"{status} Exp {i}: {result['experiment_id']} "
                  f"({result['duration']:.1f}s, {result['retries']} retries)")

        # Generate summary CSV
        print("\n" + "=" * 80)
        print("📊 Generating session summary...")
        print("=" * 80)
        csv_file = self.session.generate_summary_csv()
        if csv_file:
            print(f"✅ Summary CSV: {csv_file}")

        print("\n✨ All experiments completed!\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mutation-based Training Energy Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single mutation for ResNet20 (full arguments)
  python mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \\
                     --mutate epochs,learning_rate,seed

  # Same using abbreviations
  python mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate,seed

  # Run 5 mutations for VulBERTa MLP (full arguments)
  python mutation.py --repo VulBERTa --model mlp --mutate all --runs 5

  # Same using abbreviations
  python mutation.py -r VulBERTa -m mlp -mt all -n 5

  # Run with performance governor (full arguments)
  python mutation.py --repo Person_reID_baseline_pytorch --model densenet121 \\
                     --mutate epochs,learning_rate --governor performance

  # Same using abbreviations
  python mutation.py -r Person_reID_baseline_pytorch -m densenet121 \\
                     -mt epochs,learning_rate -g performance

  # List available models
  python mutation.py --list  # or: python mutation.py -l

  # Run from experiment configuration file
  python mutation.py --experiment-config settings/all.json  # or: -ec settings/all.json
  python mutation.py --experiment-config settings/default.json
        """
    )

    parser.add_argument(
        "-ec", "--experiment-config",
        type=str,
        help="Path to experiment configuration JSON file (e.g., settings/all.json, settings/default.json)"
    )

    parser.add_argument(
        "-r", "--repo",
        type=str,
        help="Repository name (e.g., pytorch_resnet_cifar10, VulBERTa)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name (e.g., resnet20, mlp, densenet121)"
    )

    parser.add_argument(
        "-mt", "--mutate",
        type=str,
        help="Comma-separated list of hyperparameters to mutate, or 'all' "
             "(e.g., epochs,learning_rate,seed,dropout,weight_decay)"
    )

    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=1,
        help="Number of mutation runs (default: 1)"
    )

    parser.add_argument(
        "-g", "--governor",
        type=str,
        choices=["performance", "powersave", "ondemand", "conservative"],
        help="CPU governor mode to set before experiments"
    )

    parser.add_argument(
        "-mr", "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries on training failure (default: 2)"
    )

    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available repositories and models"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/models_config.json",
        help="Path to models configuration file (default: config/models_config.json)"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None, random)"
    )

    args = parser.parse_args()

    # Initialize runner
    try:
        runner = MutationRunner(config_path=args.config, random_seed=args.seed)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Handle --list flag
    if args.list:
        print("\n📋 Available Repositories and Models:\n")
        for repo_name, repo_config in runner.config["models"].items():
            print(f"  {repo_name}:")
            print(f"    Models: {', '.join(repo_config['models'])}")
            print(f"    Supported hyperparameters: {', '.join(repo_config['supported_hyperparams'].keys())}")
            print()
        sys.exit(0)

    # Handle --experiment-config flag
    if args.experiment_config:
        try:
            runner.run_from_experiment_config(args.experiment_config)
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Validate required arguments for command-line mode
    if not args.repo or not args.model or not args.mutate:
        parser.print_help()
        sys.exit(1)

    # Parse mutate parameters
    mutate_params = [p.strip() for p in args.mutate.split(",")]

    # Validate repository and model
    if args.repo not in runner.config["models"]:
        print(f"❌ Error: Repository '{args.repo}' not found in configuration")
        print(f"Available repositories: {', '.join(runner.config['models'].keys())}")
        sys.exit(1)

    repo_config = runner.config["models"][args.repo]
    if args.model not in repo_config["models"]:
        print(f"❌ Error: Model '{args.model}' not available for repository '{args.repo}'")
        print(f"Available models: {', '.join(repo_config['models'])}")
        sys.exit(1)

    # Run experiments
    try:
        runner.run_mutation_experiments(
            repo=args.repo,
            model=args.model,
            mutate_params=mutate_params,
            num_runs=args.runs,
            governor=args.governor,
            max_retries=args.max_retries
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
