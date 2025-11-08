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
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class MutationRunner:
    """Main class for running mutation-based training experiments"""

    # Timing constants (seconds)
    GOVERNOR_TIMEOUT_SECONDS = 10
    RETRY_SLEEP_SECONDS = 30
    RUN_SLEEP_SECONDS = 60
    CONFIG_SLEEP_SECONDS = 120
    DEFAULT_TRAINING_TIMEOUT_SECONDS = 36000  # 10 hours max

    # Validation constants
    MIN_LOG_FILE_SIZE_BYTES = 1000
    DEFAULT_MAX_RETRIES = 2

    # Mutation constants
    SCALE_FACTOR_MIN = 0.5
    SCALE_FACTOR_MAX = 1.5
    MAX_MUTATION_ATTEMPTS = 1000  # Maximum attempts to generate unique mutations

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

    def _load_config(self) -> Dict:
        """Load models configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

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

    def mutate_hyperparameter(self, param_config: Dict, strategy: str = "random") -> Any:
        """Mutate a single hyperparameter

        Args:
            param_config: Configuration for the hyperparameter
            strategy: Mutation strategy (random, scale, discrete)

        Returns:
            Mutated value
        """
        param_type = param_config["type"]
        param_range = param_config["range"]
        default_value = param_config.get("default")

        if strategy == "random":
            # Random value within range
            if param_type == "int":
                return random.randint(param_range[0], param_range[1])
            elif param_type == "float":
                return random.uniform(param_range[0], param_range[1])

        elif strategy == "scale" and default_value is not None:
            # Scale from default value
            scale_factor = random.uniform(self.SCALE_FACTOR_MIN, self.SCALE_FACTOR_MAX)
            if param_type == "int":
                new_value = int(default_value * scale_factor)
                return max(param_range[0], min(param_range[1], new_value))
            elif param_type == "float":
                new_value = default_value * scale_factor
                return max(param_range[0], min(param_range[1], new_value))

        # Fallback to random
        return self.mutate_hyperparameter(param_config, "random")

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
                mutation[param] = self.mutate_hyperparameter(param_config)

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
        """Build the training command with mutated hyperparameters

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

        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # Add mutated hyperparameters
        for param, value in mutation.items():
            if param in supported_params:
                flag = supported_params[param]["flag"]
                param_type = supported_params[param]["type"]

                # Format value based on type
                if param_type == "int":
                    cmd.extend([flag, str(int(value))])
                elif param_type == "float":
                    cmd.extend([flag, f"{value:.6f}"])

        return cmd, log_file

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

        # Check for common error patterns
        error_patterns = [
            r"CUDA out of memory",
            r"RuntimeError",
            r"AssertionError",
            r"FileNotFoundError",
            r"KeyboardInterrupt",
            r"Traceback \(most recent call last\)",
            r"FAILED",
            r"Error:",
            r"错误:",
        ]

        for pattern in error_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                return False, f"Error pattern found: {pattern}"

        # Check for success indicators
        success_patterns = [
            r"Training completed",
            r"训练完成",
            r"SUCCESS",
            r"✓",
            r"Evaluation completed",
        ]

        for pattern in success_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                return True, "Training completed successfully"

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
            return {"avg": None, "max": None, "min": None, "sum": None}

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
                return {"avg": None, "max": None, "min": None, "sum": None}

            return {
                "avg": total / count,
                "max": max_val,
                "min": min_val,
                "sum": total
            }

        except Exception as e:
            print(f"⚠️  Warning: Error parsing {csv_file}: {e}")
            return {"avg": None, "max": None, "min": None, "sum": None}

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
        print(f"   Timeout: {timeout}s ({timeout/3600:.1f}h)")

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
        """Save experiment results to JSON file

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

        result_file = self.results_dir / f"{experiment_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 Results saved to: {result_file}")

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
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{repo}_{model}"

        print("\n" + "=" * 80)
        print(f"🔬 EXPERIMENT: {experiment_id}")
        print(f"   Repository: {repo}")
        print(f"   Model: {model}")
        print(f"   Hyperparameters: {mutation}")
        print("=" * 80)

        success = False
        retries = 0
        exit_code = -1
        duration = 0
        energy_metrics = {}
        performance_metrics = {}
        error_message = ""

        while not success and retries <= max_retries:
            if retries > 0:
                print(f"\n🔄 Retry {retries}/{max_retries}")

            # Build training command with energy directory
            energy_dir = f"results/energy_{experiment_id}_attempt{retries}"
            cmd, log_file = self.build_training_command(repo, model, mutation, energy_dir)

            # Run training with monitoring
            exit_code, duration, energy_metrics = self.run_training_with_monitoring(
                cmd, log_file, f"{experiment_id}_attempt{retries}"
            )

            # Check training success
            success, error_message = self.check_training_success(log_file, repo)

            if success:
                print(f"✅ Training successful!")
                # Extract performance metrics
                performance_metrics = self.extract_performance_metrics(log_file, repo)
                if performance_metrics:
                    print(f"📊 Performance metrics: {performance_metrics}")

                # Clean up failed attempt directories (keep only the successful one)
                for i in range(retries):
                    failed_dir = self.results_dir / f"energy_{experiment_id}_attempt{i}"
                    if failed_dir.exists():
                        try:
                            shutil.rmtree(failed_dir)
                            print(f"🗑️  Cleaned up failed attempt directory: {failed_dir.name}")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not clean up {failed_dir}: {e}")
            else:
                print(f"❌ Training failed: {error_message}")
                retries += 1

                if retries <= max_retries:
                    print(f"⏳ Waiting {self.RETRY_SLEEP_SECONDS} seconds before retry...")
                    time.sleep(self.RETRY_SLEEP_SECONDS)

        # Save results
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
                if exp_mode == "default":
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

        print("\n✨ All experiments completed!\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mutation-based Training Energy Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single mutation for ResNet20
  python mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \\
                     --mutate epochs,learning_rate,seed

  # Run 5 mutations for VulBERTa MLP, mutating all supported hyperparameters
  python mutation.py --repo VulBERTa --model mlp \\
                     --mutate all --runs 5

  # Run with performance governor
  python mutation.py --repo Person_reID_baseline_pytorch --model densenet121 \\
                     --mutate epochs,learning_rate --governor performance

  # List available models
  python mutation.py --list

  # Run from experiment configuration file
  python mutation.py --experiment-config settings/all.json
  python mutation.py --experiment-config settings/default.json
        """
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        help="Path to experiment configuration JSON file (e.g., settings/all.json, settings/default.json)"
    )

    parser.add_argument(
        "--repo",
        type=str,
        help="Repository name (e.g., pytorch_resnet_cifar10, VulBERTa)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., resnet20, mlp, densenet121)"
    )

    parser.add_argument(
        "--mutate",
        type=str,
        help="Comma-separated list of hyperparameters to mutate, or 'all' "
             "(e.g., epochs,learning_rate,seed,dropout,weight_decay)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of mutation runs (default: 1)"
    )

    parser.add_argument(
        "--governor",
        type=str,
        choices=["performance", "powersave", "ondemand", "conservative"],
        help="CPU governor mode to set before experiments"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries on training failure (default: 2)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available repositories and models"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/models_config.json",
        help="Path to models configuration file (default: config/models_config.json)"
    )

    parser.add_argument(
        "--seed",
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
