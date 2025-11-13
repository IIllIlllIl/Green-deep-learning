"""Experiment session management and result persistence"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


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

    def generate_summary_csv(self) -> Optional[Path]:
        """Generate CSV summary of all experiments in this session

        Returns:
            Path to generated CSV file, or None if no experiments
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
