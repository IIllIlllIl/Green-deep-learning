#!/usr/bin/env python3
"""
Boundary Test Results Analyzer

Analyzes the results of boundary value testing to determine if hyperparameter
ranges are reasonable based on performance impact.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def load_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all experiment results grouped by model"""
    model_results = defaultdict(list)

    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            repo = data["repository"]
            model = data["model"]
            key = f"{repo}/{model}"

            model_results[key].append({
                "experiment_id": data["experiment_id"],
                "hyperparameters": data["hyperparameters"],
                "performance_metrics": data["performance_metrics"],
                "training_success": data["training_success"],
                "duration_seconds": data["duration_seconds"]
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not parse {result_file}: {e}", file=sys.stderr)

    return model_results

def get_primary_metric(model_key: str, perf_metrics: Dict) -> float:
    """Get the primary performance metric for a model"""
    # Define primary metrics for each model type
    if "mnist" in model_key.lower():
        return perf_metrics.get("test_accuracy", 0.0)
    elif "resnet" in model_key.lower():
        return perf_metrics.get("test_accuracy", 0.0)
    elif "reid" in model_key.lower():
        return perf_metrics.get("rank1", 0.0)
    elif "mrt-oast" in model_key.lower():
        return perf_metrics.get("accuracy", 0.0)
    else:
        # Default: return first metric
        return list(perf_metrics.values())[0] if perf_metrics else 0.0

def identify_baseline(results: List[Dict], model_key: str) -> Dict:
    """Identify the baseline (default) configuration"""
    # Default configurations by model
    defaults = {
        "examples/mnist": {"epochs": 10, "learning_rate": 0.01},
        "pytorch_resnet_cifar10/resnet20": {"epochs": 200, "learning_rate": 0.1, "weight_decay": 0.0001},
        "Person_reID_baseline_pytorch/densenet121": {"epochs": 60, "learning_rate": 0.05, "dropout": 0.5},
        "MRT-OAST/default": {"epochs": 10, "learning_rate": 0.0001, "dropout": 0.2, "weight_decay": 0.0}
    }

    target_params = defaults.get(model_key, {})

    # Find result matching default configuration
    for r in results:
        match = True
        for key, val in target_params.items():
            if abs(r["hyperparameters"].get(key, -999) - val) > 1e-6:
                match = False
                break
        if match:
            return r

    # If no exact match, return first result
    return results[0] if results else None

def analyze_model(model_key: str, results: List[Dict]) -> None:
    """Analyze boundary test results for a single model"""
    print(f"\n{'='*80}")
    print(f"Model: {model_key}")
    print(f"{'='*80}")

    if not results:
        print("⚠️  No results found")
        return

    # Identify baseline
    baseline = identify_baseline(results, model_key)
    if not baseline:
        print("❌ Could not identify baseline configuration")
        return

    baseline_perf = get_primary_metric(model_key, baseline["performance_metrics"])
    print(f"\n📊 Baseline Configuration:")
    print(f"   Hyperparameters: {baseline['hyperparameters']}")
    print(f"   Performance: {baseline_perf:.2f}%")
    print(f"   Duration: {baseline['duration_seconds']:.1f}s")

    # Analyze each boundary configuration
    print(f"\n📈 Boundary Value Analysis:")
    print(f"{'Status':<8} {'Performance':<12} {'Change':<10} {'Hyperparameters'}")
    print(f"{'-'*80}")

    boundary_results = []
    for r in results:
        if r == baseline:
            continue

        perf = get_primary_metric(model_key, r["performance_metrics"])
        diff = perf - baseline_perf
        diff_pct = (diff / baseline_perf * 100) if baseline_perf > 0 else 0

        # Determine status
        if diff_pct >= -3:
            status = "✅"
            rating = "Good"
        elif diff_pct >= -5:
            status = "✅"
            rating = "OK"
        elif diff_pct >= -10:
            status = "⚠️"
            rating = "Warning"
        else:
            status = "❌"
            rating = "Bad"

        boundary_results.append({
            "status": status,
            "rating": rating,
            "perf": perf,
            "diff": diff,
            "diff_pct": diff_pct,
            "hyperparams": r["hyperparameters"]
        })

        print(f"{status:<8} {perf:>6.2f}%      {diff:>+6.2f}%    {r['hyperparameters']}")

    # Summary
    print(f"\n📋 Summary:")
    good = sum(1 for r in boundary_results if r["diff_pct"] >= -5)
    warning = sum(1 for r in boundary_results if -10 <= r["diff_pct"] < -5)
    bad = sum(1 for r in boundary_results if r["diff_pct"] < -10)

    print(f"   ✅ Good/OK (drop < 5%): {good}/{len(boundary_results)}")
    print(f"   ⚠️  Warning (drop 5-10%): {warning}/{len(boundary_results)}")
    print(f"   ❌ Bad (drop > 10%): {bad}/{len(boundary_results)}")

    if bad > 0:
        print(f"\n   ❌ RECOMMENDATION: Narrow the range for parameters causing >10% performance drop")
    elif warning > 0:
        print(f"\n   ⚠️  RECOMMENDATION: Consider narrowing ranges for parameters causing 5-10% drop")
    else:
        print(f"\n   ✅ RECOMMENDATION: Current ranges are reasonable, can proceed with mutation experiments")

def main():
    """Main analysis function"""
    results_dir = Path("results")

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("="*80)
    print("🔬 BOUNDARY VALUE TEST ANALYSIS")
    print("="*80)

    # Load all results
    model_results = load_results(results_dir)

    if not model_results:
        print("❌ No results found in results directory")
        sys.exit(1)

    print(f"\nFound results for {len(model_results)} model(s):")
    for model_key, results in model_results.items():
        print(f"  - {model_key}: {len(results)} configurations")

    # Analyze each model
    for model_key in sorted(model_results.keys()):
        analyze_model(model_key, model_results[model_key])

    # Overall summary
    print(f"\n{'='*80}")
    print("🎯 OVERALL RECOMMENDATION")
    print(f"{'='*80}")
    print("""
Based on the boundary value test results above:

1. Check if any model has configurations with >10% performance drop (❌)
   → If YES: You MUST narrow those hyperparameter ranges before mutation experiments

2. Check if any model has configurations with 5-10% performance drop (⚠️)
   → If YES: Consider narrowing those ranges for better mutation quality

3. If all boundary values cause <5% performance drop (✅):
   → Current ranges are reasonable, proceed with mutation experiments

Next steps:
- Review the analysis above
- Adjust config/models_config.json if needed
- Re-run boundary tests for adjusted parameters
- Once satisfied, run: python3 mutation.py -ec settings/mutation_experiment_elite_plus.json
    """)

if __name__ == "__main__":
    main()
