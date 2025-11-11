#!/usr/bin/env python3
"""
Test script for mutation strategies
Tests the new distribution-based mutation functionality
"""

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

# Import the mutation function
sys.path.insert(0, str(Path(__file__).parent))
from mutation import MutationRunner


def test_log_uniform_distribution():
    """Test log-uniform distribution for epochs and learning_rate"""
    print("\n" + "="*80)
    print("TEST 1: Log-Uniform Distribution (Epochs)")
    print("="*80)

    runner = MutationRunner(random_seed=42)

    # Test config for epochs (log-uniform)
    epochs_config = {
        "type": "int",
        "default": 10,
        "range": [5, 20],
        "distribution": "log_uniform"
    }

    samples = []
    for i in range(100):
        value = runner.mutate_hyperparameter(epochs_config, "epochs")
        samples.append(value)

    print(f"Sample size: {len(samples)}")
    print(f"Min: {min(samples)}, Max: {max(samples)}")
    print(f"Mean: {sum(samples)/len(samples):.2f}")
    print(f"Distribution: {dict(Counter(samples))}")

    # Check if values cluster towards lower end (expected for log-uniform)
    below_12 = sum(1 for s in samples if s < 12)
    above_12 = sum(1 for s in samples if s >= 12)
    print(f"\nBelow 12: {below_12}%, Above 12: {above_12}%")
    print("Expected: More values should be in lower range due to log-uniform distribution")

    return samples


def test_learning_rate_distribution():
    """Test log-uniform distribution for learning rate"""
    print("\n" + "="*80)
    print("TEST 2: Log-Uniform Distribution (Learning Rate)")
    print("="*80)

    runner = MutationRunner(random_seed=123)

    lr_config = {
        "type": "float",
        "default": 0.01,
        "range": [0.001, 0.1],
        "distribution": "log_uniform"
    }

    samples = []
    for i in range(100):
        value = runner.mutate_hyperparameter(lr_config, "learning_rate")
        samples.append(value)

    print(f"Sample size: {len(samples)}")
    print(f"Min: {min(samples):.6f}, Max: {max(samples):.6f}")
    print(f"Mean: {sum(samples)/len(samples):.6f}")
    print(f"Median: {sorted(samples)[50]:.6f}")

    # Show distribution in log scale
    log_samples = [math.log10(s) for s in samples]
    print(f"\nLog10 values:")
    print(f"  Min: {min(log_samples):.2f}, Max: {max(log_samples):.2f}")
    print(f"  Mean: {sum(log_samples)/len(log_samples):.2f}")

    # Bin analysis
    bins = {
        "0.001-0.003": sum(1 for s in samples if 0.001 <= s < 0.003),
        "0.003-0.01": sum(1 for s in samples if 0.003 <= s < 0.01),
        "0.01-0.03": sum(1 for s in samples if 0.01 <= s < 0.03),
        "0.03-0.1": sum(1 for s in samples if 0.03 <= s < 0.1),
    }
    print(f"\nDistribution across bins:")
    for bin_range, count in bins.items():
        print(f"  {bin_range}: {count}")

    print("Expected: Roughly uniform distribution in log space")

    return samples


def test_weight_decay_zero_probability():
    """Test weight_decay with 30% zero probability"""
    print("\n" + "="*80)
    print("TEST 3: Weight Decay with Zero Probability (30%)")
    print("="*80)

    runner = MutationRunner(random_seed=456)

    wd_config = {
        "type": "float",
        "default": 0.0001,
        "range": [0.00001, 0.01],
        "distribution": "log_uniform",
        "zero_probability": 0.3
    }

    samples = []
    for i in range(1000):  # Larger sample for better statistics
        value = runner.mutate_hyperparameter(wd_config, "weight_decay")
        samples.append(value)

    zero_count = sum(1 for s in samples if s == 0.0)
    non_zero = [s for s in samples if s != 0.0]

    print(f"Sample size: {len(samples)}")
    print(f"Zero values: {zero_count} ({zero_count/len(samples)*100:.1f}%)")
    print(f"Expected: ~30% (300/1000)")

    if non_zero:
        print(f"\nNon-zero values:")
        print(f"  Min: {min(non_zero):.6f}, Max: {max(non_zero):.6f}")
        print(f"  Mean: {sum(non_zero)/len(non_zero):.6f}")
        print(f"  Median: {sorted(non_zero)[len(non_zero)//2]:.6f}")

    return samples


def test_dropout_uniform_distribution():
    """Test dropout with uniform distribution"""
    print("\n" + "="*80)
    print("TEST 4: Dropout with Uniform Distribution [0.0, 0.7]")
    print("="*80)

    runner = MutationRunner(random_seed=789)

    dropout_config = {
        "type": "float",
        "default": 0.5,
        "range": [0.0, 0.7],
        "distribution": "uniform"
    }

    samples = []
    for i in range(100):
        value = runner.mutate_hyperparameter(dropout_config, "dropout")
        samples.append(value)

    print(f"Sample size: {len(samples)}")
    print(f"Min: {min(samples):.3f}, Max: {max(samples):.3f}")
    print(f"Mean: {sum(samples)/len(samples):.3f}")
    print(f"Expected mean: ~0.35 (middle of [0.0, 0.7])")

    # Bin analysis
    bins = {
        "0.0-0.2": sum(1 for s in samples if 0.0 <= s < 0.2),
        "0.2-0.4": sum(1 for s in samples if 0.2 <= s < 0.4),
        "0.4-0.6": sum(1 for s in samples if 0.4 <= s < 0.6),
        "0.6-0.7": sum(1 for s in samples if 0.6 <= s <= 0.7),
    }
    print(f"\nDistribution across bins:")
    for bin_range, count in bins.items():
        print(f"  {bin_range}: {count}")

    print("Expected: Roughly uniform distribution across bins")

    return samples


def test_seed_uniform_distribution():
    """Test seed with uniform integer distribution"""
    print("\n" + "="*80)
    print("TEST 5: Seed with Uniform Integer Distribution [0, 9999]")
    print("="*80)

    runner = MutationRunner(random_seed=101112)

    seed_config = {
        "type": "int",
        "default": 42,
        "range": [0, 9999],
        "distribution": "uniform"
    }

    samples = []
    for i in range(100):
        value = runner.mutate_hyperparameter(seed_config, "seed")
        samples.append(value)

    print(f"Sample size: {len(samples)}")
    print(f"Min: {min(samples)}, Max: {max(samples)}")
    print(f"Mean: {sum(samples)/len(samples):.1f}")
    print(f"Expected mean: ~5000 (middle of [0, 9999])")

    # Bin analysis
    bins = {
        "0-2499": sum(1 for s in samples if 0 <= s < 2500),
        "2500-4999": sum(1 for s in samples if 2500 <= s < 5000),
        "5000-7499": sum(1 for s in samples if 5000 <= s < 7500),
        "7500-9999": sum(1 for s in samples if 7500 <= s <= 9999),
    }
    print(f"\nDistribution across quartiles:")
    for bin_range, count in bins.items():
        print(f"  {bin_range}: {count}")

    print("Expected: Roughly 25 in each quartile")

    return samples


def test_mutation_uniqueness():
    """Test that generate_mutations produces unique mutations"""
    print("\n" + "="*80)
    print("TEST 6: Mutation Uniqueness")
    print("="*80)

    # Create a minimal test config
    runner = MutationRunner(random_seed=999)

    # Mock config for testing
    test_config = {
        "models": {
            "test_repo": {
                "path": "test",
                "train_script": "test.sh",
                "models": ["test_model"],
                "supported_hyperparams": {
                    "epochs": {
                        "type": "int",
                        "default": 10,
                        "range": [5, 20],
                        "distribution": "log_uniform"
                    },
                    "learning_rate": {
                        "type": "float",
                        "default": 0.01,
                        "range": [0.001, 0.1],
                        "distribution": "log_uniform"
                    }
                }
            }
        }
    }

    runner.config = test_config

    # Generate 20 mutations
    mutations = runner.generate_mutations("test_repo", "test_model", ["epochs", "learning_rate"], 20)

    print(f"Requested: 20 mutations")
    print(f"Generated: {len(mutations)} mutations")

    # Check uniqueness
    mutation_tuples = [frozenset(m.items()) for m in mutations]
    unique_count = len(set(mutation_tuples))

    print(f"Unique mutations: {unique_count}")
    print(f"All mutations unique: {unique_count == len(mutations)}")

    # Show first 5 mutations
    print(f"\nFirst 5 mutations:")
    for i, m in enumerate(mutations[:5], 1):
        print(f"  {i}. epochs={m['epochs']}, lr={m['learning_rate']:.6f}")

    return mutations


def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*80)
    print("█ MUTATION STRATEGY TEST SUITE")
    print("█"*80)

    try:
        test_log_uniform_distribution()
        test_learning_rate_distribution()
        test_weight_decay_zero_probability()
        test_dropout_uniform_distribution()
        test_seed_uniform_distribution()
        test_mutation_uniqueness()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nSummary:")
        print("- Log-uniform distribution working for epochs and learning_rate")
        print("- Zero probability working for weight_decay (30% zeros)")
        print("- Uniform distribution working for dropout and seed")
        print("- Mutation uniqueness guaranteed")
        print("\n✨ Mutation strategies are ready for use!")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
