#!/usr/bin/env python3
"""
Quick demo of mutation strategies
Generates 5 mutations for ResNet20 and shows the distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation import MutationRunner

def main():
    print("="*80)
    print("🧬 MUTATION STRATEGY DEMO")
    print("="*80)

    # Initialize runner with seed for reproducibility
    runner = MutationRunner(random_seed=2025)

    # Check available hyperparameters for pytorch_resnet_cifar10
    repo = "pytorch_resnet_cifar10"
    model = "resnet20"

    print(f"\nTarget: {repo}/{model}")

    repo_config = runner.config["models"][repo]
    supported_params = repo_config["supported_hyperparams"]

    print(f"\nSupported hyperparameters:")
    for param, config in supported_params.items():
        dist = config.get("distribution", "uniform")
        zero_prob = config.get("zero_probability", 0.0)
        print(f"  - {param}:")
        print(f"      Type: {config['type']}")
        print(f"      Range: {config['range']}")
        print(f"      Default: {config.get('default', 'N/A')}")
        print(f"      Distribution: {dist}")
        if zero_prob > 0:
            print(f"      Zero Probability: {zero_prob*100:.0f}%")

    # Generate 5 mutations
    print(f"\n{'='*80}")
    print("Generating 5 mutations...")
    print(f"{'='*80}")

    mutations = runner.generate_mutations(
        repo=repo,
        model=model,
        mutate_params=["epochs", "learning_rate"],
        num_mutations=5
    )

    print(f"\n{'='*80}")
    print("MUTATION RESULTS")
    print(f"{'='*80}")

    for i, mutation in enumerate(mutations, 1):
        print(f"\nMutation {i}:")
        for param, value in mutation.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")

    # Show distribution characteristics
    print(f"\n{'='*80}")
    print("DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    epochs_vals = [m['epochs'] for m in mutations]
    lr_vals = [m['learning_rate'] for m in mutations]

    print(f"\nEpochs (log-uniform):")
    print(f"  Min: {min(epochs_vals)}, Max: {max(epochs_vals)}")
    print(f"  Mean: {sum(epochs_vals)/len(epochs_vals):.1f}")

    print(f"\nLearning Rate (log-uniform):")
    print(f"  Min: {min(lr_vals):.6f}, Max: {max(lr_vals):.6f}")
    print(f"  Mean: {sum(lr_vals)/len(lr_vals):.6f}")
    print(f"  Geometric Mean: {(min(lr_vals) * max(lr_vals)) ** 0.5:.6f}")

    print(f"\n{'='*80}")
    print("✨ Demo Complete!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Run actual training: python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate -n 10")
    print("  2. Or use config file: python3 mutation.py -ec settings/mutation_test.json")
    print()

if __name__ == "__main__":
    main()
