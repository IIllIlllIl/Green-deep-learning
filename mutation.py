#!/usr/bin/env python3
"""
Mutation-based Training Energy Profiler - CLI Entry Point

This script provides a minimalist CLI wrapper for the mutation framework.

Usage:
    python mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \\
                       --mutate epochs,learning_rate,seed \\
                       --governor performance

    python mutation.py --repo VulBERTa --model mlp \\
                       --mutate all \\
                       --runs 5
"""

import argparse
import sys

from mutation.runner import MutationRunner


def main():
    """Main CLI entry point"""
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
        default=None,
        help="Path to models configuration file (default: mutation/models_config.json)"
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
