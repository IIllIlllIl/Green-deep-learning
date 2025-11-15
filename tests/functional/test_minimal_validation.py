"""
Functional test to verify basic CLI and mutation functionality

This test performs a minimal validation to ensure:
1. CLI argument parsing works
2. Mutation generation works
3. Config loading works
4. No import or basic runtime errors

This is a smoke test - does NOT run actual training.
"""

import subprocess
import sys
import json
from pathlib import Path


def test_cli_help():
    """Test that --help works"""
    print('=== Test 1: CLI --help ===')
    result = subprocess.run(
        ['python', 'mutation.py', '--help'],
        capture_output=True,
        text=True,
        timeout=5
    )

    assert result.returncode == 0, f"--help failed with code {result.returncode}"
    assert 'Mutation-based Training Energy Profiler' in result.stdout
    assert '--repo' in result.stdout
    assert '--model' in result.stdout
    assert '--mutate' in result.stdout

    print('✅ PASSED\n')


def test_cli_list():
    """Test that --list works"""
    print('=== Test 2: CLI --list ===')
    result = subprocess.run(
        ['python', 'mutation.py', '--list'],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, f"--list failed with code {result.returncode}"
    assert 'Available Repositories and Models' in result.stdout
    assert 'Person_reID_baseline_pytorch' in result.stdout
    assert 'densenet121' in result.stdout
    assert 'pytorch_resnet_cifar10' in result.stdout
    assert 'resnet20' in result.stdout

    print(result.stdout)
    print('✅ PASSED\n')


def test_mutation_import():
    """Test that mutation module imports correctly"""
    print('=== Test 3: Mutation Module Import ===')

    try:
        # Add parent directory to path to access mutation package
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))

        from mutation.hyperparams import generate_mutations
        from mutation.energy import extract_performance_metrics, check_training_success
        from mutation.runner import MutationRunner
        from mutation.session import ExperimentSession
        from mutation.command_runner import CommandRunner

        print('✓ hyperparams.generate_mutations imported')
        print('✓ energy.extract_performance_metrics imported')
        print('✓ energy.check_training_success imported')
        print('✓ runner.MutationRunner imported')
        print('✓ session.ExperimentSession imported')
        print('✓ command_runner.CommandRunner imported')

        print('✅ PASSED\n')

    except ImportError as e:
        raise AssertionError(f"Failed to import mutation modules: {e}")


def test_config_loading():
    """Test that config file loads correctly"""
    print('=== Test 4: Config Loading ===')

    config_path = Path('mutation/models_config.json')
    assert config_path.exists(), "Config file not found"

    with open(config_path, 'r') as f:
        config = json.load(f)

    assert 'models' in config
    assert 'Person_reID_baseline_pytorch' in config['models']
    assert 'pytorch_resnet_cifar10' in config['models']

    # Verify all epochs use uniform distribution
    for model_name, model_config in config['models'].items():
        if 'epochs' in model_config['supported_hyperparams']:
            epochs_dist = model_config['supported_hyperparams']['epochs']['distribution']
            assert epochs_dist == 'uniform', \
                f"{model_name} epochs should use uniform distribution, got {epochs_dist}"

    # Verify all weight_decay have no zero_probability
    for model_name, model_config in config['models'].items():
        if 'weight_decay' in model_config['supported_hyperparams']:
            wd_config = model_config['supported_hyperparams']['weight_decay']
            assert 'zero_probability' not in wd_config, \
                f"{model_name} weight_decay should not have zero_probability"

    print('✓ Config file loaded')
    print('✓ All epochs use uniform distribution')
    print('✓ All weight_decay configs have no zero_probability')
    print('✅ PASSED\n')


def test_mutation_generation_basic():
    """Test basic mutation generation without running training"""
    print('=== Test 5: Mutation Generation ===')

    # Add parent directory to path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))

    from mutation.hyperparams import generate_mutations

    # Load config
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    # Test mutation generation for DenseNet121
    repo_config = config['models']['Person_reID_baseline_pytorch']
    supported_params = repo_config['supported_hyperparams']

    # Generate 3 mutations
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=['learning_rate'],
        num_mutations=3
    )

    assert len(mutations) == 3, f"Should generate 3 mutations, got {len(mutations)}"

    # Verify all mutations are different from default
    default_lr = 0.05
    for mutation in mutations:
        assert 'learning_rate' in mutation
        assert mutation['learning_rate'] != default_lr, \
            f"Mutation should not equal default {default_lr}"
        assert 0.025 <= mutation['learning_rate'] <= 0.1, \
            f"LR {mutation['learning_rate']} out of range [0.025, 0.1]"

    print(f'✓ Generated {len(mutations)} mutations')
    print(f'✓ All mutations different from default (0.05)')
    print(f'✓ All mutations in range [0.025, 0.1]')
    print('✅ PASSED\n')


def test_runner_initialization():
    """Test that MutationRunner can be initialized"""
    print('=== Test 6: MutationRunner Initialization ===')

    # Add parent directory to path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))

    from mutation.runner import MutationRunner

    try:
        runner = MutationRunner()
        assert runner.config is not None
        assert runner.session is not None
        assert runner.cmd_runner is not None
        runner.close()  # Clean up

        print('✓ MutationRunner initialized')
        print('✓ Config loaded')
        print('✓ Session created')
        print('✓ CommandRunner created')
        print('✅ PASSED\n')

    except Exception as e:
        raise AssertionError(f"Failed to initialize MutationRunner: {e}")


if __name__ == '__main__':
    print('=' * 70)
    print('Minimal Functional Validation Tests')
    print('=' * 70)
    print('NOTE: These tests do NOT run actual training.')
    print('They only verify CLI, imports, config, and mutation generation.')
    print('=' * 70)
    print()

    try:
        test_cli_help()
        test_cli_list()
        test_mutation_import()
        test_config_loading()
        test_mutation_generation_basic()
        test_runner_initialization()

        print('=' * 70)
        print('✅ ALL FUNCTIONAL TESTS PASSED')
        print('=' * 70)
        print()
        print('Summary:')
        print('- CLI argument parsing works')
        print('- All mutation modules import correctly')
        print('- Config file loads and validates correctly')
        print('- Mutation generation works as expected')
        print('- MutationRunner initializes successfully')
        print()
        print('The system is ready for actual training experiments.')

    except AssertionError as e:
        print('=' * 70)
        print(f'❌ TEST FAILED: {e}')
        print('=' * 70)
        sys.exit(1)
    except Exception as e:
        print('=' * 70)
        print(f'❌ UNEXPECTED ERROR: {e}')
        print('=' * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
