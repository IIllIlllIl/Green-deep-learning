"""
Unit tests for hyperparameter mutation distributions

Tests verify:
1. Epochs uses uniform (linear) distribution
2. Weight Decay uses 100% log-uniform sampling (no zero probability)
3. Default values are excluded from mutations
"""

import sys
import json
from pathlib import Path

# Add mutation package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mutation"))

from hyperparams import mutate_hyperparameter, generate_mutations


def test_epochs_uniform_distribution():
    """Test that epochs uses uniform distribution"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    epochs_config = config['models']['Person_reID_baseline_pytorch']['supported_hyperparams']['epochs']

    print('=== Test 1: Epochs Uniform Distribution ===')
    print(f'Config: {epochs_config}')

    # Generate 20 samples
    values = [mutate_hyperparameter(epochs_config, 'epochs') for _ in range(20)]

    mean = sum(values) / len(values)
    min_val = min(values)
    max_val = max(values)

    print(f'Generated values: {values}')
    print(f'Mean: {mean:.1f} (expected ~60 for uniform [30,90])')
    print(f'Min: {min_val}, Max: {max_val}')

    # Assertions
    assert epochs_config['distribution'] == 'uniform', "Epochs should use uniform distribution"
    assert 30 <= min_val <= 90, f"Min value {min_val} out of range [30, 90]"
    assert 30 <= max_val <= 90, f"Max value {max_val} out of range [30, 90]"
    assert 50 <= mean <= 70, f"Mean {mean} should be near 60 for uniform distribution"

    print('✅ PASSED\n')


def test_weight_decay_100_percent_log_uniform():
    """Test that weight decay uses 100% log-uniform sampling (no zeros)"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    wd_config = config['models']['pytorch_resnet_cifar10']['supported_hyperparams']['weight_decay']

    print('=== Test 2: Weight Decay 100% Log-Uniform ===')
    print(f'Config: {wd_config}')

    # Generate 20 samples
    values = [mutate_hyperparameter(wd_config, 'weight_decay') for _ in range(20)]

    print(f'Generated values: {[f"{v:.6f}" for v in values]}')
    print(f'Any zeros?: {0.0 in values}')
    print(f'Min: {min(values):.6f}, Max: {max(values):.6f}')

    # Assertions
    assert wd_config['distribution'] == 'log_uniform', "Weight decay should use log_uniform"
    assert 'zero_probability' not in wd_config, "Weight decay should not have zero_probability"
    assert 0.0 not in values, "Weight decay should never generate 0.0"
    assert all(1e-5 <= v <= 0.01 for v in values), "All values should be in range [1e-5, 0.01]"

    print('✅ PASSED\n')


def test_default_value_exclusion_single_param():
    """Test that default values are excluded from mutations (single parameter)"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    params = config['models']['Person_reID_baseline_pytorch']['supported_hyperparams']

    print('=== Test 3: Default Value Exclusion (Single Parameter) ===')
    print('Generating 10 mutations for learning_rate')
    print(f'Default LR: 0.05')

    mutations = generate_mutations(params, ['learning_rate'], num_mutations=10)
    lr_values = [m['learning_rate'] for m in mutations]

    print(f'\nGenerated LRs: {[f"{lr:.6f}" for lr in lr_values]}')
    print(f'Contains default 0.05?: {0.05 in lr_values}')
    print(f'Contains values close to 0.05?: {any(abs(lr - 0.05) < 0.0001 for lr in lr_values)}')

    # Assertions
    assert 0.05 not in lr_values, "Generated mutations should not contain default value 0.05"
    assert len(mutations) == 10, f"Should generate 10 mutations, got {len(mutations)}"
    assert all(0.025 <= lr <= 0.1 for lr in lr_values), "All LR values should be in range [0.025, 0.1]"

    print('✅ PASSED\n')


def test_default_value_exclusion_multiple_params():
    """Test that default values are excluded from mutations (multiple parameters)"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    params = config['models']['MRT-OAST']['supported_hyperparams']

    print('=== Test 4: Default Value Exclusion (Multiple Parameters) ===')
    print('Generating 5 mutations for epochs + learning_rate')
    print(f'Defaults: epochs=10, lr=0.0001')

    mutations = generate_mutations(params, ['epochs', 'learning_rate'], num_mutations=5)

    print('\nGenerated mutations:')
    for i, m in enumerate(mutations, 1):
        print(f'  Mutation {i}: epochs={m["epochs"]}, lr={m["learning_rate"]:.6f}')

    has_default_epochs = any(m['epochs'] == 10 for m in mutations)
    has_default_lr = any(abs(m['learning_rate'] - 0.0001) < 1e-7 for m in mutations)
    has_both_defaults = any(m['epochs'] == 10 and abs(m['learning_rate'] - 0.0001) < 1e-7 for m in mutations)

    print(f'\nContains default epochs=10?: {has_default_epochs}')
    print(f'Contains default lr=0.0001?: {has_default_lr}')
    print(f'Contains both defaults?: {has_both_defaults}')

    # Assertions
    assert not has_both_defaults, "Should not generate mutation with both default values"
    assert len(mutations) == 5, f"Should generate 5 mutations, got {len(mutations)}"

    print('✅ PASSED\n')


def test_all_models_epochs_uniform():
    """Test that all models use uniform distribution for epochs"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    print('=== Test 5: All Models Use Uniform Distribution for Epochs ===')

    models_with_epochs = []
    for model_name, model_config in config['models'].items():
        if 'epochs' in model_config['supported_hyperparams']:
            epochs_config = model_config['supported_hyperparams']['epochs']
            distribution = epochs_config.get('distribution', 'uniform')
            models_with_epochs.append((model_name, distribution))
            print(f'{model_name}: {distribution}')

    # Assertions
    for model_name, distribution in models_with_epochs:
        assert distribution == 'uniform', f"{model_name} should use uniform distribution for epochs, got {distribution}"

    print(f'\n✅ PASSED - All {len(models_with_epochs)} models use uniform distribution\n')


def test_all_models_weight_decay_no_zero_prob():
    """Test that all models have no zero_probability for weight_decay"""
    with open('mutation/models_config.json', 'r') as f:
        config = json.load(f)

    print('=== Test 6: All Models Have No Zero Probability for Weight Decay ===')

    models_with_wd = []
    for model_name, model_config in config['models'].items():
        if 'weight_decay' in model_config['supported_hyperparams']:
            wd_config = model_config['supported_hyperparams']['weight_decay']
            has_zero_prob = 'zero_probability' in wd_config
            models_with_wd.append((model_name, has_zero_prob))
            print(f'{model_name}: zero_probability={wd_config.get("zero_probability", "N/A")}')

    # Assertions
    for model_name, has_zero_prob in models_with_wd:
        assert not has_zero_prob, f"{model_name} should not have zero_probability for weight_decay"

    print(f'\n✅ PASSED - All {len(models_with_wd)} models have no zero_probability\n')


if __name__ == '__main__':
    print('=' * 70)
    print('Hyperparameter Distribution Tests')
    print('=' * 70)
    print()

    try:
        test_epochs_uniform_distribution()
        test_weight_decay_100_percent_log_uniform()
        test_default_value_exclusion_single_param()
        test_default_value_exclusion_multiple_params()
        test_all_models_epochs_uniform()
        test_all_models_weight_decay_no_zero_prob()

        print('=' * 70)
        print('✅ ALL TESTS PASSED')
        print('=' * 70)
    except AssertionError as e:
        print('=' * 70)
        print(f'❌ TEST FAILED: {e}')
        print('=' * 70)
        sys.exit(1)
