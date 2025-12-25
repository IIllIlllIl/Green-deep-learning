#!/usr/bin/env python3
"""
验证 mutation 框架的变异方法
"""

import json
import sys
from pathlib import Path

# Add mutation package to path
sys.path.insert(0, str(Path(__file__).parent))

from mutation.hyperparams import generate_mutations, mutate_hyperparameter

def test_mutation_ranges():
    """测试各模型的变异范围是否符合预期"""

    # Load models config
    config_path = Path(__file__).parent / "mutation" / "models_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=" * 80)
    print("MUTATION 框架变异方法验证")
    print("=" * 80)

    # Test each model
    test_models = [
        ("examples", "mnist_ff"),
        ("pytorch_resnet_cifar10", "resnet20"),
        ("MRT-OAST", "default"),
        ("VulBERTa", "mlp"),
        ("bug-localization-by-dnn-and-rvsm", "default"),
        ("Person_reID_baseline_pytorch", "densenet121"),
    ]

    for repo, model in test_models:
        print(f"\n{'─' * 80}")
        print(f"模型: {repo}/{model}")
        print(f"{'─' * 80}")

        repo_config = config["models"][repo]
        supported_params = repo_config["supported_hyperparams"]

        # Print parameter configurations
        for param_name, param_config in supported_params.items():
            print(f"\n参数: {param_name}")
            print(f"  默认值: {param_config.get('default')}")
            print(f"  范围: {param_config['range']}")
            print(f"  分布: {param_config['distribution']}")
            print(f"  类型: {param_config['type']}")

            # Check if range matches expected multiplier pattern
            default = param_config.get('default')
            range_min, range_max = param_config['range']

            if default is not None and default > 0:
                multiplier_min = range_min / default
                multiplier_max = range_max / default
                print(f"  倍数范围: {multiplier_min:.2f}× ~ {multiplier_max:.2f}×")

            # Generate sample mutations
            print(f"\n  生成10个样本变异值:")
            for i in range(10):
                value = mutate_hyperparameter(param_config, param_name)
                if param_config['type'] == 'int':
                    print(f"    样本 {i+1}: {value}")
                else:
                    print(f"    样本 {i+1}: {value:.6f}")

def test_generate_mutations():
    """测试生成多个变异的功能"""

    print("\n\n" + "=" * 80)
    print("测试 generate_mutations 函数")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent / "mutation" / "models_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Test mnist_ff
    repo_config = config["models"]["examples"]
    supported_params = repo_config["supported_hyperparams"]

    print("\n测试: 为 mnist_ff 生成 3 个 epochs 变异")
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["epochs"],
        num_mutations=3
    )

    print("\n测试: 为 mnist_ff 生成 3 个 learning_rate 变异")
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["learning_rate"],
        num_mutations=3
    )

    print("\n测试: 为 mnist_ff 生成 3 个 seed 变异")
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["seed"],
        num_mutations=3
    )

    # Test resnet20
    repo_config = config["models"]["pytorch_resnet_cifar10"]
    supported_params = repo_config["supported_hyperparams"]

    print("\n测试: 为 resnet20 生成 3 个 epochs 变异")
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["epochs"],
        num_mutations=3
    )

    print("\n测试: 为 resnet20 生成 3 个 weight_decay 变异")
    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=["weight_decay"],
        num_mutations=3
    )

def compare_with_expected():
    """对比实际变异范围与预期"""

    print("\n\n" + "=" * 80)
    print("对比实际变异范围与预期")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent / "mutation" / "models_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n预期的变异策略（根据文档）:")
    print("  - epochs/max_iter: Log-uniform [default×0.5, default×2.0]")
    print("  - learning_rate/alpha: Log-uniform [default×0.1, default×10.0]")
    print("  - dropout: Uniform [0.0, 0.7]")
    print("  - weight_decay: 30% zero + 70% log-uniform [0.0, default×100]")
    print("  - seed: Uniform integer [0, 9999]")

    print("\n实际配置的范围:")

    # Check examples
    examples_params = config["models"]["examples"]["supported_hyperparams"]
    print("\nexamples (mnist_ff, mnist, etc.):")
    print(f"  epochs: range={examples_params['epochs']['range']}, default={examples_params['epochs']['default']}")
    print(f"    → 实际倍数: {examples_params['epochs']['range'][0]/examples_params['epochs']['default']:.2f}× ~ {examples_params['epochs']['range'][1]/examples_params['epochs']['default']:.2f}×")
    print(f"  learning_rate: range={examples_params['learning_rate']['range']}, default={examples_params['learning_rate']['default']}")
    print(f"    → 实际倍数: {examples_params['learning_rate']['range'][0]/examples_params['learning_rate']['default']:.2f}× ~ {examples_params['learning_rate']['range'][1]/examples_params['learning_rate']['default']:.2f}×")
    print(f"  seed: range={examples_params['seed']['range']}, default={examples_params['seed']['default']}")

    # Check resnet20
    resnet_params = config["models"]["pytorch_resnet_cifar10"]["supported_hyperparams"]
    print("\npytorch_resnet_cifar10 (resnet20):")
    print(f"  epochs: range={resnet_params['epochs']['range']}, default={resnet_params['epochs']['default']}")
    print(f"    → 实际倍数: {resnet_params['epochs']['range'][0]/resnet_params['epochs']['default']:.2f}× ~ {resnet_params['epochs']['range'][1]/resnet_params['epochs']['default']:.2f}×")
    print(f"  learning_rate: range={resnet_params['learning_rate']['range']}, default={resnet_params['learning_rate']['default']}")
    print(f"    → 实际倍数: {resnet_params['learning_rate']['range'][0]/resnet_params['learning_rate']['default']:.2f}× ~ {resnet_params['learning_rate']['range'][1]/resnet_params['learning_rate']['default']:.2f}×")
    print(f"  weight_decay: range={resnet_params['weight_decay']['range']}, default={resnet_params['weight_decay']['default']}")
    print(f"    → 实际倍数: {resnet_params['weight_decay']['range'][0]/resnet_params['weight_decay']['default']:.2f}× ~ {resnet_params['weight_decay']['range'][1]/resnet_params['weight_decay']['default']:.2f}×")

if __name__ == "__main__":
    # Run tests
    test_mutation_ranges()
    test_generate_mutations()
    compare_with_expected()

    print("\n\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)
