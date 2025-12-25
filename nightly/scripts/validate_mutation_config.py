#!/usr/bin/env python3
"""
验证 mutation 配置 JSON 是否能正确调用脚本中的变异方法
"""

import json
import sys
from pathlib import Path

# Add mutation package to path
sys.path.insert(0, str(Path(__file__).parent))

from mutation.runner import MutationRunner
from mutation.hyperparams import generate_mutations

def validate_json_format(config_file):
    """验证 JSON 格式正确性"""
    print("=" * 80)
    print(f"验证配置文件: {config_file}")
    print("=" * 80)

    config_path = Path(__file__).parent / config_file

    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ JSON 格式正确")

        # Check required fields
        required_fields = ["experiment_name", "experiments"]
        for field in required_fields:
            if field not in config:
                print(f"❌ 错误: 缺少必需字段 '{field}'")
                return False
            else:
                print(f"✅ 包含字段 '{field}'")

        # Print summary
        print(f"\n配置摘要:")
        print(f"  实验名称: {config.get('experiment_name')}")
        print(f"  描述: {config.get('description')}")
        print(f"  总配置数: {len(config['experiments'])}")
        print(f"  每配置运行次数: {config.get('runs_per_config', 1)}")
        print(f"  Governor: {config.get('governor', 'None')}")
        print(f"  模式: {config.get('mode', 'default')}")

        return True, config

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {e}")
        return False, None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False, None

def validate_mutation_mode_experiments(config):
    """验证 mutation 模式的实验配置"""
    print("\n" + "=" * 80)
    print("验证 Mutation 模式实验")
    print("=" * 80)

    experiments = config.get("experiments", [])
    mutation_count = 0
    default_count = 0
    parallel_count = 0

    for idx, exp in enumerate(experiments, 1):
        exp_mode = exp.get("mode")

        if exp_mode == "parallel":
            parallel_count += 1
            fg_config = exp.get("foreground", {})
            fg_mode = fg_config.get("mode")

            if fg_mode == "mutation":
                print(f"\n实验 {idx}: Parallel + Mutation (foreground)")
                print(f"  前台: {fg_config.get('repo')}/{fg_config.get('model')}")
                print(f"  变异参数: {fg_config.get('mutate')}")

                bg_config = exp.get("background", {})
                print(f"  后台: {bg_config.get('repo')}/{bg_config.get('model')}")
                mutation_count += 1

        elif exp_mode == "mutation":
            mutation_count += 1
            print(f"\n实验 {idx}: Mutation (sequential)")
            print(f"  模型: {exp.get('repo')}/{exp.get('model')}")
            print(f"  变异参数: {exp.get('mutate')}")

        elif exp_mode == "default":
            default_count += 1

    print(f"\n统计:")
    print(f"  Mutation 模式实验: {mutation_count}")
    print(f"  Default 模式实验: {default_count}")
    print(f"  Parallel 模式实验: {parallel_count}")

    return mutation_count > 0

def test_mutation_generation(config):
    """测试是否能正确调用 generate_mutations"""
    print("\n" + "=" * 80)
    print("测试变异生成功能")
    print("=" * 80)

    # Load models config
    models_config_path = Path(__file__).parent / "mutation" / "models_config.json"
    with open(models_config_path, 'r') as f:
        models_config = json.load(f)

    experiments = config.get("experiments", [])
    runs_per_config = config.get("runs_per_config", 1)

    # Test first few mutation experiments
    test_count = 0
    max_tests = 5

    for idx, exp in enumerate(experiments, 1):
        if test_count >= max_tests:
            break

        exp_mode = exp.get("mode")

        # Handle parallel mode
        if exp_mode == "parallel":
            fg_config = exp.get("foreground", {})
            if fg_config.get("mode") == "mutation":
                repo = fg_config.get("repo")
                model = fg_config.get("model")
                mutate_params = fg_config.get("mutate", [])
            else:
                continue
        elif exp_mode == "mutation":
            repo = exp.get("repo")
            model = exp.get("model")
            mutate_params = exp.get("mutate", [])
        else:
            continue

        # Get repository configuration
        if repo not in models_config["models"]:
            print(f"\n❌ 实验 {idx}: 仓库 '{repo}' 未在 models_config.json 中找到")
            continue

        repo_config = models_config["models"][repo]
        supported_params = repo_config["supported_hyperparams"]

        print(f"\n✅ 实验 {idx}: {repo}/{model}")
        print(f"   变异参数: {mutate_params}")
        print(f"   运行次数: {runs_per_config}")

        try:
            # Test mutation generation
            mutations = generate_mutations(
                supported_params=supported_params,
                mutate_params=mutate_params,
                num_mutations=runs_per_config
            )

            print(f"   ✅ 成功生成 {len(mutations)} 个变异")
            for i, mut in enumerate(mutations, 1):
                print(f"      变异 {i}: {mut}")

            test_count += 1

        except Exception as e:
            print(f"   ❌ 生成变异失败: {e}")
            import traceback
            traceback.print_exc()

    if test_count > 0:
        print(f"\n✅ 成功测试 {test_count} 个 mutation 实验")
        return True
    else:
        print(f"\n❌ 没有成功测试任何 mutation 实验")
        return False

def validate_config_file(config_file):
    """完整验证配置文件"""
    print("\n\n" + "=" * 80)
    print(f"完整验证: {config_file}")
    print("=" * 80)

    # Step 1: Validate JSON format
    valid, config = validate_json_format(config_file)
    if not valid:
        return False

    # Step 2: Validate mutation mode experiments
    has_mutation = validate_mutation_mode_experiments(config)
    if not has_mutation:
        print("\n⚠️ 警告: 配置文件中没有 mutation 模式实验")

    # Step 3: Test mutation generation
    success = test_mutation_generation(config)

    print("\n" + "=" * 80)
    if success:
        print("✅ 配置验证成功！可以正确调用变异方法")
    else:
        print("❌ 配置验证失败")
    print("=" * 80)

    return success

if __name__ == "__main__":
    # Test both config files
    config_files = [
        "settings/mutation_validation_1x.json",
        "settings/mutation_all_models_3x_dynamic.json"
    ]

    results = {}
    for config_file in config_files:
        print("\n\n" + "█" * 80)
        print(f"█ 验证配置文件: {config_file}")
        print("█" * 80)

        results[config_file] = validate_config_file(config_file)

    # Final summary
    print("\n\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)

    for config_file, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {config_file}")

    if all(results.values()):
        print("\n🎉 所有配置文件验证通过！")
    else:
        print("\n⚠️ 部分配置文件验证失败")
