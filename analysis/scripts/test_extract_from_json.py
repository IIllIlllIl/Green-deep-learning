#!/usr/bin/env python3
"""测试extract_from_json_with_defaults.py的核心功能

用途: 验证数据提取脚本的正确性
作者: Claude
日期: 2025-12-24
"""

import json
import sys
from pathlib import Path

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from extract_from_json_with_defaults import (
    apply_field_mapping,
    extract_complete_hyperparams,
    extract_energy_metrics,
    extract_performance_metrics,
    load_models_config
)


def test_field_mapping():
    """测试字段映射功能"""
    print("=" * 80)
    print("测试1: 字段映射")
    print("=" * 80)

    test_cases = [
        # (repo, param_name, expected)
        ('bug-localization-by-dnn-and-rvsm', 'max_iter', 'training_duration'),
        ('bug-localization-by-dnn-and-rvsm', 'alpha', 'l2_regularization'),
        ('VulBERTa', 'epochs', 'training_duration'),
        ('VulBERTa', 'learning_rate', 'hyperparam_learning_rate'),
        ('VulBERTa', 'weight_decay', 'l2_regularization'),
        ('examples', 'batch_size', 'hyperparam_batch_size'),
        ('examples', 'seed', 'seed'),
        ('MRT-OAST', 'alpha', 'l2_regularization'),
    ]

    passed = 0
    failed = 0

    for repo, param, expected in test_cases:
        result = apply_field_mapping(repo, param)
        if result == expected:
            print(f"✅ {repo}.{param} → {result}")
            passed += 1
        else:
            print(f"❌ {repo}.{param} → {result} (期望: {expected})")
            failed += 1

    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_hyperparam_extraction():
    """测试超参数提取功能（使用真实JSON）"""
    print("\n" + "=" * 80)
    print("测试2: 超参数提取（真实数据）")
    print("=" * 80)

    # 加载models_config
    models_config = load_models_config()

    # 测试用例：Bug定位默认值实验（只记录seed）
    test_json_path = Path('../../results/run_20251126_224751/bug-localization-by-dnn-and-rvsm_default_017/experiment.json')
    if not test_json_path.exists():
        test_json_path = Path('../results/run_20251126_224751/bug-localization-by-dnn-and-rvsm_default_017/experiment.json')

    if not test_json_path.exists():
        print(f"⚠️ 测试文件不存在: {test_json_path}")
        # 使用模拟数据
        test_exp = {
            'repository': 'bug-localization-by-dnn-and-rvsm',
            'model': 'default',
            'hyperparameters': {'seed': 917},  # 只记录seed
        }
    else:
        with open(test_json_path) as f:
            test_exp = json.load(f)

    print(f"测试实验: {test_exp.get('experiment_id', 'mock_exp')}")
    print(f"记录的超参数: {test_exp.get('hyperparameters', {})}")

    # 提取完整超参数
    complete_params = extract_complete_hyperparams(test_exp, models_config)

    print(f"\n提取的完整超参数:")
    for key, value in sorted(complete_params.items()):
        print(f"  {key}: {value}")

    # 验证
    expected_keys = {'training_duration', 'l2_regularization', 'hyperparam_kfold', 'seed'}
    actual_keys = set(complete_params.keys())

    if expected_keys == actual_keys:
        print(f"\n✅ 提取的字段正确: {expected_keys}")
    else:
        print(f"\n❌ 字段不匹配")
        print(f"  期望: {expected_keys}")
        print(f"  实际: {actual_keys}")
        print(f"  缺少: {expected_keys - actual_keys}")
        print(f"  多余: {actual_keys - expected_keys}")
        return False

    # 验证默认值
    if complete_params.get('training_duration') == 10000:
        print(f"✅ training_duration默认值正确: 10000")
    else:
        print(f"❌ training_duration默认值错误: {complete_params.get('training_duration')}")
        return False

    if complete_params.get('seed') == 917:
        print(f"✅ seed记录值正确: 917 (覆盖默认值42)")
    else:
        print(f"❌ seed记录值错误: {complete_params.get('seed')}")
        return False

    return True


def test_parallel_mode():
    """测试并行模式数据提取"""
    print("\n" + "=" * 80)
    print("测试3: 并行模式提取")
    print("=" * 80)

    models_config = load_models_config()

    # 查找并行模式实验
    test_json_path = Path('../../results/run_20251222_214929/bug-localization-by-dnn-and-rvsm_default_001_parallel/experiment.json')
    if not test_json_path.exists():
        test_json_path = Path('../results/run_20251222_214929/bug-localization-by-dnn-and-rvsm_default_001_parallel/experiment.json')

    if not test_json_path.exists():
        print(f"⚠️ 测试文件不存在: {test_json_path}")
        # 使用模拟数据
        test_exp = {
            'experiment_id': 'bug-localization-by-dnn-and-rvsm_default_001_parallel',
            'mode': 'parallel',
            'foreground': {
                'repository': 'bug-localization-by-dnn-and-rvsm',
                'model': 'default',
                'hyperparameters': {},  # 空的
                'energy_metrics': {'cpu_energy_total_joules': 100},
                'performance_metrics': {'top1_accuracy': 0.5}
            }
        }
    else:
        with open(test_json_path) as f:
            test_exp = json.load(f)

    print(f"测试实验: {test_exp.get('experiment_id', 'mock_parallel')}")
    print(f"模式: {test_exp.get('mode')}")
    print(f"foreground.hyperparameters: {test_exp.get('foreground', {}).get('hyperparameters', {})}")

    # 提取完整超参数
    complete_params = extract_complete_hyperparams(test_exp, models_config)

    print(f"\n提取的完整超参数:")
    for key, value in sorted(complete_params.items()):
        print(f"  {key}: {value}")

    # 验证：并行模式应该也能提取完整超参数
    if len(complete_params) > 0:
        print(f"\n✅ 并行模式成功提取 {len(complete_params)} 个超参数")
        return True
    else:
        print(f"\n❌ 并行模式提取失败")
        return False


def test_energy_extraction():
    """测试能耗指标提取"""
    print("\n" + "=" * 80)
    print("测试4: 能耗指标提取")
    print("=" * 80)

    # 使用模拟数据
    test_exp = {
        'energy_metrics': {
            'cpu_energy_total_joules': 50000.0,
            'gpu_energy_total_joules': 100000.0,
            'gpu_power_avg_watts': 200.0,
            'gpu_util_avg_percent': 85.5,
            'gpu_temp_max_celsius': 75.0,
        }
    }

    energy = extract_energy_metrics(test_exp)

    print("提取的能耗指标:")
    for key, value in sorted(energy.items()):
        print(f"  {key}: {value}")

    # 验证
    if energy.get('energy_cpu_total_joules') == 50000.0:
        print("\n✅ 能耗指标提取正确")
        return True
    else:
        print("\n❌ 能耗指标提取失败")
        return False


def dry_run_test():
    """Dry run测试：处理少量数据"""
    print("\n" + "=" * 80)
    print("测试5: Dry Run（前10个实验）")
    print("=" * 80)

    from extract_from_json_with_defaults import load_all_experiments, experiments_to_dataframe

    models_config = load_models_config()
    results_dir = Path('../../results')
    if not results_dir.exists():
        results_dir = Path('../results')

    # 加载前10个实验
    all_experiments = load_all_experiments(results_dir)
    test_experiments = all_experiments[:10]

    print(f"加载 {len(test_experiments)} 个测试实验")

    # 转换为DataFrame
    df = experiments_to_dataframe(test_experiments, models_config)

    print(f"\n生成的DataFrame:")
    print(f"  - 行数: {len(df)}")
    print(f"  - 列数: {len(df.columns)}")

    # 检查超参数列
    hyperparam_cols = [c for c in df.columns if
                       c.startswith('hyperparam_') or
                       c in ['training_duration', 'l2_regularization', 'seed']]

    print(f"\n超参数列 ({len(hyperparam_cols)} 个):")
    for col in sorted(hyperparam_cols):
        missing = df[col].isnull().sum()
        print(f"  {col}: {len(df) - missing}/{len(df)} 填充 ({missing} 缺失)")

    # 显示前3行
    print(f"\n前3行数据预览:")
    print(df[['experiment_id', 'repository', 'mode'] + hyperparam_cols[:5]].head(3).to_string())

    return True


def main():
    """运行所有测试"""
    print("=" * 80)
    print("数据提取脚本测试套件")
    print("=" * 80)

    tests = [
        ("字段映射", test_field_mapping),
        ("超参数提取（真实数据）", test_hyperparam_extraction),
        ("并行模式提取", test_parallel_mode),
        ("能耗指标提取", test_energy_extraction),
        ("Dry Run", dry_run_test),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")

    print(f"\n总计: {passed_count}/{total_count} 通过")

    if passed_count == total_count:
        print("\n✅ 所有测试通过！可以执行完整提取")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} 个测试失败，需要修复")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
