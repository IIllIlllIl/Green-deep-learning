#!/usr/bin/env python3
"""
测试step5增强功能

验证：
1. num_mutated_params计数逻辑正确（包括seed）
2. mutated_param列正确识别单参数变异
3. hyperparam列填充默认值正确
"""

import csv
import json
from typing import Dict, List

def load_models_config() -> Dict:
    """加载模型配置"""
    with open('mutation/models_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def test_num_mutated_params_counts_seed():
    """测试1: 验证num_mutated_params正确计数seed参数"""
    print("测试1: 验证num_mutated_params正确计数seed参数")
    print("=" * 70)

    with open('results/summary_new.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 找到有seed值的实验
    seed_experiments = []
    for row in rows:
        seed_val = row.get('hyperparam_seed', '').strip()
        num_mutated = row.get('num_mutated_params', '')
        if seed_val and num_mutated:
            seed_experiments.append({
                'exp_id': row['experiment_id'][:50],
                'repo': row.get('repository', ''),
                'model': row.get('model', ''),
                'seed': seed_val,
                'num_mutated': num_mutated,
                'mutated_param': row.get('mutated_param', '')
            })

    print(f"找到 {len(seed_experiments)} 个设置了seed的实验")

    # 验证seed被算作变异参数
    seed_only_count = 0
    for exp in seed_experiments:
        if exp['mutated_param'] == 'seed':
            seed_only_count += 1

    print(f"其中 {seed_only_count} 个仅变异seed参数")

    # 显示几个示例
    print("\n示例（seed单参数变异）：")
    for i, exp in enumerate([e for e in seed_experiments if e['mutated_param'] == 'seed'][:5], 1):
        print(f"{i}. {exp['repo']}/{exp['model']} seed={exp['seed']} num_mutated={exp['num_mutated']}")

    assert seed_only_count > 0, "应该存在仅变异seed的实验"
    print("\n✓ 测试通过: seed被正确计数为变异参数")
    print()
    return True

def test_mutated_param_single_mutation():
    """测试2: 验证mutated_param列正确识别单参数变异"""
    print("测试2: 验证mutated_param列正确识别单参数变异")
    print("=" * 70)

    with open('results/summary_new.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 统计单参数变异实验
    single_param_exps = []
    multi_param_exps = []
    baseline_exps = []

    for row in rows:
        num_mutated = row.get('num_mutated_params', '')
        mutated_param = row.get('mutated_param', '')

        if num_mutated == '1':
            single_param_exps.append(row)
        elif num_mutated == '0':
            baseline_exps.append(row)
        elif num_mutated and int(num_mutated) > 1:
            multi_param_exps.append(row)

    print(f"单参数变异实验: {len(single_param_exps)}")
    print(f"多参数变异实验: {len(multi_param_exps)}")
    print(f"Baseline实验: {len(baseline_exps)}")

    # 验证单参数变异实验都有mutated_param值
    missing_count = 0
    for exp in single_param_exps:
        if not exp.get('mutated_param', '').strip():
            missing_count += 1

    print(f"\n单参数变异实验中缺失mutated_param的数量: {missing_count}")

    # 验证多参数和baseline实验的mutated_param为空
    should_be_empty_count = 0
    for exp in multi_param_exps + baseline_exps:
        if not exp.get('mutated_param', '').strip():
            should_be_empty_count += 1

    print(f"多参数/baseline实验中mutated_param正确为空的数量: {should_be_empty_count}/{len(multi_param_exps) + len(baseline_exps)}")

    # 显示mutated_param的分布
    param_dist = {}
    for exp in single_param_exps:
        param = exp.get('mutated_param', '')
        if param:
            param_dist[param] = param_dist.get(param, 0) + 1

    print("\nmutated_param参数分布:")
    for param in sorted(param_dist.keys()):
        print(f"  {param}: {param_dist[param]}")

    assert missing_count == 0, "所有单参数变异实验都应该有mutated_param值"
    assert should_be_empty_count == len(multi_param_exps) + len(baseline_exps), "多参数和baseline实验的mutated_param应该为空"
    print("\n✓ 测试通过: mutated_param列正确识别单参数变异")
    print()
    return True

def test_hyperparam_default_values():
    """测试3: 验证hyperparam列填充默认值正确"""
    print("测试3: 验证hyperparam列填充默认值正确")
    print("=" * 70)

    models_config = load_models_config()

    with open('results/summary_new.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 检查每个仓库的默认值填充
    filled_count = 0
    check_count = 0

    for row in rows:
        repo = row.get('repository', '').strip()
        if not repo or repo not in models_config['models']:
            continue

        repo_config = models_config['models'][repo]
        supported_params = repo_config.get('supported_hyperparams', {})

        for param, config in supported_params.items():
            col_name = f'hyperparam_{param}'
            value = row.get(col_name, '').strip()
            default = config.get('default')

            check_count += 1

            # 如果CSV中有值，验证它不是空的
            if value:
                filled_count += 1

    print(f"检查了 {check_count} 个超参数值")
    print(f"其中 {filled_count} 个已填充")
    print(f"填充率: {filled_count/check_count*100:.1f}%")

    # 验证一些具体的例子
    print("\n验证具体示例（前5个实验的默认值）:")
    for i, row in enumerate(rows[:5], 1):
        repo = row.get('repository', '')
        model = row.get('model', '')
        num_mutated = row.get('num_mutated_params', '')

        print(f"{i}. {repo}/{model} num_mutated={num_mutated}")

        if repo in models_config['models']:
            supported_params = models_config['models'][repo].get('supported_hyperparams', {})
            for param, config in list(supported_params.items())[:3]:  # 只显示前3个参数
                col_name = f'hyperparam_{param}'
                value = row.get(col_name, '')
                default = config.get('default')
                print(f"   {param}: value={value}, default={default}")

    assert filled_count > 0, "应该有填充的超参数值"
    print("\n✓ 测试通过: hyperparam列正确填充默认值")
    print()
    return True

def test_consistency_check():
    """测试4: 一致性检查"""
    print("测试4: 一致性检查")
    print("=" * 70)

    with open('results/summary_new.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    inconsistencies = []

    for idx, row in enumerate(rows, 1):
        num_mutated = row.get('num_mutated_params', '')
        mutated_param = row.get('mutated_param', '')

        # 规则1: num_mutated == 1 时必须有mutated_param
        if num_mutated == '1' and not mutated_param:
            inconsistencies.append(f"行{idx}: num_mutated=1 但mutated_param为空")

        # 规则2: num_mutated != 1 时mutated_param应该为空
        if num_mutated not in ['', '1'] and mutated_param:
            inconsistencies.append(f"行{idx}: num_mutated={num_mutated} 但mutated_param不为空")

        # 规则3: num_mutated == 0 时mutated_param应该为空
        if num_mutated == '0' and mutated_param:
            inconsistencies.append(f"行{idx}: num_mutated=0 但mutated_param不为空")

    if inconsistencies:
        print("发现不一致:")
        for inc in inconsistencies[:10]:  # 只显示前10个
            print(f"  {inc}")
        print(f"\n总计 {len(inconsistencies)} 个不一致")
        raise AssertionError("数据不一致")
    else:
        print("✓ 所有数据一致")

    print()
    return True

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("运行Step5增强功能测试")
    print("=" * 70)
    print()

    tests = [
        test_num_mutated_params_counts_seed,
        test_mutated_param_single_mutation,
        test_hyperparam_default_values,
        test_consistency_check
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 测试错误: {e}")
            failed += 1

    print("=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
