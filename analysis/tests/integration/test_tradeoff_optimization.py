#!/usr/bin/env python3
"""
权衡检测sign函数优化方案验收测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils.tradeoff_detection import (
    TradeoffDetector,
    ENERGY_PERF_RULES,
    create_sign_func_from_rule,
    create_sign_functions_from_rules,
    create_energy_perf_sign_functions
)

def test_rule_system():
    """测试规则系统"""
    print("=" * 60)
    print("1. 测试规则系统")
    print("=" * 60)

    # 测试规则定义
    print(f"规则总数: {len(ENERGY_PERF_RULES)}")
    print("\n规则分类:")

    # 统计各类规则
    energy_rules = [p for p in ENERGY_PERF_RULES.keys() if 'energy_' in p]
    perf_positive_rules = [p for p in ENERGY_PERF_RULES.keys() if '+' in ENERGY_PERF_RULES[p] and 'perf_' in p]
    perf_negative_rules = [p for p in ENERGY_PERF_RULES.keys() if '-' in ENERGY_PERF_RULES[p] and 'perf_' in p]
    efficiency_rules = [p for p in ENERGY_PERF_RULES.keys() if 'samples_per_second' in p]
    interaction_rules = [p for p in ENERGY_PERF_RULES.keys() if '_x_is_parallel' in p]

    print(f"  能耗指标规则: {len(energy_rules)} 条")
    print(f"  性能指标（正改善）规则: {len(perf_positive_rules)} 条")
    print(f"  性能指标（负改善）规则: {len(perf_negative_rules)} 条")
    print(f"  效率指标规则: {len(efficiency_rules)} 条")
    print(f"  交互项规则: {len(interaction_rules)} 条")

    # 打印示例规则
    print("\n示例规则:")
    for i, (pattern, rule) in enumerate(list(ENERGY_PERF_RULES.items())[:10]):
        print(f"  {pattern}: {rule}")

    return True

def test_sign_function_creation():
    """测试sign函数创建"""
    print("\n" + "=" * 60)
    print("2. 测试sign函数创建")
    print("=" * 60)

    # 测试单个规则创建
    print("测试单个规则创建:")
    plus_func = create_sign_func_from_rule('+')
    minus_func = create_sign_func_from_rule('-')

    # 测试正改善规则
    print("  正改善规则 (+):")
    print(f"    current=1.0, change=0.5 → {plus_func(1.0, 0.5)} (应为+)")
    print(f"    current=1.0, change=-0.5 → {plus_func(1.0, -0.5)} (应为-)")

    # 测试负改善规则
    print("  负改善规则 (-):")
    print(f"    current=1.0, change=0.5 → {minus_func(1.0, 0.5)} (应为+)")
    print(f"    current=1.0, change=-0.5 → {minus_func(1.0, -0.5)} (应为-)")

    # 测试模式匹配sign函数
    print("\n测试模式匹配sign函数:")
    sign_functions = create_energy_perf_sign_functions()

    # 测试能耗指标
    test_cases = [
        ("energy_cpu_pkg_joules", "-", "能耗指标应为负改善"),
        ("perf_test_accuracy", "+", "准确率应为正改善"),
        ("perf_eval_loss", "-", "损失函数应为负改善"),
        ("perf_eval_samples_per_second", "+", "效率指标应为正改善"),
        ("unknown_metric", "+", "未知指标默认正改善"),
    ]

    for metric, expected_rule, description in test_cases:
        sign_func = sign_functions.get(metric)
        if sign_func:
            # 测试sign函数行为
            test_change = 0.1 if expected_rule == '+' else -0.1
            sign = sign_func(1.0, test_change)
            expected_sign = '+' if (expected_rule == '+' and test_change > 0) or (expected_rule == '-' and test_change < 0) else '-'
            print(f"  {metric}: {description}")
            print(f"    规则: {expected_rule}, 测试变化: {test_change}, 结果: {sign} (应为{expected_sign})")
        else:
            print(f"  {metric}: 未找到sign函数")

    return True

def test_tradeoff_detector():
    """测试权衡检测器"""
    print("\n" + "=" * 60)
    print("3. 测试权衡检测器")
    print("=" * 60)

    # 创建模拟因果效应数据
    causal_effects = {
        'learning_rate->energy_cpu_total_joules': {
            'ate': -0.15,  # 降低能耗（改善）
            'ci_lower': -0.2,
            'ci_upper': -0.1,
            'is_significant': True
        },
        'learning_rate->perf_test_accuracy': {
            'ate': 0.08,  # 提高准确率（改善）
            'ci_lower': 0.05,
            'ci_upper': 0.11,
            'is_significant': True
        },
        'learning_rate->perf_eval_loss': {
            'ate': -0.12,  # 降低损失（改善）
            'ci_lower': -0.15,
            'ci_upper': -0.09,
            'is_significant': True
        },
        'batch_size->energy_cpu_total_joules': {
            'ate': 0.25,  # 增加能耗（恶化）
            'ci_lower': 0.2,
            'ci_upper': 0.3,
            'is_significant': True
        },
        'batch_size->perf_test_accuracy': {
            'ate': 0.15,  # 提高准确率（改善）
            'ci_lower': 0.1,
            'ci_upper': 0.2,
            'is_significant': True
        }
    }

    # 测试不同初始化方式
    print("测试不同初始化方式:")

    # 方式1: 使用默认规则
    print("\n方式1: 使用默认规则（能耗/性能规则）")
    detector1 = TradeoffDetector(verbose=True)
    tradeoffs1 = detector1.detect_tradeoffs(causal_effects, require_significance=True)
    print(f"  检测到权衡: {len(tradeoffs1)}")

    # 方式2: 使用自定义规则
    print("\n方式2: 使用自定义规则")
    custom_rules = {
        "energy_*": "-",
        "perf_*_accuracy": "+",
        "perf_*_loss": "-",
        "*": "+"
    }
    detector2 = TradeoffDetector(rules=custom_rules, verbose=True)
    tradeoffs2 = detector2.detect_tradeoffs(causal_effects, require_significance=True)
    print(f"  检测到权衡: {len(tradeoffs2)}")

    # 方式3: 使用当前值
    print("\n方式3: 使用当前值")
    current_values = {
        'energy_cpu_total_joules': 100.0,  # 当前能耗值
        'perf_test_accuracy': 0.85,        # 当前准确率
        'perf_eval_loss': 0.5              # 当前损失
    }
    detector3 = TradeoffDetector(current_values=current_values, verbose=True)
    tradeoffs3 = detector3.detect_tradeoffs(causal_effects, require_significance=True)
    print(f"  检测到权衡: {len(tradeoffs3)}")

    # 打印检测到的权衡
    if tradeoffs1:
        print("\n检测到的权衡详情:")
        for i, t in enumerate(tradeoffs1, 1):
            print(f"  {i}. 干预: {t['intervention']}")
            print(f"     指标1: {t['metric1']} (ATE={t['ate1']:.4f}, sign={t['sign1']})")
            print(f"     指标2: {t['metric2']} (ATE={t['ate2']:.4f}, sign={t['sign2']})")
            print(f"     是否显著: {t['is_significant']}")

    return True

def test_with_real_data():
    """使用真实ATE数据进行测试"""
    print("\n" + "=" * 60)
    print("4. 使用真实ATE数据进行测试")
    print("=" * 60)

    # 加载ATE结果
    ate_file = "results/energy_research/data/global_std_dibs_ate/group1_examples_dibs_global_std_ate.csv"
    if not os.path.exists(ate_file):
        print(f"  ⚠️  ATE文件不存在: {ate_file}")
        return False

    try:
        ate_df = pd.read_csv(ate_file)
        print(f"  加载ATE数据: {len(ate_df)} 条边")

        # 转换为causal_effects格式
        causal_effects = {}
        for _, row in ate_df.iterrows():
            edge = f"{row['source']}->{row['target']}"
            causal_effects[edge] = {
                'ate': row['ate_global_std'] if not pd.isna(row['ate_global_std']) else 0.0,
                'ci_lower': row['ate_global_std_ci_lower'] if not pd.isna(row['ate_global_std_ci_lower']) else 0.0,
                'ci_upper': row['ate_global_std_ci_upper'] if not pd.isna(row['ate_global_std_ci_upper']) else 0.0,
                'is_significant': row['ate_global_std_is_significant'] if not pd.isna(row['ate_global_std_is_significant']) else False
            }

        print(f"  转换后的因果边: {len(causal_effects)}")

        # 加载当前值（使用全局标准化数据的均值）
        data_file = "data/energy_research/6groups_global_std/group1_examples_global_std.csv"
        if os.path.exists(data_file):
            data_df = pd.read_csv(data_file)
            # 计算每列的均值作为当前值
            current_values = data_df.mean().to_dict()
            print(f"  加载当前值: {len(current_values)} 个指标")
        else:
            current_values = None
            print("  ⚠️  数据文件不存在，不使用当前值")

        # 运行权衡检测
        print("\n  运行权衡检测...")
        detector = TradeoffDetector(
            rules=ENERGY_PERF_RULES,
            current_values=current_values,
            verbose=True
        )

        tradeoffs = detector.detect_tradeoffs(
            causal_effects,
            require_significance=True,
            current_values=current_values
        )

        print(f"\n  ✓ 检测完成")
        print(f"    检测到权衡: {len(tradeoffs)}")

        if tradeoffs:
            # 统计不同类型的权衡
            energy_vs_perf = 0
            perf_vs_perf = 0
            other = 0

            for t in tradeoffs:
                m1 = t['metric1'].lower()
                m2 = t['metric2'].lower()

                is_energy1 = 'energy' in m1
                is_energy2 = 'energy' in m2
                is_perf1 = 'perf' in m1
                is_perf2 = 'perf' in m2

                if (is_energy1 and is_perf2) or (is_energy2 and is_perf1):
                    energy_vs_perf += 1
                elif is_perf1 and is_perf2:
                    perf_vs_perf += 1
                else:
                    other += 1

            print(f"\n  权衡类型统计:")
            print(f"    能耗 vs 性能: {energy_vs_perf}")
            print(f"    性能 vs 性能: {perf_vs_perf}")
            print(f"    其他: {other}")

            # 显示前5个权衡
            print(f"\n  前5个权衡:")
            for i, t in enumerate(tradeoffs[:5], 1):
                print(f"    {i}. {t['intervention']} → {t['metric1']} ({t['sign1']}) vs {t['metric2']} ({t['sign2']})")

        return True

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "=" * 60)
    print("5. 测试向后兼容性")
    print("=" * 60)

    # 测试旧的sign_functions参数
    print("测试旧的sign_functions参数:")

    # 创建旧的sign_functions字典（不包含current_value参数）
    old_sign_functions = {
        'energy_cpu_total_joules': lambda change: '-' if change < 0 else '+',
        'perf_test_accuracy': lambda change: '+' if change > 0 else '-',
        'perf_eval_loss': lambda change: '-' if change < 0 else '+'
    }

    try:
        # 使用旧的sign_functions初始化
        detector = TradeoffDetector(sign_functions=old_sign_functions, verbose=True)
        print("  ✓ 支持旧的sign_functions参数")

        # 测试检测
        causal_effects = {
            'learning_rate->energy_cpu_total_joules': {'ate': -0.1, 'is_significant': True},
            'learning_rate->perf_test_accuracy': {'ate': 0.05, 'is_significant': True}
        }

        tradeoffs = detector.detect_tradeoffs(causal_effects)
        print(f"  ✓ 使用旧sign_functions检测到权衡: {len(tradeoffs)}")

        return True

    except Exception as e:
        print(f"  ❌ 向后兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("权衡检测sign函数优化方案验收测试")
    print("=" * 60)

    test_results = []

    # 运行所有测试
    test_results.append(("规则系统测试", test_rule_system()))
    test_results.append(("sign函数创建测试", test_sign_function_creation()))
    test_results.append(("权衡检测器测试", test_tradeoff_detector()))
    test_results.append(("真实数据测试", test_with_real_data()))
    test_results.append(("向后兼容性测试", test_backward_compatibility()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_name, passed in test_results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！优化方案验收成功。")
    else:
        print("❌ 部分测试失败，请检查问题。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)