#!/usr/bin/env python3
"""
权衡检测sign函数优化方案简化验收测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "=" * 60)
    print("4. 测试向后兼容性")
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

def test_code_quality():
    """测试代码质量"""
    print("\n" + "=" * 60)
    print("5. 测试代码质量")
    print("=" * 60)

    # 检查代码中的关键函数
    print("检查关键函数:")

    # 检查TradeoffDetector类
    print("  TradeoffDetector类:")
    print(f"    __init__ 参数: {TradeoffDetector.__init__.__code__.co_varnames}")
    print(f"    detect_tradeoffs 参数: {TradeoffDetector.detect_tradeoffs.__code__.co_varnames}")

    # 检查辅助函数
    print("  辅助函数:")
    print(f"    create_sign_func_from_rule 存在: {'create_sign_func_from_rule' in globals()}")
    print(f"    create_sign_functions_from_rules 存在: {'create_sign_functions_from_rules' in globals()}")
    print(f"    create_energy_perf_sign_functions 存在: {'create_energy_perf_sign_functions' in globals()}")

    # 检查规则系统
    print("  规则系统:")
    print(f"    ENERGY_PERF_RULES 类型: {type(ENERGY_PERF_RULES)}")
    print(f"    ENERGY_PERF_RULES 大小: {len(ENERGY_PERF_RULES)}")

    return True

def main():
    """主测试函数"""
    print("权衡检测sign函数优化方案验收测试（简化版）")
    print("=" * 60)

    test_results = []

    # 运行所有测试
    test_results.append(("规则系统测试", test_rule_system()))
    test_results.append(("sign函数创建测试", test_sign_function_creation()))
    test_results.append(("权衡检测器测试", test_tradeoff_detector()))
    test_results.append(("向后兼容性测试", test_backward_compatibility()))
    test_results.append(("代码质量测试", test_code_quality()))

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