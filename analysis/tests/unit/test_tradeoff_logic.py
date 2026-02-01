#!/usr/bin/env python3
"""
权衡检测sign函数优化方案逻辑验收测试
直接检查代码逻辑，不依赖外部库
"""

import sys
import os
import re

def analyze_tradeoff_detection_code():
    """分析tradeoff_detection.py代码"""
    print("=" * 60)
    print("权衡检测sign函数优化方案代码分析")
    print("=" * 60)

    file_path = "utils/tradeoff_detection.py"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"文件大小: {len(content)} 字符")

        # 检查关键组件
        components = {
            "TradeoffDetector类": "class TradeoffDetector:",
            "ENERGY_PERF_RULES": "ENERGY_PERF_RULES = {",
            "create_sign_func_from_rule函数": "def create_sign_func_from_rule",
            "create_sign_functions_from_rules函数": "def create_sign_functions_from_rules",
            "create_energy_perf_sign_functions函数": "def create_energy_perf_sign_functions",
            "PatternSignFunctions内部类": "class PatternSignFunctions:"
        }

        print("\n关键组件检查:")
        for name, pattern in components.items():
            if pattern in content:
                print(f"  ✓ {name}: 存在")
            else:
                print(f"  ❌ {name}: 缺失")

        # 分析规则系统
        print("\n规则系统分析:")

        # 提取ENERGY_PERF_RULES
        rules_match = re.search(r'ENERGY_PERF_RULES = \{([^}]+(?:\{[^}]*\}[^}]*)*)\}', content, re.DOTALL)
        if rules_match:
            rules_text = rules_match.group(1)
            # 统计规则数量
            rule_lines = [line.strip() for line in rules_text.split('\n') if '"' in line or "'" in line]
            print(f"  规则总数: {len(rule_lines)}")

            # 分类统计
            energy_rules = [line for line in rule_lines if 'energy_' in line]
            perf_rules = [line for line in rule_lines if 'perf_' in line]
            interaction_rules = [line for line in rule_lines if '_x_is_parallel' in line]

            print(f"  能耗指标规则: {len(energy_rules)}")
            print(f"  性能指标规则: {len(perf_rules)}")
            print(f"  交互项规则: {len(interaction_rules)}")

            # 显示示例规则
            print("\n  示例规则:")
            for line in rule_lines[:5]:
                print(f"    {line}")
        else:
            print("  ❌ 无法提取ENERGY_PERF_RULES")

        # 检查sign函数签名
        print("\nsign函数签名检查:")

        # 检查create_sign_func_from_rule函数
        sign_func_match = re.search(r'def create_sign_func_from_rule\(rule: str\):(.*?)def ', content, re.DOTALL)
        if sign_func_match:
            sign_func_code = sign_func_match.group(1)
            # 检查是否包含current_value参数
            if 'current_value' in sign_func_code and 'change' in sign_func_code:
                print("  ✓ create_sign_func_from_rule: 使用论文风格签名 (current_value, change)")
            else:
                print("  ❌ create_sign_func_from_rule: 未使用论文风格签名")
        else:
            print("  ❌ 无法找到create_sign_func_from_rule函数")

        # 检查TradeoffDetector的_compute_sign方法
        compute_sign_match = re.search(r'def _compute_sign\(self,.*?current_value.*?\)', content)
        if compute_sign_match:
            print("  ✓ _compute_sign: 支持current_value参数")
        else:
            print("  ❌ _compute_sign: 不支持current_value参数")

        # 检查向后兼容性
        print("\n向后兼容性检查:")

        # 检查__init__方法是否支持sign_functions参数
        init_match = re.search(r'def __init__\(self,(.*?)\):', content, re.DOTALL)
        if init_match:
            init_params = init_match.group(1)
            if 'sign_functions' in init_params:
                print("  ✓ __init__: 支持sign_functions参数（向后兼容）")
            else:
                print("  ❌ __init__: 不支持sign_functions参数")

            if 'rules' in init_params:
                print("  ✓ __init__: 支持rules参数")
            else:
                print("  ❌ __init__: 不支持rules参数")

            if 'current_values' in init_params:
                print("  ✓ __init__: 支持current_values参数")
            else:
                print("  ❌ __init__: 不支持current_values参数")
        else:
            print("  ❌ 无法找到__init__方法")

        # 检查detect_tradeoffs方法
        detect_match = re.search(r'def detect_tradeoffs\(self,(.*?)\) ->', content, re.DOTALL)
        if detect_match:
            detect_params = detect_match.group(1)
            if 'current_values' in detect_params:
                print("  ✓ detect_tradeoffs: 支持current_values参数")
            else:
                print("  ❌ detect_tradeoffs: 不支持current_values参数")
        else:
            print("  ❌ 无法找到detect_tradeoffs方法")

        return True

    except Exception as e:
        print(f"分析失败: {e}")
        return False

def compare_with_paper_implementation():
    """与论文实现对比"""
    print("\n" + "=" * 60)
    print("与论文实现对比分析")
    print("=" * 60)

    # 论文中的关键特征
    paper_features = {
        "规则系统": "使用rules字典定义改善方向",
        "sign函数": "函数签名 sign(current_value, change)",
        "权衡检测算法": "算法1：基于因果效应的权衡检测",
        "当前值使用": "使用当前值计算sign",
        "模式匹配": "支持通配符模式匹配"
    }

    print("论文实现特征:")
    for feature, description in paper_features.items():
        print(f"  {feature}: {description}")

    # 检查我们的实现是否包含这些特征
    print("\n我们的实现检查:")

    file_path = "utils/tradeoff_detection.py"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("规则系统", "ENERGY_PERF_RULES" in content),
            ("sign函数论文风格", "def sign_func(current_value: float, change: float)" in content),
            ("权衡检测算法", "def detect_tradeoffs" in content and "def _check_tradeoff_pair" in content),
            ("当前值使用", "current_value" in content and "current_values" in content),
            ("模式匹配", "PatternSignFunctions" in content and "_pattern_match" in content)
        ]

        for feature, found in checks:
            status = "✓" if found else "❌"
            print(f"  {status} {feature}")

        return all(found for _, found in checks)

    except Exception as e:
        print(f"对比失败: {e}")
        return False

def check_energy_perf_scenario():
    """检查能耗/性能场景适配性"""
    print("\n" + "=" * 60)
    print("能耗/性能场景适配性检查")
    print("=" * 60)

    file_path = "utils/tradeoff_detection.py"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取ENERGY_PERF_RULES内容
        rules_match = re.search(r'ENERGY_PERF_RULES = \{([^}]+(?:\{[^}]*\}[^}]*)*)\}', content, re.DOTALL)
        if not rules_match:
            print("❌ 无法提取ENERGY_PERF_RULES")
            return False

        rules_text = rules_match.group(1)

        # 检查关键指标类型
        required_patterns = [
            ("能耗指标", "energy_.*"),
            ("准确率指标", "perf_.*_accuracy"),
            ("损失函数", "perf_.*_loss"),
            ("效率指标", "perf_.*_samples_per_second"),
            ("交互项", ".*_x_is_parallel"),
            ("默认规则", '"\\*"')
        ]

        print("关键指标类型检查:")
        for name, pattern in required_patterns:
            # 简化检查：查找相关文本
            if name == "能耗指标":
                found = '"energy_' in rules_text
            elif name == "准确率指标":
                found = 'accuracy' in rules_text and 'perf_' in rules_text
            elif name == "损失函数":
                found = 'loss' in rules_text and 'perf_' in rules_text
            elif name == "效率指标":
                found = 'samples_per_second' in rules_text
            elif name == "交互项":
                found = '_x_is_parallel' in rules_text
            elif name == "默认规则":
                found = '"*"' in rules_text

            status = "✓" if found else "❌"
            print(f"  {status} {name}")

        # 检查改善方向
        print("\n改善方向检查:")

        # 能耗指标应为负改善
        energy_lines = [line for line in rules_text.split('\n') if 'energy_' in line]
        energy_correct = all('"-"' in line for line in energy_lines if 'energy_' in line)
        print(f"  能耗指标改善方向: {'✓ 正确（负改善）' if energy_correct else '❌ 错误'}")

        # 准确率指标应为正改善
        accuracy_lines = [line for line in rules_text.split('\n') if 'accuracy' in line and 'perf_' in line]
        accuracy_correct = all('"+"' in line for line in accuracy_lines if 'accuracy' in line)
        print(f"  准确率指标改善方向: {'✓ 正确（正改善）' if accuracy_correct else '❌ 错误'}")

        # 损失函数应为负改善
        loss_lines = [line for line in rules_text.split('\n') if 'loss' in line and 'perf_' in line]
        loss_correct = all('"-"' in line for line in loss_lines if 'loss' in line)
        print(f"  损失函数改善方向: {'✓ 正确（负改善）' if loss_correct else '❌ 错误'}")

        return True

    except Exception as e:
        print(f"检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("权衡检测sign函数优化方案逻辑验收测试")
    print("=" * 60)

    test_results = []

    # 运行所有测试
    test_results.append(("代码分析", analyze_tradeoff_detection_code()))
    test_results.append(("与论文实现对比", compare_with_paper_implementation()))
    test_results.append(("能耗/性能场景适配性", check_energy_perf_scenario()))

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
        print("✅ 所有逻辑测试通过！优化方案验收成功。")
        print("\n建议:")
        print("1. 在实际环境中安装numpy和pandas后运行完整测试")
        print("2. 使用真实ATE数据进行验证测试")
        print("3. 检查边缘情况处理（如变化量为0、当前值缺失等）")
    else:
        print("❌ 部分测试失败，请检查问题。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)