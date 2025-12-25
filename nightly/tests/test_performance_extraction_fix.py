#!/usr/bin/env python3
"""
Unit tests for Phase 3: Performance Extraction Fix

Tests the updated log_patterns in models_config.json to ensure
they correctly extract performance metrics from actual model outputs.

Phase 3 修复了4个问题模型的正则表达式:
1. examples/mnist_ff - 添加可选[SUCCESS]标签
2. VulBERTa/mlp - 添加字典格式('eval_loss':)提取
3. bug-localization - 支持Top-k格式（带空格）
4. MRT-OAST - 增强中英文混合支持

执行方式:
    python3 tests/test_performance_extraction_fix.py
"""

import re
import sys
import json
from pathlib import Path


def test_mnist_ff_extraction():
    """测试 examples/mnist_ff 的提取模式

    修复前: "Final Test Accuracy[:\\s]+([0-9.]+)%"
    修复后: "(?:\\[SUCCESS\\]\\s+)?Test Accuracy[:\\s]+([0-9.]+)%?"

    新模式特点:
    - 可选匹配 [SUCCESS] 标签
    - 移除 "Final" 要求
    - % 符号可选
    """
    # 加载配置
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path) as f:
        config = json.load(f)

    patterns = config["models"]["examples"]["performance_metrics"]["log_patterns"]

    # 测试用例1: 带[SUCCESS]标签的输出（实际输出格式）
    log_content_1 = "[SUCCESS] Test Accuracy: 9.5599994063377400%"
    match = re.search(patterns["test_accuracy"], log_content_1)
    assert match is not None, "Failed to match [SUCCESS] format"
    assert float(match.group(1)) == 9.5599994063377400
    print("✓ mnist_ff test 1: [SUCCESS] format matched")

    # 测试用例2: 不带百分号
    log_content_2 = "Test Accuracy: 0.9560"
    match = re.search(patterns["test_accuracy"], log_content_2)
    assert match is not None, "Failed to match without % format"
    assert float(match.group(1)) == 0.9560
    print("✓ mnist_ff test 2: Without % format matched")

    # 测试用例3: Test Error（新增指标）
    log_content_3 = "Test Error: 0.9044000059366226"
    match = re.search(patterns["test_error"], log_content_3)
    assert match is not None, "Failed to match test_error"
    assert float(match.group(1)) == 0.9044000059366226
    print("✓ mnist_ff test 3: test_error matched")

    print("✅ All mnist_ff tests passed\n")


def test_vulberta_mlp_extraction():
    """测试 VulBERTa/mlp 的提取模式

    修复前: "Accuracy[:\\s]+([0-9.]+)"
    修复后: "(?:'eval_loss':|eval_loss:)\\s*([0-9.]+)"

    新模式特点:
    - 支持字典格式 'eval_loss':
    - 支持键值格式 eval_loss:
    """
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path) as f:
        config = json.load(f)

    patterns = config["models"]["VulBERTa"]["performance_metrics"]["log_patterns"]

    # 测试用例1: 字典格式（实际输出）
    log_content_1 = "{'eval_loss': 5.012244701385498, 'epoch': 18.0}"
    match = re.search(patterns["eval_loss"], log_content_1)
    assert match is not None, "Failed to match dict format"
    assert float(match.group(1)) == 5.012244701385498
    print("✓ VulBERTa/mlp test 1: Dict format matched")

    # 测试用例2: 键值格式
    log_content_2 = "  eval_loss: 0.776414692401886"
    match = re.search(patterns["eval_loss"], log_content_2)
    assert match is not None, "Failed to match key-value format"
    assert float(match.group(1)) == 0.776414692401886
    print("✓ VulBERTa/mlp test 2: Key-value format matched")

    # 测试用例3: Final training loss（新增指标）
    log_content_3 = "Final training loss: 0.4189"
    match = re.search(patterns["final_training_loss"], log_content_3)
    assert match is not None, "Failed to match final_training_loss"
    assert float(match.group(1)) == 0.4189
    print("✓ VulBERTa/mlp test 3: final_training_loss matched")

    print("✅ All VulBERTa/mlp tests passed\n")


def test_bug_localization_extraction():
    """测试 bug-localization 的提取模式

    修复前: "Top-1[:\\s@]+([0-9.]+)"
    修复后: "Top-\\s*1\\s+(?:Accuracy:)?\\s*([0-9.]+)"

    新模式特点:
    - 支持 Top-1 和 Top- 1 格式（空格可选）
    - "Accuracy:" 可选
    """
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path) as f:
        config = json.load(f)

    patterns = config["models"]["bug-localization-by-dnn-and-rvsm"]["performance_metrics"]["log_patterns"]

    # 测试用例1: 带空格格式（实际输出）
    log_content_1 = "  Top- 1 Accuracy: 0.380 (38.0%)"
    match = re.search(patterns["top1_accuracy"], log_content_1)
    assert match is not None, "Failed to match with space format"
    assert float(match.group(1)) == 0.380
    print("✓ bug-localization test 1: With space format matched")

    # 测试用例2: 不带空格格式
    log_content_2 = "  Top-1: 0.380"
    match = re.search(patterns["top1_accuracy"], log_content_2)
    assert match is not None, "Failed to match without space format"
    assert float(match.group(1)) == 0.380
    print("✓ bug-localization test 2: Without space format matched")

    # 测试用例3: Top-5
    log_content_3 = "  Top- 5 Accuracy: 0.628 (62.8%)"
    match = re.search(patterns["top5_accuracy"], log_content_3)
    assert match is not None, "Failed to match Top-5"
    assert float(match.group(1)) == 0.628
    print("✓ bug-localization test 3: top5_accuracy matched")

    # 测试用例4: Top-10（新增指标）
    log_content_4 = "  Top-10 Accuracy: 0.740"
    match = re.search(patterns["top10_accuracy"], log_content_4)
    assert match is not None, "Failed to match Top-10"
    assert float(match.group(1)) == 0.740
    print("✓ bug-localization test 4: top10_accuracy matched")

    print("✅ All bug-localization tests passed\n")


def test_mrt_oast_extraction():
    """测试 MRT-OAST 的提取模式

    修复前: "Accuracy[:\\s]+([0-9.]+)"
    修复后: "(?:Accuracy|准确率)[:\\s()]+([0-9.]+)"

    新模式特点:
    - 支持英文 "Accuracy" 和中文 "准确率"
    - 支持括号
    """
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path) as f:
        config = json.load(f)

    patterns = config["models"]["MRT-OAST"]["performance_metrics"]["log_patterns"]

    # 测试用例1: 英文格式
    log_content_1 = "    Precision: 0.979006"
    match = re.search(patterns["precision"], log_content_1)
    assert match is not None, "Failed to match English format"
    assert float(match.group(1)) == 0.979006
    print("✓ MRT-OAST test 1: English format matched")

    # 测试用例2: 中文格式（实际输出）
    log_content_2 = "  准确率 (Accuracy): 0.8632"
    match = re.search(patterns["accuracy"], log_content_2)
    assert match is not None, "Failed to match Chinese format"
    assert float(match.group(1)) == 0.8632
    print("✓ MRT-OAST test 2: Chinese format matched")

    # 测试用例3: Recall with spaces
    log_content_3 = "    Recall   : 0.733140"
    match = re.search(patterns["recall"], log_content_3)
    assert match is not None, "Failed to match Recall with spaces"
    assert float(match.group(1)) == 0.733140
    print("✓ MRT-OAST test 3: recall with spaces matched")

    # 测试用例4: F1 score（支持 F1 score 和 F1-score）
    log_content_4 = "F1 score: 0.9071"
    match = re.search(patterns["f1"], log_content_4)
    assert match is not None, "Failed to match F1 score"
    assert float(match.group(1)) == 0.9071
    print("✓ MRT-OAST test 4: F1 score matched")

    print("✅ All MRT-OAST tests passed\n")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("Phase 3: Performance Extraction Fix - Unit Tests")
    print("=" * 80)
    print()

    total_tests = 0
    passed_tests = 0

    tests = [
        ("examples/mnist_ff", test_mnist_ff_extraction),
        ("VulBERTa/mlp", test_vulberta_mlp_extraction),
        ("bug-localization", test_bug_localization_extraction),
        ("MRT-OAST", test_mrt_oast_extraction)
    ]

    for test_name, test_func in tests:
        try:
            print(f"Testing {test_name}...")
            print("-" * 80)
            test_func()
            passed_tests += 1
            total_tests += 1
        except AssertionError as e:
            print(f"❌ Test failed for {test_name}: {e}")
            total_tests += 1
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            total_tests += 1

    print("=" * 80)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Performance extraction fix verified!")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
