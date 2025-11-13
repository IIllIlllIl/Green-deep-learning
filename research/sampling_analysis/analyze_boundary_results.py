#!/usr/bin/env python3
"""
Analyze boundary test results and validate mutation ranges
"""
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Load the test configuration to get baselines
with open('settings/boundary_test_elite_4models.json', 'r') as f:
    test_configs = json.load(f)['experiments']

# Group configs by model to identify baselines
model_configs = defaultdict(list)
for idx, config in enumerate(test_configs):
    model_key = f"{config['repo']}_{config['model']}"
    model_configs[model_key].append((idx, config))

# Identify baseline configs (should be first for each model)
baselines = {}
for model_key, configs in model_configs.items():
    # First config for each model should be baseline
    baselines[model_key] = configs[0]

print("="*80)
print("边界测试结果分析 - Boundary Test Results Analysis")
print("="*80)
print(f"\n分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load all result files from Nov 10 20:00 onwards (boundary test period)
results_dir = Path('results')
result_files = sorted(results_dir.glob('*.json'))

# Filter to boundary test timeframe (after Nov 10 20:00)
boundary_results = []
for rf in result_files:
    if rf.stem >= '20251110_200000':  # After Nov 10 20:00
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                boundary_results.append(data)
        except Exception as e:
            print(f"⚠️  Error reading {rf}: {e}")

print(f"📊 总共找到 {len(boundary_results)} 个训练结果\n")

# Group results by model
model_results = defaultdict(list)
for result in boundary_results:
    model_key = f"{result['repository']}_{result['model']}"
    model_results[model_key].append(result)

# Analyze each model
all_models_summary = []

for model_key in sorted(model_results.keys()):
    results = model_results[model_key]
    print("="*80)
    print(f"📌 模型: {model_key}")
    print("="*80)

    # Find baseline result
    baseline_idx, baseline_config = baselines.get(model_key, (None, None))
    if baseline_config is None:
        print("⚠️  未找到baseline配置\n")
        continue

    baseline_hp = baseline_config['hyperparameters']

    # Find baseline result
    baseline_result = None
    for result in results:
        result_hp = result['hyperparameters']
        # Check if all hyperparameters match baseline
        if all(result_hp.get(k) == v for k, v in baseline_hp.items()):
            baseline_result = result
            break

    if baseline_result is None:
        print(f"⚠️  未找到baseline结果")
        print(f"   期望超参数: {baseline_hp}\n")
        continue

    baseline_success = baseline_result['training_success']
    baseline_perf = baseline_result['performance_metrics']

    print(f"\n🎯 Baseline结果:")
    print(f"   超参数: {baseline_hp}")
    print(f"   训练状态: {'✅ 成功' if baseline_success else '❌ 失败'}")
    if baseline_success:
        print(f"   性能指标: {baseline_perf}")
    else:
        print(f"   错误信息: {baseline_result.get('error_message', 'Unknown')}")

    if not baseline_success:
        print(f"\n⚠️  Baseline训练失败，无法进行性能对比分析\n")
        all_models_summary.append({
            'model': model_key,
            'baseline_success': False,
            'tests_completed': len(results),
            'tests_failed': sum(1 for r in results if not r['training_success']),
            'performance_degradations': 'N/A'
        })
        continue

    # Extract baseline metric
    baseline_metric = None
    metric_name = None
    if 'test_accuracy' in baseline_perf:
        baseline_metric = baseline_perf['test_accuracy']
        metric_name = 'test_accuracy'
    elif 'map' in baseline_perf:
        baseline_metric = baseline_perf['map']
        metric_name = 'map'
    elif 'accuracy' in baseline_perf:
        baseline_metric = baseline_perf['accuracy']
        metric_name = 'accuracy'

    if baseline_metric is None:
        print(f"\n⚠️  未找到性能指标\n")
        continue

    print(f"\n   主要性能指标: {metric_name} = {baseline_metric}")

    # Analyze boundary results
    print(f"\n📈 边界测试结果:")
    print(f"   {'配置描述':<60} {'状态':<8} {metric_name:<12} {'性能下降':<10} {'判定'}")
    print("   " + "-"*105)

    failed_tests = []
    degradation_tests = []

    for result in sorted(results, key=lambda r: (r['hyperparameters'].get('epochs', 0),
                                                   r['hyperparameters'].get('learning_rate', 0))):
        result_hp = result['hyperparameters']

        # Skip baseline
        if all(result_hp.get(k) == baseline_hp.get(k) for k in result_hp.keys()):
            continue

        # Find description from config
        desc = "Unknown"
        for idx, cfg in model_configs[model_key]:
            if all(result_hp.get(k) == cfg['hyperparameters'].get(k) for k in result_hp.keys()):
                desc = cfg.get('description', 'Unknown')
                break

        success = result['training_success']
        perf = result['performance_metrics']

        if not success:
            status_icon = "❌ 失败"
            metric_val = "N/A"
            degradation = "N/A"
            verdict = "⛔ 训练失败"
            failed_tests.append((desc, result))
        else:
            status_icon = "✅ 成功"
            result_metric = perf.get(metric_name, None)

            if result_metric is None:
                metric_val = "N/A"
                degradation = "N/A"
                verdict = "⚠️  无指标"
            else:
                metric_val = f"{result_metric:.4f}" if isinstance(result_metric, float) else str(result_metric)

                # Calculate degradation
                if baseline_metric > 0:
                    deg_pct = (baseline_metric - result_metric) / baseline_metric * 100
                else:
                    deg_pct = 0

                degradation = f"{deg_pct:+.2f}%"

                # Verdict
                if deg_pct < 0:  # Performance improved
                    verdict = "✅ 提升"
                elif deg_pct < 5:
                    verdict = "✅ 优秀"
                elif deg_pct < 10:
                    verdict = "⚠️  警告"
                else:
                    verdict = "❌ 不可接受"
                    degradation_tests.append((desc, result, deg_pct))

        # Truncate description for display
        short_desc = desc[:58] + ".." if len(desc) > 60 else desc
        print(f"   {short_desc:<60} {status_icon:<8} {metric_val:<12} {degradation:<10} {verdict}")

    # Summary for this model
    print(f"\n📋 模型总结:")
    print(f"   - 总测试数: {len(results)}")
    print(f"   - 训练失败: {len(failed_tests)}")
    print(f"   - 性能下降>10%: {len(degradation_tests)}")

    if failed_tests:
        print(f"\n   ❌ 失败的测试:")
        for desc, result in failed_tests:
            print(f"      - {desc}")
            print(f"        超参数: {result['hyperparameters']}")
            print(f"        错误: {result.get('error_message', 'Unknown')}")

    if degradation_tests:
        print(f"\n   ⚠️  性能下降>10%的测试:")
        for desc, result, deg_pct in degradation_tests:
            print(f"      - {desc}")
            print(f"        超参数: {result['hyperparameters']}")
            print(f"        性能下降: {deg_pct:.2f}%")

    all_models_summary.append({
        'model': model_key,
        'baseline_success': True,
        'baseline_metric': baseline_metric,
        'metric_name': metric_name,
        'tests_completed': len(results),
        'tests_failed': len(failed_tests),
        'tests_degraded': len(degradation_tests)
    })

    print()

# Overall summary
print("="*80)
print("🎯 总体分析结论")
print("="*80)

for summary in all_models_summary:
    print(f"\n📌 {summary['model']}:")
    if not summary['baseline_success']:
        print(f"   ⚠️  Baseline训练失败，需要检查模型配置")
    else:
        print(f"   ✅ Baseline {summary['metric_name']}: {summary['baseline_metric']}")
        print(f"   📊 测试完成: {summary['tests_completed']}")
        print(f"   ❌ 训练失败: {summary['tests_failed']}")
        print(f"   ⚠️  性能下降>10%: {summary['tests_degraded']}")

        if summary['tests_failed'] == 0 and summary['tests_degraded'] == 0:
            print(f"   ✅ 结论: 当前变异范围合理")
        elif summary['tests_failed'] > 0:
            print(f"   ⚠️  结论: 存在训练失败，需要检查失败原因")
        else:
            print(f"   ⚠️  结论: 存在显著性能下降，建议缩小变异范围")

print("\n" + "="*80)
print("📝 标准化变异范围:")
print("="*80)
print("""
| 超参数 | 变异范围表达式 | 分布类型 | 说明 |
|--------|---------------|----------|------|
| Epochs | [default×0.5, default×2.0] | 对数均匀 | 半倍到两倍 |
| Learning Rate | [default×0.1, default×10.0] | 对数均匀 | 十倍范围 |
| Weight Decay | [0.0, default×100] | 30%零值+70%对数 | 允许无正则化 |
| Dropout | [0.0, 0.7] | 均匀分布 | 绝对值范围 |
| Seed | [0, 9999] | 均匀整数 | 评估稳定性 |
""")

print("\n✅ 分析完成！")
