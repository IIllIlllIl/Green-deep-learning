#!/usr/bin/env python3
"""
性能度量可视化脚本

生成项目中所有模型性能度量的可视化图表
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_metrics():
    """分析配置文件中的性能度量"""

    config_path = Path(__file__).parent.parent / "config" / "models_config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    # 统计数据
    metrics_count = defaultdict(int)
    task_types = {"classification": [], "retrieval": []}
    model_metrics = []

    for repo_name, repo_config in config["models"].items():
        models = repo_config.get("models", [])
        log_patterns = repo_config.get("performance_metrics", {}).get("log_patterns", {})

        for model_name in models:
            model_info = {
                "repo": repo_name,
                "model": model_name,
                "metrics": list(log_patterns.keys())
            }
            model_metrics.append(model_info)

            # 统计度量出现次数
            for metric in log_patterns.keys():
                metrics_count[metric] += 1

            # 判断任务类型
            if any(m in log_patterns for m in ["accuracy", "test_accuracy"]):
                task_types["classification"].append(f"{repo_name}/{model_name}")
            elif any(m in log_patterns for m in ["map", "rank1", "rank5"]):
                task_types["retrieval"].append(f"{repo_name}/{model_name}")

    return metrics_count, task_types, model_metrics

def print_metrics_report():
    """打印性能度量分析报告"""

    metrics_count, task_types, model_metrics = analyze_metrics()

    total_models = len(model_metrics)

    print("=" * 80)
    print("性能度量分析报告")
    print("=" * 80)
    print()

    # 1. 总体统计
    print("📊 总体统计")
    print("-" * 80)
    print(f"总模型数: {total_models}")
    print(f"分类任务模型: {len(task_types['classification'])} ({len(task_types['classification'])/total_models*100:.1f}%)")
    print(f"检索任务模型: {len(task_types['retrieval'])} ({len(task_types['retrieval'])/total_models*100:.1f}%)")
    print()

    # 2. 度量统计
    print("📈 性能度量统计 (按出现次数排序)")
    print("-" * 80)
    sorted_metrics = sorted(metrics_count.items(), key=lambda x: x[1], reverse=True)

    for metric, count in sorted_metrics:
        percentage = count / total_models * 100
        bar_length = int(percentage / 2)  # 每2%一个字符
        bar = "█" * bar_length
        print(f"{metric:20s} {bar:50s} {count:2d} 个模型 ({percentage:5.1f}%)")
    print()

    # 3. 任务类型详情
    print("🎯 分类任务模型 (11个)")
    print("-" * 80)
    for i, model in enumerate(task_types['classification'], 1):
        print(f"{i:2d}. {model}")
    print()

    print("🔍 检索任务模型 (4个)")
    print("-" * 80)
    for i, model in enumerate(task_types['retrieval'], 1):
        print(f"{i:2d}. {model}")
    print()

    # 4. 公共度量分析
    print("💡 公共度量分析")
    print("-" * 80)

    # 分类任务的公共度量
    classification_metrics = defaultdict(int)
    for model_full in task_types['classification']:
        for model_info in model_metrics:
            if f"{model_info['repo']}/{model_info['model']}" == model_full:
                for metric in model_info['metrics']:
                    if 'accuracy' in metric.lower():
                        classification_metrics['accuracy'] += 1

    # 检索任务的公共度量
    retrieval_metrics = defaultdict(int)
    for model_full in task_types['retrieval']:
        for model_info in model_metrics:
            if f"{model_info['repo']}/{model_info['model']}" == model_full:
                for metric in model_info['metrics']:
                    if 'map' in metric.lower():
                        retrieval_metrics['map'] += 1

    print(f"✅ 分类任务公共度量: Accuracy")
    print(f"   覆盖率: {classification_metrics['accuracy']}/{len(task_types['classification'])} = {classification_metrics['accuracy']/len(task_types['classification'])*100:.1f}%")
    print()
    print(f"✅ 检索任务公共度量: mAP")
    print(f"   覆盖率: {retrieval_metrics['map']}/{len(task_types['retrieval'])} = {retrieval_metrics['map']/len(task_types['retrieval'])*100:.1f}%")
    print()

    # 5. 详细模型列表
    print("📋 详细模型度量列表")
    print("-" * 80)
    print(f"{'#':<3} {'仓库':<30} {'模型':<20} {'度量指标':<30}")
    print("-" * 80)

    for i, model_info in enumerate(model_metrics, 1):
        metrics_str = ", ".join(model_info['metrics'])
        if len(metrics_str) > 28:
            metrics_str = metrics_str[:25] + "..."
        print(f"{i:<3} {model_info['repo']:<30} {model_info['model']:<20} {metrics_str:<30}")

    print()
    print("=" * 80)
    print("结论: 不存在全局公共度量，建议采用分层度量策略")
    print("  - 分类任务: 使用 Accuracy")
    print("  - 检索任务: 使用 mAP")
    print("=" * 80)

if __name__ == "__main__":
    print_metrics_report()
