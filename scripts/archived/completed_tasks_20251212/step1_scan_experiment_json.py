#!/usr/bin/env python3
"""
步骤1: 扫描所有新实验的experiment.json，收集所有可能的字段

遍历results/run_*/目录下的所有实验，分析experiment.json的结构，
收集所有可能出现的字段，为构建新CSV表头做准备。
"""

import json
import os
import glob
from collections import defaultdict
from datetime import datetime

def scan_experiment_jsons():
    """扫描所有experiment.json，收集字段信息"""

    all_fields = set()
    field_examples = defaultdict(set)  # 记录每个字段的示例值
    experiments_count = 0

    print("扫描所有run目录下的experiment.json...")
    print("=" * 70)

    # 遍历所有run目录
    run_dirs = sorted(glob.glob("results/run_*"))
    print(f"找到 {len(run_dirs)} 个run目录")
    print()

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        # 查找该run下的所有experiment.json
        exp_json_files = glob.glob(f"{run_dir}/*/experiment.json")

        if not exp_json_files:
            continue

        print(f"处理 {run_name}: {len(exp_json_files)} 个实验")

        for exp_json_path in exp_json_files:
            experiments_count += 1

            try:
                with open(exp_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 递归收集所有字段
                def collect_fields(obj, prefix=''):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            full_key = f"{prefix}_{key}" if prefix else key
                            all_fields.add(full_key)

                            # 记录示例值（限制长度）
                            if isinstance(value, (str, int, float, bool)):
                                example = str(value)[:50]
                                field_examples[full_key].add(example)

                            # 递归处理嵌套结构
                            if isinstance(value, dict):
                                collect_fields(value, full_key)
                            elif isinstance(value, list) and value and isinstance(value[0], dict):
                                # 处理列表中的字典（取第一个元素的结构）
                                collect_fields(value[0], full_key)

                collect_fields(data)

            except Exception as e:
                print(f"  ⚠️  读取失败: {exp_json_path}")
                print(f"      错误: {e}")

    print()
    print("=" * 70)
    print(f"扫描完成！")
    print(f"  总实验数: {experiments_count}")
    print(f"  总字段数: {len(all_fields)}")
    print()

    # 按字段名排序并分类显示
    sorted_fields = sorted(all_fields)

    # 分类字段
    categories = {
        '基本信息': [],
        'hyperparameters': [],
        'energy_metrics': [],
        'performance_metrics': [],
        '其他': []
    }

    for field in sorted_fields:
        if field.startswith('hyperparameters_'):
            categories['hyperparameters'].append(field)
        elif field.startswith('energy_metrics_'):
            categories['energy_metrics'].append(field)
        elif field.startswith('performance_metrics_'):
            categories['performance_metrics'].append(field)
        elif field in ['experiment_id', 'timestamp', 'repository', 'model',
                       'training_success', 'duration_seconds', 'retries', 'error_message']:
            categories['基本信息'].append(field)
        else:
            categories['其他'].append(field)

    # 打印分类结果
    print("字段分类：")
    print("=" * 70)

    for category, fields in categories.items():
        if not fields:
            continue

        print(f"\n【{category}】 ({len(fields)} 个字段)")
        print("-" * 70)

        for field in fields:
            # 获取示例值（取前3个）
            examples = list(field_examples[field])[:3]
            examples_str = ", ".join(examples) if examples else "N/A"

            # 截断过长的示例
            if len(examples_str) > 60:
                examples_str = examples_str[:57] + "..."

            print(f"  {field:40s} 示例: {examples_str}")

    print()
    print("=" * 70)

    # 保存字段列表到文件
    output_file = "results/experiment_json_fields.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"扫描时间: {datetime.now().isoformat()}\n")
        f.write(f"总实验数: {experiments_count}\n")
        f.write(f"总字段数: {len(all_fields)}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        for category, fields in categories.items():
            if not fields:
                continue
            f.write(f"【{category}】 ({len(fields)} 个字段)\n")
            f.write("-" * 70 + "\n")
            for field in fields:
                examples = list(field_examples[field])[:3]
                examples_str = ", ".join(examples)
                f.write(f"  {field}\n")
                f.write(f"    示例: {examples_str}\n")
            f.write("\n")

    print(f"✓ 字段列表已保存到: {output_file}")

    return sorted_fields, field_examples, experiments_count

if __name__ == '__main__':
    fields, examples, count = scan_experiment_jsons()

    print()
    print("下一步: 设计新的CSV表头结构")
