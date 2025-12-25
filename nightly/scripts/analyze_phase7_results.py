#!/usr/bin/env python3
"""分析 Phase 7 执行结果"""

import csv
from datetime import datetime
from collections import defaultdict

def analyze_phase7():
    csv_path = "results/run_20251217_211341/summary.csv"

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    success = sum(1 for r in rows if r['training_success'] == 'True')
    failed = total - success

    # 按模型分组
    model_counts = defaultdict(int)
    mode_counts = defaultdict(int)
    for row in rows:
        model_counts[row['model']] += 1
        # 判断模式
        if '_parallel' in row['experiment_id']:
            mode_counts['parallel'] += 1
        else:
            mode_counts['nonparallel'] += 1

    # 时间统计
    start_time = datetime.fromisoformat(rows[0]['timestamp'])
    end_time = datetime.fromisoformat(rows[-1]['timestamp'])
    duration_hours = (end_time - start_time).total_seconds() / 3600

    # 能耗数据完整性
    energy_complete = sum(1 for r in rows if r['energy_cpu_total_joules'] and r['energy_gpu_total_joules'])

    # 性能数据完整性
    perf_complete = sum(1 for r in rows if any([
        r.get('perf_top1_accuracy'),
        r.get('perf_eval_loss'),
        r.get('perf_final_training_loss')
    ]))

    print("=" * 70)
    print("Phase 7 执行结果分析")
    print("=" * 70)
    print(f"\n【基本统计】")
    print(f"  总实验数: {total}")
    print(f"  训练成功: {success} ({success/total*100:.1f}%)")
    print(f"  训练失败: {failed} ({failed/total*100:.1f}%)")

    print(f"\n【模型分布】")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}个实验")

    print(f"\n【模式分布】")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}个实验")

    print(f"\n【执行时间】")
    print(f"  开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总时长: {duration_hours:.2f}小时 ({duration_hours/24:.2f}天)")

    print(f"\n【数据完整性】")
    print(f"  能耗数据: {energy_complete}/{total} ({energy_complete/total*100:.1f}%)")
    print(f"  性能数据: {perf_complete}/{total} ({perf_complete/total*100:.1f}%)")

    # 检查列数
    print(f"\n【CSV格式】")
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        print(f"  列数: {len(header)}")
        print(f"  预期列数: 32 (非并行格式)")

    # 详细分析每个配置
    print(f"\n【详细配置分析】")

    # VulBERTa/mlp 非并行
    mlp_nonparallel = [r for r in rows if r['model'] == 'mlp' and '_parallel' not in r['experiment_id']]
    if mlp_nonparallel:
        print(f"\n  VulBERTa/mlp (非并行): {len(mlp_nonparallel)}个实验")
        params = defaultdict(set)
        for row in mlp_nonparallel:
            if row['hyperparam_epochs']:
                params['epochs'].add(row['hyperparam_epochs'])
            if row['hyperparam_learning_rate']:
                params['learning_rate'].add(row['hyperparam_learning_rate'])
            if row['hyperparam_seed']:
                params['seed'].add(row['hyperparam_seed'])
            if row['hyperparam_weight_decay']:
                params['weight_decay'].add(row['hyperparam_weight_decay'])

        for param, values in sorted(params.items()):
            print(f"    {param}: {len(values)}个唯一值")

    # bug-localization 非并行
    bug_nonparallel = [r for r in rows if r['model'] == 'default' and '_parallel' not in r['experiment_id']]
    if bug_nonparallel:
        print(f"\n  bug-localization (非并行): {len(bug_nonparallel)}个实验")
        params = defaultdict(set)
        for row in bug_nonparallel:
            if row['hyperparam_alpha']:
                params['alpha'].add(row['hyperparam_alpha'])
            if row['hyperparam_kfold']:
                params['kfold'].add(row['hyperparam_kfold'])
            if row['hyperparam_max_iter']:
                params['max_iter'].add(row['hyperparam_max_iter'])
            if row['hyperparam_seed']:
                params['seed'].add(row['hyperparam_seed'])

        for param, values in sorted(params.items()):
            print(f"    {param}: {len(values)}个唯一值")

    # bug-localization 并行
    bug_parallel = [r for r in rows if r['model'] == 'default' and '_parallel' in r['experiment_id']]
    if bug_parallel:
        print(f"\n  bug-localization (并行): {len(bug_parallel)}个实验")
        params = defaultdict(set)
        for row in bug_parallel:
            if row['hyperparam_alpha']:
                params['alpha'].add(row['hyperparam_alpha'])
            if row['hyperparam_kfold']:
                params['kfold'].add(row['hyperparam_kfold'])
            if row['hyperparam_max_iter']:
                params['max_iter'].add(row['hyperparam_max_iter'])
            if row['hyperparam_seed']:
                params['seed'].add(row['hyperparam_seed'])

        for param, values in sorted(params.items()):
            print(f"    {param}: {len(values)}个唯一值")

    print("=" * 70)

if __name__ == '__main__':
    analyze_phase7()
