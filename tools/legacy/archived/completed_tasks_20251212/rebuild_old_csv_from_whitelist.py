#!/usr/bin/env python3
"""
从白名单experiment.json重建老实验CSV

输入:
- results/old_experiment_whitelist.json: 安全实验白名单
- results/run_*/*/experiment.json: 各实验的JSON文件

输出:
- results/summary_old_rebuilt.csv: 重建的老实验CSV

重建原理:
1. 加载白名单（experiment_id -> experiment_source映射）
2. 在所有run目录中查找白名单中的experiment.json文件
3. 从JSON重建CSV行数据
4. 保留原有的experiment_source信息
"""

import json
import os
import glob
import csv

# 标准37列CSV表头（与summary_all.csv一致）
STANDARD_HEADER = [
    'experiment_id', 'timestamp', 'repository', 'model',
    'training_success', 'duration_seconds', 'retries',
    'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
    'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
    'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',
    'perf_accuracy', 'perf_best_val_accuracy', 'perf_map',
    'perf_precision', 'perf_rank1', 'perf_rank5',
    'perf_recall', 'perf_test_accuracy', 'perf_test_loss',
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',
    'experiment_source'
]

def load_whitelist():
    """加载白名单"""
    with open('results/old_experiment_whitelist.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def find_experiment_json(experiment_id, run_dirs):
    """在run目录中查找指定experiment_id的JSON文件

    Args:
        experiment_id: 实验ID（可能包含source前缀，如default__、mutation_1x__）
        run_dirs: 所有run目录列表

    Returns:
        找到的JSON文件路径，或None
    """
    # 去掉source前缀（如default__、mutation_1x__、mutation_2x_safe__）
    # 因为目录名不包含这些前缀
    if '__' in experiment_id:
        actual_id = experiment_id.split('__', 1)[1]  # 取__后面的部分
    else:
        actual_id = experiment_id

    for run_dir in run_dirs:
        # 尝试多种目录命名模式
        possible_patterns = [
            f"{run_dir}/{actual_id}/experiment.json",
            f"{run_dir}/*{actual_id}/experiment.json",
            f"{run_dir}/*{actual_id}*/experiment.json",
        ]

        for pattern in possible_patterns:
            files = glob.glob(pattern)
            if files:
                return files[0]

    return None

def process_experiment_json(json_path, experiment_source, full_experiment_id):
    """处理单个experiment.json文件，提取CSV行数据

    Args:
        json_path: experiment.json文件路径
        experiment_source: 实验来源（default/mutation_1x/mutation_2x_safe）
        full_experiment_id: 完整的实验ID（包含source前缀）

    Returns:
        dict: CSV行数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    row = {col: '' for col in STANDARD_HEADER}  # 初始化所有列为空

    # 基础信息 - 使用完整的experiment_id（包含前缀）
    row['experiment_id'] = full_experiment_id

    # 判断是否为并行实验
    is_parallel = data.get('mode') == 'parallel'

    if is_parallel:
        # 并行实验：数据在foreground中
        fg = data.get('foreground', {})

        row['timestamp'] = data.get('timestamp', '')
        row['repository'] = fg.get('repository', '')
        row['model'] = fg.get('model', '')
        row['training_success'] = fg.get('training_success', '')
        row['duration_seconds'] = fg.get('duration_seconds', '')
        row['retries'] = fg.get('retries', 0)

        # Hyperparameters (from foreground)
        hyper = fg.get('hyperparameters', {})
        row['hyperparam_alpha'] = hyper.get('alpha', '')
        row['hyperparam_batch_size'] = hyper.get('batch_size', '')
        row['hyperparam_dropout'] = hyper.get('dropout', '')
        row['hyperparam_epochs'] = hyper.get('epochs', '')
        row['hyperparam_kfold'] = hyper.get('kfold', '')
        row['hyperparam_learning_rate'] = hyper.get('learning_rate', '')
        row['hyperparam_max_iter'] = hyper.get('max_iter', '')
        row['hyperparam_seed'] = hyper.get('seed', '')
        row['hyperparam_weight_decay'] = hyper.get('weight_decay', '')

        # Performance metrics (from foreground)
        perf = fg.get('performance_metrics', {})
        row['perf_accuracy'] = perf.get('accuracy', '')
        row['perf_best_val_accuracy'] = perf.get('best_val_accuracy', '')
        row['perf_map'] = perf.get('map', '')
        row['perf_precision'] = perf.get('precision', '')
        row['perf_rank1'] = perf.get('rank1', '')
        row['perf_rank5'] = perf.get('rank5', '')
        row['perf_recall'] = perf.get('recall', '')
        row['perf_test_accuracy'] = perf.get('test_accuracy', '')
        row['perf_test_loss'] = perf.get('test_loss', '')

        # Energy metrics (from foreground)
        energy = fg.get('energy_metrics', {})
        row['energy_cpu_pkg_joules'] = energy.get('cpu_energy_pkg_joules', '')
        row['energy_cpu_ram_joules'] = energy.get('cpu_energy_ram_joules', '')
        row['energy_cpu_total_joules'] = energy.get('cpu_energy_total_joules', '')
        row['energy_gpu_avg_watts'] = energy.get('gpu_power_avg_watts', '')
        row['energy_gpu_max_watts'] = energy.get('gpu_power_max_watts', '')
        row['energy_gpu_min_watts'] = energy.get('gpu_power_min_watts', '')
        row['energy_gpu_total_joules'] = energy.get('gpu_energy_total_joules', '')
        row['energy_gpu_temp_avg_celsius'] = energy.get('gpu_temp_avg_celsius', '')
        row['energy_gpu_temp_max_celsius'] = energy.get('gpu_temp_max_celsius', '')
        row['energy_gpu_util_avg_percent'] = energy.get('gpu_util_avg_percent', '')
        row['energy_gpu_util_max_percent'] = energy.get('gpu_util_max_percent', '')

    else:
        # 非并行实验：数据在顶层
        row['timestamp'] = data.get('timestamp', '')
        row['repository'] = data.get('repository', '')
        row['model'] = data.get('model', '')
        row['training_success'] = data.get('training_success', '')
        row['duration_seconds'] = data.get('duration_seconds', '')
        row['retries'] = data.get('retries', 0)

        # Hyperparameters
        hyper = data.get('hyperparameters', {})
        row['hyperparam_alpha'] = hyper.get('alpha', '')
        row['hyperparam_batch_size'] = hyper.get('batch_size', '')
        row['hyperparam_dropout'] = hyper.get('dropout', '')
        row['hyperparam_epochs'] = hyper.get('epochs', '')
        row['hyperparam_kfold'] = hyper.get('kfold', '')
        row['hyperparam_learning_rate'] = hyper.get('learning_rate', '')
        row['hyperparam_max_iter'] = hyper.get('max_iter', '')
        row['hyperparam_seed'] = hyper.get('seed', '')
        row['hyperparam_weight_decay'] = hyper.get('weight_decay', '')

        # Performance metrics
        perf = data.get('performance_metrics', {})
        row['perf_accuracy'] = perf.get('accuracy', '')
        row['perf_best_val_accuracy'] = perf.get('best_val_accuracy', '')
        row['perf_map'] = perf.get('map', '')
        row['perf_precision'] = perf.get('precision', '')
        row['perf_rank1'] = perf.get('rank1', '')
        row['perf_rank5'] = perf.get('rank5', '')
        row['perf_recall'] = perf.get('recall', '')
        row['perf_test_accuracy'] = perf.get('test_accuracy', '')
        row['perf_test_loss'] = perf.get('test_loss', '')

        # Energy metrics
        energy = data.get('energy_metrics', {})
        row['energy_cpu_pkg_joules'] = energy.get('cpu_energy_pkg_joules', '')
        row['energy_cpu_ram_joules'] = energy.get('cpu_energy_ram_joules', '')
        row['energy_cpu_total_joules'] = energy.get('cpu_energy_total_joules', '')
        row['energy_gpu_avg_watts'] = energy.get('gpu_power_avg_watts', '')
        row['energy_gpu_max_watts'] = energy.get('gpu_power_max_watts', '')
        row['energy_gpu_min_watts'] = energy.get('gpu_power_min_watts', '')
        row['energy_gpu_total_joules'] = energy.get('gpu_energy_total_joules', '')
        row['energy_gpu_temp_avg_celsius'] = energy.get('gpu_temp_avg_celsius', '')
        row['energy_gpu_temp_max_celsius'] = energy.get('gpu_temp_max_celsius', '')
        row['energy_gpu_util_avg_percent'] = energy.get('gpu_util_avg_percent', '')
        row['energy_gpu_util_max_percent'] = energy.get('gpu_util_max_percent', '')

    # experiment_source（从白名单）
    row['experiment_source'] = experiment_source

    return row

def rebuild_old_csv():
    """从白名单重建老实验CSV"""

    print("=" * 70)
    print("从白名单experiment.json重建老实验CSV")
    print("=" * 70)
    print()

    # 1. 加载白名单
    print("步骤1: 加载白名单...")
    whitelist = load_whitelist()
    print(f"  ✓ 白名单包含 {len(whitelist)} 个实验")
    print()

    # 2. 获取所有run目录 + 老实验目录
    print("步骤2: 扫描实验目录...")
    run_dirs = sorted(glob.glob("results/run_*"))

    # 添加老实验的三个源目录
    old_experiment_dirs = [
        "results/default",  # 修正拼写（原为defualt）
        "results/mutation_1x",
        "results/mutation_2x_20251122_175401"
    ]

    # 检查这些目录是否存在
    for dir_path in old_experiment_dirs:
        if os.path.exists(dir_path):
            run_dirs.append(dir_path)
        else:
            print(f"  ⚠️  目录不存在: {dir_path}")

    print(f"  ✓ 找到 {len(run_dirs)} 个实验目录")
    print(f"     - {len(run_dirs)-len([d for d in old_experiment_dirs if os.path.exists(d)])} 个run_*目录")
    print(f"     - {len([d for d in old_experiment_dirs if os.path.exists(d)])} 个老实验源目录")
    print()

    # 3. 查找并处理每个白名单实验
    print("步骤3: 查找并处理白名单实验...")
    print()

    all_rows = []
    found_count = 0
    missing_count = 0
    missing_experiments = []

    for i, (experiment_id, experiment_source) in enumerate(whitelist.items(), 1):
        print(f"  [{i}/{len(whitelist)}] {experiment_id:50s} ", end='')

        # 查找JSON文件
        json_path = find_experiment_json(experiment_id, run_dirs)

        if json_path:
            try:
                row = process_experiment_json(json_path, experiment_source, experiment_id)
                all_rows.append(row)
                found_count += 1
                print("✓")
            except Exception as e:
                print(f"✗ (处理失败: {e})")
                missing_count += 1
                missing_experiments.append((experiment_id, f"处理失败: {e}"))
        else:
            print("✗ (未找到)")
            missing_count += 1
            missing_experiments.append((experiment_id, "JSON文件未找到"))

    print()
    print("=" * 70)
    print("处理结果:")
    print(f"  ✓ 成功重建: {found_count} 个实验")
    print(f"  ✗ 缺失/失败: {missing_count} 个实验")
    print()

    # 4. 如果有缺失实验，显示详情
    if missing_experiments:
        print("缺失/失败实验详情:")
        for exp_id, reason in missing_experiments[:10]:  # 最多显示10个
            print(f"  - {exp_id}: {reason}")
        if len(missing_experiments) > 10:
            print(f"  ... 还有 {len(missing_experiments)-10} 个")
        print()

    # 5. 写入CSV
    if all_rows:
        output_file = 'results/summary_old_rebuilt.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=STANDARD_HEADER)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"✓ 重建的老实验CSV已保存: {output_file}")
        print(f"  总行数: {len(all_rows)}")
        print(f"  总列数: {len(STANDARD_HEADER)}")
        print()

        # 统计信息
        source_dist = {}
        repo_dist = {}
        for row in all_rows:
            source = row.get('experiment_source', '')
            source_dist[source] = source_dist.get(source, 0) + 1

            repo = row.get('repository', '')
            repo_dist[repo] = repo_dist.get(repo, 0) + 1

        print("实验源分布:")
        for source, count in sorted(source_dist.items()):
            print(f"  {source:20s}: {count:3d} 个实验")

        print()
        print("Repository分布:")
        for repo, count in sorted(repo_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {repo:40s}: {count:3d} 个实验")

        print()
        print("=" * 70)

        return all_rows, missing_experiments
    else:
        print("✗ 错误: 没有成功重建任何实验")
        return [], missing_experiments

if __name__ == '__main__':
    rows, missing = rebuild_old_csv()

    if missing:
        print()
        print("⚠️  警告: 有实验缺失或处理失败")
        print(f"   建议检查这些实验的数据源")

    if rows:
        print()
        print("✓ 重建成功！")
        print()
        print("下一步: 验证重建数据的正确性")
        print("  python3 tests/test_old_csv_rebuild.py")
