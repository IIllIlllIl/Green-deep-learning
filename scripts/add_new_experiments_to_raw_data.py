#!/usr/bin/env python3
"""
从run_20251212_224937会话中提取4个新实验并追加到raw_data.csv

这4个实验是Phase 2诊断实验，在Phase 3修复之前运行，
需要从terminal_output.txt重新提取性能数据。
"""

import json
import csv
import re
from pathlib import Path
from datetime import datetime

# 配置
SESSION_DIR = Path('results/run_20251212_224937')
RAW_DATA_CSV = Path('results/raw_data.csv')
MODELS_CONFIG = Path('mutation/models_config.json')

def load_models_config():
    """加载models_config.json"""
    with open(MODELS_CONFIG, 'r') as f:
        return json.load(f)['models']

def extract_performance_from_terminal_output(terminal_output_path, log_patterns):
    """从terminal_output.txt提取性能指标"""
    if not terminal_output_path.exists():
        return {}

    with open(terminal_output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    metrics = {}
    for metric_name, pattern in log_patterns.items():
        try:
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1))
                metrics[f'perf_{metric_name}'] = value
        except (ValueError, IndexError):
            pass

    return metrics

def load_experiment_json(exp_dir):
    """加载experiment.json"""
    json_path = exp_dir / 'experiment.json'
    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    print('=' * 80)
    print('添加4个新实验到raw_data.csv')
    print('=' * 80)
    print()

    # 加载models配置
    models_config = load_models_config()

    # 读取现有raw_data.csv
    with open(RAW_DATA_CSV, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    print(f'✅ 加载现有数据: {len(existing_rows)}行')
    print()

    # 查找新实验
    new_experiments = []
    for exp_dir in sorted(SESSION_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == '__pycache__':
            continue

        # 加载experiment.json
        exp_data = load_experiment_json(exp_dir)
        if not exp_data:
            print(f'⚠️  跳过 {exp_dir.name}: 无experiment.json')
            continue

        # 获取log_patterns
        repo = exp_data.get('repository')
        model = exp_data.get('model')

        if repo not in models_config:
            print(f'⚠️  跳过 {exp_dir.name}: 仓库配置未找到 ({repo})')
            continue

        log_patterns = models_config[repo].get('performance_metrics', {}).get('log_patterns', {})

        # 从terminal_output.txt提取性能数据
        terminal_output = exp_dir / 'terminal_output.txt'
        perf_metrics = extract_performance_from_terminal_output(terminal_output, log_patterns)

        # 合并experiment.json和性能数据
        # 复制基础数据
        row = {key: '' for key in fieldnames}

        # 填充experiment.json中的数据
        row['experiment_id'] = exp_data.get('experiment_id', '')
        row['timestamp'] = exp_data.get('timestamp', '')
        row['repository'] = exp_data.get('repository', '')
        row['model'] = exp_data.get('model', '')
        row['training_success'] = str(exp_data.get('training_success', ''))
        row['duration_seconds'] = str(exp_data.get('duration_seconds', ''))
        row['retries'] = str(exp_data.get('retries', 0))
        row['experiment_source'] = exp_data.get('experiment_source', '')
        row['num_mutated_params'] = str(exp_data.get('num_mutated_params', ''))
        row['mutated_param'] = exp_data.get('mutated_param', '')
        row['mode'] = exp_data.get('mode', '')
        row['error_message'] = exp_data.get('error_message', '')

        # 填充超参数
        hyperparams = exp_data.get('hyperparameters', {})
        for key, value in hyperparams.items():
            col_name = f'hyperparam_{key}'
            if col_name in fieldnames:
                row[col_name] = str(value)

        # 填充能耗数据
        energy = exp_data.get('energy_consumption', {})
        for key, value in energy.items():
            col_name = f'energy_{key}'
            if col_name in fieldnames:
                row[col_name] = str(value)

        # 填充性能数据（从terminal_output.txt提取）
        for key, value in perf_metrics.items():
            if key in fieldnames:
                row[key] = str(value)

        new_experiments.append(row)

        print(f'✅ {exp_dir.name}:')
        print(f'   训练成功: {row["training_success"]}')
        print(f'   性能指标: {list(perf_metrics.keys())}')
        print()

    if not new_experiments:
        print('⚠️  未找到新实验')
        return

    print(f'=== 总结 ===')
    print(f'找到新实验: {len(new_experiments)}个')
    print()

    # 备份raw_data.csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = RAW_DATA_CSV.parent / f'raw_data.csv.backup_{timestamp}'
    import shutil
    shutil.copy(RAW_DATA_CSV, backup_path)
    print(f'✅ 已备份: {backup_path}')

    # 追加新实验到raw_data.csv
    all_rows = existing_rows + new_experiments

    with open(RAW_DATA_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'✅ 已更新: {RAW_DATA_CSV}')
    print(f'   原始: {len(existing_rows)}行')
    print(f'   新增: {len(new_experiments)}行')
    print(f'   总计: {len(all_rows)}行')
    print()

    # 验证
    with open(RAW_DATA_CSV, 'r') as f:
        reader = csv.DictReader(f)
        final_rows = list(reader)

    print(f'✅ 验证: {len(final_rows)}行 (预期{len(all_rows)}行)')

    if len(final_rows) == len(all_rows):
        print('✅ 数据完整性验证通过')
    else:
        print('❌ 数据完整性验证失败')

    print()
    print('=' * 80)
    print('完成')
    print('=' * 80)

if __name__ == '__main__':
    main()
