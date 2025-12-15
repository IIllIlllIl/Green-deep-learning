#!/usr/bin/env python3
"""
深度分析共用性能指标问题

功能：
1. 找到所有共享experiment_id的实验
2. 通过时间戳匹配对应的JSON文件
3. 验证JSON中的性能指标与CSV中的是否一致
4. 检查是否存在数据加载错误

用法: python3 scripts/analyze_shared_performance_issue.py

版本: 1.0
创建日期: 2025-12-15
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def find_json_by_timestamp(experiment_id, timestamp):
    """通过experiment_id和timestamp查找对应的JSON文件"""

    # 尝试在所有results/run_*目录中查找
    results_dir = Path('results')

    for session_dir in sorted(results_dir.glob('run_*')):
        if not session_dir.is_dir():
            continue

        # 查找匹配experiment_id的目录
        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # 检查是否包含experiment_id
            if experiment_id not in exp_dir.name:
                continue

            json_path = exp_dir / 'experiment.json'
            if not json_path.exists():
                continue

            # 读取JSON检查时间戳
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                json_timestamp = data.get('timestamp', '')

                # 匹配时间戳
                if json_timestamp == timestamp:
                    return json_path, data
            except:
                continue

    return None, None

def extract_performance_from_json(exp_data):
    """从experiment.json提取性能指标"""

    perf_metrics = {}

    # 检查是否为并行模式
    if exp_data.get('mode') == 'parallel':
        # 从foreground提取性能指标
        fg_perf = exp_data.get('foreground', {}).get('performance_metrics', {})
        for key, value in fg_perf.items():
            perf_metrics[f'perf_{key}'] = value
    else:
        # 从顶层提取性能指标
        top_perf = exp_data.get('performance_metrics', {})
        for key, value in top_perf.items():
            perf_metrics[f'perf_{key}'] = value

    return perf_metrics

def analyze_shared_performance():
    """分析共用性能指标问题"""

    csv_path = Path('results/raw_data.csv')

    print('=' * 80)
    print('共用性能指标深度分析')
    print('=' * 80)
    print('')

    # 读取CSV数据
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'总数据行数: {len(rows)}')
    print('')

    # 按experiment_id分组
    by_exp_id = defaultdict(list)
    for row in rows:
        exp_id = row.get('experiment_id', '')
        by_exp_id[exp_id].append(row)

    # 找出共享experiment_id的情况
    shared_exp_ids = {exp_id: rows_list for exp_id, rows_list in by_exp_id.items()
                     if len(rows_list) > 1}

    print(f'发现 {len(shared_exp_ids)} 个experiment_id有多行数据')
    print('')

    # 分析每个共享ID
    issues_found = []

    for exp_id, exp_rows in sorted(shared_exp_ids.items()):
        print('=' * 80)
        print(f'experiment_id: {exp_id}')
        print('=' * 80)
        print(f'数据行数: {len(exp_rows)}')
        print('')

        # 提取每行的性能指标
        csv_perf_data = []
        for i, row in enumerate(exp_rows, 1):
            timestamp = row.get('timestamp', '')

            # 提取CSV中的性能指标
            csv_perf = {}
            for key, value in row.items():
                if key.startswith('perf_') and value:
                    csv_perf[key] = value

            csv_perf_data.append({
                'row_num': i,
                'timestamp': timestamp,
                'csv_perf': csv_perf,
                'row': row
            })

            print(f'行 {i}:')
            print(f'  timestamp: {timestamp}')
            print(f'  training_success: {row.get("training_success", "")}')
            print(f'  mode: {row.get("mode", "")}')
            print(f'  CSV中的性能指标: {len(csv_perf)}个')
            if csv_perf:
                for key, value in list(csv_perf.items())[:3]:
                    print(f'    {key}: {value}')
                if len(csv_perf) > 3:
                    print(f'    ... 还有{len(csv_perf)-3}个')

        print('')

        # 检查CSV中的性能指标是否相同
        if len(csv_perf_data) > 1:
            # 比较性能指标
            first_perf = csv_perf_data[0]['csv_perf']
            all_same = True

            for data in csv_perf_data[1:]:
                if data['csv_perf'] != first_perf:
                    all_same = False
                    break

            if all_same and first_perf:
                print('⚠️  CSV中所有行的性能指标完全相同！')
                print('')
            else:
                print('✅ CSV中各行的性能指标不同')
                print('')

        # 尝试匹配JSON文件
        print('查找对应的JSON文件:')
        print('')

        json_perf_data = []

        for data in csv_perf_data:
            timestamp = data['timestamp']
            print(f'行 {data["row_num"]} (timestamp: {timestamp}):')

            json_path, json_data = find_json_by_timestamp(exp_id, timestamp)

            if json_path:
                print(f'  ✅ 找到JSON: {json_path}')

                # 提取JSON中的性能指标
                json_perf = extract_performance_from_json(json_data)

                print(f'  JSON中的性能指标: {len(json_perf)}个')
                if json_perf:
                    for key, value in list(json_perf.items())[:3]:
                        print(f'    {key}: {value}')
                    if len(json_perf) > 3:
                        print(f'    ... 还有{len(json_perf)-3}个')

                json_perf_data.append({
                    'row_num': data['row_num'],
                    'json_path': json_path,
                    'json_perf': json_perf
                })

                # 比较CSV和JSON的性能指标
                csv_perf = data['csv_perf']

                if csv_perf and json_perf:
                    # 检查是否匹配
                    match = True
                    for key, csv_value in csv_perf.items():
                        json_value = json_perf.get(key)
                        if json_value is None:
                            continue

                        # 转换为字符串比较
                        if str(csv_value) != str(json_value):
                            match = False
                            print(f'  ❌ 不匹配: {key}')
                            print(f'     CSV: {csv_value}')
                            print(f'     JSON: {json_value}')

                    if match:
                        print('  ✅ CSV和JSON的性能指标匹配')
                else:
                    print('  ⚠️  无法比较（性能指标为空）')
            else:
                print(f'  ❌ 未找到JSON文件')

            print('')

        # 检查JSON中的性能指标是否相同
        if len(json_perf_data) > 1:
            first_json_perf = json_perf_data[0]['json_perf']
            all_json_same = True

            for data in json_perf_data[1:]:
                if data['json_perf'] != first_json_perf:
                    all_json_same = False
                    break

            if all_json_same and first_json_perf:
                print('❌ 问题确认: JSON文件中的性能指标也完全相同！')
                print('   这意味着问题出在训练过程，不是数据提取过程')
                print('')

                issues_found.append({
                    'exp_id': exp_id,
                    'type': 'json_identical',
                    'row_count': len(exp_rows),
                    'timestamps': [d['timestamp'] for d in csv_perf_data]
                })
            else:
                print('✅ JSON文件中的性能指标不同')
                print('   CSV中的相同性能指标可能是数据提取错误')
                print('')

                issues_found.append({
                    'exp_id': exp_id,
                    'type': 'extraction_error',
                    'row_count': len(exp_rows),
                    'timestamps': [d['timestamp'] for d in csv_perf_data]
                })

        print('')

    # 总结
    print('=' * 80)
    print('总结')
    print('=' * 80)
    print('')

    print(f'检查的共享experiment_id数: {len(shared_exp_ids)}')
    print(f'发现问题的案例数: {len(issues_found)}')
    print('')

    if issues_found:
        print('问题详情:')
        print('')

        json_identical_cases = [i for i in issues_found if i['type'] == 'json_identical']
        extraction_error_cases = [i for i in issues_found if i['type'] == 'extraction_error']

        if json_identical_cases:
            print(f'1. JSON文件中性能指标就相同 ({len(json_identical_cases)}个案例):')
            for case in json_identical_cases:
                print(f'   - {case["exp_id"]}: {case["row_count"]}行数据')
                print(f'     时间范围: {min(case["timestamps"])} ~ {max(case["timestamps"])}')
            print('')
            print('   原因: 可能是实验配置完全相同，产生了相同的性能结果')
            print('   或者: 实验重跑时加载了缓存的结果')
            print('')

        if extraction_error_cases:
            print(f'2. 数据提取错误 ({len(extraction_error_cases)}个案例):')
            for case in extraction_error_cases:
                print(f'   - {case["exp_id"]}: {case["row_count"]}行数据')
                print(f'     时间范围: {min(case["timestamps"])} ~ {max(case["timestamps"])}')
            print('')
            print('   原因: 数据提取脚本可能使用了错误的键查询')
            print('   建议: 检查append_session_to_raw_data.py的逻辑')
            print('')
    else:
        print('✅ 未发现明确的问题')
        print('')

    return issues_found

if __name__ == '__main__':
    analyze_shared_performance()
