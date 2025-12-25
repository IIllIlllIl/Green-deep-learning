#!/usr/bin/env python3
"""
步骤3: 填充默认超参数

功能:
- 读取models_config.json获取默认配置
- 填充空的超参数列为默认值
- 只填充空值，不覆盖已有数据

日期: 2025-12-11
版本: v1.0
"""

import csv
import json
import sys
from typing import Dict


def load_models_config(config_path: str) -> Dict:
    """加载模型配置文件"""
    with open(config_path, 'r') as f:
        data = json.load(f)
        return data.get('models', {})


def get_default_hyperparams(models_config: Dict, repository: str, model: str) -> Dict:
    """获取指定模型的默认超参数"""
    if repository not in models_config:
        return {}

    repo_config = models_config[repository]
    supported = repo_config.get('supported_hyperparams', {})

    defaults = {}
    for param, config in supported.items():
        default_value = config.get('default')
        if default_value is not None:
            defaults[param] = default_value

    return defaults


def fill_default_hyperparams(row: Dict, defaults: Dict) -> tuple:
    """
    填充空的超参数为默认值

    返回: (填充数量, 填充的参数列表)
    """
    filled_count = 0
    filled_params = []

    for param, default_value in defaults.items():
        col_name = f'hyperparam_{param}'

        # 确保列存在
        if col_name not in row:
            continue

        # 只填充空值
        if not row[col_name].strip():
            row[col_name] = str(default_value)
            filled_count += 1
            filled_params.append(param)

    return filled_count, filled_params


def fill_hyperparams(csv_path: str, config_path: str, dry_run: bool = False):
    """填充默认超参数"""

    print("="*70)
    print("步骤3: 填充默认超参数")
    print("="*70)
    print(f"输入文件: {csv_path}")
    print(f"配置文件: {config_path}")
    print(f"模式: {'DRY-RUN（预览）' if dry_run else '实际执行'}")
    print("="*70 + "\n")

    # 1. 加载配置
    print("📖 加载模型配置...")
    models_config = load_models_config(config_path)
    print(f"✓ 加载了 {len(models_config)} 个模型配置\n")

    # 2. 读取CSV
    print("📊 读取CSV文件...")
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"✓ 读取了 {len(rows)} 行数据")
    print(f"✓ 列数: {len(fieldnames)}\n")

    # 3. 处理数据
    stats = {
        'total': len(rows),
        'rows_modified': 0,
        'total_filled': 0,
        'no_config': 0
    }

    # 统计每个参数的填充次数
    param_stats = {}
    examples = []

    print("🔧 开始处理数据...")
    print("="*70)

    for i, row in enumerate(rows, 1):
        exp_id = row['experiment_id']
        repo = row['repository']
        model = row['model']

        # 获取默认配置
        defaults = get_default_hyperparams(models_config, repo, model)

        if not defaults:
            stats['no_config'] += 1
            if i <= 3:
                print(f"[{i}/{len(rows)}] {exp_id}")
                print(f"  ⚠️  模型 {repo}/{model} 无配置")
            continue

        # 填充默认值
        filled_count, filled_params = fill_default_hyperparams(row, defaults)

        if filled_count > 0:
            stats['rows_modified'] += 1
            stats['total_filled'] += filled_count

            # 统计每个参数
            for param in filled_params:
                param_stats[param] = param_stats.get(param, 0) + 1

            # 记录示例
            if len(examples) < 15:
                examples.append({
                    'row': i,
                    'exp_id': exp_id,
                    'count': filled_count,
                    'params': filled_params
                })

            if i <= 10:
                print(f"[{i}/{len(rows)}] {exp_id}")
                print(f"  填充了 {filled_count} 个参数: {', '.join(filled_params)}")

    print("\n" + "="*70)

    # 4. 保存结果
    if not dry_run:
        print(f"\n💾 写入修复后的CSV: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ 已保存 {len(rows)} 行数据")
    else:
        print("\n🔍 DRY-RUN模式：不实际写入文件")

    # 5. 打印统计
    print("\n" + "="*70)
    print("📈 填充统计")
    print("="*70)
    print(f"总行数:            {stats['total']}")
    print(f"修改的行数:        {stats['rows_modified']}")
    print(f"填充的参数值总数:  {stats['total_filled']}")
    print(f"无模型配置:        {stats['no_config']}")
    print("="*70)

    # 6. 参数统计
    if param_stats:
        print("\n各参数填充次数:")
        for param, count in sorted(param_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {count}")

    # 7. 显示示例
    if examples:
        print(f"\n填充示例（前{len(examples)}个）:")
        for ex in examples:
            print(f"  行{ex['row']}: 填充了{ex['count']}个参数")
            print(f"    实验: {ex['exp_id']}")
            print(f"    参数: {', '.join(ex['params'])}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='步骤3: 填充默认超参数')
    parser.add_argument('--input', default='results/summary_all.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--config', default='mutation/models_config.json',
                       help='模型配置文件路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不实际修改文件')

    args = parser.parse_args()

    success = fill_hyperparams(args.input, args.config, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
