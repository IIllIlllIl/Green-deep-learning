#!/usr/bin/env python3
"""
步骤1: 从目录名修复experiment_source

功能:
- 查找results/run_*下的实际实验目录
- 根据目录名推断experiment_source
- 仅修改experiment_source列

日期: 2025-12-11
版本: v1.0
"""

import csv
import os
import glob
import sys
from datetime import datetime


def find_experiment_directory(experiment_id: str) -> str:
    """
    根据experiment_id查找实际的目录名

    命名规则:
    - default__repo_model_001 -> repo_model_001
    - mutation_1x__repo_model_002 -> repo_model_002
    - repo_model_003 -> repo_model_003 (已经是目录名)
    """
    # 尝试去掉前缀
    possible_names = [experiment_id]

    if '__' in experiment_id:
        # 去掉 default__ 或 mutation_*__ 前缀
        folder_name = experiment_id.split('__', 1)[1]
        possible_names.append(folder_name)

    # 在results下所有run_*目录中查找
    for name in possible_names:
        pattern = f"results/run_*/{name}"
        matches = glob.glob(pattern)
        if matches:
            # 返回目录名（不含路径）
            return os.path.basename(matches[0])

    return None


def extract_experiment_source(folder_name: str, experiment_id: str) -> str:
    """
    从experiment_id推断experiment_source

    逻辑:
    - 如果experiment_id以"default__"开头 -> "default"
    - 如果experiment_id以"mutation_"开头 -> 提取mutation部分 (如"mutation_1x")
    - 否则 -> 空字符串
    """
    if experiment_id.startswith('default__'):
        return 'default'
    elif experiment_id.startswith('mutation_'):
        # 提取mutation_1x, mutation_2x_safe等
        parts = experiment_id.split('__')
        if len(parts) >= 2:
            return parts[0]  # mutation_1x, mutation_2x_safe等

    # 对于没有前缀的，返回空字符串
    return ''


def fix_experiment_source(csv_path: str, dry_run: bool = False):
    """修复experiment_source列"""

    print("="*70)
    print("步骤1: 修复experiment_source列")
    print("="*70)
    print(f"输入文件: {csv_path}")
    print(f"模式: {'DRY-RUN（预览）' if dry_run else '实际执行'}")
    print("="*70 + "\n")

    # 1. 读取CSV
    print("📊 读取CSV文件...")
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"✓ 读取了 {len(rows)} 行数据")
    print(f"✓ 列数: {len(fieldnames)}\n")

    # 2. 处理数据
    stats = {
        'total': len(rows),
        'found_dir': 0,
        'not_found_dir': 0,
        'modified': 0,
        'unchanged': 0
    }

    modifications = []

    print("🔧 开始处理数据...")
    print("="*70)

    for i, row in enumerate(rows, 1):
        exp_id = row['experiment_id']
        current_source = row.get('experiment_source', '').strip()

        # 查找实验目录
        folder_name = find_experiment_directory(exp_id)

        if folder_name:
            stats['found_dir'] += 1
            # 从experiment_id推断source
            new_source = extract_experiment_source(folder_name, exp_id)

            if new_source != current_source:
                stats['modified'] += 1
                row['experiment_source'] = new_source
                modifications.append({
                    'row': i,
                    'exp_id': exp_id,
                    'old': current_source,
                    'new': new_source
                })

                # 打印前10个修改
                if len(modifications) <= 10:
                    print(f"[{i}/{len(rows)}] {exp_id}")
                    print(f"  '{current_source}' -> '{new_source}'")
            else:
                stats['unchanged'] += 1
        else:
            stats['not_found_dir'] += 1
            if i <= 5:
                print(f"[{i}/{len(rows)}] {exp_id}")
                print(f"  ⚠️  找不到实验目录")

    print("\n" + "="*70)

    # 3. 保存结果
    if not dry_run:
        print(f"\n💾 写入修复后的CSV: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ 已保存 {len(rows)} 行数据")
    else:
        print("\n🔍 DRY-RUN模式：不实际写入文件")

    # 4. 打印统计
    print("\n" + "="*70)
    print("📈 修复统计")
    print("="*70)
    print(f"总行数:          {stats['total']}")
    print(f"找到目录:        {stats['found_dir']}")
    print(f"未找到目录:      {stats['not_found_dir']}")
    print(f"修改行数:        {stats['modified']}")
    print(f"未变化行数:      {stats['unchanged']}")
    print("="*70)

    # 5. 详细修改列表
    if modifications and len(modifications) <= 20:
        print("\n修改详情:")
        for mod in modifications:
            print(f"  行{mod['row']}: '{mod['old']}' -> '{mod['new']}'")
            print(f"    实验: {mod['exp_id']}")
    elif len(modifications) > 20:
        print(f"\n修改详情（仅显示前20个）:")
        for mod in modifications[:20]:
            print(f"  行{mod['row']}: '{mod['old']}' -> '{mod['new']}'")
        print(f"  ... 还有 {len(modifications)-20} 个修改")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='步骤1: 修复experiment_source列')
    parser.add_argument('--input', default='results/summary_all.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不实际修改文件')

    args = parser.parse_args()

    success = fix_experiment_source(args.input, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
