#!/usr/bin/env python3
"""
步骤2: 新增mutated_param列

功能:
- 读取models_config.json获取默认配置
- 比较每行的超参数与默认值
- 识别被变异的参数
- 新增mutated_param列

日期: 2025-12-11
版本: v1.0
"""

import csv
import json
import sys
from typing import Dict, Optional


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


def identify_mutated_param(row: Dict, defaults: Dict) -> Optional[str]:
    """
    识别被变异的超参数

    逻辑: 比较实际值与默认值，找出唯一不同的参数
    """
    mutated_params = []

    for param, default_value in defaults.items():
        col_name = f'hyperparam_{param}'
        actual_value = row.get(col_name, '').strip()

        if not actual_value:
            continue

        # 类型转换并比较
        try:
            if isinstance(default_value, int):
                if int(float(actual_value)) != default_value:
                    mutated_params.append(param)
            elif isinstance(default_value, float):
                if abs(float(actual_value) - default_value) > 1e-9:
                    mutated_params.append(param)
            else:
                if str(actual_value) != str(default_value):
                    mutated_params.append(param)
        except (ValueError, TypeError):
            # 无法转换，跳过比较
            pass

    # 只返回单参数变异的情况
    if len(mutated_params) == 1:
        return mutated_params[0]
    elif len(mutated_params) > 1:
        # 多参数变异，记录但标记异常
        return f"MULTIPLE:[{','.join(mutated_params)}]"

    return None


def add_mutated_param_column(csv_path: str, config_path: str, dry_run: bool = False):
    """新增mutated_param列"""

    print("="*70)
    print("步骤2: 新增mutated_param列")
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
    print(f"✓ 当前列数: {len(fieldnames)}\n")

    # 3. 添加mutated_param列（如果不存在）
    if 'mutated_param' in fieldnames:
        print("⚠️  mutated_param列已存在，将覆盖其值\n")
    else:
        # 在experiment_source后面插入
        if 'experiment_source' in fieldnames:
            idx = fieldnames.index('experiment_source') + 1
            fieldnames.insert(idx, 'mutated_param')
            print(f"✓ 在experiment_source后面插入mutated_param列")
        else:
            fieldnames.append('mutated_param')
            print(f"✓ 在末尾添加mutated_param列")
        print(f"✓ 新列数: {len(fieldnames)}\n")

    # 4. 处理数据
    stats = {
        'total': len(rows),
        'has_mutated': 0,
        'no_mutated': 0,
        'multiple_mutated': 0,
        'no_config': 0
    }

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
            row['mutated_param'] = ''
            if i <= 3:
                print(f"[{i}/{len(rows)}] {exp_id}")
                print(f"  ⚠️  模型 {repo}/{model} 无配置")
            continue

        # 识别变异参数
        mutated = identify_mutated_param(row, defaults)

        if mutated:
            if mutated.startswith('MULTIPLE:'):
                stats['multiple_mutated'] += 1
            else:
                stats['has_mutated'] += 1

            row['mutated_param'] = mutated

            # 记录示例
            if len(examples) < 10:
                examples.append({
                    'row': i,
                    'exp_id': exp_id,
                    'mutated': mutated
                })

            if i <= 10 or mutated.startswith('MULTIPLE:'):
                print(f"[{i}/{len(rows)}] {exp_id}")
                print(f"  mutated_param: {mutated}")
        else:
            stats['no_mutated'] += 1
            row['mutated_param'] = ''

    print("\n" + "="*70)

    # 5. 保存结果
    if not dry_run:
        print(f"\n💾 写入修复后的CSV: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ 已保存 {len(rows)} 行数据")
    else:
        print("\n🔍 DRY-RUN模式：不实际写入文件")

    # 6. 打印统计
    print("\n" + "="*70)
    print("📈 处理统计")
    print("="*70)
    print(f"总行数:            {stats['total']}")
    print(f"识别到单参数变异:  {stats['has_mutated']}")
    print(f"识别到多参数变异:  {stats['multiple_mutated']}")
    print(f"无变异（空值）:    {stats['no_mutated']}")
    print(f"无模型配置:        {stats['no_config']}")
    print("="*70)

    # 7. 显示示例
    if examples:
        print("\n变异参数示例（前10个）:")
        for ex in examples:
            print(f"  行{ex['row']}: {ex['mutated']}")
            print(f"    实验: {ex['exp_id']}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='步骤2: 新增mutated_param列')
    parser.add_argument('--input', default='results/summary_all.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--config', default='mutation/models_config.json',
                       help='模型配置文件路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不实际修改文件')

    args = parser.parse_args()

    success = add_mutated_param_column(args.input, args.config, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
