#!/usr/bin/env python3
"""从models_config.json填充raw_data.csv中缺失的超参数默认值

用途: 为单参数变异实验填充未变异的超参数默认值，用于6分组回归分析
方法: 从models_config.json提取默认值，填充空值
数据来源追踪: 添加source列记录每个值的来源（recorded/backfilled）
作者: Claude
日期: 2026-01-14
版本: v1.0

核心逻辑:
1. 加载raw_data.csv和models_config.json
2. 对于每个实验记录：
   - 如果超参数已有值 → 标记为"recorded"
   - 如果超参数为空 → 从models_config.json填充默认值，标记为"backfilled"
3. 生成填充后的数据文件和来源追踪文件
4. 生成填充报告

注意事项:
- 仅填充models_config.json中定义的超参数
- 保留所有原始数据不变
- 添加*_source列追踪每个超参数的数据来源
- 支持--dry-run模式预览填充结果
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


# 超参数列名映射 (raw_data.csv列名 -> models_config.json参数名)
HYPERPARAM_MAPPING = {
    'hyperparam_epochs': 'epochs',
    'hyperparam_max_iter': 'max_iter',
    'hyperparam_learning_rate': 'learning_rate',
    'hyperparam_batch_size': 'batch_size',
    'hyperparam_dropout': 'dropout',
    'hyperparam_weight_decay': 'weight_decay',
    'hyperparam_alpha': 'alpha',
    'hyperparam_kfold': 'kfold',
    'hyperparam_seed': 'seed',
}

# 前台超参数列名映射 (fg_*列)
FG_HYPERPARAM_MAPPING = {
    'fg_hyperparam_epochs': 'epochs',
    'fg_hyperparam_max_iter': 'max_iter',
    'fg_hyperparam_learning_rate': 'learning_rate',
    'fg_hyperparam_batch_size': 'batch_size',
    'fg_hyperparam_dropout': 'dropout',
    'fg_hyperparam_weight_decay': 'weight_decay',
    'fg_hyperparam_alpha': 'alpha',
    'fg_hyperparam_kfold': 'kfold',
    'fg_hyperparam_seed': 'seed',
}


def load_models_config(config_path='mutation/models_config.json'):
    """加载models_config.json"""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def get_default_value(models_config, repository, param_name):
    """
    从models_config.json获取指定仓库和参数的默认值

    Args:
        models_config: models_config.json配置字典
        repository: 仓库名
        param_name: 参数名（models_config.json中的名称）

    Returns:
        默认值，如果不存在返回None
    """
    if repository not in models_config['models']:
        return None

    repo_config = models_config['models'][repository]
    supported_params = repo_config.get('supported_hyperparams', {})

    if param_name not in supported_params:
        return None

    return supported_params[param_name].get('default')


def backfill_hyperparameters(df, models_config, is_foreground=False):
    """
    填充超参数默认值并追踪数据来源

    Args:
        df: 数据DataFrame
        models_config: models_config.json配置
        is_foreground: 是否处理前台(fg_*)列

    Returns:
        tuple: (填充后的DataFrame, 填充统计字典)
    """
    df_filled = df.copy()

    # 选择映射表
    mapping = FG_HYPERPARAM_MAPPING if is_foreground else HYPERPARAM_MAPPING
    repo_col = 'fg_repository' if is_foreground else 'repository'

    # 统计信息
    stats = {
        'total_cells': 0,
        'originally_filled': 0,
        'backfilled': 0,
        'not_applicable': 0,
        'by_param': {},
    }

    # 为每个超参数列添加source列
    for csv_col in mapping.keys():
        if csv_col not in df_filled.columns:
            continue

        source_col = f"{csv_col}_source"
        df_filled[source_col] = None
        stats['by_param'][csv_col] = {
            'total': 0,
            'recorded': 0,
            'backfilled': 0,
            'not_applicable': 0,
        }

    # 逐行填充
    for idx, row in df_filled.iterrows():
        repository = row.get(repo_col)

        if pd.isna(repository):
            continue

        for csv_col, config_param in mapping.items():
            if csv_col not in df_filled.columns:
                continue

            stats['total_cells'] += 1
            stats['by_param'][csv_col]['total'] += 1

            current_value = row.get(csv_col)

            # 判断当前值是否存在
            if pd.notna(current_value) and current_value != '':
                # 已有值，标记为recorded
                df_filled.loc[idx, f"{csv_col}_source"] = 'recorded'
                stats['originally_filled'] += 1
                stats['by_param'][csv_col]['recorded'] += 1
            else:
                # 值缺失，尝试填充
                default_value = get_default_value(models_config, repository, config_param)

                if default_value is not None:
                    # 填充默认值
                    df_filled.loc[idx, csv_col] = default_value
                    df_filled.loc[idx, f"{csv_col}_source"] = 'backfilled'
                    stats['backfilled'] += 1
                    stats['by_param'][csv_col]['backfilled'] += 1
                else:
                    # 该仓库不支持该参数，标记为N/A
                    df_filled.loc[idx, f"{csv_col}_source"] = 'not_applicable'
                    stats['not_applicable'] += 1
                    stats['by_param'][csv_col]['not_applicable'] += 1

    return df_filled, stats


def generate_report(stats_main, stats_fg, output_dir):
    """
    生成填充报告

    Args:
        stats_main: 主超参数统计
        stats_fg: 前台超参数统计
        output_dir: 输出目录
    """
    report_path = output_dir / 'backfill_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("超参数默认值填充报告\n")
        f.write("=" * 80 + "\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        total_cells = stats_main['total_cells'] + stats_fg['total_cells']
        total_recorded = stats_main['originally_filled'] + stats_fg['originally_filled']
        total_backfilled = stats_main['backfilled'] + stats_fg['backfilled']
        total_na = stats_main['not_applicable'] + stats_fg['not_applicable']

        f.write(f"总单元格数: {total_cells}\n")
        f.write(f"  - 原始已有值 (recorded): {total_recorded} ({total_recorded/total_cells*100:.2f}%)\n")
        f.write(f"  - 新填充值 (backfilled): {total_backfilled} ({total_backfilled/total_cells*100:.2f}%)\n")
        f.write(f"  - 不适用 (N/A): {total_na} ({total_na/total_cells*100:.2f}%)\n")
        f.write(f"\n填充后完整性: {(total_recorded + total_backfilled)/total_cells*100:.2f}%\n\n")

        # 主超参数统计
        f.write("## 主超参数列 (repository字段)\n\n")
        for param, stat in stats_main['by_param'].items():
            if stat['total'] == 0:
                continue
            f.write(f"### {param}\n")
            f.write(f"  - 总数: {stat['total']}\n")
            f.write(f"  - recorded: {stat['recorded']} ({stat['recorded']/stat['total']*100:.2f}%)\n")
            f.write(f"  - backfilled: {stat['backfilled']} ({stat['backfilled']/stat['total']*100:.2f}%)\n")
            f.write(f"  - N/A: {stat['not_applicable']} ({stat['not_applicable']/stat['total']*100:.2f}%)\n")
            f.write(f"  - 填充后完整性: {(stat['recorded']+stat['backfilled'])/stat['total']*100:.2f}%\n\n")

        # 前台超参数统计
        f.write("## 前台超参数列 (fg_repository字段)\n\n")
        for param, stat in stats_fg['by_param'].items():
            if stat['total'] == 0:
                continue
            f.write(f"### {param}\n")
            f.write(f"  - 总数: {stat['total']}\n")
            f.write(f"  - recorded: {stat['recorded']} ({stat['recorded']/stat['total']*100:.2f}%)\n")
            f.write(f"  - backfilled: {stat['backfilled']} ({stat['backfilled']/stat['total']*100:.2f}%)\n")
            f.write(f"  - N/A: {stat['not_applicable']} ({stat['not_applicable']/stat['total']*100:.2f}%)\n")
            f.write(f"  - 填充后完整性: {(stat['recorded']+stat['backfilled'])/stat['total']*100:.2f}%\n\n")

    print(f"\n✅ 填充报告已生成: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='从models_config.json填充raw_data.csv中缺失的超参数默认值'
    )
    parser.add_argument(
        '--input',
        default='data/raw_data.csv',
        help='输入CSV文件路径 (默认: data/raw_data.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis/data/energy_research/backfilled',
        help='输出目录 (默认: analysis/data/energy_research/backfilled)'
    )
    parser.add_argument(
        '--models-config',
        default='mutation/models_config.json',
        help='models_config.json路径 (默认: mutation/models_config.json)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不保存文件'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='仅处理前N行（用于测试）'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("超参数默认值填充工具 v1.0")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"配置文件: {args.models_config}")
    if args.dry_run:
        print("⚠️ DRY RUN 模式 - 不会保存文件")
    if args.limit:
        print(f"⚠️ 测试模式 - 仅处理前 {args.limit} 行")
    print("=" * 80)

    # 1. 加载配置
    print("\n阶段1: 加载配置")
    models_config = load_models_config(args.models_config)
    print(f"✅ 成功加载 models_config.json")
    print(f"  - 仓库数: {len(models_config['models'])}")

    # 2. 加载数据
    print("\n阶段2: 加载数据")
    df = pd.read_csv(args.input)
    print(f"✅ 成功加载 {args.input}")
    print(f"  - 原始行数: {len(df)}")
    print(f"  - 原始列数: {len(df.columns)}")

    if args.limit:
        df = df.head(args.limit)
        print(f"  - 限制处理: {len(df)} 行")

    # 3. 填充主超参数
    print("\n阶段3: 填充主超参数 (repository字段)")
    df_filled, stats_main = backfill_hyperparameters(df, models_config, is_foreground=False)
    print(f"✅ 主超参数填充完成")
    print(f"  - 原始已有值: {stats_main['originally_filled']}")
    print(f"  - 新填充值: {stats_main['backfilled']}")
    print(f"  - 不适用: {stats_main['not_applicable']}")

    # 4. 填充前台超参数
    print("\n阶段4: 填充前台超参数 (fg_repository字段)")
    df_filled, stats_fg = backfill_hyperparameters(df_filled, models_config, is_foreground=True)
    print(f"✅ 前台超参数填充完成")
    print(f"  - 原始已有值: {stats_fg['originally_filled']}")
    print(f"  - 新填充值: {stats_fg['backfilled']}")
    print(f"  - 不适用: {stats_fg['not_applicable']}")

    # 5. 保存结果
    output_dir = Path(args.output_dir)

    if not args.dry_run:
        print("\n阶段5: 保存结果")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存填充后的数据
        output_csv = output_dir / 'raw_data_backfilled.csv'
        df_filled.to_csv(output_csv, index=False)
        print(f"✅ 填充后数据已保存: {output_csv}")
        print(f"  - 行数: {len(df_filled)}")
        print(f"  - 列数: {len(df_filled.columns)}")

        # 生成报告
        report_path = generate_report(stats_main, stats_fg, output_dir)

        # 保存统计JSON
        stats_json = output_dir / 'backfill_stats.json'
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump({
                'main_hyperparams': stats_main,
                'fg_hyperparams': stats_fg,
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ 统计JSON已保存: {stats_json}")

    else:
        print("\n⚠️ DRY RUN 模式 - 预览前10行填充结果:")
        print("\n主超参数列:")
        for col in HYPERPARAM_MAPPING.keys():
            if col in df_filled.columns:
                source_col = f"{col}_source"
                print(f"\n{col}:")
                print(df_filled[[col, source_col]].head(10))

    # 6. 总结
    print("\n" + "=" * 80)
    print("填充完成总结")
    print("=" * 80)
    total_cells = stats_main['total_cells'] + stats_fg['total_cells']
    total_recorded = stats_main['originally_filled'] + stats_fg['originally_filled']
    total_backfilled = stats_main['backfilled'] + stats_fg['backfilled']
    total_na = stats_main['not_applicable'] + stats_fg['not_applicable']

    print(f"✅ 总单元格数: {total_cells}")
    print(f"✅ 原始已有值: {total_recorded} ({total_recorded/total_cells*100:.2f}%)")
    print(f"✅ 新填充值: {total_backfilled} ({total_backfilled/total_cells*100:.2f}%)")
    print(f"✅ 不适用: {total_na} ({total_na/total_cells*100:.2f}%)")
    print(f"✅ 填充后完整性: {(total_recorded + total_backfilled)/total_cells*100:.2f}%")

    if not args.dry_run:
        print(f"\n输出文件:")
        print(f"  - 数据: {output_dir / 'raw_data_backfilled.csv'}")
        print(f"  - 报告: {output_dir / 'backfill_report.txt'}")
        print(f"  - 统计: {output_dir / 'backfill_stats.json'}")

    return df_filled


if __name__ == '__main__':
    df = main()
    print("\n✅ 脚本执行完成！")
