#!/usr/bin/env python3
"""
预处理全局标准化数据供DiBS使用

DiBS要求：
1. 无缺失值（NaN）
2. 所有列都是数值型
3. 适度的特征数

全局标准化数据的特殊处理：
1. 保留 hyperparam_seed = -1（表示"未设置seed"）
2. 结构性NaN填充策略：
   - 组特有超参数：用该组内非NaN值的均值填充
   - 性能指标：用-999标记（区分结构性缺失）
3. 能耗列：无缺失，保持不变
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime


def preprocess_group_for_dibs(group_name: str, input_dir: Path, output_dir: Path,
                             verbose: bool = True) -> pd.DataFrame:
    """预处理单个组的数据供DiBS使用"""

    print(f"\n预处理 {group_name}...")
    print("-" * 40)

    # 1. 加载全局标准化数据
    input_file = input_dir / f"{group_name}_global_std.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"文件不存在: {input_file}")

    df = pd.read_csv(input_file)

    if verbose:
        print(f"  原始数据: {len(df)}行 × {len(df.columns)}列")
        print(f"  缺失值总数: {df.isnull().sum().sum()}")

    # 2. 移除timestamp列（DiBS无法处理字符串）
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        if verbose:
            print(f"  移除 timestamp 列")

    # 3. 处理 hyperparam_seed（保留-1，表示"未设置seed"）
    if 'hyperparam_seed' in df.columns:
        seed_nan_count = df['hyperparam_seed'].isna().sum()
        if seed_nan_count > 0:
            # 理论上不应该有NaN，因为我们在生成时已经填充了-1
            df['hyperparam_seed'].fillna(-1, inplace=True)
            if verbose:
                print(f"  填充 hyperparam_seed {seed_nan_count}个NaN为-1")

    # 4. 识别列类型
    energy_cols = [col for col in df.columns if 'energy' in col.lower()]
    hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
    perf_cols = [col for col in df.columns if col.startswith('perf_')]
    model_cols = [col for col in df.columns if col.startswith('model_')]
    interaction_cols = [col for col in df.columns if '_x_is_parallel' in col]

    if verbose:
        print(f"  列分类:")
        print(f"    能耗列: {len(energy_cols)}个")
        print(f"    超参数列: {len(hyperparam_cols)}个")
        print(f"    性能指标列: {len(perf_cols)}个")
        print(f"    模型列: {len(model_cols)}个")
        print(f"    交互项列: {len(interaction_cols)}个")

    # 5. 处理缺失值（按列类型不同策略）
    fill_info = {
        'columns_filled': [],
        'strategy_used': {},
        'missing_counts': {}
    }

    for col in df.columns:
        missing_count = df[col].isna().sum()

        if missing_count == 0:
            continue

        fill_info['missing_counts'][col] = missing_count

        # 检查是否全列都是NaN
        is_all_nan = missing_count == len(df)

        if col in energy_cols:
            # 能耗列理论上不应该有NaN
            if is_all_nan:
                df[col].fillna(0, inplace=True)  # 全NaN用0填充
                strategy = "energy_all_nan_fill_0"
            else:
                group_mean = df[col].mean()
                df[col].fillna(group_mean, inplace=True)
                strategy = f"energy_col_group_mean({group_mean:.4f})"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

        elif col in hyperparam_cols:
            # 超参数列：结构性NaN
            if is_all_nan:
                df[col].fillna(0, inplace=True)  # 全NaN用0填充
                strategy = "hyperparam_all_nan_fill_0"
            else:
                group_mean = df[col].mean()
                df[col].fillna(group_mean, inplace=True)
                strategy = f"hyperparam_group_mean({group_mean:.4f})"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

        elif col in perf_cols:
            # 性能指标：结构性NaN
            if is_all_nan:
                df[col].fillna(-999, inplace=True)  # 全NaN用-999标记
                strategy = "perf_all_nan_marked_missing"
            else:
                df[col].fillna(-999, inplace=True)  # 部分NaN也用-999标记
                strategy = "perf_marked_missing(-999)"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

        elif col in model_cols:
            # 模型列：二值变量
            if is_all_nan:
                df[col].fillna(0, inplace=True)  # 全NaN用0填充（表示该模型不存在）
                strategy = "model_all_nan_fill_0"
            else:
                df[col].fillna(0, inplace=True)  # 部分NaN也用0填充
                strategy = "model_fill_0"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

        elif col in interaction_cols:
            # 交互项
            if is_all_nan:
                df[col].fillna(0, inplace=True)
                strategy = "interaction_all_nan_fill_0"
            else:
                df[col].fillna(0, inplace=True)
                strategy = "interaction_fill_0"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

        else:
            # 其他列
            if is_all_nan:
                df[col].fillna(0, inplace=True)
                strategy = "other_all_nan_fill_0"
            else:
                col_mean = df[col].mean()
                df[col].fillna(col_mean, inplace=True)
                strategy = f"other_col_mean({col_mean:.4f})"
            fill_info['strategy_used'][col] = strategy
            fill_info['columns_filled'].append(col)

    # 6. 验证无缺失值
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        raise ValueError(f"预处理后仍有 {remaining_missing} 个缺失值!")

    if verbose:
        print(f"  缺失值处理:")
        print(f"    处理的列数: {len(fill_info['columns_filled'])}")
        print(f"    剩余缺失值: {remaining_missing}")

    # 7. 保存预处理后的数据
    output_file = output_dir / f"{group_name}_dibs_ready.csv"
    df.to_csv(output_file, index=False)

    if verbose:
        print(f"  保存至: {output_file}")

    # 8. 保存填充信息（转换numpy类型为Python类型）
    info_file = output_dir / f"{group_name}_preprocess_info.json"

    # 转换numpy类型为Python原生类型
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        else:
            return obj

    fill_info_python = convert_to_python(fill_info)

    with open(info_file, 'w') as f:
        json.dump(fill_info_python, f, indent=2)

    return df, fill_info


def main():
    parser = argparse.ArgumentParser(description="预处理全局标准化数据供DiBS使用")
    parser.add_argument("--group", type=str, required=True,
                       help="组名 (如: group1_examples)")
    parser.add_argument("--input-dir", type=str,
                       default="data/energy_research/6groups_global_std",
                       help="输入目录 (默认: data/energy_research/6groups_global_std)")
    parser.add_argument("--output-dir", type=str,
                       default="data/energy_research/6groups_dibs_ready",
                       help="输出目录 (默认: data/energy_research/6groups_dibs_ready)")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细信息")

    args = parser.parse_args()

    print("=" * 80)
    print("全局标准化数据预处理 (DiBS版本)")
    print("=" * 80)

    # 创建目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 预处理
    try:
        df, fill_info = preprocess_group_for_dibs(
            args.group, input_dir, output_dir, args.verbose
        )

        print(f"\n✅ 预处理完成: {args.group}")
        print(f"   输出文件: {output_dir / f'{args.group}_dibs_ready.csv'}")
        print(f"   数据形状: {df.shape}")
        print(f"   处理的缺失值列: {len(fill_info['columns_filled'])}")

    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        raise

    print("\n" + "=" * 80)
    print("预处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
