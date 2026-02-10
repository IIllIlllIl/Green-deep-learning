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

v2.0 修改 (2026-02-09):
- 删除全0值的列（不适用于该组的超参数和模型）
- Group4超参数语义合并：max_iter→epochs, alpha→l2_regularization
- 删除重命名后的原始列（避免重复）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime


def remove_all_zero_columns(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    删除全0值的列（表示该变量不适用于此组）

    参数:
        df: 输入DataFrame
        verbose: 是否输出详细信息

    返回:
        df: 删除全0列后的DataFrame
        removed_cols: 被删除的列名列表
    """
    removed_cols = []
    for col in df.columns:
        # 检查是否全为0（或接近0的浮点数）
        if (df[col] == 0).all() or (df[col].abs() < 1e-10).all():
            removed_cols.append(col)

    if removed_cols:
        df = df.drop(columns=removed_cols)
        if verbose:
            print(f"  删除全0列: {len(removed_cols)}列")
            for col in removed_cols:
                print(f"    - {col}")

    return df, removed_cols


def rename_group4_hyperparams(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Group4超参数语义合并：
    - hyperparam_max_iter → hyperparam_epochs
    - hyperparam_alpha → hyperparam_l2_regularization

    参数:
        df: 输入DataFrame
        verbose: 是否输出详细信息

    返回:
        df: 重命名后的DataFrame
        rename_info: 重命名信息字典
    """
    rename_info = {
        'renamed_columns': {},
        'dropped_columns': []
    }

    # 检查是否为Group4（通过特征列判断）
    is_group4 = 'hyperparam_max_iter' in df.columns or 'hyperparam_alpha' in df.columns

    if not is_group4:
        return df, rename_info

    # 重命名 hyperparam_max_iter → hyperparam_epochs
    if 'hyperparam_max_iter' in df.columns:
        # 检查是否已存在 hyperparam_epochs
        if 'hyperparam_epochs' in df.columns:
            # 检查hyperparam_epochs是否全为NaN（如果是，说明是占位列，可以替换）
            epochs_notna = df['hyperparam_epochs'].notna().sum()
            max_iter_notna = df['hyperparam_max_iter'].notna().sum()

            if epochs_notna == 0 and max_iter_notna > 0:
                # hyperparam_epochs全为NaN，hyperparam_max_iter有数据：替换
                df = df.drop(columns=['hyperparam_epochs'])
                df = df.rename(columns={'hyperparam_max_iter': 'hyperparam_epochs'})
                rename_info['renamed_columns']['hyperparam_max_iter'] = 'hyperparam_epochs'
                if verbose:
                    print(f"  替换占位列: hyperparam_epochs(全NaN) ← hyperparam_max_iter({max_iter_notna}有效值)")
            elif max_iter_notna == 0:
                # hyperparam_max_iter全为NaN，删除它
                df = df.drop(columns=['hyperparam_max_iter'])
                rename_info['dropped_columns'].append('hyperparam_max_iter')
                if verbose:
                    print(f"  删除空列: hyperparam_max_iter(全NaN)")
            else:
                # 两列都有数据，删除源列避免重复
                df = df.drop(columns=['hyperparam_max_iter'])
                rename_info['dropped_columns'].append('hyperparam_max_iter')
                if verbose:
                    print(f"  删除原始列: hyperparam_max_iter(保留已有的hyperparam_epochs)")
        else:
            # 不存在，进行重命名
            df = df.rename(columns={'hyperparam_max_iter': 'hyperparam_epochs'})
            rename_info['renamed_columns']['hyperparam_max_iter'] = 'hyperparam_epochs'
            if verbose:
                print(f"  重命名: hyperparam_max_iter → hyperparam_epochs")

    # 重命名 hyperparam_alpha → hyperparam_l2_regularization
    if 'hyperparam_alpha' in df.columns:
        # 检查是否已存在 hyperparam_l2_regularization
        if 'hyperparam_l2_regularization' in df.columns:
            # 检查hyperparam_l2_regularization是否全为NaN
            l2_notna = df['hyperparam_l2_regularization'].notna().sum()
            alpha_notna = df['hyperparam_alpha'].notna().sum()

            if l2_notna == 0 and alpha_notna > 0:
                # hyperparam_l2_regularization全为NaN，hyperparam_alpha有数据：替换
                df = df.drop(columns=['hyperparam_l2_regularization'])
                df = df.rename(columns={'hyperparam_alpha': 'hyperparam_l2_regularization'})
                rename_info['renamed_columns']['hyperparam_alpha'] = 'hyperparam_l2_regularization'
                if verbose:
                    print(f"  替换占位列: hyperparam_l2_regularization(全NaN) ← hyperparam_alpha({alpha_notna}有效值)")
            elif alpha_notna == 0:
                # hyperparam_alpha全为NaN，删除它
                df = df.drop(columns=['hyperparam_alpha'])
                rename_info['dropped_columns'].append('hyperparam_alpha')
                if verbose:
                    print(f"  删除空列: hyperparam_alpha(全NaN)")
            else:
                # 两列都有数据，删除源列避免重复
                df = df.drop(columns=['hyperparam_alpha'])
                rename_info['dropped_columns'].append('hyperparam_alpha')
                if verbose:
                    print(f"  删除原始列: hyperparam_alpha(保留已有的hyperparam_l2_regularization)")
        else:
            # 不存在，进行重命名
            df = df.rename(columns={'hyperparam_alpha': 'hyperparam_l2_regularization'})
            rename_info['renamed_columns']['hyperparam_alpha'] = 'hyperparam_l2_regularization'
            if verbose:
                print(f"  重命名: hyperparam_alpha → hyperparam_l2_regularization")

    # v2.0修复：重命名交互项列
    # 交互项格式: hyperparam_{name}_x_is_parallel
    interaction_rename_map = {
        'hyperparam_max_iter_x_is_parallel': 'hyperparam_epochs_x_is_parallel',
        'hyperparam_alpha_x_is_parallel': 'hyperparam_l2_regularization_x_is_parallel'
    }

    for old_col, new_col in interaction_rename_map.items():
        if old_col in df.columns:
            # 检查目标列是否已存在
            if new_col in df.columns:
                # 检查new_col是否全为0/NaN
                new_notna = df[new_col].notna().sum() & (df[new_col] != 0).sum()
                old_notna = df[old_col].notna().sum() & (df[old_col] != 0).sum()

                if new_notna == 0 and old_notna > 0:
                    # 目标列为空，源列有数据：替换
                    df = df.drop(columns=[new_col])
                    df = df.rename(columns={old_col: new_col})
                    rename_info['renamed_columns'][old_col] = new_col
                    if verbose:
                        print(f"  替换交互项: {new_col}(全0) ← {old_col}({old_notna}有效值)")
                else:
                    # 删除源列
                    df = df.drop(columns=[old_col])
                    rename_info['dropped_columns'].append(old_col)
                    if verbose and old_notna > 0:
                        print(f"  删除交互项: {old_col}(保留已有的{new_col})")
            else:
                # 目标列不存在，直接重命名
                df = df.rename(columns={old_col: new_col})
                rename_info['renamed_columns'][old_col] = new_col
                if verbose:
                    print(f"  重命名交互项: {old_col} → {new_col}")

    return df, rename_info


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

    # 2.1 Group4超参数语义合并（在缺失值处理之前）
    df, rename_info = rename_group4_hyperparams(df, verbose=verbose)

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
    rows_to_drop = set()  # 记录需要删除的行索引
    cols_to_drop = set()  # 记录需要删除的列名

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
            # 性能指标：删除缺失行（而非填充-999）
            if missing_count > 0:
                if is_all_nan:
                    # 整列都是NaN：标记删除该列
                    cols_to_drop.add(col)
                    strategy = "perf_column_dropped"
                    # 不添加到 columns_filled（因为列被删除，未填充）
                else:
                    # 部分缺失：记录缺失行索引
                    missing_indices = df[df[col].isna()].index.tolist()
                    rows_to_drop.update(missing_indices)
                    strategy = f"perf_delete_rows({missing_count} rows)"
                    # 添加到 columns_filled（因为缺失行将被删除）
                    fill_info['columns_filled'].append(col)

                fill_info['strategy_used'][col] = strategy
            else:
                # 无缺失值
                strategy = "perf_no_missing"
                fill_info['strategy_used'][col] = strategy

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

    # 5.1 删除标记的列（全NaN性能指标列）
    if cols_to_drop:
        original_cols = len(df.columns)
        df.drop(columns=list(cols_to_drop), inplace=True)
        if verbose:
            print(f"  删除全NaN性能指标列: {len(cols_to_drop)}列 (原始{original_cols}列 → 剩余{len(df.columns)}列)")
        # 记录删除的列
        fill_info['cols_dropped'] = {
            'count': len(cols_to_drop),
            'columns': sorted(list(cols_to_drop))
        }
    else:
        fill_info['cols_dropped'] = {'count': 0, 'columns': []}

    # 5.2 删除标记的行（性能指标缺失行）
    if rows_to_drop:
        original_rows = len(df)
        df.drop(index=list(rows_to_drop), inplace=True)
        df.reset_index(drop=True, inplace=True)
        dropped_rows = original_rows - len(df)
        if verbose:
            print(f"  删除性能指标缺失行: {dropped_rows}行 (原始{original_rows}行 → 剩余{len(df)}行)")
        # 记录删除信息
        fill_info['rows_dropped'] = {
            'count': dropped_rows,
            'indices': sorted(list(rows_to_drop))
        }
    else:
        fill_info['rows_dropped'] = {'count': 0, 'indices': []}

    # 6. 删除全0列（v2.0新增：删除不适用于该组的超参数和模型列）
    df, removed_zero_cols = remove_all_zero_columns(df, verbose=verbose)

    # 7. 验证无缺失值
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

    # 添加v2.0新增信息：重命名和删除全0列
    fill_info['v2_changes'] = {
        'rename_info': rename_info,
        'removed_zero_columns': removed_zero_cols
    }

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
