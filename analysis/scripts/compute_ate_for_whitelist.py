#!/usr/bin/env python3
"""
白名单数据ATE计算脚本

基于已验收的CausalInferenceEngine为白名单数据计算ATE
为每个白名单文件添加ATE相关列，并保存到新目录

使用方法:
    python compute_ate_for_whitelist.py [--dry-run] [--group GROUP_NUM]

参数:
    --dry-run: 只测试不写入文件
    --group: 只处理指定组（1-6）
    --output-dir: 输出目录（默认: whitelist_with_ate）
    --t-strategy: T0/T1计算策略（默认: quantile）
    --ref-strategy: ref_df构建策略（默认: non_parallel）

输出列:
    - ate: 平均处理效应
    - ci_lower: 置信区间下界
    - ci_upper: 置信区间上界
    - is_significant: 是否统计显著
    - ate_computed: ATE是否计算成功
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.causal_inference import CausalInferenceEngine


class WhitelistATEComputer:
    """白名单数据ATE计算器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.engine = CausalInferenceEngine(verbose=verbose)
        self.results_summary = {}

    def load_whitelist_data(self, whitelist_path: str) -> pd.DataFrame:
        """加载白名单数据"""
        if not os.path.exists(whitelist_path):
            raise FileNotFoundError(f"白名单文件不存在: {whitelist_path}")

        df = pd.read_csv(whitelist_path)
        if self.verbose:
            print(f"  加载白名单: {whitelist_path}")
            print(f"    记录数: {len(df)}")
            print(f"    列: {list(df.columns)}")

        return df

    def load_group_data(self, group_data_path: str) -> pd.DataFrame:
        """加载组数据"""
        if not os.path.exists(group_data_path):
            raise FileNotFoundError(f"组数据文件不存在: {group_data_path}")

        df = pd.read_csv(group_data_path)
        if self.verbose:
            print(f"  加载组数据: {group_data_path}")
            print(f"    记录数: {len(df)}")
            print(f"    列数: {len(df.columns)}")

        return df

    def build_reference_df(self, data: pd.DataFrame, strategy: str = "non_parallel") -> pd.DataFrame:
        """构建参考数据集"""
        return self.engine.build_reference_df(data, strategy=strategy)

    def compute_T0_T1(self, data: pd.DataFrame, treatment: str, strategy: str = "quantile") -> Tuple[float, float]:
        """计算T0和T1"""
        return self.engine.compute_T0_T1(data, treatment, strategy=strategy)

    def estimate_ate_for_edge(self,
                             data: pd.DataFrame,
                             source: str,
                             target: str,
                             ref_df: Optional[pd.DataFrame] = None,
                             t_strategy: str = "quantile") -> Dict:
        """
        为单条边计算ATE

        返回:
            Dict with keys: ate, ci_lower, ci_upper, is_significant, ate_computed, error_message
        """
        result = {
            'ate': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'is_significant': False,
            'ate_computed': False,
            'error_message': ''
        }

        try:
            # 检查source和target是否在数据中
            # 处理交互项：如果source包含'_x_'，尝试分解
            original_source = source
            if '_x_' in source:
                # 交互项处理：尝试使用原始变量
                # 例如：hyperparam_batch_size_x_is_parallel -> 使用hyperparam_batch_size和is_parallel
                parts = source.split('_x_')
                if len(parts) == 2:
                    var1, var2 = parts
                    if var1 in data.columns and var2 in data.columns:
                        # 创建交互项
                        data_with_interaction = data.copy()
                        interaction_col = f"{var1}_interaction_{var2}"
                        data_with_interaction[interaction_col] = data_with_interaction[var1] * data_with_interaction[var2]
                        source = interaction_col
                        data = data_with_interaction

                        # 如果提供了ref_df，也需要在其中创建交互项
                        if ref_df is not None and var1 in ref_df.columns and var2 in ref_df.columns:
                            ref_df_with_interaction = ref_df.copy()
                            ref_df_with_interaction[interaction_col] = ref_df_with_interaction[var1] * ref_df_with_interaction[var2]
                            ref_df = ref_df_with_interaction

                        if self.verbose:
                            print(f"    创建交互项: {interaction_col}")
                    else:
                        raise ValueError(f"交互项 '{original_source}' 的组成部分不在数据中: {var1}, {var2}")
                else:
                    raise ValueError(f"无法解析交互项: {original_source}")

            if source not in data.columns:
                raise ValueError(f"source '{source}' 不在数据列中")
            if target not in data.columns:
                raise ValueError(f"target '{target}' 不在数据列中")

            # 检查数据类型
            source_dtype = data[source].dtype
            target_dtype = data[target].dtype

            # 处理布尔/分类变量
            if source_dtype == bool or (hasattr(source_dtype, 'categories') and len(data[source].unique()) <= 2):
                # 布尔或二分类变量：使用0/1编码
                data_encoded = data.copy()
                data_encoded[source] = data_encoded[source].astype(int)
                data = data_encoded
                if self.verbose:
                    print(f"    将布尔变量 '{source}' 编码为0/1")

            if target_dtype == bool or (hasattr(target_dtype, 'categories') and len(data[target].unique()) <= 2):
                # 布尔或二分类结果变量：使用0/1编码
                data_encoded = data.copy()
                data_encoded[target] = data_encoded[target].astype(int)
                data = data_encoded
                if self.verbose:
                    print(f"    将布尔变量 '{target}' 编码为0/1")

            # 识别混淆因素（简化版本：使用所有其他变量）
            # 在实际应用中，应该使用因果图来识别混淆因素
            # 这里使用简化方法：排除source和target的所有其他变量
            confounders = [col for col in data.columns
                          if col not in [source, target]
                          and pd.api.types.is_numeric_dtype(data[col])]

            # 如果混淆因素太少，添加一些关键变量
            if len(confounders) < 5:
                # 添加一些常见的混淆因素
                common_confounders = ['is_parallel', 'hyperparam_batch_size', 'hyperparam_learning_rate']
                for conf in common_confounders:
                    if conf in data.columns and conf not in confounders and conf not in [source, target]:
                        confounders.append(conf)

            if self.verbose and len(confounders) > 0:
                print(f"    使用 {len(confounders)} 个混淆因素")

            # 计算T0/T1（对于连续变量）
            if pd.api.types.is_numeric_dtype(data[source]) and len(data[source].unique()) > 2:
                try:
                    T0, T1 = self.compute_T0_T1(data, source, strategy=t_strategy)
                    use_T0_T1 = True
                except Exception as e:
                    if self.verbose:
                        print(f"    T0/T1计算失败: {e}，使用默认ATE")
                    T0, T1 = None, None
                    use_T0_T1 = False
            else:
                # 对于分类/布尔变量，不使用T0/T1
                T0, T1 = None, None
                use_T0_T1 = False

            # 估计ATE
            ate, ci = self.engine.estimate_ate(
                data=data,
                treatment=source,
                outcome=target,
                confounders=confounders,
                ref_df=ref_df,
                T0=T0 if use_T0_T1 else None,
                T1=T1 if use_T0_T1 else None,
                t_strategy=t_strategy if use_T0_T1 else None
            )

            # 更新结果
            result.update({
                'ate': float(ate),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'is_significant': not (ci[0] <= 0 <= ci[1]),
                'ate_computed': True,
                'error_message': ''
            })

        except Exception as e:
            result['error_message'] = str(e)
            if self.verbose:
                print(f"    ⚠ ATE计算失败: {e}")

        return result

    def process_whitelist(self,
                         whitelist_path: str,
                         group_data_path: str,
                         output_path: str,
                         dry_run: bool = False,
                         t_strategy: str = "quantile",
                         ref_strategy: str = "non_parallel") -> Dict:
        """
        处理单个白名单文件

        返回:
            处理统计信息
        """
        group_name = os.path.basename(whitelist_path).replace('_causal_edges_whitelist.csv', '')

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"处理组: {group_name}")
            print(f"白名单: {whitelist_path}")
            print(f"组数据: {group_data_path}")
            print(f"输出: {output_path}")
            print(f"策略: T0/T1={t_strategy}, ref_df={ref_strategy}")
            print(f"Dry run: {dry_run}")
            print(f"{'='*60}")

        # 加载数据
        whitelist_df = self.load_whitelist_data(whitelist_path)
        group_data = self.load_group_data(group_data_path)

        # 构建ref_df
        ref_df = self.build_reference_df(group_data, strategy=ref_strategy)

        # 初始化结果列
        n_edges = len(whitelist_df)
        whitelist_df['ate'] = np.nan
        whitelist_df['ci_lower'] = np.nan
        whitelist_df['ci_upper'] = np.nan
        whitelist_df['is_significant'] = False
        whitelist_df['ate_computed'] = False
        whitelist_df['error_message'] = ''

        # 为每条边计算ATE
        success_count = 0
        for idx, row in whitelist_df.iterrows():
            source = row['source']
            target = row['target']

            if self.verbose:
                print(f"\n  [{idx+1}/{n_edges}] 计算ATE: {source} → {target}")

            # 计算ATE
            ate_result = self.estimate_ate_for_edge(
                data=group_data,
                source=source,
                target=target,
                ref_df=ref_df,
                t_strategy=t_strategy
            )

            # 更新结果
            for key in ['ate', 'ci_lower', 'ci_upper', 'is_significant', 'ate_computed', 'error_message']:
                whitelist_df.at[idx, key] = ate_result[key]

            if ate_result['ate_computed']:
                success_count += 1
                if self.verbose:
                    print(f"    ✓ ATE: {ate_result['ate']:.4f}")
                    print(f"      95% CI: [{ate_result['ci_lower']:.4f}, {ate_result['ci_upper']:.4f}]")
                    print(f"      显著: {ate_result['is_significant']}")

        # 保存结果（如果不是dry run）
        if not dry_run:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            whitelist_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"\n✓ 结果已保存到: {output_path}")

        # 统计信息
        stats = {
            'group': group_name,
            'total_edges': n_edges,
            'successful_ate': success_count,
            'success_rate': success_count / n_edges * 100 if n_edges > 0 else 0,
            'significant_edges': int(whitelist_df['is_significant'].sum()),
            'output_path': output_path if not dry_run else 'N/A (dry run)'
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"处理完成: {group_name}")
            print(f"  总边数: {stats['total_edges']}")
            print(f"  ATE计算成功: {stats['successful_ate']} ({stats['success_rate']:.1f}%)")
            print(f"  统计显著边: {stats['significant_edges']}")
            print(f"{'='*60}")

        return stats

    def process_all_groups(self,
                          whitelist_dir: str,
                          group_data_dir: str,
                          output_dir: str,
                          dry_run: bool = False,
                          t_strategy: str = "quantile",
                          ref_strategy: str = "non_parallel",
                          specific_group: Optional[int] = None) -> Dict:
        """
        处理所有组

        返回:
            所有组的统计信息
        """
        # 查找所有白名单文件
        whitelist_files = []
        for i in range(1, 7):
            pattern = f"group{i}_*_causal_edges_whitelist.csv"
            matches = list(Path(whitelist_dir).glob(pattern))
            if matches:
                whitelist_files.append(matches[0])

        if not whitelist_files:
            raise FileNotFoundError(f"在 {whitelist_dir} 中未找到白名单文件")

        if self.verbose:
            print(f"找到 {len(whitelist_files)} 个白名单文件")

        # 如果指定了特定组，只处理该组
        if specific_group is not None:
            whitelist_files = [f for f in whitelist_files if f"group{specific_group}_" in str(f)]
            if not whitelist_files:
                raise FileNotFoundError(f"未找到组 {specific_group} 的白名单文件")

        all_stats = {}

        # 处理每个组
        for whitelist_path in whitelist_files:
            # 提取组名和编号
            filename = os.path.basename(whitelist_path)
            group_match = filename.split('_')[0]  # group1, group2, etc.
            group_num = int(group_match.replace('group', ''))

            # 构建组数据路径
            group_data_pattern = f"group{group_num}_*.csv"
            group_data_matches = list(Path(group_data_dir).glob(group_data_pattern))
            if not group_data_matches:
                warnings.warn(f"未找到组 {group_num} 的数据文件，跳过")
                continue

            group_data_path = group_data_matches[0]

            # 构建输出路径
            output_filename = filename.replace('.csv', '_with_ate.csv')
            output_path = os.path.join(output_dir, output_filename)

            # 处理该组
            try:
                stats = self.process_whitelist(
                    whitelist_path=str(whitelist_path),
                    group_data_path=str(group_data_path),
                    output_path=output_path,
                    dry_run=dry_run,
                    t_strategy=t_strategy,
                    ref_strategy=ref_strategy
                )
                all_stats[group_match] = stats
            except Exception as e:
                warnings.warn(f"处理组 {group_match} 失败: {e}")
                all_stats[group_match] = {
                    'group': group_match,
                    'error': str(e),
                    'total_edges': 0,
                    'successful_ate': 0,
                    'success_rate': 0,
                    'significant_edges': 0
                }

        return all_stats

    def generate_summary_report(self, all_stats: Dict, output_path: Optional[str] = None, dry_run: bool = False):
        """生成汇总报告"""
        summary_df = pd.DataFrame(all_stats.values())

        if self.verbose:
            print(f"\n{'='*80}")
            print("ATE计算汇总报告")
            print(f"{'='*80}")

            for _, row in summary_df.iterrows():
                print(f"\n{row['group']}:")
                print(f"  总边数: {row['total_edges']}")
                print(f"  ATE计算成功: {row['successful_ate']} ({row.get('success_rate', 0):.1f}%)")
                print(f"  统计显著边: {row['significant_edges']}")
                if 'error' in row and pd.notna(row['error']):
                    print(f"  错误: {row['error']}")

            # 总计
            total_edges = summary_df['total_edges'].sum()
            total_success = summary_df['successful_ate'].sum()
            total_significant = summary_df['significant_edges'].sum()

            print(f"\n{'='*80}")
            print("总计:")
            print(f"  总边数: {total_edges}")
            print(f"  ATE计算成功: {total_success} ({total_success/total_edges*100:.1f}% of total)")
            print(f"  统计显著边: {total_significant} ({total_significant/total_success*100:.1f}% of successful)")
            print(f"{'='*80}")

        # 保存报告
        if output_path and not dry_run:
            summary_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"\n✓ 汇总报告已保存到: {output_path}")

        return summary_df


def main():
    parser = argparse.ArgumentParser(description='为白名单数据计算ATE')
    parser.add_argument('--dry-run', action='store_true', help='只测试不写入文件')
    parser.add_argument('--group', type=int, choices=range(1, 7), help='只处理指定组（1-6）')
    parser.add_argument('--output-dir', type=str, default='whitelist_with_ate',
                       help='输出目录（默认: whitelist_with_ate）')
    parser.add_argument('--t-strategy', type=str, default='quantile',
                       choices=['quantile', 'min_max', 'mean_std'],
                       help='T0/T1计算策略（默认: quantile）')
    parser.add_argument('--ref-strategy', type=str, default='non_parallel',
                       choices=['non_parallel', 'mean', 'group_mean'],
                       help='ref_df构建策略（默认: non_parallel）')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细信息（默认: True）')

    args = parser.parse_args()

    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    whitelist_dir = os.path.join(base_dir, 'results', 'energy_research', 'data', 'interaction', 'whitelist')
    group_data_dir = os.path.join(base_dir, 'data', 'energy_research', '6groups_final')
    output_dir = os.path.join(base_dir, 'results', 'energy_research', 'data', 'interaction', args.output_dir)

    print(f"白名单目录: {whitelist_dir}")
    print(f"组数据目录: {group_data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"策略: T0/T1={args.t_strategy}, ref_df={args.ref_strategy}")
    print(f"Dry run: {args.dry_run}")
    print(f"指定组: {args.group if args.group else '全部'}")

    # 创建ATE计算器
    computer = WhitelistATEComputer(verbose=args.verbose)

    try:
        # 处理所有组
        all_stats = computer.process_all_groups(
            whitelist_dir=whitelist_dir,
            group_data_dir=group_data_dir,
            output_dir=output_dir,
            dry_run=args.dry_run,
            t_strategy=args.t_strategy,
            ref_strategy=args.ref_strategy,
            specific_group=args.group
        )

        # 生成汇总报告
        summary_path = os.path.join(output_dir, 'ate_computation_summary.csv') if not args.dry_run else None
        computer.generate_summary_report(all_stats, output_path=summary_path, dry_run=args.dry_run)

        print(f"\n{'='*80}")
        print("ATE计算任务完成!")
        if args.dry_run:
            print("注意: 这是dry run，未写入任何文件")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()