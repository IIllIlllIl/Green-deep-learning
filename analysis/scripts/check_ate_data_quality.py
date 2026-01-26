#!/usr/bin/env python3
"""
ATE数据质量检查脚本

检查白名单数据ATE计算的质量和完整性
生成详细的质量报告

使用方法:
    python check_ate_data_quality.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class ATEDataQualityChecker:
    """ATE数据质量检查器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.quality_reports = {}

    def load_ate_data(self, filepath: str) -> pd.DataFrame:
        """加载ATE数据"""
        df = pd.read_csv(filepath)
        return df

    def check_file_quality(self, filepath: str) -> Dict:
        """检查单个文件的数据质量"""
        filename = os.path.basename(filepath)
        group_name = filename.replace('_causal_edges_whitelist_with_ate.csv', '')

        if self.verbose:
            print(f"\n检查文件: {filename}")

        try:
            df = self.load_ate_data(filepath)

            # 基本统计
            total_edges = len(df)
            ate_computed = df['ate_computed'].sum() if 'ate_computed' in df.columns else 0
            ate_success_rate = ate_computed / total_edges * 100 if total_edges > 0 else 0

            # ATE统计
            ate_values = df['ate'].dropna() if 'ate' in df.columns else pd.Series()
            ate_mean = ate_values.mean() if len(ate_values) > 0 else np.nan
            ate_std = ate_values.std() if len(ate_values) > 0 else np.nan
            ate_min = ate_values.min() if len(ate_values) > 0 else np.nan
            ate_max = ate_values.max() if len(ate_values) > 0 else np.nan

            # 显著性统计
            significant_edges = df['is_significant'].sum() if 'is_significant' in df.columns else 0
            significance_rate = significant_edges / ate_computed * 100 if ate_computed > 0 else 0

            # 置信区间检查
            ci_valid = 0
            if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                for idx, row in df.iterrows():
                    if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']):
                        if row['ci_upper'] > row['ci_lower']:
                            ci_valid += 1

            ci_valid_rate = ci_valid / total_edges * 100 if total_edges > 0 else 0

            # 错误检查
            errors = df['error_message'].fillna('').str.strip()
            error_count = (errors != '').sum()
            error_rate = error_count / total_edges * 100 if total_edges > 0 else 0

            # 质量报告
            report = {
                'group': group_name,
                'filename': filename,
                'total_edges': total_edges,
                'ate_computed': int(ate_computed),
                'ate_success_rate': ate_success_rate,
                'significant_edges': int(significant_edges),
                'significance_rate': significance_rate,
                'ate_mean': ate_mean,
                'ate_std': ate_std,
                'ate_min': ate_min,
                'ate_max': ate_max,
                'ci_valid': ci_valid,
                'ci_valid_rate': ci_valid_rate,
                'error_count': error_count,
                'error_rate': error_rate,
                'file_size_kb': os.path.getsize(filepath) / 1024
            }

            if self.verbose:
                print(f"  总边数: {total_edges}")
                print(f"  ATE计算成功: {ate_computed} ({ate_success_rate:.1f}%)")
                print(f"  统计显著边: {significant_edges} ({significance_rate:.1f}%)")
                print(f"  ATE统计: 均值={ate_mean:.4f}, 标准差={ate_std:.4f}")
                print(f"  ATE范围: [{ate_min:.4f}, {ate_max:.4f}]")
                print(f"  置信区间有效: {ci_valid} ({ci_valid_rate:.1f}%)")
                print(f"  错误数: {error_count} ({error_rate:.1f}%)")

            return report

        except Exception as e:
            if self.verbose:
                print(f"  错误: {e}")
            return {
                'group': group_name,
                'filename': filename,
                'error': str(e),
                'total_edges': 0,
                'ate_computed': 0,
                'ate_success_rate': 0,
                'significant_edges': 0,
                'significance_rate': 0,
                'ate_mean': np.nan,
                'ate_std': np.nan,
                'ate_min': np.nan,
                'ate_max': np.nan,
                'ci_valid': 0,
                'ci_valid_rate': 0,
                'error_count': 1,
                'error_rate': 100,
                'file_size_kb': 0
            }

    def check_all_files(self, directory: str) -> Dict:
        """检查目录中的所有文件"""
        ate_files = list(Path(directory).glob('*_causal_edges_whitelist_with_ate.csv'))

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"检查目录: {directory}")
            print(f"找到 {len(ate_files)} 个ATE文件")
            print(f"{'='*80}")

        all_reports = {}

        for filepath in ate_files:
            report = self.check_file_quality(str(filepath))
            all_reports[report['group']] = report

        return all_reports

    def generate_summary_report(self, all_reports: Dict, output_dir: str):
        """生成汇总报告"""
        summary_df = pd.DataFrame(all_reports.values())

        if self.verbose:
            print(f"\n{'='*80}")
            print("ATE数据质量汇总报告")
            print(f"{'='*80}")

            # 按组显示
            for _, row in summary_df.iterrows():
                print(f"\n{row['group']}:")
                print(f"  文件: {row['filename']}")
                print(f"  总边数: {row['total_edges']}")
                print(f"  ATE计算成功: {row['ate_computed']} ({row['ate_success_rate']:.1f}%)")
                print(f"  统计显著边: {row['significant_edges']} ({row['significance_rate']:.1f}%)")
                if 'error' in row and pd.notna(row['error']):
                    print(f"  错误: {row['error']}")

            # 总计
            total_edges = summary_df['total_edges'].sum()
            total_ate_computed = summary_df['ate_computed'].sum()
            total_significant = summary_df['significant_edges'].sum()
            total_ci_valid = summary_df['ci_valid'].sum()
            total_errors = summary_df['error_count'].sum()

            overall_success_rate = total_ate_computed / total_edges * 100 if total_edges > 0 else 0
            overall_significance_rate = total_significant / total_ate_computed * 100 if total_ate_computed > 0 else 0
            overall_ci_valid_rate = total_ci_valid / total_edges * 100 if total_edges > 0 else 0
            overall_error_rate = total_errors / total_edges * 100 if total_edges > 0 else 0

            print(f"\n{'='*80}")
            print("总计:")
            print(f"  总边数: {total_edges}")
            print(f"  ATE计算成功: {total_ate_computed} ({overall_success_rate:.1f}%)")
            print(f"  统计显著边: {total_significant} ({overall_significance_rate:.1f}%)")
            print(f"  置信区间有效: {total_ci_valid} ({overall_ci_valid_rate:.1f}%)")
            print(f"  错误数: {total_errors} ({overall_error_rate:.1f}%)")
            print(f"{'='*80}")

            # ATE统计汇总
            valid_ate_rows = summary_df[summary_df['ate_mean'].notna()]
            if len(valid_ate_rows) > 0:
                ate_mean_overall = valid_ate_rows['ate_mean'].mean()
                ate_std_overall = valid_ate_rows['ate_std'].mean()
                ate_min_overall = valid_ate_rows['ate_min'].min()
                ate_max_overall = valid_ate_rows['ate_max'].max()

                print(f"\nATE统计汇总:")
                print(f"  平均ATE均值: {ate_mean_overall:.4f}")
                print(f"  平均ATE标准差: {ate_std_overall:.4f}")
                print(f"  总体ATE最小值: {ate_min_overall:.4f}")
                print(f"  总体ATE最大值: {ate_max_overall:.4f}")

        # 保存报告
        summary_path = os.path.join(output_dir, 'ate_data_quality_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        if self.verbose:
            print(f"\n✓ 质量报告已保存到: {summary_path}")

        # 生成详细报告
        detailed_path = os.path.join(output_dir, 'ate_data_quality_detailed.csv')
        detailed_df = self._generate_detailed_report(all_reports)
        detailed_df.to_csv(detailed_path, index=False)

        if self.verbose:
            print(f"✓ 详细报告已保存到: {detailed_path}")

        return summary_df

    def _generate_detailed_report(self, all_reports: Dict) -> pd.DataFrame:
        """生成详细报告"""
        records = []
        for group, report in all_reports.items():
            records.append({
                'group': report['group'],
                'metric': 'total_edges',
                'value': report['total_edges'],
                'percentage': 100.0
            })
            records.append({
                'group': report['group'],
                'metric': 'ate_computed',
                'value': report['ate_computed'],
                'percentage': report['ate_success_rate']
            })
            records.append({
                'group': report['group'],
                'metric': 'significant_edges',
                'value': report['significant_edges'],
                'percentage': report['significance_rate']
            })
            records.append({
                'group': report['group'],
                'metric': 'ci_valid',
                'value': report['ci_valid'],
                'percentage': report['ci_valid_rate']
            })
            records.append({
                'group': report['group'],
                'metric': 'errors',
                'value': report['error_count'],
                'percentage': report['error_rate']
            })

        return pd.DataFrame(records)

    def create_visualizations(self, all_reports: Dict, output_dir: str):
        """创建可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            summary_df = pd.DataFrame(all_reports.values())

            # 设置样式
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")

            # 1. ATE计算成功率条形图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ATE数据质量分析', fontsize=16, fontweight='bold')

            # 成功率
            ax1 = axes[0, 0]
            groups = summary_df['group']
            success_rates = summary_df['ate_success_rate']
            ax1.bar(groups, success_rates, color='skyblue', edgecolor='black')
            ax1.set_title('ATE计算成功率 (%)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('成功率 (%)')
            ax1.set_ylim(0, 110)
            ax1.tick_params(axis='x', rotation=45)
            for i, v in enumerate(success_rates):
                ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

            # 显著性率
            ax2 = axes[0, 1]
            significance_rates = summary_df['significance_rate']
            ax2.bar(groups, significance_rates, color='lightgreen', edgecolor='black')
            ax2.set_title('统计显著性率 (%)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('显著性率 (%)')
            ax2.set_ylim(0, 110)
            ax2.tick_params(axis='x', rotation=45)
            for i, v in enumerate(significance_rates):
                ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

            # ATE均值分布
            ax3 = axes[1, 0]
            ate_means = summary_df['ate_mean'].dropna()
            if len(ate_means) > 0:
                ax3.hist(ate_means, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
                ax3.set_title('ATE均值分布', fontsize=12, fontweight='bold')
                ax3.set_xlabel('ATE均值')
                ax3.set_ylabel('频数')
                mean_val = ate_means.mean()
                ax3.axvline(mean_val, color='red', linestyle='--', label=f'总体均值: {mean_val:.2f}')
                ax3.legend()

            # 错误率
            ax4 = axes[1, 1]
            error_rates = summary_df['error_rate']
            ax4.bar(groups, error_rates, color='salmon', edgecolor='black')
            ax4.set_title('错误率 (%)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('错误率 (%)')
            ax4.set_ylim(0, max(error_rates) * 1.2 if max(error_rates) > 0 else 10)
            ax4.tick_params(axis='x', rotation=45)
            for i, v in enumerate(error_rates):
                ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'ate_data_quality_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            if self.verbose:
                print(f"✓ 可视化图表已保存到: {viz_path}")

        except Exception as e:
            if self.verbose:
                print(f"⚠ 可视化创建失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='检查ATE数据质量')
    parser.add_argument('--input-dir', type=str,
                       default='/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/whitelist_with_ate',
                       help='输入目录（包含ATE文件）')
    parser.add_argument('--output-dir', type=str,
                       default='/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/whitelist_with_ate',
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细信息')

    args = parser.parse_args()

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")

    # 创建检查器
    checker = ATEDataQualityChecker(verbose=args.verbose)

    try:
        # 检查所有文件
        all_reports = checker.check_all_files(args.input_dir)

        # 生成汇总报告
        summary_df = checker.generate_summary_report(all_reports, args.output_dir)

        # 创建可视化
        checker.create_visualizations(all_reports, args.output_dir)

        print(f"\n{'='*80}")
        print("ATE数据质量检查完成!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()