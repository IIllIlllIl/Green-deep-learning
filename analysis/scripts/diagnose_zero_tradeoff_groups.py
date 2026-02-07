#!/usr/bin/env python3
"""
诊断group2和group4零权衡问题

深入调查为什么group2_vulberta和group4_bug_localization没有检测到任何权衡。

作者: Claude
日期: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 路径配置
ANALYSIS_DIR = Path('/home/green/energy_dl/nightly/analysis')
RESULTS_DIR = ANALYSIS_DIR / 'results/energy_research/data'

# 数据文件路径
GLOBAL_STD_ATE_DIR = RESULTS_DIR / 'global_std_dibs_ate'
WHITELIST_ATE_DIR = RESULTS_DIR / 'interaction/whitelist_with_ate'
GLOBAL_STD_DATA_DIR = ANALYSIS_DIR / 'data/energy_research/6groups_global_std'


class ZeroTradeoffDiagnoser:
    """零权衡组诊断器"""

    def __init__(self):
        self.groups = [
            'group1_examples',
            'group2_vulberta',
            'group3_person_reid',
            'group4_bug_localization',
            'group5_mrt_oast',
            'group6_resnet'
        ]
        self.ate_data = {}
        self.whitelist_ate_data = {}
        self.global_std_data = {}

    def load_data(self):
        """加载数据"""
        print("加载数据...")

        for group in self.groups:
            # 加载全局标准化ATE
            global_std_ate_path = GLOBAL_STD_ATE_DIR / f'{group}_dibs_global_std_ate.csv'
            if global_std_ate_path.exists():
                df = pd.read_csv(global_std_ate_path)
                self.ate_data[group] = df
                print(f"  {group}: 全局标准化ATE = {len(df)} 条")
            else:
                print(f"  {group}: 全局标准化ATE文件不存在")
                self.ate_data[group] = None

            # 加载whitelist ATE
            whitelist_ate_path = WHITELIST_ATE_DIR / f'{group}_causal_edges_whitelist_with_ate.csv'
            if whitelist_ate_path.exists():
                df = pd.read_csv(whitelist_ate_path)
                self.whitelist_ate_data[group] = df
                print(f"  {group}: Whitelist ATE = {len(df)} 条")
            else:
                print(f"  {group}: Whitelist ATE文件不存在")
                self.whitelist_ate_data[group] = None

            # 加载全局标准化数据
            global_std_path = GLOBAL_STD_DATA_DIR / f'{group}_global_std.csv'
            if global_std_path.exists():
                df = pd.read_csv(global_std_path)
                self.global_std_data[group] = df
                print(f"  {group}: 全局标准化数据 = {len(df)} 行, {len(df.columns)} 列")
            else:
                print(f"  {group}: 全局标准化数据文件不存在")
                self.global_std_data[group] = None

    def compare_edge_statistics(self):
        """比较因果图边数统计"""
        print("\n" + "="*60)
        print("比较因果图边数统计")
        print("="*60)

        print("\n因果边数对比:")
        print(f"{'Group':<25} {'全局ATE边数':<15} {'Whitelist边数':<15} {'有效ATE数':<15}")
        print("-" * 70)

        for group in self.groups:
            global_ate_edges = 0
            whitelist_edges = 0
            valid_ate_count = 0

            # 全局标准化ATE
            if self.ate_data[group] is not None:
                df = self.ate_data[group]
                global_ate_edges = len(df)
                # 统计有效ATE（非NaN）
                if 'ate' in df.columns:
                    valid_ate_count = df['ate'].notna().sum()
                elif 'ATE' in df.columns:
                    valid_ate_count = df['ATE'].notna().sum()

            # Whitelist ATE
            if self.whitelist_ate_data[group] is not None:
                whitelist_edges = len(self.whitelist_ate_data[group])

            # 标记零权衡组
            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""

            print(f"{group:<25} {global_ate_edges:<15} {whitelist_edges:<15} {valid_ate_count:<15} {marker}")

    def analyze_ate_quality(self):
        """分析ATE数据质量"""
        print("\n" + "="*60)
        print("ATE数据质量分析")
        print("="*60)

        print("\n各组的ATE统计特征:")
        print(f"{'Group':<25} {'ATE均值':<12} {'ATE中位数':<12} {'ATE非Null%':<12} {'零方差列':<10}")
        print("-" * 75)

        for group in self.groups:
            if self.ate_data[group] is None or len(self.ate_data[group]) == 0:
                print(f"{group:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
                continue

            df = self.ate_data[group]

            # 获取ATE列
            ate_col = None
            if 'ate' in df.columns:
                ate_col = 'ate'
            elif 'ATE' in df.columns:
                ate_col = 'ATE'

            if ate_col is None:
                print(f"{group:<25} {'无ATE列':<12} {'-':<12} {'-':<12} {'-':<10}")
                continue

            ate_values = df[ate_col]
            ate_mean = ate_values.mean()
            ate_median = ate_values.median()
            ate_nonnull_pct = ate_values.notna().sum() / len(ate_values) * 100

            # 检查零方差列
            zero_variance = 0
            if self.global_std_data[group] is not None:
                numeric_cols = self.global_std_data[group].select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if self.global_std_data[group][col].var() == 0:
                        zero_variance += 1

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {ate_mean:<12.2f} {ate_median:<12.2f} {ate_nonnull_pct:<12.1f}% {zero_variance:<10} {marker}")

    def investigate_specific_columns(self):
        """调查特定列（batch_size等）的零方差问题"""
        print("\n" + "="*60)
        print("调查零方差列")
        print("="*60)

        for group in ['group2_vulberta', 'group4_bug_localization']:
            print(f"\n{group}:")
            if self.global_std_data[group] is None:
                print("  数据文件不存在")
                continue

            df = self.global_std_data[group]
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            zero_var_cols = []
            for col in numeric_cols:
                if df[col].var() == 0:
                    zero_var_cols.append(col)

            if zero_var_cols:
                print(f"  发现 {len(zero_var_cols)} 个零方差列:")
                for col in zero_var_cols[:10]:  # 只显示前10个
                    unique_vals = df[col].unique()
                    print(f"    - {col}: 唯一值 = {unique_vals[:5]}")
                if len(zero_var_cols) > 10:
                    print(f"    ... 还有 {len(zero_var_cols) - 10} 个")
            else:
                print("  未发现零方差列")

    def analyze_sign_function_failure(self):
        """分析Sign函数可能失败的情况"""
        print("\n" + "="*60)
        print("分析Sign函数判定情况")
        print("="*60)

        # 读取权衡检测的详细结果
        tradeoff_detailed_path = ANALYSIS_DIR / 'results/energy_research/tradeoff_detection/tradeoff_detailed.csv'
        if not tradeoff_detailed_path.exists():
            print("权衡检测文件不存在，跳过此分析")
            return

        tradeoff_df = pd.read_csv(tradeoff_detailed_path)

        # 统计各组的Sign模式
        print("\nSign模式分布:")
        sign_cross = pd.crosstab(
            tradeoff_df['group_id'],
            tradeoff_df['sign1'] + tradeoff_df['sign2'],
            margins=True
        )
        print(sign_cross)

    def analyze_missing_values(self):
        """分析缺失值模式"""
        print("\n" + "="*60)
        print("分析缺失值模式")
        print("="*60)

        print(f"\n{'Group':<25} {'行数':<8} {'列数':<8} {'缺失值%':<10} {'能耗数据%':<12}")
        print("-" * 65)

        energy_cols = [
            'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts',
            'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
            'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent',
            'energy_gpu_util_max_percent'
        ]

        for group in self.groups:
            if self.global_std_data[group] is None:
                print(f"{group:<25} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<12}")
                continue

            df = self.global_std_data[group]
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isna().sum().sum()
            missing_pct = missing_cells / total_cells * 100

            # 计算能耗数据完整性
            energy_cells = len(df) * len(energy_cols)
            energy_missing = df[energy_cols].isna().sum().sum()
            energy_missing_pct = energy_missing / energy_cells * 100 if energy_cells > 0 else 0
            energy_completeness = 100 - energy_missing_pct

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {len(df):<8} {len(df.columns):<8} {missing_pct:<10.1f}% {energy_completeness:<12.1f}% {marker}")

    def generate_diagnostic_report(self):
        """生成诊断报告"""
        print("\n" + "="*60)
        print("诊断结论与建议")
        print("="*60)

        print("""
根据以上分析，group2和group4零权衡的可能原因：

1. **因果边数不足**
   - 如果因果边数显著少于其他组，权衡检测的基础就不足
   - 建议检查DiBS因果发现结果

2. **ATE计算问题**
   - 如果ATE非Null比例低，说明很多边没有计算出有效的ATE
   - 可能是数据质量问题或统计显著性不足

3. **数据质量问题**
   - 零方差列会导致无法计算ATE
   - 缺失值过多会影响因果发现

4. **Sign函数判定失败**
   - 如果ATE接近0或方向不一致，Sign函数可能无法判定为权衡
   - 需要检查ATE的符号和大小

建议的后续调查：
1. 检查group2和group4的DiBS因果图
2. 查看ATE计算失败的边
3. 对比数据质量指标
4. 考虑调整Sign函数的判定阈值
        """)

    def run_diagnosis(self):
        """运行完整诊断"""
        print("\n" + "="*60)
        print("开始诊断group2和group4零权衡问题")
        print("="*60)

        self.load_data()
        self.compare_edge_statistics()
        self.analyze_ate_quality()
        self.analyze_missing_values()
        self.investigate_specific_columns()
        self.analyze_sign_function_failure()
        self.generate_diagnostic_report()

        print("\n" + "="*60)
        print("诊断完成！")
        print("="*60)


def main():
    """主函数"""
    diagnoser = ZeroTradeoffDiagnoser()
    diagnoser.run_diagnosis()


if __name__ == '__main__':
    main()
