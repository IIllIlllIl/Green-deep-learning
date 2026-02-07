#!/usr/bin/env python3
"""
深入分析ATE数据质量问题

诊断为什么group2和group4没有检测到权衡的根本原因。

作者: Claude
日期: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 路径配置
ANALYSIS_DIR = Path('/home/green/energy_dl/nightly/analysis')
WHITELIST_ATE_DIR = ANALYSIS_DIR / 'results/energy_research/data/interaction/whitelist_with_ate'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ATEQualityAnalyzer:
    """ATE数据质量分析器"""

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

    def load_data(self):
        """加载数据"""
        print("加载whitelist ATE数据...")

        for group in self.groups:
            file_path = WHITELIST_ATE_DIR / f'{group}_causal_edges_whitelist_with_ate.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                self.ate_data[group] = df
                print(f"  {group}: {len(df)} 条边")
            else:
                self.ate_data[group] = None

    def analyze_ate_completeness(self):
        """分析ATE完整性"""
        print("\n" + "="*60)
        print("ATE完整性分析")
        print("="*60)

        print(f"\n{'Group':<25} {'总边数':<8} {'ATE计算成功':<12} {'ATE缺失':<10} {'ATE完整率':<10}")
        print("-" * 70)

        for group in self.groups:
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            total_edges = len(df)
            ate_computed = df['ate_computed'].sum()
            ate_missing = total_edges - ate_computed
            ate_completeness = ate_computed / total_edges * 100 if total_edges > 0 else 0

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {total_edges:<8} {ate_computed:<12} {ate_missing:<10} {ate_completeness:<10.1f}% {marker}")

    def analyze_significance_filtering(self):
        """分析统计显著性过滤"""
        print("\n" + "="*60)
        print("统计显著性分析")
        print("="*60)

        print(f"\n{'Group':<25} {'总边数':<8} {'显著边数':<10} {'显著边%':<10} {'显著且ATE完整':<15}")
        print("-" * 75)

        for group in self.groups:
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            total_edges = len(df)
            significant_edges = df['is_significant'].sum()
            significant_pct = significant_edges / total_edges * 100 if total_edges > 0 else 0

            # 同时满足显著和ATE完整的边数
            significant_and_computed = df[(df['is_significant'] == True) &
                                          (df['ate_computed'] == True)].shape[0]

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {total_edges:<8} {significant_edges:<10} {significant_pct:<10.1f}% {significant_and_computed:<15} {marker}")

    def analyze_target_category_distribution(self):
        """分析目标类别分布（Q1, Q2, Q3, other）"""
        print("\n" + "="*60)
        print("目标类别分布分析")
        print("="*60)

        print("\n各组的question_relevance分布:")
        print(f"{'Group':<25} {'Q1':<6} {'Q2':<6} {'Q3':<6} {'other':<6} {'Q1+Q3':<8}")
        print("-" * 60)

        for group in self.groups:
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            relevance_counts = df['question_relevance'].value_counts()

            q1 = relevance_counts.get('Q1', 0)
            q2 = relevance_counts.get('Q2', 0)
            q3 = relevance_counts.get('Q3', 0)
            other = relevance_counts.get('other', 0)
            q1_q3 = q1 + q3

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {q1:<6} {q2:<6} {q3:<6} {other:<6} {q1_q3:<8} {marker}")

    def analyze_edge_type_distribution(self):
        """分析边类型分布"""
        print("\n" + "="*60)
        print("边类型分布分析")
        print("="*60)

        print("\n各组的edge_type分布:")
        print(f"{'Group':<25} {'main_effect':<12} {'moderation':<12} {'mode_effect':<12} {'mediator':<12}")
        print("-" * 75)

        for group in self.groups:
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            edge_type_counts = df['edge_type'].value_counts()

            main_effect = edge_type_counts.get('main_effect', 0)
            moderation = edge_type_counts.get('moderation', 0)
            mode_effect = edge_type_counts.get('mode_effect', 0)
            mediator = edge_type_counts.get('mediator', 0)

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {main_effect:<12} {moderation:<12} {mode_effect:<12} {mediator:<12} {marker}")

    def analyze_tradeoff_detection_candidates(self):
        """分析权衡检测的候选边"""
        print("\n" + "="*60)
        print("权衡检测候选边分析")
        print("="*60)

        print("\n算法1需要的条件:")
        print("  1. is_significant = True")
        print("  2. ate_computed = True")
        print("  3. source_category != target_category（不同类别）")
        print("  4. Sign函数判定为权衡（+- 或 -+）")

        print(f"\n{'Group':<25} {'满足1+2+3':<12} {'占总边数%':<10}")
        print("-" * 45)

        for group in self.groups:
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            total_edges = len(df)

            # 筛选满足条件的边
            candidates = df[
                (df['is_significant'] == True) &
                (df['ate_computed'] == True) &
                (df['source_category'] != df['target_category'])
            ]

            candidate_pct = len(candidates) / total_edges * 100 if total_edges > 0 else 0

            marker = " <-- 零权衡" if group in ['group2_vulberta', 'group4_bug_localization'] else ""
            print(f"{group:<25} {len(candidates):<12} {candidate_pct:<10.1f}% {marker}")

            # 如果是零权衡组，详细分析
            if group in ['group2_vulberta', 'group4_bug_localization'] and len(candidates) > 0:
                print(f"\n  {group} 的候选边详情:")
                print(f"  {'source':<30} -> {'target':<30} {'ate':<12} {'source_cat':<12} {'target_cat':<12}")
                print("  " + "-" * 110)
                for _, row in candidates.head(10).iterrows():
                    print(f"  {row['source']:<30} -> {row['target']:<30} {row['ate']:<12.4f} {row['source_category']:<12} {row['target_category']:<12}")
                if len(candidates) > 10:
                    print(f"  ... 还有 {len(candidates) - 10} 条")

    def analyze_ate_sign_distribution(self):
        """分析ATE符号分布"""
        print("\n" + "="*60)
        print("ATE符号分布分析")
        print("="*60)

        for group in ['group2_vulberta', 'group4_bug_localization']:
            print(f"\n{group}:")
            if self.ate_data[group] is None:
                continue

            df = self.ate_data[group]
            valid_ate_df = df[df['ate_computed'] == True]

            if len(valid_ate_df) == 0:
                print("  没有计算出任何ATE！")
                continue

            ate_positive = (valid_ate_df['ate'] > 0).sum()
            ate_negative = (valid_ate_df['ate'] < 0).sum()
            ate_zero = (valid_ate_df['ate'] == 0).sum()

            print(f"  ATE>0: {ate_positive}, ATE<0: {ate_negative}, ATE=0: {ate_zero}")

    def generate_diagnostic_summary(self):
        """生成诊断总结"""
        print("\n" + "="*60)
        print("诊断总结")
        print("="*60)

        print("""
根据以上分析，group2和group4零权衡的根本原因：

1. **ATE完整性不足**
   - 如果很多边的ATE没有被计算出来（ate_computed=False），
     算法1就无法使用这些边进行权衡检测

2. **统计显著性不足**
   - 如果很多边不显著（is_significant=False），
     这些边会被过滤掉

3. **候选边数量不足**
   - 同时满足显著、ATE完整、不同类别的边可能很少
   - 没有足够的候选边就无法检测到权衡

4. **Sign函数判定失败**
   - 即使有候选边，如果ATE方向不符合权衡的定义
     （一个改善，一个恶化），Sign函数也不会判定为权衡

建议：
1. 检查为什么某些边的ATE没有被计算出来
2. 检查置信区间（ci_lower, ci_upper）是否有效
3. 考虑放宽统计显著性阈值
4. 检查数据质量是否影响ATE计算
        """)

    def run_analysis(self):
        """运行完整分析"""
        print("\n" + "="*60)
        print("开始ATE数据质量深入分析")
        print("="*60)

        self.load_data()
        self.analyze_ate_completeness()
        self.analyze_significance_filtering()
        self.analyze_target_category_distribution()
        self.analyze_edge_type_distribution()
        self.analyze_tradeoff_detection_candidates()
        self.analyze_ate_sign_distribution()
        self.generate_diagnostic_summary()

        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)


def main():
    """主函数"""
    analyzer = ATEQualityAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
