#!/usr/bin/env python3
"""
Group2数据问题诊断脚本

深入诊断group2_vulberta没有检测到权衡的根本原因。

作者: Claude
日期: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 路径配置
ANALYSIS_DIR = Path('/home/green/energy_dl/nightly/analysis')
GLOBAL_STD_DATA_DIR = ANALYSIS_DIR / 'data/energy_research/6groups_global_std'
WHITELIST_ATE_DIR = ANALYSIS_DIR / 'results/energy_research/data/interaction/whitelist_with_ate'


class Group2Diagnoser:
    """Group2数据诊断器"""

    def __init__(self):
        self.group2_data = None
        self.group2_ate = None
        self.other_groups_data = {}
        self.other_groups_ate = {}

    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载数据")
        print("="*60)

        # 加载group2全局标准化数据
        group2_path = GLOBAL_STD_DATA_DIR / 'group2_vulberta_global_std.csv'
        self.group2_data = pd.read_csv(group2_path)
        print(f"\ngroup2全局标准化数据: {len(self.group2_data)} 行, {len(self.group2_data.columns)} 列")

        # 加载group2 whitelist ATE
        group2_ate_path = WHITELIST_ATE_DIR / 'group2_vulberta_causal_edges_whitelist_with_ate.csv'
        self.group2_ate = pd.read_csv(group2_ate_path)
        print(f"group2 whitelist ATE: {len(self.group2_ate)} 条边")

        # 加载其他组数据用于对比
        other_groups = ['group1_examples', 'group3_person_reid', 'group5_mrt_oast']
        for group in other_groups:
            data_path = GLOBAL_STD_DATA_DIR / f'{group}_global_std.csv'
            if data_path.exists():
                self.other_groups_data[group] = pd.read_csv(data_path)
                print(f"{group}全局标准化数据: {len(self.other_groups_data[group])} 行")

            ate_path = WHITELIST_ATE_DIR / f'{group}_causal_edges_whitelist_with_ate.csv'
            if ate_path.exists():
                self.other_groups_ate[group] = pd.read_csv(ate_path)
                print(f"{group} whitelist ATE: {len(self.other_groups_ate[group])} 条边")

    def check_batch_size_variance(self):
        """检查batch_size方差"""
        print("\n" + "="*60)
        print("1. 检查batch_size数据分布和方差")
        print("="*60)

        # 检查全局标准化数据中的batch_size
        if 'batch_size' in self.group2_data.columns:
            batch_size_vals = self.group2_data['batch_size']
            print(f"\ngroup2 batch_size:")
            print(f"  唯一值数量: {batch_size_vals.nunique()}")
            print(f"  唯一值: {sorted(batch_size_vals.unique())}")
            print(f"  方差: {batch_size_vals.var():.6f}")
            print(f"  标准差: {batch_size_vals.std():.6f}")
            print(f"  均值: {batch_size_vals.mean():.6f}")

            # 检查是否为零方差
            if batch_size_vals.var() == 0:
                print("  ⚠️  零方差！这会导致因果边无法被检测")
            else:
                print("  ✅ 有方差，应该能被检测到")
        else:
            print("\n⚠️  batch_size列不存在！")

        # 与其他组对比
        print("\n与其他组对比:")
        print(f"{'Group':<20} {'唯一值数':<10} {'方差':<15} {'标准差':<15}")
        print("-" * 60)

        for group, df in self.other_groups_data.items():
            if 'batch_size' in df.columns:
                bs = df['batch_size']
                print(f"{group:<20} {bs.nunique():<10} {bs.var():<15.6f} {bs.std():<15.6f}")

    def check_all_zero_variance_columns(self):
        """检查所有零方差列"""
        print("\n" + "="*60)
        print("2. 检查所有零方差列")
        print("="*60)

        numeric_cols = self.group2_data.select_dtypes(include=[np.number]).columns
        zero_var_cols = []

        for col in numeric_cols:
            if self.group2_data[col].var() == 0:
                zero_var_cols.append(col)

        if zero_var_cols:
            print(f"\n发现 {len(zero_var_cols)} 个零方差列:")
            for col in zero_var_cols:
                unique_vals = self.group2_data[col].unique()
                print(f"  - {col}: 唯一值 = {unique_vals[:5]}")
        else:
            print("\n✅ 未发现零方差列")

        return zero_var_cols

    def analyze_data_quality(self):
        """分析数据质量"""
        print("\n" + "="*60)
        print("3. 分析数据质量")
        print("="*60)

        # 样本数对比
        print("\n样本数对比:")
        print(f"{'Group':<20} {'样本数':<10}")
        print("-" * 30)
        print(f"{'group2_vulberta':<20} {len(self.group2_data):<10}")
        for group, df in self.other_groups_data.items():
            print(f"{group:<20} {len(df):<10}")

        # 缺失值检查
        print("\n缺失值检查:")
        print(f"{'Group':<20} {'缺失值数':<10} {'缺失比例':<10}")
        print("-" * 40)
        missing_count = self.group2_data.isna().sum().sum()
        total_cells = len(self.group2_data) * len(self.group2_data.columns)
        missing_pct = missing_count / total_cells * 100
        print(f"{'group2_vulberta':<20} {missing_count:<10} {missing_pct:<10.2f}%")

        # 因果图边数对比
        print("\n因果图边数对比:")
        print(f"{'Group':<20} {'whitelist边数':<15}")
        print("-" * 35)
        print(f"{'group2_vulberta':<20} {len(self.group2_ate):<15}")
        for group, df in self.other_groups_ate.items():
            print(f"{group:<20} {len(df):<15}")

    def analyze_tradeoff_detection_candidates(self):
        """分析权衡检测候选边"""
        print("\n" + "="*60)
        print("4. 分析权衡检测候选边")
        print("="*60)

        # 筛选候选边
        candidates = self.group2_ate[
            (self.group2_ate['is_significant'] == True) &
            (self.group2_ate['ate_computed'] == True) &
            (self.group2_ate['source_category'] != self.group2_ate['target_category'])
        ]

        print(f"\ngroup2候选边数: {len(candidates)} / {len(self.group2_ate)} ({len(candidates)/len(self.group2_ate)*100:.1f}%)")

        # 与其他组对比
        print("\n候选边对比:")
        print(f"{'Group':<20} {'候选边数':<12} {'总边数':<10} {'候选边比例':<12}")
        print("-" * 55)
        print(f"{'group2_vulberta':<20} {len(candidates):<12} {len(self.group2_ate):<10} {len(candidates)/len(self.group2_ate)*100:.12}")

        for group, df in self.other_groups_ate.items():
            cands = df[
                (df['is_significant'] == True) &
                (df['ate_computed'] == True) &
                (df['source_category'] != df['target_category'])
            ]
            print(f"{group:<20} {len(cands):<12} {len(df):<10} {len(cands)/len(df)*100:.1f}%")

        # 详细查看group2的候选边
        if len(candidates) > 0:
            print(f"\ngroup2的 {len(candidates)} 条候选边详情:")
            print(f"{'source':<35} -> {'target':<35} {'ate':<12} {'source_cat':<12} {'target_cat':<12}")
            print("-" * 115)
            for _, row in candidates.iterrows():
                print(f"{row['source']:<35} -> {row['target']:<35} {row['ate']:<12.4f} {row['source_category']:<12} {row['target_category']:<12}")

    def check_sign_function_candidates(self):
        """检查哪些候选边可能被Sign函数判定为权衡"""
        print("\n" + "="*60)
        print("5. 检查Sign函数判定可能性")
        print("="*60)

        print("\n根据CTF论文算法1，权衡需要满足:")
        print("  1. 干预对metric1的效应: sign1")
        print("  2. 干预对metric2的效应: sign2")
        print("  3. 权衡条件: sign1=+, sign2=- OR sign1=-, sign2=+")

        # 分析group2的候选边组合
        candidates = self.group2_ate[
            (self.group2_ate['is_significant'] == True) &
            (self.group2_ate['ate_computed'] == True) &
            (self.group2_ate['source_category'] != self.group2_ate['target_category'])
        ]

        print(f"\ngroup2有 {len(candidates)} 条候选边，可能的干预-指标对数量:")

        # 统计可能的干预
        interventions = candidates['source'].unique()
        metrics = candidates[~candidates['source'].isin(candidates['target'].unique())]['target'].unique()

        print(f"  可能的干预数: {len(interventions)}")
        print(f"  可能的指标数: {len(metrics)}")
        print(f"  理论上的干预-指标对数量: {len(interventions)} * {len(metrics)} = {len(interventions) * len(metrics)}")

        # 实际检测到的权衡
        print(f"\n实际上检测到的权衡数: 0")

        print("\n可能的原因:")
        print("  1. 候选边太少，无法形成有效的干预-指标对")
        print("  2. ATE的符号分布不符合权衡条件（都是+或都是-）")
        print("  3. 需要查看实际的ATE符号分布")

    def analyze_ate_sign_distribution(self):
        """分析ATE符号分布"""
        print("\n" + "="*60)
        print("6. 分析ATE符号分布")
        print("="*60)

        valid_ate = self.group2_ate[self.group2_ate['ate_computed'] == True]

        ate_positive = (valid_ate['ate'] > 0).sum()
        ate_negative = (valid_ate['ate'] < 0).sum()
        ate_zero = (valid_ate['ate'] == 0).sum()

        print(f"\ngroup2 ATE符号分布:")
        print(f"  ATE > 0: {ate_positive} ({ate_positive/len(valid_ate)*100:.1f}%)")
        print(f"  ATE < 0: {ate_negative} ({ate_negative/len(valid_ate)*100:.1f}%)")
        print(f"  ATE = 0: {ate_zero} ({ate_zero/len(valid_ate)*100:.1f}%)")

        # 与其他组对比
        print("\n与其他组对比:")
        print(f"{'Group':<20} {'ATE>0':<8} {'ATE<0':<8} {'ATE>0%':<10} {'ATE<0%':<10}")
        print("-" * 55)

        pos_pct = ate_positive / len(valid_ate) * 100
        neg_pct = ate_negative / len(valid_ate) * 100
        print(f"{'group2_vulberta':<20} {ate_positive:<8} {ate_negative:<8} {pos_pct:<10.1f}% {neg_pct:<10.1f}%")

        for group, df in self.other_groups_ate.items():
            v = df[df['ate_computed'] == True]
            pos = (v['ate'] > 0).sum()
            neg = (v['ate'] < 0).sum()
            pos_pct = pos / len(v) * 100
            neg_pct = neg / len(v) * 100
            print(f"{group:<20} {pos:<8} {neg:<8} {pos_pct:<10.1f}% {neg_pct:<10.1f}%")

    def generate_diagnostic_report(self):
        """生成诊断报告"""
        print("\n" + "="*60)
        print("诊断结论")
        print("="*60)

        report = {
            "group": "group2_vulberta",
            "sample_size": len(self.group2_data),
            "whitelist_edges": len(self.group2_ate),
            "candidate_edges": len(self.group2_ate[
                (self.group2_ate['is_significant'] == True) &
                (self.group2_ate['ate_computed'] == True) &
                (self.group2_ate['source_category'] != self.group2_ate['target_category'])
            ]),
            "tradeoffs_detected": 0,
            "key_issues": [],
            "recommendations": []
        }

        # 检查batch_size
        if 'batch_size' in self.group2_data.columns:
            if self.group2_data['batch_size'].var() == 0:
                report["key_issues"].append("batch_size零方差")
            else:
                report["key_issues"].append("batch_size有方差，不是问题")

        # 分析候选边
        candidates = self.group2_ate[
            (self.group2_ate['is_significant'] == True) &
            (self.group2_ate['ate_computed'] == True) &
            (self.group2_ate['source_category'] != self.group2_ate['target_category'])
        ]

        if len(candidates) < 10:
            report["key_issues"].append(f"候选边太少({len(candidates)}条)，无法形成足够的干预-指标对")

        report["recommendations"] = [
            "增加样本量以提高统计功效",
            "检查数据质量问题",
            "考虑调整统计显著性阈值",
            "零权衡可能是真实的（该任务确实没有权衡）"
        ]

        print("\n诊断报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        return report

    def run_diagnosis(self):
        """运行完整诊断"""
        print("\n" + "="*60)
        print("开始Group2数据问题诊断")
        print("="*60)

        self.load_data()
        self.check_batch_size_variance()
        zero_var_cols = self.check_all_zero_variance_columns()
        self.analyze_data_quality()
        self.analyze_tradeoff_detection_candidates()
        self.check_sign_function_candidates()
        self.analyze_ate_sign_distribution()
        report = self.generate_diagnostic_report()

        print("\n" + "="*60)
        print("诊断完成")
        print("="*60)

        return report


def main():
    """主函数"""
    diagnoser = Group2Diagnoser()
    diagnoser.run_diagnosis()


if __name__ == '__main__':
    main()
