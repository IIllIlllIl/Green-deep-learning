#!/usr/bin/env python3
"""
权衡结果深入分析脚本

分析算法1检测到的37个权衡关系，包括：
1. 权衡类型分布（能耗vs能耗、能耗vs性能、性能vs性能）
2. ATE幅度分布
3. 干预类型分布
4. 任务组对比
5. 可视化生成

作者: Claude
日期: 2026-02-01
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
ANALYSIS_DIR = Path('/home/green/energy_dl/nightly/analysis')
RESULTS_DIR = ANALYSIS_DIR / 'results/energy_research/tradeoff_detection'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# 数据文件路径
DETAILED_CSV = RESULTS_DIR / 'tradeoff_detailed.csv'
ALL_TRADEOFFS_JSON = RESULTS_DIR / 'all_tradeoffs.json'


class TradeoffAnalyzer:
    """权衡结果分析器"""

    def __init__(self):
        self.df_detailed = None
        self.all_tradeoffs = None
        self.metrics_energy = None
        self.metrics_perf = None
        self.metrics_hyperparam = None

    def load_data(self):
        """加载权衡结果数据"""
        print("加载权衡结果数据...")

        # 加载详细权衡表
        self.df_detailed = pd.read_csv(DETAILED_CSV)
        print(f"  - 权衡记录数: {len(self.df_detailed)}")

        # 加载完整JSON
        with open(ALL_TRADEOFFS_JSON, 'r') as f:
            self.all_tradeoffs = json.load(f)

        # 定义指标类型
        self.metrics_energy = [
            'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts',
            'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
            'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent',
            'energy_gpu_util_max_percent'
        ]

        self.metrics_perf = [
            'perf_test_accuracy', 'perf_map', 'perf_accuracy', 'perf_precision'
        ]

        self.metrics_hyperparam = [
            'model_siamese', 'model_hrnet18', 'model_pcb',
            'batch_size', 'learning_rate', 'weight_decay',
            'is_parallel'
        ]

    def classify_metric_type(self, metric_name):
        """分类指标类型"""
        if metric_name in self.metrics_energy:
            return 'energy'
        elif metric_name in self.metrics_perf:
            return 'performance'
        else:
            return 'hyperparameter'

    def classify_tradeoff_type(self, row):
        """分类权衡类型"""
        m1_type = self.classify_metric_type(row['metric1'])
        m2_type = self.classify_metric_type(row['metric2'])

        if m1_type == 'energy' and m2_type == 'energy':
            return 'Energy vs Energy'
        elif m1_type == 'performance' and m2_type == 'performance':
            return 'Performance vs Performance'
        elif ((m1_type == 'energy' and m2_type == 'performance') or
              (m1_type == 'performance' and m2_type == 'energy')):
            return 'Energy vs Performance'
        else:
            return 'Other'

    def classify_intervention_type(self, intervention):
        """分类干预类型"""
        if intervention in self.metrics_energy:
            return 'Energy Metric'
        elif intervention in self.metrics_perf:
            return 'Performance Metric'
        else:
            return 'Hyperparameter'

    def analyze_tradeoff_types(self):
        """分析1: 权衡类型分布"""
        print("\n" + "="*60)
        print("分析1: 权衡类型分布")
        print("="*60)

        # 添加权衡类型列
        self.df_detailed['tradeoff_type'] = self.df_detailed.apply(
            self.classify_tradeoff_type, axis=1
        )

        # 统计各类型数量
        type_counts = self.df_detailed['tradeoff_type'].value_counts()
        print("\n权衡类型分布:")
        for tradeoff_type, count in type_counts.items():
            pct = count / len(self.df_detailed) * 100
            print(f"  - {tradeoff_type}: {count} ({pct:.1f}%)")

        # 按任务组统计
        print("\n各任务组的权衡类型分布:")
        group_type_cross = pd.crosstab(
            self.df_detailed['group_id'],
            self.df_detailed['tradeoff_type'],
            margins=True
        )
        print(group_type_cross)

        return type_counts

    def analyze_ate_distribution(self):
        """分析2: ATE幅度分布"""
        print("\n" + "="*60)
        print("分析2: ATE幅度分布")
        print("="*60)

        # 分析ATE1和ATE2的分布
        print("\nATE1统计:")
        print(self.df_detailed['ate1'].describe())

        print("\nATE2统计:")
        print(self.df_detailed['ate2'].describe())

        # 按权衡类型分析ATE幅度
        print("\n各权衡类型的ATE1中位数:")
        ate_by_type = self.df_detailed.groupby('tradeoff_type')[['ate1', 'ate2']].median()
        print(ate_by_type)

        # 极端ATE值
        print("\n极端ATE值（|ATE| > 1000）:")
        extreme_ate = self.df_detailed[
            (np.abs(self.df_detailed['ate1']) > 1000) |
            (np.abs(self.df_detailed['ate2']) > 1000)
        ]
        if len(extreme_ate) > 0:
            print(extreme_ate[['group_id', 'intervention', 'metric1', 'metric2', 'ate1', 'ate2']])
        else:
            print("  无")

        return self.df_detailed[['ate1', 'ate2']]

    def analyze_intervention_types(self):
        """分析3: 干预类型分布"""
        print("\n" + "="*60)
        print("分析3: 干预类型分布")
        print("="*60)

        # 添加干预类型列
        self.df_detailed['intervention_type'] = self.df_detailed['intervention'].apply(
            self.classify_intervention_type
        )

        # 统计干预类型
        intervention_counts = self.df_detailed['intervention_type'].value_counts()
        print("\n干预类型分布:")
        for intv_type, count in intervention_counts.items():
            pct = count / len(self.df_detailed) * 100
            print(f"  - {intv_type}: {count} ({pct:.1f}%)")

        # 详细干预列表
        print("\n具体干预分布:")
        intervention_list = self.df_detailed['intervention'].value_counts()
        print(intervention_list)

        # 按任务组统计干预类型
        print("\n各任务组的干预类型分布:")
        group_intervention_cross = pd.crosstab(
            self.df_detailed['group_id'],
            self.df_detailed['intervention_type'],
            margins=True
        )
        print(group_intervention_cross)

        return intervention_counts

    def analyze_group_comparison(self):
        """分析4: 任务组对比"""
        print("\n" + "="*60)
        print("分析4: 任务组对比")
        print("="*60)

        # 按任务组统计
        group_stats = self.df_detailed.groupby('group_id').agg({
            'intervention': 'count',
            'ate1': ['mean', 'median', 'std'],
            'ate2': ['mean', 'median', 'std']
        }).round(2)

        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns]
        group_stats = group_stats.rename(columns={'intervention_count': 'tradeoff_count'})

        print("\n各任务组统计:")
        print(group_stats)

        return group_stats

    def investigate_zero_tradeoff_groups(self):
        """分析5: 调查零权衡组（group2, group4）"""
        print("\n" + "="*60)
        print("分析5: 调查零权衡组（group2, group4）")
        print("="*60)

        zero_tradeoff_groups = ['group2_vulberta', 'group4_bug_localization']

        for group_id in zero_tradeoff_groups:
            print(f"\n{group_id}:")
            print(f"  - 权衡数: {len(self.all_tradeoffs.get(group_id, []))}")

        # 读取各组因果图统计
        print("\n可能原因分析:")
        print("  1. 数据质量问题（缺失值、方差）")
        print("  2. 因果图结构差异（边数、节点）")
        print("  3. ATE计算问题（统计显著性不足）")
        print("  4. Sign函数判定问题")

        print("\n需要进一步调查:")
        print("  - 检查各组的数据完整性")
        print("  - 对比各组的因果图边数")
        print("  - 检查各组的ATE计算结果")

    def create_visualizations(self):
        """生成可视化图表"""
        print("\n" + "="*60)
        print("生成可视化图表")
        print("="*60)

        # 图1: 权衡类型分布饼图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1.1 权衡类型分布
        type_counts = self.df_detailed['tradeoff_type'].value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                       colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        axes[0, 0].set_title('Tradeoff Type Distribution', fontsize=14, fontweight='bold')

        # 1.2 任务组权衡数量
        group_counts = self.df_detailed['group_id'].value_counts()
        axes[0, 1].bar(range(len(group_counts)), group_counts.values,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(group_counts)])
        axes[0, 1].set_xticks(range(len(group_counts)))
        axes[0, 1].set_xticklabels([g.replace('_', '\n') for g in group_counts.index],
                                    rotation=0, fontsize=8)
        axes[0, 1].set_title('Tradeoffs by Group', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 1.3 ATE1分布
        axes[1, 0].hist(np.abs(self.df_detailed['ate1']), bins=20,
                        color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('|ATE1|', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('|ATE1| Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 1.4 ATE2分布
        axes[1, 1].hist(np.abs(self.df_detailed['ate2']), bins=20,
                        color='coral', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('|ATE2|', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('|ATE2| Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'tradeoff_overview.png', dpi=300, bbox_inches='tight')
        print(f"  保存: {FIGURES_DIR / 'tradeoff_overview.png'}")
        plt.close()

        # 图2: 干预类型分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 2.1 干预类型
        intervention_counts = self.df_detailed['intervention_type'].value_counts()
        axes[0].bar(range(len(intervention_counts)), intervention_counts.values,
                   color=['#e74c3c', '#3498db', '#2ecc71'][:len(intervention_counts)])
        axes[0].set_xticks(range(len(intervention_counts)))
        axes[0].set_xticklabels(intervention_counts.index, fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Intervention Type Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # 2.2 按任务组的干预类型堆叠柱状图
        group_intervention_cross = pd.crosstab(
            self.df_detailed['group_id'],
            self.df_detailed['intervention_type']
        )
        group_intervention_cross.plot(kind='bar', stacked=True, ax=axes[1],
                                      color=['#e74c3c', '#3498db', '#2ecc71'])
        axes[1].set_xlabel('Group', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Intervention Types by Group', fontsize=13, fontweight='bold')
        axes[1].legend(title='Intervention Type', fontsize=9)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'intervention_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  保存: {FIGURES_DIR / 'intervention_distribution.png'}")
        plt.close()

        # 图3: ATE幅度对比（按权衡类型）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 3.1 ATE1 by tradeoff type
        tradeoff_types = self.df_detailed['tradeoff_type'].unique()
        ate1_by_type = [self.df_detailed[self.df_detailed['tradeoff_type'] == t]['ate1'].values
                        for t in tradeoff_types]
        bp1 = axes[0].boxplot(ate1_by_type, labels=[t.replace(' ', '\n') for t in tradeoff_types],
                              patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('ATE1', fontsize=11)
        axes[0].set_title('ATE1 by Tradeoff Type', fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)

        # 3.2 ATE2 by tradeoff type
        ate2_by_type = [self.df_detailed[self.df_detailed['tradeoff_type'] == t]['ate2'].values
                        for t in tradeoff_types]
        bp2 = axes[1].boxplot(ate2_by_type, labels=[t.replace(' ', '\n') for t in tradeoff_types],
                              patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        axes[1].set_ylabel('ATE2', fontsize=11)
        axes[1].set_title('ATE2 by Tradeoff Type', fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ate_by_tradeoff_type.png', dpi=300, bbox_inches='tight')
        print(f"  保存: {FIGURES_DIR / 'ate_by_tradeoff_type.png'}")
        plt.close()

        print("\n可视化图表生成完成！")

    def generate_summary_report(self):
        """生成摘要报告"""
        print("\n" + "="*60)
        print("生成摘要报告")
        print("="*60)

        report_lines = [
            "# 权衡结果分析摘要报告",
            "",
            f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 概览",
            "",
            f"- **总权衡数**: {len(self.df_detailed)}",
            f"- **涉及任务组**: {self.df_detailed['group_id'].nunique()}",
            f"- **零权衡组**: group2_vulberta, group4_bug_localization",
            "",
            "## 2. 权衡类型分布",
            "",
        ]

        type_counts = self.df_detailed['tradeoff_type'].value_counts()
        for tradeoff_type, count in type_counts.items():
            pct = count / len(self.df_detailed) * 100
            report_lines.append(f"- **{tradeoff_type}**: {count} ({pct:.1f}%)")

        report_lines.extend([
            "",
            "## 3. 干预类型分布",
            "",
        ])

        intervention_counts = self.df_detailed['intervention_type'].value_counts()
        for intv_type, count in intervention_counts.items():
            pct = count / len(self.df_detailed) * 100
            report_lines.append(f"- **{intv_type}**: {count} ({pct:.1f}%)")

        report_lines.extend([
            "",
            "## 4. ATE幅度统计",
            "",
            "### ATE1:",
            f"- 均值: {self.df_detailed['ate1'].mean():.2f}",
            f"- 中位数: {self.df_detailed['ate1'].median():.2f}",
            f"- 标准差: {self.df_detailed['ate1'].std():.2f}",
            "",
            "### ATE2:",
            f"- 均值: {self.df_detailed['ate2'].mean():.2f}",
            f"- 中位数: {self.df_detailed['ate2'].median():.2f}",
            f"- 标准差: {self.df_detailed['ate2'].std():.2f}",
            "",
            "## 5. 任务组对比",
            "",
        ])

        group_counts = self.df_detailed['group_id'].value_counts()
        for group_id, count in group_counts.items():
            report_lines.append(f"- **{group_id}**: {count} 个权衡")

        report_lines.extend([
            "",
            "## 6. 主要发现",
            "",
            "1. **group2和group4没有检测到权衡**",
            "   - 可能原因：数据质量、因果图结构、统计显著性",
            "   - 需要进一步调查",
            "",
            "2. **能耗vs性能权衡占比较高**",
            "   - 说明性能优化和能耗优化存在冲突",
            "   - 这是核心研究问题的证据",
            "",
            "3. **干预类型分布**",
            "   - 能耗指标干预是最常见的权衡触发因素",
            "   - 超参数干预也产生显著权衡",
            "",
            "4. **ATE幅度差异大**",
            "   - 某些权衡的效应量极大（>1000）",
            "   - 需要检查异常值处理",
            "",
        ])

        report_text = "\n".join(report_lines)

        # 保存报告
        report_path = FIGURES_DIR / 'TRADEOFF_ANALYSIS_SUMMARY.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n报告已保存: {report_path}")
        print("\n" + "="*60)
        print(report_text)
        print("="*60)

        return report_text

    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "="*60)
        print("开始权衡结果深入分析")
        print("="*60)

        # 加载数据
        self.load_data()

        # 运行各项分析
        self.analyze_tradeoff_types()
        self.analyze_ate_distribution()
        self.analyze_intervention_types()
        self.analyze_group_comparison()
        self.investigate_zero_tradeoff_groups()

        # 生成可视化
        self.create_visualizations()

        # 生成摘要报告
        self.generate_summary_report()

        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)


def main():
    """主函数"""
    analyzer = TradeoffAnalyzer()
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
