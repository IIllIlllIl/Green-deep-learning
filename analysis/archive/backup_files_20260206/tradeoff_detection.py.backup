"""
权衡检测模块 - 实现论文算法1
Trade-off Detection based on Causal Inference

基于因果推断结果，检测不同指标之间的权衡关系。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
import warnings


class TradeoffDetector:
    """
    权衡检测器

    实现论文算法1：基于因果推断的权衡检测

    算法核心思想：
    1. 对于共享源节点的边对 (A→B, A→C)
    2. 计算各自的因果效应（ATE）
    3. 使用sign函数判断效应方向
    4. 如果sign相反且统计显著 → 检测到权衡

    参数:
        sign_functions: sign函数字典 {metric_name: sign_func}
        verbose: 是否输出详细信息

    示例:
        >>> detector = TradeoffDetector(sign_functions)
        >>> tradeoffs = detector.detect_tradeoffs(causal_effects)
        >>> for t in tradeoffs:
        ...     print(f"权衡: {t['intervention']} → {t['metric1']} vs {t['metric2']}")
    """

    def __init__(self,
                 sign_functions: Dict[str, Callable],
                 verbose: bool = False):
        if not sign_functions:
            raise ValueError("sign_functions不能为空")

        self.sign_functions = sign_functions
        self.verbose = verbose
        self.detected_tradeoffs = []

    def detect_tradeoffs(self,
                        causal_effects: Dict[str, Dict],
                        require_significance: bool = True) -> List[Dict]:
        """
        检测权衡关系（论文算法1）

        算法流程：
        1. 遍历所有边对 (A→B, A→C)
        2. 检查是否共享源节点A
        3. 计算各自的ATE sign
        4. 如果sign相反且统计显著 → 权衡

        参数:
            causal_effects: 因果效应字典
                {
                    'A->B': {'ate': float, 'ci_lower': float, 'ci_upper': float, ...},
                    ...
                }
            require_significance: 是否要求统计显著

        返回:
            tradeoffs: 检测到的权衡列表
                [
                    {
                        'intervention': str,  # 干预变量（源节点）
                        'metric1': str,       # 指标1（目标节点1）
                        'metric2': str,       # 指标2（目标节点2）
                        'ate1': float,        # 指标1的ATE
                        'ate2': float,        # 指标2的ATE
                        'sign1': str,         # 指标1的sign ('+' or '-')
                        'sign2': str,         # 指标2的sign ('+' or '-')
                        'is_significant': bool
                    },
                    ...
                ]
        """
        if not causal_effects:
            warnings.warn("causal_effects为空，无法检测权衡")
            return []

        tradeoffs = []

        # 将边按源节点分组
        edges_by_source = {}
        for edge, result in causal_effects.items():
            source, target = edge.split('->')
            if source not in edges_by_source:
                edges_by_source[source] = []
            edges_by_source[source].append((target, result))

        if self.verbose:
            print(f"\n开始权衡检测...")
            print(f"  因果边总数: {len(causal_effects)}")
            print(f"  源节点数: {len(edges_by_source)}")

        # 对每个源节点，检查其所有目标节点对
        for source, targets in edges_by_source.items():
            if len(targets) < 2:
                continue  # 至少需要2个目标节点才能检测权衡

            # 检查所有目标节点对
            for i, (target1, result1) in enumerate(targets):
                for target2, result2 in targets[i+1:]:
                    # 检查是否为权衡
                    tradeoff = self._check_tradeoff_pair(
                        source, target1, result1, target2, result2,
                        require_significance
                    )

                    if tradeoff:
                        tradeoffs.append(tradeoff)

        self.detected_tradeoffs = tradeoffs

        if self.verbose:
            print(f"\n✓ 权衡检测完成")
            print(f"  检测到权衡: {len(tradeoffs)}")

        return tradeoffs

    def _check_tradeoff_pair(self,
                            source: str,
                            target1: str,
                            result1: Dict,
                            target2: str,
                            result2: Dict,
                            require_significance: bool) -> Optional[Dict]:
        """
        检查一对边是否构成权衡

        参数:
            source: 源节点（干预变量）
            target1: 目标节点1
            result1: target1的因果效应结果
            target2: 目标节点2
            result2: target2的因果效应结果
            require_significance: 是否要求统计显著

        返回:
            tradeoff_info: 如果是权衡返回详细信息，否则返回None
        """
        ate1 = result1['ate']
        ate2 = result2['ate']

        # 提取指标名称（移除前缀如'Te_'）
        metric1 = self._extract_metric_name(target1)
        metric2 = self._extract_metric_name(target2)

        # 检查是否有对应的sign函数
        if metric1 not in self.sign_functions or metric2 not in self.sign_functions:
            return None

        # 计算sign（使用简化版本：只看ATE的正负）
        # 注意：论文中sign函数需要当前值和变化量，这里简化为只看变化量
        sign1 = self._compute_sign(metric1, ate1)
        sign2 = self._compute_sign(metric2, ate2)

        # 检查sign是否相反
        if sign1 == sign2:
            return None  # 不是权衡（都改善或都恶化）

        # 检查统计显著性
        is_sig1 = result1.get('is_significant', False)
        is_sig2 = result2.get('is_significant', False)

        if require_significance and not (is_sig1 and is_sig2):
            return None  # 不满足显著性要求

        # 构造权衡信息
        tradeoff = {
            'intervention': source,
            'metric1': target1,
            'metric2': target2,
            'ate1': ate1,
            'ate2': ate2,
            'sign1': sign1,
            'sign2': sign2,
            'is_significant': is_sig1 and is_sig2
        }

        if self.verbose:
            print(f"\n  ⚠️  检测到权衡:")
            print(f"    干预: {source}")
            print(f"    {target1}: ATE={ate1:.4f} ({sign1})")
            print(f"    {target2}: ATE={ate2:.4f} ({sign2})")

        return tradeoff

    def _extract_metric_name(self, target: str) -> str:
        """
        从目标节点名提取指标名

        例如: 'Te_Acc' -> 'Acc', 'Tr_SPD' -> 'SPD'
        """
        # 移除常见前缀
        for prefix in ['Te_', 'Tr_', 'D_']:
            if target.startswith(prefix):
                return target[len(prefix):]
        return target

    def _compute_sign(self, metric: str, ate: float) -> str:
        """
        计算sign

        参数:
            metric: 指标名称
            ate: 平均处理效应

        返回:
            sign: '+' (改善) 或 '-' (恶化)
        """
        # 获取sign函数
        sign_func = self.sign_functions.get(metric)
        if sign_func is None:
            # 默认：正值为改善
            return '+' if ate > 0 else '-'

        # 使用sign函数（传入当前值=0，变化量=ate）
        # 这是简化版本，实际应用中需要真实的当前值
        try:
            sign = sign_func(0, ate)
            return sign
        except Exception as e:
            warnings.warn(f"计算sign失败 (metric={metric}): {e}")
            return '+' if ate > 0 else '-'

    def summarize_tradeoffs(self, tradeoffs: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        生成权衡摘要表

        参数:
            tradeoffs: 权衡列表（如果为None，使用上次检测结果）

        返回:
            summary: 摘要DataFrame
        """
        if tradeoffs is None:
            tradeoffs = self.detected_tradeoffs

        if not tradeoffs:
            return pd.DataFrame()

        records = []
        for t in tradeoffs:
            records.append({
                'Intervention': t['intervention'],
                'Metric1': t['metric1'],
                'Metric2': t['metric2'],
                'ATE1': t['ate1'],
                'ATE2': t['ate2'],
                'Sign1': t['sign1'],
                'Sign2': t['sign2'],
                'Significant': t['is_significant']
            })

        return pd.DataFrame(records)

    def visualize_tradeoffs(self, tradeoffs: Optional[List[Dict]] = None, output_path: str = 'tradeoffs.png'):
        """
        可视化权衡关系

        参数:
            tradeoffs: 权衡列表
            output_path: 输出图片路径
        """
        if tradeoffs is None:
            tradeoffs = self.detected_tradeoffs

        if not tradeoffs:
            warnings.warn("没有权衡可视化")
            return

        try:
            import matplotlib.pyplot as plt

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 提取数据
            ate1_values = [t['ate1'] for t in tradeoffs]
            ate2_values = [t['ate2'] for t in tradeoffs]
            labels = [f"{t['intervention']}\n{t['metric1']} vs {t['metric2']}"
                     for t in tradeoffs]

            # 绘制散点图
            colors = ['red' if t['is_significant'] else 'gray' for t in tradeoffs]
            ax.scatter(ate1_values, ate2_values, c=colors, s=100, alpha=0.6)

            # 添加标签
            for i, label in enumerate(labels):
                ax.annotate(label, (ate1_values[i], ate2_values[i]),
                           fontsize=8, ha='center')

            # 添加参考线
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

            # 标注象限
            ax.text(0.95, 0.95, '双赢', transform=ax.transAxes,
                   fontsize=12, ha='right', va='top')
            ax.text(0.05, 0.05, '双输', transform=ax.transAxes,
                   fontsize=12, ha='left', va='bottom')
            ax.text(0.95, 0.05, '权衡', transform=ax.transAxes,
                   fontsize=12, ha='right', va='bottom', color='red')
            ax.text(0.05, 0.95, '权衡', transform=ax.transAxes,
                   fontsize=12, ha='left', va='top', color='red')

            ax.set_xlabel('Metric 1 ATE', fontsize=12)
            ax.set_ylabel('Metric 2 ATE', fontsize=12)
            ax.set_title('因果效应权衡分析', fontsize=14)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 权衡可视化已保存到: {output_path}")

        except ImportError:
            warnings.warn("matplotlib未安装，跳过可视化")
        except Exception as e:
            warnings.warn(f"可视化失败: {e}")


# 辅助函数
def analyze_tradeoff_pattern(tradeoffs: List[Dict]) -> Dict[str, int]:
    """
    分析权衡模式

    参数:
        tradeoffs: 权衡列表

    返回:
        pattern_counts: 各类模式的统计
    """
    patterns = {
        'accuracy_vs_fairness': 0,
        'fairness_vs_robustness': 0,
        'accuracy_vs_robustness': 0,
        'other': 0
    }

    for t in tradeoffs:
        m1 = t['metric1'].lower()
        m2 = t['metric2'].lower()

        # 判断模式
        is_acc = any(x in m1 or x in m2 for x in ['acc', 'f1'])
        is_fair = any(x in m1 or x in m2 for x in ['spd', 'di', 'aod', 'cons'])
        is_robust = any(x in m1 or x in m2 for x in ['fgsm', 'pgd'])

        if is_acc and is_fair:
            patterns['accuracy_vs_fairness'] += 1
        elif is_fair and is_robust:
            patterns['fairness_vs_robustness'] += 1
        elif is_acc and is_robust:
            patterns['accuracy_vs_robustness'] += 1
        else:
            patterns['other'] += 1

    return patterns


def format_tradeoff_report(tradeoffs: List[Dict]) -> str:
    """
    生成权衡报告文本

    参数:
        tradeoffs: 权衡列表

    返回:
        report: 格式化的报告文本
    """
    if not tradeoffs:
        return "未检测到权衡关系。"

    report = f"检测到 {len(tradeoffs)} 个权衡关系：\n\n"

    for i, t in enumerate(tradeoffs, 1):
        sig_marker = "***" if t['is_significant'] else ""
        report += f"{i}. 干预: {t['intervention']} {sig_marker}\n"
        report += f"   - {t['metric1']}: ATE={t['ate1']:+.4f} ({t['sign1']})\n"
        report += f"   - {t['metric2']}: ATE={t['ate2']:+.4f} ({t['sign2']})\n"
        report += f"   说明: 改进{t['metric1']}会恶化{t['metric2']}\n\n"

    # 添加模式统计
    patterns = analyze_tradeoff_pattern(tradeoffs)
    report += "\n权衡模式统计:\n"
    for pattern, count in patterns.items():
        if count > 0:
            report += f"  - {pattern}: {count}\n"

    return report
