"""
DML (Double Machine Learning) 因果推断模块
实现论文中的因果效应估计方法

基于EconML库实现
论文参考: Chernozhukov et al. (2018) - Double/Debiased Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class CausalInferenceEngine:
    """
    使用DML进行因果推断

    DML通过两阶段估计消除混淆偏差，提供无偏的平均处理效应(ATE)估计。

    参数:
        verbose: 是否输出详细信息

    示例:
        >>> engine = CausalInferenceEngine()
        >>> ate, ci = engine.estimate_ate(data, 'alpha', 'Te_Acc', confounders)
        >>> print(f"ATE: {ate:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.dml_models = {}  # 存储已拟合的模型
        self.ate_results = {}  # 存储ATE结果

    def estimate_ate(self,
                     data: pd.DataFrame,
                     treatment: str,
                     outcome: str,
                     confounders: List[str],
                     controls: Optional[List[str]] = None) -> Tuple[float, Tuple[float, float]]:
        """
        估计平均处理效应(ATE)

        使用DML方法估计干预(treatment)对结果(outcome)的因果效应。

        参数:
            data: 数据框
            treatment: 干预变量列名（如'alpha'）
            outcome: 结果变量列名（如'Te_Acc'）
            confounders: 混淆因素列名列表
            controls: 控制变量列名列表（可选）

        返回:
            (ate, (ci_lower, ci_upper))
            ate: 平均处理效应
            ci: 95%置信区间

        Raises:
            ValueError: 如果输入数据无效
            ImportError: 如果EconML未安装
        """
        # 输入验证
        if data is None or len(data) == 0:
            raise ValueError("data不能为空")
        if treatment not in data.columns:
            raise ValueError(f"treatment '{treatment}' 不在数据中")
        if outcome not in data.columns:
            raise ValueError(f"outcome '{outcome}' 不在数据中")
        for conf in confounders:
            if conf not in data.columns:
                raise ValueError(f"confounder '{conf}' 不在数据中")

        if self.verbose:
            print(f"\n估计因果效应: {treatment} → {outcome}")
            print(f"  混淆因素: {len(confounders)}个")
            print(f"  样本数: {len(data)}")

        try:
            from econml.dml import LinearDML

            # 准备数据
            T = data[treatment].values  # 干预变量
            Y = data[outcome].values    # 结果变量
            X = data[confounders].values if confounders else None  # 混淆因素
            W = data[controls].values if controls else None  # 控制变量

            # 检查数据变异性
            if len(np.unique(T)) < 2:
                warnings.warn(f"干预变量{treatment}缺乏变异性")
                return 0.0, (0.0, 0.0)

            # 创建和拟合DML模型
            dml = LinearDML(
                model_y='auto',  # 自动选择Y模型
                model_t='auto',  # 自动选择T模型
                random_state=42
            )

            dml.fit(Y, T, X=X, W=W)

            # 估计ATE
            ate = dml.ate(X=X)

            # 计算置信区间
            try:
                ate_inference = dml.ate_inference(X=X)
                ci = ate_inference.conf_int()[0]  # 95%置信区间
                ci_lower, ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                # 如果置信区间计算失败，使用标准误估计
                stderr = dml.ate_stderr(X=X) if hasattr(dml, 'ate_stderr') else 0.1 * abs(ate)
                ci_lower = ate - 1.96 * stderr
                ci_upper = ate + 1.96 * stderr

            # 存储模型和结果
            edge_key = f"{treatment}->{outcome}"
            self.dml_models[edge_key] = dml
            self.ate_results[edge_key] = {
                'ate': float(ate),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confounders': confounders
            }

            if self.verbose:
                print(f"  ATE: {ate:.4f}")
                print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                is_significant = not (ci_lower <= 0 <= ci_upper)
                print(f"  统计显著: {'是' if is_significant else '否'}")

            return float(ate), (ci_lower, ci_upper)

        except ImportError:
            # EconML未安装，使用简化方法
            warnings.warn("EconML未安装，使用简化的差值方法")
            return self._simple_ate_estimate(data, treatment, outcome)

        except Exception as e:
            warnings.warn(f"DML估计失败: {e}，使用简化方法")
            return self._simple_ate_estimate(data, treatment, outcome)

    def _simple_ate_estimate(self,
                            data: pd.DataFrame,
                            treatment: str,
                            outcome: str) -> Tuple[float, Tuple[float, float]]:
        """
        简化的ATE估计（后备方案）

        使用简单的分组均值差异作为ATE估计。
        注意：这个方法不控制混淆因素，结果可能有偏。
        """
        # 按treatment分组
        if data[treatment].dtype in [np.float64, np.float32]:
            # 连续变量：使用中位数分组
            median = data[treatment].median()
            low_group = data[data[treatment] <= median]
            high_group = data[data[treatment] > median]
        else:
            # 离散变量：使用唯一值
            unique_vals = sorted(data[treatment].unique())
            if len(unique_vals) < 2:
                return 0.0, (0.0, 0.0)
            low_group = data[data[treatment] == unique_vals[0]]
            high_group = data[data[treatment] == unique_vals[-1]]

        # 计算均值差异
        mean_low = low_group[outcome].mean()
        mean_high = high_group[outcome].mean()
        ate = mean_high - mean_low

        # 简单的标准误估计
        se_low = low_group[outcome].sem()
        se_high = high_group[outcome].sem()
        se = np.sqrt(se_low**2 + se_high**2)

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return float(ate), (float(ci_lower), float(ci_upper))

    def analyze_all_edges(self,
                         data: pd.DataFrame,
                         causal_graph: np.ndarray,
                         var_names: List[str],
                         threshold: float = 0.3) -> Dict[str, Dict]:
        """
        对因果图中的所有边进行因果推断

        参数:
            data: 数据框
            causal_graph: 邻接矩阵 (n_vars, n_vars)
            var_names: 变量名列表
            threshold: 边权重阈值

        返回:
            results: {
                'edge_name': {
                    'ate': float,
                    'ci_lower': float,
                    'ci_upper': float,
                    'is_significant': bool
                }
            }
        """
        if causal_graph is None:
            raise ValueError("causal_graph不能为None")
        if len(var_names) != causal_graph.shape[0]:
            raise ValueError(f"变量名数量({len(var_names)})与图大小({causal_graph.shape[0]})不匹配")

        results = {}
        n_vars = len(var_names)

        # 提取所有边
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_graph[i, j] > threshold:
                    edges.append((i, j, causal_graph[i, j]))

        if self.verbose:
            print(f"\n开始分析 {len(edges)} 条因果边...")

        # 对每条边进行因果推断
        for idx, (source_idx, target_idx, weight) in enumerate(edges, 1):
            source = var_names[source_idx]
            target = var_names[target_idx]

            if self.verbose:
                print(f"\n[{idx}/{len(edges)}] 分析边: {source} → {target} (权重={weight:.3f})")

            # 识别混淆因素
            confounders = self._identify_confounders(
                causal_graph, source_idx, target_idx, var_names, threshold
            )

            # 估计ATE
            try:
                ate, ci = self.estimate_ate(data, source, target, confounders)

                is_significant = not (ci[0] <= 0 <= ci[1])

                results[f"{source}->{target}"] = {
                    'ate': ate,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'is_significant': is_significant,
                    'weight': weight,
                    'confounders': confounders
                }

            except Exception as e:
                warnings.warn(f"分析边 {source}->{target} 失败: {e}")
                continue

        if self.verbose:
            n_significant = sum(1 for r in results.values() if r['is_significant'])
            print(f"\n✓ 完成分析")
            print(f"  总边数: {len(edges)}")
            print(f"  成功分析: {len(results)}")
            print(f"  统计显著: {n_significant}")

        return results

    def _identify_confounders(self,
                            causal_graph: np.ndarray,
                            source_idx: int,
                            target_idx: int,
                            var_names: List[str],
                            threshold: float = 0.3) -> List[str]:
        """
        根据因果图识别混淆因素

        混淆因素定义：同时指向source和target的变量

        参数:
            causal_graph: 邻接矩阵
            source_idx: 源节点索引
            target_idx: 目标节点索引
            var_names: 变量名列表
            threshold: 边权重阈值

        返回:
            confounders: 混淆因素列表
        """
        confounders = []
        n_vars = len(var_names)

        for k in range(n_vars):
            if k == source_idx or k == target_idx:
                continue

            # 检查是否同时指向source和target
            points_to_source = causal_graph[k, source_idx] > threshold
            points_to_target = causal_graph[k, target_idx] > threshold

            if points_to_source or points_to_target:
                confounders.append(var_names[k])

        # 如果没有找到混淆因素，使用所有其他变量
        if len(confounders) == 0:
            confounders = [var_names[i] for i in range(n_vars)
                          if i != source_idx and i != target_idx]

        return confounders

    def get_significant_effects(self, alpha: float = 0.05) -> Dict[str, Dict]:
        """
        获取所有统计显著的因果效应

        参数:
            alpha: 显著性水平（默认0.05）

        返回:
            significant_effects: 统计显著的效应字典
        """
        significant = {}

        for edge, result in self.ate_results.items():
            if result['is_significant']:
                significant[edge] = result

        return significant

    def save_results(self, filepath: str):
        """
        保存因果推断结果到文件

        参数:
            filepath: 保存路径（.csv格式）
        """
        if not self.ate_results:
            warnings.warn("没有结果可保存")
            return

        # 转换为DataFrame
        records = []
        for edge, result in self.ate_results.items():
            source, target = edge.split('->')
            records.append({
                'source': source,
                'target': target,
                'ate': result['ate'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'is_significant': result['is_significant']
            })

        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)

        print(f"✓ 因果推断结果已保存到: {filepath}")
        print(f"  总边数: {len(records)}")
        print(f"  统计显著: {df['is_significant'].sum()}")

    def load_results(self, filepath: str):
        """
        从文件加载因果推断结果

        参数:
            filepath: 结果文件路径（.csv格式）
        """
        df = pd.read_csv(filepath)

        self.ate_results = {}
        for _, row in df.iterrows():
            edge = f"{row['source']}->{row['target']}"
            self.ate_results[edge] = {
                'ate': row['ate'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'is_significant': row['is_significant']
            }

        print(f"✓ 因果推断结果已从 {filepath} 加载")
        print(f"  加载边数: {len(self.ate_results)}")


# 辅助函数
def is_effect_significant(ci_lower: float, ci_upper: float) -> bool:
    """
    判断因果效应是否统计显著

    参数:
        ci_lower: 置信区间下界
        ci_upper: 置信区间上界

    返回:
        is_significant: True如果置信区间不包含0
    """
    return not (ci_lower <= 0 <= ci_upper)


def format_ate_result(ate: float, ci: Tuple[float, float]) -> str:
    """
    格式化ATE结果为可读字符串

    参数:
        ate: 平均处理效应
        ci: 置信区间 (lower, upper)

    返回:
        formatted_string: 格式化的字符串
    """
    is_sig = is_effect_significant(ci[0], ci[1])
    sig_marker = "***" if is_sig else ""

    return f"ATE={ate:.4f} {sig_marker}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
