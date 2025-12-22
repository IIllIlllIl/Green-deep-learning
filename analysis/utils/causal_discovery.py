"""
DiBS因果图学习模块
实现论文中的因果图发现算法

基于DiBS (Differentiable Bayesian Structure Learning)
论文: https://arxiv.org/abs/2105.11839
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings


class CausalGraphLearner:
    """
    使用DiBS学习因果图

    DiBS是一种基于变分推断的因果发现算法，可以从观测数据中学习有向无环图(DAG)。

    参数:
        n_vars: 变量数量（论文中为46个指标）
        alpha: DAG惩罚参数（论文中为0.9），控制图的稀疏性
        n_steps: 迭代次数（论文中为10000次）
        random_seed: 随机种子，用于结果可复现

    示例:
        >>> learner = CausalGraphLearner(n_vars=10, n_steps=1000)
        >>> graph = learner.fit(data_df)
        >>> edges = learner.get_edges(threshold=0.5)
    """

    def __init__(self,
                 n_vars: int = 46,
                 alpha: float = 0.9,
                 n_steps: int = 10000,
                 random_seed: int = 42):

        # 输入验证
        if n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {n_vars}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        self.n_vars = n_vars
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_seed = random_seed

        # 初始化DiBS模型
        self.model = None
        self.learned_graph = None

    def fit(self, data: pd.DataFrame, verbose: bool = True) -> np.ndarray:
        """
        学习因果图

        使用DiBS算法从数据中学习变量之间的因果关系。

        参数:
            data: 训练数据，shape (n_samples, n_vars)
            verbose: 是否输出进度信息

        返回:
            learned_graph: 邻接矩阵，shape (n_vars, n_vars)
                          learned_graph[i,j] > 0 表示存在因果边 i → j

        Raises:
            ValueError: 如果数据形状不匹配
            RuntimeError: 如果DiBS拟合失败
        """
        # 输入验证
        if data is None or len(data) == 0:
            raise ValueError("data cannot be None or empty")
        if data.shape[1] != self.n_vars:
            raise ValueError(
                f"Expected {self.n_vars} variables, got {data.shape[1]}"
            )

        # 数据预处理：离散变量→连续变量
        data_continuous = self._discretize_to_continuous(data)

        if verbose:
            print(f"开始DiBS因果图学习...")
            print(f"  变量数: {self.n_vars}")
            print(f"  样本数: {len(data)}")
            print(f"  迭代次数: {self.n_steps}")
            print(f"  Alpha参数: {self.alpha}")

        try:
            # 尝试导入DiBS
            from dibs.inference import JointDiBS
            from dibs.target import make_linear_gaussian_model
            import jax.random as random

            # 运行DiBS（这是耗时操作）
            if verbose:
                print(f"\n正在运行DiBS算法（这可能需要几分钟）...")

            # 创建JAX随机密钥
            key = random.PRNGKey(self.random_seed)

            # 创建图模型和似然模型
            # 使用线性高斯模型作为默认模型
            key, subk = random.split(key)
            _, graph_model, likelihood_model = make_linear_gaussian_model(
                key=subk,
                n_vars=self.n_vars,
                n_observations=len(data_continuous),
                graph_prior_str='er'  # Erdős-Rényi先验
            )

            # 初始化DiBS模型
            self.model = JointDiBS(
                x=data_continuous,
                interv_mask=None,  # 观测数据，无干预
                graph_model=graph_model,
                likelihood_model=likelihood_model,
                alpha_linear=self.alpha  # DAG惩罚参数
            )

            # 运行SVGD采样
            key, subk = random.split(key)
            n_particles = 10  # 粒子数量
            gs, thetas = self.model.sample(
                key=subk,
                n_particles=n_particles,
                steps=self.n_steps,
                callback_every=100 if verbose else None
            )

            # 获取学习到的图：对所有粒子的图求平均
            # gs的形状是 [n_particles, n_vars, n_vars]
            import jax.numpy as jnp
            self.learned_graph = jnp.mean(gs, axis=0)  # 平均边概率

            # 转换为numpy数组
            self.learned_graph = np.array(self.learned_graph)

            if verbose:
                n_edges = np.sum(self.learned_graph > 0)
                density = n_edges / (self.n_vars * (self.n_vars - 1))
                print(f"\n✓ DiBS学习完成")
                print(f"  学到的边数: {n_edges}")
                print(f"  图密度: {density:.3f}")
                print(f"  是否为DAG: {self._is_dag(self.learned_graph)}")

        except ImportError as e:
            raise RuntimeError(
                f"无法导入DiBS库。请确保已正确安装DiBS: {e}\n"
                "安装方法: pip install -e /tmp/dibs"
            )
        except Exception as e:
            raise RuntimeError(f"DiBS拟合失败: {e}")

        return self.learned_graph

    def _discretize_to_continuous(self, data: pd.DataFrame) -> np.ndarray:
        """
        将离散变量转换为连续变量

        DiBS需要连续数据输入。对于离散变量，我们添加小的随机噪声。
        论文未详细说明此步骤，这里使用简单的噪声注入方法。

        参数:
            data: 原始数据

        返回:
            data_continuous: 连续化后的数据
        """
        data_continuous = data.values.copy().astype(float)

        # 检测离散列（唯一值数量 < 10）
        for i in range(data.shape[1]):
            n_unique = len(np.unique(data.iloc[:, i]))
            if n_unique < 10:
                # 离散列：添加小噪声（标准差为值域的1%）
                col_range = data.iloc[:, i].max() - data.iloc[:, i].min()
                noise_std = max(0.01 * col_range, 0.01)  # 至少0.01
                noise = np.random.normal(0, noise_std, size=len(data))
                data_continuous[:, i] += noise

        return data_continuous

    def _is_dag(self, graph: np.ndarray) -> bool:
        """
        检查图是否为有向无环图(DAG)

        使用深度优先搜索检测环
        """
        n = len(graph)
        visited = np.zeros(n, dtype=bool)
        rec_stack = np.zeros(n, dtype=bool)

        def dfs(v):
            visited[v] = True
            rec_stack[v] = True

            # 检查所有邻居
            for u in range(n):
                if graph[v, u] > 0:  # 存在边 v → u
                    if not visited[u]:
                        if not dfs(u):
                            return False
                    elif rec_stack[u]:
                        return False  # 发现环

            rec_stack[v] = False
            return True

        # 对所有未访问节点运行DFS
        for i in range(n):
            if not visited[i]:
                if not dfs(i):
                    return False

        return True

    def get_edges(self, threshold: float = 0.5) -> list:
        """
        获取因果边列表

        参数:
            threshold: 边权重阈值，只返回权重大于此值的边

        返回:
            edges: [(source, target, weight), ...]

        Raises:
            RuntimeError: 如果还未调用fit()方法
        """
        if self.learned_graph is None:
            raise RuntimeError("Must call fit() first")

        edges = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                weight = self.learned_graph[i, j]
                if weight > threshold:
                    edges.append((i, j, float(weight)))

        return edges

    def save_graph(self, filepath: str):
        """
        保存学习到的图到文件

        参数:
            filepath: 保存路径（.npy格式）

        Raises:
            RuntimeError: 如果还未调用fit()方法
        """
        if self.learned_graph is None:
            raise RuntimeError("Must call fit() first")

        np.save(filepath, self.learned_graph)
        print(f"✓ 因果图已保存到: {filepath}")

    def load_graph(self, filepath: str):
        """
        从文件加载已保存的图

        参数:
            filepath: 图文件路径（.npy格式）
        """
        self.learned_graph = np.load(filepath)
        self.n_vars = self.learned_graph.shape[0]
        print(f"✓ 因果图已从 {filepath} 加载")
        print(f"  变量数: {self.n_vars}")
        print(f"  边数: {np.sum(self.learned_graph > 0)}")


# 辅助函数：可视化因果图
def visualize_causal_graph(graph: np.ndarray,
                          var_names: list,
                          output_path: str = 'causal_graph.png',
                          threshold: float = 0.5,
                          top_k: Optional[int] = None):
    """
    可视化因果图

    参数:
        graph: 邻接矩阵
        var_names: 变量名列表
        output_path: 输出图片路径
        threshold: 只显示权重大于此值的边
        top_k: 只显示权重最大的k条边（可选）
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        # 提取边
        edges = []
        for i in range(len(graph)):
            for j in range(len(graph)):
                if graph[i, j] > threshold:
                    edges.append((i, j, graph[i, j]))

        # 按权重排序
        edges.sort(key=lambda x: x[2], reverse=True)

        # 如果指定top_k，只保留前k条边
        if top_k is not None:
            edges = edges[:top_k]

        # 创建有向图
        G = nx.DiGraph()
        for i, j, weight in edges:
            G.add_edge(var_names[i], var_names[j], weight=weight)

        # 绘制
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightblue',
                              node_size=500,
                              alpha=0.9)

        # 绘制边
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              width=2,
                              alpha=0.6)

        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 因果图已保存到: {output_path}")
        print(f"  显示边数: {len(edges)}")

    except ImportError:
        warnings.warn("matplotlib或networkx未安装，跳过可视化")
    except Exception as e:
        warnings.warn(f"可视化失败: {e}")
