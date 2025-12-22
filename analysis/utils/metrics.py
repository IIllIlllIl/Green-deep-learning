"""
指标计算模块（精简版）
只计算核心指标，避免复杂的鲁棒性测试
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from utils.aif360_utils import to_aif360_dataset
import warnings


class MetricsCalculator:
    """计算所有类型的指标（精简版）"""

    def __init__(self, model, sensitive_attr='sex'):
        self.model = model
        self.sensitive_attr = sensitive_attr

    def compute_all_metrics(self, X, y, sensitive_features, phase='Te'):
        """
        计算所有指标

        参数:
            X: 特征数据，shape (n_samples, n_features)
            y: 真实标签，shape (n_samples,)
            sensitive_features: 敏感属性值，shape (n_samples,)
            phase: 'D'(dataset), 'Tr'(train), 'Te'(test)

        返回: Dict[str, float]

        Raises:
            ValueError: 如果输入数据无效
        """
        # 输入验证
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if sensitive_features is None or len(sensitive_features) == 0:
            raise ValueError("sensitive_features cannot be None or empty")
        if len(X) != len(y) or len(X) != len(sensitive_features):
            raise ValueError(
                f"Shape mismatch: X={len(X)}, y={len(y)}, "
                f"sensitive={len(sensitive_features)}"
            )

        metrics = {}

        # 1. 性能指标
        if phase != 'D':
            try:
                y_pred = self.model.predict(X)
                metrics[f'{phase}_Acc'] = accuracy_score(y, y_pred)
                metrics[f'{phase}_F1'] = f1_score(y, y_pred, average='binary', zero_division=0)
            except Exception as e:
                warnings.warn(f"Failed to compute performance metrics: {e}")
                metrics[f'{phase}_Acc'] = 0.0
                metrics[f'{phase}_F1'] = 0.0
        else:
            y_pred = None

        # 2. 公平性指标（使用AIF360）
        try:
            # 创建AIF360数据集 - 使用工具函数
            dataset = to_aif360_dataset(X, y, sensitive_features, self.sensitive_attr)
            dataset_metric = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=[{self.sensitive_attr: 0}],
                privileged_groups=[{self.sensitive_attr: 1}]
            )

            # 群体公平性
            try:
                metrics[f'{phase}_DI'] = dataset_metric.disparate_impact()
            except (ZeroDivisionError, ValueError):
                metrics[f'{phase}_DI'] = 0.0

            try:
                metrics[f'{phase}_SPD'] = dataset_metric.statistical_parity_difference()
            except (ZeroDivisionError, ValueError):
                metrics[f'{phase}_SPD'] = 0.0

            # 如果有预测结果，计算AOD
            if phase != 'D' and y_pred is not None:
                try:
                    pred_dataset = to_aif360_dataset(X, y_pred, sensitive_features, self.sensitive_attr)
                    clf_metric = ClassificationMetric(
                        dataset, pred_dataset,
                        unprivileged_groups=[{self.sensitive_attr: 0}],
                        privileged_groups=[{self.sensitive_attr: 1}]
                    )
                    metrics[f'{phase}_AOD'] = clf_metric.average_odds_difference()
                except Exception as e:
                    warnings.warn(f"Failed to compute AOD: {e}")
                    metrics[f'{phase}_AOD'] = 0.0
            else:
                metrics[f'{phase}_AOD'] = 0.0

            # 个体公平性
            try:
                metrics[f'{phase}_Cons'] = dataset_metric.consistency()
            except Exception as e:
                warnings.warn(f"Failed to compute Consistency: {e}")
                metrics[f'{phase}_Cons'] = 0.0

            try:
                metrics[f'{phase}_TI'] = dataset_metric.theil_index()
            except Exception as e:
                warnings.warn(f"Failed to compute Theil Index: {e}")
                metrics[f'{phase}_TI'] = 0.0

        except ValueError as e:
            warnings.warn(f"Invalid data format for fairness metrics: {e}")
            # 使用默认值
            for metric_name in ['DI', 'SPD', 'AOD', 'Cons', 'TI']:
                metrics[f'{phase}_{metric_name}'] = 0.0
        except Exception as e:
            warnings.warn(f"Unexpected error computing fairness metrics: {e}")
            for metric_name in ['DI', 'SPD', 'AOD', 'Cons', 'TI']:
                metrics[f'{phase}_{metric_name}'] = 0.0

        # 3. 鲁棒性指标（精简版：仅模拟）
        if phase == 'Te':
            try:
                # 使用简单的随机扰动代替真实的对抗攻击
                metrics['A_FGSM'] = self._simple_robustness_test(X, y, epsilon=0.1)
                metrics['A_PGD'] = self._simple_robustness_test(X, y, epsilon=0.05)
            except Exception as e:
                warnings.warn(f"Failed to compute robustness metrics: {e}")
                metrics['A_FGSM'] = 0.0
                metrics['A_PGD'] = 0.0

        return metrics

    def _simple_robustness_test(self, X, y, epsilon=0.1):
        """
        简化的鲁棒性测试（不使用真实的对抗攻击）

        通过添加随机噪声测试模型的稳定性。
        注意：这不是真实的对抗攻击，仅用于快速评估。

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 真实标签，shape (n_samples,)
            epsilon: 噪声强度（标准差）

        Returns:
            float: 攻击成功率，范围 [0, 1]
        """
        # 添加随机噪声
        noise = np.random.normal(0, epsilon, X.shape)
        X_noisy = X + noise

        # 预测
        y_pred_original = self.model.predict(X)
        y_pred_noisy = self.model.predict(X_noisy)

        # 计算预测改变的比例（攻击成功率）
        success_rate = np.mean(y_pred_original != y_pred_noisy)

        return success_rate


def define_sign_functions():
    """
    定义所有指标的sign函数

    返回: Dict[str, Callable]
    """
    sign_funcs = {}

    # 性能指标：越高越好
    sign_funcs['Acc'] = lambda cur, change: '+' if change > 0 else '-'
    sign_funcs['F1'] = lambda cur, change: '+' if change > 0 else '-'

    # 公平性指标
    # DI: 理想值=1
    sign_funcs['DI'] = lambda cur, change: '+' if abs(cur + change - 1) < abs(cur - 1) else '-'
    # SPD/AOD: 理想值=0
    sign_funcs['SPD'] = lambda cur, change: '+' if abs(cur + change) < abs(cur) else '-'
    sign_funcs['AOD'] = lambda cur, change: '+' if abs(cur + change) < abs(cur) else '-'

    # 个体公平性
    sign_funcs['Cons'] = lambda cur, change: '+' if change > 0 else '-'
    sign_funcs['TI'] = lambda cur, change: '+' if change < 0 else '-'

    # 鲁棒性：越低越好
    sign_funcs['FGSM'] = lambda cur, change: '+' if change < 0 else '-'
    sign_funcs['PGD'] = lambda cur, change: '+' if change < 0 else '-'

    return sign_funcs
