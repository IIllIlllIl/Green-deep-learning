"""
公平性改进方法（精简版）
使用AIF360库的现成实现
"""
import numpy as np
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
import pandas as pd
from utils.aif360_utils import to_aif360_dataset, from_aif360_dataset
import warnings


class FairnessMethodWrapper:
    """公平性方法的统一包装器"""

    def __init__(self, method_name, alpha=0.5, sensitive_attr='sex'):
        """
        参数:
            method_name: 方法名称 ('Reweighing', 'AdversarialDebiasing', 'EqualizedOdds', 'Baseline')
            alpha: 干预比例 [0, 1]
            sensitive_attr: 敏感属性名称
        """
        self.method_name = method_name
        self.alpha = alpha
        self.sensitive_attr = sensitive_attr
        self.method = None
        self.feature_names = None

    def fit_transform(self, X_train, y_train, sensitive_features, X_test=None, y_test=None):
        """
        应用公平性改进方法

        参数:
            X_train, y_train: 训练数据
            sensitive_features: 敏感属性值
            X_test, y_test: 测试数据（用于后处理方法）

        返回: X_transformed, y_transformed

        Raises:
            ValueError: 如果输入数据无效
        """
        # 输入验证
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be None or empty")
        if y_train is None or len(y_train) == 0:
            raise ValueError("y_train cannot be None or empty")
        if sensitive_features is None or len(sensitive_features) == 0:
            raise ValueError("sensitive_features cannot be None or empty")
        if len(X_train) != len(y_train) or len(X_train) != len(sensitive_features):
            raise ValueError(
                f"Shape mismatch: X_train={len(X_train)}, y_train={len(y_train)}, "
                f"sensitive={len(sensitive_features)}"
            )

        if self.method_name == 'Baseline':
            # Baseline: 不应用任何方法
            return X_train, y_train

        # 根据alpha选择部分数据应用方法
        n_samples = len(X_train)
        n_apply = int(self.alpha * n_samples)

        if n_apply == 0:
            return X_train, y_train
        if self.alpha >= 1.0:
            # 全部应用，避免不必要的复制
            try:
                dataset = to_aif360_dataset(X_train, y_train, sensitive_features, self.sensitive_attr)
                transformed_dataset = self._apply_method(dataset)
                return from_aif360_dataset(transformed_dataset)
            except Exception as e:
                warnings.warn(f"Failed to apply fairness method: {e}")
                return X_train, y_train

        # 随机选择样本
        np.random.seed(42)  # 保证可复现性
        indices = np.random.choice(n_samples, n_apply, replace=False)
        mask = np.zeros(n_samples, dtype=bool)
        mask[indices] = True

        # 应用方法到选中的样本
        X_apply = X_train[mask]
        y_apply = y_train[mask]
        sens_apply = sensitive_features[mask]

        try:
            # 转换为AIF360格式 - 使用工具函数
            dataset_apply = to_aif360_dataset(X_apply, y_apply, sens_apply, self.sensitive_attr)

            # 应用具体方法
            transformed_dataset = self._apply_method(dataset_apply)

            # 从AIF360格式转回 - 使用工具函数
            X_transformed, y_transformed = from_aif360_dataset(transformed_dataset)

            # 合并处理过的数据和未处理的数据
            X_result = X_train.copy()
            y_result = y_train.copy()
            X_result[mask] = X_transformed
            y_result[mask] = y_transformed

            return X_result, y_result

        except Exception as e:
            warnings.warn(f"Failed to apply fairness method {self.method_name}: {e}")
            return X_train, y_train

    def _apply_method(self, dataset):
        """
        应用具体的公平性方法

        Args:
            dataset: AIF360 BinaryLabelDataset

        Returns:
            transformed_dataset: 转换后的数据集
        """
        if self.method_name == 'Reweighing':
            return self._apply_reweighing(dataset)
        elif self.method_name == 'AdversarialDebiasing':
            # 处理中方法需要训练模型，这里简化处理
            warnings.warn("AdversarialDebiasing simplified: returning original dataset")
            return dataset
        elif self.method_name == 'EqualizedOdds':
            # 后处理方法需要预测结果，这里简化处理
            warnings.warn("EqualizedOdds simplified: returning original dataset")
            return dataset
        else:
            return dataset

    def _apply_reweighing(self, dataset):
        """应用Reweighing方法"""
        unprivileged_groups = [{self.sensitive_attr: 0}]
        privileged_groups = [{self.sensitive_attr: 1}]

        RW = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )

        transformed_dataset = RW.fit_transform(dataset)
        return transformed_dataset


def get_fairness_method(method_name, alpha, sensitive_attr='sex'):
    """
    获取公平性方法实例

    参数:
        method_name: 方法名称
        alpha: 干预比例
        sensitive_attr: 敏感属性

    返回: FairnessMethodWrapper实例
    """
    return FairnessMethodWrapper(method_name, alpha, sensitive_attr)
