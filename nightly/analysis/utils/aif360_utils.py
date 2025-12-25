"""
AIF360工具函数 - 提取重复代码
"""
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from typing import Tuple


def to_aif360_dataset(X: np.ndarray, y: np.ndarray,
                     sensitive_features: np.ndarray,
                     sensitive_attr: str = 'sex') -> BinaryLabelDataset:
    """
    通用的AIF360数据集转换函数

    Args:
        X: 特征数组，shape (n_samples, n_features)
        y: 标签数组，shape (n_samples,)
        sensitive_features: 敏感属性数组，shape (n_samples,)
        sensitive_attr: 敏感属性名称

    Returns:
        BinaryLabelDataset

    Raises:
        ValueError: 如果输入数据维度不匹配
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
            f"Shape mismatch: X={len(X)}, y={len(y)}, sensitive={len(sensitive_features)}"
        )

    # 创建DataFrame
    n_features = X.shape[1]
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
    df['label'] = y
    df[sensitive_attr] = sensitive_features

    # 转换为BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr],
        favorable_label=1.0,
        unfavorable_label=0.0
    )

    return dataset


def from_aif360_dataset(dataset: BinaryLabelDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    从AIF360数据集格式转回numpy数组

    Args:
        dataset: AIF360 BinaryLabelDataset

    Returns:
        Tuple[X, y]: 特征数组和标签数组
    """
    df = dataset.convert_to_dataframe()[0]

    # 提取特征
    feature_cols = [col for col in df.columns if col.startswith('f')]
    X = df[feature_cols].values

    # 提取标签
    y = df['label'].values

    return X, y
