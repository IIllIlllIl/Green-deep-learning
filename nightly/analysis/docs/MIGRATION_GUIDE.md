# 因果分析系统迁移指南

**目的**: 将本系统应用到新的数据集进行因果分析
**适用场景**: 机器学习公平性、性能权衡、超参数分析等
**更新时间**: 2025-12-21

---

## 📋 目录

1. [快速开始检查清单](#快速开始检查清单)
2. [数据集要求详解](#数据集要求详解)
3. [迁移步骤详解](#迁移步骤详解)
4. [配置文件定制](#配置文件定制)
5. [常见场景案例](#常见场景案例)
6. [注意事项与陷阱](#注意事项与陷阱)
7. [故障排查指南](#故障排查指南)

---

## 快速开始检查清单

### ✅ 在开始之前确认

**必须条件** (缺一不可):
```
□ 数据集包含敏感属性 (如性别、种族、年龄等)
□ 有明确的预测任务 (二分类或多分类)
□ 至少有 500+ 样本 (越多越好)
□ 特征已经是数值型或可以编码为数值型
□ 标签是离散的类别 (0/1 或多类)
```

**推荐条件** (提高效果):
```
□ 样本量 > 5000 (统计功效更高)
□ 有多个公平性敏感属性可选
□ 数据集已经过初步清洗 (无缺失值/异常值)
□ 有领域知识指导指标选择
□ 有GPU资源 (加速训练)
```

### 🎯 核心输入确认

在开始迁移前，您需要明确以下信息：

| 输入项 | 说明 | 示例 |
|--------|------|------|
| **数据来源** | CSV文件路径 | `data/my_dataset.csv` |
| **特征列** | 用于预测的列名列表 | `['age', 'income', 'education', ...]` |
| **标签列** | 预测目标列名 | `'approved'` (贷款是否批准) |
| **敏感属性** | 公平性关注的列 | `'gender'`, `'race'` |
| **特权类别** | 敏感属性的特权组 | `gender=Male`, `race=White` |
| **分析目标** | 想发现什么权衡 | `accuracy vs fairness` |

---

## 数据集要求详解

### 1. 数据格式要求

#### 1.1 文件格式

**支持的格式**:
```python
✅ CSV文件 (推荐)
✅ Pandas DataFrame
✅ NumPy数组 (需要额外处理)
⚠️ Excel文件 (需要转换为CSV)
❌ 图像/文本数据 (需要预先提取特征)
```

**CSV示例**:
```csv
id,age,income,education,gender,race,credit_score,approved
1,25,35000,Bachelor,Female,Asian,650,0
2,45,75000,Master,Male,White,720,1
3,33,52000,Bachelor,Female,Black,680,1
...
```

#### 1.2 特征类型要求

**数值型特征** (直接使用):
```python
age: [25, 45, 33, ...]           # 连续型
income: [35000, 75000, 52000]    # 连续型
credit_score: [650, 720, 680]    # 连续型
```

**分类特征** (需要编码):
```python
# 方式1: One-Hot编码 (推荐)
education: ['Bachelor', 'Master', 'PhD']
    ↓
education_Bachelor: [1, 0, 0]
education_Master: [0, 1, 0]
education_PhD: [0, 0, 1]

# 方式2: 标签编码 (谨慎使用)
education: ['Bachelor', 'Master', 'PhD']
    ↓
education_encoded: [0, 1, 2]  # 可能暗示顺序关系
```

**二值特征** (保持原样):
```python
is_student: [0, 1, 0, ...]  # 0=No, 1=Yes
```

#### 1.3 敏感属性要求

**必须是二值或可以二值化**:
```python
✅ 正确示例:
gender: [0, 1, 0, 1, ...]        # 0=Female, 1=Male
race: [0, 1, 0, 0, ...]          # 0=Minority, 1=Majority

⚠️ 需要处理:
gender: ['F', 'M', 'F', 'M']     # 需要映射为 0/1
race: ['Asian', 'White', 'Black'] # 需要二值化 (如 White vs Non-White)

❌ 不支持:
age_group: [1, 2, 3, 4]          # 多类别，需要选择二值分割
```

#### 1.4 标签要求

**分类任务**:
```python
✅ 二分类 (最常见):
y: [0, 1, 0, 1, ...]  # 0=Negative, 1=Positive

✅ 多分类 (需要转换):
y: [0, 1, 2]  # 3类
    ↓ 转换为一对多 (One-vs-Rest)
y_binary: [0, 1, 0]  # 类别1 vs 其他

❌ 回归任务 (当前不支持):
y: [35000.5, 72000.3, ...]  # 连续值
```

### 2. 数据规模要求

#### 2.1 样本量建议

| 样本量 | 效果评估 | 建议配置数 | DiBS迭代数 | 预期耗时 |
|--------|---------|-----------|-----------|---------|
| **< 500** | ❌ 不推荐 | - | - | - |
| **500 - 2K** | ⚠️ 勉强可行 | 6-10 | 2000 | ~30分钟 |
| **2K - 10K** | ✅ 良好 | 10-20 | 3000 | ~1-2小时 |
| **10K - 50K** | ✅ 很好 | 20-50 | 5000 | ~3-6小时 |
| **> 50K** | ✅ 优秀 | 50-100 | 5000-10000 | ~6-12小时 |

**Adult数据集参考** (45K样本):
- 配置数: 10
- DiBS迭代: 3000
- 总耗时: 61分钟

#### 2.2 特征数量建议

| 特征数 | 效果 | 注意事项 |
|--------|------|---------|
| **< 10** | ⚠️ 可能信息不足 | 增加特征工程 |
| **10 - 50** | ✅ 理想 | 平衡性能和计算成本 |
| **50 - 200** | ✅ 良好 | 需要GPU加速 |
| **> 200** | ⚠️ 可能过多 | 考虑特征选择/降维 |

**Adult数据集参考**: 102个特征 (One-Hot编码后)

#### 2.3 配置数量建议

**配置 = 方法 × 超参数组合**

```python
# 示例1: 基础配置
METHODS = ['Baseline', 'Reweighing']  # 2个方法
ALPHA_VALUES = [0.0, 0.5, 1.0]        # 3个alpha
总配置数 = 2 × 3 = 6个

# 示例2: 扩展配置
METHODS = ['Baseline', 'Reweighing', 'Adversarial']  # 3个方法
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]           # 5个alpha
MODEL_WIDTHS = [1, 2, 3]                             # 3个模型宽度
总配置数 = 3 × 5 × 3 = 45个

# 推荐配置数
最小: 6个 (快速验证)
标准: 10-20个 (平衡)
完整: 50-100个 (论文级别)
```

### 3. 数据质量要求

#### 3.1 缺失值处理

**检查缺失值**:
```python
import pandas as pd

df = pd.read_csv('your_data.csv')
missing_summary = df.isnull().sum()
print(missing_summary)

# 输出示例:
# age           0
# income        15    ← 有缺失
# education     0
# gender        3     ← 有缺失
```

**处理策略**:
```python
# 策略1: 删除缺失样本 (推荐，如果缺失<5%)
df_clean = df.dropna()

# 策略2: 填充数值特征 (均值/中位数)
df['income'].fillna(df['income'].median(), inplace=True)

# 策略3: 填充分类特征 (众数/新类别)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
# 或
df['gender'].fillna('Unknown', inplace=True)

# ⚠️ 不推荐: 复杂插补 (可能引入偏差)
```

**Adult数据集示例**:
```python
# 原始: 48,842样本
# 缺失: 3,620样本 (7.4%)
# 清洗后: 45,222样本
# 策略: 直接删除缺失行
```

#### 3.2 异常值检测

**数值特征异常值**:
```python
import numpy as np

# 方法1: IQR (四分位距) 方法
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值边界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
outliers = (df['income'] < lower_bound) | (df['income'] > upper_bound)
print(f"异常值数量: {outliers.sum()}")

# 处理: 删除或截断
df_clean = df[~outliers]  # 删除
# 或
df['income'] = df['income'].clip(lower_bound, upper_bound)  # 截断
```

**分类特征异常值**:
```python
# 检查罕见类别
value_counts = df['education'].value_counts()
print(value_counts)

# 输出示例:
# Bachelor     15000
# HS-grad      12000
# Master        8000
# PhD           3000
# Preschool       50  ← 罕见类别

# 处理: 合并罕见类别
rare_threshold = 100
rare_categories = value_counts[value_counts < rare_threshold].index
df['education'] = df['education'].replace(rare_categories, 'Other')
```

#### 3.3 类别不平衡

**检查标签分布**:
```python
label_dist = df['approved'].value_counts()
print(label_dist)
print(f"不平衡比例: {label_dist.max() / label_dist.min():.2f}")

# 输出示例:
# 0 (拒绝)    34000
# 1 (批准)    11000
# 不平衡比例: 3.09
```

**处理策略**:
```python
# 策略1: 什么都不做 (比例<5:1)
# 适用: Adult数据集 (3:1比例)

# 策略2: 重采样 (比例5:1 ~ 10:1)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 策略3: 加权损失 (比例>10:1)
# 在模型训练中自动处理
class_weights = {0: 1.0, 1: 3.0}  # 给少数类更高权重

# ⚠️ 注意: 本系统的Reweighing会自动平衡，无需额外处理
```

---

## 迁移步骤详解

### 步骤1: 数据准备与验证

#### 1.1 创建数据加载脚本

**创建文件**: `load_my_dataset.py`

```python
"""
自定义数据集加载脚本
替换Adult数据集为您的数据集
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_my_dataset():
    """
    加载并预处理您的数据集

    返回:
        X_train, X_test: 特征矩阵
        y_train, y_test: 标签向量
        sensitive_train, sensitive_test: 敏感属性
        feature_names: 特征名称列表
    """

    # === 1. 加载原始数据 ===
    print("加载数据...")
    df = pd.read_csv('data/my_dataset.csv')
    print(f"原始数据: {len(df)} 样本, {len(df.columns)} 列")

    # === 2. 处理缺失值 ===
    print("\n处理缺失值...")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # 删除缺失值 (或使用其他策略)
    df_clean = df.dropna()
    print(f"清洗后: {len(df_clean)} 样本 (删除 {len(df) - len(df_clean)} 行)")

    # === 3. 定义特征、标签、敏感属性 ===
    # ⚠️ 根据您的数据集修改这些列名
    label_col = 'approved'           # 预测目标
    sensitive_col = 'gender'          # 敏感属性

    # 要排除的列 (ID、标签、敏感属性等)
    exclude_cols = ['id', label_col, sensitive_col]

    # 特征列 = 所有列 - 排除列
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    print(f"\n特征列 ({len(feature_cols)}个): {feature_cols}")

    # === 4. 编码分类特征 ===
    print("\n编码分类特征...")

    # 识别分类列
    categorical_cols = df_clean[feature_cols].select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    print(f"分类特征: {categorical_cols}")

    # One-Hot编码
    df_encoded = pd.get_dummies(
        df_clean,
        columns=categorical_cols,
        drop_first=False  # 保留所有类别
    )

    # 更新特征列名 (因为One-Hot编码会改变列名)
    feature_cols = [col for col in df_encoded.columns
                   if col not in exclude_cols]

    print(f"编码后特征数: {len(feature_cols)}")

    # === 5. 提取数据 ===
    X = df_encoded[feature_cols].values
    y = df_encoded[label_col].values

    # === 6. 处理敏感属性 (必须是0/1) ===
    print("\n处理敏感属性...")

    if df_clean[sensitive_col].dtype == 'object':
        # 分类型 → 二值化
        unique_vals = df_clean[sensitive_col].unique()
        print(f"敏感属性唯一值: {unique_vals}")

        # ⚠️ 定义哪个是特权组 (privilege=1)
        privilege_group = 'Male'  # 根据您的数据修改

        sensitive = (df_clean[sensitive_col] == privilege_group).astype(int).values
        print(f"特权组 ({privilege_group}): {sensitive.sum()} 样本")
        print(f"非特权组: {len(sensitive) - sensitive.sum()} 样本")
    else:
        # 已经是数值型
        sensitive = df_clean[sensitive_col].values
        assert set(sensitive) <= {0, 1}, "敏感属性必须是0或1"

    # === 7. 处理标签 (必须是0/1) ===
    print("\n处理标签...")

    if y.dtype == 'object' or len(np.unique(y)) > 2:
        # 分类型或多类 → 二值化
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    print(f"标签分布: {np.bincount(y)}")

    # === 8. 数据分割 ===
    print("\n分割数据...")
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive,
        test_size=0.3,
        random_state=42,
        stratify=y  # 保持标签分布
    )

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # === 9. 特征标准化 ===
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === 10. 验证数据 ===
    print("\n数据验证...")
    assert X_train.shape[1] == X_test.shape[1], "训练集和测试集特征数不一致"
    assert not np.any(np.isnan(X_train)), "训练集包含NaN"
    assert not np.any(np.isnan(X_test)), "测试集包含NaN"
    assert set(y_train) <= {0, 1}, "标签必须是0或1"
    assert set(sensitive_train) <= {0, 1}, "敏感属性必须是0或1"

    print("✅ 数据验证通过")

    # === 11. 返回结果 ===
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sensitive_train,
        'sensitive_test': sensitive_test,
        'n_features': X_train.shape[1],
        'feature_names': feature_cols
    }

# 测试加载
if __name__ == '__main__':
    data = load_my_dataset()
    print("\n最终数据形状:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test: {data['X_test'].shape}")
    print(f"  特征数: {data['n_features']}")
```

#### 1.2 数据验证检查

**创建验证脚本**: `validate_data.py`

```python
"""
数据质量验证脚本
在正式训练前运行，确保数据符合要求
"""
import numpy as np
from load_my_dataset import load_my_dataset

def validate_dataset():
    """验证数据集是否符合要求"""

    print("="*70)
    print("数据集验证报告")
    print("="*70)

    # 加载数据
    data = load_my_dataset()

    X_train = data['X_train']
    y_train = data['y_train']
    sensitive_train = data['sensitive_train']

    # === 验证1: 样本量 ===
    print("\n1. 样本量检查")
    n_samples = len(X_train)
    print(f"   训练样本数: {n_samples}")

    if n_samples < 500:
        print("   ❌ 警告: 样本量太少 (<500), 结果可能不可靠")
    elif n_samples < 2000:
        print("   ⚠️  注意: 样本量较少 (<2000), 建议增加样本")
    else:
        print("   ✅ 样本量充足")

    # === 验证2: 特征数 ===
    print("\n2. 特征数检查")
    n_features = X_train.shape[1]
    print(f"   特征数: {n_features}")

    if n_features < 5:
        print("   ❌ 警告: 特征太少, 可能信息不足")
    elif n_features > 500:
        print("   ⚠️  注意: 特征很多, 考虑降维")
    else:
        print("   ✅ 特征数合理")

    # === 验证3: 数据类型 ===
    print("\n3. 数据类型检查")
    print(f"   X类型: {X_train.dtype}")
    print(f"   y类型: {y_train.dtype}")

    assert X_train.dtype in [np.float32, np.float64], "❌ X必须是浮点型"
    assert y_train.dtype in [np.int32, np.int64], "❌ y必须是整数型"
    print("   ✅ 数据类型正确")

    # === 验证4: 缺失值 ===
    print("\n4. 缺失值检查")
    n_missing = np.isnan(X_train).sum()
    print(f"   缺失值数量: {n_missing}")

    assert n_missing == 0, "❌ 发现缺失值, 请先处理"
    print("   ✅ 无缺失值")

    # === 验证5: 标签分布 ===
    print("\n5. 标签分布检查")
    label_counts = np.bincount(y_train)
    print(f"   类别0: {label_counts[0]} ({label_counts[0]/n_samples*100:.1f}%)")
    print(f"   类别1: {label_counts[1]} ({label_counts[1]/n_samples*100:.1f}%)")

    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"   不平衡比例: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 10:
        print("   ⚠️  警告: 严重不平衡, 考虑重采样")
    elif imbalance_ratio > 5:
        print("   ⚠️  注意: 中度不平衡")
    else:
        print("   ✅ 分布相对平衡")

    # === 验证6: 敏感属性分布 ===
    print("\n6. 敏感属性分布检查")
    sensitive_counts = np.bincount(sensitive_train)
    print(f"   非特权组 (0): {sensitive_counts[0]} ({sensitive_counts[0]/n_samples*100:.1f}%)")
    print(f"   特权组 (1): {sensitive_counts[1]} ({sensitive_counts[1]/n_samples*100:.1f}%)")

    sensitive_ratio = sensitive_counts.max() / sensitive_counts.min()
    print(f"   比例: {sensitive_ratio:.2f}:1")

    if sensitive_counts.min() < 100:
        print("   ⚠️  警告: 某组样本太少 (<100)")
    else:
        print("   ✅ 两组样本量充足")

    # === 验证7: 特征分布 ===
    print("\n7. 特征分布检查")
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)

    print(f"   均值范围: [{feature_means.min():.3f}, {feature_means.max():.3f}]")
    print(f"   标准差范围: [{feature_stds.min():.3f}, {feature_stds.max():.3f}]")

    # 检查是否标准化
    if np.abs(feature_means.mean()) < 0.1 and np.abs(feature_stds.mean() - 1.0) < 0.1:
        print("   ✅ 特征已标准化")
    else:
        print("   ⚠️  注意: 特征可能未标准化")

    # === 验证8: 常数特征 ===
    print("\n8. 常数特征检查")
    constant_features = (feature_stds < 1e-8).sum()
    print(f"   常数特征数: {constant_features}")

    if constant_features > 0:
        print("   ⚠️  警告: 发现常数特征, 应该移除")
    else:
        print("   ✅ 无常数特征")

    # === 验证9: 相关性检查 ===
    print("\n9. 特征相关性检查")
    if n_features < 100:  # 特征不太多时才检查
        corr_matrix = np.corrcoef(X_train.T)
        high_corr = (np.abs(corr_matrix) > 0.95) & (np.abs(corr_matrix) < 1.0)
        n_high_corr = high_corr.sum() // 2  # 除以2因为对称

        print(f"   高度相关特征对数: {n_high_corr} (相关系数>0.95)")

        if n_high_corr > n_features * 0.1:
            print("   ⚠️  警告: 过多高度相关特征, 考虑移除")
        else:
            print("   ✅ 特征相关性合理")
    else:
        print("   ⏭️  特征太多, 跳过相关性检查")

    # === 总结 ===
    print("\n" + "="*70)
    print("验证完成")
    print("="*70)
    print("\n如果所有检查都通过 (✅), 数据集可以用于训练")
    print("如果有警告 (⚠️), 建议先解决再继续")
    print("如果有错误 (❌), 必须修复才能继续")

    return True

if __name__ == '__main__':
    validate_dataset()
```

**运行验证**:
```bash
python validate_data.py
```

### 步骤2: 创建主实验脚本

#### 2.1 复制并修改模板

**创建文件**: `demo_my_dataset.py`

```python
"""
自定义数据集因果分析
基于 demo_adult_full_analysis.py 修改
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import torch
from datetime import datetime

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义数据加载器
from load_my_dataset import load_my_dataset

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator
from utils.fairness_methods import get_fairness_method

# ============================================================================
# 配置区 - ⚠️ 根据您的需求修改
# ============================================================================

# 数据集配置
DATASET_NAME = 'MyDataset'  # 修改为您的数据集名称

# 公平性方法配置
METHODS = ['Baseline', 'Reweighing']  # 可添加其他方法

# 超参数配置
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # 公平性权重

# 模型配置
EPOCHS = 50           # 训练轮数 (可减少以加快速度)
MODEL_WIDTH = 2       # 网络宽度倍数
BATCH_SIZE = 256      # 批次大小

# DiBS配置
DIBS_STEPS = 3000     # DiBS迭代次数 (样本多可增加到5000)

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 主程序
# ============================================================================

print("="*70)
print(f"  {DATASET_NAME} 完整因果分析")
print("="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

start_time = time.time()
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================================
# 步骤1: 加载数据
# ============================================================================
print("\n" + "="*70)
print("  步骤1: 加载数据")
print("="*70)

data = load_my_dataset()
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
sensitive_train = data['sensitive_train']
sensitive_test = data['sensitive_test']
n_features = data['n_features']

print(f"✅ 数据加载完成")
print(f"  训练集: {len(X_train)} 样本")
print(f"  测试集: {len(X_test)} 样本")
print(f"  特征数: {n_features}")

# ============================================================================
# 步骤2: 数据收集
# ============================================================================
print("\n" + "="*70)
print("  步骤2: 数据收集")
print("="*70)

results = []
total_configs = len(METHODS) * len(ALPHA_VALUES)

for idx, (method_name, alpha) in enumerate(
    [(m, a) for m in METHODS for a in ALPHA_VALUES], 1
):
    config_start = time.time()
    print(f"\n  [{idx}/{total_configs}] {method_name}, α={alpha:.2f}")

    try:
        # 应用公平性方法
        method = get_fairness_method(
            method_name, alpha,
            sensitive_attr='sensitive'  # ⚠️ 修改为您的敏感属性名
        )
        X_transformed, y_transformed = method.fit_transform(
            X_train, y_train, sensitive_train
        )

        # 训练模型
        model = FFNN(input_dim=n_features, width=MODEL_WIDTH)
        trainer = ModelTrainer(model, device=device, lr=0.001)
        trainer.train(
            X_transformed, y_transformed,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=False
        )

        # 计算指标
        calculator = MetricsCalculator(
            trainer,
            sensitive_attr='sensitive'  # ⚠️ 修改为您的敏感属性名
        )

        dataset_metrics = calculator.compute_all_metrics(
            X_train, y_train, sensitive_train, phase='D'
        )
        train_metrics = calculator.compute_all_metrics(
            X_transformed, y_transformed, sensitive_train, phase='Tr'
        )
        test_metrics = calculator.compute_all_metrics(
            X_test, y_test, sensitive_test, phase='Te'
        )

        # 收集结果
        row = {
            'method': method_name,
            'alpha': alpha,
            'Width': MODEL_WIDTH
        }
        row.update(dataset_metrics)
        row.update(train_metrics)
        row.update(test_metrics)
        results.append(row)

        # 显示进度
        elapsed = time.time() - start_time
        eta = (total_configs - idx) * (elapsed / idx) / 60
        print(f"    ✓ Acc={test_metrics.get('Te_Acc', 0):.3f} | "
              f"耗时={time.time()-config_start:.0f}s | ETA={eta:.1f}min")

    except Exception as e:
        print(f"    ✗ 失败: {e}")
        continue

# 保存数据
df = pd.DataFrame(results)
output_path = f'data/{DATASET_NAME.lower()}_training_data.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ 数据收集完成，保存到: {output_path}")

# ============================================================================
# 步骤3: DiBS因果图学习
# ============================================================================
print("\n" + "="*70)
print("  步骤3: DiBS因果图学习")
print("="*70)

try:
    from utils.causal_discovery import CausalGraphLearner

    # 准备数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Width' in numeric_cols:
        numeric_cols.remove('Width')

    causal_data = df[numeric_cols]

    print(f"  变量数: {len(numeric_cols)}")
    print(f"  数据点: {len(causal_data)}")

    # 创建学习器
    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=DIBS_STEPS,
        alpha=0.1,
        random_seed=42
    )

    # 学习因果图
    print(f"\n  开始DiBS学习...")
    dibs_start = time.time()

    causal_graph = learner.fit(causal_data, verbose=True)

    print(f"\n  ✓ DiBS完成，耗时: {(time.time()-dibs_start)/60:.1f}分钟")

    # 分析边
    edges = learner.get_edges(threshold=0.3)
    print(f"  检测到 {len(edges)} 条因果边")

    # 保存结果
    graph_path = f'results/{DATASET_NAME.lower()}_causal_graph.npy'
    learner.save_graph(graph_path)
    print(f"  ✓ 因果图已保存到: {graph_path}")

    # 显示关键边
    if len(edges) > 0:
        print(f"\n  前10条最强因果边:")
        for i, (source, target, weight) in enumerate(edges[:10], 1):
            print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")

except Exception as e:
    print(f"  ✗ DiBS失败: {e}")
    import traceback
    traceback.print_exc()
    causal_graph = None
    edges = []

# ============================================================================
# 步骤4: DML因果推断
# ============================================================================
if causal_graph is not None and len(edges) > 0:
    print("\n" + "="*70)
    print("  步骤4: DML因果推断")
    print("="*70)

    try:
        from utils.causal_inference import CausalInferenceEngine

        engine = CausalInferenceEngine(verbose=True)

        print(f"\n  开始DML分析...")
        dml_start = time.time()

        causal_effects = engine.analyze_all_edges(
            data=causal_data,
            causal_graph=causal_graph,
            var_names=numeric_cols,
            threshold=0.3
        )

        print(f"\n  ✓ DML完成，耗时: {(time.time()-dml_start)/60:.1f}分钟")

        if causal_effects:
            effects_path = f'results/{DATASET_NAME.lower()}_causal_effects.csv'
            engine.save_results(effects_path)
            print(f"  ✓ 因果效应已保存到: {effects_path}")

            significant = engine.get_significant_effects()
            print(f"\n  因果效应统计:")
            print(f"    总边数: {len(causal_effects)}")
            print(f"    统计显著: {len(significant)}")

            if significant:
                print(f"\n  显著的因果效应 (前5个):")
                for i, (edge, result) in enumerate(list(significant.items())[:5], 1):
                    print(f"    {i}. {edge}")
                    print(f"       ATE={result['ate']:.4f}, "
                          f"95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    except Exception as e:
        print(f"  ✗ DML失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*70)
print("  分析完成！")
print("="*70)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")

print(f"\n生成的文件:")
for file in [output_path, graph_path]:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"  ✓ {file} ({size:.1f} KB)")

print("\n" + "="*70)
```

### 步骤3: 运行实验

#### 3.1 小规模测试

**先用少量配置测试**:

```python
# 在 demo_my_dataset.py 中临时修改:
METHODS = ['Baseline']  # 只测试1个方法
ALPHA_VALUES = [0.0, 1.0]  # 只测试2个alpha
EPOCHS = 10  # 减少训练轮数

# 运行
python demo_my_dataset.py
```

**预期输出**:
```
======================================================================
  MyDataset 完整因果分析
======================================================================
开始时间: 2025-12-21 18:00:00
设备: cuda
GPU: NVIDIA GeForce RTX 3080

======================================================================
  步骤1: 加载数据
======================================================================
加载数据...
原始数据: 10000 样本, 25 列
...
✅ 数据加载完成
  训练集: 7000 样本
  测试集: 3000 样本
  特征数: 50

======================================================================
  步骤2: 数据收集
======================================================================

  [1/2] Baseline, α=0.00
    ✓ Acc=0.756 | 耗时=45s | ETA=0.8min

  [2/2] Baseline, α=1.00
    ✓ Acc=0.752 | 耗时=43s | ETA=0.0min

✓ 数据收集完成，保存到: data/mydataset_training_data.csv
...
```

#### 3.2 完整实验

**确认测试成功后，运行完整实验**:

```python
# 恢复完整配置
METHODS = ['Baseline', 'Reweighing']
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCHS = 50

# 使用后台运行
nohup python demo_my_dataset.py > my_experiment.log 2>&1 &

# 监控进度
tail -f my_experiment.log
```

---

## 配置文件定制

### 方法选择

#### 可用的公平性方法

```python
# 预处理方法
METHODS = [
    'Baseline',       # 不做任何处理（基准）
    'Reweighing',     # 样本重加权
    'Sampling',       # 重采样（过采样+欠采样）
]

# 处理中方法（需要额外实现）
METHODS = [
    'Adversarial',    # 对抗去偏
    'PrejudiceRemover',  # 偏见移除
]

# 后处理方法（需要额外实现）
METHODS = [
    'Calibration',    # 校准
    'RejectOption',   # 拒绝选项分类
]
```

### 超参数网格

#### Alpha参数（公平性强度）

```python
# 粗粒度搜索
ALPHA_VALUES = [0.0, 0.5, 1.0]  # 3个点

# 标准搜索
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5个点

# 细粒度搜索
ALPHA_VALUES = [0.0, 0.1, 0.2, ..., 0.9, 1.0]  # 11个点

# 对数搜索（如果效应非线性）
ALPHA_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0]
```

#### 模型宽度（容量）

```python
# 单一宽度
MODEL_WIDTH = 2

# 多宽度对比
MODEL_WIDTHS = [1, 2, 3]  # 浅 → 中 → 深

# 嵌套循环
for width in MODEL_WIDTHS:
    for method in METHODS:
        for alpha in ALPHA_VALUES:
            # 训练配置...
```

### DiBS参数调优

```python
# 样本量 < 20
DIBS_STEPS = 2000
DIBS_ALPHA = 0.2  # 更强稀疏性

# 样本量 20-50
DIBS_STEPS = 3000
DIBS_ALPHA = 0.1  # 标准

# 样本量 > 50
DIBS_STEPS = 5000
DIBS_ALPHA = 0.05  # 更弱稀疏性
```

---

## 常见场景案例

### 场景1: 信贷审批公平性

**数据特征**:
- 样本量: 10,000
- 特征: 年龄、收入、信用分、教育等 (15个)
- 标签: 是否批准贷款 (0/1)
- 敏感属性: 性别 (Female/Male)

**配置**:
```python
DATASET_NAME = 'CreditApproval'
METHODS = ['Baseline', 'Reweighing']
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EPOCHS = 50
DIBS_STEPS = 3000

# 预期权衡: approval rate vs fairness
```

**预期结果**:
- DiBS发现: alpha → Te_SPD, alpha → Te_Acc
- DML量化: ATE(alpha → Te_SPD) > 0, ATE(alpha → Te_Acc) < 0
- 权衡: 提高公平性降低批准准确率

### 场景2: 招聘系统偏见

**数据特征**:
- 样本量: 5,000
- 特征: 教育、经验、技能评分等 (20个)
- 标签: 是否录用 (0/1)
- 敏感属性: 种族 (Minority/Majority)

**配置**:
```python
DATASET_NAME = 'HiringDecision'
METHODS = ['Baseline', 'Reweighing', 'Adversarial']
ALPHA_VALUES = [0.0, 0.5, 1.0]
EPOCHS = 30  # 样本量较少
DIBS_STEPS = 2000

# 预期权衡: hiring quality vs demographic parity
```

### 场景3: 医疗诊断公平性

**数据特征**:
- 样本量: 20,000
- 特征: 年龄、BMI、血压、检查结果等 (30个)
- 标签: 是否患病 (0/1)
- 敏感属性: 年龄组 (Young/Old)

**配置**:
```python
DATASET_NAME = 'MedicalDiagnosis'
METHODS = ['Baseline', 'Reweighing']
ALPHA_VALUES = np.linspace(0, 1, 11)  # 细粒度
EPOCHS = 50
DIBS_STEPS = 5000  # 样本量大

# 预期权衡: diagnostic accuracy vs age fairness
```

---

## 注意事项与陷阱

### ⚠️ 陷阱1: 测试集指标不变

**现象**:
```
所有配置的 Te_SPD 都相同
所有配置的 Te_DI 都相同
```

**原因**:
- Reweighing等方法只处理**训练集**
- 测试集保持原样不变
- 因此测试集的公平性指标也不变

**解决**:
```python
# 观察训练集指标的变化
print(df[['method', 'alpha', 'Tr_SPD', 'Tr_DI']])

# 正确期望:
# - Tr_SPD 应该随alpha变化
# - Te_SPD 保持不变（这是正常的）
```

### ⚠️ 陷阱2: 样本量太少导致DiBS失败

**现象**:
```
DiBS学习出的图完全稀疏（无边）
或完全稠密（全连接）
```

**原因**:
- 样本量 < 10: 统计功效不足
- DiBS无法可靠估计因果关系

**解决**:
```python
# 方案1: 增加配置数
ALPHA_VALUES = np.linspace(0, 1, 20)  # 从5个增加到20个

# 方案2: 增加模型多样性
for width in [1, 2, 3]:
    for method in METHODS:
        for alpha in ALPHA_VALUES:
            # ...

# 方案3: 降低DiBS稀疏性惩罚
DIBS_ALPHA = 0.05  # 从0.1降低（允许更多边）
```

### ⚠️ 陷阱3: 特征未标准化

**现象**:
```
模型训练不收敛
损失函数NaN
准确率随机波动
```

**原因**:
- 特征尺度差异大（如age=25, income=50000）
- 梯度爆炸/消失

**解决**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 验证
print(f"均值: {X_train.mean():.3f}")  # 应接近0
print(f"标准差: {X_train.std():.3f}")  # 应接近1
```

### ⚠️ 陷阱4: 分类特征编码错误

**现象**:
```
模型性能异常差
公平性指标无意义
```

**错误示例**:
```python
# ❌ 错误: 标签编码暗示顺序
education = ['HS', 'Bachelor', 'Master', 'PhD']
education_encoded = [0, 1, 2, 3]  # 模型认为PhD=3×HS
```

**正确方法**:
```python
# ✅ 正确: One-Hot编码
df_encoded = pd.get_dummies(df, columns=['education'])

# 结果:
# education_HS:       [1, 0, 0, 0]
# education_Bachelor: [0, 1, 0, 0]
# education_Master:   [0, 0, 1, 0]
# education_PhD:      [0, 0, 0, 1]
```

### ⚠️ 陷阱5: GPU内存不足

**现象**:
```
CUDA out of memory
RuntimeError: CUDA error
```

**原因**:
- 批次大小太大
- 模型太大
- 多个进程共享GPU

**解决**:
```python
# 方案1: 减小批次
BATCH_SIZE = 128  # 从256降低

# 方案2: 减小模型
MODEL_WIDTH = 1  # 从2降低

# 方案3: 清理GPU缓存
import torch
torch.cuda.empty_cache()

# 方案4: 使用CPU
device = 'cpu'  # 放弃GPU加速
```

### ⚠️ 陷阱6: 因果图过于复杂

**现象**:
```
DiBS检测到数百条边
图密度 > 0.5
```

**原因**:
- DiBS稀疏性惩罚太弱
- 样本量不足导致过拟合

**解决**:
```python
# 增加稀疏性惩罚
learner = CausalGraphLearner(
    n_vars=len(numeric_cols),
    n_steps=3000,
    alpha=0.3,  # 从0.1增加到0.3
    random_seed=42
)

# 或提高阈值
edges = learner.get_edges(threshold=0.5)  # 从0.3提高
```

---

## 故障排查指南

### 问题诊断流程

```
1. 检查数据加载
   ├─ 运行 validate_data.py
   ├─ 确认样本量、特征数、标签分布
   └─ 如果失败 → 修复数据加载脚本

2. 检查模型训练
   ├─ 运行1个配置测试
   ├─ 观察损失函数曲线
   └─ 如果失败 → 调整学习率/网络结构

3. 检查指标计算
   ├─ 打印中间结果
   ├─ 确认AIF360兼容性
   └─ 如果失败 → 检查敏感属性编码

4. 检查DiBS学习
   ├─ 查看收敛曲线
   ├─ 检查边的数量和分布
   └─ 如果失败 → 调整超参数

5. 检查DML推断
   ├─ 查看每条边的估计
   ├─ 检查置信区间是否合理
   └─ 如果失败 → 增加样本量
```

### 常见错误信息

**错误1: KeyError: 'is_significant'**
```
原因: DML保存结果时缺少字段
解决: 已在最新代码中修复，更新 utils/causal_inference.py
```

**错误2: ValueError: could not convert string to float**
```
原因: 分类特征未编码
解决: 使用 pd.get_dummies() 编码所有分类列
```

**错误3: AssertionError: 敏感属性必须是0或1**
```
原因: 敏感属性不是二值
解决: 在load_my_dataset.py中添加二值化逻辑
```

**错误4: RuntimeError: CUDA out of memory**
```
原因: GPU内存不足
解决: 减小BATCH_SIZE或MODEL_WIDTH
```

**错误5: np.linalg.LinAlgError: Singular matrix**
```
原因: DML中的协方差矩阵奇异（变量缺乏变异性）
解决: 检查是否所有配置的某些指标完全相同
```

---

## 总结

### 迁移检查清单

**准备阶段**:
```
□ 数据集满足最低要求（>500样本）
□ 敏感属性是二值或可二值化
□ 标签是分类型（0/1）
□ 特征已清洗（无缺失值）
□ 完成数据验证脚本
```

**实施阶段**:
```
□ 创建 load_my_dataset.py
□ 创建 validate_data.py 并通过
□ 创建 demo_my_dataset.py
□ 运行小规模测试（2-3个配置）
□ 确认结果合理后运行完整实验
```

**验证阶段**:
```
□ 检查数据收集结果（CSV文件）
□ 检查DiBS学习结果（边的数量和意义）
□ 检查DML推断结果（统计显著性）
□ 生成分析报告
□ 与领域知识对比验证
```

### 关键成功因素

1. **数据质量** > 算法复杂度
2. **充足样本** > 复杂模型
3. **领域知识** > 盲目调参
4. **小步验证** > 一次到位
5. **耐心调试** > 快速放弃

---

**文档版本**: v1.0
**最后更新**: 2025-12-21
**适用系统版本**: 基于Adult数据集完整因果分析
