# 能耗DL项目 - 数据使用主指南

**文档版本**: 1.0
**创建日期**: 2026-01-15
**最后更新**: 2026-01-15
**状态**: ✅ 当前有效

> **重要**: 这是项目数据使用的**唯一权威指南**。所有数据处理工作应首先参考本文档。

---

## 📑 目录

1. [快速开始](#快速开始)
2. [⚠️ 关键注意事项](#关键注意事项)
3. [数据文件选择](#数据文件选择)
4. [数据质量现状](#数据质量现状)
5. [数据处理最佳实践](#数据处理最佳实践)
6. [常见错误和解决方案](#常见错误和解决方案)
7. [参考文档](#参考文档)

---

## 快速开始

### 推荐使用方案

| 使用场景 | 推荐文件 | 理由 |
|---------|---------|------|
| **✅ 回归分析** | `data/data.csv` | 统一格式，无需处理并行/非并行差异 |
| **✅ 因果分析** | `data/data.csv` | 统一格式，适合DiBS等因果工具 |
| **✅ 性能分析** | `data/data.csv` | 字段清晰，易于处理 |
| **⚠️ 最大完整性需求** | `data/raw_data.csv` | 970行（data.csv有726行），但需特殊处理 |
| **⚠️ 调试和验证** | `data/raw_data.csv` | 包含所有原始字段，便于问题排查 |

### 快速加载代码

#### 推荐：使用 data.csv

```python
import pandas as pd

# 读取数据（推荐）
df = pd.read_csv('data/data.csv')

# 基本检查
print(f"总记录数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(f"唯一timestamp: {df['timestamp'].nunique()}")

# 验证唯一性
assert df['timestamp'].nunique() == len(df), "timestamp 应该是唯一的！"

# 使用数据
# 直接访问字段，无需考虑 fg_ 前缀
learning_rate = df['learning_rate']
batch_size = df['batch_size']
energy = df['energy_gpu_avg']
```

#### 备选：使用 raw_data.csv

```python
import pandas as pd

# 读取数据（需要特殊处理）
df = pd.read_csv('data/raw_data.csv')

# ⚠️ 必读：参考 RAW_DATA_CSV_USAGE_GUIDE.md 获取完整处理方法

def get_field(row, field_name):
    """
    智能获取字段值，自动处理并行/非并行模式

    Args:
        row: DataFrame行
        field_name: 字段名（不带fg_前缀）

    Returns:
        字段值
    """
    # 检查是否是并行模式
    if pd.notna(row.get('fg_learning_rate', None)):
        # 并行模式：使用 fg_ 前缀
        return row.get(f'fg_{field_name}', row.get(field_name, ''))
    else:
        # 非并行模式：直接使用字段名
        return row.get(field_name, '')

# 应用到所有行
df['learning_rate_unified'] = df.apply(lambda row: get_field(row, 'learning_rate'), axis=1)
```

---

## ⚠️ 关键注意事项

### 🚨 极其重要：唯一标识符

**错误认识** ❌:
```python
# ❌ 错误！experiment_id 不是唯一的
df_unique = df.drop_duplicates(subset=['experiment_id'])
# 这会丢失数据！
```

**正确认识** ✅:
```python
# ✅ 正确！timestamp 是唯一键
df_unique = df.drop_duplicates(subset=['timestamp'])

# ✅ 或使用复合键（raw_data.csv中）
df_unique = df.drop_duplicates(subset=['experiment_id', 'timestamp'])
```

**原因说明**:
- **experiment_id**: 代表实验**配置**（可以运行多次）
  - 例如：`default__mnist_default_001` 可以在不同时间运行多次
  - 不同轮次（runs）会复用相同的 experiment_id

- **timestamp**: 代表实验**运行实例**（应该唯一）
  - 例如：`2025-11-18T20:37:37.187907`
  - 每次运行产生唯一的时间戳

**详细说明**: 参考 [analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md](../analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md) ⭐⭐⭐

### 🚨 重要：并行模式数据处理

**data.csv** ✅:
- 已自动合并并行/非并行字段
- 添加了 `is_parallel` 列标识模式
- **直接使用字段名**，无需考虑 `fg_` 前缀

**raw_data.csv** ⚠️:
- 并行模式数据在 `fg_` 前缀字段中
- 非并行模式数据在顶层字段中
- **必须**根据模式选择正确的字段
- **强烈推荐**阅读: [RAW_DATA_CSV_USAGE_GUIDE.md](RAW_DATA_CSV_USAGE_GUIDE.md) ⭐⭐⭐⭐⭐

### 🚨 重要：空字符串 vs 缺失值

在 raw_data.csv 中：
- **空字符串 `""`**: 表示使用默认值或不适用（不是数据缺失）
- **NaN/空**: 表示真正的数据缺失

```python
# 错误的判断方式 ❌
missing = df[df['learning_rate'] == '']  # 这可能不是缺失！

# 正确的判断方式 ✅
import pandas as pd
missing = df[df['learning_rate'].isna()]  # 真正的缺失值
has_default = df[df['learning_rate'] == '']  # 使用默认值
```

---

## 数据文件选择

### data.csv vs raw_data.csv 详细对比

| 特征 | data.csv ✅ 推荐 | raw_data.csv ⚠️ 备选 |
|------|-----------------|---------------------|
| **记录数** | 726行 | 970行 |
| **可用记录** | 692行 (95.3%) | 577行 (59.5%) |
| **字段数** | 56列 | 87列 |
| **格式统一性** | ✅ 完全统一 | ⚠️ 需要特殊处理 |
| **并行模式处理** | ✅ 已自动合并 | ❌ 需要手动处理 |
| **is_parallel标识** | ✅ 有 | ❌ 无 |
| **timestamp唯一性** | ✅ 100%唯一 | ✅ 100%唯一 (去重后) |
| **适用场景** | 回归分析、因果分析、通用分析 | 调试、最大完整性需求 |
| **上手难度** | ⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**详细对比**: 参考 [analysis/docs/DATA_FILES_COMPARISON.md](../analysis/docs/DATA_FILES_COMPARISON.md) ⭐⭐⭐

### 什么时候使用 raw_data.csv？

**仅在以下情况下使用**:
1. 需要**最大数据完整性**（970行 vs 726行）
2. 需要**所有原始字段**（87列 vs 56列）
3. 需要**调试数据问题**或验证数据处理流程
4. 需要访问**特殊字段**（如某些未合并的字段）

**使用前必读**: [RAW_DATA_CSV_USAGE_GUIDE.md](RAW_DATA_CSV_USAGE_GUIDE.md) ⭐⭐⭐⭐⭐

---

## 数据质量现状

### 最新统计（2026-01-15更新）

#### data.csv 数据质量 ✅ 优秀

| 指标 | 数值 | 百分比 |
|------|------|--------|
| **总记录数** | 726 | 100% |
| **✅ 可用记录** | 692 | **95.3%** |
| **训练成功** | 726 | **100%** |
| **有能耗数据** | 692 | **95.3%** |
| **有性能指标** | 726 | **100%** |
| **timestamp唯一** | 726 | **100%** |

**结论**: data.csv 数据质量优秀，**强烈推荐用于所有分析工作** ✅

#### raw_data.csv 数据质量 ⚠️ 混合

| 指标 | 数值 | 百分比 |
|------|------|--------|
| **总记录数** | 970 | 100% |
| **✅ 可用记录** | 577 | **59.5%** |
| **❌ 不可用记录** | 393 | **40.5%** |
| **训练成功** | 854 | 88.0% |
| **有能耗数据** | 828 | 85.4% |
| **有性能指标** | 577 | 59.5% |
| **timestamp唯一** | 970 | **100%** (去重后) |

**不可用原因分析**:
- 🚨 **性能指标缺失**: 393条 (100%不可用记录)
- ⚡ **能耗数据缺失**: 142条 (36.1%不可用记录)
- ❌ **训练失败**: 116条 (29.5%不可用记录)

**结论**: raw_data.csv 有更多记录但质量较低，仅在需要最大完整性时使用 ⚠️

### 推荐使用的高质量数据

**8个100%可用模型** (487条记录，来自data.csv):
- `pytorch_resnet_cifar10/resnet20` (53条)
- `Person_reID_baseline_pytorch/densenet121` (53条)
- `Person_reID_baseline_pytorch/hrnet18` (53条)
- `Person_reID_baseline_pytorch/pcb` (53条)
- `examples/mnist` (69条)
- `examples/mnist_rnn` (69条)
- `examples/siamese` (69条)
- `examples/mnist_ff` (68条)

**详细分析**: 参考 [DATA_USABILITY_FOR_REGRESSION_20260114.md](DATA_USABILITY_FOR_REGRESSION_20260114.md) ⭐

### 6分组回归分析数据可用性

针对研究问题1（超参数对能耗的影响），数据按模型特征分为6组：

| 组别 | 模型 | 可用记录 | 占比 |
|------|------|---------|------|
| **Group 1** | Image Classification (3模型) | 159 | 100% |
| **Group 2** | Image-based Matching (3模型) | 207 | 100% |
| **Group 3** | Text-based Tasks (2模型) | 0 | 0% ⚠️ |
| **Group 4** | Defect Localization (1模型) | 25 | 100% |
| **Group 5** | Bug Localization (1模型) | 0 | 0% ⚠️ |
| **Group 6** | Vulnerability Detection (1模型) | 0 | 0% ⚠️ |

**可用组别**: Group 1, 2, 4 **(3/6组，391条记录)**

**详细报告**: [DATA_USABILITY_FOR_REGRESSION_20260114.md](DATA_USABILITY_FOR_REGRESSION_20260114.md)

---

## 数据处理最佳实践

### 1. 去重处理

```python
import pandas as pd

# 读取数据
df = pd.read_csv('data/data.csv')

# 方法1: 基于 timestamp 去重（推荐）
df_unique = df.drop_duplicates(subset=['timestamp'], keep='first')

# 验证
print(f"原始行数: {len(df)}")
print(f"去重后行数: {len(df_unique)}")
print(f"移除行数: {len(df) - len(df_unique)}")

# 检查唯一性
assert df_unique['timestamp'].nunique() == len(df_unique), "去重失败！"
```

**工具脚本**: `tools/data_management/deduplicate_by_timestamp.py`

### 2. 缺失值处理

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/data.csv')

# 检查缺失值
print("缺失值统计:")
print(df.isnull().sum())

# 能耗数据缺失检查
energy_cols = [col for col in df.columns if col.startswith('energy_')]
energy_missing = df[energy_cols].isnull().all(axis=1)
print(f"缺失所有能耗数据的记录: {energy_missing.sum()}")

# 性能指标缺失检查
perf_cols = [col for col in df.columns if col.startswith('perf_')]
perf_missing = df[perf_cols].isnull().all(axis=1)
print(f"缺失所有性能指标的记录: {perf_missing.sum()}")

# 筛选可用数据（推荐）
df_usable = df[
    (df['status'] == 'success') &  # 训练成功
    (~energy_missing) &             # 有能耗数据
    (~perf_missing)                 # 有性能指标
].copy()

print(f"可用记录数: {len(df_usable)} / {len(df)} ({len(df_usable)/len(df)*100:.1f}%)")
```

### 3. 数据验证流程

```python
def validate_data(df, file_name='unknown'):
    """
    验证数据质量

    Args:
        df: DataFrame
        file_name: 文件名（用于报告）

    Returns:
        dict: 验证结果
    """
    results = {
        'file': file_name,
        'total_rows': len(df),
        'unique_timestamps': df['timestamp'].nunique(),
        'has_duplicates': df['timestamp'].nunique() < len(df),
        'training_success': (df['status'] == 'success').sum(),
        'has_energy': (~df[[col for col in df.columns if col.startswith('energy_')]].isnull().all(axis=1)).sum(),
        'has_performance': (~df[[col for col in df.columns if col.startswith('perf_')]].isnull().all(axis=1)).sum(),
    }

    # 计算可用记录
    results['usable'] = len(df[
        (df['status'] == 'success') &
        (~df[[col for col in df.columns if col.startswith('energy_')]].isnull().all(axis=1)) &
        (~df[[col for col in df.columns if col.startswith('perf_')]].isnull().all(axis=1))
    ])

    # 输出报告
    print(f"\n数据验证报告 - {file_name}")
    print("=" * 60)
    print(f"总记录数: {results['total_rows']}")
    print(f"唯一timestamp: {results['unique_timestamps']}")
    print(f"有重复: {'是' if results['has_duplicates'] else '否'}")
    print(f"训练成功: {results['training_success']} ({results['training_success']/results['total_rows']*100:.1f}%)")
    print(f"有能耗数据: {results['has_energy']} ({results['has_energy']/results['total_rows']*100:.1f}%)")
    print(f"有性能指标: {results['has_performance']} ({results['has_performance']/results['total_rows']*100:.1f}%)")
    print(f"✅ 可用记录: {results['usable']} ({results['usable']/results['total_rows']*100:.1f}%)")

    return results

# 使用示例
df = pd.read_csv('data/data.csv')
results = validate_data(df, 'data.csv')
```

**工具脚本**: `tools/data_management/validate_raw_data.py`

### 4. 数据追加流程

当有新实验结果时，使用标准追加流程：

```bash
# 追加新实验结果到 raw_data.csv
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS

# 重新生成 data.csv（如果需要）
python3 tools/data_management/create_unified_data_csv.py
```

**详细指南**: [APPEND_SESSION_TO_RAW_DATA_GUIDE.md](APPEND_SESSION_TO_RAW_DATA_GUIDE.md)

---

## 常见错误和解决方案

### ❌ 错误1: 使用 experiment_id 作为唯一键

**错误代码**:
```python
df_unique = df.drop_duplicates(subset=['experiment_id'])
# 🚨 这会丢失大量有效数据！
```

**正确方法**:
```python
df_unique = df.drop_duplicates(subset=['timestamp'])
```

**原因**: experiment_id 代表配置（可重复），timestamp 代表运行实例（唯一）

**参考**: [analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md](../analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md)

### ❌ 错误2: 直接读取 raw_data.csv 不处理并行模式

**错误代码**:
```python
df = pd.read_csv('data/raw_data.csv')
learning_rate = df['learning_rate']  # 🚨 并行模式数据会丢失！
```

**正确方法**:
- **方案A**: 使用 data.csv（推荐）
```python
df = pd.read_csv('data/data.csv')
learning_rate = df['learning_rate']  # ✅ 自动处理
```

- **方案B**: 使用 raw_data.csv + 特殊处理
```python
df = pd.read_csv('data/raw_data.csv')
# 参考 RAW_DATA_CSV_USAGE_GUIDE.md 中的 get_field() 函数
```

**参考**: [RAW_DATA_CSV_USAGE_GUIDE.md](RAW_DATA_CSV_USAGE_GUIDE.md)

### ❌ 错误3: 误判空字符串为缺失值

**错误代码**:
```python
# 在 raw_data.csv 中
missing = df[df['learning_rate'] == '']  # 🚨 可能不是缺失！
```

**正确方法**:
```python
# 空字符串通常表示使用默认值，不是缺失
truly_missing = df[df['learning_rate'].isna()]

# 或使用 data.csv（已处理好）
df = pd.read_csv('data/data.csv')
```

### ❌ 错误4: 忽略训练失败的记录

**错误代码**:
```python
# 不检查 status 字段
df_analysis = df[~df['energy_gpu_avg'].isna()]  # 🚨 可能包含失败记录
```

**正确方法**:
```python
# 明确筛选训练成功的记录
df_analysis = df[
    (df['status'] == 'success') &
    (~df['energy_gpu_avg'].isna())
]
```

### ❌ 错误5: 混淆 data.csv 和 raw_data.csv 的行数

**错误认识**:
- "我的分析只有726行，但应该有970行，数据丢失了！"

**正确认识**:
- data.csv: 726行（精选的高质量数据）
- raw_data.csv: 970行（包含所有记录，含低质量数据）
- 两者都是有效的，根据需求选择

### ❌ 错误6: 不验证数据质量就开始分析

**错误做法**:
```python
df = pd.read_csv('data/data.csv')
# 直接开始分析，不检查数据质量
model = LinearRegression()
model.fit(X, y)  # 🚨 可能包含异常值、缺失值等
```

**正确方法**:
```python
df = pd.read_csv('data/data.csv')

# 1. 验证数据质量
results = validate_data(df, 'data.csv')

# 2. 筛选可用数据
df_usable = df[
    (df['status'] == 'success') &
    (~df[[col for col in df.columns if col.startswith('energy_')]].isnull().all(axis=1)) &
    (~df[[col for col in df.columns if col.startswith('perf_')]].isnull().all(axis=1))
].copy()

# 3. 检查异常值
# ... (根据具体情况)

# 4. 开始分析
model = LinearRegression()
model.fit(X, y)
```

---

## 参考文档

### 核心文档 ⭐⭐⭐⭐⭐

| 文档 | 用途 |
|------|------|
| **本文档** | 数据使用主指南（单一真实来源） |
| [RAW_DATA_CSV_USAGE_GUIDE.md](RAW_DATA_CSV_USAGE_GUIDE.md) | raw_data.csv 详细使用指南 ⭐⭐⭐⭐⭐ |
| [analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md](../analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md) | 唯一标识说明 ⭐⭐⭐ |
| [analysis/docs/DATA_UNDERSTANDING_CORRECTION_20251228.md](../analysis/docs/DATA_UNDERSTANDING_CORRECTION_20251228.md) | 数据理解关键更正 ⭐⭐⭐ |

### 数据质量报告

| 文档 | 用途 |
|------|------|
| [DATA_USABILITY_FOR_REGRESSION_20260114.md](DATA_USABILITY_FOR_REGRESSION_20260114.md) | 6分组回归分析数据可用性 |
| [DATA_REPAIR_FINAL_SUMMARY_20260113.md](DATA_REPAIR_FINAL_SUMMARY_20260113.md) | 数据修复最终总结 |
| [analysis/data/energy_research/DATA_STATUS_REPORT_20260114.md](../analysis/data/energy_research/DATA_STATUS_REPORT_20260114.md) | 数据现状完整报告 |

### 文件对比分析

| 文档 | 用途 |
|------|------|
| [analysis/docs/DATA_FILES_COMPARISON.md](../analysis/docs/DATA_FILES_COMPARISON.md) | data.csv vs raw_data.csv 详细对比 |
| [analysis/data/energy_research/RAW_DATA_VS_DATA_CSV_COMPARISON.md](../analysis/data/energy_research/RAW_DATA_VS_DATA_CSV_COMPARISON.md) | 两文件对比分析 |

### 数据处理工具

| 脚本 | 用途 |
|------|------|
| `tools/data_management/validate_raw_data.py` | 数据验证 |
| `tools/data_management/deduplicate_by_timestamp.py` | 去重处理 |
| `tools/data_management/append_session_to_raw_data.py` | 追加新数据 |
| `tools/data_management/create_unified_data_csv.py` | 生成统一data.csv |
| `tools/data_management/analyze_experiment_status.py` | 实验状态分析 |

### 历史文档归档

所有历史数据报告（2025-12到2026-01）已归档到:
- `docs/archived/data_reports_archive_20260115/`

归档文档仍可访问，但不再维护更新。

---

## 📞 获取帮助

### 遇到问题时

1. **首先**: 检查本文档的"常见错误和解决方案"章节
2. **其次**: 查阅相关的参考文档
3. **工具**: 使用 `tools/data_management/` 中的验证脚本
4. **文档**: 查看 CLAUDE.md 或 analysis/docs/INDEX.md

### 报告问题

如果发现文档错误或有改进建议，请：
1. 创建详细的问题描述
2. 包含复现步骤和数据示例
3. 提及相关的文档版本和日期

---

## 📝 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0 | 2026-01-15 | 初始版本，整合78个数据文档的核心内容 |

---

**文档维护者**: Claude Assistant
**下次审查**: 2026-02-15
**反馈**: 请通过项目主文档 CLAUDE.md 联系
