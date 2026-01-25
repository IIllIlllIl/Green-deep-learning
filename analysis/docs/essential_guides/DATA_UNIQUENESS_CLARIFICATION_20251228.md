# 数据唯一标识说明 ⚠️ 重要

**日期**: 2025-12-28
**优先级**: 🚨 高
**目的**: 澄清experiment_id不是唯一标识，防止数据处理错误

---

## 📋 核心要点

### ⚠️ 关键事实

**experiment_id 不是唯一标识符！**

- ✅ **设计特性**：不同轮次的实验**可以且应该**使用相同的experiment_id
- ✅ **唯一标识**：需要使用 **experiment_id + timestamp** 的组合
- ❌ **常见错误**：将experiment_id误认为唯一标识，导致数据覆盖或统计错误

---

## 🔍 为什么experiment_id会重复？

### 设计原理

主项目的实验设计允许：
1. **多轮实验**：同一个配置可以在不同时间运行多次
2. **参数调优**：重新运行相同的experiment_id以验证结果
3. **失败重试**：失败的实验可以用相同ID重新运行
4. **补充实验**：后续补齐之前缺失的实验

### 实际示例

```csv
experiment_id,timestamp,repository,energy_gpu_avg_watts,perf_test_accuracy
VulBERTa_mlp_001,2025-12-06T23:16:47,VulBERTa,222.82,
VulBERTa_mlp_001,2025-12-13T21:28:09,VulBERTa,245.38,
VulBERTa_mlp_001,2025-12-15T23:43:37,VulBERTa,,0.875
VulBERTa_mlp_001,2025-12-17T22:26:33,VulBERTa,236.32,0.881
```

**说明**：
- 同一个`VulBERTa_mlp_001`有4条记录
- 每条记录有不同的timestamp
- 第1、2条：有能耗，无性能（早期实验）
- 第3条：无能耗，有性能（监控故障）
- 第4条：完整数据（补齐实验）

---

## ✅ 正确的唯一标识方法

### 方法1：复合键（推荐）⭐⭐⭐

```python
import pandas as pd

df = pd.read_csv('data.csv')

# ✅ 正确：使用复合键作为唯一标识
df['unique_key'] = df['experiment_id'] + '|' + df['timestamp']

# 使用复合键去重
df_dedup = df.drop_duplicates(subset=['unique_key'])

# 或者直接使用两列去重
df_dedup = df.drop_duplicates(subset=['experiment_id', 'timestamp'])
```

### 方法2：分组聚合

```python
# 如果需要合并同一experiment_id的多条记录
df_grouped = df.groupby('experiment_id').agg({
    'timestamp': 'max',  # 选择最新的记录
    'energy_gpu_avg_watts': 'mean',  # 或者取平均值
    'perf_test_accuracy': 'last'  # 或者取最后一个
}).reset_index()
```

### 方法3：选择策略

```python
# 策略1: 只保留最新的记录
df_latest = df.sort_values('timestamp').groupby('experiment_id').tail(1)

# 策略2: 只保留数据最完整的记录
def completeness_score(row):
    return sum([
        pd.notna(row['energy_gpu_avg_watts']),
        pd.notna(row['perf_test_accuracy'])
    ])

df['completeness'] = df.apply(completeness_score, axis=1)
df_best = df.sort_values('completeness').groupby('experiment_id').tail(1)
```

---

## ❌ 错误的处理方式

### 错误1：使用experiment_id作为字典键

```python
# ❌ 错误：会导致数据覆盖
data_dict = {}
for row in rows:
    exp_id = row['experiment_id']
    data_dict[exp_id] = row  # 后面的记录会覆盖前面的！
```

**后果**：
- 只保留最后一条记录
- 726行数据变成637行（丢失89条）
- 统计结果错误

### 错误2：假设experiment_id唯一

```python
# ❌ 错误：假设每个experiment_id只有一条记录
df = pd.read_csv('data.csv')
df_indexed = df.set_index('experiment_id')  # 会产生警告或覆盖
```

**后果**：
- pandas会发出重复索引警告
- 某些操作结果不符合预期

### 错误3：用experiment_id计数

```python
# ❌ 错误：统计唯一experiment_id数量
unique_experiments = df['experiment_id'].nunique()
# 结果: 637（实际行数是726）
```

**后果**：
- 样本量统计错误
- 可能低估实验数量

---

## 📊 当前数据统计

### data.csv和raw_data.csv

| 指标 | 数值 | 说明 |
|------|------|------|
| **总行数** | 726 | 实际实验记录数 |
| **唯一experiment_id** | 637 | 不同的experiment_id |
| **重复experiment_id** | 89行 | 有多条记录的ID数量 |
| **有重复的ID数** | 68个 | 这些ID各有2-5条记录 |

### 重复情况分布

```python
from collections import Counter

# 统计每个experiment_id的记录数
id_counts = Counter(df['experiment_id'])

# 分布
1条记录: 569个ID
2条记录: 48个ID
3条记录: 15个ID
4条记录: 4个ID
5条记录: 1个ID
```

### 重复最多的experiment_id

| experiment_id | 记录数 | 任务 |
|---------------|--------|------|
| VulBERTa_mlp_002 | 5条 | VulBERTa |
| VulBERTa_mlp_001 | 4条 | VulBERTa |
| VulBERTa_mlp_003-007 | 4条 | VulBERTa |
| MRT-OAST_default_001 | 3条 | MRT-OAST |

---

## 🎯 不同场景的处理建议

### 场景1：因果分析（DiBS, DML）

**建议**：选择数据最完整的记录

```python
# 定义完整性评分
def completeness_score(row):
    score = 0
    # 检查能耗
    if pd.notna(row['energy_gpu_avg_watts']):
        score += 2
    # 检查性能
    if pd.notna(row['perf_test_accuracy']):
        score += 2
    # 检查超参数
    if pd.notna(row['hyperparam_learning_rate']):
        score += 1
    return score

df['score'] = df.apply(completeness_score, axis=1)
df_analysis = df.sort_values(['experiment_id', 'score']).groupby('experiment_id').tail(1)
```

**理由**：
- 因果分析需要完整的输入-输出数据
- 选择数据最完整的记录可最大化可用样本

### 场景2：时间序列分析

**建议**：保留所有记录，使用复合键

```python
df['unique_key'] = df['experiment_id'] + '|' + df['timestamp']
df = df.sort_values('timestamp')
# 分析同一配置在不同时间的表现变化
```

**理由**：
- 可以观察实验的时间演化
- 可以分析重复实验的一致性

### 场景3：统计分析（描述性统计）

**建议**：明确说明使用的记录选择策略

```python
# 策略A：使用所有记录（726行）
df_all = df  # 726行

# 策略B：每个experiment_id只用一条（637行）
df_dedup = df.groupby('experiment_id').first()  # 637行

# 策略C：只用数据完整的记录（569行）
df_complete = df[(df['energy_gpu_avg_watts'].notna()) &
                  (df['perf_test_accuracy'].notna())]  # 569行
```

**理由**：
- 不同策略会得到不同的样本量
- 必须在文档中明确说明使用的策略

### 场景4：数据合并/追加

**建议**：始终使用复合键检查重复

```python
# 读取现有数据
df_existing = pd.read_csv('data.csv')
df_existing['key'] = df_existing['experiment_id'] + '|' + df_existing['timestamp']

# 读取新数据
df_new = pd.read_csv('new_data.csv')
df_new['key'] = df_new['experiment_id'] + '|' + df_new['timestamp']

# 检查重复
duplicates = set(df_existing['key']) & set(df_new['key'])
if duplicates:
    print(f"警告：发现{len(duplicates)}条重复记录")

# 合并（去重）
df_merged = pd.concat([df_existing, df_new]).drop_duplicates(subset=['key'])
```

**理由**：
- 防止重复追加相同的实验记录
- 确保数据完整性

---

## 📚 相关文档

### 已修改的文档

1. **COLUMN_USAGE_ANALYSIS.md** ✅
   - 修改了experiment_id和timestamp的描述

2. **DATA_UNDERSTANDING_CORRECTION_20251228.md** （待更新）
   - 需要补充唯一标识说明

3. **DATA_FILES_COMPARISON.md** （待更新）
   - 需要补充唯一标识说明

### 需要注意的主项目文档

主项目中也提到了这个问题：
- **CLAUDE.md**: Phase 5性能指标缺失问题分析提到了复合键的重要性

---

## 🔧 实用工具函数

### 检查重复experiment_id

```python
def check_duplicate_ids(df):
    """检查并报告重复的experiment_id"""
    from collections import Counter

    id_counts = Counter(df['experiment_id'])
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    print(f"总行数: {len(df)}")
    print(f"唯一experiment_id: {len(id_counts)}")
    print(f"重复的experiment_id: {len(duplicates)}个")
    print(f"重复的记录数: {sum(v-1 for v in duplicates.values())}条")

    return duplicates
```

### 生成唯一标识列

```python
def add_unique_key(df):
    """为DataFrame添加唯一标识列"""
    df['unique_key'] = df['experiment_id'] + '|' + df['timestamp']

    # 验证唯一性
    assert df['unique_key'].nunique() == len(df), \
        "警告：unique_key不唯一！可能timestamp格式有问题"

    return df
```

### 选择最佳记录

```python
def select_best_records(df):
    """为每个experiment_id选择数据最完整的记录"""

    def score_row(row):
        """计算记录的完整性分数"""
        score = 0
        # 能耗数据（权重2）
        if pd.notna(row.get('energy_gpu_avg_watts')):
            score += 2
        # 性能数据（权重2）
        if pd.notna(row.get('perf_test_accuracy')):
            score += 2
        # 超参数数据（权重1）
        hyperparam_cols = [c for c in df.columns if c.startswith('hyperparam_')]
        score += sum(pd.notna(row.get(c)) for c in hyperparam_cols) / len(hyperparam_cols)

        return score

    df['completeness_score'] = df.apply(score_row, axis=1)
    df_best = df.sort_values(['experiment_id', 'completeness_score', 'timestamp']) \
                .groupby('experiment_id').tail(1)

    return df_best.drop('completeness_score', axis=1)
```

---

## 🚨 重要提醒

### 给数据分析人员

1. ⚠️ **永远不要**假设experiment_id是唯一的
2. ✅ **始终使用**experiment_id + timestamp作为唯一标识
3. ✅ **明确说明**你的记录选择策略（全部/去重/最新/最完整）
4. ✅ **在报告中**注明实际使用的样本量

### 给代码开发人员

1. ⚠️ **不要**用experiment_id作为字典的键
2. ✅ **使用**复合键或multi-index
3. ✅ **添加**重复检查和警告
4. ✅ **提供**清晰的去重选项

### 给文档编写人员

1. ⚠️ **不要**说"experiment_id是唯一标识符"
2. ✅ **说明**"experiment_id + timestamp组合才是唯一标识"
3. ✅ **解释**为什么会有重复（设计特性，不是错误）
4. ✅ **提供**处理建议

---

## 📝 常见问题

### Q1: 为什么不设计成experiment_id唯一？

**A**: 因为实验需要可重复性。同一个配置在不同时间运行可能得到不同结果（随机性、硬件状态等）。允许重复ID可以：
- 验证结果的可重复性
- 补齐失败或缺失数据的实验
- 进行多轮实验比较

### Q2: 如果我只想要一条记录怎么办？

**A**: 使用去重策略，推荐按完整性排序后取最佳记录：

```python
df_best = select_best_records(df)  # 见上面的工具函数
```

### Q3: 分析时应该用726行还是637行？

**A**: 取决于你的研究问题：
- **因果分析**：建议用637行（每个ID一条最佳记录）
- **描述性统计**：可以用726行（全部记录）
- **时间分析**：必须用726行（观察变化）

**关键**：在报告中明确说明你使用的策略和样本量。

### Q4: stage2_mediators.csv也有重复吗？

**A**: 是的，与data.csv一样：
- 总行数：726
- 唯一experiment_id：637
- 这是正常的设计特性

---

## 📖 相关阅读

- [DATA_UNDERSTANDING_CORRECTION_20251228.md](DATA_UNDERSTANDING_CORRECTION_20251228.md) - 数据理解纠正
- [DATA_FILES_COMPARISON.md](DATA_FILES_COMPARISON.md) - data.csv vs raw_data.csv对比
- 主项目 [CLAUDE.md](../../CLAUDE.md) - 提到复合键的重要性

---

**文档维护者**: Analysis模块
**创建日期**: 2025-12-28
**重要性**: 🚨 极高 - 影响所有数据处理和分析
**状态**: ✅ 完成

---

## 🏁 总结

**核心原则**：
1. ✅ experiment_id **不是**唯一标识符
2. ✅ 使用 **experiment_id + timestamp** 作为唯一标识
3. ✅ 重复是**设计特性**，不是错误
4. ✅ 选择合适的去重策略，并在文档中说明

**立即行动**：
- 检查你的代码是否正确处理了重复ID
- 更新你的分析报告，明确说明样本量和去重策略
- 在新代码中使用本文档提供的工具函数
