# 6分组数据问题分析报告

**日期**: 2026-01-15
**分析目的**: 系统分析data.csv到6分组数据的工作流程问题
**数据源**: data.csv (970行) → 6分组数据 (423行)

---

## 📋 执行摘要

### 核心发现
1. **数据大量丢失**: 970行 → 423行 (丢失56.5%数据)
2. **根本原因**: 统一40%缺失率阈值不适用于多模型数据集
3. **设计与实现不匹配**: 原始设计指定每组特定超参数，实际实现使用全局阈值
4. **环境配置遗漏**: 未使用已配置的conda环境导致DiBS分析失败
5. **文档信息过时**: CLAUDE.md中data.csv行数错误（726→970）

### 影响
- ❌ 损失547行可用数据
- ❌ 某些模型组数据量不足（如MRT-OAST仅46行）
- ❌ DiBS分析全部失败（0/6成功）
- ❌ 重复错误：agent未阅读文档，使用错误环境

---

## 🔍 问题详细分析

### 问题1: 统一缺失率阈值导致数据丢失

#### 现象
```
原始data.csv: 970行
训练成功且有数据: 818行
6分组后总数: 423行
丢失数据: 395行 (48.3%的可用数据被丢弃)
```

#### 根本原因
脚本使用统一的40%缺失率阈值：

```python
# generate_dibs_6groups_from_data_csv.py
MISSING_THRESHOLD = 0.40

# 对所有列一视同仁
cols_to_keep = missing_rates[missing_rates <= missing_threshold].index.tolist()
```

#### 具体例子
1. **hyperparam_alpha (93.5%缺失)**
   - 只有bug-localization使用（约5%的数据）
   - 其他95%的数据自然缺失
   - 被错误地删除

2. **hyperparam_kfold (94.6%缺失)**
   - 只有bug-localization使用
   - 同样被删除

3. **hyperparam_batch_size**
   - examples组使用（约15%的数据）
   - 其他组不使用，导致高缺失率
   - 可能被错误删除

---

### 问题2: 设计与实现不匹配

#### 原始设计 (6GROUPS_DATA_GENERATION_PLAN_20251224.md)

每个组有特定的超参数集合：

| 任务组 | 指定超参数 |
|--------|-----------|
| examples | learning_rate, batch_size, epochs, seed |
| resnet | learning_rate, l2_regularization, seed |
| person_reid | learning_rate, dropout, seed |
| vulberta | learning_rate, l2_regularization, seed |
| bug_localization | l2_regularization, kfold, seed |
| mrt_oast | dropout, epochs, learning_rate, seed, weight_decay |

#### 实际实现

```python
# 没有按组选择超参数
hyperparam_cols = [col for col in group_df.columns if col.startswith('hyperparam_')]
# 所有超参数列都被包含，然后用统一阈值过滤
```

---

### 问题3: 超参数语义相同但名称不同

从VARIABLE_EXPANSION_PLAN.md发现：

| 参数名 | 使用模型 | 含义 | 框架 |
|--------|---------|------|------|
| hyperparam_weight_decay | VulBERTa, MRT-OAST, ResNet | L2正则化系数 | PyTorch |
| hyperparam_alpha | bug-localization | L2正则化系数 | scikit-learn |

**问题**: 相同功能的参数未统一，导致：
- MRT-OAST组：weight_decay有值，alpha为NaN
- bug_localization组：alpha有值，weight_decay为NaN

---

### 问题4: 环境配置问题

#### 发现的conda环境
```bash
# 存在专用环境但未使用
/home/green/miniconda3/envs/causal-research/
- 已安装DiBS
- 已安装其他因果推断工具
```

#### 实际执行
```python
# 使用默认python（base环境）
python scripts/run_dibs_on_new_6groups.py

# 结果：ModuleNotFoundError: No module named 'dibs'
```

---

### 问题5: 文档信息过时

#### CLAUDE.md中的错误信息
- ❌ "data/data.csv (726行，95.3%可用)"
- ✅ 实际：970行，约818条可用记录

#### 缺失的重要信息
- 未提及conda环境要求
- 未说明6分组数据生成的注意事项
- 未更新超参数统一的说明

---

## 💡 解决方案

### 1. 修复数据生成脚本

```python
# 为每个组指定需要的列
TASK_GROUP_COLUMNS = {
    'group1_examples': {
        'hyperparams': ['hyperparam_batch_size', 'hyperparam_learning_rate',
                       'hyperparam_epochs', 'hyperparam_seed'],
        'performance': ['perf_test_accuracy'],
    },
    'group5_bug_localization': {
        'hyperparams': ['hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_seed'],
        'performance': ['perf_top1_accuracy', 'perf_top5_accuracy'],
    },
    # ... 其他组
}

# 按组选择列，而不是使用统一阈值
selected_cols = TASK_GROUP_COLUMNS[group_id]['hyperparams'] + \
                TASK_GROUP_COLUMNS[group_id]['performance'] + \
                energy_cols + control_cols
```

### 2. 超参数统一

在数据预处理时统一语义相同的参数：
```python
# 统一L2正则化参数
df['l2_regularization'] = df['hyperparam_weight_decay'].fillna(df['hyperparam_alpha'])
```

### 3. 更新CLAUDE.md

✅ 已完成更新：
- 修正data.csv行数信息
- 添加conda环境使用说明
- 添加6分组数据生成注意事项
- 更新版本至v5.7.0

### 4. 创建环境检查脚本

```python
def check_dibs_environment():
    """检查DiBS分析环境"""
    try:
        import dibs
        print("✅ DiBS已安装")
        return True
    except ImportError:
        print("❌ DiBS未安装")
        print("请运行: conda activate causal-research")
        return False
```

---

## 📊 数据恢复预期

如果正确实现按组选择列：

| 任务组 | 当前行数 | 预期行数 | 恢复数据 |
|--------|---------|---------|----------|
| examples | 126 | ~200 | +74 |
| vulberta | 52 | ~140 | +88 |
| person_reid | 118 | ~180 | +62 |
| bug_localization | 40 | ~100 | +60 |
| mrt_oast | 46 | ~80 | +34 |
| resnet | 41 | ~70 | +29 |
| **总计** | 423 | ~770 | +347 |

预期恢复率：从43.5% → 79.4%

---

## 🎯 行动建议

### 立即行动
1. ✅ 更新CLAUDE.md文档（已完成）
2. ⏳ 修改数据生成脚本，使用按组列选择
3. ⏳ 重新生成6分组数据
4. ⏳ 使用正确的conda环境重新运行DiBS

### 长期改进
1. 建立数据处理标准操作流程(SOP)
2. 添加自动化测试验证数据完整性
3. 统一语义相同的变量名
4. 在脚本开始时检查环境配置

---

## 📚 相关文档

- [analysis/docs/reports/VARIABLE_EXPANSION_PLAN.md](VARIABLE_EXPANSION_PLAN.md) - 变量统一方案
- [analysis/docs/reports/6GROUPS_DATA_GENERATION_PLAN_20251224.md](6GROUPS_DATA_GENERATION_PLAN_20251224.md) - 原始设计
- [analysis/scripts/generate_dibs_6groups_from_data_csv.py](../../scripts/generate_dibs_6groups_from_data_csv.py) - 需要修复的脚本
- [CLAUDE.md](../../../CLAUDE.md) - 已更新至v5.7.0

---

**分析人**: Claude
**创建日期**: 2026-01-15
**状态**: ✅ 问题已识别，解决方案已提出