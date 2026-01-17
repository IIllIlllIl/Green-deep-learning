# 能耗数据相对值转换方案

**创建日期**: 2026-01-16
**文档版本**: v1.0
**状态**: ✅ 方案确认
**评估状态**: ✅ 已通过Subagent评估（评分 ⭐⭐⭐⭐/5）

---

## 📋 执行摘要

本文档记录了为DiBS因果图学习准备能耗数据的相对值转换方案。通过将能耗指标转换为"与默认值的差值"，消除并行/非并行训练模式的系统性偏差，使DiBS能够更准确地发现超参数对能耗的真实因果关系。

### 核心问题

**并行vs非并行模式的系统性能耗差异**会干扰DiBS因果发现：
- 并行模式天然能耗更高（多进程/线程开销）
- CPU Package能耗: 并行模式高24.4% (p<0.01) ***
- CPU总能耗: 并行模式高23.4% (p<0.01) ***
- GPU平均功率: 并行模式高11.2% (p<0.001) ***

这种系统性偏差可能导致DiBS错误地将`is_parallel`识别为能耗的主要原因，掩盖超参数的真实因果影响。

### 解决方案

**使用相对值转换**：
```
相对能耗 = 实际能耗 - 同模式下的默认值能耗
```

**优势**：
- ✅ 消除并行/非并行系统性差异
- ✅ 保留超参数变异引起的增量变化
- ✅ 更适合DiBS因果图学习
- ✅ 结果更易解释（"相比默认配置的变化"）

---

## 1️⃣ 默认值实验识别结果

### 1.1 识别成功 ✅

**所有6组都找到了默认值实验**（每组至少有1个并行和1个非并行）：

| 组别 | 模型 | 默认值实验总数 | 并行 | 非并行 | 并行GPU能耗 | 非并行GPU能耗 |
|-----|------|--------------|-----|--------|------------|--------------|
| **group1_examples** | mnist, mnist_ff, mnist_rnn, siamese | 10 | 4 | 6 | 50,972 J | 12,157 J |
| **group2_vulberta** | mlp | 2 | 1 | 1 | 847,272 J | 726,127 J |
| **group3_person_reid** | densenet121, hrnet18, pcb | 6 | 3 | 3 | 962,694 J | 919,491 J |
| **group4_bug_localization** | default | 2 | 1 | 1 | 227,915 J | 22,796 J |
| **group5_mrt_oast** | default | 2 | 1 | 1 | 303,568 J | 331,876 J |
| **group6_resnet** | resnet20 | 2 | 1 | 1 | 252,255 J | 246,851 J |

### 1.2 Group2 和 Group4 数据来源

⚠️ **重要发现**：group2_vulberta和group4_bug_localization的默认值实验在6groups数据中缺失，但在`data.csv`中存在。

**原因**：
- 这些默认值实验缺少性能指标数据
- 6groups生成脚本要求同时有能耗+性能数据，因此被过滤掉
- 但能耗数据是完整的，可以正常用作基准值

**解决方案**：
- 从`data.csv`中提取这些默认值实验的能耗数据作为基准值
- 仅用于能耗基准计算，不影响DiBS分析（DiBS不需要这些实验的性能数据）
- 文档中标注数据来源

**验证**：
- ✅ data.csv中的能耗数据与raw_data.csv一致
- ✅ 基准值计算不需要性能指标
- ✅ 对DiBS能耗分析无影响

---

## 2️⃣ 数据转换规则

### 2.1 转换规则表

| 变量类别 | 列名 | 转换方式 | 输出列名 | 保留原列 | 理由 |
|---------|------|---------|---------|---------|------|
| **基础能耗** | `energy_cpu_pkg_joules` | 相对值 | `rel_energy_cpu_pkg_joules` | ❌ | 并行高24.4% (p<0.01) |
| **基础能耗** | `energy_cpu_ram_joules` | 相对值 | `rel_energy_cpu_ram_joules` | ❌ | 并行高9.3% (趋势) |
| **基础能耗** | `energy_cpu_total_joules` | 相对值 | `rel_energy_cpu_total_joules` | ❌ | 并行高23.4% (p<0.01) |
| **基础能耗** | `energy_gpu_total_joules` | 相对值 | `rel_energy_gpu_total_joules` | ❌ | 并行高9.2% (趋势) |
| **功率** | `energy_gpu_avg_watts` | 相对值 | `rel_gpu_avg_watts` | ❌ | 并行高11.2% (p<0.001) |
| **功率** | `energy_gpu_max_watts` | 相对值 | `rel_gpu_max_watts` | ❌ | 并行高5.8% (p<0.05) |
| **功率** | `energy_gpu_min_watts` | 相对值 | `rel_gpu_min_watts` | ❌ | 逻辑一致性 |
| **温度** | `energy_gpu_temp_avg_celsius` | 保留绝对值 | `energy_gpu_temp_avg_celsius` | ✅ | 状态量，非累积量 |
| **温度** | `energy_gpu_temp_max_celsius` | 保留绝对值 | `energy_gpu_temp_max_celsius` | ✅ | 状态量，非累积量 |
| **利用率** | `energy_gpu_util_avg_percent` | 保留绝对值 | `energy_gpu_util_avg_percent` | ✅ | 比例量，已归一化 |
| **利用率** | `energy_gpu_util_max_percent` | 保留绝对值 | `energy_gpu_util_max_percent` | ✅ | 比例量，已归一化 |
| **控制变量** | `is_parallel`, `timestamp`, `model_*`, `hyperparam_*` | 保留 | 原列名 | ✅ | 元数据和自变量 |

### 2.2 转换公式

```python
# 对于每个实验
if row['is_parallel'] == True:
    baseline = parallel_mode_baseline[group_id][metric]
else:
    baseline = nonparallel_mode_baseline[group_id][metric]

rel_value = actual_value - baseline
```

### 2.3 基准值计算（稳健方法）⭐ 关键

根据Subagent评估建议，采用**稳健基准值计算方法**：

```python
def get_robust_baseline(defaults_df, is_parallel, metric):
    """
    获取稳健的基准值

    方法：
    1. 使用中位数（比平均值更稳健）
    2. 剔除离群值（z-score > 2.5）
    3. 检查变异系数（CV < 20%）
    """
    baseline_values = defaults_df[defaults_df['is_parallel'] == is_parallel][metric]

    # 剔除离群值
    from scipy import stats
    z_scores = np.abs(stats.zscore(baseline_values))
    values_clean = baseline_values[z_scores < 2.5]

    # 使用中位数
    baseline = values_clean.median()
    mad = np.median(np.abs(values_clean - baseline))  # 中位数绝对偏差

    # 检查稳定性
    cv_mad = mad / baseline
    if cv_mad > 0.2:
        print(f"⚠️ 基准值不稳定 (CV_MAD={cv_mad:.1%})")

    return baseline, mad, len(values_clean)
```

**为什么使用中位数而不是平均值**：
- ✅ 对离群值更稳健
- ✅ 默认值实验数量少（大部分组只有1-2个），中位数更可靠
- ✅ 避免极端值影响整体基准

---

## 3️⃣ 输出数据结构

### 3.1 输出目录

```
data/energy_research/6groups_relative_value/
├── group1_examples.csv
├── group2_vulberta.csv
├── group3_person_reid.csv
├── group4_bug_localization.csv
├── group5_mrt_oast.csv
├── group6_resnet.csv
├── baseline_values.json                # 记录每组的基准值
├── conversion_report.md                # 转换报告
└── data_dictionary.md                  # 数据字典
```

### 3.2 数据列结构示例 (group1_examples.csv)

```
# 元数据
timestamp, is_parallel

# 模型 one-hot 编码
model_mnist_ff, model_mnist_rnn, model_siamese

# 超参数（自变量）
hyperparam_batch_size, hyperparam_learning_rate, hyperparam_epochs, hyperparam_seed

# 性能指标
perf_test_accuracy

# 相对能耗（基础指标）⭐ 新增
rel_energy_cpu_pkg_joules, rel_energy_cpu_ram_joules,
rel_energy_cpu_total_joules, rel_energy_gpu_total_joules

# 相对功率 ⭐ 新增
rel_gpu_avg_watts, rel_gpu_max_watts, rel_gpu_min_watts

# 绝对温度（保留）
energy_gpu_temp_avg_celsius, energy_gpu_temp_max_celsius

# 绝对利用率（保留）
energy_gpu_util_avg_percent, energy_gpu_util_max_percent
```

### 3.3 baseline_values.json 结构

```json
{
  "group1_examples": {
    "parallel": {
      "energy_gpu_total_joules": {
        "baseline": 50972.15,
        "mad": 15234.56,
        "n_samples": 4,
        "method": "median"
      },
      "energy_cpu_total_joules": {
        "baseline": 8780.27,
        "mad": 2456.78,
        "n_samples": 4,
        "method": "median"
      }
    },
    "nonparallel": {
      "energy_gpu_total_joules": {
        "baseline": 12157.08,
        "mad": 4321.09,
        "n_samples": 6,
        "method": "median"
      }
    }
  },
  "group2_vulberta": {
    "note": "Baseline from data.csv, missing from 6groups due to no performance metrics",
    "parallel": {
      "energy_gpu_total_joules": {
        "baseline": 847272.32,
        "n_samples": 1,
        "method": "single_value"
      }
    },
    "nonparallel": {
      "energy_gpu_total_joules": {
        "baseline": 726127.40,
        "n_samples": 1,
        "method": "single_value"
      }
    }
  }
}
```

---

## 4️⃣ 数据验证

### 4.1 验证步骤

**验证1: 默认值实验的相对值应为0**
```python
# 对于有默认值的组，默认值实验的所有 rel_* 列应接近0
for group_id in groups:
    defaults = df_relative[df_relative['num_mutated_params'] == 0]
    for col in rel_energy_cols:
        assert abs(defaults[col].mean()) < 100, f"{group_id}-{col} 默认值相对值不为0"
```

**验证2: is_parallel效应显著减弱**
```python
# 转换前后对比
from scipy import stats

# 转换前
t_stat_before, p_before = stats.ttest_ind(
    df_original[df_original['is_parallel']]['energy_gpu_total_joules'],
    df_original[~df_original['is_parallel']]['energy_gpu_total_joules']
)

# 转换后
t_stat_after, p_after = stats.ttest_ind(
    df_relative[df_relative['is_parallel']]['rel_energy_gpu_total_joules'],
    df_relative[~df_relative['is_parallel']]['rel_energy_gpu_total_joules']
)

print(f"转换前 p值: {p_before:.4f}")
print(f"转换后 p值: {p_after:.4f}")
print(f"效应减弱: {(1 - p_after/p_before)*100:.1f}%")
```

**验证3: 相对值分布合理性**
```python
# 相对值应该正负分布
for col in rel_energy_cols:
    positive_pct = (df_relative[col] > 0).sum() / len(df_relative)
    negative_pct = (df_relative[col] < 0).sum() / len(df_relative)

    print(f"{col}: 正值{positive_pct:.1%}, 负值{negative_pct:.1%}")

    # 检查异常值（超过基准±300%）
    outliers = abs(df_relative[col]) > baseline[col] * 3
    if outliers.sum() > 0:
        print(f"  ⚠️ {outliers.sum()} 个异常值")
```

**验证4: 数据完整性**
```python
# 记录数保持不变
assert len(df_relative) == len(df_original)

# 无新增缺失值
for col in rel_energy_cols:
    assert df_relative[col].isna().sum() == df_original[col.replace('rel_', 'energy_')].isna().sum()
```

---

## 5️⃣ Subagent评估结果总结 ⭐⭐⭐⭐

### 5.1 总体评价

**评分**: ⭐⭐⭐⭐/5 (推荐，但需改进)

### 5.2 核心优势

1. ✅ **正确识别系统性偏差** - 并行模式确实显著影响能耗
2. ✅ **理论合理** - 相对值转换适合DiBS因果分析
3. ✅ **变量选择正确** - 能耗转换，温度/利用率保留
4. ✅ **适合DiBS** - 消除基础偏差后，因果图会更清晰

### 5.3 主要风险与缓解

| 风险 | 影响 | 缓解方案 | 状态 |
|------|------|---------|------|
| **基准值稳定性不足** | 高 | 使用稳健基准值计算（中位数+离群值剔除） | ✅ 已采纳 |
| **Group2/4数据来源** | 中 | 验证data.csv能耗数据一致性，文档标注 | ✅ 已验证 |
| **绝对值信息损失** | 中 | 保留原始数据，双轨并行 | ⚠️ 可选 |
| **非线性效应线性化** | 低 | 分组分析，避免跨组对比 | ✅ 已采纳 |

### 5.4 关键改进建议（已采纳）

1. ✅ **稳健基准值计算** - 使用中位数代替平均值，剔除离群值
2. ✅ **完整验证流程** - 添加4个验证步骤
3. ✅ **文档记录Group2/4数据来源** - 标注从data.csv恢复
4. ⚠️ **双轨保留数据**（可选）- 同时保留绝对值和相对值

---

## 6️⃣ 预期效果

### 6.1 DiBS分析改进

**转换前**:
```
能耗分布特征:
- 并行模式: 平均能耗 = 500,000 J
- 非并行模式: 平均能耗 = 300,000 J
- is_parallel → energy_gpu_total_joules (强相关, p<0.001)

DiBS可能错误学习:
- is_parallel → energy (强边) ❌ 系统偏差
- learning_rate → energy (弱边) ⚠️ 被掩盖
```

**转换后**:
```
相对能耗分布特征:
- 并行模式: 平均相对能耗 ≈ 0 J (以并行基准为0)
- 非并行模式: 平均相对能耗 ≈ 0 J (以非并行基准为0)
- is_parallel → rel_energy (相关性大幅减弱, p>0.05)

DiBS应该学习:
- is_parallel → rel_energy (弱边或无边) ✅ 偏差消除
- learning_rate → rel_energy (强边) ✅ 真实因果
- batch_size → rel_energy (强边) ✅ 真实因果
```

### 6.2 可解释性提升

**相对值语义**:
- `rel_energy_gpu_total_joules = +50,000` → 相比默认配置，GPU多消耗50,000焦耳（约+10%）
- `rel_energy_gpu_total_joules = -30,000` → 相比默认配置，GPU节省30,000焦耳（约-6%）

**应用示例**:
```
能耗优化建议:
1. learning_rate 从 0.1 降到 0.05 → 节省 50,000J GPU能耗
2. batch_size 从 128 降到 64 → 节省 30,000J GPU能耗
3. 总节省: 80,000J (相比默认配置)
```

---

## 7️⃣ 实施计划

### 7.1 脚本设计

**主脚本**: `scripts/generate_relative_value_data.py`

**功能模块**:
1. **加载基准值** (`load_baseline_values()`)
   - 从`identified_default_experiments.json`加载默认值实验
   - 计算稳健基准值（中位数+离群值剔除）

2. **转换数据** (`convert_to_relative_values()`)
   - 对每个能耗/功率列计算相对值
   - 保留温度/利用率绝对值
   - 删除原始能耗列

3. **验证数据** (`validate_conversion()`)
   - 检查默认值实验相对值是否为0
   - 检查is_parallel效应是否减弱
   - 检查相对值分布合理性
   - 生成验证报告

4. **保存数据** (`save_converted_data()`)
   - 保存新的6组CSV
   - 保存`baseline_values.json`
   - 生成`conversion_report.md`和`data_dictionary.md`

### 7.2 命令行使用

```bash
# 基本用法
python3 scripts/generate_relative_value_data.py \
    --input-dir data/energy_research/6groups_final \
    --output-dir data/energy_research/6groups_relative_value \
    --baseline-file data/energy_research/identified_default_experiments.json

# 使用稳健基准值计算
python3 scripts/generate_relative_value_data.py \
    --input-dir data/energy_research/6groups_final \
    --output-dir data/energy_research/6groups_relative_value \
    --baseline-method median \
    --remove-outliers \
    --outlier-threshold 2.5

# Dry run（验证不保存）
python3 scripts/generate_relative_value_data.py \
    --input-dir data/energy_research/6groups_final \
    --dry-run
```

### 7.3 执行流程

```
步骤1: 从data.csv恢复group2和group4的默认值实验
  ↓
步骤2: 计算所有6组的稳健基准值
  ↓
步骤3: 生成相对值数据（6个CSV文件）
  ↓
步骤4: 验证转换（4个验证步骤）
  ↓
步骤5: 生成报告和数据字典
  ↓
步骤6: DiBS分析（使用相对值数据）
```

---

## 8️⃣ 关键决策记录

### 决策1: 采用相对值转换 ✅

**决定**: 将基础能耗和功率指标转换为相对值，删除绝对值

**理由**:
- 消除并行/非并行系统性偏差
- 适合DiBS因果图学习
- 相对值语义清晰（"相比默认配置的变化"）

**决策时间**: 2026-01-16
**决策者**: Green + Claude + Subagent评估

---

### 决策2: 温度和利用率保留绝对值 ✅

**决定**: 温度和利用率指标保持绝对值，不转换

**理由**:
- 温度是状态量，不是累积量
- 利用率是比例量，已经归一化（0-100%）
- 转换为相对值没有物理意义

**决策时间**: 2026-01-16
**决策者**: Green + Claude + Subagent评估

---

### 决策3: 使用稳健基准值计算 ✅

**决定**: 使用中位数+离群值剔除计算基准值

**理由**:
- 默认值实验数量少（大部分组只有1-2个）
- 中位数对离群值更稳健
- 避免极端值影响基准

**决策时间**: 2026-01-16
**决策者**: Claude（基于Subagent建议）

---

### 决策4: Group2和Group4从data.csv恢复 ✅

**决定**: 从data.csv中提取group2和group4的默认值实验能耗数据

**理由**:
- 这些实验在6groups中缺失（因为缺少性能指标）
- 但能耗数据完整，可用作基准
- 不影响DiBS能耗分析

**决策时间**: 2026-01-16
**决策者**: Green + Claude

---

## 9️⃣ 风险管理

### 风险1: 基准值不稳定 ⚠️

**影响**: 如果基准值有±10%噪声，相对值可能有100%误差

**缓解措施**:
- ✅ 使用中位数代替平均值
- ✅ 剔除离群值（z-score > 2.5）
- ✅ 检查变异系数（CV < 20%）
- ✅ 使用多个默认值实验的平均（Group1有10个）

**监控**: 在`baseline_values.json`中记录CV和样本量

---

### 风险2: Group2和Group4数据质量 ⚠️

**影响**: 这两组的默认值实验来自data.csv，需验证数据一致性

**缓解措施**:
- ✅ 验证data.csv和raw_data.csv能耗数据一致性
- ✅ 文档中标注数据来源
- ✅ 只用于基准计算，不参与DiBS训练

**验证脚本**:
```python
# 验证Group2和Group4的能耗数据一致性
def validate_group2_group4_data():
    data_csv = pd.read_csv('data/data.csv')
    raw_csv = pd.read_csv('data/raw_data.csv')

    # 对比能耗字段
    for metric in ['energy_gpu_total_joules', 'energy_cpu_total_joules']:
        diff = abs(data_csv[metric] - raw_csv[metric])
        max_diff = diff.max()

        if max_diff > 1.0:  # 允许浮点误差
            print(f"⚠️ {metric} 数据不一致! 最大差异: {max_diff}")
        else:
            print(f"✅ {metric} 数据一致")
```

---

## 🔟 下一步行动

### 立即任务（优先级：高）⭐⭐⭐

1. **实现数据转换脚本**
   - 文件: `scripts/generate_relative_value_data.py`
   - 功能: 加载基准值、转换数据、验证、保存
   - 预估时间: 2-3小时

2. **生成相对值数据**
   - 运行转换脚本
   - 生成6组CSV + baseline_values.json
   - 预估时间: 10分钟

3. **数据验证**
   - 运行4个验证步骤
   - 生成验证报告
   - 预估时间: 30分钟

### 后续任务（优先级：中）

4. **DiBS分析**
   - 使用相对值数据运行DiBS
   - 检查is_parallel边是否消失
   - 预估时间: 1-2小时

5. **效果评估**
   - 对比转换前后的DiBS结果
   - 生成对比报告
   - 预估时间: 1小时

---

## 📚 参考文档

- [identified_default_experiments.json](../data/energy_research/identified_default_experiments.json) - 默认值实验识别结果
- [defaults_by_group.json](../data/energy_research/defaults_by_group.json) - 默认超参数定义
- [QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md](QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md) - DiBS分析方案
- [6GROUPS_DATA_DESIGN_CORRECT_20260115.md](reports/6GROUPS_DATA_DESIGN_CORRECT_20260115.md) - 6分组数据设计

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-01-16 | 初始版本：确认转换规则、基准值计算方法、验证流程 | Green + Claude |
| v1.0 | 2026-01-16 | 添加Subagent评估结果（⭐⭐⭐⭐评分） | Claude |
| v1.0 | 2026-01-16 | 采纳稳健基准值计算建议（中位数+离群值剔除） | Claude |

---

**文档状态**: ✅ 方案确认，待实施
**维护者**: Green + Claude
**下次更新**: 完成数据转换后更新实际结果
