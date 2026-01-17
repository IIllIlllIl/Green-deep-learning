# 差值转换方案 v2.0（模型级基准值）

**创建日期**: 2026-01-16
**状态**: 待验证
**方法**: 差值法 + 模型级基准值 + 部分保留绝对值

---

## 📋 目录

- [执行摘要](#执行摘要)
- [核心变更](#核心变更)
- [转换规则](#转换规则)
- [模型级基准值计算](#模型级基准值计算)
- [实施细节](#实施细节)
- [验证步骤](#验证步骤)
- [预期效果](#预期效果)
- [风险评估](#风险评估)

---

## 执行摘要

### 问题
并行/非并行训练模式存在系统性能耗偏差，导致DiBS因果分析时可能错误识别`is_parallel`为强因果因子。

### 解决方案
**差值转换（模型级基准值）**: 在每个模型内部，将能耗值转换为相对于该模型默认值的差值。

**转换公式**:
```
差值 = 实际值 - 模型基准值
```

其中基准值在**每个模型的并行/非并行模式内部分别计算**。

### 核心优势
1. ✅ **消除模型间的绝对值差异**: 不同模型转换到相同的"相对于默认值的偏差"尺度
2. ✅ **保留物理单位**: 差值仍然是焦耳(J)或瓦特(W)，易于解释
3. ✅ **处理模型异质性**: 每个模型有自己的基准值，避免Group1高变异问题
4. ✅ **适度保留信息**: 功率的最大/最小值保留绝对值，提供额外信息

---

## 核心变更

相比v1.0版本的关键变更：

| 维度 | v1.0（组级基准值） | v2.0（模型级基准值） |
|------|------------------|---------------------|
| **基准值粒度** | 按组计算（6组） | 按模型计算（11个模型） ⭐ |
| **转换公式** | 混乱（差值vs百分比） | 统一差值法 ⭐ |
| **功率处理** | 全部转换 | 最大/最小值保留绝对值 ⭐ |
| **Group1处理** | 高CV(>80%) | 拆分为4个模型，各自稳定 ⭐ |
| **适用范围** | 所有组 | 所有模型（11个） |

---

## 转换规则

### 转换字段（4个能量字段）

**转换为差值**:
```
差值 = 实际值 - 模型基准值
```

| 字段 | 单位 | 转换后含义 |
|------|------|-----------|
| `energy_cpu_pkg_joules` | J | 相对于默认值的CPU包能耗差异 |
| `energy_cpu_ram_joules` | J | 相对于默认值的RAM能耗差异 |
| `energy_cpu_total_joules` | J | 相对于默认值的CPU总能耗差异 |
| `energy_gpu_total_joules` | J | 相对于默认值的GPU总能耗差异 |

**示例**:
```
某模型默认GPU能耗（并行）: 50,000 J
某次实验实际GPU能耗: 55,000 J
转换后差值: 55,000 - 50,000 = +5,000 J （表示高出默认值5000焦耳）
```

### 转换字段（1个功率字段）

**转换为差值**:
```
差值 = 实际值 - 模型基准值
```

| 字段 | 单位 | 转换后含义 |
|------|------|-----------|
| `energy_gpu_avg_watts` | W | 相对于默认值的平均功率差异 |

### 保留绝对值字段（6个字段）

**不转换，保留原始值**:

| 字段 | 单位 | 原因 |
|------|------|------|
| `energy_gpu_max_watts` | W | 峰值功率是硬件限制指标，保留绝对值更有意义 |
| `energy_gpu_min_watts` | W | 最低功率反映空闲状态，保留绝对值更有意义 |
| `energy_gpu_temp_avg_celsius` | °C | 温度是绝对物理量 |
| `energy_gpu_temp_max_celsius` | °C | 温度是绝对物理量 |
| `energy_gpu_util_avg_percent` | % | 利用率是绝对百分比 |
| `energy_gpu_util_max_percent` | % | 利用率是绝对百分比 |

**理由**:
- 最大/最小功率反映硬件性能边界，不同实验间的绝对值可比性更重要
- 温度和利用率是物理绝对量，转换为差值会失去物理意义

---

## 模型级基准值计算

### 模型列表及默认值实验数量

| 组 | 模型 | 并行默认值实验 | 非并行默认值实验 | 总计 | 稳定性 |
|----|------|--------------|----------------|------|--------|
| **Group1** | mnist | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group1** | mnist_rnn | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group1** | siamese | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group1** | mnist_ff | 1 | 3 | 4 | 🟡 可接受 |
| **Group2** | mlp | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group3** | densenet121 | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group3** | hrnet18 | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group3** | pcb | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group4** | default | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group5** | default | 1 | 1 | 2 | ⚠️ 样本不足 |
| **Group6** | resnet20 | 1 | 1 | 2 | ⚠️ 样本不足 |

**观察**:
- ✅ 所有模型都有至少1个并行+1个非并行默认值实验
- ⚠️ 大部分模型只有1个/模式（n=1），无法计算标准差
- 🟡 只有mnist_ff有3个非并行实验（n=3）

### 基准值计算方法

**对于n=1的情况**（大部分模型）:
```python
baseline = 唯一实验的能耗值
```

**对于n≥2的情况**（如mnist_ff非并行）:
```python
# 稳健方法：中位数
baseline = median(values)

# 或者：检测并移除离群值后的中位数
if n >= 3:
    z_scores = (values - mean) / std
    clean_values = values[abs(z_scores) <= 2.5]
    baseline = median(clean_values)
else:
    baseline = median(values)
```

### 基准值数据结构

```json
{
  "model_baselines": {
    "mnist": {
      "parallel": {
        "energy_cpu_pkg_joules": {"median": 12345.67, "count": 1},
        "energy_cpu_ram_joules": {"median": 1234.56, "count": 1},
        "energy_cpu_total_joules": {"median": 13580.23, "count": 1},
        "energy_gpu_total_joules": {"median": 98765.43, "count": 1},
        "energy_gpu_avg_watts": {"median": 234.56, "count": 1}
      },
      "nonparallel": {
        "energy_cpu_pkg_joules": {"median": 5678.90, "count": 1},
        ...
      }
    },
    "mnist_rnn": { ... },
    ...
  }
}
```

---

## 实施细节

### 步骤1: 重新计算模型级基准值

**输入**:
- `data/energy_research/identified_default_experiments.json` (默认值实验timestamps)
- `../data/data.csv` (完整数据)

**脚本**: `scripts/calculate_model_level_baselines.py`

**输出**: `data/energy_research/model_level_baselines.json`

**核心逻辑**:
```python
for group in groups:
    for model in group.models:
        # 获取该模型的默认值实验
        df_model_defaults = get_default_experiments(model)

        for mode in ['parallel', 'nonparallel']:
            df_mode = df_model_defaults[df_model_defaults['is_parallel'] == (mode == 'parallel')]

            for field in energy_fields:
                values = df_mode[field].dropna()

                if len(values) == 0:
                    baseline = None
                elif len(values) == 1:
                    baseline = float(values.iloc[0])
                else:
                    # n >= 2: 使用中位数
                    baseline = float(values.median())

                baselines[model][mode][field] = {
                    'median': baseline,
                    'count': len(values)
                }
```

---

### 步骤2: 生成差值数据

**输入**:
- 6groups原始数据: `data/energy_research/6groups_*.csv`
- 模型级基准值: `data/energy_research/model_level_baselines.json`

**脚本**: `scripts/generate_difference_value_data.py`

**输出**:
- 6组差值数据: `data/energy_research/6groups_*_difference.csv`
- 应用记录: `data/energy_research/model_baselines_applied.json`

**核心逻辑**:
```python
for row in df.itertuples():
    model = row.model
    is_parallel = row.is_parallel
    mode = 'parallel' if is_parallel else 'nonparallel'

    # 获取该模型的基准值
    baseline = model_baselines[model][mode]

    # 转换能量字段（4个）
    for field in ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
                  'energy_cpu_total_joules', 'energy_gpu_total_joules']:
        actual = row[field]
        base = baseline[field]['median']

        if pd.notna(actual) and base is not None:
            diff = actual - base
            row_new[field] = diff
        else:
            row_new[field] = np.nan

    # 转换平均功率字段（1个）
    field = 'energy_gpu_avg_watts'
    actual = row[field]
    base = baseline[field]['median']
    if pd.notna(actual) and base is not None:
        diff = actual - base
        row_new[field] = diff
    else:
        row_new[field] = np.nan

    # 保留绝对值字段（6个）
    for field in ['energy_gpu_max_watts', 'energy_gpu_min_watts',
                  'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
                  'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent']:
        row_new[field] = row[field]  # 不转换
```

---

## 验证步骤

### 验证1: 默认值实验的差值应≈0

**目的**: 确认默认值实验转换后差值接近0

**方法**:
```python
# 提取默认值实验
df_defaults = df_diff[df_diff['timestamp'].isin(default_timestamps)]

# 检查差值
for field in converted_fields:
    diffs = df_defaults[field].dropna()
    print(f"{field}:")
    print(f"  最小差值: {diffs.min():.2f}")
    print(f"  最大差值: {diffs.max():.2f}")
    print(f"  均值: {diffs.mean():.2f}")

    # 对于n=1的情况，差值应该恰好为0
    # 对于n>=2的情况，差值应该接近0（因为用的是中位数）
    assert abs(diffs.mean()) < 1000, "默认值实验差值异常大"
```

**预期结果**:
- 对于n=1的模型: 差值 = 0（完全相等）
- 对于n=2的模型: 差值 ≈ 0（两个值关于中位数对称）
- 对于n≥3的模型: 差值 ≈ 0（中位数附近）

---

### 验证2: 模型间差值分布可比性

**目的**: 验证不同模型转换后差值在可比范围内

**方法**:
```python
# 按模型统计差值分布
for model in models:
    df_model = df_diff[df_diff['model'] == model]

    for field in converted_fields:
        diffs = df_model[field].dropna()
        print(f"{model} - {field}:")
        print(f"  范围: {diffs.min():.0f} ~ {diffs.max():.0f}")
        print(f"  均值: {diffs.mean():.0f}")
        print(f"  标准差: {diffs.std():.0f}")
```

**预期结果**:
- 不同模型的差值范围应该在同一数量级（如 -10,000 ~ +50,000 J）
- 如果某模型差值异常大/小，需要检查基准值是否合理

---

### 验证3: 保留字段未改变

**目的**: 确认绝对值字段没有被错误转换

**方法**:
```python
# 对比原始数据和转换后数据
for field in ['energy_gpu_max_watts', 'energy_gpu_min_watts',
              'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
              'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent']:

    # 随机抽取10行对比
    sample_idx = np.random.choice(len(df_original), 10)

    original_values = df_original.iloc[sample_idx][field]
    diff_values = df_diff.iloc[sample_idx][field]

    assert (original_values == diff_values).all(), f"{field}被错误修改"
```

**预期结果**: 所有保留字段完全相同

---

### 验证4: is_parallel效应分析

**目的**: 检查转换后is_parallel对能耗的影响是否减弱

**方法**:
```python
# 对每个组，分别分析转换前后is_parallel的效应
for group in groups:
    df_group_orig = df_original[df_original['group'] == group]
    df_group_diff = df_diff[df_diff['group'] == group]

    for field in converted_fields:
        # 转换前：t检验
        parallel_orig = df_group_orig[df_group_orig['is_parallel']][field]
        nonparallel_orig = df_group_orig[~df_group_orig['is_parallel']][field]
        t_stat_orig, p_orig = stats.ttest_ind(parallel_orig, nonparallel_orig)

        # 转换后：t检验
        parallel_diff = df_group_diff[df_group_diff['is_parallel']][field]
        nonparallel_diff = df_group_diff[~df_group_diff['is_parallel']][field]
        t_stat_diff, p_diff = stats.ttest_ind(parallel_diff, nonparallel_diff)

        print(f"{group} - {field}:")
        print(f"  转换前: t={t_stat_orig:.2f}, p={p_orig:.4f}")
        print(f"  转换后: t={t_stat_diff:.2f}, p={p_diff:.4f}")
        print(f"  t统计量变化: {abs(t_stat_diff)/abs(t_stat_orig):.2%}")
```

**预期结果**:
- ⚠️ **注意**: 差值法可能无法显著减弱t统计量（见风险评估）
- 主要验证：两组的均值差异是否缩小
- 更重要的验证：在DiBS分析中is_parallel的因果效应是否减弱

---

## 预期效果

### 1. 消除模型间绝对值差异 ✅

**转换前** (Group1):
```
mnist GPU能耗:      10,000 J (绝对值)
mnist_rnn GPU能耗:  50,000 J (绝对值)
siamese GPU能耗:   100,000 J (绝对值)
→ 模型间差异巨大，DiBS可能错误学习"模型类型 → 能耗"
```

**转换后**:
```
mnist GPU能耗差值:      +2,000 J (相对于mnist默认值)
mnist_rnn GPU能耗差值:  +3,000 J (相对于mnist_rnn默认值)
siamese GPU能耗差值:    +5,000 J (相对于siamese默认值)
→ 都是"高于各自默认值的增量"，可比性强
```

---

### 2. 保留物理意义 ✅

**差值的解释**:
```
energy_gpu_total_joules = +5,000 J
→ 解释: "该超参数配置导致GPU能耗比默认配置高5000焦耳"
→ 物理意义清晰，易于理解
```

---

### 3. 处理Group1的高变异性 ✅

**问题**: v1.0中Group1的组级基准值CV > 80%

**解决**: v2.0中每个模型有自己的基准值
```
mnist基准值:     10,000 J (mnist的默认值)
mnist_rnn基准值: 50,000 J (mnist_rnn的默认值)
siamese基准值:  100,000 J (siamese的默认值)
→ 每个模型内部稳定（n=1或2，CV=0或很小）
```

---

### 4. 适度保留信息 ✅

**保留绝对值字段的优势**:
```
energy_gpu_max_watts = 319 W
→ 告诉DiBS: "该实验接近GPU硬件功率上限"
→ 可能学习到: "大batch size → 高功率 → 接近硬件限制"

energy_gpu_temp_max_celsius = 82°C
→ 告诉DiBS: "该实验温度较高"
→ 可能学习到: "高温 → 散热不良 → 需要降频"
```

---

## 风险评估

### 风险1: 差值法可能无法完全消除is_parallel偏差 ⚠️⚠️⚠️

**问题**:
- 差值法只平移分布（减去均值），不归一化尺度（不除以标准差）
- 如果并行/非并行模式的**变异幅度**不同，DiBS仍可能检测到差异

**示例**:
```
并行模式: GPU差值范围 -10,000 ~ +50,000 J (大变异)
非并行模式: GPU差值范围 -2,000 ~ +8,000 J (小变异)
→ DiBS可能学习: is_parallel → 能耗变异幅度
```

**缓解措施**:
1. ✅ **验证4**会检验这个问题
2. ✅ 如果验证4失败，考虑在差值基础上再做**标准化**:
   ```python
   z = (diff - mean(diff)) / std(diff)
   ```
3. ✅ 或在DiBS中添加**方差建模**

**评估**: 🟡 中等风险，需要通过验证4确认

---

### 风险2: n=1时基准值不可靠 ⚠️⚠️

**问题**: 大部分模型只有1个默认值实验/模式
- 如果该实验恰好是异常值（系统故障、后台干扰等），所有转换都会错误

**示例**:
```
某模型非并行默认值实验: 10,000 J
但实际该模型正常非并行能耗: 5,000 J (该默认值实验异常)
→ 所有非并行实验转换后: actual - 10,000 = 偏低
→ 可能错误学习: "该超参数总是降低能耗"
```

**缓解措施**:
1. ✅ **手动检查**默认值实验的合理性（查看日志、检查status）
2. ✅ 对比模型间基准值，识别异常模型
3. ✅ 如果发现异常，从data.csv中寻找其他"接近默认配置"的实验

**评估**: 🟡 中等风险，但可通过人工检查缓解

---

### 风险3: 模型级基准值导致样本量进一步减少 ⚠️

**问题**:
- v1.0: Group1有10个默认值实验用于计算基准值
- v2.0: mnist只有2个，mnist_rnn只有2个，...

**影响**: 基准值估计更不稳定（但模型内部更一致）

**权衡**:
```
v1.0 (组级): 基准值稳定性高 ✅，但模型间异质性大 ❌
v2.0 (模型级): 基准值稳定性低 ❌，但模型内部一致性高 ✅
```

**评估**: 🟢 可接受的权衡，模型一致性更重要

---

### 风险4: 保留字段可能引入混淆 ⚠️

**问题**:
- 能量字段转换为差值（相对量）
- 功率最大/最小值保留绝对值
- DiBS可能混淆相对量和绝对量的关系

**示例**:
```
energy_gpu_total_joules = +5,000 J (差值)
energy_gpu_max_watts = 319 W (绝对值)

DiBS可能错误学习: "GPU总能耗差值 → 最大功率绝对值"
实际上这两个变量在不同尺度上
```

**缓解措施**:
1. ✅ 在DiBS分析时**明确标注**哪些是差值、哪些是绝对值
2. ✅ 考虑对保留字段也做**标准化**（但保持单独的标准化）
3. ✅ 敏感性分析: 对比"全部转换"vs"部分保留"的结果

**评估**: 🟢 轻微风险，DiBS应该能处理不同尺度的变量

---

## 实施路线图

### 阶段1: 计算模型级基准值（1天）

1. 实现 `scripts/calculate_model_level_baselines.py`
2. 生成 `data/energy_research/model_level_baselines.json`
3. 手动检查基准值合理性

### 阶段2: 生成差值数据（1天）

1. 实现 `scripts/generate_difference_value_data.py`
2. 生成6个差值CSV文件
3. 执行验证1-3

### 阶段3: 效应验证（1-2天）

1. 执行验证4（is_parallel效应分析）
2. 如果效应未显著减弱，考虑补充标准化
3. 可视化对比转换前后的分布

### 阶段4: DiBS分析（1周）

1. 使用差值数据进行DiBS因果图学习
2. 对比差值数据 vs 原始数据的因果图
3. 评估is_parallel的因果效应是否减弱

---

## 与其他方案对比

| 方案 | 消除模型差异 | 消除模式偏差 | 物理意义 | 实施复杂度 | 推荐度 |
|------|------------|------------|---------|-----------|--------|
| **差值法(v2.0)** | ✅ 模型级 | 🟡 部分 | ✅ 清晰 | 🟢 简单 | ⭐⭐⭐⭐ |
| 百分比法(v1.0) | ✅ 组级 | ✅ 完全 | 🟡 尚可 | 🟢 简单 | ⭐⭐⭐ |
| Z-score标准化 | ✅ 任意级 | ✅ 完全 | ❌ 弱 | 🟢 简单 | ⭐⭐⭐⭐ |
| DiBS混淆因子 | ❌ 不处理 | ✅ 控制 | ✅ 保留原值 | 🟡 中等 | ⭐⭐⭐⭐⭐ |

**推荐**:
1. 先尝试**差值法v2.0**（本方案）
2. 如果验证4失败，补充**标准化**或改用**Z-score**
3. 平行尝试**DiBS混淆因子控制**作为对比

---

## 相关文档

| 文档 | 用途 |
|------|------|
| `RELATIVE_VALUE_TRANSFORMATION_PLAN.md` | v1.0方案（已废弃） |
| `RELATIVE_VALUE_TRANSFORMATION_SUMMARY.md` | v1.0总结（已废弃） |
| `data/energy_research/identified_default_experiments.json` | 默认值实验识别（按组） |
| `data/energy_research/model_level_baselines.json` | 模型级基准值（待生成） |

---

## 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|---------|
| v2.0 | 2026-01-16 | 改用差值法+模型级基准值+部分保留绝对值 |
| v1.0 | 2026-01-16 | 初始版本（百分比法+组级基准值，已废弃） |

---

**维护者**: Claude
**最后更新**: 2026-01-16
**状态**: 待Subagent验证
