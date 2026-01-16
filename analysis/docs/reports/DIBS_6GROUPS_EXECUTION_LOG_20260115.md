# DiBS 6分组因果分析执行日志

**日期**: 2026-01-15
**任务**: 使用DiBS对新生成的6组数据进行因果分析
**状态**: 🔄 进行中

---

## 📋 任务背景

### 目标
对新生成的6组能耗研究数据执行DiBS因果发现分析：
1. **Question 1**: 超参数对能耗的影响
2. **Question 2**: 能耗与性能的权衡关系
3. **Question 3**: 中介变量的中介效应

### 数据源
- **数据位置**: `analysis/data/energy_research/dibs_training/group*.csv`
- **数据生成**: 2026-01-15 18:03:17
- **总样本数**: 423 (分布在6组)
- **关键更新**: ✅ 包含 `is_parallel` 控制变量

---

## 🔍 发现的关键问题

### 问题1: numpy 2.x兼容性 ⚠️

**症状**: DiBS执行失败，错误信息：
```
numpy boolean subtract, the `-` operator, is not supported,
use the bitwise_xor, the `^` operator, or the logical_xor function instead.
```

**根本原因**:
- 项目环境使用 numpy 2.4.1
- DiBS库设计时基于 numpy 1.x
- numpy 2.x引入了重大变更：不再支持布尔数组的减法操作 (`1 - bool_array`)

**影响范围**:
- 所有历史DiBS运行都失败 (0/6成功率)
- 问题存在于DiBS库的多个核心文件中

---

## 🛠️ 修复方案

### 修复的文件 (4个) ⭐ 最终完整修复

#### 1. `/tmp/dibs/dibs/utils/func.py`
**位置**: Line 144
**问题代码**:
```python
submat = mask * m + (1 - mask) * jnp.eye(n_vars)
```
**修复后**:
```python
# Fix for numpy 2.x: convert boolean mask to float before subtraction
mask_float = mask.astype(jnp.float32)
submat = mask_float * m + (1 - mask_float) * jnp.eye(n_vars)
```

#### 2. `/tmp/dibs/dibs/models/linearGaussian.py`
**位置**: Lines 83-88
**问题代码**:
```python
x = x * (1 - interv_targets[..., j, None])
N = (1 - interv_targets[..., j]).sum()
x_center = (x - x_bar) * (1 - interv_targets[..., j, None])
```
**修复后**:
```python
# Fix for numpy 2.x: convert boolean to float before subtraction
interv_j = interv_targets[..., j].astype(jnp.float32)
interv_j_expanded = interv_j[..., None]
x = x * (1 - interv_j_expanded)
N = (1 - interv_j).sum()
x_center = (x - x_bar) * (1 - interv_j_expanded)
```

#### 3. `/tmp/dibs/dibs/inference/dibs.py`
**位置**: Line 224
**问题代码**:
```python
log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p
```
**修复后**:
```python
# Fix for numpy 2.x: convert boolean/int to float before subtraction
single_g_float = single_g.astype(jnp.float32)
log_prob_g_ij = single_g_float * log_p + (1 - single_g_float) * log_1_p
```

#### 4. `analysis/utils/causal_discovery.py` ⭐ 我们自己的代码
**位置**: Lines 205-208
**问题代码**:
```python
col_range = data.iloc[:, i].max() - data.iloc[:, i].min()
```
**根本原因**: `is_parallel` 列是布尔类型，`max()`返回True，`min()`返回False，相减触发numpy 2.x错误
**修复后**:
```python
# Fix for numpy 2.x: convert to float before subtraction (handles boolean columns)
col_max = float(data.iloc[:, i].max())
col_min = float(data.iloc[:, i].min())
col_range = col_max - col_min
```

---

## 📊 执行记录

### 失败的尝试 (修复前)

| 时间 | 日志文件 | 结果 | 错误 |
|------|---------|------|------|
| 18:07 | `dibs_6groups_run_20260115_180713.log` | 0/6成功 | numpy boolean subtract error |
| 18:09 | `dibs_6groups_run_20260115_180944.log` | 0/6成功 | numpy boolean subtract error |
| 18:11 | `dibs_6groups_run_20260115_181111.log` | 0/6成功 | numpy boolean subtract error |
| 18:13 | `dibs_6groups_run_20260115_181322.log` | 0/6成功 | numpy boolean subtract error |

### 最终执行 (所有修复应用) ✅

| 时间 | 日志文件 | 进程ID | 状态 |
|------|---------|--------|------|
| 18:24+ | `dibs_6groups_final_20260115_*.log` | 3374055 | 🔄 运行中 |

**应用的修复**:
- ✅ DiBS库3处numpy 2.x兼容性修复
- ✅ 我们自己代码1处布尔列处理修复

**预计运行时间**: 40-90分钟（取决于数据复杂度）

---

## 📁 相关文件

### 脚本文件
- **主执行脚本**: `scripts/run_dibs_on_new_6groups.py`
- **数据生成脚本**: `scripts/generate_dibs_6groups_from_data_csv.py`

### 数据文件
- **Group 1**: `data/energy_research/dibs_training/group1_examples.csv` (126 samples, 19 features)
- **Group 2**: `data/energy_research/dibs_training/group2_vulberta.csv` (52 samples, 17 features)
- **Group 3**: `data/energy_research/dibs_training/group3_person_reid.csv` (118 samples, 20 features)
- **Group 4**: `data/energy_research/dibs_training/group4_bug_localization.csv` (40 samples, 18 features)
- **Group 5**: `data/energy_research/dibs_training/group5_mrt_oast.csv` (46 samples, 17 features)
- **Group 6**: `data/energy_research/dibs_training/group6_resnet.csv` (41 samples, 19 features)

### 配置信息
```json
{
  "generation_time": "2026-01-15 18:03:17",
  "input_file": "/home/green/energy_dl/nightly/data/data.csv",
  "total_samples": 423,
  "total_tasks": 6,
  "successful_tasks": 6,
  "control_variables": ["duration_seconds", "is_parallel", "num_mutated_params"]
}
```

---

## ✅ is_parallel 变量验证

### 数据分布验证

所有6个分组都成功包含 `is_parallel` 控制变量：

| 分组 | 样本数 | 并行模式 | 非并行模式 |
|------|--------|---------|-----------|
| group1_examples | 126 | 62 (49.2%) | 64 (50.8%) |
| group2_vulberta | 52 | 32 (61.5%) | 20 (38.5%) |
| group3_person_reid | 118 | 72 (61.0%) | 46 (39.0%) |
| group4_bug_localization | 40 | 20 (50.0%) | 20 (50.0%) |
| group5_mrt_oast | 46 | 21 (45.7%) | 25 (54.3%) |
| group6_resnet | 41 | 28 (68.3%) | 13 (31.7%) |

✅ **数据分布合理，各组都包含并行和非并行样本**

---

## 📝 经验教训

### 1. 依赖库版本兼容性
- 在使用第三方库时，需要注意版本兼容性
- numpy 2.x引入了多个不向后兼容的变更（特别是布尔数组操作）
- **关键发现**：不仅第三方库需要修复，我们自己的代码也需要适配numpy 2.x
- 建议在项目文档中明确记录依赖版本和兼容性问题

### 2. 历史运行结果验证
- 发现所有历史DiBS运行（包括2026-01-05）都失败了（0/6成功率）
- **教训**：在复用"成功"的代码前，必须验证它是否真的成功过
- 应该定期检查关键任务的成功率
- 失败的运行应该有明确的错误日志和修复记录

### 3. 布尔数据类型处理
- **新增问题**：`is_parallel` 布尔列在pandas中的max()/min()操作
- numpy 2.x不允许 `True - False` 这样的布尔减法
- **解决方案**：在算术操作前显式转换为float类型
- 这是添加新控制变量时容易忽略的问题

### 4. 调试策略
- 问题定位顺序：
  1. 查看是否有历史成功记录（本次发现都失败）
  2. 检查环境变化（numpy版本升级）
  3. 启用完整堆栈跟踪定位精确位置
  4. 从错误信息向上溯源，检查所有涉及的代码路径

---

## 🔮 下一步计划

1. **✅ 完成**: numpy 2.x兼容性修复（DiBS库 + 我们的代码）
2. **🔄 进行中**: 监控DiBS分析运行（进程 3374055）
3. **⏳ 待执行**: 验证修复效果 - 检查是否所有6组都成功完成
4. **⏳ 待执行**: 生成分析报告 - 基于DiBS结果提取因果证据
5. **⏳ 待执行**: 更新项目主文档 - 记录numpy兼容性问题和解决方案
6. **⏳ 备选方案**: 如果DiBS仍有问题，使用回归分析作为备选方法

---

## 🎯 核心发现总结

### 问题定位过程
1. **初始误判**: 以为只是DiBS库的问题
2. **第一轮修复**: 修复了DiBS库的3处numpy 2.x不兼容代码
3. **持续失败**: 修复后仍然全部失败
4. **启用调试**: 添加完整堆栈跟踪
5. **真相大白**: 发现我们自己的 `causal_discovery.py` 中有布尔列处理问题
6. **完整修复**: 4处修复全部完成

### 关键教训
- ✅ 环境升级（numpy 2.x）影响范围比预期更广
- ✅ 不仅第三方库需要适配，自己的代码也需要检查
- ✅ 添加新的布尔类型控制变量（`is_parallel`）暴露了潜在问题
- ✅ 历史"成功"记录需要验证（本次发现从未真正成功过）

---

**记录人**: Claude Code
**最后更新**: 2026-01-15 18:27
**状态**: ✅ 所有修复完成，DiBS正在后台运行
**修复总数**: 4处（DiBS库3处 + 我们的代码1处）
