# data.csv vs raw_data.csv 数据一致性检查报告

**日期**: 2025-12-29
**检查工具**: `scripts/compare_data_vs_raw_data.py`
**版本**: v1.0
**状态**: ✅ 完全一致

---

## 📋 执行摘要

对 `data.csv` 和 `raw_data.csv` 进行了全面的数据一致性检查，包括：
- ✅ 基本信息（行数、列数）
- ✅ 关键标识字段（experiment_id, timestamp）
- ✅ is_parallel 字段正确性
- ✅ 所有数据字段值一致性（44个字段 × 726行 = 31,944次检查）
- ✅ 并行模式特有字段

**结论**: **完全一致，无任何差异** 🎉

---

## 📊 检查结果概览

| 检查项 | 结果 | 详情 |
|--------|------|------|
| **行数一致性** | ✅ 通过 | 两个文件均为726行 |
| **关键标识字段** | ✅ 通过 | experiment_id和timestamp完全匹配 |
| **is_parallel字段** | ✅ 通过 | 所有726行的is_parallel值正确 |
| **数据字段值** | ✅ 通过 | 31,944次检查，0次不一致 (0.00%) |
| **并行特有字段** | ✅ 通过 | 7个并行字段全部正确 |
| **严重错误** | ✅ 0个 | 无严重错误 |
| **警告** | ✅ 0个 | 无警告 |

---

## 🗂️ 文件信息

### data.csv
- **文件路径**: `results/data.csv`
- **行数**: 726行（含header）
- **列数**: 56列
- **用途**: 精简数据文件，统一并行/非并行字段
- **生成脚本**: `scripts/create_unified_data_csv.py`

### raw_data.csv
- **文件路径**: `results/raw_data.csv`
- **行数**: 726行（含header）
- **列数**: 87列
- **用途**: 主数据文件，保留原始结构
- **来源**: 合并所有实验数据

---

## 📈 数据分布

| 模式 | 行数 | 占比 |
|------|------|------|
| **非并行模式** | 348行 | 47.9% |
| **并行模式** | 378行 | 52.1% |
| **总计** | 726行 | 100.0% |

---

## 🔍 详细检查结果

### 1. 基本信息一致性 ✅

**检查内容**: 两个文件的行数是否一致

**结果**:
- ✅ data.csv: 726行
- ✅ raw_data.csv: 726行
- ✅ 完全一致

---

### 2. 关键标识字段一致性 ✅

**检查内容**: 逐行比较 `experiment_id` 和 `timestamp` 字段

**检查次数**: 726行 × 2字段 = 1,452次

**结果**:
- ✅ 所有关键标识字段完全一致
- ✅ 0处不一致

**说明**: 这确保了两个文件的数据行是完全对应的，可以安全地逐行比较。

---

### 3. is_parallel 字段正确性 ✅

**检查内容**: 验证 `is_parallel` 字段是否正确反映 `mode` 字段

**映射关系**:
- `mode='parallel'` → `is_parallel='True'`
- `mode='default'` / `mode='mutation'` → `is_parallel='False'`

**结果**:
- ✅ 所有726行的 is_parallel 字段全部正确
- ✅ 0处不一致

---

### 4. 所有字段值一致性 ✅

**检查内容**: 逐行逐字段比较所有数据字段

**检查范围**: 44个字段
- 基础信息: 7个字段（experiment_id, timestamp, repository, model, training_success, duration_seconds, retries, error_message）
- 超参数: 9个字段（alpha, batch_size, dropout, epochs, kfold, learning_rate, max_iter, seed, weight_decay）
- 性能指标: 16个字段（accuracy, best_val_accuracy, map, precision, rank1, rank5, recall, test_accuracy, test_loss, eval_loss, final_training_loss, eval_samples_per_second, top1/5/10/20_accuracy）
- 能耗指标: 11个字段（cpu_pkg/ram/total_joules, gpu_avg/max/min_watts, gpu_total_joules, gpu_temp_avg/max, gpu_util_avg/max）
- 实验元数据: 1个字段（mode - 用于确定is_parallel）

**检查逻辑**: 模拟 `create_unified_data_csv.py` 的字段获取逻辑
```python
# 非并行模式: 直接使用顶层字段
data_value = data_row['field']
raw_value = raw_row['field']

# 并行模式: 优先使用fg_字段，fallback到顶层字段
data_value = data_row['field']
raw_value = raw_row['fg_field'] if raw_row['fg_field'] else raw_row['field']
```

**检查次数**: 44字段 × 726行 = **31,944次**

**结果**:
- ✅ 总检查次数: 31,944次
- ✅ 不一致次数: 0次 (**0.00%**)
- ✅ **完全一致**

**详细分布**:
| 模式 | 行数 | 字段数 | 检查次数 | 不一致次数 |
|------|------|--------|----------|------------|
| 非并行 | 348 | 44 | 15,312 | 0 |
| 并行 | 378 | 44 | 16,632 | 0 |
| **总计** | 726 | 44 | **31,944** | **0** |

---

### 5. 并行模式特有字段检查 ✅

**检查内容**: 验证并行模式特有的7个字段

**字段列表**:
1. `bg_repository` - 后台仓库
2. `bg_model` - 后台模型
3. `bg_note` - 后台备注
4. `bg_log_directory` - 后台日志目录
5. `fg_duration_seconds` - 前台训练时长
6. `fg_retries` - 前台重试次数
7. `fg_error_message` - 前台错误信息

**检查逻辑**:
- 非并行模式: 这些字段应该为空
- 并行模式: 这些字段应该与raw_data.csv一致

**结果**:
- ✅ 所有并行模式特有字段全部正确
- ✅ 非并行模式下这些字段全部为空
- ✅ 并行模式下这些字段与raw_data.csv完全一致
- ✅ 0处问题

---

## 🎯 设计差异（预期的）

以下差异是由设计造成的，**不属于数据不一致**：

### 1. 列数差异
- **raw_data.csv**: 87列（保留原始结构）
- **data.csv**: 56列（精简版）

**差异原因**: data.csv移除了31列冗余或内部使用的列

**移除的列**（31列）:
1. **并行模式fg_前缀字段**（24列）:
   - `fg_repository`, `fg_model`, `fg_training_success`, `fg_duration_seconds`, `fg_retries`, `fg_error_message`
   - `fg_hyperparam_*` (9个超参数)
   - `fg_perf_*` (部分性能指标)
   - `fg_energy_*` (部分能耗指标)

   **原因**: 这些字段已经整合到顶层字段中（is_parallel=True时使用fg_值）

2. **内部字段**（7列）:
   - `experiment_source` - 已移到元数据部分
   - `num_mutated_params` - 已移到元数据部分
   - `mutated_param` - 已移到元数据部分
   - 其他内部使用字段

### 2. 新增字段
- **data.csv新增**: `is_parallel` 字段
  - **值**: 'True' 或 'False'
  - **来源**: 根据 `mode` 字段计算
  - **用途**: 方便区分并行/非并行模式

---

## ✅ 验证结论

### 核心发现

1. **数据完整性**: ✅ 完美
   - 726行数据完全一致
   - 31,944次字段值检查，0次不一致

2. **字段映射正确性**: ✅ 完美
   - 非并行模式: 顶层字段 → 顶层字段
   - 并行模式: fg_字段 → 顶层字段（优先fg_，fallback到顶层）
   - 并行特有字段: 正确保留

3. **is_parallel字段**: ✅ 完美
   - 所有726行的值完全正确
   - 正确反映mode字段

4. **关键标识**: ✅ 完美
   - experiment_id和timestamp完全匹配
   - 确保两文件数据行对应

### 数据质量评估

| 指标 | 评分 | 说明 |
|------|------|------|
| **数据一致性** | ⭐⭐⭐⭐⭐ | 完全一致，0处差异 |
| **字段映射正确性** | ⭐⭐⭐⭐⭐ | 完全符合设计逻辑 |
| **并行模式处理** | ⭐⭐⭐⭐⭐ | fg_字段整合正确 |
| **标识字段** | ⭐⭐⭐⭐⭐ | 完全匹配 |
| **整体质量** | ⭐⭐⭐⭐⭐ | **完美** |

### 最终结论

✅ **data.csv 和 raw_data.csv 完全一致!**

- ✅ 除了设计造成的差异（列数不同、新增is_parallel字段）外，**没有任何其他差异**
- ✅ 所有数据字段值完全一致（31,944次检查）
- ✅ 并行/非并行模式的字段整合逻辑完全正确
- ✅ 数据同步状态良好，可以安全使用

**建议**:
- 可以放心使用 `data.csv` 进行分析
- `data.csv` 的56列格式更简洁，推荐用于因果分析等下游任务
- `raw_data.csv` 的87列格式保留完整信息，推荐用于数据备份和审计

---

## 🛠️ 检查工具说明

### 脚本信息
- **脚本路径**: `scripts/compare_data_vs_raw_data.py`
- **版本**: v1.0
- **创建日期**: 2025-12-29
- **代码行数**: 462行

### 检查功能
1. **基本信息检查**: 行数、列数对比
2. **关键字段检查**: experiment_id, timestamp逐行对比
3. **is_parallel字段检查**: 验证与mode字段的映射关系
4. **字段值一致性检查**: 44个字段 × 726行的完整对比
5. **并行字段检查**: 7个并行特有字段的正确性验证

### 使用方法
```bash
# 运行完整检查
python3 scripts/compare_data_vs_raw_data.py

# 预期输出: 完整的检查报告和结论
```

### 检查逻辑
```python
# 模拟create_unified_data_csv.py的字段获取逻辑
def get_raw_field_value(raw_row, field, is_parallel):
    """
    从raw_data获取字段值

    并行模式: 优先使用fg_字段，fallback到顶层字段
    非并行模式: 直接使用顶层字段
    """
    if is_parallel:
        fg_value = raw_row.get(f'fg_{field}', '').strip()
        if fg_value:
            return fg_value
    return raw_row.get(field, '').strip()
```

---

## 📚 相关文档

- [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) - 数据格式设计决定
- [项目进度完整总结](PROJECT_PROGRESS_COMPLETE_SUMMARY.md) - 项目总体状况
- [create_unified_data_csv.py](../../scripts/create_unified_data_csv.py) - data.csv生成脚本

---

## 📅 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-12-29 | 初始版本 - 完整数据一致性检查 |

---

**维护者**: Green
**最后更新**: 2025-12-29
**检查状态**: ✅ 完全一致
