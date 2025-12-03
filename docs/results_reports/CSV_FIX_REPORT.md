# summary_all.csv 格式修复报告

**修复日期**: 2025-12-03
**问题**: GitHub报错 "row 321 should actually have 37 columns, instead of 29 in line 320"

---

## 问题分析

### 根本原因

Stage1 (run_20251202_185830) 追加到 summary_all.csv 的12行数据只有29列，而标准格式需要37列。

**缺少的8个列**:
1. `hyperparam_alpha` (索引7)
2. `hyperparam_kfold` (索引11)
3. `hyperparam_max_iter` (索引13)
4. `hyperparam_weight_decay` (索引15)
5. `perf_best_val_accuracy` (索引17)
6. `perf_test_accuracy` (索引23)
7. `perf_test_loss` (索引24)
8. `experiment_source` (索引36)

### 受影响的行

- **行范围**: 第321-332行 (共12行)
- **实验**: Stage1运行的12个实验结果
- **症状**: 这些行只有29列而非37列，导致CSV格式不一致

### 为什么会发生

Stage1的 `summary.csv` 生成时使用的列模板与 `summary_all.csv` 的标准模板不一致：
- `summary_all.csv`: 37列 (完整模板，包含所有可能的超参数和性能指标)
- `stage1 summary.csv`: 29列 (只包含Stage1实验中实际使用的列)

追加时直接将29列数据写入，没有对齐到37列格式。

---

## 修复过程

### 1. 备份

```bash
cp results/summary_all.csv results/summary_all.csv.backup
```

### 2. 列映射

创建了stage1列到summary_all列的映射关系：

| Stage1列 | 索引 | → | Summary_all列 | 索引 |
|---------|------|---|--------------|------|
| experiment_id | 0 | → | experiment_id | 0 |
| timestamp | 1 | → | timestamp | 1 |
| repository | 2 | → | repository | 2 |
| model | 3 | → | model | 3 |
| ... | ... | → | ... | ... |
| hyperparam_batch_size | 7 | → | hyperparam_batch_size | 8 |
| hyperparam_dropout | 8 | → | hyperparam_dropout | 9 |
| hyperparam_epochs | 9 | → | hyperparam_epochs | 10 |
| ... | ... | → | ... | ... |

**关键点**: Stage1的列在summary_all中不是连续的，需要跳过8个缺失列的位置。

### 3. 重新映射数据

对每一行29列数据：
1. 创建37列的空行
2. 根据列映射关系，将29列数据填入正确位置
3. 缺失的8列位置保持空值

### 4. 写回文件

使用Python csv模块安全写回：
```python
with open('results/summary_all.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # 37列header
    writer.writerows(all_rows)  # 所有行都是37列
```

---

## 修复结果

### 修复统计

- ✅ **修复行数**: 12行 (第321-332行)
- ✅ **总行数**: 331行数据 + 1行header = 332行
- ✅ **所有行列数**: 37列
- ✅ **CSV格式**: 完全一致

### 验证结果

```bash
✓ CSV Header: 37 columns
✓ Total rows: 331
✓ All 331 rows have correct 37 columns
✓ CSV解析成功
✓ GitHub应该可以正常显示
```

### 第321行修复示例

**修复前** (29列):
```csv
MRT-OAST_default_001,2025-12-02T19:20:10.686267,MRT-OAST,default,True,1300.203...,0,,0.0427...,,...
```
- 列错位，缺少8个空列
- GitHub无法解析

**修复后** (37列):
```csv
MRT-OAST_default_001,2025-12-02T19:20:10.686267,MRT-OAST,default,True,1300.203...,0,,,0.0427...,,,,,...
```
- 所有列对齐到正确位置
- 缺失列填充空值
- GitHub可以正常解析

---

## 修复后的列结构

### 第321行 (MRT-OAST_default_001) 详细信息

**非空字段** (18个):
```
experiment_id: MRT-OAST_default_001
timestamp: 2025-12-02T19:20:10.686267
repository: MRT-OAST
model: default
training_success: True
duration_seconds: 1300.2032346725464
retries: 0
hyperparam_dropout: 0.04270586130294716
perf_accuracy: 4934.0
perf_precision: 0.999124
perf_recall: 0.944146
energy_cpu_pkg_joules: 38422.61
energy_cpu_ram_joules: 2977.18
energy_cpu_total_joules: 41399.79
energy_gpu_avg_watts: 250.51941971383167
energy_gpu_max_watts: 318.71
energy_gpu_min_watts: 68.4
energy_gpu_total_joules: 315153.4300000002
energy_gpu_temp_avg_celsius: 78.25039745627981
energy_gpu_temp_max_celsius: 84.0
energy_gpu_util_avg_percent: 92.18124006359301
energy_gpu_util_max_percent: 98.0
```

**空字段** (19个，符合预期):
```
hyperparam_alpha (MRT-OAST不使用)
hyperparam_batch_size (MRT-OAST不使用)
hyperparam_epochs (此次实验未变异)
hyperparam_kfold (MRT-OAST不使用)
hyperparam_learning_rate (此次实验未变异)
hyperparam_max_iter (MRT-OAST不使用)
hyperparam_seed (此次实验未变异)
hyperparam_weight_decay (MRT-OAST不使用)
perf_best_val_accuracy (MRT-OAST不记录)
perf_map (MRT-OAST不记录)
perf_rank1 (MRT-OAST不记录)
perf_rank5 (MRT-OAST不记录)
perf_test_accuracy (MRT-OAST不记录)
perf_test_loss (MRT-OAST不记录)
experiment_source (未记录)
```

---

## 预防措施

### 问题根源

`mutation.py` 的 `_append_to_summary_all()` 方法没有确保列对齐。

### 建议改进

1. **统一列模板**: 所有session的summary.csv应使用相同的37列模板
2. **追加时验证**: 追加前检查列数和列名是否匹配
3. **使用DictWriter**: 用字典方式写入，自动处理列对齐
4. **单元测试**: 添加CSV格式一致性测试

### 代码建议

```python
# 建议的改进方案
def _append_to_summary_all(self, session_csv):
    # 读取summary_all的header作为标准
    with open(self.summary_all_path, 'r') as f:
        reader = csv.DictReader(f)
        standard_fieldnames = reader.fieldnames

    # 使用DictReader读取session数据
    with open(session_csv, 'r') as f:
        reader = csv.DictReader(f)
        session_rows = list(reader)

    # 使用DictWriter追加，自动对齐列
    with open(self.summary_all_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=standard_fieldnames)
        writer.writerows(session_rows)  # 缺失的列自动填空
```

---

## 备份文件

原始文件已备份至:
```
results/summary_all.csv.backup
```

如需恢复原始文件：
```bash
mv results/summary_all.csv.backup results/summary_all.csv
```

---

## 总结

✅ **问题**: Stage1追加的12行数据列数不匹配（29列 vs 37列）
✅ **原因**: Session CSV与summary_all.csv列模板不一致
✅ **修复**: 重新映射所有29列数据到正确的37列位置
✅ **验证**: 所有331行数据现在都是37列，CSV格式完全正确
✅ **结果**: GitHub可以正常解析和显示CSV文件

**修复者**: Claude Code
**修复时间**: 2025-12-03
**状态**: ✅ 完成
