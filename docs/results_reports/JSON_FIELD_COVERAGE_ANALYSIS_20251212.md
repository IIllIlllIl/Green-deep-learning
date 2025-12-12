# Experiment.json字段完整覆盖分析报告

**日期**: 2025-12-12
**版本**: 1.0
**状态**: ✅ 分析完成

---

## 执行摘要

本报告分析了老实验JSON文件的字段覆盖情况，发现**80列CSV格式不足以覆盖所有experiment.json信息**。

### 关键发现

1. **80列格式不足** ⚠️
   - 当前80列格式
   - 缺少13个JSON字段（背景超参数6个 + 背景能耗7个）
   - 需要扩展到**93列**才能完整覆盖

2. **字段映射Bug修复** ✅
   - 原映射逻辑错误地重复添加单位后缀
   - 例：`gpu_power_avg_watts` → `energy_gpu_avg_watts_watts`（错误）
   - 已修复为：`gpu_power_avg_watts` → `energy_gpu_avg_watts`（正确）

3. **转换脚本问题** ⚠️
   - `convert_summary_old_to_80col.py`未能加载JSON文件
   - 原因：老实验JSON文件路径结构与新实验不同
   - 结果：使用37列CSV数据转换，**可能丢失JSON中的额外信息**

---

## 详细分析

### 1. 数据源统计

#### JSON文件扫描结果
```
总JSON文件数: 569个
- 老实验目录: mutation_2x_20251122_175401/, default/, mutation_1x/, archived/
- 新实验目录: run_YYYYMMDD_HHMMSS/

唯一JSON字段数: 92个
- 已映射字段: 83个
- 未映射字段: 9个（均为父级对象，如 "energy_metrics", "foreground" 等）
```

#### 字段分类统计
| 类别 | 字段数 | 说明 |
|------|--------|------|
| 基础字段 | 9个 | experiment_id, timestamp, repository, model, etc. |
| 超参数 | 9个 | alpha, batch_size, dropout, epochs, etc. |
| 性能指标 | 9个 | accuracy, map, precision, recall, etc. |
| 能耗指标 | 11个 | CPU能耗(3) + GPU能耗(8) |
| 前景实验 | 35个 | 前景完整信息（超参数+性能+能耗） |
| 背景实验 | 10个 | 背景基础信息（4个） + 超参数（6个，80列格式缺失） |

### 2. 80列格式缺失字段

#### 缺失的背景实验字段（13个）

**背景超参数（6个）**:
1. `bg_hyperparam_batch_size` ← background.hyperparameters.batch_size
2. `bg_hyperparam_dropout` ← background.hyperparameters.dropout
3. `bg_hyperparam_epochs` ← background.hyperparameters.epochs
4. `bg_hyperparam_learning_rate` ← background.hyperparameters.learning_rate
5. `bg_hyperparam_seed` ← background.hyperparameters.seed
6. `bg_hyperparam_weight_decay` ← background.hyperparameters.weight_decay

**背景能耗指标（7个）**:
1. `bg_energy_cpu_pkg_joules` ← background.energy_metrics.cpu_energy_pkg_joules
2. `bg_energy_cpu_ram_joules` ← background.energy_metrics.cpu_energy_ram_joules
3. `bg_energy_cpu_total_joules` ← background.energy_metrics.cpu_energy_total_joules
4. `bg_energy_gpu_avg_watts` ← background.energy_metrics.gpu_power_avg_watts
5. `bg_energy_gpu_max_watts` ← background.energy_metrics.gpu_power_max_watts
6. `bg_energy_gpu_min_watts` ← background.energy_metrics.gpu_power_min_watts
7. `bg_energy_gpu_total_joules` ← background.energy_metrics.gpu_energy_total_joules

### 3. 字段映射Bug修复

#### 修复前的映射逻辑
```python
# 错误映射（原代码）
if metric.startswith('gpu_power_'):
    return f"energy_gpu_{metric.replace('gpu_power_', '')}_watts"
    # gpu_power_avg_watts → energy_gpu_avg_watts_watts ❌ (重复 "watts")
```

#### 修复后的映射逻辑
```python
# 正确映射（修复后）
if metric.startswith('gpu_power_'):
    return f"energy_{metric.replace('gpu_power_', 'gpu_')}"
    # gpu_power_avg_watts → energy_gpu_avg_watts ✓
```

#### 映射规则总结

**JSON → CSV字段映射规则**:

| JSON字段模式 | CSV列名模式 | 示例 |
|-------------|-------------|------|
| `hyperparameters.{param}` | `hyperparam_{param}` | epochs → hyperparam_epochs |
| `performance_metrics.{metric}` | `perf_{metric}` | accuracy → perf_accuracy |
| `energy_metrics.cpu_energy_{x}` | `energy_cpu_{x}` | cpu_energy_pkg_joules → energy_cpu_pkg_joules |
| `energy_metrics.gpu_power_{x}` | `energy_gpu_{x}` | gpu_power_avg_watts → energy_gpu_avg_watts |
| `energy_metrics.gpu_energy_{x}` | `energy_gpu_{x}` | gpu_energy_total_joules → energy_gpu_total_joules |
| `energy_metrics.gpu_{x}` | `energy_gpu_{x}` | gpu_temp_avg_celsius → energy_gpu_temp_avg_celsius |
| `foreground.{field}` | `fg_{field}` | 同上规则，添加fg_前缀 |
| `background.{field}` | `bg_{field}` | 同上规则，添加bg_前缀 |

### 4. 93列格式定义

#### 列数分布
```
总列数: 93列 (原80列 + 新增13列)

按类别统计:
- 基础信息: 7列
- 超参数: 9列
- 性能指标: 9列
- 能耗指标: 11列
- 元数据: 5列
- 前景实验: 42列 (6基础 + 9超参 + 9性能 + 11能耗 + 7额外能耗)
- 背景实验: 17列 (4基础 + 6超参数 + 7能耗) [新增13列]
```

#### 完整93列表头

详见自动生成的文件：
- 定义文件：`results/100col_schema_definition.txt`
- Python代码：`results/100col_header_code.py`

### 5. 转换脚本问题分析

#### 问题描述
`scripts/convert_summary_old_to_80col.py` 存在以下问题：

1. **JSON文件查找失败**
   - 老实验JSON路径: `results/mutation_2x_20251122_175401/{exp_id}/experiment.json`
   - 新实验JSON路径: `results/run_YYYYMMDD_HHMMSS/{exp_id}/experiment.json`
   - 脚本未能适配老实验路径结构

2. **数据来源问题**
   - 预期：从experiment.json重建数据
   - 实际：从37列CSV直接转换到80列CSV
   - 结果：**可能丢失JSON中的额外信息**

3. **缺失字段**
   - 80列格式本身缺少13个字段
   - 即使加载JSON，也无法填充这些字段（因为表头定义不包含）

---

## 验证结果

### JSON字段覆盖验证

```
✅ 93列格式验证结果:

总列数: 93列
从JSON映射得到的列数: 83列
元数据列（CSV特有）: 10列
JSON中有但CSV缺失: 0列

✓ 93列格式完整覆盖所有experiment.json字段！
```

### 未映射字段说明

以下9个字段"未映射"是**正常的**，它们是JSON对象的父级节点：
- `background` (对象)
- `background.hyperparameters` (对象)
- `energy_metrics` (对象)
- `foreground` (对象)
- `foreground.energy_metrics` (对象)
- `foreground.hyperparameters` (对象)
- `foreground.performance_metrics` (对象)
- `hyperparameters` (对象)
- `performance_metrics` (对象)

这些父级对象不需要映射到CSV列，只有它们的子字段需要映射。

---

## 建议与下一步

### 1. 升级CSV格式到93列 ⭐

**优先级**: 高

**行动**:
1. 使用生成的93列表头定义（`results/100col_header_code.py`）
2. 创建新的转换脚本 `convert_to_93col.py`
3. 重新转换 `summary_old.csv` 和 `summary_new.csv`

**预期收益**:
- 完整保留所有JSON信息
- 支持背景实验的完整分析
- 统一格式便于数据合并

### 2. 修复转换脚本的JSON加载逻辑

**优先级**: 高

**问题**:
- 当前脚本无法找到老实验的JSON文件
- 使用CSV数据转换可能丢失信息

**解决方案**:
```python
def find_experiment_json(experiment_id):
    """改进的JSON文件查找逻辑"""
    results_dir = Path('results')

    # 1. 尝试老实验目录
    old_dirs = [
        'mutation_2x_20251122_175401',
        'default',
        'mutation_1x',
        'archived'
    ]
    for old_dir in old_dirs:
        json_path = results_dir / old_dir / experiment_id / 'experiment.json'
        if json_path.exists():
            return json_path

    # 2. 尝试新实验目录
    for run_dir in results_dir.glob('run_*'):
        json_path = run_dir / experiment_id / 'experiment.json'
        if json_path.exists():
            return json_path

    return None
```

### 3. 创建完整的93列转换脚本

**优先级**: 高

**要求**:
1. 使用修复后的字段映射逻辑（不重复添加单位后缀）
2. 正确查找和加载JSON文件
3. 支持80列 → 93列的升级
4. 保留完整的备份机制

**新脚本名称**: `scripts/convert_to_93col_complete.py`

### 4. 验证数据完整性

**优先级**: 中

**验证项**:
1. 随机抽样10个实验，对比JSON与CSV数据一致性
2. 检查新增的13列是否正确填充
3. 验证能耗字段映射正确性
4. 确认背景实验信息完整

### 5. 更新文档

**优先级**: 中

**需要更新的文档**:
1. `CLAUDE.md` - 更新CSV格式说明（80列 → 93列）
2. `README.md` - 更新项目状态
3. `docs/CSV_REBUILD_FROM_EXPERIMENT_JSON.md` - 补充93列格式说明
4. `docs/results_reports/SUMMARY_OLD_REBUILD_80COL_REPORT_20251212.md` - 添加后续行动

---

## 附录

### A. 生成的文件清单

| 文件路径 | 说明 | 大小 |
|---------|------|------|
| `scripts/generate_100col_schema.py` | 93列格式生成脚本 | ~8KB |
| `results/100col_schema_definition.txt` | 93列格式完整定义 | ~6KB |
| `results/100col_header_code.py` | 93列表头Python代码 | ~2KB |
| `results/json_field_analysis.txt` | JSON字段分析详情 | ~4KB |

### B. 字段映射示例

#### 能耗指标映射（修复后）

| JSON字段 | CSV列名 | 说明 |
|---------|---------|------|
| `energy_metrics.cpu_energy_pkg_joules` | `energy_cpu_pkg_joules` | CPU封装能耗 |
| `energy_metrics.cpu_energy_ram_joules` | `energy_cpu_ram_joules` | CPU内存能耗 |
| `energy_metrics.cpu_energy_total_joules` | `energy_cpu_total_joules` | CPU总能耗 |
| `energy_metrics.gpu_power_avg_watts` | `energy_gpu_avg_watts` | GPU平均功率 |
| `energy_metrics.gpu_power_max_watts` | `energy_gpu_max_watts` | GPU最大功率 |
| `energy_metrics.gpu_power_min_watts` | `energy_gpu_min_watts` | GPU最小功率 |
| `energy_metrics.gpu_energy_total_joules` | `energy_gpu_total_joules` | GPU总能耗 |
| `energy_metrics.gpu_temp_avg_celsius` | `energy_gpu_temp_avg_celsius` | GPU平均温度 |
| `energy_metrics.gpu_temp_max_celsius` | `energy_gpu_temp_max_celsius` | GPU最大温度 |
| `energy_metrics.gpu_util_avg_percent` | `energy_gpu_util_avg_percent` | GPU平均利用率 |
| `energy_metrics.gpu_util_max_percent` | `energy_gpu_util_max_percent` | GPU最大利用率 |

### C. 命令行工具

#### 生成93列定义
```bash
python3 scripts/generate_100col_schema.py
```

#### 查看93列表头
```bash
python3 -c "exec(open('results/100col_header_code.py').read()); print('\\n'.join(f'{i:2d}. {col}' for i, col in enumerate(HEADER_100COL, 1)))"
```

#### 验证JSON字段覆盖
```bash
python3 scripts/analyze_json_field_coverage.py
```

---

## 结论

1. **80列CSV格式不足以覆盖所有experiment.json信息** ❌
   - 缺少13个背景实验相关字段
   - 需要扩展到93列格式

2. **字段映射逻辑已修复** ✅
   - 修复了能耗字段重复添加单位后缀的bug
   - 所有83个JSON叶子字段正确映射到CSV列

3. **转换脚本存在问题** ⚠️
   - 未能加载老实验的JSON文件
   - 使用CSV数据转换可能丢失信息
   - 需要重写转换脚本

4. **下一步行动** 📋
   - 创建93列转换脚本
   - 修复JSON文件加载逻辑
   - 重新转换summary_old.csv和summary_new.csv
   - 验证数据完整性
   - 更新相关文档

---

**报告生成**: Claude Code
**生成时间**: 2025-12-12
**版本**: v1.0
**状态**: ✅ 分析完成，等待实施
