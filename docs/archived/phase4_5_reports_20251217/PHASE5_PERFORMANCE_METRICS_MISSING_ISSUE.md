# Phase 5性能指标缺失问题分析与解决方案

**发现日期**: 2025-12-15
**影响范围**: Phase 5的20个并行实验（VulBERTa/mlp和bug-localization）
**严重程度**: 中等

---

## 问题描述

Phase 5执行的72个并行实验中，有20个实验的性能指标未能成功追加到raw_data.csv：
- VulBERTa/mlp: 12个实验缺失性能数据
- bug-localization: 8个实验缺失性能数据

**根本原因**: raw_data.csv的列定义中缺少这两个模型使用的性能指标列。

---

## 技术分析

### 1. 数据源验证

✅ **实验原始数据完整** - 在session的summary.csv和experiment.json中，所有72个实验都有性能指标

**VulBERTa/mlp的性能指标字段**:
- `eval_loss`
- `final_training_loss`
- `eval_samples_per_second`

**bug-localization的性能指标字段**:
- `top1_accuracy`
- `top5_accuracy`
- `top10_accuracy`
- `top20_accuracy`

### 2. 数据提取脚本分析

`tools/data_management/append_session_to_raw_data.py` 已包含正确的性能指标映射（修复后）:
```python
perf_mapping = {
    'eval_loss': 'perf_eval_loss',
    'final_training_loss': 'perf_final_training_loss',
    'eval_samples_per_second': 'perf_eval_samples_per_second',
    '...': '...',
    'top1_accuracy': 'perf_top1_accuracy',
    'top5_accuracy': 'perf_top5_accuracy',
    'top10_accuracy': 'perf_top10_accuracy',
    'top20_accuracy': 'perf_top20_accuracy'
}
```

### 3. raw_data.csv列定义问题

**当前列** (80列):
```
perf_accuracy, perf_best_val_accuracy, perf_map, perf_precision, perf_rank1,
perf_rank5, perf_recall, perf_test_accuracy, perf_test_loss
```

**缺失列**:
- `perf_eval_loss` (VulBERTa/mlp需要)
- `perf_final_training_loss` (VulBERTa/mlp需要)
- `perf_eval_samples_per_second` (VulBERTa/mlp需要)
- `perf_top1_accuracy` (bug-localization需要)
- `perf_top5_accuracy` (bug-localization需要)
- `perf_top10_accuracy` (bug-localization需要)
- `perf_top20_accuracy` (bug-localization需要)

---

## 解决方案

### 方案1: 扩展raw_data.csv列定义（推荐）⭐⭐⭐

**步骤**:
1. 创建扩展脚本，添加7个缺失的性能指标列到raw_data.csv
2. 恢复到Phase 5追加前的状态（512行）
3. 重新运行追加脚本，提取完整的性能数据

**优点**:
- ✅ 数据完整，不丢失任何信息
- ✅ 支持未来所有模型的性能指标
- ✅ 一次修复，永久解决

**缺点**:
- 列数增加（80→87列）
- 需要更新相关脚本和文档

### 方案2: 仅记录已有列的性能指标（不推荐）

**优点**:
- 保持列定义不变

**缺点**:
- ❌ 丢失20个实验的性能数据
- ❌ 数据不完整
- ❌ 无法满足Phase 5的实验目标

---

## 推荐行动

### 立即行动（今天）

1. **创建列扩展脚本**: `scripts/expand_raw_data_columns.py`
   - 添加7个缺失的性能指标列
   - 备份现有raw_data.csv

2. **重新追加Phase 5数据**:
   - 恢复到512行状态
   - 运行修复后的追加脚本
   - 验证所有72个实验的性能数据完整

3. **更新文档**:
   - 记录列扩展的原因和过程
   - 更新raw_data.csv的列定义文档

### 后续优化（本周）

1. **创建标准化工具**: 自动检测并扩展CSV列定义
2. **测试套件**: 添加测试验证所有模型的性能指标列存在
3. **文档完善**: 更新CLAUDE.md中的重要注意事项

---

## 重要经验教训 ⭐⭐⭐

### 1. **实验ID不唯一，必须使用复合键**

**问题**: 不同批次的实验会产生相同的experiment_id（如 `VulBERTa_mlp_001_parallel`）

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

**示例**:
```python
# ❌ 错误 - 仅使用experiment_id
if exp_id == 'VulBERTa_mlp_001_parallel':
    ...

# ✅ 正确 - 使用复合键
composite_key = f"{exp_id}|{timestamp}"
if composite_key in existing_keys:
    ...

# ✅ 或使用时间范围过滤
phase5_start = datetime.fromisoformat('2025-12-14T17:48:00')
phase5_end = datetime.fromisoformat('2025-12-15T17:06:00')
if phase5_start <= timestamp <= phase5_end:
    ...
```

**影响**:
- 数据查询错误
- 去重机制失效
- 统计分析不准确

**已更新**:
- ✅ `tools/data_management/append_session_to_raw_data.py` 已使用复合键去重
- ⚠️ 需要更新CLAUDE.md，添加此重要注意事项

### 2. **CSV列定义需要提前规划**

**问题**: 不同模型使用不同的性能指标字段名，导致列定义不完整

**正确做法**:
- 在开始实验前，分析所有模型的性能指标字段
- 创建包含所有可能字段的完整列定义
- 定期审查和扩展列定义

### 3. **数据提取脚本需要与CSV列定义同步**

**问题**: 脚本有正确的映射，但CSV缺少目标列

**正确做法**:
- 脚本映射和CSV列定义需要一致
- 添加列存在性检查
- 自动扩展列定义

---

## 修复脚本示例

```python
#!/usr/bin/env python3
"""
扩展raw_data.csv，添加缺失的性能指标列

用法: python3 scripts/expand_raw_data_columns.py
"""

import csv
from datetime import datetime
from pathlib import Path

# 新增列
NEW_COLUMNS = [
    'perf_eval_loss',
    'perf_final_training_loss',
    'perf_eval_samples_per_second',
    'perf_top1_accuracy',
    'perf_top5_accuracy',
    'perf_top10_accuracy',
    'perf_top20_accuracy'
]

def expand_raw_data_columns(csv_path='data/raw_data.csv'):
    """扩展raw_data.csv的列定义"""

    # 备份
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'{csv_path}.backup_80col_{timestamp}'

    import shutil
    shutil.copy(csv_path, backup_path)
    print(f'✅ 备份: {backup_path}')

    # 读取现有数据
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames
        rows = list(reader)

    # 创建新的fieldnames
    new_fieldnames = list(old_fieldnames)

    # 在perf_test_loss后插入新列
    insert_index = new_fieldnames.index('perf_test_loss') + 1
    for col in NEW_COLUMNS:
        if col not in new_fieldnames:
            new_fieldnames.insert(insert_index, col)
            insert_index += 1

    print(f'列数: {len(old_fieldnames)} → {len(new_fieldnames)}')
    print(f'新增列: {NEW_COLUMNS}')

    # 写回文件
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ 已更新: {csv_path}')
    print(f'   列数: {len(new_fieldnames)}')

if __name__ == '__main__':
    expand_raw_data_columns()
```

---

## 状态

- ⏳ **待修复**: 需要执行方案1，扩展CSV列并重新追加数据
- 📝 **已文档化**: 此问题和解决方案
- ⚠️ **需更新**: CLAUDE.md中添加"实验ID不唯一"的重要注意事项

---

**报告人**: Claude Code Assistant
**报告日期**: 2025-12-15
**优先级**: 高
**预计修复时间**: 1-2小时
