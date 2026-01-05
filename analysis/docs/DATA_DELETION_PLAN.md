# 数据删除方案

**文档版本**: v1.0
**创建时间**: 2025-12-22
**状态**: ⚠️ 方案待审核，未执行

---

## 📋 执行摘要

本文档提供从676行原始数据筛选到284行严格完整数据的详细删除方案。

**⚠️ 重要**: 本方案仅为建议，未执行任何删除操作。请审核后决定是否执行。

---

## 🎯 删除规则

### 严格完整性标准

保留满足**以下所有条件**的行：

1. ✅ **能耗数据完整**
   - `energy_cpu_total_joules` 有值（或 `fg_energy_cpu_total_joules` 有值）
   - `energy_gpu_total_joules` 有值（或 `fg_energy_gpu_total_joules` 有值）

2. ✅ **性能数据完整**
   - 至少一个性能指标有值：
     - `perf_test_accuracy`
     - `perf_map`
     - `perf_eval_loss`
     - `perf_top1_accuracy`
   - （或对应的 `fg_perf_*` 列有值）

3. ✅ **训练时长超参数完整**
   - `hyperparam_epochs` 有值 **或** `hyperparam_max_iter` 有值
   - （或对应的 `fg_hyperparam_*` 列有值）

4. ✅ **分组标识完整**
   - `repository` 有值（或 `fg_repository`）
   - `model` 有值（或 `fg_model`）

5. ✅ **敏感属性完整**
   - `mode` 有值（或为空但可填充为'default'）

---

## 📊 删除影响分析

### 总体统计

| 指标 | 数值 |
|------|------|
| **原始数据** | 676行 |
| **保留数据** | 284行 (42.0%) |
| **删除数据** | 392行 (58.0%) |

### 按删除原因分类

```python
# 运行以下脚本查看详细分类
python3 scripts/analyze_deletion_reasons.py
```

**预期分类**（需验证）：

| 删除原因 | 行数（估计） | 百分比 |
|---------|------------|-------|
| 缺失能耗数据 | ~139行 | 20.6% |
| 缺失性能数据 | ~285行 | 42.2% |
| 缺失训练时长超参数 | ~30行 | 4.4% |
| 缺失分组标识 | 0行 | 0% |
| **总删除** | **392行** | **58.0%** |

**注**: 一行可能因多个原因被删除，上述统计可能有重叠。

### 按任务组统计

#### 保留数据分布

| 任务组 | 保留行数 | 原始行数 | 保留率 |
|-------|---------|---------|-------|
| examples (MNIST) | 153 | 189 | 81.0% |
| Person_reID_baseline_pytorch | 86 | 93 | 92.5% |
| pytorch_resnet_cifar10 | 21 | 26 | 80.8% |
| VulBERTa | 14 | 119 | 11.8% ⚠️ |
| bug-localization-by-dnn-and-rvsm | 10 | 82 | 12.2% ⚠️ |
| MRT-OAST | 0 | 62 | 0% ❌ |
| (empty repository) | 0 | 105 | 0% ❌ |
| **总计** | **284** | **676** | **42.0%** |

#### 关键观察

⚠️ **VulBERTa保留率极低 (11.8%)**
- 原因：性能数据缺失率高（60.5%）
- 建议：如果该任务组重要，考虑补充数据

⚠️ **Bug定位保留率极低 (12.2%)**
- 原因：性能数据缺失率高（48.8%）
- 建议：如果该任务组重要，考虑补充数据

❌ **MRT-OAST完全删除**
- 原因：性能指标完全缺失
- 建议：该任务组无法进行性能-能耗因果分析

❌ **Empty repository完全删除**
- 原因：并行实验中repository为空（数据结构问题）
- 建议：检查数据收集逻辑

---

## 🔧 执行方案

### 方案A：创建新文件（推荐） ✅

**优点**: 保留原始数据，可回滚

```python
#!/usr/bin/env python3
"""
方案A: 创建新的cleaned数据文件

文件路径: analysis/scripts/create_cleaned_data.py
"""

import pandas as pd
import sys
import os

def extract_value(row, base_col):
    """统一提取数据：优先主列，然后fg_列"""
    if row.get(base_col) and str(row[base_col]).strip():
        return row[base_col]

    fg_col = 'fg_' + base_col
    if row.get(fg_col) and str(row[fg_col]).strip():
        return row[fg_col]

    return None

def is_valid_row(row):
    """判断行是否满足严格完整性标准"""

    # 1. 能耗数据
    has_energy = (extract_value(row, 'energy_cpu_total_joules') and
                  extract_value(row, 'energy_gpu_total_joules'))
    if not has_energy:
        return False, "缺失能耗数据"

    # 2. 性能数据
    perf_cols = ['perf_test_accuracy', 'perf_map', 'perf_eval_loss', 'perf_top1_accuracy']
    has_perf = any(extract_value(row, col) for col in perf_cols)
    if not has_perf:
        return False, "缺失性能数据"

    # 3. 训练时长
    has_duration = (extract_value(row, 'hyperparam_epochs') or
                   extract_value(row, 'hyperparam_max_iter'))
    if not has_duration:
        return False, "缺失训练时长超参数"

    # 4. 分组标识
    has_grouping = (extract_value(row, 'repository') and extract_value(row, 'model'))
    if not has_grouping:
        return False, "缺失分组标识"

    # 5. mode（可以为空，后续填充）
    # 不作为删除条件

    return True, "valid"

def main():
    # 路径设置
    input_path = '../../data/raw_data.csv'
    output_path = '../data/cleaned_data.csv'
    report_path = '../data/deletion_report.txt'

    print("="*80)
    print("数据清洗：创建cleaned数据文件")
    print("="*80)

    # 1. 读取原始数据
    print("\n1. 读取原始数据...")
    df = pd.read_csv(input_path)
    print(f"   原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 2. 检查每行
    print("\n2. 检查数据完整性...")
    valid_rows = []
    deletion_reasons = []

    for idx, row in df.iterrows():
        is_valid, reason = is_valid_row(row)
        if is_valid:
            valid_rows.append(idx)
        else:
            deletion_reasons.append({
                'index': idx,
                'reason': reason,
                'repository': row.get('repository', ''),
                'model': row.get('model', ''),
                'mode': row.get('mode', '')
            })

    # 3. 创建cleaned数据
    df_cleaned = df.loc[valid_rows].copy()

    print(f"\n3. 数据清洗结果:")
    print(f"   保留行数: {len(df_cleaned)} 行 ({len(df_cleaned)/len(df)*100:.1f}%)")
    print(f"   删除行数: {len(deletion_reasons)} 行 ({len(deletion_reasons)/len(df)*100:.1f}%)")

    # 4. 统计删除原因
    print(f"\n4. 删除原因统计:")
    reason_counts = {}
    for item in deletion_reasons:
        reason = item['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"   {reason:30s}: {count:3d} 行 ({count/len(df)*100:5.1f}%)")

    # 5. 按任务组统计
    print(f"\n5. 按任务组统计:")

    # 原始分布
    original_counts = df['repository'].value_counts().to_dict()
    # Cleaned分布
    cleaned_counts = df_cleaned['repository'].value_counts().to_dict()

    for repo in sorted(set(list(original_counts.keys()) + list(cleaned_counts.keys()))):
        orig = original_counts.get(repo, 0)
        clean = cleaned_counts.get(repo, 0)
        rate = clean / orig * 100 if orig > 0 else 0
        print(f"   {repo:45s}: {clean:3d}/{orig:3d} ({rate:5.1f}%)")

    # 6. 保存cleaned数据
    print(f"\n6. 保存数据...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_cleaned.to_csv(output_path, index=False)
    print(f"   ✅ 已保存到: {output_path}")

    # 7. 保存删除报告
    print(f"\n7. 保存删除报告...")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("数据删除报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"原始数据: {len(df)} 行\n")
        f.write(f"保留数据: {len(df_cleaned)} 行 ({len(df_cleaned)/len(df)*100:.1f}%)\n")
        f.write(f"删除数据: {len(deletion_reasons)} 行 ({len(deletion_reasons)/len(df)*100:.1f}%)\n\n")

        f.write("删除原因统计:\n")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {reason:30s}: {count:3d} 行\n")

        f.write("\n按任务组统计:\n")
        for repo in sorted(set(list(original_counts.keys()) + list(cleaned_counts.keys()))):
            orig = original_counts.get(repo, 0)
            clean = cleaned_counts.get(repo, 0)
            rate = clean / orig * 100 if orig > 0 else 0
            f.write(f"  {repo:45s}: {clean:3d}/{orig:3d} ({rate:5.1f}%)\n")

        f.write("\n删除详情 (前50行):\n")
        f.write("-"*80 + "\n")
        for i, item in enumerate(deletion_reasons[:50]):
            f.write(f"{i+1:3d}. Index {item['index']:4d}: {item['reason']:30s} ")
            f.write(f"[{item['repository']}/{item['model']}/{item['mode']}]\n")

        if len(deletion_reasons) > 50:
            f.write(f"\n... 还有 {len(deletion_reasons) - 50} 行被删除\n")

    print(f"   ✅ 已保存报告到: {report_path}")

    print("\n" + "="*80)
    print("数据清洗完成！")
    print("="*80)
    print(f"\n文件位置:")
    print(f"  原始数据: {input_path}")
    print(f"  清洗后: {output_path}")
    print(f"  删除报告: {report_path}")

    return df_cleaned, deletion_reasons

if __name__ == '__main__':
    df_cleaned, reasons = main()
```

**使用方法**:

```bash
cd /home/green/energy_dl/nightly/analysis
python3 scripts/create_cleaned_data.py
```

**输出文件**:
- `analysis/data/cleaned_data.csv` - 清洗后的数据（284行）
- `analysis/data/deletion_report.txt` - 详细删除报告

---

### 方案B：就地修改（不推荐） ❌

**缺点**: 会永久删除原始数据，无法回滚

```python
# ❌ 不推荐，仅作参考
df = pd.read_csv('raw_data.csv')
df_cleaned = df[df.apply(is_valid_row, axis=1)]
df_cleaned.to_csv('raw_data.csv', index=False)  # 覆盖原文件
```

**⚠️ 警告**: 如果采用此方案，务必先备份！

---

## ✅ 验证方案

### 验证脚本

创建验证脚本确保删除正确：

```python
#!/usr/bin/env python3
"""
验证cleaned数据的完整性

文件路径: analysis/scripts/validate_cleaned_data.py
"""

import pandas as pd

def extract_value(row, base_col):
    """统一提取数据"""
    if row.get(base_col) and str(row[base_col]).strip():
        return row[base_col]
    fg_col = 'fg_' + base_col
    if row.get(fg_col) and str(row[fg_col]).strip():
        return row[fg_col]
    return None

def validate_cleaned_data(csv_path):
    """验证cleaned数据"""

    print("="*80)
    print("验证cleaned数据完整性")
    print("="*80)

    df = pd.read_csv(csv_path)
    print(f"\n总行数: {len(df)}")

    # 检查每个完整性标准
    checks = {
        '能耗数据': lambda r: extract_value(r, 'energy_cpu_total_joules') and
                              extract_value(r, 'energy_gpu_total_joules'),
        '性能数据': lambda r: any(extract_value(r, col) for col in
                                ['perf_test_accuracy', 'perf_map', 'perf_eval_loss', 'perf_top1_accuracy']),
        '训练时长': lambda r: extract_value(r, 'hyperparam_epochs') or
                              extract_value(r, 'hyperparam_max_iter'),
        '分组标识': lambda r: extract_value(r, 'repository') and extract_value(r, 'model')
    }

    all_valid = True

    print("\n完整性检查:")
    for check_name, check_func in checks.items():
        valid_count = sum(1 for _, row in df.iterrows() if check_func(row))
        invalid_count = len(df) - valid_count

        if invalid_count > 0:
            print(f"  ❌ {check_name:12s}: {valid_count}/{len(df)} 行有效, {invalid_count}行缺失")
            all_valid = False
        else:
            print(f"  ✅ {check_name:12s}: {valid_count}/{len(df)} 行全部有效")

    if all_valid:
        print("\n" + "="*80)
        print("✅ 验证通过！所有行都满足完整性标准。")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("❌ 验证失败！存在不满足标准的行。")
        print("="*80)
        return False

if __name__ == '__main__':
    csv_path = '../data/cleaned_data.csv'
    validate_cleaned_data(csv_path)
```

**使用方法**:

```bash
python3 scripts/validate_cleaned_data.py
```

---

## 🔄 回滚方案

### 如果删除错误

如果采用方案A（创建新文件），可以直接使用原始文件：

```bash
# 原始文件位置
/home/green/energy_dl/nightly/data/raw_data.csv

# 如果需要，可以重新生成cleaned文件
cd /home/green/energy_dl/nightly/analysis
python3 scripts/create_cleaned_data.py
```

### 如果采用方案B（就地修改）

需要提前备份：

```bash
# 备份原始数据
cp data/raw_data.csv data/raw_data.csv.backup_$(date +%Y%m%d_%H%M%S)

# 如果需要回滚
cp data/raw_data.csv.backup_YYYYMMDD_HHMMSS data/raw_data.csv
```

---

## 📝 执行清单

### 执行前

- [ ] 审核删除规则（见"删除规则"章节）
- [ ] 审核删除影响（见"删除影响分析"章节）
- [ ] 确认保留率可接受（特别是VulBERTa和Bug定位）
- [ ] 选择执行方案（推荐方案A）
- [ ] 如果选择方案B，先备份原始数据

### 执行中

- [ ] 运行执行脚本 `python3 scripts/create_cleaned_data.py`
- [ ] 检查输出日志，确认统计数字与预期一致
- [ ] 验证输出文件存在：`analysis/data/cleaned_data.csv`
- [ ] 验证报告文件存在：`analysis/data/deletion_report.txt`

### 执行后

- [ ] 运行验证脚本 `python3 scripts/validate_cleaned_data.py`
- [ ] 确认验证通过（所有检查✅）
- [ ] 检查cleaned数据形状：284行 × 87列
- [ ] 抽查几行数据，确认关键列有值
- [ ] 阅读deletion_report.txt，确认删除合理

---

## ⚠️ 风险提示

### 高风险任务组

1. **VulBERTa (11.8%保留率)**
   - 风险：样本量从119 → 14，降低87.8%
   - 影响：DiBS分析统计功效下降
   - 建议：如果该任务组关键，考虑补充数据或放宽标准

2. **Bug定位 (12.2%保留率)**
   - 风险：样本量从82 → 10，降低87.8%
   - 影响：刚好满足DiBS最低要求（10个），边界情况
   - 建议：谨慎评估分析质量

3. **MRT-OAST (0%保留率)**
   - 风险：完全删除
   - 影响：无法进行该任务的因果分析
   - 建议：确认是否可以放弃该任务组

### 数据结构问题

发现105行实验的repository为空（并行实验数据结构问题）：
- 建议：修复数据收集逻辑，补充repository信息
- 或：检查fg_repository列是否有值但未提取

---

## 📚 参考文档

- [DATA_PREPROCESSING_DECISIONS.md](./DATA_PREPROCESSING_DECISIONS.md) - 预处理决策详解
- [VARIABLE_EXPANSION_PLAN.md](./reports/VARIABLE_EXPANSION_PLAN.md) - 变量扩展方案

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2025-12-22 | 初始版本：数据删除方案（未执行） | Green + Claude |

---

**维护者**: Green
**文档状态**: ⚠️ 方案待审核
**执行状态**: ❌ 未执行
**下次更新**: 执行删除后（添加实际结果）
