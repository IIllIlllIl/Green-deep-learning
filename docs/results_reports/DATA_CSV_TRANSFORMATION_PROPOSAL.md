# Data CSV 转换方案

**创建日期**: 2025-12-19
**版本**: v1.0
**状态**: 提案待审核

---

## 目标

从 `raw_data.csv` (676行, 87列) 提取出简化的、易于分析的 `data.csv`，解决以下问题：
1. 性能指标列名不统一（同一模型在不同实验中使用不同列名）
2. 并行/非并行模式字段重复（顶层字段 vs fg_字段）
3. 大量空列影响数据处理效率

---

## 当前问题详细分析

### 问题1: 性能指标列名不统一

**原因**: 不同模型使用不同的性能指标，导致87列中有25列是性能指标（16个顶层 + 9个fg_），但每个实验只使用其中2-4个。

**影响**:
- 每个模型的主要性能指标散布在不同列
- 需要查阅模型配置才能知道应该看哪个指标
- 数据分析时需要针对不同模型编写不同逻辑

**示例**:
| 模型 | 使用的性能指标 | 列名 |
|------|---------------|------|
| examples/mnist | 准确率 | `perf_test_accuracy` |
| examples/mnist_ff | 准确率 | `perf_test_accuracy` |
| pytorch_resnet_cifar10/resnet20 | 准确率 | `perf_test_accuracy` 或 `perf_best_val_accuracy` |
| MRT-OAST/default | 准确率、精确率、召回率 | `perf_accuracy`, `perf_precision`, `perf_recall` |
| VulBERTa/mlp | 损失、吞吐量 | `perf_eval_loss`, `perf_eval_samples_per_second` |

### 问题2: 并行/非并行字段重复

**原因**: 并行模式有三种数据存储格式（历史演进导致）

**数据格式分析** (328个并行实验):
- **老格式** (114个, 34.8%): 仅有顶层字段（`training_success`, `perf_*`, `energy_*`等）
- **新格式** (105个, 32.0%): 仅有fg_字段（`fg_training_success`, `fg_perf_*`, `fg_energy_*`等）
- **混合格式** (109个, 33.2%): 两者都有

**影响**:
- 87列中有42列是fg_前缀字段（与顶层字段重复）
- 数据提取需要同时检查两套字段
- 增加数据处理复杂度

### 问题3: 大量空列

**统计**:
- 87列中，每个实验平均只填充约30-40列
- 50%以上的单元格为空
- 影响CSV文件读取和处理效率

---

## 解决方案

### 核心设计原则

1. **标准化性能指标**: 为每个模型定义主性能指标，统一映射到新列
2. **统一并行数据**: 增加 `is_parallel` 列区分模式，合并顶层/fg_字段
3. **精简列结构**: 仅保留有意义的列，移除冗余和空列

### 目标列结构 (约50列)

#### 1. 基础信息 (8列)
```
experiment_id          - 实验ID
timestamp             - 时间戳
repository            - 仓库名
model                 - 模型名
is_parallel           - 是否并行模式 (True/False) ⭐ 新增
training_success      - 训练是否成功
duration_seconds      - 训练时长
retries              - 重试次数
```

#### 2. 超参数 (9列)
```
hyperparam_alpha
hyperparam_batch_size
hyperparam_dropout
hyperparam_epochs
hyperparam_kfold
hyperparam_learning_rate
hyperparam_max_iter
hyperparam_seed
hyperparam_weight_decay
```

#### 3. 标准化性能指标 (10列) ⭐ 核心改进
```
# 主性能指标 (每个模型必填其中之一)
primary_metric_name   - 主性能指标名称 (如 "test_accuracy", "mAP", "eval_loss") ⭐ 新增
primary_metric_value  - 主性能指标值 ⭐ 新增

# 次要性能指标 (根据模型不同选填)
secondary_metric_1_name   - 次要指标1名称 ⭐ 新增
secondary_metric_1_value  - 次要指标1值 ⭐ 新增
secondary_metric_2_name   - 次要指标2名称 ⭐ 新增
secondary_metric_2_value  - 次要指标2值 ⭐ 新增
secondary_metric_3_name   - 次要指标3名称 ⭐ 新增
secondary_metric_3_value  - 次要指标3值 ⭐ 新增
secondary_metric_4_name   - 次要指标4名称 ⭐ 新增
secondary_metric_4_value  - 次要指标4值 ⭐ 新增
```

**设计理由**:
- 使用 `name-value` 对存储性能指标，避免列名不统一
- 每个模型只需填充实际使用的指标
- 便于后续分析（按 `primary_metric_name` 分组即可）

#### 4. 能耗指标 (11列)
```
energy_cpu_pkg_joules
energy_cpu_ram_joules
energy_cpu_total_joules
energy_gpu_avg_watts
energy_gpu_max_watts
energy_gpu_min_watts
energy_gpu_total_joules
energy_gpu_temp_avg_celsius
energy_gpu_temp_max_celsius
energy_gpu_util_avg_percent
energy_gpu_util_max_percent
```

#### 5. 实验元数据 (5列)
```
experiment_source     - 实验来源 (default/mutation_1x等)
num_mutated_params   - 变异参数数量
mutated_param        - 变异的参数名
mode                 - 模式 (default/mutation/parallel)
error_message        - 错误信息
```

#### 6. 并行模式额外信息 (7列，仅并行模式填充)
```
bg_repository        - 背景仓库
bg_model            - 背景模型
bg_note             - 背景备注
bg_log_directory    - 背景日志目录
fg_duration_seconds - 前景训练时长
fg_retries          - 前景重试次数
fg_error_message    - 前景错误信息
```

**总计**: 8 + 9 + 10 + 11 + 5 + 7 = **50列**

---

## 性能指标映射规则

### 每个模型的主性能指标定义

| 模型 | primary_metric_name | secondary_metrics |
|------|--------------------|--------------------|
| examples/mnist | test_accuracy | test_loss |
| examples/mnist_ff | test_accuracy | - |
| examples/mnist_rnn | test_accuracy | test_loss |
| examples/siamese | test_accuracy | test_loss |
| pytorch_resnet_cifar10/resnet20 | test_accuracy | best_val_accuracy |
| Person_reID/densenet121 | mAP | rank1, rank5 |
| Person_reID/hrnet18 | mAP | rank1, rank5 |
| Person_reID/pcb | mAP | rank1, rank5 |
| MRT-OAST/default | accuracy | precision, recall |
| VulBERTa/mlp | eval_loss | eval_samples_per_second, final_training_loss |
| bug-localization/default | top1_accuracy | top5_accuracy, top10_accuracy, top20_accuracy |

### 字段合并规则（并行模式）

**数据提取优先级**:
1. 如果 `fg_*` 字段有值 → 使用 `fg_*`
2. 否则使用顶层字段
3. 设置 `is_parallel = True`

**代码伪逻辑**:
```python
if row['mode'] == 'parallel':
    is_parallel = True
    # 优先使用fg_字段，fallback到顶层字段
    repository = row['fg_repository'] or row['repository']
    model = row['fg_model'] or row['model']
    duration_seconds = row['fg_duration_seconds'] or row['duration_seconds']
    training_success = row['fg_training_success'] or row['training_success']

    # 超参数优先使用fg_字段
    for param in ['alpha', 'batch_size', 'dropout', 'epochs', ...]:
        value = row[f'fg_hyperparam_{param}'] or row[f'hyperparam_{param}']

    # 能耗优先使用fg_字段
    for metric in ['cpu_pkg_joules', 'gpu_total_joules', ...]:
        value = row[f'fg_energy_{metric}'] or row[f'energy_{metric}']

    # 性能指标优先使用fg_字段
    for metric in all_perf_metrics:
        value = row[f'fg_perf_{metric}'] or row[f'perf_{metric}']
else:
    is_parallel = False
    # 直接使用顶层字段
    repository = row['repository']
    model = row['model']
    ...
```

---

## 数据转换实现

### 脚本设计: `scripts/transform_raw_to_data.py`

```python
#!/usr/bin/env python3
"""
将 raw_data.csv (87列) 转换为 data.csv (50列)

功能:
1. 统一并行/非并行字段 (fg_ vs 顶层)
2. 标准化性能指标 (name-value对)
3. 精简列结构，移除冗余
"""

import csv
from typing import Dict, List, Optional

# 每个模型的性能指标映射
MODEL_PERF_MAPPING = {
    'examples/mnist': {
        'primary': 'test_accuracy',
        'secondary': ['test_loss']
    },
    'examples/mnist_ff': {
        'primary': 'test_accuracy',
        'secondary': []
    },
    'examples/mnist_rnn': {
        'primary': 'test_accuracy',
        'secondary': ['test_loss']
    },
    'examples/siamese': {
        'primary': 'test_accuracy',
        'secondary': ['test_loss']
    },
    'pytorch_resnet_cifar10/resnet20': {
        'primary': 'test_accuracy',
        'secondary': ['best_val_accuracy']
    },
    'Person_reID_baseline_pytorch/densenet121': {
        'primary': 'map',
        'secondary': ['rank1', 'rank5']
    },
    'Person_reID_baseline_pytorch/hrnet18': {
        'primary': 'map',
        'secondary': ['rank1', 'rank5']
    },
    'Person_reID_baseline_pytorch/pcb': {
        'primary': 'map',
        'secondary': ['rank1', 'rank5']
    },
    'MRT-OAST/default': {
        'primary': 'accuracy',
        'secondary': ['precision', 'recall']
    },
    'VulBERTa/mlp': {
        'primary': 'eval_loss',
        'secondary': ['eval_samples_per_second', 'final_training_loss']
    },
    'bug-localization-by-dnn-and-rvsm/default': {
        'primary': 'top1_accuracy',
        'secondary': ['top5_accuracy', 'top10_accuracy', 'top20_accuracy']
    }
}

def get_field_value(row: Dict, field: str, is_parallel: bool) -> str:
    """
    获取字段值，处理并行/非并行模式

    并行模式: 优先使用fg_字段，fallback到顶层字段
    非并行模式: 直接使用顶层字段
    """
    if is_parallel:
        fg_value = row.get(f'fg_{field}', '').strip()
        if fg_value:
            return fg_value
    return row.get(field, '').strip()

def extract_performance_metrics(row: Dict, model_key: str, is_parallel: bool) -> Dict:
    """
    提取性能指标，转换为name-value对格式
    """
    mapping = MODEL_PERF_MAPPING.get(model_key, {'primary': '', 'secondary': []})

    perf_prefix = 'fg_perf_' if is_parallel else 'perf_'

    # 提取主指标
    primary_name = mapping['primary']
    primary_value = row.get(f'{perf_prefix}{primary_name}', '').strip()

    # 提取次要指标
    secondaries = {}
    for i, sec_name in enumerate(mapping['secondary'][:4], 1):  # 最多4个次要指标
        sec_value = row.get(f'{perf_prefix}{sec_name}', '').strip()
        secondaries[f'secondary_{i}'] = {'name': sec_name, 'value': sec_value}

    return {
        'primary_metric_name': primary_name,
        'primary_metric_value': primary_value,
        **secondaries
    }

def transform_row(row: Dict) -> Dict:
    """
    转换单行数据从raw_data格式到data格式
    """
    is_parallel = (row['mode'] == 'parallel')

    # 基础信息
    if is_parallel:
        repository = get_field_value(row, 'repository', True)
        model = get_field_value(row, 'model', True)
    else:
        repository = row['repository']
        model = row['model']

    model_key = f"{repository}/{model}"

    # 提取性能指标
    perf_metrics = extract_performance_metrics(row, model_key, is_parallel)

    # 构建新行
    new_row = {
        # 基础信息
        'experiment_id': row['experiment_id'],
        'timestamp': row['timestamp'],
        'repository': repository,
        'model': model,
        'is_parallel': str(is_parallel),
        'training_success': get_field_value(row, 'training_success', is_parallel),
        'duration_seconds': get_field_value(row, 'duration_seconds', is_parallel),
        'retries': get_field_value(row, 'retries', is_parallel),

        # 超参数
        'hyperparam_alpha': get_field_value(row, 'hyperparam_alpha', is_parallel),
        'hyperparam_batch_size': get_field_value(row, 'hyperparam_batch_size', is_parallel),
        'hyperparam_dropout': get_field_value(row, 'hyperparam_dropout', is_parallel),
        'hyperparam_epochs': get_field_value(row, 'hyperparam_epochs', is_parallel),
        'hyperparam_kfold': get_field_value(row, 'hyperparam_kfold', is_parallel),
        'hyperparam_learning_rate': get_field_value(row, 'hyperparam_learning_rate', is_parallel),
        'hyperparam_max_iter': get_field_value(row, 'hyperparam_max_iter', is_parallel),
        'hyperparam_seed': get_field_value(row, 'hyperparam_seed', is_parallel),
        'hyperparam_weight_decay': get_field_value(row, 'hyperparam_weight_decay', is_parallel),

        # 性能指标
        'primary_metric_name': perf_metrics['primary_metric_name'],
        'primary_metric_value': perf_metrics['primary_metric_value'],
        'secondary_metric_1_name': perf_metrics.get('secondary_1', {}).get('name', ''),
        'secondary_metric_1_value': perf_metrics.get('secondary_1', {}).get('value', ''),
        'secondary_metric_2_name': perf_metrics.get('secondary_2', {}).get('name', ''),
        'secondary_metric_2_value': perf_metrics.get('secondary_2', {}).get('value', ''),
        'secondary_metric_3_name': perf_metrics.get('secondary_3', {}).get('name', ''),
        'secondary_metric_3_value': perf_metrics.get('secondary_3', {}).get('value', ''),
        'secondary_metric_4_name': perf_metrics.get('secondary_4', {}).get('name', ''),
        'secondary_metric_4_value': perf_metrics.get('secondary_4', {}).get('value', ''),

        # 能耗指标
        'energy_cpu_pkg_joules': get_field_value(row, 'energy_cpu_pkg_joules', is_parallel),
        'energy_cpu_ram_joules': get_field_value(row, 'energy_cpu_ram_joules', is_parallel),
        'energy_cpu_total_joules': get_field_value(row, 'energy_cpu_total_joules', is_parallel),
        'energy_gpu_avg_watts': get_field_value(row, 'energy_gpu_avg_watts', is_parallel),
        'energy_gpu_max_watts': get_field_value(row, 'energy_gpu_max_watts', is_parallel),
        'energy_gpu_min_watts': get_field_value(row, 'energy_gpu_min_watts', is_parallel),
        'energy_gpu_total_joules': get_field_value(row, 'energy_gpu_total_joules', is_parallel),
        'energy_gpu_temp_avg_celsius': get_field_value(row, 'energy_gpu_temp_avg_celsius', is_parallel),
        'energy_gpu_temp_max_celsius': get_field_value(row, 'energy_gpu_temp_max_celsius', is_parallel),
        'energy_gpu_util_avg_percent': get_field_value(row, 'energy_gpu_util_avg_percent', is_parallel),
        'energy_gpu_util_max_percent': get_field_value(row, 'energy_gpu_util_max_percent', is_parallel),

        # 实验元数据
        'experiment_source': row['experiment_source'],
        'num_mutated_params': row['num_mutated_params'],
        'mutated_param': row['mutated_param'],
        'mode': row['mode'],
        'error_message': get_field_value(row, 'error_message', is_parallel),

        # 并行模式额外信息
        'bg_repository': row['bg_repository'] if is_parallel else '',
        'bg_model': row['bg_model'] if is_parallel else '',
        'bg_note': row['bg_note'] if is_parallel else '',
        'bg_log_directory': row['bg_log_directory'] if is_parallel else '',
        'fg_duration_seconds': row.get('fg_duration_seconds', '') if is_parallel else '',
        'fg_retries': row.get('fg_retries', '') if is_parallel else '',
        'fg_error_message': row.get('fg_error_message', '') if is_parallel else '',
    }

    return new_row

def main():
    """主函数"""
    input_file = 'data/raw_data.csv'
    output_file = 'data/data.csv'

    print("开始转换 raw_data.csv → data.csv")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()

    # 定义输出列顺序
    output_fieldnames = [
        # 基础信息 (8)
        'experiment_id', 'timestamp', 'repository', 'model', 'is_parallel',
        'training_success', 'duration_seconds', 'retries',

        # 超参数 (9)
        'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
        'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
        'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',

        # 性能指标 (10)
        'primary_metric_name', 'primary_metric_value',
        'secondary_metric_1_name', 'secondary_metric_1_value',
        'secondary_metric_2_name', 'secondary_metric_2_value',
        'secondary_metric_3_name', 'secondary_metric_3_value',
        'secondary_metric_4_name', 'secondary_metric_4_value',

        # 能耗指标 (11)
        'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
        'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
        'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
        'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',

        # 实验元数据 (5)
        'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',

        # 并行模式额外信息 (7)
        'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
        'fg_duration_seconds', 'fg_retries', 'fg_error_message'
    ]

    transformed_count = 0

    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            new_row = transform_row(row)
            writer.writerow(new_row)
            transformed_count += 1

    print(f"✓ 转换完成: {transformed_count} 行")
    print(f"✓ 列数: 87 → 50")
    print(f"✓ 输出文件: {output_file}")

if __name__ == '__main__':
    main()
```

---

## 转换结果验证

### 验证脚本: `scripts/validate_data_csv.py`

```python
#!/usr/bin/env python3
"""
验证 data.csv 的数据完整性和正确性
"""

import csv
from collections import defaultdict

def validate_data_csv():
    """验证data.csv"""
    data_file = 'data/data.csv'

    print("=" * 80)
    print("Data CSV 验证报告")
    print("=" * 80)
    print()

    stats = {
        'total_rows': 0,
        'parallel_rows': 0,
        'nonparallel_rows': 0,
        'primary_metric_filled': 0,
        'energy_filled': 0,
        'model_counts': defaultdict(int),
        'primary_metrics': defaultdict(set)
    }

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            stats['total_rows'] += 1

            # 统计并行/非并行
            if row['is_parallel'] == 'True':
                stats['parallel_rows'] += 1
            else:
                stats['nonparallel_rows'] += 1

            # 统计主性能指标填充率
            if row['primary_metric_value'].strip():
                stats['primary_metric_filled'] += 1

            # 统计能耗填充率
            if row['energy_cpu_total_joules'].strip():
                stats['energy_filled'] += 1

            # 统计模型分布
            model_key = f"{row['repository']}/{row['model']}"
            stats['model_counts'][model_key] += 1

            # 统计每个模型的主性能指标
            if row['primary_metric_name'].strip():
                stats['primary_metrics'][model_key].add(row['primary_metric_name'])

    # 输出报告
    print(f"总行数: {stats['total_rows']}")
    print(f"总列数: {len(fieldnames)}")
    print()

    print("模式分布:")
    print(f"  非并行: {stats['nonparallel_rows']} ({stats['nonparallel_rows']/stats['total_rows']*100:.1f}%)")
    print(f"  并行: {stats['parallel_rows']} ({stats['parallel_rows']/stats['total_rows']*100:.1f}%)")
    print()

    print("数据完整性:")
    print(f"  主性能指标: {stats['primary_metric_filled']}/{stats['total_rows']} ({stats['primary_metric_filled']/stats['total_rows']*100:.1f}%)")
    print(f"  能耗数据: {stats['energy_filled']}/{stats['total_rows']} ({stats['energy_filled']/stats['total_rows']*100:.1f}%)")
    print()

    print("模型分布:")
    for model_key in sorted(stats['model_counts'].keys()):
        count = stats['model_counts'][model_key]
        primary_metrics = ', '.join(sorted(stats['primary_metrics'][model_key]))
        print(f"  {model_key}: {count} 实验")
        print(f"    主性能指标: {primary_metrics}")

    print()
    print("=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == '__main__':
    validate_data_csv()
```

---

## 使用示例

### 转换数据
```bash
# 生成 data.csv
python3 scripts/transform_raw_to_data.py

# 验证转换结果
python3 scripts/validate_data_csv.py
```

### 数据分析示例

**分析不同超参数对主性能指标的影响**:
```python
import pandas as pd

df = pd.read_csv('data/data.csv')

# 按模型分组
for model in df['model'].unique():
    model_df = df[df['model'] == model]

    # 获取该模型的主性能指标名
    primary_metric = model_df['primary_metric_name'].iloc[0]

    # 分析learning_rate对主性能指标的影响
    lr_analysis = model_df.groupby('hyperparam_learning_rate')['primary_metric_value'].mean()

    print(f"\n{model} - {primary_metric} vs learning_rate:")
    print(lr_analysis)
```

**比较并行/非并行模式的能耗差异**:
```python
import pandas as pd

df = pd.read_csv('data/data.csv')

# 转换能耗为数值
df['energy_total'] = pd.to_numeric(df['energy_cpu_total_joules'], errors='coerce')

# 按模式分组统计
energy_by_mode = df.groupby(['model', 'is_parallel'])['energy_total'].mean()

print("各模型并行/非并行能耗对比:")
print(energy_by_mode.unstack())
```

---

## 优势总结

### 1. 简化数据结构
- **列数**: 87 → 50 (减少42.5%)
- **空单元格**: 50% → <20%
- **文件大小**: 预计减少30-40%

### 2. 统一性能指标
- 所有模型的主性能指标在同一列 (`primary_metric_value`)
- 按 `primary_metric_name` 分组即可对比不同模型
- 不需要记忆每个模型用的是哪个列

### 3. 并行/非并行统一
- 增加 `is_parallel` 列明确区分
- 自动合并 fg_/顶层字段
- 简化数据提取逻辑

### 4. 向后兼容
- 保留 `raw_data.csv` 作为原始数据
- `data.csv` 作为分析友好格式
- 所有原始数据仍可追溯

---

## 下一步

1. **审核方案**: 用户确认设计合理性
2. **实现脚本**: 编写并测试转换脚本
3. **验证结果**: 确保转换后数据完整性
4. **更新文档**: 更新README和CLAUDE.md说明data.csv的使用

---

**文档状态**: 提案待审核
**预期完成时间**: 1-2小时（实现+测试+验证）
