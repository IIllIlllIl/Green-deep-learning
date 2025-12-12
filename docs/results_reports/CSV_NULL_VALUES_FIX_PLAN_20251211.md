# CSV空值修复方案

**日期**: 2025-12-11
**版本**: v1.0
**状态**: 设计中

---

## 📋 问题概述

### 当前状况
- 总实验数: 476条记录
- CSV列数: 37列
- 主要空值问题:
  1. **变异模式列** (`experiment_source`): 55.7%空值 (265/476)
  2. **超参数列**: 多个列存在高比例空值（正常，因为不同模型使用不同超参数）
  3. **性能指标列**: 多个列存在空值（需要从日志提取）

### 根本原因
1. **变异模式**: 早期版本未记录训练模式，仅记录了被变异的超参数值
2. **性能指标**: 不同模型输出不同指标，某些指标从日志提取失败或未提取

---

## 🎯 修复目标

### 1. 变异模式补全（核心任务）
**目标**: 将`experiment_source`列从`default`修改为`default/{mutated_param}`格式

**逻辑**:
- 对于每一行，比较其超参数与模型的默认配置
- 如果只有一个参数与默认值不同，则该参数为被变异参数
- 格式: `{原值}/{变异参数名}` (例如: `default/epochs`, `mutation_1x/learning_rate`)

**示例**:
```
原始记录:
- experiment_id: mutation_1x__examples_mnist_007
- experiment_source: mutation_1x (或空)
- hyperparam_epochs: 5 (默认10)
- 其他超参数: 空

修复后:
- experiment_source: mutation_1x/epochs
```

### 2. 性能指标补全（次要任务）
**目标**: 从训练日志中提取缺失的性能指标

**方法**:
1. 根据`experiment_id`或`timestamp`匹配日志文件
2. 使用`models_config.json`中定义的正则表达式提取指标
3. 只填充当前为空的列（不覆盖已有数据）

**注意**: 允许某些指标为空（因为不同模型输出不同指标）

---

## 🔧 技术方案

### 数据结构

```python
# models_config.json中每个模型的默认配置
{
  "MRT-OAST": {
    "supported_hyperparams": {
      "epochs": {"default": 10},
      "learning_rate": {"default": 0.0001},
      "seed": {"default": 1334},
      "dropout": {"default": 0.2},
      "weight_decay": {"default": 0.0}
    }
  }
}
```

### 算法流程

```python
for each row in summary_all.csv:
    # 1. 识别模型
    repo = row['repository']
    model = row['model']

    # 2. 获取默认配置
    defaults = load_defaults(repo, model)

    # 3. 识别变异参数
    mutated_params = []
    for param, default_value in defaults.items():
        actual_value = row[f'hyperparam_{param}']
        if actual_value and actual_value != default_value:
            mutated_params.append(param)

    # 4. 更新experiment_source
    if len(mutated_params) == 1:
        # 单参数变异（正常情况）
        base_source = row['experiment_source'] or 'default'
        row['experiment_source'] = f"{base_source}/{mutated_params[0]}"
    elif len(mutated_params) == 0:
        # 默认配置（无变异）
        row['experiment_source'] = row['experiment_source'] or 'default'
    else:
        # 多参数变异（异常，需要记录）
        print(f"Warning: {row['experiment_id']} has {len(mutated_params)} mutations")
```

---

## 🛡️ 安全措施

### 1. 数据备份
```bash
# 自动创建带时间戳的备份
cp results/summary_all.csv results/summary_all.csv.backup_$(date +%Y%m%d_%H%M%S)
```

### 2. 验证检查
- 修复前后行数一致
- 修复前后列数一致
- 所有必填列无空值
- CSV格式正确（可用`python -m csv`验证）

### 3. 增量更新
- 只修改空值或需要更新的列
- 保留所有已有数据
- 生成修复报告（哪些行被修改）

---

## 📊 预期结果

### 变异模式列
- **修复前**: 265行空值（55.7%）
- **修复后**: 0行空值（目标100%）
- **格式示例**:
  - `default` → `default` (无变异)
  - `default` → `default/epochs` (变异epochs)
  - `mutation_1x` → `mutation_1x/learning_rate` (变异lr)
  - `parallel` → `parallel/dropout` (并行模式变异dropout)

### 性能指标列
- 尽最大努力从日志提取
- 允许合理的空值存在（不同模型不同指标）
- 记录无法提取的情况

---

## 📝 实现计划

### 脚本结构
```
scripts/fix_csv_null_values.py
├── 1. load_models_config()        # 加载默认配置
├── 2. identify_mutated_param()    # 识别变异参数
├── 3. fix_experiment_source()     # 修复变异模式列
├── 4. extract_performance_metrics() # 从日志提取性能指标
├── 5. validate_csv()              # 验证CSV完整性
└── 6. generate_report()           # 生成修复报告
```

### 执行命令
```bash
# 运行修复脚本
python3 scripts/fix_csv_null_values.py \
    --input results/summary_all.csv \
    --output results/summary_all.csv \
    --config mutation/models_config.json \
    --backup-dir results/backups \
    --report-dir docs/results_reports \
    --dry-run  # 先预览不实际修改

# 实际执行
python3 scripts/fix_csv_null_values.py \
    --input results/summary_all.csv \
    --output results/summary_all.csv \
    --config mutation/models_config.json \
    --backup-dir results/backups \
    --report-dir docs/results_reports
```

---

## ⚠️ 注意事项

### 1. 特殊情况处理
- **多参数变异**: 记录警告但不修改（可能是错误配置）
- **无法识别模型**: 跳过并记录
- **默认值不匹配**: 可能是配置更新，需要人工检查

### 2. 性能指标提取
- 不同模型有不同的指标集
- 某些模型可能没有某些指标（正常情况）
- 只提取空值列，不覆盖已有数据

### 3. 兼容性
- 保持CSV格式不变（37列）
- 保持列顺序不变
- 保持数据类型不变

---

## 🔄 后续优化

1. 在`runner.py`中自动记录变异模式（避免未来空值）
2. 增强日志解析器，提高指标提取成功率
3. 添加实时验证机制，确保数据完整性

---

**维护者**: Claude + Green
**审核状态**: 待审核
**实现状态**: 设计完成，待实现
