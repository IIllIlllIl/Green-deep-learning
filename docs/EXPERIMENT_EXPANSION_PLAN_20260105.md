# 实验扩展方案 - 多参数变异实验设计

**文档版本**: 1.0
**创建日期**: 2026-01-05
**状态**: 📋 规划中

---

## 📋 方案概述

### 背景

当前项目已完成：
- **836个实验**（795个有效能耗数据，95.1%完整性）
- **11个模型**全覆盖（并行/非并行模式）
- **90/90参数-模式组合**100%达标

### 新目标

从原有的单参数变异实验设计扩展为：
- **每个模型1次默认值实验** - 建立基线
- **每个模型n次变异实验** - **不再限制每次只变异一个超参数**
- **保留去重机制** - 避免重复实验
- **目标运行时间**: 约1周（7天）

### 关键变化

| 维度 | 原设计 | 新设计 |
|------|--------|--------|
| **变异约束** | 每次只变异1个超参数 | 可同时变异多个超参数 |
| **实验数/模型** | 1默认 + 4参数×5变异 = 21个 | 1默认 + n个变异 |
| **去重机制** | ✅ 启用 | ✅ 保留 |
| **配置格式** | `"mutate": ["param"]` | 灵活配置（单参数或多参数） |

---

## 📊 运行时间分析

### 当前实验统计

基于836个已完成实验的数据分析：

| 指标 | 数值 |
|------|------|
| **总实验数** | 836个 |
| **平均运行时间** | 30.84分钟（0.51小时） |
| **中位数运行时间** | 16.49分钟 |
| **最短实验** | 0.12分钟（examples/mnist_ff） |
| **最长实验** | 164.51分钟（Person_reID/pcb） |
| **累计运行时间** | 429.76小时（17.91天） |

### 模型速度分类

#### 快速模型（< 5分钟）- 2个模型

| 模型 | 平均时间 | 当前实验数 |
|------|---------|-----------|
| examples/mnist_ff | 0.25分钟 | 87个 |
| examples/mnist | 3.39分钟 | 69个 |

#### 中速模型（5-30分钟）- 5个模型

| 模型 | 平均时间 | 当前实验数 |
|------|---------|-----------|
| examples/mnist_rnn | 5.16分钟 | 53个 |
| examples/siamese | 7.96分钟 | 50个 |
| bug-localization-by-dnn-and-rvsm/default | 15.66分钟 | 142个 |
| MRT-OAST/default | 18.48分钟 | 88个 |
| pytorch_resnet_cifar10/resnet20 | 27.43分钟 | 49个 |

#### 慢速模型（> 30分钟）- 4个模型

| 模型 | 平均时间 | 当前实验数 |
|------|---------|-----------|
| VulBERTa/mlp | 56.05分钟 | 152个 |
| Person_reID_baseline_pytorch/densenet121 | 56.74分钟 | 53个 |
| Person_reID_baseline_pytorch/hrnet18 | 84.42分钟 | 47个 |
| Person_reID_baseline_pytorch/pcb | 90.67分钟 | 46个 |

### 1周运行能力估算

| 并行模式 | 可运行实验数 | 说明 |
|---------|-------------|------|
| **串行模式** | 327个 | 168小时 ÷ 0.51小时/实验 |
| **2倍并行** | 654个 | 前台+后台同时运行 |
| **1.5倍并行（保守）** | 490个 | 考虑资源竞争和调度开销 |

**保守估计**: 1周内可完成约 **490个实验**（1.5倍并行能力）

---

## 🎯 n值推荐方案

### 方案1：统一n值（所有模型使用相同n值）

适用于：简单配置，易于管理

| n值 | 总实验数 | 预计时间 | 可行性 |
|-----|---------|---------|--------|
| n=30 | 341个 | 7.88天 | ✅ 可行（留有余量） |
| n=36 | 407个 | 9.41天 | ⚠️ 稍超（但可接受） |
| n=40 | 451个 | 10.43天 | ⚠️ 超出1周（需10.4天） |
| n=50 | 561个 | 12.97天 | ❌ 超时 |

**推荐**:
- **保守选择**: n=30（341个实验，7.88天）
- **平衡选择**: n=36（407个实验，9.41天）

### 方案2：分层n值（根据模型速度调整）⭐⭐⭐ 推荐

适用于：最大化利用1周时间，获得更多实验数据

#### 激进方案

| 模型分类 | n值 | 模型数 | 实验数 |
|---------|-----|--------|--------|
| 快速模型 | n=100 | 2个 | 202个 |
| 中速模型 | n=50 | 5个 | 255个 |
| 慢速模型 | n=30 | 4个 | 124个 |
| **总计** | - | **11个** | **581个** |

- **预计时间**: 218.3小时（9.10天）
- **可行性**: ✅ 可行（稍超1周，但可接受）

#### 平衡方案 ⭐⭐⭐ 强烈推荐

| 模型分类 | n值 | 模型数 | 实验数 |
|---------|-----|--------|--------|
| 快速模型 | n=80 | 2个 | 162个 |
| 中速模型 | n=40 | 5个 | 205个 |
| 慢速模型 | n=25 | 4个 | 104个 |
| **总计** | - | **11个** | **471个** |

- **预计时间**: 180.7小时（7.53天）✅
- **可行性**: ✅ 可行（在1周内完成）
- **优势**: 平衡实验数量和时间，留有10%余量

#### 保守方案

| 模型分类 | n值 | 模型数 | 实验数 |
|---------|-----|--------|--------|
| 快速模型 | n=60 | 2个 | 122个 |
| 中速模型 | n=35 | 5个 | 180个 |
| 慢速模型 | n=20 | 4个 | 84个 |
| **总计** | - | **11个** | **386个** |

- **预计时间**: 149.3小时（6.22天）
- **可行性**: ✅ 可行（留有较大余量）

---

## ⭐ 最终推荐

### 推荐方案：分层n值 - 平衡方案

**理由**:
1. ✅ **时间可控**: 7.53天，在1周内完成
2. ✅ **实验数量充足**: 471个实验，比当前836个的一半还多
3. ✅ **充分利用资源**: 快速模型多跑，慢速模型适量
4. ✅ **留有余量**: 约10%的时间缓冲，应对失败重试
5. ✅ **数据丰富**: 允许多参数变异，探索参数交互效应

### 具体配置

#### 快速模型（2个）- n=80

- examples/mnist_ff
- examples/mnist

**配置示例**:
```json
{
  "comment": "examples/mnist - 扩展变异实验",
  "repo": "examples",
  "model": "mnist",
  "mode": "mutation",
  "runs_per_config": 80,
  "mutate_strategy": "random_multi_param",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"]
}
```

#### 中速模型（5个）- n=40

- examples/mnist_rnn
- examples/siamese
- bug-localization-by-dnn-and-rvsm/default
- MRT-OAST/default
- pytorch_resnet_cifar10/resnet20

**配置示例**:
```json
{
  "comment": "pytorch_resnet_cifar10/resnet20 - 扩展变异实验",
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mode": "mutation",
  "runs_per_config": 40,
  "mutate_strategy": "random_multi_param",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"]
}
```

#### 慢速模型（4个）- n=25

- VulBERTa/mlp
- Person_reID_baseline_pytorch/densenet121
- Person_reID_baseline_pytorch/hrnet18
- Person_reID_baseline_pytorch/pcb

**配置示例**:
```json
{
  "comment": "VulBERTa/mlp - 扩展变异实验",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "runs_per_config": 25,
  "mutate_strategy": "random_multi_param",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"]
}
```

### 预期成果

| 指标 | 当前值 | 新增值 | 总计 |
|------|--------|--------|------|
| **实验总数** | 836个 | +471个 | 1307个 |
| **能耗数据** | 795个 | +471个（预计） | 1266个 |
| **数据完整性** | 95.1% | 预计100% | 96.9% |
| **运行时间** | 17.91天 | +7.53天 | 25.44天 |

---

## 🔧 实现细节

### 1. 多参数变异策略

#### 随机多参数变异（推荐）

```python
# mutation/hyperparameter_mutator.py 新增功能

def mutate_random_multi_param(config, num_params_range=(1, 3)):
    """
    随机选择1-3个超参数进行变异

    Args:
        config: 模型配置
        num_params_range: 变异参数数量范围

    Returns:
        mutated_config: 变异后的配置
    """
    import random

    available_params = list(config['hyperparameters'].keys())
    num_params_to_mutate = random.randint(*num_params_range)
    params_to_mutate = random.sample(available_params, num_params_to_mutate)

    mutated_config = config.copy()
    for param in params_to_mutate:
        mutated_config[param] = mutate_single_param(param, config[param])

    return mutated_config
```

#### 全参数组合变异（可选）

适用于探索参数空间的特定区域：

```python
def mutate_all_combinations(config, sample_size=None):
    """
    生成所有参数组合的变异（或采样）

    Args:
        config: 模型配置
        sample_size: 采样数量（None则生成全部）

    Returns:
        list of mutated_configs
    """
    import itertools

    param_ranges = get_param_ranges(config)
    all_combinations = list(itertools.product(*param_ranges.values()))

    if sample_size and len(all_combinations) > sample_size:
        all_combinations = random.sample(all_combinations, sample_size)

    return [dict(zip(param_ranges.keys(), combo)) for combo in all_combinations]
```

### 2. 去重机制增强

确保多参数变异也能正确去重：

```python
# mutation/deduplication.py 增强功能

def generate_experiment_hash(config):
    """
    生成实验配置的哈希值（用于去重）

    Args:
        config: 实验配置字典

    Returns:
        hash_value: 配置的哈希值
    """
    import hashlib
    import json

    # 提取所有超参数（排序以确保一致性）
    hyperparams = {k: v for k, v in sorted(config.items())
                   if k.startswith('hyperparam_')}

    # 添加模型标识
    hash_input = {
        'repo': config['repo'],
        'model': config['model'],
        'mode': config.get('mode', 'mutation'),
        'hyperparams': hyperparams
    }

    # 生成哈希
    hash_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()
```

### 3. 配置文件格式

#### 新增字段

```json
{
  "comment": "模型描述",
  "repo": "仓库名",
  "model": "模型名",
  "mode": "mutation",

  // 新增字段
  "mutate_strategy": "random_multi_param",  // 变异策略
  "runs_per_config": 40,                    // 变异次数（即n值）
  "num_params_per_mutation": [1, 3],        // 每次变异的参数数量范围

  // 去重配置
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],
  "dedup_method": "config_hash"             // 去重方法
}
```

#### 完整示例

```json
{
  "comment": "VulBERTa/mlp - 多参数变异实验（25次）",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate_strategy": "random_multi_param",
  "runs_per_config": 25,
  "num_params_per_mutation": [1, 3],
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],
  "dedup_method": "config_hash",
  "parallel": false
}
```

### 4. 执行流程

```bash
# 步骤1: 生成配置文件
python3 scripts/generate_expansion_configs.py --strategy balanced --output settings/expansion_phase1.json

# 步骤2: 验证配置
python3 scripts/validate_config.py settings/expansion_phase1.json

# 步骤3: 运行实验（建议分批）
# 批次1: 快速模型（2个模型 × 81实验 = 162个实验，预计1.5天）
python3 mutation.py --config settings/expansion_phase1_fast.json

# 批次2: 中速模型（5个模型 × 41实验 = 205个实验，预计3.5天）
python3 mutation.py --config settings/expansion_phase1_medium.json

# 批次3: 慢速模型（4个模型 × 26实验 = 104个实验，预计2.5天）
python3 mutation.py --config settings/expansion_phase1_slow.json

# 步骤4: 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 步骤5: 更新data.csv
python3 tools/data_management/create_unified_data_csv.py
```

---

## 📝 配置书写规范更新

### 原规范回顾

**核心原则**（原设计）:
- ✅ 每个实验配置只能变异一个超参数
- ✅ 使用 `"mutate": ["参数名"]` 数组格式
- ✅ 使用 `"repo"` 而非 `"repository"`
- ✅ 使用 `"mode"` 而非 `"mutation_type"`

### 新规范扩展

**核心原则**（新设计）:
- ✅ **允许多参数变异**（不再限制单参数）
- ✅ 使用 `"mutate_strategy"` 指定变异策略
- ✅ 使用 `"runs_per_config": n` 指定变异次数
- ✅ 保留去重机制（必须启用）
- ✅ 其他规范保持不变

### 配置类型对比

| 配置类型 | 原设计 | 新设计 |
|---------|--------|--------|
| **默认值实验** | `"mode": "default"` | 保持不变 |
| **单参数变异** | `"mode": "mutation", "mutate": ["param"]` | 保持支持 |
| **多参数变异** | ❌ 不支持 | ✅ `"mutate_strategy": "random_multi_param"` |
| **变异次数** | `"runs_per_config": 5` | `"runs_per_config": n`（根据模型速度） |

### 示例对比

#### 原设计（单参数变异）

```json
{
  "comment": "VulBERTa/mlp - learning_rate变异",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 5
}
```

#### 新设计（多参数变异）

```json
{
  "comment": "VulBERTa/mlp - 多参数随机变异",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate_strategy": "random_multi_param",
  "runs_per_config": 25,
  "num_params_per_mutation": [1, 3],
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"]
}
```

---

## 🔬 研究意义

### 单参数变异 vs 多参数变异

#### 单参数变异（原设计）

**优势**:
- ✅ 清晰的因果关系（某参数对能耗的直接影响）
- ✅ 易于解释和分析
- ✅ 适合回归分析

**局限**:
- ❌ 无法探索参数交互效应
- ❌ 无法发现组合优化策略
- ❌ 实验数量受限于参数数量

#### 多参数变异（新设计）

**优势**:
- ✅ 探索参数交互效应（如learning_rate × batch_size）
- ✅ 发现最优参数组合
- ✅ 更丰富的数据支持机器学习建模
- ✅ 适合因果森林、SHAP分析等高级方法

**注意事项**:
- ⚠️ 需要更复杂的分析方法
- ⚠️ 因果推断难度增加
- ⚠️ 需要足够的样本量

### 数据分析策略

#### 1. 描述性分析
- 参数分布统计
- 能耗和性能指标分布
- 相关性分析

#### 2. 回归分析
- 多元线性回归（主效应）
- 交互项回归（交互效应）
- 随机森林回归（非线性关系）

#### 3. 因果推断
- 因果森林（Causal Forest）- 异质性因果效应
- SHAP值分析 - 特征重要性和交互
- 中介效应分析 - 因果机制探索

#### 4. 优化分析
- Pareto前沿分析 - 能耗-性能权衡
- 参数优化建议 - 基于实验数据的推荐

---

## ✅ 检查清单

### 实施前准备

- [ ] 确认GPU资源可用性（1周持续运行）
- [ ] 备份当前data/raw_data.csv
- [ ] 验证去重机制工作正常
- [ ] 准备监控脚本（实时跟踪进度）

### 配置文件准备

- [ ] 生成快速模型配置（n=80）
- [ ] 生成中速模型配置（n=40）
- [ ] 生成慢速模型配置（n=25）
- [ ] 验证所有配置文件格式正确

### 执行过程监控

- [ ] 每天检查实验进度
- [ ] 监控GPU/CPU利用率
- [ ] 及时处理失败实验
- [ ] 记录异常情况

### 完成后验证

- [ ] 验证实验总数（预期471个新实验）
- [ ] 检查数据完整性（能耗+性能）
- [ ] 更新data/raw_data.csv
- [ ] 重新生成data/data.csv
- [ ] 更新项目文档

---

## 📚 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目总指南
- [JSON_CONFIG_WRITING_STANDARDS.md](JSON_CONFIG_WRITING_STANDARDS.md) - JSON配置规范
- [PROJECT_PROGRESS_COMPLETE_SUMMARY.md](results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) - 项目进度
- [DATA_REPAIR_REPORT_20260104.md](results_reports/DATA_REPAIR_REPORT_20260104.md) - 数据修复报告

---

## 📞 问题与支持

如有疑问，请参考：
1. 本文档的相关章节
2. [CLAUDE.md](../CLAUDE.md) 中的常见问题
3. 项目Issues记录

---

**维护者**: Green
**文档版本**: 1.0
**最后更新**: 2026-01-05
**状态**: 📋 规划中 - 待用户确认后实施
