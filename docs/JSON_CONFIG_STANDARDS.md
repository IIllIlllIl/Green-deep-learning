# JSON配置编写规范

**文档版本**: 2.0 (统一版)
**创建日期**: 2025-12-13
**最后更新**: 2026-01-25
**适用版本**: v4.7.3+

> **文档合并说明 (2026-01-25)**:
> 本文档整合了原有的 `JSON_CONFIG_WRITING_STANDARDS.md` 和 `guides/JSON_CONFIG_BEST_PRACTICES.md`，
> 消除了约60%的重复内容，提供统一的配置编写指南。

---

## 📋 快速参考

| 配置类型 | 使用场景 | 格式 |
|---------|---------|------|
| **默认值实验** | 建立基线 | `"mode": "default"` |
| **单参数变异** | 研究单个参数影响 | `"mutate": ["参数名"]` |
| **多参数变异** | 研究参数交互 | `"mutate": ["参数1", "参数2"]` ⭐ 2026-01-05新增 |
| **并行模式** | foreground/background同时训练 | 使用 `foreground`/`background` 结构 |

---

## 🎯 核心概念

### `runs_per_config` 的语义

**定义**: 该配置项运行的次数

- ❌ **常见误解**: "每个参数运行N次"
- ✅ **正确理解**: "这个配置项运行N次"

**示例**:
```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "runs_per_config": 7,
  "mutate": ["epochs"]
}
```
**结果**: 运行7个实验，每个实验变异epochs参数

### `mutate` 的语义

**定义**: 每次运行时同时变异的参数列表

**重要规则**:
- 列表中的所有参数会在**每次运行时同时变异**
- 单参数变异: `"mutate": ["param1"]`
- 多参数变异: `"mutate": ["param1", "param2"]` (2026-01-05起支持)

---

## 📐 配置文件结构

### 顶层结构

```json
{
  "experiment_name": "配置名称",
  "description": "描述",
  "comment": "注释说明",

  "mode": "mutation",
  "max_retries": 2,
  "governor": "performance",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],

  "experiments": [
    // 实验配置列表
  ]
}
```

### 必须字段

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `mode` | string | 实验模式 | `"mutation"` |
| `max_retries` | number | 最大重试次数 | `2` |
| `governor` | string | CPU调频策略 | `"performance"` |
| `use_deduplication` | boolean | 启用去重 | `true` |
| `historical_csvs` | array | 历史数据文件列表 | `["data/raw_data.csv"]` |
| `experiments` | array | 实验配置列表 | **必需** |

---

## 🔧 实验配置类型

### 1. 默认值实验（Default）

**用途**: 建立基线，使用所有参数的默认值

**格式**:
```json
{
  "comment": "模型名 - 默认值实验",
  "repo": "仓库名",
  "model": "模型名",
  "mode": "default",
  "runs_per_config": 1
}
```

**示例**:
```json
{
  "comment": "VulBERTa/mlp - 默认值实验",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "default",
  "runs_per_config": 3
}
```

### 2. 单参数变异（Mutation）

**用途**: 研究单个参数对模型的影响

**格式**:
```json
{
  "comment": "模型名 - 参数变异说明",
  "repo": "仓库名",
  "model": "模型名",
  "mode": "mutation",
  "mutate": ["参数名"],
  "runs_per_config": 5
}
```

**示例**:
```json
{
  "comment": "VulBERTa/mlp - 变异epochs",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["epochs"],
  "runs_per_config": 7
}
```

### 3. 多参数变异（Mutation - Multi）⭐ 2026-01-05新增

**用途**: 研究多个参数的交互影响

**格式**:
```json
{
  "comment": "模型名 - 多参数变异",
  "repo": "仓库名",
  "model": "模型名",
  "mode": "mutation",
  "mutate": ["参数1", "参数2"],
  "runs_per_config": 5
}
```

**示例**:
```json
{
  "comment": "VulBERTa/mlp - epochs + learning_rate",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["epochs", "learning_rate"],
  "runs_per_config": 10
}
```

**注意**: 多参数变异实验设计详见 [docs/EXPERIMENT_EXPANSION_PLAN_20260105.md](EXPERIMENT_EXPANSION_PLAN_20260105.md)

### 4. 并行模式（Parallel）

**用途**: foreground和background同时训练

**格式**:
```json
{
  "comment": "并行训练 - 前台和后台",
  "repo": "仓库名",
  "model": "模型名",
  "mode": "parallel",

  "foreground": {
    "mode": "mutation",
    "mutate": ["参数名"],
    "runs_per_config": 5
  },

  "background": {
    "mode": "mutation",
    "mutate": ["参数名"],
    "runs_per_config": 5
  }
}
```

**示例**:
```json
{
  "comment": "Person_reID_baseline_pytorch - 并行epochs变异",
  "repo": "Person_reID_baseline_pytorch",
  "model": "pcb",
  "mode": "parallel",

  "foreground": {
    "mode": "mutation",
    "mutate": ["epochs"],
    "runs_per_config": 10
  },

  "background": {
    "mode": "mutation",
    "mutate": ["epochs"],
    "runs_per_config": 10
  }
}
```

---

## ❌ 常见错误与修正

### 错误1: 使用`mutate_params`对象（旧格式）

**❌ 错误**:
```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "mutate_params": ["epochs", "learning_rate"]
}
```

**✅ 正确**:
```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "mutate": ["epochs", "learning_rate"]
}
```

**原因**: 旧版格式使用`mutate_params`，新版统一为`mutate`

### 错误2: 使用`repository`而非`repo`

**❌ 错误**:
```json
{
  "repository": "VulBERTa"
}
```

**✅ 正确**:
```json
{
  "repo": "VulBERTa"
}
```

### 错误3: 使用`mutation_type`而非`mode`

**❌ 错误**:
```json
{
  "mutation_type": "mutation"
}
```

**✅ 正确**:
```json
{
  "mode": "mutation"
}
```

### 错误4: 误解多参数变异的影响

**问题配置**:
```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "runs_per_config": 7,
  "mutate": ["epochs", "learning_rate", "seed", "weight_decay"]
}
```

**常见误解**:
- ❌ 预期: 4参数 × 7次 = 28个实验（每个参数独立）
- ✅ 实际: 7个实验（每个实验同时变异4个参数）

**实际行为**:
```python
# 生成7个mutations，每个mutation同时包含4个参数的变异
{
  "epochs": 15,           # 同时变异
  "learning_rate": 0.001,  # 同时变异
  "seed": 4287,           # 同时变异
  "weight_decay": 0.0001  # 同时变异
}
```

**建议**:
- 如果需要单参数分析: 为每个参数创建独立配置项
- 如果需要多参数交互: 使用多参数变异配置（2026-01-05起支持）

---

## 📝 完整示例

### 示例1: 非并行单参数变异

```json
{
  "experiment_name": "mnist_batch_size_experiments",
  "description": "研究batch_size对MNIST训练的影响",
  "comment": "examples/mnist - batch_size变异",

  "mode": "mutation",
  "max_retries": 2,
  "governor": "performance",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],

  "experiments": [
    {
      "comment": "mnist - 默认值",
      "repo": "examples",
      "model": "mnist",
      "mode": "default",
      "runs_per_config": 3
    },
    {
      "comment": "mnist - 变异batch_size",
      "repo": "examples",
      "model": "mnist",
      "mode": "mutation",
      "mutate": ["batch_size"],
      "runs_per_config": 10
    }
  ]
}
```

### 示例2: 并行模式单参数变异

```json
{
  "experiment_name": "resnet_epochs_parallel",
  "description": "并行训练ResNet，变异epochs",
  "comment": "pytorch_resnet_cifar10/resnet20 - 并行epochs变异",

  "mode": "mutation",
  "max_retries": 2,
  "governor": "performance",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],

  "experiments": [
    {
      "comment": "resnet20 - 并行epochs变异",
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mode": "parallel",

      "foreground": {
        "mode": "mutation",
        "mutate": ["epochs"],
        "runs_per_config": 10
      },

      "background": {
        "mode": "mutation",
        "mutate": ["epochs"],
        "runs_per_config": 10
      }
    }
  ]
}
```

### 示例3: 多参数变异（2026-01-05新增）

```json
{
  "experiment_name": "vulberta_multi_param",
  "description": "研究多参数交互影响",
  "comment": "VulBERTa/mlp - 多参数变异",

  "mode": "mutation",
  "max_retries": 2,
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],

  "experiments": [
    {
      "comment": "mlp - epochs + learning_rate交互",
      "repo": "VulBERTa",
      "model": "mlp",
      "mode": "mutation",
      "mutate": ["epochs", "learning_rate"],
      "runs_per_config": 20
    }
  ]
}
```

---

## ✅ 最佳实践

### 1. 参数变异原则

**单参数变异**（传统方法）:
- 每个配置项只变异一个参数
- 适合独立参数影响分析
- 配置简单，结果易解释

**多参数变异**（2026-01-05起支持）:
- 可同时变异多个参数
- 适合研究参数交互效应
- 需要更多实验才能充分探索

### 2. 配置项命名规范

**格式**: `模型名 - 变异说明`

**示例**:
- `"mnist - 默认值"`
- `"mnist - batch_size变异"`
- `"resnet20 - 并行epochs变异"`

### 3. 实验数估算

**公式**: `总实验数 = Σ(runs_per_config × 配置项数)`

**示例**:
```json
{
  "experiments": [
    {"runs_per_config": 3, "mutate": ["epochs"]},      // 3个实验
    {"runs_per_config": 5, "mutate": ["batch_size"]},  // 5个实验
    {"runs_per_config": 2, "mode": "default"}          // 2个实验
  ]
}
// 总计: 3 + 5 + 2 = 10个实验
```

### 4. 去重配置最佳实践

- **启用去重**: `"use_deduplication": true`
- **指定历史数据**: `"historical_csvs": ["data/raw_data.csv"]`
- **自动跳过**: 已存在的实验配置不会重复运行

### 5. 版本控制

- 配置文件命名: `experiment_name_YYYYMMDD.json`
- 记录变更: 在 `comment` 字段说明变更原因
- 备份配置: 重要配置文件备份到 `settings/backups/`

---

## ✅ 验证清单

### 1. JSON格式验证

```bash
# 使用python验证JSON格式
python -m json.tool settings/your_config.json
```

### 2. 字段验证

- [ ] 使用 `"repo"` 而非 `"repository"`
- [ ] 使用 `"mode"` 而非 `"mutation_type"`
- [ ] 使用 `"mutate"` 而非 `"mutate_params"`
- [ ] `"runs_per_config"` 为正整数
- [ ] `"experiments"` 列表非空

### 3. 结构验证

- [ ] 每个配置项有明确的 `comment`
- [ ] `mode` 值为: `default`, `mutation`, 或 `parallel`
- [ ] 并行模式包含 `foreground` 和 `background`
- [ ] 启用了 `use_deduplication`
- [ ] 指定了 `historical_csvs`

### 4. 语义验证

- [ ] 理解 `runs_per_config` 的正确含义
- [ ] 单参数变异: `"mutate": ["param1"]`
- [ ] 多参数变异: `"mutate": ["param1", "param2"]`
- [ ] 理解多参数会同时变异

---

## 📚 相关文档

- [实验扩展方案 2026-01-05](EXPERIMENT_EXPANSION_PLAN_20260105.md) - 多参数变异详细设计
- [参考/SCRIPTS_QUICKREF.md](reference/SCRIPTS_QUICKREF.md) - 配置管理脚本快速参考
- [CLAUDE_FULL_REFERENCE.md](CLAUDE_FULL_REFERENCE.md) - 项目完整参考

---

**文档维护**: 本文档合并了原有的 STANDARDS 和 BEST_PRACTICES 文档
**归档位置**: `archived/JSON_CONFIG_*.backup_20260125_2`
**合并日期**: 2026-01-25
**重复内容消除**: 约60%
