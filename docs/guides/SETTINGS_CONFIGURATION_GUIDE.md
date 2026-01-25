# Settings配置文件完全指南

**最后更新**: 2025-11-09

## 概述

`settings/`目录包含实验配置文件（JSON格式），用于批量运行训练实验。每个配置文件定义了一组实验，包括要训练的模型、超参数设置、运行次数等。

---

## 配置文件结构

### 完整示例

```json
{
  "experiment_name": "my_experiment",
  "description": "实验描述",
  "governor": "performance",
  "runs_per_config": 5,
  "max_retries": 2,
  "mode": "default",
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mode": "default",
      "hyperparameters": {
        "epochs": 200,
        "learning_rate": 0.1,
        "seed": 42
      },
      "comment": "可选注释"
    }
  ]
}
```

---

## 顶层配置字段

### 1. `experiment_name` (必需)

**类型**: 字符串
**作用**: 实验名称，用于标识这组实验

**示例**:
```json
"experiment_name": "baseline_reproduction"
```

---

### 2. `description` (可选)

**类型**: 字符串
**作用**: 详细描述实验目的和内容

**示例**:
```json
"description": "Reproduce original training process for all models using default hyperparameters"
```

---

### 3. `governor` (可选)

**类型**: 字符串
**可选值**: `"performance"`, `"powersave"`, `"ondemand"`, `"conservative"`, `"userspace"`
**默认值**: 不设置（保持系统默认）

**作用**: 设置CPU频率调度器，影响能耗测量的准确性

**推荐**:
- 能耗研究实验: `"performance"` (固定最高频率，减少干扰)
- 普通训练: 不设置或使用默认值

**示例**:
```json
"governor": "performance"
```

---

### 4. `runs_per_config` (可选)

**类型**: 整数
**默认值**: 1
**作用**: 每个实验配置运行的次数

**用途**:
- `1`: 单次运行，快速验证
- `3-5`: 基本统计，获得平均值
- `10+`: 详细研究，更可靠的统计结果

**示例**:
```json
"runs_per_config": 5
```

**注意**:
- 对于`mode: "mutate"`，这是**每个变异配置**运行的次数
- 对于`mode: "default"`，这是该固定配置运行的次数

---

### 5. `max_retries` (可选)

**类型**: 整数
**默认值**: 2 (mutation.py中定义的DEFAULT_MAX_RETRIES)
**作用**: 训练失败时的最大重试次数

**示例**:
```json
"max_retries": 3
```

**推荐**:
- 稳定环境: 2-3次
- 测试阶段: 0-1次（快速发现问题）
- 生产环境: 3-5次（确保可靠性）

---

### 6. `mode` (可选)

**类型**: 字符串
**可选值**: `"default"`, `"mutate"`
**默认值**: 由实验条目决定

**作用**: 顶层默认模式（可被实验条目覆盖）

**示例**:
```json
"mode": "default"
```

---

### 7. `experiments` (必需)

**类型**: 数组
**作用**: 包含所有实验配置的列表

**示例**:
```json
"experiments": [
  { "repo": "...", "model": "..." },
  { "repo": "...", "model": "..." }
]
```

---

## 实验条目配置

每个实验条目定义一个模型的训练配置。有两种模式：

### 模式1: 固定超参数 (mode: "default")

使用明确指定的超参数值，不进行变异。

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mode": "default",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 0.0001,
    "seed": 42
  },
  "comment": "ResNet20原始配置"
}
```

**运行结果**:
- 使用指定的超参数值
- 运行`runs_per_config`次
- 结果可重复（如果设置了seed）

---

### 模式2: 超参数变异 (mode: "mutate")

自动生成超参数变体进行实验。

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mutate": ["learning_rate", "dropout"],
  "comment": "变异学习率和dropout"
}
```

或变异所有支持的超参数：

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mutate": ["all"]
}
```

**运行结果**:
- 系统自动在参数范围内生成随机值
- 每个变异配置运行`runs_per_config`次
- 适合探索超参数空间

---

## 实验条目字段详解

### 1. `repo` (必需)

**类型**: 字符串
**作用**: 模型仓库名称

**可用值**:
- `"pytorch_resnet_cifar10"`
- `"VulBERTa"`
- `"Person_reID_baseline_pytorch"`
- `"MRT-OAST"`
- `"bug-localization-by-dnn-and-rvsm"`
- `"examples"`

**查看所有可用仓库**:
```bash
python3 mutation.py --list
```

---

### 2. `model` (必需)

**类型**: 字符串
**作用**: 模型名称

**示例** (取决于仓库):
- `pytorch_resnet_cifar10`: `"resnet20"`, `"resnet32"`, `"resnet44"`, `"resnet56"`
- `VulBERTa`: `"mlp"`, `"cnn"`
- `Person_reID_baseline_pytorch`: `"densenet121"`, `"hrnet18"`, `"pcb"`
- `MRT-OAST`: `"default"`
- `bug-localization-by-dnn-and-rvsm`: `"default"`
- `examples`: `"mnist_cnn"`, `"mnist_rnn"`, `"siamese"`

---

### 3. `mode` (可选)

**类型**: 字符串
**可选值**: `"default"`, `"mutate"`
**作用**: 该实验的运行模式

**自动检测规则**:
- 如果提供`hyperparameters`字段 → `"default"`模式
- 如果提供`mutate`字段 → `"mutate"`模式
- 如果都不提供 → 使用顶层`mode`或默认为`"default"`

---

### 4. `hyperparameters` (mode: default时必需)

**类型**: 对象
**作用**: 固定的超参数值

**可用字段** (取决于模型):

#### 通用超参数:
- `epochs` (整数): 训练轮数
- `learning_rate` (浮点): 学习率
- `seed` (整数): 随机种子

#### 模型特定超参数:
- `weight_decay` (浮点): L2正则化系数
  - 支持: pytorch_resnet_cifar10, VulBERTa, MRT-OAST

- `dropout` (浮点): Dropout概率
  - 支持: Person_reID_baseline_pytorch, MRT-OAST

- `max_iter` (整数): 最大迭代次数
  - 支持: bug-localization-by-dnn-and-rvsm

- `kfold` (整数): K折交叉验证
  - 支持: bug-localization-by-dnn-and-rvsm

- `alpha` (浮点): L2正则化参数
  - 支持: bug-localization-by-dnn-and-rvsm

**示例**:
```json
"hyperparameters": {
  "epochs": 100,
  "learning_rate": 0.01,
  "seed": 42,
  "weight_decay": 0.0001
}
```

**查看模型支持的超参数**:
```bash
python3 mutation.py --list
# 或查看
cat config/models_config.json | jq '.models["仓库名"].supported_hyperparams'
```

---

### 5. `mutate` (mode: mutate时必需)

**类型**: 字符串数组
**作用**: 指定要变异的超参数

**可选值**:
- `["all"]`: 变异所有支持的超参数
- `["epochs"]`: 只变异epochs
- `["learning_rate", "dropout"]`: 变异多个指定参数

**示例**:
```json
"mutate": ["learning_rate", "seed"]
```

**变异范围**:
- 由`config/models_config.json`中的`range`字段定义
- 系统在范围内随机选择值

---

### 6. `comment` (可选)

**类型**: 字符串
**作用**: 注释说明，帮助理解配置

**示例**:
```json
"comment": "这是baseline配置，使用原始默认值"
```

---

## 现有配置文件说明

### 1. `default.json` ⭐ (推荐先运行)

**用途**: 复现所有模型的原始训练
**模式**: 固定超参数
**实验数**: 15个模型
**运行次数**: 每个1次

**特点**:
- 使用各模型的默认超参数
- 建立性能基线
- 验证环境配置正确

**运行时间**: 数小时到数天（取决于模型）

```bash
sudo python3 mutation.py --experiment-config settings/default.json
```

---

### 2. `all.json`

**用途**: 变异所有模型的所有超参数
**模式**: 超参数变异
**实验数**: 15个模型
**运行次数**: 每个变异5次

**特点**:
- 全面探索超参数空间
- 生成大量数据
- 适合深入研究

**运行时间**: 非常长（数周）

**注意**: ⚠️ 这会生成大量实验，建议先运行部分测试

```bash
# 警告：这会运行非常多的实验
sudo python3 mutation.py --experiment-config settings/all.json
```

---

### 3. `learning_rate_study.json`

**用途**: 研究学习率对能耗和性能的影响
**模式**: 仅变异learning_rate
**实验数**: 3个模型
**运行次数**: 每个变异10次

**特点**:
- 专注研究单一超参数
- 适合发表论文/报告
- 统计结果可靠

**运行时间**: 中等（数小时到1天）

```bash
sudo python3 mutation.py --experiment-config settings/learning_rate_study.json
```

---

### 4. `resnet_all_models.json`

**用途**: 对比ResNet系列所有模型
**模式**: 固定超参数
**实验数**: 4个ResNet模型

**特点**:
- 研究模型深度影响
- 同一架构不同规模

---

### 5. `full_test_run.json` ⭐

**用途**: 完整测试所有仓库（每个仓库选1个模型）
**模式**: 固定超参数
**实验数**: 6个模型
**运行次数**: 每个1次

**特点**:
- 验证框架功能
- 测试sudo环境兼容性
- 快速全面检查

**运行时间**: 数小时

```bash
sudo python3 mutation.py --experiment-config settings/full_test_run.json
```

---

### 6. `failed_models_quick_test.json` ⚠️

**用途**: 快速验证之前失败的模型修复
**模式**: 固定超参数（极小训练量）
**实验数**: 3个模型
**运行次数**: 每个1次

**特点**:
- 训练量极小（1 epoch或1000 iterations）
- 快速验证修复
- 仅用于测试，不产生有意义的性能指标

**运行时间**: 10-20分钟

```bash
sudo python3 mutation.py --experiment-config settings/failed_models_quick_test.json
```

---

### 7. `mixed_mode_demo.json`

**用途**: 演示混合使用default和mutate模式
**模式**: 混合
**实验数**: 少量

**特点**:
- 展示配置文件灵活性
- 学习示例

---

## 创建自定义配置文件

### 示例1: 单一模型详细研究

```json
{
  "experiment_name": "resnet20_detailed_study",
  "description": "深入研究ResNet20的超参数影响",
  "governor": "performance",
  "runs_per_config": 10,
  "max_retries": 3,
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["all"],
      "comment": "变异所有超参数，每个配置运行10次"
    }
  ]
}
```

---

### 示例2: 对比实验

```json
{
  "experiment_name": "model_comparison",
  "description": "对比不同模型在相同超参数下的表现",
  "governor": "performance",
  "runs_per_config": 3,
  "mode": "default",
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "hyperparameters": {
        "epochs": 100,
        "learning_rate": 0.1,
        "seed": 42
      }
    },
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet32",
      "hyperparameters": {
        "epochs": 100,
        "learning_rate": 0.1,
        "seed": 42
      }
    },
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet44",
      "hyperparameters": {
        "epochs": 100,
        "learning_rate": 0.1,
        "seed": 42
      }
    }
  ]
}
```

---

### 示例3: 快速原型验证

```json
{
  "experiment_name": "quick_prototype",
  "description": "快速验证新模型或配置",
  "runs_per_config": 1,
  "max_retries": 0,
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist_cnn",
      "mode": "default",
      "hyperparameters": {
        "epochs": 1,
        "learning_rate": 0.01,
        "seed": 1
      },
      "comment": "极小训练量，快速验证"
    }
  ]
}
```

---

## 配置文件最佳实践

### 1. 命名规范

**推荐格式**: `<purpose>_<scope>.json`

**示例**:
- `baseline_all_models.json` - 所有模型的基线
- `lr_study_resnet.json` - ResNet学习率研究
- `quick_test_bugs.json` - 快速测试bug修复

---

### 2. 注释说明

**始终添加**:
- `description`: 实验目的
- 每个实验的`comment`: 特殊说明

**示例**:
```json
{
  "experiment_name": "energy_efficiency_study",
  "description": "研究不同超参数对能耗效率的影响，用于2025论文",
  "experiments": [
    {
      "repo": "...",
      "model": "...",
      "comment": "这是baseline配置，作为对照组"
    }
  ]
}
```

---

### 3. 渐进式测试

**建议顺序**:
1. **快速验证** (1-2个模型，1 epoch)
2. **中等测试** (3-5个模型，正常epochs)
3. **完整实验** (所有配置)

**示例流程**:
```bash
# 步骤1: 快速验证
sudo python3 mutation.py --experiment-config settings/quick_test.json

# 步骤2: 检查结果
ls results/*.json | tail -3

# 步骤3: 运行完整实验
sudo python3 mutation.py --experiment-config settings/full_experiment.json
```

---

### 4. 版本控制

**建议**:
- 使用git管理配置文件
- 在`description`中记录日期和版本
- 保留历史配置以便重现

**示例**:
```json
{
  "experiment_name": "baseline_v2",
  "description": "第二版baseline实验（2025-11-09），修复了Person_reID的scipy问题",
  ...
}
```

---

### 5. 参数验证

**运行前检查**:
```bash
# 检查JSON语法
python3 -m json.tool settings/your_config.json

# 预览配置
cat settings/your_config.json | jq .

# 查看会运行多少实验
cat settings/your_config.json | jq '.experiments | length'
```

---

## 常见配置模式

### 模式1: Ablation Study (消融研究)

研究每个超参数的独立影响：

```json
{
  "experiment_name": "ablation_study",
  "description": "消融研究：独立测试每个超参数的影响",
  "runs_per_config": 5,
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["learning_rate"],
      "comment": "只变异学习率"
    },
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["weight_decay"],
      "comment": "只变异weight decay"
    },
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["epochs"],
      "comment": "只变异epochs"
    }
  ]
}
```

---

### 模式2: Grid Search (网格搜索)

系统测试参数组合（需要手动列举）：

```json
{
  "experiment_name": "grid_search",
  "runs_per_config": 3,
  "mode": "default",
  "experiments": [
    {"repo": "...", "hyperparameters": {"learning_rate": 0.1, "weight_decay": 0.0001}},
    {"repo": "...", "hyperparameters": {"learning_rate": 0.1, "weight_decay": 0.001}},
    {"repo": "...", "hyperparameters": {"learning_rate": 0.01, "weight_decay": 0.0001}},
    {"repo": "...", "hyperparameters": {"learning_rate": 0.01, "weight_decay": 0.001}}
  ]
}
```

---

### 模式3: Reproducibility Test (可重复性测试)

使用固定seed多次运行：

```json
{
  "experiment_name": "reproducibility_test",
  "runs_per_config": 10,
  "mode": "default",
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist_cnn",
      "hyperparameters": {
        "epochs": 10,
        "learning_rate": 0.01,
        "seed": 42
      },
      "comment": "固定seed，测试结果的可重复性"
    }
  ]
}
```

---

## 故障排查

### 问题1: JSON语法错误

**错误信息**: `json.decoder.JSONDecodeError`

**解决**:
```bash
# 验证JSON语法
python3 -m json.tool settings/your_config.json
```

**常见错误**:
- 缺少逗号
- 多余的逗号（最后一项后）
- 引号不匹配
- 注释（JSON不支持//或/* */）

---

### 问题2: 配置字段错误

**错误信息**: `KeyError` 或 `Invalid configuration`

**检查**:
- `repo`名称是否正确
- `model`名称是否该仓库支持
- `hyperparameters`字段名是否正确

**验证**:
```bash
python3 mutation.py --list
```

---

### 问题3: 超参数不支持

**错误信息**: `Unsupported hyperparameter`

**原因**: 该模型不支持指定的超参数

**解决**:
查看`config/models_config.json`:
```bash
cat config/models_config.json | jq '.models["your_repo"].supported_hyperparams'
```

---

## 相关文档

- [修复与测试指南](FIXES_AND_TESTING.md) - 已知问题和修复
- [配置文件说明](CONFIG_EXPLANATION.md) - models_config.json详解
- [快速参考](quick_reference.md) - 常用命令
- [主README](../README.md) - 项目整体说明

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
