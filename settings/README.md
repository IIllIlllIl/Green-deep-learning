# Experiment Configuration Files

本目录包含实验配置文件，用于批量运行训练实验，比命令行模式更方便。

## 🎯 预设配置文件

### all.json - 全面变异实验
**用途**: 变异所有模型的所有可变异超参数

**包含**:
- 所有6个仓库的所有16个模型
- 每个模型变异所有支持的超参数
- 每个配置运行5次

**运行方式**:
```bash
python3 mutation_runner.py --experiment-config settings/all.json
```

**预计时间**: 根据模型大小和epochs，约10-50小时

**用途场景**:
- 全面的超参数探索
- 建立完整的性能-能耗数据库
- 长期实验

---

### default.json - 基线复现实验
**用途**: 复现所有模型的原始训练过程（使用默认超参数）

**包含**:
- 所有16个模型
- 使用各模型的原始默认超参数
- 每个模型运行1次

**运行方式**:
```bash
python3 mutation_runner.py --experiment-config settings/default.json
```

**预计时间**: 约5-20小时

**用途场景**:
- 建立性能基线（baseline）
- 与变异实验对比
- 验证原始训练流程
- 能耗基准测试

**特点**:
✅ **这是唯一能复现原始训练过程的方式**
- 使用各仓库train.sh中定义的默认超参数
- 不进行任何随机变异
- 可作为后续实验的对照组

---

### resnet_all_models.json - 单仓库多模型实验
**用途**: 对pytorch_resnet_cifar10的所有模型进行变异实验

**包含**:
- resnet20, resnet32, resnet44, resnet56
- 每个模型运行3次

**运行方式**:
```bash
python3 mutation_runner.py --experiment-config settings/resnet_all_models.json
```

**预计时间**: 约6-10小时

**用途场景**:
- 比较不同ResNet深度的能耗特征
- 单一模型家族的系统研究

---

### learning_rate_study.json - 学习率影响研究
**用途**: 研究学习率对能耗和性能的影响

**包含**:
- 3个代表性模型
- 只变异learning_rate
- 每个模型10次变异

**运行方式**:
```bash
python3 mutation_runner.py --experiment-config settings/learning_rate_study.json
```

**预计时间**: 约5-8小时

**用途场景**:
- 研究单一超参数的影响
- 控制变量实验
- 快速验证假设

---

### mixed_mode_demo.json - 混合模式演示
**用途**: 演示如何在一个配置中混合default和mutation模式

**包含**:
- 1个基线实验（default模式）
- 2个变异实验（mutation模式）

**运行方式**:
```bash
python3 mutation_runner.py --experiment-config settings/mixed_mode_demo.json
```

**预计时间**: 约30-60分钟

**用途场景**:
- 学习配置文件格式
- 测试新想法
- 快速对比实验

---

## 📝 配置文件格式

### 基本结构

```json
{
  "experiment_name": "实验名称",
  "description": "实验描述",
  "governor": "performance",           // CPU调度器模式（可选）
  "runs_per_config": 5,               // 每个配置运行几次
  "max_retries": 2,                   // 失败时最大重试次数
  "mode": "mutation",                 // 全局模式：mutation 或 default
  "experiments": [                    // 实验列表
    {
      "repo": "repository_name",
      "model": "model_name",
      "mode": "mutation",              // 单个实验的模式（可覆盖全局）
      "mutate": ["all"],               // mutation模式：要变异的参数
      "hyperparameters": {...},        // default模式：固定的超参数
      "comment": "注释"                 // 可选的注释
    }
  ]
}
```

### 字段说明

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| `experiment_name` | 是 | string | 实验名称，用于标识 |
| `description` | 否 | string | 实验描述 |
| `governor` | 否 | string | CPU调度器：performance/powersave/ondemand |
| `runs_per_config` | 否 | int | 每个配置运行次数（默认1） |
| `max_retries` | 否 | int | 失败重试次数（默认2） |
| `mode` | 否 | string | 全局模式：mutation/default（默认mutation） |
| `experiments` | 是 | array | 实验配置列表 |

### experiments数组元素

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| `repo` | 是 | string | 仓库名称 |
| `model` | 是 | string | 模型名称 |
| `mode` | 否 | string | 该实验的模式（覆盖全局mode） |
| `mutate` | mutation模式必需 | array | 要变异的超参数列表或["all"] |
| `hyperparameters` | default模式必需 | object | 固定的超参数值 |
| `comment` | 否 | string | 注释说明 |

---

## 🔧 使用模式

### Mode 1: Mutation模式（变异）

**用途**: 自动生成随机的超参数变体进行探索

**配置示例**:
```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mutate": ["epochs", "learning_rate"],
  "comment": "变异epochs和learning_rate，其他参数使用train.sh的默认值"
}
```

**行为**:
- 从配置的range范围内随机生成超参数值
- 未指定的参数由train.sh使用默认值（不会传递）
- `"mutate": ["all"]` 表示变异所有支持的超参数

### Mode 2: Default模式（默认）

**用途**: 使用指定的超参数值运行，用于复现或基线实验

**配置示例**:
```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mode": "default",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 0.0001
  },
  "comment": "使用原始默认超参数"
}
```

**行为**:
- 直接使用hyperparameters中指定的值
- 不进行任何随机变异
- 适合复现原始训练过程

---

## 📊 实验结果

所有实验结果保存在 `results/` 目录：

```
results/
├── 20251105_180000_pytorch_resnet_cifar10_resnet20.json
├── 20251105_180500_VulBERTa_mlp.json
├── energy_20251105_180000_pytorch_resnet_cifar10_resnet20/
│   ├── cpu_energy.txt
│   └── gpu_power.csv
└── training_pytorch_resnet_cifar10_resnet20_20251105_180000.log
```

每个JSON结果文件包含:
- 实验ID和时间戳
- 使用的超参数
- 训练时长
- CPU和GPU能耗
- 性能指标（准确率、损失等）
- 训练是否成功
- 重试次数

---

## 💡 最佳实践

### 1. 先运行default.json建立基线

```bash
# 第一步：建立基线
python3 mutation_runner.py --experiment-config settings/default.json

# 第二步：运行变异实验
python3 mutation_runner.py --experiment-config settings/all.json

# 第三步：对比结果
```

### 2. 从小规模开始测试

```bash
# 先用小配置测试
python3 mutation_runner.py --experiment-config settings/mixed_mode_demo.json

# 确认无误后运行大规模实验
python3 mutation_runner.py --experiment-config settings/all.json
```

### 3. 使用性能模式减少干扰

所有配置文件都建议设置:
```json
"governor": "performance"
```

运行时使用sudo:
```bash
sudo python3 mutation_runner.py --experiment-config settings/all.json
```

### 4. 监控实验进度

```bash
# 在另一个终端监控结果
watch -n 10 'ls -lh results/*.json | wc -l'

# 监控最新日志
tail -f results/training_*.log
```

### 5. 分批运行长期实验

将all.json拆分成多个文件：
- all_part1.json: 仓库1-3
- all_part2.json: 仓库4-6

分批运行避免单次实验时间过长。

---

## 🎓 示例场景

### 场景1: 复现原始训练 + 对比变异

```bash
# 步骤1: 运行基线
sudo python3 mutation_runner.py --experiment-config settings/default.json

# 步骤2: 运行变异（只变异learning_rate）
sudo python3 mutation_runner.py --experiment-config settings/learning_rate_study.json

# 步骤3: 分析结果
cd results
cat *.json | jq '[.repository, .model, .hyperparameters.learning_rate, .performance_metrics, .energy_metrics.cpu_energy_total_joules] | @csv'
```

### 场景2: 研究特定模型家族

```bash
# 只研究ResNet系列
sudo python3 mutation_runner.py --experiment-config settings/resnet_all_models.json
```

### 场景3: 快速原型验证

创建custom.json:
```json
{
  "experiment_name": "quick_test",
  "runs_per_config": 1,
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist_cnn",
      "mutate": ["epochs", "learning_rate"]
    }
  ]
}
```

运行:
```bash
python3 mutation_runner.py --experiment-config custom.json
```

---

## 📁 创建自定义配置

### ���板

```json
{
  "experiment_name": "my_experiment",
  "description": "我的实验描述",
  "governor": "performance",
  "runs_per_config": 3,
  "max_retries": 2,
  "experiments": [
    {
      "repo": "仓库名",
      "model": "模型名",
      "mutate": ["要变异的超参数"],
      "comment": "可选注释"
    }
  ]
}
```

### 查看可用的仓库和模型

```bash
python3 mutation_runner.py --list
```

### 验证配置文件

```bash
# Python验证JSON语法
python3 -c "import json; print(json.load(open('settings/my_config.json')))"

# 或使用jq
jq . settings/my_config.json
```

---

## ⚠️ 注意事项

1. **时间估算**: 大规模实验可能需要数十小时，建议使用screen/tmux
2. **磁盘空间**: 确保有足够空间存储日志和结果（每个实验约10-50MB）
3. **GPU占用**: 实验会占用GPU，确保没有其他任务在运行
4. **能耗监控**: 需要root权限访问perf，建议使用sudo运行
5. **休眠时间**: 配置文件中的休眠时间（60秒/120秒）可根据需要调整

---

## 🔗 相关文档

- [主文档](../README.md)
- [配置说明](../docs/CONFIG_EXPLANATION.md)
- [使用示例](../docs/USAGE_EXAMPLES.md)

---

## 📞 获取帮助

```bash
# 查看命令行帮助
python3 mutation_runner.py --help

# 列出可用模型
python3 mutation_runner.py --list

# 验证配置文件（会输出实验数量）
python3 -c "import json; c=json.load(open('settings/all.json')); print(f\"Total: {len(c['experiments'])} experiments\")"
```
