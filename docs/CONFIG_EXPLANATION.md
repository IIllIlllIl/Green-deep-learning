# models_config.json 配置文件说明

## 为什么需要这个配置文件？

### 1. 统一不同仓库的参数接口

每个仓库的train.sh使用不同的参数命名和格式：

| 仓库 | Epochs参数 | Learning Rate参数 | Dropout参数 |
|------|-----------|------------------|-------------|
| pytorch_resnet_cifar10 | `-e` 或 `--epochs` | `--lr` | 不支持 |
| VulBERTa | `--epochs` | `--learning_rate` | 不支持 |
| Person_reID_baseline_pytorch | `--total_epoch` | `--lr` | `--droprate` |
| MRT-OAST | `--epochs` | `--lr` | `--dropout` |

**配置文件通过`flag`字段统一了这些差异。**

### 2. 明确每个仓库支持的超参数

不同仓库支持的超参数集合不同：

- **pytorch_resnet_cifar10**: epochs, learning_rate, seed, weight_decay
- **VulBERTa**: epochs, learning_rate, seed, weight_decay
- **Person_reID**: epochs, learning_rate, seed, dropout
- **MRT-OAST**: epochs, learning_rate, seed, dropout, weight_decay

**配置文件的`supported_hyperparams`明确记录了每个仓库支持哪些超参数。**

### 3. 定义合理的超参数变异范围

不同模型的合理超参数范围差异很大：

- pytorch_resnet_cifar10的epochs通常是200（范围50-200）
- VulBERTa的epochs通常是10（范围5-20）
- pytorch_resnet_cifar10的learning_rate是0.1（范围0.01-0.2）
- VulBERTa的learning_rate是0.00003（范围0.00001-0.0001）

**配置文件的`range`字段确保变异在合理范围内，避免生成无效的超参数组合。**

### 4. 定义性能指标提取规则

不同仓库的训练日志格式不同：

- pytorch_resnet_cifar10: "测试准确率: 91.25%"
- VulBERTa: "Accuracy: 0.8532"
- Person_reID: "Rank@1: 88.5%"

**配置文件的`performance_metrics.log_patterns`使用正则表达式定义如何从日志中提取指标。**

## 关于默认值（default字段）

### ✅ 默认值与原仓库完全一致

配置文件中的`default`值已经与各仓库train.sh中的默认值核对一致：

```json
{
  "MRT-OAST": {
    "supported_hyperparams": {
      "epochs": {"default": 10},        // ✓ train.sh默认10
      "learning_rate": {"default": 0.0001}, // ✓ train.sh默认0.0001
      "dropout": {"default": 0.2},      // ✓ train.sh默认0.2
      "seed": {"default": 1334}         // ✓ train.sh默认1334
    }
  },
  "pytorch_resnet_cifar10": {
    "supported_hyperparams": {
      "epochs": {"default": 200},       // ✓ train.sh默认200
      "learning_rate": {"default": 0.1}, // ✓ train.sh默认0.1
      "weight_decay": {"default": 0.0001}, // ✓ train.sh默认0.0001
      "seed": {"default": null}         // ✓ train.sh默认不设置
    }
  }
}
```

### ⚠️ 重要：默认值不会自动传递

**关键设计决策**：配置文件中的`default`值**不会**被自动传递给train.sh。

#### 工作原理

1. **只传递被变异的参数**
   ```bash
   # 用户命令
   python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate epochs

   # 生成的实际命令（假设变异后epochs=100）
   ./train.sh -n resnet20 -e 100
   # 注意：只传递了 -e 100，没有传递 lr、weight_decay 等
   ```

2. **train.sh使用自己的默认值**
   ```bash
   # train.sh收到的命令
   ./train.sh -n resnet20 -e 100

   # train.sh内部使用的参数
   # epochs=100      （从命令行接收）
   # lr=0.1          （train.sh自己的默认值）
   # weight_decay=0.0001  （train.sh自己的默认值）
   # seed=不设置      （train.sh自己的默认值）
   ```

#### 实际验证

```python
from mutation_runner import MutationRunner
runner = MutationRunner()

# 只变异epochs
mutation = runner.generate_mutations('pytorch_resnet_cifar10', 'resnet20', ['epochs'], 1)[0]
# mutation = {'epochs': 150}

cmd, _ = runner.build_training_command('pytorch_resnet_cifar10', 'resnet20', mutation)
# cmd = [..., '-n', 'resnet20', '-e', '150']
# 注意：只有 -e 150，没有 --lr 或 --wd
```

### 配置文件中default值的实际用途

1. **Scale变异策略的参考点**
   ```python
   # 如果使用scale策略（目前未实现，但预留接口）
   # new_value = default * random.uniform(0.5, 1.5)
   ```

2. **文档记录**
   - 明确该超参数的典型值
   - 帮助理解合理的range范围

3. **验证和校验**
   - 可用于检查range是否合理（default应在range内）

## 如何复现原始训练？

### 方法1：直接使用各仓库的train.sh（推荐）

```bash
# 复现pytorch_resnet_cifar10的原始训练
cd repos/pytorch_resnet_cifar10
./train.sh 2>&1 | tee training.log

# 复现VulBERTa的原始训练
cd repos/VulBERTa
./train.sh -n mlp 2>&1 | tee training.log
```

这样���使用train.sh内部定义的所有默认值，完全复现原始训练。

### 方法2：不使用mutation.py

mutation.py的设计目的是**变异**超参数，不是用来复现原始训练的。

## 变异策略示例

### 示例1：只变异epochs

```bash
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate epochs --runs 5
```

**效果**：
- epochs会在[50, 200]范围内随机选择不同的值
- learning_rate、weight_decay、seed使用train.sh的默认值（0.1, 0.0001, 不设置）

### 示例2：变异epochs和learning_rate

```bash
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate epochs,learning_rate --runs 5
```

**效果**：
- epochs在[50, 200]范围内随机变异
- learning_rate在[0.01, 0.2]范围内随机变异
- weight_decay、seed使用train.sh的默认值

### 示例3：变异所有支持的参数

```bash
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate all --runs 5
```

**效果**：
- epochs、learning_rate、seed、weight_decay都会被随机变异
- 每次运行都会产生完全不同的超参数组合

## 配置文件结构说明

```json
{
  "models": {
    "repository_name": {
      "path": "repos/repository_name",           // 仓库路径
      "train_script": "./train.sh",              // 训练脚本名称
      "models": ["model1", "model2"],            // 支持的模型列表

      "supported_hyperparams": {
        "parameter_name": {
          "flag": "--param",                     // train.sh接受的参数名
          "type": "int|float",                   // 参数类型
          "default": value,                      // 默认值���与train.sh一致）
          "range": [min, max]                    // 变异的合理范围
        }
      },

      "model_flag": "-n",                        // 指定模型的参数标志

      "required_args": {                         // 必需的额外参数
        "arg_name": "value"
      },

      "performance_metrics": {                   // 性能指标提取规则
        "log_patterns": {
          "metric_name": "regex_pattern"
        }
      }
    }
  }
}
```

## 添加新仓库

如果要添加新的模型仓库，需要：

1. **确认train.sh的参数格式**
   ```bash
   ./train.sh --help
   ```

2. **确认默认值**
   - 查看train.sh源码中的默认值定义
   - 或运行不带参数的train.sh查看输出

3. **确认合理的参数范围**
   - 基于模型类型和数据集特点
   - 参考原始论文或仓库文档

4. **定义性能指标提取规则**
   - 运行一次训练，查看日志格式
   - 编写正则表达式提取关键指标

5. **添加到config/models_config.json**
   ```json
   {
     "new_repo": {
       "path": "repos/new_repo",
       "train_script": "./train.sh",
       "models": ["model_name"],
       "supported_hyperparams": {
         "epochs": {
           "flag": "--epochs",
           "type": "int",
           "default": 20,
           "range": [10, 50]
         }
       }
     }
   }
   ```

6. **验证配置**
   ```bash
   python3 mutation.py --list
   ```

## 总结

- ✅ **配置文件的default值与原仓库一致** - 用于文档记录和变异策略
- ✅ **只传递被变异的参数** - 未变异的参数由train.sh使用自己的默认值
- ✅ **复现原始训练请直接使用train.sh** - mutation.py用于变异实验
- ✅ **配置文件统一了6个仓库的差异** - 提供统一的接口进行超参数变异
