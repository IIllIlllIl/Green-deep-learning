# mutation.py 参数缩写手册

本文档提供 `mutation.py` 命令行参数的缩写版本，便于快速使用。

---

## 📋 完整参数对照表

| 完整参数名 | 缩写 | 类型 | 默认值 | 说明 |
|-----------|-----|------|-------|------|
| `--experiment-config` | `-ec` | str | - | 实验配置文件路径 |
| `--repo` | `-r` | str | - | 仓库名称 |
| `--model` | `-m` | str | - | 模型名称 |
| `--mutate` | `-mt` | str | - | 要变异的超参数列表 |
| `--runs` | `-n` | int | 1 | 变异运行次数 |
| `--governor` | `-g` | str | - | CPU调度器模式 |
| `--max-retries` | `-mr` | int | 2 | 最大重试次数 |
| `--list` | `-l` | flag | - | 列出可用模型 |
| `--config` | `-c` | str | config/models_config.json | 模型配置文件路径 |
| `--seed` | `-s` | int | None | 随机种子 |

---

## 🎯 缩写参数详解

### 1. `-ec, --experiment-config` - 实验配置文件

**用途**: 指定预定义的实验配置文件路径

**示例**:
```bash
# 完整写法
sudo python3 mutation.py --experiment-config settings/all.json

# 缩写
sudo python3 mutation.py -ec settings/all.json
```

**常用配置文件**:
- `settings/default.json` - 复现原始训练（基线）
- `settings/all.json` - 全面变异所有模型
- `settings/learning_rate_study.json` - 学习率影响研究

---

### 2. `-r, --repo` - 仓库名称

**用途**: 指定模型所属的仓库

**示例**:
```bash
# 完整写法
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate epochs

# 缩写
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs
```

**可用仓库**:
- `MRT-OAST`
- `bug-localization-by-dnn-and-rvsm`
- `pytorch_resnet_cifar10`
- `VulBERTa`
- `Person_reID_baseline_pytorch`
- `examples`

---

### 3. `-m, --model` - 模型名称

**用途**: 指定要训练的具体模型

**示例**:
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all
```

**常用模型**:
- ResNet家族: `resnet20`, `resnet32`, `resnet44`, `resnet56`
- VulBERTa: `mlp`, `cnn`
- Person ReID: `densenet121`, `hrnet18`, `pcb`

---

### 4. `-mt, --mutate` - 变异参数

**用途**: 指定要变异的超参数（逗号分隔）

**示例**:
```bash
# 完整写法
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                    --mutate epochs,learning_rate,seed

# 缩写
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 \
                    -mt epochs,learning_rate,seed

# 变异所有支持的超参数
python3 mutation.py -r VulBERTa -m mlp -mt all
```

**可用超参数**:
- `epochs` - 训练轮数
- `learning_rate` - 学习率
- `seed` - 随机种子
- `dropout` - Dropout概率
- `weight_decay` - 权重衰减
- `all` - 所有支持的超参数

---

### 5. `-n, --runs` - 运行次数

**用途**: 指定生成多少个不同的超参数变异组合

**示例**:
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all --runs 5

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all -n 5
```

**注意**:
- 默认值为 1
- 框架会自动确保生成的变异组合不重复

---

### 6. `-g, --governor` - CPU调度器模式

**用途**: 设置CPU频率调度器以减少实验干扰

**示例**:
```bash
# 完整写法
sudo python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                         --mutate epochs --governor performance

# 缩写
sudo python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 \
                         -mt epochs -g performance
```

**可选值**:
- `performance` - 高性能模式（推荐用于能耗实验）
- `powersave` - 省电模式
- `ondemand` - 按需调频
- `conservative` - 保守调频

**最佳实践**:
- 能耗实验建议使用 `performance` 模式以减少CPU频率波动带来的干扰
- 需要 `sudo` 权限

---

### 7. `-mr, --max-retries` - 最大重试次数

**用途**: 训练失败时自动重试的最大次数

**示例**:
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all --max-retries 3

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all -mr 3
```

**注意**:
- 默认值为 2
- 重试间隔为 30 秒（`RETRY_SLEEP_SECONDS`）

---

### 8. `-l, --list` - 列出可用模型

**用途**: 显示所有可用的仓库、模型和支持的超参数

**示例**:
```bash
# 完整写法
python3 mutation.py --list

# 缩写
python3 mutation.py -l
```

**输出示例**:
```
📋 Available Repositories and Models:

  pytorch_resnet_cifar10:
    Models: resnet20, resnet32, resnet44, resnet56
    Supported hyperparameters: epochs, learning_rate, seed, weight_decay

  VulBERTa:
    Models: mlp, cnn
    Supported hyperparameters: epochs, learning_rate, seed, weight_decay
```

---

### 9. `-c, --config` - 模型配置文件

**用途**: 指定自定义的模型配置文件路径

**示例**:
```bash
# 完整写法
python3 mutation.py --config my_config/custom_models.json --list

# 缩写
python3 mutation.py -c my_config/custom_models.json -l
```

**注意**:
- 默认使用 `config/models_config.json`
- 一般情况下不需要修改

---

### 10. `-s, --seed` - 随机种子

**用途**: 设置随机种子以确保实验可复现

**示例**:
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all \
                    --runs 5 --seed 42

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all -n 5 -s 42
```

**注意**:
- 默认为 `None`（使用系统时间）
- 设置后可以确保每次运行生成相同的变异组合

---

## 🚀 快速使用示例

### 示例 1: 基础单次变异（使用缩写）

```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate
```

等价于:
```bash
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                    --mutate epochs,learning_rate
```

---

### 示例 2: 多次变异 + Performance模式（使用缩写）

```bash
sudo python3 mutation.py -r VulBERTa -m mlp -mt all -n 5 -g performance
```

等价于:
```bash
sudo python3 mutation.py --repo VulBERTa --model mlp \
                         --mutate all --runs 5 --governor performance
```

---

### 示例 3: 使用配置文件（使用缩写）

```bash
sudo python3 mutation.py -ec settings/default.json
```

等价于:
```bash
sudo python3 mutation.py --experiment-config settings/default.json
```

---

### 示例 4: 可复现实验（使用缩写）

```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 \
                    -mt all -n 10 -s 42 -mr 3
```

等价于:
```bash
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                    --mutate all --runs 10 --seed 42 --max-retries 3
```

---

## 📊 参数组合建议

### 1. 快速测试
```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs -n 1
```

### 2. 学习率研究
```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt learning_rate -n 10 -s 42
```

### 3. 全面变异实验
```bash
sudo python3 mutation.py -r VulBERTa -m mlp -mt all -n 20 -g performance -mr 3
```

### 4. 批量实验（配置文件）
```bash
# 推荐：使用预设配置文件
sudo python3 mutation.py -ec settings/all.json
```

---

## ⚠️ 注意事项

1. **Governor参数需要sudo权限**
   ```bash
   sudo python3 mutation.py -r ... -g performance
   ```

2. **缩写与完整参数不能混用同一参数**
   ```bash
   # ❌ 错误：同时使用缩写和完整参数
   python3 mutation.py -r pytorch_resnet_cifar10 --repo VulBERTa

   # ✅ 正确：统一使用缩写或完整参数
   python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20
   ```

3. **命令行模式需要必需参数**
   - 必需: `-r/--repo`, `-m/--model`, `-mt/--mutate`
   - 配置文件模式只需: `-ec/--experiment-config`

4. **逗号分隔的参数不要加空格**
   ```bash
   # ✅ 正确
   -mt epochs,learning_rate,seed

   # ❌ 错误
   -mt epochs, learning_rate, seed
   ```

---

## 📝 更新日志

- **2025-11-09**: 初始版本，包含所有10个参数的缩写定义
- 创建者: Green
- 项目: 深度学习模型训练能耗研究

---

## 🔗 相关文档

- [README.md](../README.md) - 项目主文档
- [settings/README.md](../settings/README.md) - 实验配置文件说明
- [config/models_config.json](../config/models_config.json) - 模型配置文件

