# mutation.py 快速参考卡片

## 📋 参数缩写速查表

| 缩写 | 完整参数 | 说明 |
|-----|---------|-----|
| `-ec` | `--experiment-config` | 实验配置文件 |
| `-r` | `--repo` | 仓库名称 |
| `-m` | `--model` | 模型名称 |
| `-mt` | `--mutate` | 变异参数列表 |
| `-n` | `--runs` | 运行次数 |
| `-g` | `--governor` | CPU调度器 |
| `-mr` | `--max-retries` | 最大重试次数 |
| `-l` | `--list` | 列出可用模型 |
| `-c` | `--config` | 模型配置文件 |
| `-s` | `--seed` | 随机种子 |

---

## 🚀 常用命令速查

### 1. 列出可用模型
```bash
python3 mutation.py -l
```

### 2. 单次变异实验
```bash
# 完整写法
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                    --mutate epochs,learning_rate

# 缩写
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate
```

### 3. 多次变异实验
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all --runs 5

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all -n 5
```

### 4. Performance模式
```bash
# 完整写法
sudo python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 \
                         --mutate all --governor performance

# 缩写
sudo python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt all -g performance
```

### 5. 配置文件模式（推荐）
```bash
# 完整写法
sudo python3 mutation.py --experiment-config settings/default.json

# 缩写
sudo python3 mutation.py -ec settings/default.json
```

### 6. 可复现实验
```bash
# 完整写法
python3 mutation.py --repo VulBERTa --model mlp --mutate all \
                    --runs 10 --seed 42

# 缩写
python3 mutation.py -r VulBERTa -m mlp -mt all -n 10 -s 42
```

---

## 🎯 超参数速查

### 可用超参数
- `epochs` - 训练轮数
- `learning_rate` - 学习率
- `seed` - 随机种子
- `dropout` - Dropout概率
- `weight_decay` - 权重衰减
- `all` - 所有支持的超参数

### 超参数组合示例
```bash
# 单个参数
-mt epochs

# 多个参数（逗号分隔，无空格）
-mt epochs,learning_rate

# 三个参数
-mt epochs,learning_rate,seed

# 所有支持的参数
-mt all
```

---

## 🏃 仓库和模型速查

### pytorch_resnet_cifar10
```bash
-r pytorch_resnet_cifar10 -m resnet20
-r pytorch_resnet_cifar10 -m resnet32
-r pytorch_resnet_cifar10 -m resnet44
-r pytorch_resnet_cifar10 -m resnet56
```

### VulBERTa
```bash
-r VulBERTa -m mlp
-r VulBERTa -m cnn
```

### Person_reID_baseline_pytorch
```bash
-r Person_reID_baseline_pytorch -m densenet121
-r Person_reID_baseline_pytorch -m hrnet18
-r Person_reID_baseline_pytorch -m pcb
```

### MRT-OAST
```bash
-r MRT-OAST -m default
```

### bug-localization-by-dnn-and-rvsm
```bash
-r bug-localization-by-dnn-and-rvsm -m default
```

### examples
```bash
-r examples -m mnist_cnn
-r examples -m mnist_rnn
-r examples -m mnist_forward_forward
-r examples -m siamese
```

---

## ⚙️ Governor 模式速查

```bash
-g performance    # 高性能（推荐用于能耗实验）
-g powersave      # 省电模式
-g ondemand       # 按需调频
-g conservative   # 保守调频
```

**注意**: 使用 `-g` 参数需要 `sudo` 权限

---

## 📄 预设配置文件速查

```bash
-ec settings/default.json              # 复现原始训练（基线）
-ec settings/all.json                  # 变异所有模型
-ec settings/learning_rate_study.json  # 学习率研究
-ec settings/resnet_all_models.json    # ResNet家族实验
```

---

## ⚡ 超级快捷命令

### 最小命令（列表）
```bash
python3 mutation.py -l
```

### 最小命令（单次实验）
```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs
```

### 最小命令（配置文件）
```bash
sudo python3 mutation.py -ec settings/default.json
```

### 推荐命令（完整实验）
```bash
sudo python3 mutation.py -r VulBERTa -m mlp -mt all -n 5 -g performance -s 42
```

---

## 💡 使用技巧

1. **混合完整和缩写参数**
   ```bash
   # ✅ 可以混用
   python3 mutation.py -r pytorch_resnet_cifar10 --model resnet20 -mt all
   ```

2. **优先使用配置文件模式**
   ```bash
   # 批量实验推荐使用配置文件
   sudo python3 mutation.py -ec settings/all.json
   ```

3. **Performance模式减少干扰**
   ```bash
   # 能耗实验建议使用performance模式
   sudo python3 mutation.py -r ... -m ... -mt ... -g performance
   ```

4. **设置随机种子确保可复现**
   ```bash
   # 添加 -s 42 确保实验可复现
   python3 mutation.py -r ... -m ... -mt ... -s 42
   ```

---

## 📊 输出目录速查

```bash
results/                          # 实验结果目录
├── <experiment_id>.json          # 实验结果JSON文件
└── energy_<experiment_id>/       # 能耗监控数据
    ├── cpu_energy.txt            # CPU能耗总结
    ├── cpu_energy_raw.txt        # perf原始输出
    ├── gpu_power.csv             # GPU功耗时间序列
    ├── gpu_temperature.csv       # GPU温度时间序列
    └── gpu_utilization.csv       # GPU利用率时间序列
```

---

**提示**: 详细文档请参考 [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md)
