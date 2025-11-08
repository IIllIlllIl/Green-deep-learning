# Usage Examples - Mutation Runner

本文档提供详细的使用示例和实际应用场景。

## 基础示例

### 1. 列出所有可用模型

```bash
python3 mutation.py --list
```

### 2. 最简单的单次实验

```bash
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate epochs \
    --runs 1
```

### 3. 变异多个超参数

```bash
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate epochs,learning_rate,seed \
    --runs 3
```

### 4. 变异所有支持的超参数

```bash
python3 mutation.py \
    --repo VulBERTa \
    --model mlp \
    --mutate all \
    --runs 5
```

## 高级示例

### 5. 使用性能模式（减少干扰）

```bash
sudo python3 mutation.py \
    --repo Person_reID_baseline_pytorch \
    --model densenet121 \
    --mutate epochs,learning_rate \
    --governor performance \
    --runs 5
```

### 6. 增加重试次数

```bash
python3 mutation.py \
    --repo examples \
    --model mnist_cnn \
    --mutate all \
    --runs 10 \
    --max-retries 3
```

### 7. 使用自定义配置文件

```bash
python3 mutation.py \
    --repo test_repo \
    --model model_a \
    --mutate all \
    --config test/test_config.json \
    --runs 1
```

## 研究场景示例

### 场景1: 研究学习率对能耗的影响

**目标**: 观察不同学习率下模型训练的能耗变化

```bash
# 运行10次，每次随机变异learning_rate
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate learning_rate \
    --runs 10 \
    --governor performance

# 分析结果
cd results
for f in *.json; do
    echo "$(basename $f):"
    jq '{lr: .hyperparameters.learning_rate, energy: .energy_metrics.cpu_energy_total_joules}' $f
done
```

### 场景2: 对比不同Dropout率的性能

**目标**: 研究dropout对模型准确率和能耗的影响

```bash
python3 mutation.py \
    --repo Person_reID_baseline_pytorch \
    --model densenet121 \
    --mutate dropout \
    --runs 15 \
    --governor performance

# 提取结果到CSV
echo "dropout,accuracy,energy" > dropout_analysis.csv
for f in results/*.json; do
    jq -r '[.hyperparameters.dropout, .performance_metrics.rank1, .energy_metrics.cpu_energy_total_joules] | @csv' $f >> dropout_analysis.csv
done
```

### 场景3: Seed稳定性研究

**目标**: 研究随机种子对训练结果的影响

```bash
python3 mutation.py \
    --repo VulBERTa \
    --model mlp \
    --mutate seed \
    --runs 20 \
    --governor performance

# 分析种子对准确率的影响
cat results/*.json | jq -r '[.hyperparameters.seed, .performance_metrics.accuracy] | @csv' | sort -t, -k2 -n
```

### 场景4: 全面超参数探索

**目标**: 系统性探索所有超参数组合的影响

```bash
# 运行大规模实验
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate all \
    --runs 50 \
    --governor performance \
    --max-retries 2

# 生成统计报告
python3 << 'EOF'
import json
import glob
from statistics import mean, stdev

results = []
for f in glob.glob('results/*.json'):
    with open(f) as file:
        results.append(json.load(file))

# 能耗统计
energies = [r['energy_metrics']['cpu_energy_total_joules'] for r in results if r['training_success']]
print(f"CPU Energy: mean={mean(energies):.2f}J, std={stdev(energies):.2f}J")

# 性能统计
accuracies = [r['performance_metrics'].get('test_accuracy', 0) for r in results if r['training_success']]
print(f"Accuracy: mean={mean(accuracies):.2f}%, std={stdev(accuracies):.2f}%")

# 训练时长统计
durations = [r['duration_seconds'] for r in results if r['training_success']]
print(f"Duration: mean={mean(durations):.1f}s, std={stdev(durations):.1f}s")
EOF
```

## 批量实验脚本

### 对多个模型运行相同实验

```bash
#!/bin/bash
# batch_experiments.sh

REPO="pytorch_resnet_cifar10"
MODELS=("resnet20" "resnet32" "resnet44")
MUTATE_PARAMS="epochs,learning_rate,weight_decay"
RUNS=5

for model in "${MODELS[@]}"; do
    echo "========================================="
    echo "Running experiments for $model"
    echo "========================================="

    sudo python3 mutation.py \
        --repo $REPO \
        --model $model \
        --mutate $MUTATE_PARAMS \
        --runs $RUNS \
        --governor performance

    echo "Completed $model, sleeping 5 minutes..."
    sleep 300
done

echo "All experiments completed!"
```

### 对多个仓库运行实验

```bash
#!/bin/bash
# multi_repo_experiments.sh

# 定义实验配置
declare -A experiments=(
    ["pytorch_resnet_cifar10:resnet20"]="epochs,learning_rate"
    ["VulBERTa:mlp"]="all"
    ["Person_reID_baseline_pytorch:densenet121"]="epochs,dropout"
    ["examples:mnist_cnn"]="learning_rate,seed"
)

RUNS=3

for config in "${!experiments[@]}"; do
    IFS=':' read -r repo model <<< "$config"
    params="${experiments[$config]}"

    echo "========================================="
    echo "Experiment: $repo / $model"
    echo "Parameters: $params"
    echo "========================================="

    sudo python3 mutation.py \
        --repo "$repo" \
        --model "$model" \
        --mutate "$params" \
        --runs $RUNS \
        --governor performance

    echo "Sleeping 10 minutes..."
    sleep 600
done
```

## 结果分析示例

### 使用jq提取关键指标

```bash
# 1. 查看所有实验的成功率
cat results/*.json | jq -r '[.experiment_id, .training_success] | @csv'

# 2. 提取能耗信息
cat results/*.json | jq '.energy_metrics | {cpu: .cpu_energy_total_joules, gpu: .gpu_energy_total_joules}'

# 3. 查找能耗最低的实验
cat results/*.json | jq -s 'sort_by(.energy_metrics.cpu_energy_total_joules) | .[0] | {id: .experiment_id, energy: .energy_metrics.cpu_energy_total_joules, hyperparams: .hyperparameters}'

# 4. 查找准确率最高的实验
cat results/*.json | jq -s 'sort_by(.performance_metrics.accuracy) | reverse | .[0] | {id: .experiment_id, accuracy: .performance_metrics.accuracy, hyperparams: .hyperparameters}'
```

### Python分析脚本示例

```python
#!/usr/bin/env python3
"""
analyze_results.py - 分析mutation实验结果
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 加载所有结果
results = []
for f in glob.glob('results/*.json'):
    with open(f) as file:
        results.append(json.load(file))

# 转换为DataFrame
data = []
for r in results:
    if not r['training_success']:
        continue

    row = {
        'experiment_id': r['experiment_id'],
        'repository': r['repository'],
        'model': r['model'],
        'duration': r['duration_seconds'],
        'cpu_energy': r['energy_metrics']['cpu_energy_total_joules'],
        'gpu_energy': r['energy_metrics'].get('gpu_energy_total_joules', 0),
    }

    # 添加超参数
    row.update(r['hyperparameters'])

    # 添加性能指标
    row.update(r['performance_metrics'])

    data.append(row)

df = pd.DataFrame(data)

# 保存到CSV
df.to_csv('experiment_results.csv', index=False)
print(f"Saved {len(df)} experiments to experiment_results.csv")

# 统计信息
print("\n=== Summary Statistics ===")
print(df[['duration', 'cpu_energy', 'gpu_energy']].describe())

# 相关性分析
if 'epochs' in df.columns and 'cpu_energy' in df.columns:
    correlation = df[['epochs', 'learning_rate', 'cpu_energy']].corr()
    print("\n=== Correlation Matrix ===")
    print(correlation)

# 绘图
if 'learning_rate' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['learning_rate'], df['cpu_energy'])
    plt.xlabel('Learning Rate')
    plt.ylabel('CPU Energy (Joules)')
    plt.title('Learning Rate vs CPU Energy')
    plt.savefig('lr_vs_energy.png')
    print("\nPlot saved to lr_vs_energy.png")
```

## 实时监控

### 监控运行中的实验

```bash
# 在一个终端运行实验
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate all --runs 10 &

# 在另一个终端监控进度
watch -n 5 'ls results/*.json | wc -l && tail -20 results/*.log | grep -i "epoch\|accuracy\|loss"'
```

### 监控能耗

```bash
# 实时监控GPU功耗
watch -n 1 nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu --format=csv

# 监控CPU频率
watch -n 1 "grep MHz /proc/cpuinfo | head -8"
```

## 清理和维护

### 清理旧结果

```bash
# 备份结果
mkdir -p results_backup/$(date +%Y%m%d)
cp results/*.json results_backup/$(date +%Y%m%d)/

# 清理旧结果
rm -f results/*.json
rm -rf results/energy_*
```

### 归档实验

```bash
# 创建实验归档
tar -czf experiment_$(date +%Y%m%d_%H%M%S).tar.gz results/

# 清理
rm -rf results/*
```

## 故障恢复

### 恢复中断的实验

如果实验被中断，可以检查results目录：

```bash
# 查看哪些实验成功
cat results/*.json | jq -r 'select(.training_success == true) | .experiment_id'

# 查看失败的实验
cat results/*.json | jq -r 'select(.training_success == false) | {id: .experiment_id, error: .error_message}'

# 统计成功/失败
echo "Success: $(cat results/*.json | jq -r 'select(.training_success == true)' | grep experiment_id | wc -l)"
echo "Failed: $(cat results/*.json | jq -r 'select(.training_success == false)' | grep experiment_id | wc -l)"
```

## 性能优化

### 减少实验时间

1. **减少epochs**：在配置文件中调整range
2. **使用小批量测试**：先用`--runs 1`测试
3. **并行运行**：在不同GPU上同时运行多个实验

### 减少能耗干扰

1. **使用performance governor**：`--governor performance`
2. **关闭不必要的服务**
3. **增加休眠时间**：修改mutation.py中的sleep时间

## 帮助和支持

如有问题，请：
1. 查看 `python3 mutation.py --help`
2. 阅读 [README.md](../README.md)
3. 检查 [test/README.md](../test/README.md) 中的测试用例
