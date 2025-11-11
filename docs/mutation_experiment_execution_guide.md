# 变异实验执行指南

**配置文件**: `settings/mutation_experiment_elite_plus.json`
**方案**: 方案一（精英模型）+ MNIST基线
**日期**: 2025-11-10

---

## 实验配置概览

### 选定模型（4个）

| 模型 | 变异参数 | 性能 | 单次时长 | 研究价值 |
|------|---------|------|---------|---------|
| **pytorch_resnet_cifar10/resnet20** | epochs, lr, wd | 91.45% Acc | 20分钟 | ⭐⭐⭐ CV分类 |
| **Person_reID_baseline_pytorch/densenet121** | epochs, lr, dropout | 90.11% Rank@1 | 38分钟 | ⭐⭐⭐ 行人重识别 |
| **MRT-OAST/default** | epochs, lr, wd, dropout | 90.10% Acc | 42分钟 | ⭐⭐⭐ 文本分类 |
| **examples/mnist** | epochs, lr | 99.0% Acc | 1.5分钟 | ⭐⭐ 快速基线 |

### 变异策略

所有变异自动使用配置文件中定义的分布策略：

- **Epochs**: 对数均匀分布 [default×0.5, default×2.0]
- **Learning Rate**: 对数均匀分布 [default×0.1, default×10.0]
- **Weight Decay**: 30%零值 + 70%对数均匀 [1e-5, 0.01]
- **Dropout**: 均匀分布 [0.0, 0.7]

---

## 执行步骤

### 步骤1: 验证环境

```bash
# 确保在正确的目录
cd /home/green/energy_dl/nightly

# 检查配置文件
cat settings/mutation_experiment_elite_plus.json

# 验证所有模型仓库存在
ls -d repos/pytorch_resnet_cifar10 repos/Person_reID_baseline_pytorch repos/MRT-OAST repos/examples
```

### 步骤2: 快速测试（推荐）

在正式运行前，先用少量变异测试是否正常工作：

```bash
# 测试1: MNIST快速验证（1.5分钟）
python3 mutation.py -r examples -m mnist \
                    -mt epochs,learning_rate \
                    -n 3 -g performance

# 检查结果
ls -lh results/examples/mnist/mutation_*
```

**预期输出**:
- 生成3个唯一的变异配置
- 每个变异训练约1.5分钟
- 总耗时约5分钟
- 能耗数据正常记录

### 步骤3: 正式执行实验

#### 方式A: 直接运行（前台）

```bash
python3 mutation.py -ec settings/mutation_experiment_elite_plus.json
```

⚠️ **注意**: 此方式会占用终端，需保持SSH连接

#### 方式B: Screen后台运行（推荐）

```bash
# 创建新的screen会话
screen -S mutation_elite

# 在screen中执行
python3 mutation.py -ec settings/mutation_experiment_elite_plus.json

# 分离screen: Ctrl+A 然后按 D
```

**恢复查看**:
```bash
# 重新连接到screen
screen -r mutation_elite

# 列出所有screen会话
screen -ls
```

#### 方式C: nohup后台运行

```bash
# 后台运行并记录日志
nohup python3 mutation.py -ec settings/mutation_experiment_elite_plus.json \
      > logs/mutation_elite_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 记录进程ID
echo $! > mutation_elite.pid

# 查看日志
tail -f logs/mutation_elite_*.log

# 停止进程（如需要）
kill $(cat mutation_elite.pid)
```

### 步骤4: 监控进度

#### 实时监控日志
```bash
# 如果使用screen
screen -r mutation_elite

# 如果使用nohup
tail -f logs/mutation_elite_*.log
```

#### 检查结果目录
```bash
# 查看已完成的实验
find results/ -name "*.json" -mmin -60 | wc -l  # 最近1小时的结果数

# 查看每个模型的进度
ls -lh results/pytorch_resnet_cifar10/resnet20/mutation_*
ls -lh results/Person_reID_baseline_pytorch/densenet121/mutation_*
ls -lh results/MRT-OAST/default/mutation_*
ls -lh results/examples/mnist/mutation_*
```

#### 监控系统资源
```bash
# CPU和GPU使用率
watch -n 5 nvidia-smi

# 查看Python进程
ps aux | grep mutation.py
```

---

## 预期结果

### 时间估算

基于 `runs_per_config: 1`（每个模型1个变异）:

| 模型 | 单次时长 | 变异数 | 总时长 |
|------|---------|-------|--------|
| ResNet20 | 20分钟 | 1 | 20分钟 |
| DenseNet121 | 38分钟 | 1 | 38分钟 |
| MRT-OAST | 42分钟 | 1 | 42分钟 |
| MNIST | 1.5分钟 | 1 | 1.5分钟 |
| **总计** | - | **4** | **~101.5分钟 (1.7小时)** |

⚠️ **注意**: 当前配置 `runs_per_config: 1` 仅生成1个变异点作为快速验证。

### 如需增加变异数量

编辑配置文件修改 `runs_per_config`:

```bash
# 编辑配置
nano settings/mutation_experiment_elite_plus.json

# 修改这一行（例如改为20）:
"runs_per_config": 20,
```

**时间估算（20个变异）**:
- ResNet20: 20 × 20分钟 = 6.7小时
- DenseNet121: 20 × 38分钟 = 12.7小时
- MRT-OAST: 20 × 42分钟 = 14小时
- MNIST: 20 × 1.5分钟 = 0.5小时
- **总计**: ~34小时 (1.4天)

---

## 结果分析

### 步骤1: 查看原始数据

```bash
# 查看所有JSON结果
find results/ -name "mutation_*.json" -type f

# 查看特定模型的结果
cat results/examples/mnist/mutation_000/results.json | jq '.'
```

### 步骤2: 提取关键指标

创建简单的分析脚本 `analyze_mutations.py`:

```python
#!/usr/bin/env python3
import json
from pathlib import Path

results_dir = Path("results")

for model_dir in results_dir.glob("*/*/mutation_*"):
    results_file = model_dir / "results.json"
    if results_file.exists():
        data = json.load(open(results_file))

        print(f"\n{model_dir.parent.parent.name}/{model_dir.parent.name}:")
        print(f"  Config: {data.get('hyperparameters', {})}")
        print(f"  Performance: {data.get('performance_metrics', {})}")
        print(f"  Energy: {data.get('energy_consumption', {})} J")
        print(f"  Time: {data.get('training_time', {})} s")
```

运行分析:
```bash
python3 analyze_mutations.py
```

### 步骤3: 可视化（可选）

如果需要绘制能耗-性能图:

```python
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 收集数据
energies = []
accuracies = []

for results_file in Path("results").glob("*/*/mutation_*/results.json"):
    data = json.load(open(results_file))
    energy = data.get("energy_consumption", {}).get("total_joules", 0)
    acc = data.get("performance_metrics", {}).get("test_accuracy", 0)

    if energy > 0 and acc > 0:
        energies.append(energy)
        accuracies.append(acc)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(energies, accuracies, alpha=0.6)
plt.xlabel("Energy Consumption (J)")
plt.ylabel("Test Accuracy (%)")
plt.title("Energy vs Performance - Mutation Experiments")
plt.grid(True)
plt.savefig("energy_vs_performance.png")
print("Plot saved to energy_vs_performance.png")
```

---

## 故障排除

### 问题1: 配置文件语法错误

```bash
# 验证JSON格式
python3 -m json.tool settings/mutation_experiment_elite_plus.json
```

### 问题2: 模型训练失败

查看具体错误:
```bash
# 查看最新的训练日志
ls -lt results/*/*/mutation_*/train.log | head -1 | awk '{print $NF}' | xargs cat
```

### 问题3: 内存不足

```bash
# 检查内存使用
free -h

# 减少并行度或添加内存限制
```

### 问题4: GPU不可用

```bash
# 检查GPU状态
nvidia-smi

# 如果需要指定GPU
CUDA_VISIBLE_DEVICES=0 python3 mutation.py -ec settings/mutation_experiment_elite_plus.json
```

---

## 实验检查清单

实验开始前：
- [ ] 确认在 `/home/green/energy_dl/nightly` 目录
- [ ] 配置文件存在并格式正确
- [ ] 所有模型仓库已克隆并配置
- [ ] GPU可用 (`nvidia-smi` 正常)
- [ ] 磁盘空间充足 (`df -h`)
- [ ] 已创建日志目录 (`mkdir -p logs`)

实验进行中：
- [ ] 定期检查进程是否运行 (`ps aux | grep mutation`)
- [ ] 监控结果文件生成 (`watch ls -lh results/`)
- [ ] 检查错误日志 (`grep -i error logs/*.log`)

实验完成后：
- [ ] 验证所有变异都已完成
- [ ] 检查结果文件完整性
- [ ] 备份重要数据
- [ ] 分析能耗-性能关系

---

## 快速命令参考

```bash
# 启动实验（Screen方式 - 推荐）
screen -S mutation_elite
python3 mutation.py -ec settings/mutation_experiment_elite_plus.json
# Ctrl+A D (分离)

# 查看进度
screen -r mutation_elite

# 监控结果
watch -n 10 'find results/ -name "*.json" -mmin -60 | wc -l'

# 查看最新日志
tail -f logs/mutation_elite_*.log

# 停止实验（如需要）
screen -r mutation_elite
# Ctrl+C

# 清理测试数据（谨慎！）
rm -rf results/*/*/mutation_*
```

---

## 下一步

实验完成后，您可以：

1. **分析结果**: 使用上述分析脚本提取关键指标
2. **绘制图表**: 能耗-性能关系、帕累托前沿
3. **调整配置**: 基于初步结果优化变异范围
4. **扩展实验**: 尝试方案二或方案三
5. **撰写报告**: 总结超参数-能耗关系发现

---

**文档创建**: 2025-11-10
**配置文件**: settings/mutation_experiment_elite_plus.json
**预计总时长**: 1.7小时（1个变异/模型）或 34小时（20个变异/模型）
