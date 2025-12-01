# Mutation 2x Supplement - 单参数独立变异配置

**配置文件**: `settings/mutation_2x_supplement.json`
**创建日期**: 2025-11-27
**版本**: 2.1 (单参数独立变异 + 去重优化)
**最后修正**: 2025-11-26 - 修正4个参数确保5个不同值

---

## 🎯 核心特性：单参数独立变异

### 什么是单参数独立变异？

每次实验**只变异一个超参数**，其他参数保持默认值。

**示例**：对于有4个超参数的模型（epochs, learning_rate, seed, dropout）

| 实验类型 | epochs | learning_rate | seed | dropout | 说明 |
|---------|--------|---------------|------|---------|------|
| 默认值 | 默认 | 默认 | 默认 | 默认 | 所有参数使用默认值 |
| epochs变异1 | **变异值1** | 默认 | 默认 | 默认 | 只变epochs |
| epochs变异2 | **变异值2** | 默认 | 默认 | 默认 | 只变epochs |
| learning_rate变异1 | 默认 | **变异值1** | 默认 | 默认 | 只变learning_rate |
| learning_rate变异2 | 默认 | **变异值2** | 默认 | 默认 | 只变learning_rate |
| ... | ... | ... | ... | ... | ... |

这种方式确保**每个超参数的影响可以独立分析**。

---

## ⚠️ 重要修正（2025-11-26）

### 问题发现

通过分析 `summary_all.csv`，发现部分超参数的不同值数量不足5个：

1. **bug-localization,default - kfold**: 当前只有1个值(10)，需要补充4个
2. **examples,mnist - batch_size**: 当前只有1个值(32)，需要补充4个
3. **examples,mnist_rnn - batch_size**: 当前只有1个值(32)，需要补充4个
4. **examples,siamese - batch_size**: 当前只有1个值(32)，需要补充4个

### 解决方案

将这4个参数的 `num_mutations` 从 **3 增加到 4**，确保每个参数都能达到至少5个不同值。

### 影响

- 总实验数：从 252 增加到 **260** 次
- 运行时间：从 133.8 小时增加到 **135.6 小时**（+1.8小时）
- 完成后总数：从 463 增加到 **471** 次

---

## 📊 实验目标

对11个模型补充实验到：
- **1次默认值训练**（所有参数默认）
- **每个超参数 5次独立变异**（每次只变一个参数）
- **非并行 + 并行各运行1次**（`runs_per_config=2`）

### 计算公式

对于有 N 个超参数的模型：
- 目标 = 1(默认) + N×5(变异) = 1 + 5N 次/模式
- 总计 = (1 + 5N) × 2模式

---

## 📋 11个模型配置详情

| # | Repository | Model | 超参数数 | 配置数 | 预计补充 | 目标总数 |
|---|------------|-------|---------|-------|---------|---------|
| 1 | MRT-OAST | default | 5 | 5配置 | 20次 | 52 |
| 2 | bug-localization | default | 4 | 4配置 | 22次 | 42 |
| 3 | pytorch_resnet_cifar10 | resnet20 | 4 | 4配置 | 16次 | 42 |
| 4 | VulBERTa | mlp | 4 | 4配置 | 16次 | 42 |
| 5 | Person_reID | densenet121 | 4 | 4配置 | 16次 | 42 |
| 6 | Person_reID | hrnet18 | 4 | 4配置 | 32次 | 43 |
| 7 | Person_reID | pcb ⚙️ | 4 | 4配置 | 32次 | 42 |
| 8 | examples | mnist | 4 | 4配置 | 22次 | 42 |
| 9 | examples | mnist_rnn | 4 | 4配置 | 22次 | 42 |
| 10 | examples | siamese | 4 | 4配置 | 22次 | 42 |
| 11 | examples | mnist_ff ⚙️ | 3 * | 4配置 | 32次 | 42 |

⚙️ = 包含特殊配置防止GPU OOM
\* = mnist_ff固定batch_size=10000，不变异该参数

**总计**: 45个配置，预计补充 **260次实验** (修正后)

---

## 🔍 配置详解

### 示例1：MRT-OAST (5个超参数)

```json
{
  "repo": "MRT-OAST",
  "model": "default",
  "num_mutations": 2,
  "mutate": ["epochs"],
  "comment": "epochs的2次变异"
}
```

这个配置会生成：
- **非并行模式**: 2次实验，只变异epochs，其他参数默认
- **并行模式**: 2次实验，只变异epochs，其他参数默认
- **总计**: 4次实验

MRT-OAST的5个配置：
1. epochs: 2次变异 × 2模式 = 4次
2. learning_rate: 2次变异 × 2模式 = 4次
3. seed: 2次变异 × 2模式 = 4次
4. dropout: 2次变异 × 2模式 = 4次
5. weight_decay: 2次变异 × 2模式 = 4次

**总计**: 20次实验

### 示例2：MNIST-FF (特殊配置)

```json
{
  "repo": "examples",
  "model": "mnist_ff",
  "mode": "default",
  "hyperparameters": {
    "batch_size": 10000
  },
  "comment": "默认值训练 (batch_size=10000防止OOM)"
}
```

这个配置会生成：
- **非并行 + 并行**: 各1次默认值训练（batch_size固定为10000）

MNIST-FF的4个配置：
1. 默认值: 1次 × 2模式 = 2次
2. epochs: 5次变异 × 2模式 = 10次
3. learning_rate: 5次变异 × 2模式 = 10次
4. seed: 5次变异 × 2模式 = 10次

**注意**: batch_size固定为10000，不变异（防止OOM）

**总计**: 32次实验

---

## 📝 配置文件结构

### 全局参数

```json
{
  "experiment_name": "mutation_2x_supplement_single_param_20251127",
  "mode": "batch",
  "runs_per_config": 2,              // 每个配置运行2次（非并行+并行）
  "max_retries": 2,
  "governor": "performance",
  "cleanup_gpu_memory": true,
  "cleanup_between_experiments": true,
  "use_deduplication": true,         // 启用去重
  "historical_csvs": [
    "results/summary_all.csv"        // 历史数据
  ]
}
```

### 实验配置类型

#### 类型1：单参数变异（最常见）

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "num_mutations": 2,
  "mutate": ["epochs"],              // 只变异epochs
  "comment": "epochs的2次变异"
}
```

#### 类型2：默认值（仅mnist_ff）

```json
{
  "repo": "examples",
  "model": "mnist_ff",
  "mode": "default",                 // 默认模式
  "hyperparameters": {
    "batch_size": 10000
  },
  "comment": "默认值训练"
}
```

#### 类型3：带固定参数的单参数变异

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "pcb",
  "num_mutations": 4,
  "mutate": ["epochs"],
  "hyperparameters": {
    "batchsize": 8                   // 固定batchsize防止OOM
  },
  "comment": "epochs的4次变异 (batchsize=8防止OOM)"
}
```

---

## 🔧 关键机制

### 1. 单参数变异机制

当配置 `"mutate": ["epochs"]` 时：
- 系统生成 `num_mutations` 个不同的 epochs 值
- 每个变异中，**只有epochs改变**，其他参数使用默认值
- 确保epochs的影响可以独立分析

### 2. 去重机制

```json
"use_deduplication": true,
"historical_csvs": ["results/summary_all.csv"]
```

系统会：
1. 读取 `summary_all.csv` 中的所有历史实验
2. 提取每个实验的超参数组合
3. 生成新变异时，跳过已存在的组合
4. 确保不产生重复实验

### 3. OOM防护

针对容易内存溢出的模型：

**Person_reID_baseline_pytorch pcb**:
```json
"hyperparameters": {
  "batchsize": 8
}
```

**examples mnist_ff**:
```json
"hyperparameters": {
  "batch_size": 10000
}
```

这些参数在所有变异中保持固定，防止GPU OOM。

---

## 🚀 运行方式

### 推荐：Screen后台运行

```bash
# 创建screen会话
screen -S mutation_supplement

# 以sudo运行（保留环境变量）
sudo -E python3 mutation.py -ec settings/mutation_2x_supplement.json

# 分离会话: Ctrl+A, D
# 重新连接: screen -r mutation_supplement
```

### 直接运行（测试用）

```bash
python3 mutation.py -ec settings/mutation_2x_supplement.json
```

---

## ⏱️ 运行时间估算

基于历史数据估算：

| 模型 | 单次时间 | 补充次数 | 预计时间 |
|------|---------|---------|---------|
| MRT-OAST | ~22分钟 | 20 | ~7小时 |
| bug-localization | ~18分钟 | 22 | ~7小时 |
| pytorch_resnet_cifar10 | ~20分钟 | 16 | ~5.3小时 |
| VulBERTa mlp | ~27分钟 | 16 | ~7.2小时 |
| Person_reID densenet121 | ~90分钟 | 16 | ~24小时 |
| Person_reID hrnet18 | ~130分钟 | 32 | ~69小时 |
| Person_reID pcb | ~70分钟 | 32 | ~37小时 |
| examples mnist | ~3分钟 | 22 | ~1.1小时 |
| examples mnist_rnn | ~9分钟 | 22 | ~3.3小时 |
| examples siamese | ~10分钟 | 22 | ~3.7小时 |
| examples mnist_ff | ~0.5分钟 | 32 | ~0.3小时 |

**总计预计运行时间**: **约136小时** (~5.7天)

考虑到并行效率和GPU负载，实际可能需要 **5-8天**。

---

## 🎯 与之前配置的区别

### 旧配置（多参数同时变异）

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "num_mutations": 8,
  "mutate": ["epochs", "learning_rate", "seed", "weight_decay"]
}
```

**行为**：每次变异同时改变所有4个参数
- 变异1: `{epochs: 随机1, learning_rate: 随机1, seed: 随机1, weight_decay: 随机1}`
- 变异2: `{epochs: 随机2, learning_rate: 随机2, seed: 随机2, weight_decay: 随机2}`
- ...

**问题**：无法独立分析每个参数的影响

### 新配置（单参数独立变异）

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "num_mutations": 2,
  "mutate": ["epochs"]
},
{
  "repo": "VulBERTa",
  "model": "mlp",
  "num_mutations": 2,
  "mutate": ["learning_rate"]
},
...
```

**行为**：每个配置只变异一个参数
- epochs变异1: `{epochs: 随机1, learning_rate: 默认, seed: 默认, weight_decay: 默认}`
- epochs变异2: `{epochs: 随机2, learning_rate: 默认, seed: 默认, weight_decay: 默认}`
- learning_rate变异1: `{epochs: 默认, learning_rate: 随机1, seed: 默认, weight_decay: 默认}`
- ...

**优势**：每个参数的影响可以独立分析 ✓

---

## 📈 预期结果

运行完成后，`summary_all.csv` 应包含：

| Repository | Model | 当前总数 | 补充 | 完成后 | 达标 |
|------------|-------|---------|------|-------|------|
| MRT-OAST | default | 32 | 20 | 52 | ✓ |
| bug-localization | default | 20 | 24 | 44 | ✓ |
| pytorch_resnet_cifar10 | resnet20 | 26 | 16 | 42 | ✓ |
| VulBERTa | mlp | 26 | 16 | 42 | ✓ |
| Person_reID | densenet121 | 26 | 16 | 42 | ✓ |
| Person_reID | hrnet18 | 11 | 32 | 43 | ✓ |
| Person_reID | pcb | 10 | 32 | 42 | ✓ |
| examples | mnist | 20 | 24 | 44 | ✓ |
| examples | mnist_rnn | 20 | 24 | 44 | ✓ |
| examples | siamese | 20 | 24 | 44 | ✓ |
| examples | mnist_ff | 0 | 32 | 32 | ⚠️ * |

\* mnist_ff只有3个可变超参数（batch_size固定），目标应为32次（1+3×5=16 ×2模式）

**总计**: 211 → 471 (增加260次实验)

---

## 🔍 监控和验证

### 实时监控

```bash
# 查看screen会话
screen -ls

# 连接到运行中的会话
screen -r mutation_supplement

# 查看最新结果
tail -f results/run_*/summary.csv
```

### 完成后验证

```bash
# 统计各模型实验数
awk -F',' 'NR>1 {print $3","$4}' results/summary_all.csv | sort | uniq -c | sort -rn

# 验证单参数变异
# 检查是否每次只变一个参数（与默认值比较）
```

---

## ⚠️ 注意事项

### 1. 配置数量较多

本配置包含 **45个独立配置**，每个配置运行 `runs_per_config=2` 次（非并行+并行）。

**优点**：细粒度控制，每个参数独立变异
**缺点**：配置文件较长，但逻辑清晰

### 2. 运行时间较长

预计需要 **5-8天** 完成所有实验。

**建议**：
- 使用screen后台运行
- 定期检查进度
- 准备好GPU长时间运行

### 3. 中断恢复

如果实验中断：
- 已完成的实验保存到 `summary_all.csv`
- 重新运行时，去重机制会跳过已完成实验
- 从中断点继续运行

### 4. 去重验证

运行前建议检查：
```bash
# 当前实验数
wc -l results/summary_all.csv

# 预期最终实验数
# 当前211 + 补充252 = 463
```

---

## 📚 相关文档

- **配置文件**: `settings/mutation_2x_supplement.json`
- **模型配置**: `mutation/models_config.json`
- **去重机制**: `mutation/dedup.py`
- **变异生成**: `mutation/hyperparams.py`

---

**文档版本**: 2.1 (单参数独立变异 + 去重优化)
**最后更新**: 2025-11-26
**状态**: ✅ 配置完成并已修正，45个配置，预计260次实验
