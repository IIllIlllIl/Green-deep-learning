# Mutation 2x Supplement 配置说明

**配置文件**: `settings/mutation_2x_supplement.json`
**创建日期**: 2025-11-27
**目标**: 补全11个模型的实验到目标次数

---

## 📊 实验目标

对11个模型运行：
- **1次默认值训练** (所有超参数使用默认值)
- **每个超参数 5次不同的变异值**
- **非并行 + 并行各运行1次**

### 计算公式

对于有 N 个超参数的模型：
- 目标实验数 = 1 (默认) + N × 5 (变异) = 1 + 5N 次/模式
- 总计 = (1 + 5N) × 2 模式 = 2 + 10N 次

---

## 🎯 11个模型配置

| # | Repository | Model | 超参数数 | 目标(每模式) | 当前(非+并) | 缺口(非+并) | num_mutations |
|---|------------|-------|---------|------------|------------|-----------|---------------|
| 1 | MRT-OAST | default | 5 | 26 | 16+16=32 | 10+10=20 | 10 |
| 2 | bug-localization-by-dnn-and-rvsm | default | 4 | 21 | 10+10=20 | 11+11=22 | 11 |
| 3 | pytorch_resnet_cifar10 | resnet20 | 4 | 21 | 13+13=26 | 8+8=16 | 8 |
| 4 | VulBERTa | mlp | 4 | 21 | 13+13=26 | 8+8=16 | 8 |
| 5 | Person_reID_baseline_pytorch | densenet121 | 4 | 21 | 13+13=26 | 8+8=16 | 8 |
| 6 | Person_reID_baseline_pytorch | hrnet18 | 4 | 21 | 6+5=11 | 15+16=31 | 16 |
| 7 | Person_reID_baseline_pytorch | pcb | 4 | 21 | 5+5=10 | 16+16=32 | 16 |
| 8 | examples | mnist | 4 | 21 | 10+10=20 | 11+11=22 | 11 |
| 9 | examples | mnist_rnn | 4 | 21 | 10+10=20 | 11+11=22 | 11 |
| 10 | examples | siamese | 4 | 21 | 10+10=20 | 11+11=22 | 11 |
| 11 | examples | mnist_ff | 4 | 21 | 0+0=0 | 21+21=42 | 21 |

**总计需要补充**: **261 次实验**

---

## 📝 各模型超参数列表

### 1. MRT-OAST (5个超参数)
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `seed`: 随机种子
- `dropout`: Dropout比率
- `weight_decay`: 权重衰减

### 2. bug-localization-by-dnn-and-rvsm (4个超参数)
- `max_iter`: 最大迭代次数
- `kfold`: K折交叉验证
- `alpha`: 正则化参数
- `seed`: 随机种子

### 3. pytorch_resnet_cifar10 resnet20 (4个超参数)
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `seed`: 随机种子
- `weight_decay`: 权重衰减

### 4. VulBERTa mlp (4个超参数)
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `seed`: 随机种子
- `weight_decay`: 权重衰减

### 5-7. Person_reID_baseline_pytorch (4个超参数)
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `seed`: 随机种子
- `dropout`: Dropout比率

**特殊配置**:
- **pcb**: 设置 `batchsize=8` (原因: 防止GPU OOM)

### 8-11. examples (4个超参数)
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `seed`: 随机种子

**特殊配置**:
- **mnist_ff**: 设置 `batch_size=10000` (原因: 防止GPU OOM，之前batch_size=50000全部失败)

---

## 🔧 配置参数说明

### 基本参数
```json
{
  "experiment_name": "mutation_2x_supplement_20251127",
  "mode": "batch",
  "runs_per_config": 2,              // 每个配置运行2次（非并行+并行）
  "max_retries": 2,                  // 最大重试次数
  "governor": "performance",          // CPU调度策略
  "cleanup_gpu_memory": true,         // GPU内存清理
  "cleanup_between_experiments": true // 实验间清理
}
```

### 去重机制
```json
{
  "use_deduplication": true,
  "historical_csvs": [
    "results/summary_all.csv"        // 历史实验记录
  ]
}
```

系统会：
1. 生成指定数量的变异配置
2. 与历史记录比对，过滤已运行的配置
3. 仅运行未重复的配置

---

## 🚀 运行方式

### 方式1: 直接运行（推荐用于测试）
```bash
python3 mutation.py -ec settings/mutation_2x_supplement.json
```

### 方式2: Screen后台运行（推荐用于长时间实验）
```bash
# 创建新screen会话
screen -S mutation_supplement

# 以sudo运行（保留环境变量）
sudo -E python3 mutation.py -ec settings/mutation_2x_supplement.json

# 分离会话: Ctrl+A, D
# 重新连接: screen -r mutation_supplement
```

---

## ⚠️ 注意事项

### 1. OOM (内存溢出) 处理
已针对容易OOM的模型调整参数：
- **Person_reID_baseline_pytorch pcb**: `batchsize=8`
- **examples mnist_ff**: `batch_size=10000`

如果仍然出现OOM，可以进一步降低batch size。

### 2. 运行时间估算

根据历史数据：
- **VulBERTa mlp** (5 epochs): ~27分钟
- **Person_reID densenet121** (60 epochs): ~90分钟
- **Person_reID hrnet18** (60 epochs): ~130分钟
- **Person_reID pcb** (60 epochs): ~70分钟
- **pytorch_resnet_cifar10** (200 epochs): ~20分钟
- **examples mnist**: ~3分钟
- **examples mnist_rnn**: ~9分钟
- **examples siamese**: ~10分钟

**总运行时间估算**: 约 **60-80 小时** (取决于GPU性能和并行效率)

### 3. 去重验证

系统会自动去重，但建议运行前检查：
```bash
# 检查当前实验数
grep -c "^" results/summary_all.csv

# 预计运行后的实验数 (当前211 + 补充261 = 472)
```

### 4. 中断恢复

如果实验中断：
- 已完成的实验结果会保存到 `summary_all.csv`
- 重新运行配置文件时，去重机制会跳过已完成的实验
- 仅运行未完成的部分

---

## 📈 预期结果

运行完成后，`summary_all.csv` 应包含：

| Repository | Model | 目标总数 | 当前 | 补充后 |
|------------|-------|---------|------|-------|
| MRT-OAST | default | 52 | 32 | 52 ✓ |
| bug-localization | default | 42 | 20 | 42 ✓ |
| pytorch_resnet_cifar10 | resnet20 | 42 | 26 | 42 ✓ |
| VulBERTa | mlp | 42 | 26 | 42 ✓ |
| Person_reID | densenet121 | 42 | 26 | 42 ✓ |
| Person_reID | hrnet18 | 42 | 11 | 43 ≈ |
| Person_reID | pcb | 42 | 10 | 42 ✓ |
| examples | mnist | 42 | 20 | 42 ✓ |
| examples | mnist_rnn | 42 | 20 | 42 ✓ |
| examples | siamese | 42 | 20 | 42 ✓ |
| examples | mnist_ff | 42 | 0 | 42 ✓ |

**总计**: 211 → 472 (增加261次实验)

> 注: hrnet18 由于当前非并行6次、并行5次不对称，补充16次后可能达到43次，略超目标。

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
cd /home/green/energy_dl/nightly
awk -F',' 'NR>1 {print $3","$4}' results/summary_all.csv | sort | uniq -c | sort -rn
```

---

**文档版本**: 1.0
**最后更新**: 2025-11-27
