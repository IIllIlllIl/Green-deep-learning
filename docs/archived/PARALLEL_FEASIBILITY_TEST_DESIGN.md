# 并行训练可行性测试设计文档

**实验配置文件**: `settings/parallel_feasibility_test.json`
**创建时间**: 2025-11-15 16:50
**目的**: 验证12组并行训练组合的GPU显存兼容性

---

## 1. 测试背景

### 1.1 并行训练模式

在并行训练模式下：
- **Foreground (前台)**: 被监控的训练进程（能耗、性能数据采集）
- **Background (后台)**: GPU负载进程（持续循环训练）

### 1.2 显存管理挑战

不同模型的显存占用差异巨大：
- 小模型: 500-1000MB (examples_mnist_ff)
- 中模型: 1500-2500MB (ResNet20, VulBERTa)
- 大模型: 3000-4000MB (DenseNet121, PCB)

**目标**: 找到可以安全并行的模型组合

---

## 2. 测试设计

### 2.1 测试策略

**快速验证方法**:
- ✅ 所有模型训练 **1个epoch**（最快验证）
- ✅ 使用默认超参数（稳定配置）
- ✅ 测试12个显存层级组合

**验证重点**:
1. GPU显存是否溢出 (CUDA out of memory)
2. 训练是否能正常启动
3. 性能指标是否正常输出

### 2.2 显存层级分类

| 显存层级 | 总显存范围 | 组合数 | 预期风险 |
|---------|-----------|--------|---------|
| 超低显存 | 1300MB | 1组 | ✅ 极低 |
| 低显存 | 2000-2700MB | 4组 | ✅ 低 |
| 中显存 | 3000-4000MB | 6组 | ⚠️ 中等 |
| 高显存 | 5000MB | 1组 | ⚠️ 较高 |

---

## 3. 测试组合详情

### Group 1: 超低显存 (1300MB)

```json
{
  "foreground": {
    "repo": "examples",
    "model": "mnist_ff",
    "hyperparameters": {"epochs": 1, "learning_rate": 0.01, "batch_size": 32}
  },
  "background": {
    "repo": "pytorch_resnet_cifar10",
    "model": "resnet20",
    "hyperparameters": {"epochs": 1, "learning_rate": 0.1}
  }
}
```

**显存分析**:
- mnist_ff: ~500MB (极小模型)
- resnet20: ~800MB (小型CNN)
- **总计**: 1300MB
- **风险**: ✅ 极低，应该稳定运行

---

### Group 2-3: 低显存 (2000MB)

**Group 2**: mnist + VulBERTa_mlp
- mnist: ~700MB
- VulBERTa_mlp: ~1300MB
- **总计**: 2000MB

**Group 3**: mnist + siamese
- mnist: ~700MB
- siamese: ~1300MB
- **总计**: 2000MB

**风险**: ✅ 低，显存充足

---

### Group 4-5: 低显存 (2500-2700MB)

**Group 4**: mnist_rnn + pcb
- mnist_rnn: ~500MB
- pcb: ~2000MB
- **总计**: 2500MB

**Group 5**: mnist_rnn + MRT-OAST
- mnist_rnn: ~500MB
- MRT-OAST: ~2200MB
- **总计**: 2700MB

**风险**: ✅ 低，但接近中等显存

---

### Group 6-11: 中显存 (3000-4000MB)

**Group 6**: mnist_rnn + hrnet18 (3000MB)
**Group 7**: VulBERTa_mlp + VulBERTa_cnn (3000MB)
**Group 8**: siamese + pcb (3500MB)
**Group 9**: pcb + VulBERTa_mlp (3500MB)
**Group 10**: mnist_ff + densenet121 (4000MB)
**Group 11**: pcb + bug-localization (4000MB)

**风险**: ⚠️ 中等
- 3000-3500MB: 应该安全
- 4000MB: 接近8GB GPU的一半，需验证

---

### Group 12: 高显存 (5000MB)

```json
{
  "foreground": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {"epochs": 1, "learning_rate": 0.05, "dropout": 0.5}
  },
  "background": {
    "repo": "VulBERTa",
    "model": "mlp",
    "hyperparameters": {"epochs": 1, "learning_rate": 0.00003}
  }
}
```

**显存分析**:
- densenet121: ~3700MB (大型模型)
- VulBERTa_mlp: ~1300MB
- **总计**: 5000MB
- **风险**: ⚠️ 较高（占8GB GPU的62.5%）

**注意**: 实际显存可能因batch size、中间激活值而更高

---

## 4. 预期结果

### 4.1 成功标准

**必须满足**:
- ✅ 所有12组训练启动成功
- ✅ 无CUDA out of memory错误
- ✅ Foreground训练完成1个epoch
- ✅ Background训练正常循环

**可选满足**:
- ✅ 性能指标正常提取
- ✅ 能耗数据正常采集

### 4.2 预期成功率

| 显存层级 | 组数 | 预期成功率 | 预期失败数 |
|---------|------|-----------|----------|
| 超低显存 (1300MB) | 1 | 100% | 0 |
| 低显存 (2000-2700MB) | 4 | 100% | 0 |
| 中显存 (3000-4000MB) | 6 | 95%+ | 0-1 |
| 高显存 (5000MB) | 1 | 80%+ | 0-1 |

**总预期成功率**: 95%+ (至少11/12组成功)

### 4.3 可能失败的组合

**高风险组合**:
1. Group 12 (densenet121 + VulBERTa_mlp, 5000MB)
   - 如果失败，考虑降低batch size

2. Group 11 (pcb + bug-localization, 4000MB)
   - bug-localization的max_iter可能影响显存

3. Group 10 (mnist_ff + densenet121, 4000MB)
   - densenet121单独就接近4GB

**应对策略**:
- 如果失败，记录错误信息
- 后续可调整batch size或模型配置

---

## 5. 预计时间

### 5.1 单组时间估算

| 模型类型 | 1 epoch时长 | 备注 |
|---------|-----------|------|
| examples (mnist系列) | 0.5-1分钟 | 极快 |
| VulBERTa | 1-2分钟 | 小模型 |
| ResNet20 | 1-2分钟 | CIFAR-10 |
| MRT-OAST | 2-3分钟 | 克隆检测 |
| Person ReID (pcb, hrnet) | 3-5分钟 | 中型模型 |
| DenseNet121 | 5-7分钟 | 大型模型 |
| bug-localization | 10-15分钟 | 最慢（kfold训练） |

### 5.2 总时间估算

**保守估算**:
- 12组实验，每组按最慢模型计算
- Group 1-9: 平均5分钟/组 = 45分钟
- Group 10-11: 15分钟/组 = 30分钟
- Group 12: 7分钟
- 间隔时间: 11 × 60秒 = 11分钟

**总计**: 约 **93分钟 (1.5小时)**

**乐观估算**: 1个epoch训练很快，可能仅需 **60-75分钟**

---

## 6. 运行命令

```bash
# 方式1: 直接运行
python mutation.py -ec settings/parallel_feasibility_test.json

# 方式2: 在screen中运行（推荐）
screen -S parallel_test
python mutation.py -ec settings/parallel_feasibility_test.json
# Ctrl+A D 分离

# 监控进程
watch -n 2 nvidia-smi  # 实时监控GPU显存
```

---

## 7. 监控要点

### 7.1 GPU显存监控

```bash
# 启动另一个terminal
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits'
```

**关键指标**:
- Memory Used: 应 < 7500MB (8GB GPU的94%)
- GPU Utilization: 应接近100%（双进程负载）

### 7.2 失败检测

**CUDA out of memory错误**:
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MB
```

**如果出现**:
- 记录哪个组合失败
- 检查training.log中的详细错误
- 后续调整该组合的配置

---

## 8. 预期输出

### 8.1 目录结构

```
results/run_YYYYMMDD_HHMMSS/
├── examples_mnist_ff_parallel_001/
│   ├── training.log
│   ├── experiment.json
│   ├── energy/
│   └── background_logs/
│       └── training_*.log
├── examples_mnist_parallel_001/
├── examples_mnist_parallel_002/
├── ... (12个实验目录)
└── summary.csv
```

### 8.2 关键数据

**每个实验目录包含**:
- `training.log`: Foreground训练日志
- `experiment.json`: 实验元数据（包含性能和能耗）
- `energy/`: CPU/GPU能耗监控数据
- `background_logs/`: Background训练日志

**summary.csv包含**:
- 12组实验的汇总
- 训练成功/失败状态
- 显存使用情况（如果能获取）

---

## 9. 后续分析

### 9.1 成功组合分类

运行后需要分析：

**稳定组合** (显存 < 3000MB):
- 可用于长时间并行训练
- 推荐用于能耗对比实验

**可用组合** (显存 3000-4000MB):
- 基本可用，需监控
- 可能需要调整batch size

**高风险组合** (显存 > 4000MB):
- 仅在必要时使用
- 建议降低batch size或选择其他组合

### 9.2 失败组合处理

如果有组合失败：
1. 记录失败的显存需求
2. 调整配置：
   - 降低batch size
   - 减少中间层激活值缓存
   - 使用梯度累积
3. 重新测试

---

## 10. 与正式实验的关系

### 10.1 本测试的作用

**快速验证阶段**:
- ✅ 验证显存兼容性
- ✅ 识别可行组合
- ✅ 避免正式实验中的显存溢出

**本测试不包括**:
- ❌ 完整训练（仅1 epoch）
- ❌ 能耗详细分析
- ❌ 性能对比研究

### 10.2 后续正式实验

基于本测试结果：
1. **选择稳定组合** (显存 < 3000MB)
2. **设计正式并行实验** (完整epochs)
3. **进行能耗对比分析**

---

## 11. 总结

### 11.1 测试关键点

- **快速验证**: 每组仅1 epoch，总时长约1.5小时
- **覆盖全面**: 12个显存层级组合
- **风险可控**: 大部分组合预期成功
- **易于监控**: 实时GPU显存监控

### 11.2 成功指标

| 指标 | 目标 |
|------|------|
| 成功率 | ≥ 95% (11/12组) |
| 显存安全 | < 7500MB (94% of 8GB) |
| 训练完成 | 所有foreground训练完成1 epoch |
| 无致命错误 | 无CUDA OOM |

---

**文档版本**: 1.0
**状态**: ✅ 配置已创建，等待运行
**配置文件**: `settings/parallel_feasibility_test.json`
**预计时间**: 60-93分钟 (1-1.5小时)
