# 并行训练可行性测试 - 配置完成总结

**配置文件**: `settings/parallel_feasibility_test.json`
**设计文档**: `docs/PARALLEL_FEASIBILITY_TEST_DESIGN.md`
**完成时间**: 2025-11-15 17:00

---

## ✅ 配置概览

### 测试规模

- **总实验数**: 12组并行训练组合
- **训练规模**: 每组1个epoch（快速验证）
- **测试模式**: parallel（foreground监控 + background负载）
- **显存范围**: 1300MB - 5000MB

---

## 📊 测试组合列表

| # | 显存层级 | Foreground | Background | 总显存 | 风险 |
|---|---------|-----------|-----------|--------|------|
| 1 | 超低 | examples_mnist_ff | pytorch_resnet_cifar10_resnet20 | 1300MB | ✅ 极低 |
| 2 | 低 | examples_mnist | VulBERTa_mlp | 2000MB | ✅ 低 |
| 3 | 低 | examples_mnist | examples_siamese | 2000MB | ✅ 低 |
| 4 | 低 | examples_mnist_rnn | Person_reID_baseline_pytorch_pcb | 2500MB | ✅ 低 |
| 5 | 低 | examples_mnist_rnn | MRT-OAST_default | 2700MB | ✅ 低 |
| 6 | 中 | examples_mnist_rnn | Person_reID_baseline_pytorch_hrnet18 | 3000MB | ⚠️ 中 |
| 7 | 中 | VulBERTa_mlp | VulBERTa_cnn | 3000MB | ⚠️ 中 |
| 8 | 中 | examples_siamese | Person_reID_baseline_pytorch_pcb | 3500MB | ⚠️ 中 |
| 9 | 中 | Person_reID_baseline_pytorch_pcb | VulBERTa_mlp | 3500MB | ⚠️ 中 |
| 10 | 中 | examples_mnist_ff | Person_reID_baseline_pytorch_densenet121 | 4000MB | ⚠️ 中 |
| 11 | 中 | Person_reID_baseline_pytorch_pcb | bug-localization-by-dnn-and-rvsm_default | 4000MB | ⚠️ 中 |
| 12 | 高 | Person_reID_baseline_pytorch_densenet121 | VulBERTa_mlp | 5000MB | ⚠️ 高 |

---

## 🎯 测试目标

### 验证重点

1. ✅ **显存兼容性**: 每组显存 < 8GB GPU限制
2. ✅ **训练稳定性**: 并行训练不崩溃
3. ✅ **启动成功**: 前后台进程正常启动
4. ✅ **完成验证**: 1 epoch快速验证可行性

### 成功标准

- **成功率**: ≥ 95% (11/12组)
- **显存安全**: < 7500MB (8GB的94%)
- **无致命错误**: 无CUDA out of memory
- **训练完成**: 所有foreground完成1 epoch

---

## ⏱️ 预计时间

### 时间估算

| 项目 | 时长 |
|------|------|
| 训练时间 | 60-80分钟 |
| 实验间隔 | 11分钟 |
| 系统开销 | 2-5分钟 |
| **总计** | **73-96分钟 (1.2-1.6小时)** |

**乐观估算**: 约 **75分钟 (1.25小时)**

### 按组估算

- Groups 1-5 (低显存): ~3分钟/组 = 15分钟
- Groups 6-9 (中显存): ~5分钟/组 = 20分钟
- Groups 10-11 (含bug-loc): ~12分钟/组 = 24分钟
- Group 12 (高显存): ~7分钟

---

## 🚀 运行命令

```bash
# 推荐: 在screen中运行
screen -S parallel_test
python mutation.py -ec settings/parallel_feasibility_test.json
# Ctrl+A D 分离

# 另一个terminal监控GPU
watch -n 1 nvidia-smi
```

---

## 📈 预期成功率

| 显存层级 | 组数 | 预期成功 | 备注 |
|---------|------|---------|------|
| 超低 (1300MB) | 1 | 100% | 极稳定 |
| 低 (2000-2700MB) | 4 | 100% | 稳定 |
| 中 (3000-4000MB) | 6 | 95%+ | 基本稳定 |
| 高 (5000MB) | 1 | 80%+ | 可能需要调整 |
| **总计** | **12** | **95%+** | **11/12组预期成功** |

---

## ⚠️ 高风险组合

### Group 12: densenet121 + VulBERTa_mlp (5000MB)

**风险因素**:
- DenseNet121单模型约3700MB
- 总显存接近8GB的62.5%
- 实际运行可能因激活值更高

**应对措施**:
- 如果OOM，考虑降低batch size
- 或替换为更小的模型组合

### Group 11: pcb + bug-localization (4000MB)

**风险因素**:
- bug-localization的kfold训练复杂
- max_iter=10000可能影响显存

**应对措施**:
- 如果失败，记录错误信息
- 后续可调整max_iter或kfold参数

---

## 📂 输出结构

```
results/run_YYYYMMDD_HHMMSS/
├── examples_mnist_ff_parallel_001/
│   ├── training.log              (前台训练日志)
│   ├── experiment.json           (实验元数据)
│   ├── energy/                   (能耗监控数据)
│   └── background_logs/          (后台训练日志)
│       └── training_*.log
├── ... (12个实验目录)
└── summary.csv                   (汇总表)
```

---

## 🔍 后续分析

### 分类结果

**稳定组合** (显存 < 3000MB):
- 推荐用于正式并行实验
- 可长时间运行
- 显存安全裕度充足

**可用组合** (显存 3000-4000MB):
- 基本可用，需监控
- 适合短期并行实验
- 可能需要调整batch size

**高风险组合** (显存 > 4000MB):
- 谨慎使用
- 建议优化配置或选择替代组合

### 失败处理

如果有组合失败：
1. 记录失败的显存需求和错误信息
2. 调整配置（降低batch size）
3. 或选择显存更小的替代模型

---

## 📋 验证清单

- [x] 12组并行配置完整
- [x] 所有模型epochs=1（快速验证）
- [x] 使用默认超参数（稳定配置）
- [x] 显存层级覆盖全面（1300MB-5000MB）
- [x] 预期时间合理（约1.25小时）
- [x] 监控方案清晰（nvidia-smi）
- [x] 失败应对策略明确

---

## 🎯 与正式实验的关系

### 本测试作用

**快速验证阶段** (1 epoch):
- ✅ 验证显存兼容性
- ✅ 识别稳定组合
- ✅ 避免正式实验OOM

**不包括**:
- ❌ 完整训练性能评估
- ❌ 详细能耗分析
- ❌ 并行vs单独训练对比

### 后续正式实验

基于本测试结果：
1. 选择成功的稳定组合
2. 设计完整epochs并行实验
3. 进行能耗和性能对比分析

---

## 📊 对比表格

| 项目 | minimal_validation | parallel_feasibility_test |
|------|-------------------|--------------------------|
| 实验数 | 12 | 12 |
| 训练模式 | 单独训练 (default) | 并行训练 (parallel) |
| Epochs | 完整 (10-200) | 1 epoch (快速) |
| 总时长 | 5.2小时 | 1.25小时 |
| 测试目标 | 超参数范围验证 | 显存兼容性验证 |
| 优先级 | 高（验证新范围） | 中（准备并行实验） |

---

**状态**: ✅ **配置完成，随时可运行**

**建议顺序**:
1. 先运行 `minimal_validation.json` (5.2小时)
2. 再运行 `parallel_feasibility_test.json` (1.25小时)

两个测试互补，共同验证系统的完整功能！
