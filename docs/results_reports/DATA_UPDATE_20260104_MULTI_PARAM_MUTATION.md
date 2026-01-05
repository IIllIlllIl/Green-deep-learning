# 数据更新日志 - 2026-01-04

## 多参数组合变异实验数据追加

### 实验概述
- **配置文件**: `settings/multi_param_mutation_all_models.json`
- **运行时间**: 2025-12-30 20:42 - 2026-01-02 10:42
- **实验目的**: 研究超参数组合效应，对比单参数变异与多参数同时变异的差异

### 数据统计
- **新增实验**: 110个
- **训练成功率**: 100% (110/110)
- **能耗数据完整性**: 100%
- **模式分布**: 55个非并行 + 55个并行

### 模型覆盖（11个模型 × 2模式 × 5重复）
1. examples/mnist - 10个实验（5非并行 + 5并行）
2. examples/mnist_ff - 10个实验（5非并行 + 5并行）
3. examples/mnist_rnn - 10个实验（5非并行 + 5并行）
4. examples/siamese - 10个实验（5非并行 + 5并行）
5. pytorch_resnet_cifar10/resnet20 - 10个实验（5非并行 + 5并行）
6. Person_reID_baseline_pytorch/densenet121 - 10个实验（5非并行 + 5并行）
7. Person_reID_baseline_pytorch/hrnet18 - 10个实验（5非并行 + 5并行）
8. Person_reID_baseline_pytorch/pcb - 10个实验（5非并行 + 5并行）
9. MRT-OAST/default - 10个实验（5非并行 + 5并行）
10. VulBERTa/mlp - 10个实验（5非并行 + 5并行）
11. bug-localization-by-dnn-and-rvsm/default - 10个实验（5非并行 + 5并行）

### 变异的超参数
各模型同时变异以下超参数：
- **examples模型**: batch_size, epochs, learning_rate, seed
- **resnet20**: epochs, learning_rate, weight_decay, seed
- **Person_reID**: dropout, epochs, learning_rate, seed
- **MRT-OAST**: dropout, epochs, learning_rate, weight_decay, seed
- **VulBERTa**: epochs, learning_rate, weight_decay, seed
- **bug-localization**: alpha, kfold, max_iter, seed

### 数据文件更新
- **raw_data.csv**: 726行 → 836行（+110行）
- **data.csv**: 726行 → 836行（+110行）
- **备份文件**:
  - `raw_data.csv.backup_20260104_173354`
  - `data.csv.backup_20260104_173400`

### 数据验证
- ✅ 所有实验训练成功
- ✅ 能耗数据100%完整
- ✅ 性能指标提取成功
- ✅ 无重复数据
- ✅ CSV文件完整性验证通过

### 数据提取工具
- 使用 `tools/data_management/append_session_to_raw_data.py` 从session目录提取数据
- 使用 `tools/data_management/create_unified_data_csv.py` 生成精简版data.csv

---
**更新人**: Claude
**更新时间**: 2026-01-04 17:34
