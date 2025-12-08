# 最小验证实验设计文档

**实验配置文件**: `settings/minimal_validation.json`
**创建时间**: 2025-11-15 16:25
**目的**: 验证新变异范围的安全性和有效性

---

## 1. 实验背景

### 1.1 新变异范围标准

经过boundary_test_v2的结果分析，我们将变异范围统一为：

```
Learning Rate: [default × 0.5, default × 2.0]
Epochs:        [default × 0.5, default × 1.5]
Weight Decay:  [0.00001, 0.01] (绝对值)
Dropout:       [0.0, 0.5] (绝对值)
```

### 1.2 之前的失败案例

boundary_test_v2中发现的关键问题：

| 模型 | 配置 | 结果 | 问题 |
|------|------|------|------|
| DenseNet121 | lr=0.2 (4×) | mAP 0.17% (-72.8%) | ❌ **训练崩溃** |
| DenseNet121 | lr=0.0125 (0.25×) | mAP 69.02% (-4.0%) | ⚠️ 性能下降明显 |
| MRT-OAST | lr=0.000025 (0.25×) | F1 83.10% (-10.24%) | ❌ 性能严重下降 |
| MRT-OAST | lr=0.0004 (4×) | F1 99.23% (+5.89%) | ✅ 最优（但超范围） |

---

## 2. 实验设计

### 2.1 测试策略

针对新范围的**上界和下界**进行完整训练验证，确保：
1. **安全性**: 不会导致训练崩溃
2. **有效性**: 性能下降在可接受范围内（<5%）

### 2.2 实验配置

总共12个实验，测试Learning Rate边界（6个）+ Weight Decay边界（6个）：

#### Experiment 1: DenseNet121 LR下界 (0.5×)

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "densenet121",
  "hyperparameters": {
    "epochs": 60,
    "learning_rate": 0.025,  // 0.5× default (0.05)
    "dropout": 0.5,
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.0125 (0.25×) → mAP 69.02% (-4.0%)
- 新范围下界: lr=0.025 (0.5×)
- **预期**: mAP ≥ 71% (约-2%以内)

---

#### Experiment 2: DenseNet121 LR上界 (2×)

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "densenet121",
  "hyperparameters": {
    "epochs": 60,
    "learning_rate": 0.1,  // 2× default (0.05)
    "dropout": 0.5,
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.2 (4×) → mAP 0.17% ❌ **崩溃**
- 新范围上界: lr=0.1 (2×)
- **预期**: mAP ≥ 71% (约-2%以内，**不崩溃**)

---

#### Experiment 3: ResNet20 LR下界 (0.5×)

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.05,  // 0.5× default (0.1)
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.025 (0.25×) → Acc 90.86% (-0.84%)
- 新范围下界: lr=0.05 (0.5×)
- **预期**: Acc ≥ 91.2% (约-0.5%以内)

---

#### Experiment 4: ResNet20 LR上界 (2×)

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.2,  // 2× default (0.1)
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.4 (4×) → Acc 90.85% (-0.85%)
- Baseline: lr=0.1 (1×) → Acc 91.70%
- 新范围上界: lr=0.2 (2×)
- **预期**: Acc ≥ 91.2% (约-0.5%以内)

---

#### Experiment 5: MRT-OAST LR下界 (0.5×)

```json
{
  "repo": "MRT-OAST",
  "model": "default",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.00005,  // 0.5× default (0.0001)
    "dropout": 0.2,
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.000025 (0.25×) → F1 83.10% (-10.24%)
- Baseline: lr=0.0001 (1×) → F1 93.34%
- 新范围下界: lr=0.00005 (0.5×)
- **预期**: F1 ≥ 88% (约-5%以内)

---

#### Experiment 6: MRT-OAST LR上界 (2×)

```json
{
  "repo": "MRT-OAST",
  "model": "default",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.0002,  // 2× default (0.0001)
    "dropout": 0.2,
    "seed": 1334
  }
}
```

**对比**:
- 之前测试: lr=0.0004 (4×) → F1 99.23% (+5.89%) ✅ 最优
- Baseline: lr=0.0001 (1×) → F1 93.34%
- 新范围上界: lr=0.0002 (2×)
- **预期**: F1 96-97% (+3~4%)

**注意**: MRT-OAST的最优点在4×，新范围2×无法达到，但仍应优于baseline

---

#### Experiment 7: ResNet20 WD下界 (0.00001)

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 0.00001,  // 下界 (default=0.0001)
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0001, Acc 91.70%
- 新范围下界: wd=0.00001 (0.1× default)
- **预期**: Acc ≈ 91.5% (约-0.2%，过拟合风险略增)

---

#### Experiment 8: ResNet20 WD上界 (0.01)

```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 0.01,  // 上界 (100× default)
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0001, Acc 91.70%
- 新范围上界: wd=0.01 (100× default) - **极端正则化**
- **预期**: Acc 88-90% (约-2~4%，强正则化可能欠拟合)

---

#### Experiment 9: MRT-OAST WD下界 (0.00001)

```json
{
  "repo": "MRT-OAST",
  "model": "default",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.0001,
    "dropout": 0.2,
    "weight_decay": 0.00001,  // 下界
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0 (无weight decay), F1 93.34%
- 新范围下界: wd=0.00001 (微弱正则化)
- **预期**: F1 ≈ 93% (与baseline相似，或略有改善)
- **重要**: 验证从0改为非零的影响

---

#### Experiment 10: MRT-OAST WD上界 (0.01)

```json
{
  "repo": "MRT-OAST",
  "model": "default",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.0001,
    "dropout": 0.2,
    "weight_decay": 0.01,  // 上界 (极端)
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0, F1 93.34%
- 新范围上界: wd=0.01 (极强正则化)
- **预期**: F1 85-90% (约-5~8%，强正则化可能抑制学习)

---

#### Experiment 11: VulBERTa WD下界 (0.00001)

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.00003,
    "weight_decay": 0.00001,  // 下界
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0 (无weight decay)
- 新范围下界: wd=0.00001 (微弱正则化)
- **预期**: 性能与baseline相似或略有改善
- **重要**: VulBERTa首次测试非零weight decay

---

#### Experiment 12: VulBERTa WD上界 (0.01)

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.00003,
    "weight_decay": 0.01,  // 上界 (极端)
    "seed": 1334
  }
}
```

**测试目的**:
- Default: wd=0.0
- 新范围上界: wd=0.01 (极强正则化)
- **预期**: 性能可能下降5-10%（强正则化对小模型影响显著）

---

## 3. 预期结果总览

### 3.1 Learning Rate边界测试

| 实验 | 模型 | 配置 | 对比值 | 预期性能 | 预期变化 | 验证目标 |
|------|------|------|--------|---------|---------|---------|
| **1** | DenseNet121 | lr=0.025 (0.5×) | 0.0125→-4.0% | mAP ≥ 71% | -2% | ✅ 改善 |
| **2** | DenseNet121 | lr=0.1 (2×) | 0.2→崩溃 | mAP ≥ 71% | -2% | ✅ **不崩溃** |
| **3** | ResNet20 | lr=0.05 (0.5×) | 0.025→-0.84% | Acc ≥ 91.2% | -0.5% | ✅ 改善 |
| **4** | ResNet20 | lr=0.2 (2×) | 0.4→-0.85% | Acc ≥ 91.2% | -0.5% | ✅ 相似 |
| **5** | MRT-OAST | lr=0.00005 (0.5×) | 0.000025→-10.24% | F1 ≥ 88% | -5% | ✅ 改善 |
| **6** | MRT-OAST | lr=0.0002 (2×) | 0.0004→+5.89% | F1 96-97% | +3~4% | ⚠️ 次优 |

### 3.2 Weight Decay边界测试

| 实验 | 模型 | 配置 | Default WD | 预期性能 | 预期变化 | 验证目标 |
|------|------|------|-----------|---------|---------|---------|
| **7** | ResNet20 | wd=0.00001 (0.1×) | 0.0001 | Acc ≈ 91.5% | -0.2% | ✅ 轻微影响 |
| **8** | ResNet20 | wd=0.01 (100×) | 0.0001 | Acc 88-90% | -2~4% | ⚠️ 强正则化 |
| **9** | MRT-OAST | wd=0.00001 | 0.0 | F1 ≈ 93% | ±0% | ✅ 验证非零WD |
| **10** | MRT-OAST | wd=0.01 | 0.0 | F1 85-90% | -5~8% | ⚠️ 可能过强 |
| **11** | VulBERTa | wd=0.00001 | 0.0 | 相似 | ±0% | ✅ 首次WD测试 |
| **12** | VulBERTa | wd=0.01 | 0.0 | 下降 | -5~10% | ⚠️ 小模型敏感 |

---

## 4. 成功标准

### 4.1 安全性验证 ✅ (必须满足)

**Learning Rate**:
- ✅ DenseNet121 lr=0.1 **不崩溃** (之前0.2崩溃)
- ✅ 所有LR实验训练成功完成
- ✅ 无CUDA out of memory或其他致命错误

**Weight Decay**:
- ✅ 所有WD实验训练成功完成
- ✅ 强正则化(wd=0.01)不导致完全无法学习
- ✅ 从wd=0改为wd=0.00001不破坏训练

### 4.2 有效性验证 (期望满足)

**Learning Rate边界**:
- ✅ DenseNet121: 性能下降 < 5% (预期约-2%)
- ✅ ResNet20: 性能下降 < 1% (预期约-0.5%)
- ✅ MRT-OAST:
  - 下界 (0.5×): 性能下降 < 10% (预期约-5%)
  - 上界 (2×): 性能提升 > 0% (预期+3~4%)

**Weight Decay边界**:
- ✅ ResNet20:
  - 下界(0.00001): 性能下降 < 1%
  - 上界(0.01): 性能下降 < 5%
- ✅ MRT-OAST/VulBERTa:
  - 下界(0.00001): 性能变化 < ±2%
  - 上界(0.01): 性能下降 < 10% (可接受，因为default=0)

### 4.3 对比基准

| 模型 | Baseline配置 | 性能指标 |
|------|-------------|---------|
| DenseNet121 | lr=0.05, epochs=60, dropout=0.5 | mAP 73.01% |
| ResNet20 | lr=0.1, epochs=200, wd=0.0001 | Test Acc 91.70% |
| MRT-OAST | lr=0.0001, epochs=10, dropout=0.2, wd=0.0 | F1 93.34% |
| VulBERTa | lr=0.00003, epochs=10, wd=0.0 | 未知（需baseline） |

---

## 5. 预计时间

### 5.1 按模型分组时间估算

基于boundary_test_v2的实际时间：

| 模型组 | 实验数 | 单次时长 | 小计 | 备注 |
|--------|-------|---------|------|------|
| DenseNet121 | 2 | 36分 | **72分** | Person ReID (LR边界) |
| ResNet20 | 4 | 19分 | **76分** | CIFAR-10 (LR+WD边界) |
| MRT-OAST | 4 | 22分 | **88分** | Clone Detection (LR+WD边界) |
| VulBERTa | 2 | ~25分 | **50分** | Vulnerability (WD边界，估算) |

**训练总时长**: 约 **286分钟 (4.77小时)**

### 5.2 总墙钟时间估算

```
训练时间:     286分钟
实验间隔:     11 × 60秒 = 11分钟
配置间隔:     3 × 60秒 = 3分钟
系统开销:     约10分钟
─────────────────────
总计:         约310分钟 (5.2小时)
```

---

## 6. 运行命令

```bash
# 设置性能模式
sudo cpupower frequency-set -g performance

# 运行实验
python mutation.py -ec settings/minimal_validation.json

# 或在screen中运行
screen -S minimal_val
python mutation.py -ec settings/minimal_validation.json
# Ctrl+A D 分离
```

---

## 7. 预期输出

成功运行后应生成：

```
results/run_YYYYMMDD_HHMMSS/
├── Person_reID_baseline_pytorch_densenet121_train_001/  # Exp 1: LR 0.025
│   ├── training.log
│   ├── experiment.json
│   └── energy/
├── Person_reID_baseline_pytorch_densenet121_train_002/  # Exp 2: LR 0.1
├── pytorch_resnet_cifar10_resnet20_train_001/           # Exp 3: LR 0.05
├── pytorch_resnet_cifar10_resnet20_train_002/           # Exp 4: LR 0.2
├── pytorch_resnet_cifar10_resnet20_train_003/           # Exp 7: WD 0.00001
├── pytorch_resnet_cifar10_resnet20_train_004/           # Exp 8: WD 0.01
├── MRT-OAST_default_train_001/                          # Exp 5: LR 0.00005
├── MRT-OAST_default_train_002/                          # Exp 6: LR 0.0002
├── MRT-OAST_default_train_003/                          # Exp 9: WD 0.00001
├── MRT-OAST_default_train_004/                          # Exp 10: WD 0.01
├── VulBERTa_mlp_train_001/                              # Exp 11: WD 0.00001
├── VulBERTa_mlp_train_002/                              # Exp 12: WD 0.01
└── summary.csv
```

**关键文件**:
- `experiment.json`: 包含性能指标、能耗数据、超参数配置
- `training.log`: 完整训练日志
- `summary.csv`: 所有实验的汇总表

---

## 8. 后续分析

实验完成后需要：

1. **提取性能指标**: 从experiment.json中提取所有性能数据
2. **对比分析**: 与boundary_test_v2结果对比
3. **验证结论**: 确认新范围的安全性和有效性
4. **生成报告**: 创建最小验证结果报告

---

**文档版本**: 1.0
**状态**: ✅ 配置已创建，等待运行
**配置文件**: `settings/minimal_validation.json`
