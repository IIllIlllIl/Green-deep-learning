# 超参数变异范围定义

## 版本历史

### v2.0 - 保守范围 (2025-11-12) ⭐ 当前版本

基于边界测试结果调整，避免训练崩溃。

| 超参数 | 变异范围 | 分布 | 理由 |
|--------|---------|------|------|
| **Epochs** | `[default×0.5, default×2.0]` | 对数均匀分布 | 直接影响能耗(线性)，避免欠拟合和过拟合 |
| **Learning Rate** | `[default×0.25, default×4.0]` | 对数均匀分布 ⭐ | 降低上界避免崩溃，保持合理变异 |
| **Weight Decay** | `[0.0, default×100]` | 30%零值 + 70%对数 | 防过拟合，对数敏感，允许无正则化 |
| **Dropout** | `[0.0, 0.4]` | 均匀分布 | 降低上界避免性能下降 |
| **Seed** | `[0, 9999]` | 均匀整数 | 评估稳定性，不直接影响能耗 |

**关键调整**：
- ✅ LR上界：5× → 4× (避免DenseNet121崩溃)
- ✅ LR下界：0.2× → 0.25× (避免MRT-OAST性能下降)
- ✅ Dropout上界：0.5 → 0.4 (避免MRT-OAST性能下降)

---

### v1.0 - 激进范围 (2025-11-11) ❌ 已废弃

**问题**：LR=5×default导致DenseNet121训练崩溃

| 超参数 | 变异范围 | 问题 |
|--------|---------|------|
| **Learning Rate** | `[default×0.2, default×5.0]` | ❌ 上界过大 |
| **Dropout** | `[0.0, 0.5]` | ⚠️ 上界略大 |

**测试结果** (2025-11-11):
- DenseNet121 @ LR=5×default (0.25): Rank@1 = 0% (崩溃)
- MRT-OAST @ LR=0.2×default: Precision下降9%
- MRT-OAST @ Dropout=0.5: Recall下降12%

---

## 各模型的具体范围

### Person_reID_baseline_pytorch (DenseNet121, HRNet18, PCB)

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 60 | [30, 120] | 0.5× - 2× |
| Learning Rate | 0.05 | [0.0125, 0.2] | 0.25× - 4× |
| Dropout | 0.5 | [0.0, 0.4] | 固定范围 |
| Seed | 1334 | [0, 9999] | 随机 |

**关键发现**：此模型对LR非常敏感，LR=5×导致完全崩溃。

### MRT-OAST

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 10 | [5, 20] | 0.5× - 2× |
| Learning Rate | 0.0001 | [0.000025, 0.0004] | 0.25× - 4× |
| Dropout | 0.2 | [0.0, 0.2] | 固定范围 (不超过默认) |
| Weight Decay | 0.0 | [0.0, 0.0] | 保持默认 |
| Seed | 1334 | [0, 9999] | 随机 |

**关键发现**：
- LR=0.2×时性能下降9%
- Dropout=0.5时性能下降12%
- Dropout最大值不超过默认值0.2

### pytorch_resnet_cifar10 (ResNet20, ResNet32, etc.)

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 200 | [100, 400] | 0.5× - 2× |
| Learning Rate | 0.1 | [0.025, 0.4] | 0.25× - 4× |
| Weight Decay | 0.0001 | [0.0, 0.01] | 0× - 100× |
| Seed | 1334 | [0, 9999] | 随机 |

**关键发现**：此模型对LR非常鲁棒，[0.2×, 5×]范围内性能下降<2%。

### VulBERTa (MLP, CNN)

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 10 | [5, 20] | 0.5× - 2× |
| Learning Rate | 0.001 | [0.00025, 0.004] | 0.25× - 4× |
| Weight Decay | 0.0 | [0.0, 0.0] | 保持默认 |
| Seed | 42 | [0, 9999] | 随机 |

### bug-localization-by-dnn-and-rvsm

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 10 | [5, 20] | 0.5× - 2× |
| Learning Rate | 0.001 | [0.00025, 0.004] | 0.25× - 4× |
| Seed | 42 | [0, 9999] | 随机 |

### examples (MNIST models)

| 超参数 | Default | 变异范围 (v2.0) | 具体值 |
|--------|---------|---------------|--------|
| Epochs | 10 | [5, 20] | 0.5× - 2× |
| Learning Rate | 0.001 | [0.00025, 0.004] | 0.25× - 4× |
| Seed | 1334 | [0, 9999] | 随机 |

---

## 验证建议

### 关键测试点

在使用v2.0范围进行大规模变异前，建议验证以下边界：

1. **DenseNet121 @ LR=4×default (0.2)**
   - 这是最关键的测试点
   - 需要验证不会像LR=5×那样崩溃
   - 配置文件：`boundary_test_conservative.json`

2. **MRT-OAST @ LR=0.25×default (0.000025)**
   - 验证下界不会导致性能过差
   - 可接受标准：Precision > 85%

3. **所有模型 @ Dropout=0.4**
   - 验证新的Dropout上界
   - 可接受标准：性能下降 < 10%

### 验证命令

```bash
# 运行保守边界测试
sudo python3 mutation.py -ec settings/boundary_test_conservative.json

# 预计时间：约6小时 (13个实验)
```

---

## 使用指南

### 在配置文件中使用

#### 方法1：使用统一表达式（推荐）⭐

```json
{
  "experiment_name": "my_mutation_test",
  "mode": "mutation",
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["epochs", "learning_rate", "weight_decay"],
      "runs": 10
    }
  ]
}
```

框架会自动从`config/models_config.json`读取范围定义。

#### 方法2：手动指定范围

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "densenet121",
  "hyperparameters": {
    "epochs": 60,
    "learning_rate": 0.0125,  // 0.25×default = 0.05 × 0.25
    "dropout": 0.4
  }
}
```

### 更新config/models_config.json

确保`config/models_config.json`中的range字段使用v2.0范围：

```json
"learning_rate": {
  "flag": "--lr",
  "type": "float",
  "default": 0.05,
  "range": [0.0125, 0.2],  // [0.25×, 4×]
  "distribution": "log_uniform"
},
"dropout": {
  "flag": "--dropout",
  "type": "float",
  "default": 0.5,
  "range": [0.0, 0.4],  // 降低上界
  "distribution": "uniform"
}
```

---

## 性能标准

基于边界测试，定义以下性能下降可接受标准：

| 模型类型 | 主要指标 | 可接受下降 |
|---------|---------|-----------|
| Person_reID | Rank@1 | < 5% |
| Person_reID | mAP | < 10% |
| MRT-OAST | Precision | < 10% |
| MRT-OAST | Recall | < 15% |
| ResNet | Accuracy | < 3% |

**过滤规则**：
- 在结果分析时，过滤掉性能下降超过标准的样本
- 标记为"异常样本"，不纳入能耗分析

---

## 参考文献

- 边界测试报告：`docs/BOUNDARY_TEST_RESULTS_2025-11-11.md`
- 归档配置：`settings/archive/boundary_test_lr_dropout_focused_2025-11-11.json`
- 保守测试配置：`settings/boundary_test_conservative.json`

---

**最后更新**: 2025-11-12
**状态**: 待验证 (需运行boundary_test_conservative.json)
