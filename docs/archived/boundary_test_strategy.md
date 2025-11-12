# 超参数边界值测试方案

**目标**: 验证当前变异范围对模型性能的影响，确定合理的超参数边界
**日期**: 2025-11-10
**方法**: 边界值测试（Boundary Value Testing）

---

## 测试目标

通过测试超参数范围的**边界值**（最小值、最大值、默认值），评估：

1. **范围合理性**: 边界值是否导致性能严重下降？
2. **性能影响**: 每个参数的边界值对性能的具体影响程度
3. **安全范围**: 确定不影响模型性能的"安全"变异范围
4. **极端组合**: 测试最坏情况（所有参数取极端值）

**关键原则**:
- ✅ 如果边界值仍能保持**合理性能**（如准确率下降<5%），说明范围合理
- ❌ 如果边界值导致**性能崩溃**（如准确率下降>10%），说明范围过宽，需要收窄

---

## 测试设计

### 测试策略：单因素变量法（One-Factor-at-a-Time）

对每个模型，测试以下配置：

1. **Default**: 默认值（基线性能）
2. **Min/Max单参数**: 仅改变一个参数到边界，其他保持默认
3. **Extreme Combination**: 多个参数同时取极端值（可选）

### 测试配置总数

| 模型 | 参数数 | 基线 | 单参数边界 | 极端组合 | 总配置数 |
|------|--------|------|-----------|---------|---------|
| **examples/mnist** | 2 (epochs, lr) | 1 | 4 | 2 | **7** |
| **pytorch_resnet_cifar10/resnet20** | 3 (epochs, lr, wd) | 1 | 6 | 1 | **8** |
| **Person_reID/densenet121** | 3 (epochs, lr, dropout) | 1 | 6 | 0 | **7** |
| **MRT-OAST/default** | 4 (epochs, lr, dropout, wd) | 1 | 8 | 0 | **9** |
| **总计** | - | **4** | **24** | **3** | **31** |

---

## 详细测试配置

### 1. examples/mnist (快速验证 - 1.5分钟/次)

**目的**: 快速验证边界值测试方法可行性

| # | Epochs | Learning Rate | 配置说明 | 预期结果 |
|---|--------|---------------|---------|---------|
| 1 | 10 | 0.01 | **Default** | 99.0% Acc (基线) |
| 2 | **5** | 0.01 | Min epochs | 95-97% Acc (轻微下降) |
| 3 | **20** | 0.01 | Max epochs | 99.0% Acc (持平或略升) |
| 4 | 10 | **0.001** | Min LR | 95-98% Acc (收敛慢) |
| 5 | 10 | **0.1** | Max LR | 98-99% Acc (可能震荡) |
| 6 | **5** | **0.001** | Min+Min | 90-95% Acc (最坏情况) |
| 7 | **20** | **0.1** | Max+Max | 99%+ Acc (最好情况) |

**验收标准**:
- ✅ 所有配置准确率 ≥ 90% → 范围合理
- ⚠️ 任意配置准确率 < 90% → 范围需调整

---

### 2. pytorch_resnet_cifar10/resnet20 (CIFAR-10分类 - 20分钟/次)

**目的**: 验证图像分类任务的超参数范围

| # | Epochs | Learning Rate | Weight Decay | 配置说明 | 预期结果 |
|---|--------|---------------|--------------|---------|---------|
| 1 | 200 | 0.1 | 0.0001 | **Default** | 91.45% Acc (已验证) |
| 2 | **100** | 0.1 | 0.0001 | Min epochs | 88-90% Acc |
| 3 | **400** | 0.1 | 0.0001 | Max epochs | 91-92% Acc |
| 4 | 200 | **0.01** | 0.0001 | Min LR | 85-88% Acc (欠拟合) |
| 5 | 200 | **1.0** | 0.0001 | Max LR | 80-85% Acc (震荡) |
| 6 | 200 | 0.1 | **0.00001** | Min WD | 90-91% Acc |
| 7 | 200 | 0.1 | **0.01** | Max WD | 88-90% Acc (过正则化) |
| 8 | 200 | 0.1 | **0.0** | Zero WD | 90-91% Acc |

**验收标准**:
- ✅ 边界值性能 ≥ 85% (基线91.45%，下降<7%) → 范围合理
- ⚠️ 边界值性能 < 85% → 该参数范围过宽

**关键测试**:
- **LR=1.0**: 学习率过大会导致震荡，期望此时性能显著下降，验证上界合理性
- **LR=0.01**: 学习率过小会导致欠拟合，期望此时性能轻微下降，验证下界合理性

---

### 3. Person_reID_baseline_pytorch/densenet121 (行人重识别 - 38分钟/次)

**目的**: 验证ReID任务的超参数范围

| # | Epochs | Learning Rate | Dropout | 配置说明 | 预期结果 |
|---|--------|---------------|---------|---------|---------|
| 1 | 60 | 0.05 | 0.5 | **Default** | 90.11% Rank@1 (已验证) |
| 2 | **30** | 0.05 | 0.5 | Min epochs | 85-88% Rank@1 |
| 3 | **120** | 0.05 | 0.5 | Max epochs | 90-91% Rank@1 |
| 4 | 60 | **0.005** | 0.5 | Min LR | 82-87% Rank@1 |
| 5 | 60 | **0.5** | 0.5 | Max LR | 75-85% Rank@1 (震荡) |
| 6 | 60 | 0.05 | **0.0** | Min dropout | 88-90% Rank@1 |
| 7 | 60 | 0.05 | **0.7** | Max dropout | 85-88% Rank@1 (信息损失) |

**验收标准**:
- ✅ 边界值性能 ≥ 80% Rank@1 (基线90.11%，下降<12%) → 范围合理
- ⚠️ 边界值性能 < 80% Rank@1 → 该参数范围过宽

**关键测试**:
- **Dropout=0.7**: 高dropout会阻碍信息流动，期望此时性能下降，验证上界合理性
- **LR=0.5**: 极高学习率可能导致不收敛，验证上界是否过宽

---

### 4. MRT-OAST/default (文本分类 - 42分钟/次)

**目的**: 验证NLP任务的超参数范围

| # | Epochs | Learning Rate | Dropout | Weight Decay | 配置说明 | 预期结果 |
|---|--------|---------------|---------|--------------|---------|---------|
| 1 | 10 | 0.0001 | 0.2 | 0.0 | **Default** | 90.10% Acc (已验证) |
| 2 | **5** | 0.0001 | 0.2 | 0.0 | Min epochs | 85-88% Acc |
| 3 | **20** | 0.0001 | 0.2 | 0.0 | Max epochs | 90-91% Acc |
| 4 | 10 | **0.00001** | 0.2 | 0.0 | Min LR | 80-85% Acc |
| 5 | 10 | **0.001** | 0.2 | 0.0 | Max LR | 85-90% Acc |
| 6 | 10 | 0.0001 | **0.0** | 0.0 | Min dropout | 88-90% Acc |
| 7 | 10 | 0.0001 | **0.5** | 0.0 | Max dropout | 85-88% Acc |
| 8 | 10 | 0.0001 | 0.2 | **0.00001** | Min WD | 89-90% Acc |
| 9 | 10 | 0.0001 | 0.2 | **0.01** | Max WD | 87-90% Acc |

**验收标准**:
- ✅ 边界值性能 ≥ 80% Acc (基线90.10%，下降<12%) → 范围合理
- ⚠️ 边界值性能 < 80% Acc → 该参数范围过宽

---

## 执行计划

### 阶段1: MNIST快速验证（推荐先执行）

**目的**: 15分钟内快速验证方法可行性

```bash
# 仅运行MNIST的7个配置
python3 test/extract_mnist_configs.py settings/boundary_test_elite_plus.json > settings/boundary_test_mnist_only.json

# 或手动运行
python3 mutation.py -r examples -m mnist -mt epochs,learning_rate -n 1
# 然后查看默认配置性能...
```

**预计时间**: 7个配置 × 1.5分钟 = **10.5分钟**

**决策点**:
- ✅ MNIST边界值测试通过 → 继续阶段2
- ❌ MNIST边界值测试失败 → 调整范围后重新测试

---

### 阶段2: 完整边界值测试

```bash
# 运行完整测试（31个配置）
screen -S boundary_test
python3 mutation.py -ec settings/boundary_test_elite_plus.json
# Ctrl+A D (分离)
```

**预计时间计算**:

| 模型 | 配置数 | 单次时长 | 小计 |
|------|--------|---------|------|
| MNIST | 7 | 1.5分钟 | 10.5分钟 |
| ResNet20 | 8 | 20分钟 | 160分钟 (2.7小时) |
| DenseNet121 | 7 | 38分钟 | 266分钟 (4.4小时) |
| MRT-OAST | 9 | 42分钟 | 378分钟 (6.3小时) |
| **总计** | **31** | - | **13.9小时** |

**加上间隔时间** (60秒/配置): 31分钟
**实际预计总时长**: **14.4小时** (约0.6天)

---

## 结果分析方法

### 步骤1: 提取性能指标

创建分析脚本 `analysis/analyze_boundary_test.py`:

```python
#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path("results")
model_results = defaultdict(list)

# 收集所有结果
for result_file in results_dir.glob("*.json"):
    with open(result_file) as f:
        data = json.load(f)

    repo = data["repository"]
    model = data["model"]
    hyperparams = data["hyperparameters"]
    perf = data["performance_metrics"]
    success = data["training_success"]

    key = f"{repo}/{model}"
    model_results[key].append({
        "hyperparams": hyperparams,
        "performance": perf,
        "success": success
    })

# 对每个模型分析
for model_key, results in model_results.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_key}")
    print(f"{'='*80}")

    # 找到默认配置（作为基线）
    baseline = None
    for r in results:
        # 根据具体模型判断是否为默认配置
        # 例如：ResNet20默认为epochs=200, lr=0.1, wd=0.0001
        # 这里简化为取第一个配置作为基线
        if baseline is None:
            baseline = r

    baseline_perf = list(baseline["performance"].values())[0]
    print(f"Baseline performance: {baseline_perf:.2f}%")

    # 对比每个边界值配置
    for r in results:
        if r == baseline:
            continue

        perf = list(r["performance"].values())[0]
        diff = perf - baseline_perf
        status = "✅" if diff >= -5 else "⚠️" if diff >= -10 else "❌"

        print(f"{status} {r['hyperparams']} → {perf:.2f}% (Δ{diff:+.2f}%)")
```

运行分析:
```bash
python3 analysis/analyze_boundary_test.py
```

---

### 步骤2: 性能下降分析

对每个参数，计算边界值导致的性能变化：

```
Performance Drop = (Boundary_Performance - Baseline_Performance) / Baseline_Performance × 100%
```

**判断标准**:

| 性能下降 | 评级 | 范围调整建议 |
|---------|------|-------------|
| < 3% | ✅ 优秀 | 范围合理，可以保持 |
| 3-5% | ✅ 良好 | 范围合理，可以保持 |
| 5-10% | ⚠️ 警告 | 考虑收窄范围（如×0.75） |
| > 10% | ❌ 不可接受 | 必须收窄范围（如×0.5） |

---

### 步骤3: 生成边界值测试报告

```bash
# 手动生成Markdown报告
python3 analysis/generate_boundary_report.py > docs/boundary_test_results.md
```

报告应包含：

1. **每个模型的基线性能**
2. **每个参数的边界值性能**
3. **性能下降百分比**
4. **范围调整建议**
5. **可视化图表**（性能 vs 超参数值）

---

## 预期结果与决策

### 场景1: 所有边界值性能下降 < 5%

**结论**: 当前变异范围**合理且安全**

**行动**:
- ✅ 保持当前范围不变
- ✅ 开始大规模变异实验（如20-50个变异点）

---

### 场景2: 部分边界值性能下降 5-10%

**结论**: 部分参数范围**略宽**，但可接受

**行动**:
- ⚠️ 收窄性能下降>5%的参数范围（如epochs [100,400] → [120,350]）
- ✅ 重新运行受影响参数的边界值测试
- ✅ 确认调整后，开始变异实验

---

### 场景3: 任意边界值性能下降 > 10%

**结论**: 该参数范围**过宽**，不可接受

**行动**:
- ❌ 必须收窄该参数范围（如learning_rate [0.01,1.0] → [0.05,0.5]）
- ❌ 重新设计变异范围
- ❌ 重新运行完整边界值测试
- ❌ 确认所有边界值性能下降<10%后，才能开始变异实验

---

## 范围调整示例

### 示例1: Learning Rate范围过宽

**测试发现**:
- Default (lr=0.1): 91.45% Acc
- Max (lr=1.0): 78.23% Acc (下降13.22%) ❌

**问题**: 上界lr=1.0导致性能崩溃

**调整方案**:
```json
// 原范围
"learning_rate": {
  "range": [0.01, 1.0],  // 10x倍数
  "default": 0.1
}

// 调整后范围（收窄上界至5x）
"learning_rate": {
  "range": [0.01, 0.5],  // 5x倍数
  "default": 0.1
}
```

**重新测试**: 验证lr=0.5时性能下降是否 < 10%

---

### 示例2: Epochs范围合理

**测试发现**:
- Default (epochs=200): 91.45% Acc
- Min (epochs=100): 89.12% Acc (下降2.33%) ✅
- Max (epochs=400): 92.01% Acc (提升0.56%) ✅

**结论**: Epochs范围合理，保持不变

---

## 执行检查清单

### 运行前准备
- [ ] 确认GPU可用
- [ ] 确认所有模型仓库已配置
- [ ] 确认配置文件格式正确
- [ ] 创建logs目录: `mkdir -p logs`
- [ ] 设置governor: `sudo ./governor.sh performance`

### 运行中监控
- [ ] 定期检查进程: `ps aux | grep mutation`
- [ ] 监控结果生成: `watch -n 60 'ls -lh results/*.json | wc -l'`
- [ ] 检查错误: `grep -i error results/*.log`

### 运行后分析
- [ ] 验证所有31个配置都已完成
- [ ] 提取性能指标
- [ ] 计算性能下降百分比
- [ ] 生成边界值测试报告
- [ ] 根据结果调整变异范围
- [ ] 决定是否开始变异实验

---

## 快速启动命令

```bash
# 方式1: 仅测试MNIST（10分钟快速验证）
# TODO: 手动提取MNIST配置或使用grep

# 方式2: 完整边界值测试（14小时）
screen -S boundary_test
cd /home/green/energy_dl/nightly
python3 mutation.py -ec settings/boundary_test_elite_plus.json

# 监控进度
screen -r boundary_test

# 分析结果
python3 analysis/analyze_boundary_test.py
```

---

## 总结

### 测试覆盖

| 维度 | 覆盖情况 |
|------|---------|
| **模型类型** | CV×2 (ResNet, DenseNet) + NLP×1 (MRT-OAST) + 基线×1 (MNIST) |
| **任务类型** | 图像分类×2 + 行人重识别×1 + 文本分类×1 |
| **超参数** | epochs, learning_rate, weight_decay, dropout |
| **边界测试** | 最小值、最大值、默认值、零值、极端组合 |
| **配置总数** | 31个（MNIST:7 + ResNet20:8 + DenseNet121:7 + MRT-OAST:9） |

### 时间成本

- **快速验证** (MNIST only): 10.5分钟
- **完整测试**: 14.4小时 (0.6天)

### 价值

1. **验证范围合理性**: 确保变异范围不会导致性能崩溃
2. **量化性能影响**: 精确测量每个参数边界值的影响
3. **指导后续实验**: 为大规模变异实验提供可靠基础
4. **避免浪费**: 避免在不合理范围内进行大量无效实验

---

**文档版本**: 1.0
**创建日期**: 2025-11-10
**配置文件**: settings/boundary_test_elite_plus.json
**预计完成时间**: 14.4小时
