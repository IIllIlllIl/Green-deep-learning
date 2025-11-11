# 边界值测试 - 快速启动指南

**目标**: 验证超参数范围是否合理（不影响模型性能）
**方法**: 测试边界值（最小值、最大值）对性能的影响
**原则**: 边界值性能下降应 < 10%

---

## 📋 配置概览

| 模型 | 测试配置数 | 预计时长 | 测试参数 |
|------|-----------|---------|---------|
| examples/mnist | 7 | 10.5分钟 | epochs, learning_rate |
| pytorch_resnet_cifar10/resnet20 | 8 | 2.7小时 | epochs, learning_rate, weight_decay |
| Person_reID_baseline_pytorch/densenet121 | 7 | 4.4小时 | epochs, learning_rate, dropout |
| MRT-OAST/default | 9 | 6.3小时 | epochs, learning_rate, dropout, weight_decay |
| **总计** | **31** | **14.4小时** | - |

---

## 🚀 快速启动

### 选项1: 完整测试（推荐）

```bash
# 1. 启动screen会话
screen -S boundary_test

# 2. 运行测试
cd /home/green/energy_dl/nightly
python3 mutation.py -ec settings/boundary_test_elite_plus.json

# 3. 分离screen (Ctrl+A 然后按 D)

# 4. 重新连接查看进度
screen -r boundary_test

# 5. 测试完成后分析结果
python3 analysis/analyze_boundary_test.py
```

### 选项2: 仅测试MNIST（快速验证）

```bash
# 手动运行MNIST的7个配置
python3 mutation.py -r examples -m mnist \
    --mutate epochs,learning_rate -n 1 -g performance

# 然后手动测试边界值...
```

---

## 📊 性能判断标准

| 性能下降 | 状态 | 说明 | 行动 |
|---------|------|------|------|
| < 5% | ✅ 优秀 | 范围合理 | 保持当前范围 |
| 5-10% | ⚠️ 警告 | 范围略宽 | 考虑收窄 |
| > 10% | ❌ 不可接受 | 范围过宽 | **必须收窄** |

**示例**:
- 基线性能: 91.45% Acc
- 边界值性能: 88.12% Acc
- 性能下降: (88.12 - 91.45) / 91.45 = -3.64% ✅ 合格

---

## 📈 结果分析

测试完成后运行：

```bash
python3 analysis/analyze_boundary_test.py
```

输出示例：

```
================================================================================
Model: pytorch_resnet_cifar10/resnet20
================================================================================

📊 Baseline Configuration:
   Hyperparameters: {'epochs': 200, 'learning_rate': 0.1, 'weight_decay': 0.0001}
   Performance: 91.45%
   Duration: 1200.5s

📈 Boundary Value Analysis:
Status   Performance   Change     Hyperparameters
--------------------------------------------------------------------------------
✅       89.12%        -2.33%     {'epochs': 100, 'learning_rate': 0.1, ...}
✅       92.01%        +0.56%     {'epochs': 400, 'learning_rate': 0.1, ...}
⚠️       85.23%        -6.22%     {'epochs': 200, 'learning_rate': 0.01, ...}
❌       78.45%        -13.00%    {'epochs': 200, 'learning_rate': 1.0, ...}
...

📋 Summary:
   ✅ Good/OK (drop < 5%): 5/7
   ⚠️  Warning (drop 5-10%): 1/7
   ❌ Bad (drop > 10%): 1/7

   ❌ RECOMMENDATION: Narrow the range for parameters causing >10% performance drop
```

---

## 🔧 范围调整示例

### 场景: Learning Rate上界过宽

**问题**: `learning_rate: 1.0` 导致性能下降13% ❌

**调整**:

编辑 `config/models_config.json`:

```json
// 原配置
"learning_rate": {
  "range": [0.01, 1.0],  // 上界过高
  ...
}

// 调整后（收窄至5x）
"learning_rate": {
  "range": [0.01, 0.5],  // 收窄上界
  ...
}
```

**重新测试**: 只测试受影响的配置

```bash
# 手动测试 learning_rate=0.5
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 \
    -mt learning_rate -n 1 -g performance
# 检查性能是否 > 基线 - 10%
```

---

## 📝 测试配置详情

所有31个测试配置详见:
- **配置文件**: `settings/boundary_test_elite_plus.json`
- **详细文档**: `docs/boundary_test_strategy.md`

### MNIST测试配置（7个）

1. Default: epochs=10, lr=0.01
2. Min epochs: epochs=5, lr=0.01
3. Max epochs: epochs=20, lr=0.01
4. Min LR: epochs=10, lr=0.001
5. Max LR: epochs=10, lr=0.1
6. Min+Min: epochs=5, lr=0.001
7. Max+Max: epochs=20, lr=0.1

### ResNet20测试配置（8个）

1. Default: epochs=200, lr=0.1, wd=0.0001
2. Min epochs: epochs=100
3. Max epochs: epochs=400
4. Min LR: lr=0.01
5. Max LR: lr=1.0
6. Min WD: wd=0.00001
7. Max WD: wd=0.01
8. Zero WD: wd=0.0

（详见配置文件...）

---

## ⏱️ 监控命令

```bash
# 查看进程
ps aux | grep mutation.py

# 查看已完成配置数
find results/ -name "*.json" -mmin -300 | wc -l

# 查看最新日志
tail -f results/training_*.log

# 实时监控结果数量
watch -n 60 'find results/ -name "*.json" -mmin -300 | wc -l'
```

---

## ✅ 完成检查清单

- [ ] 31个配置全部完成（检查: `ls results/*.json | wc -l`）
- [ ] 所有训练成功（检查日志中是否有"Training completed successfully"）
- [ ] 运行分析脚本: `python3 analysis/analyze_boundary_test.py`
- [ ] 查看每个模型的性能下降情况
- [ ] 如有❌（下降>10%），调整 `config/models_config.json`
- [ ] 如有调整，重新测试受影响的边界值
- [ ] 所有边界值下降<10%后，可以开始变异实验

---

## 🎯 预期结果示例

### 场景1: 范围合理 ✅

```
examples/mnist:          所有边界值 < 5% drop → ✅ 保持范围
resnet20:                所有边界值 < 5% drop → ✅ 保持范围
densenet121:             所有边界值 < 8% drop → ✅ 范围合理（可选收窄）
MRT-OAST:                所有边界值 < 6% drop → ✅ 范围合理
```

**结论**: 可以直接开始变异实验

---

### 场景2: 部分范围需调整 ⚠️

```
examples/mnist:          所有边界值 < 5% drop → ✅
resnet20:                lr=1.0 导致 13% drop → ❌ 需调整
densenet121:             lr=0.5 导致 11% drop → ❌ 需调整
MRT-OAST:                所有边界值 < 5% drop → ✅
```

**行动**:
1. 调整 resnet20 的 learning_rate 上界: `1.0 → 0.5`
2. 调整 densenet121 的 learning_rate 上界: `0.5 → 0.3`
3. 重新测试受影响的2个配置
4. 确认调整后性能下降 < 10%
5. 开始变异实验

---

## 📚 相关文档

- **配置文件**: settings/boundary_test_elite_plus.json
- **详细策略**: docs/boundary_test_strategy.md
- **分析脚本**: analysis/analyze_boundary_test.py
- **模型配置**: config/models_config.json

---

**创建日期**: 2025-11-10
**预计完成**: 14.4小时
**下一步**: 根据测试结果决定是否调整范围，然后开始变异实验
