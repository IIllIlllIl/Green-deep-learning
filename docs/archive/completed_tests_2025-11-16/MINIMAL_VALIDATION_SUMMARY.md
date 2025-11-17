# 最小验证实验配置完成总结

**配置文件**: `settings/minimal_validation.json`
**设计文档**: `docs/MINIMAL_VALIDATION_DESIGN.md`
**完成时间**: 2025-11-15 16:45

---

## ✅ 配置完成

### 实验规模

- **总实验数**: 14个
- **测试类型**: Learning Rate边界（6个）+ Dropout边界（2个）+ Weight Decay边界（6个）
- **覆盖模型**: DenseNet121, ResNet20, MRT-OAST, VulBERTa

### 实验分组

#### Learning Rate边界测试（6个）

1. **DenseNet121** lr=0.025 (0.5×下界)
2. **DenseNet121** lr=0.1 (2×上界) - 验证不崩溃
3. **ResNet20** lr=0.05 (0.5×下界)
4. **ResNet20** lr=0.2 (2×上界)
5. **MRT-OAST** lr=0.00005 (0.5×下界)
6. **MRT-OAST** lr=0.0002 (2×上界)

#### Dropout边界测试（2个）

7. **MRT-OAST** dropout=0.0 (下界) - 验证无dropout
8. **MRT-OAST** dropout=0.4 (上界) - 验证最大dropout

#### Weight Decay边界测试（6个）

9. **ResNet20** wd=0.00001 (下界)
10. **ResNet20** wd=0.01 (上界)
11. **MRT-OAST** wd=0.00001 (下界) - 验证从0改为非零
12. **MRT-OAST** wd=0.01 (上界)
13. **VulBERTa** wd=0.00001 (下界)
14. **VulBERTa** wd=0.01 (上界)

---

## 🎯 验证目标

### 安全性（必须满足）

- ✅ DenseNet121 lr=0.1 不崩溃（之前0.2崩溃）
- ✅ 所有实验训练成功完成
- ✅ Weight Decay从0改为非零不破坏训练

### 有效性（期望满足）

**Learning Rate**:
- DenseNet121: 性能下降 < 5%
- ResNet20: 性能下降 < 1%
- MRT-OAST: 下界-5%，上界+3~4%

**Dropout**:
- MRT-OAST: dropout=0.0 预期+5~6%（基于boundary_test_v2结果）
- MRT-OAST: dropout=0.4 预期±0%（与默认0.2相比）

**Weight Decay**:
- ResNet20: 下界-0.2%，上界-2~4%
- MRT-OAST: 下界±0%，上界-5~8%
- VulBERTa: 下界±0%，上界-5~10%

---

## ⏱️ 预计时间

| 项目 | 时长 |
|------|------|
| 训练时间 | 338分钟 (5.63小时) |
| 间隔时间 | 16分钟 (14个实验间隔) |
| 系统开销 | 10分钟 |
| **总计** | **364分钟 (6.1小时)** |

---

## 📝 运行命令

```bash
# 方式1: 直接运行
python mutation.py -ec settings/minimal_validation.json

# 方式2: 在screen中运行（推荐）
screen -S minimal_val
python mutation.py -ec settings/minimal_validation.json
# Ctrl+A D 分离会话
```

---

## 📊 关键对比

### 与boundary_test_v2的区别

| 项目 | boundary_test_v2 | minimal_validation |
|------|------------------|-------------------|
| LR范围 | [0.25×, 4×] | [0.5×, 2×] ✅ 更安全 |
| 测试点 | 边界极端值 | 新范围边界值 |
| DenseNet121 | lr=0.2 **崩溃** | lr=0.1 预期安全 |
| MRT-OAST | lr=4× 最优 | lr=2× 次优但可接受 |
| Weight Decay | 未测试 | 完整边界测试 ✅ |
| Dropout | 部分测试 | MRT-OAST边界测试 ✅ |
| 总实验数 | 12个 | 14个 |
| 总时长 | 5.46小时 | 6.1小时 |

---

## 🔍 关键验证点

1. **DenseNet121崩溃问题修复**
   - 之前: lr=0.2 (4×) → mAP 0.17% ❌ 崩溃
   - 现在: lr=0.1 (2×) → 预期 mAP ≥ 71% ✅

2. **MRT-OAST Dropout边界验证**
   - dropout=0.0: 预期F1 +5~6%（基于boundary_test_v2: +5.81%）
   - dropout=0.4: 预期F1 与默认值相近
   - 验证新范围[0.0, 0.4]的安全性和有效性

3. **Weight Decay非零影响**
   - MRT-OAST/VulBERTa default=0
   - 测试 wd=0.00001 vs wd=0
   - 验证100%对数采样的实际效果

4. **新范围安全性**
   - 所有模型在新范围内不崩溃
   - 性能下降在可接受范围内（<5%）

---

## 📂 输出文件

运行完成后生成：

```
results/run_YYYYMMDD_HHMMSS/
├── 14个实验目录（每个包含training.log, experiment.json, energy/）
└── summary.csv
```

**关键数据**:
- 性能指标（mAP, Accuracy, F1等）
- 能耗数据（CPU/GPU能耗、温度、利用率）
- 训练时长
- 超参数配置

---

## ✅ 配置检查清单

- [x] 14个实验配置完整
- [x] 所有超参数正确（epochs, lr, wd, dropout, seed）
- [x] Learning Rate边界：[0.5×, 2×]
- [x] Dropout边界：[0.0, 0.4]
- [x] Weight Decay边界：[0.00001, 0.01]
- [x] Governor设置：performance
- [x] Max retries: 1
- [x] 设计文档完整
- [x] 预期结果定义清晰

---

**状态**: ✅ **配置完成，等待运行**

**下一步**: 用户决定何时运行实验（约6.1小时）
