# 边界测试快速参考

## 📊 当前状态

**最新配置**: `settings/boundary_test_conservative.json`  
**状态**: ⏳ 待验证  
**版本**: v2.0 保守范围

---

## 🎯 新的统一变异范围

| 超参数 | 统一表达式 | 变更 |
|--------|-----------|------|
| **Learning Rate** | `[0.25×default, 4×default]` | ⬆️ 下界 0.2→0.25, ⬇️ 上界 5→4 |
| **Dropout** | `[0.0, 0.4]` | ⬇️ 上界 0.5→0.4 |
| **Epochs** | `[0.5×default, 2×default]` | 不变 |
| **Weight Decay** | `[0.0, 100×default]` | 不变 |

---

## 🚀 立即运行验证

```bash
# 进入项目目录
cd /home/green/energy_dl/nightly

# 运行保守边界测试
sudo python3 mutation.py -ec settings/boundary_test_conservative.json

# 或使用screen
screen -S boundary_test
sudo python3 mutation.py -ec settings/boundary_test_conservative.json
# Ctrl+A, D (detach)
```

**预计时间**: 约6小时  
**实验数量**: 13个

---

## ✅ 验证标准

### 必须通过的关键测试

1. **DenseNet121 @ LR=4×default (0.2)**
   - ✅ 训练成功完成（不崩溃）
   - ✅ Rank@1 > 85%
   - ✅ mAP > 65%

2. **MRT-OAST @ LR=0.25×default (0.000025)**
   - ✅ Precision > 85%
   - ✅ Recall > 70%

3. **所有模型 @ Dropout=0.4**
   - ✅ 性能下降 < 10%

---

## 📁 文件位置

```
settings/
├── boundary_test_conservative.json          ⭐ 新配置
├── archive/
│   ├── boundary_test_lr_dropout_focused_2025-11-11.json  # 已归档
│   └── README.md
└── [其他配置文件...]

docs/
├── HYPERPARAMETER_RANGES_V2.md              ⭐ 详细范围定义
└── PARALLEL_TRAINING_DESIGN.md              # 并行训练方案
```

---

## 📈 各模型具体范围

| 模型 | Default LR | LR范围 | Dropout范围 |
|------|-----------|--------|------------|
| DenseNet121 | 0.05 | [0.0125, 0.2] | [0.0, 0.4] |
| MRT-OAST | 0.0001 | [0.000025, 0.0004] | [0.0, 0.2]* |
| ResNet20 | 0.1 | [0.025, 0.4] | N/A |

*MRT-OAST的Dropout上界不超过默认值0.2

---

## 🔍 查看结果

### 实时监控

```bash
# 查看最新结果文件
ls -lt results/*.json | head -5

# 统计成功/失败
grep -l '"training_success": true' results/*.json | wc -l
grep -l '"training_success": false' results/*.json | wc -l
```

### 分析性能

```bash
# 提取DenseNet121的Rank@1
jq -r 'select(.repository=="Person_reID_baseline_pytorch") | 
  "\(.hyperparameters.learning_rate) -> Rank@1: \(.performance_metrics.rank1)"' \
  results/2025*.json
```

---

## ⚠️ 已知问题（v1.0范围）

| 问题 | 模型 | 配置 | 结果 |
|------|------|------|------|
| ❌ 训练崩溃 | DenseNet121 | LR=5×default (0.25) | Rank@1: 90%→0% |
| ⚠️ 性能下降9% | MRT-OAST | LR=0.2×default | Precision: 98%→89% |
| ⚠️ 性能下降12% | MRT-OAST | Dropout=0.5 | Recall: 89%→77% |

**解决**: 已在v2.0中调整范围

---

## 📞 下一步

### 验证通过后

1. ✅ 更新 `config/models_config.json` 中的范围
2. ✅ 创建完整变异实验配置
3. ✅ 开始大规模能耗实验

### 验证未通过

- 如果DenseNet121 @ LR=4×仍崩溃
  → 进一步降低上界到3×或2×
  
- 如果MRT-OAST @ LR=0.25×性能过差
  → 提高下界到0.5×

---

**最后更新**: 2025-11-12  
**作者**: Claude Code  
**参考**: `docs/HYPERPARAMETER_RANGES_V2.md`
