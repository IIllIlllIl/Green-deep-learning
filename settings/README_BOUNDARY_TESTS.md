# 边界测试配置说明

## 配置文件列表

### 1. boundary_test_critical_only.json ⭐ 推荐

**用途**: 精简版边界测试，仅测试上次有问题的边界值

**实验数量**: 5个  
**预计时间**: 2-3小时

**测试内容**:
- ✅ DenseNet121 @ lr=4×default (0.2) - 验证不会像5×那样崩溃
- ✅ MRT-OAST @ lr=0.25×default (0.000025) - 验证比0.2×性能更好
- ✅ MRT-OAST @ dropout=0.4 - 验证比0.5性能更好
- 包含2个baseline作为对比

**运行命令**:
```bash
sudo python3 mutation.py -ec settings/boundary_test_critical_only.json
```

---

### 2. boundary_test_conservative.json

**用途**: 全面的边界测试，测试所有边界值

**实验数量**: 13个  
**预计时间**: 6小时

**测试内容**:
- DenseNet121: 5个实验（baseline + 4个边界）
- MRT-OAST: 5个实验（baseline + 4个边界）
- ResNet20: 3个实验（baseline + 2个边界）

**运行命令**:
```bash
sudo python3 mutation.py -ec settings/boundary_test_conservative.json
```

**注意**: 仅在关键边界测试通过后再运行全面测试

---

## 变异范围 (v2.0)

| 超参数 | 范围 | 说明 |
|--------|------|------|
| Learning Rate | [0.25×, 4×] | 下界↑ 上界↓ |
| Dropout | [0.0, 0.4] | 上界↓ |
| Epochs | [0.5×, 2×] | 不变 |
| Weight Decay | [0.0, 100×] | 不变 |

---

## 上次测试问题总结 (v1.0)

**日期**: 2025-11-11  
**配置**: boundary_test_lr_dropout_focused.json (已归档)

| 问题 | 配置 | 结果 | v2.0修正 |
|------|------|------|---------|
| ❌ 性能崩溃 | DenseNet121 lr=5× (0.25) | Rank@1: 90%→0% | 测试4× (0.2) |
| ⚠️ 性能下降9% | MRT-OAST lr=0.2× (0.00002) | Precision: 98%→89% | 测试0.25× (0.000025) |
| ⚠️ 性能下降12% | MRT-OAST dropout=0.5 | Recall: 89%→77% | 测试0.4 |

---

## 归档配置

已归档的配置存放在 `archive/` 目录，可查看但不推荐使用。

详见: [archive/README.md](archive/README.md)

---

**最后更新**: 2025-11-12  
**详细文档**: [docs/HYPERPARAMETER_RANGES_V2.md](../docs/HYPERPARAMETER_RANGES_V2.md)
