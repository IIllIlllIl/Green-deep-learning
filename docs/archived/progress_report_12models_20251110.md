# 12模型快速测试实时进度报告
**检查时间**: 2025-11-10 16:34
**实验配置**: settings/quick_test_12models.json
**开始时间**: 2025-11-10 16:03

---

## 总体进度

### ✅ 已完成: 5 / 12 模型 (41.7%)

| 序号 | 模型 | 时长 | CPU能耗 | GPU能耗 | 状态 |
|------|------|------|---------|---------|------|
| 1 | examples/mnist | 19.5s | 0.62 kJ | 1.43 kJ | ✅ |
| 2 | MRT-OAST/default | 164.8s (2.7分钟) | 4.95 kJ | 37.47 kJ | ✅ |
| 3 | bug-localization-by-dnn-and-rvsm/default | 905.6s (15.1分钟) | 60.74 kJ | 23.13 kJ | ✅ |
| 4 | VulBERTa/mlp | 380.1s (6.3分钟) | 12.05 kJ | 86.73 kJ | ✅ |
| 5 | VulBERTa/cnn | 3.7s | 0.08 kJ | 0.07 kJ | ✅ |

**小计**:
- 总时长: 24.6分钟
- CPU总能耗: 78.44 kJ
- GPU总能耗: 148.83 kJ
- 总能耗: 227.27 kJ

---

### 🔄 当前训练中: 1 模型

**6. Person_reID_baseline_pytorch/densenet121**
- 开始时间: 16:32
- 状态: 训练中 (多进程)
- 预计时长: ~5-10分钟 (epoch=1)

---

### ⏳ 待训练: 6 模型

7. Person_reID_baseline_pytorch/hrnet18
8. Person_reID_baseline_pytorch/pcb
9. pytorch_resnet_cifar10/resnet20
10. pytorch_resnet_cifar10/resnet32
11. pytorch_resnet_cifar10/resnet44
12. pytorch_resnet_cifar10/resnet56

---

## 性能亮点

### 最快模型
1. **VulBERTa/cnn**: 3.7秒 ⚡
2. **examples/mnist**: 19.5秒
3. **MRT-OAST/default**: 2.7分钟

### 最慢模型
1. **bug-localization-by-dnn-and-rvsm**: 15.1分钟 (max_iter=100)

### 能耗效率
| 模型 | 总能耗 | 效率 (J/s) |
|------|--------|-----------|
| VulBERTa/cnn | 0.15 kJ | 40.5 J/s |
| examples/mnist | 2.05 kJ | 105.1 J/s |
| MRT-OAST/default | 42.42 kJ | 257.4 J/s |
| VulBERTa/mlp | 98.78 kJ | 259.9 J/s |
| bug-localization | 83.87 kJ | 92.6 J/s |

---

## 时间预测

### 基于已完成模型的平均速度

- **已用时间**: 24.6分钟
- **平均每模型**: 4.9分钟
- **预计剩余**: 34.4分钟 (7模型)
- **预计总时长**: ~59分钟
- **预计完成时间**: 17:02

### 考虑模型差异的估算

ResNet系列模型通常较快（epoch=1约30-60秒），Person_reID模型较慢（约5-10分钟）：

- Person_reID (hrnet18, pcb): 2×8分钟 = 16分钟
- ResNet (20/32/44/56): 4×1分钟 = 4分钟
- **调整后预计剩余**: ~25分钟
- **调整后预计完成时间**: 16:59

---

## 详细结果

### 1. examples/mnist ✅
- 训练时长: 19.5秒
- 测试准确率: 93.0%
- CPU能耗: 0.62 kJ
- GPU能耗: 1.43 kJ
- 评价: 最快的模型，非常适合快速验证

### 2. MRT-OAST/default ✅
- 训练时长: 164.8秒
- CPU能耗: 4.95 kJ
- GPU能耗: 37.47 kJ
- 评价: 代码克隆检测，训练稳定

### 3. bug-localization-by-dnn-and-rvsm/default ✅
- 训练时长: 905.6秒 (15.1分钟)
- CPU能耗: 60.74 kJ (最高)
- GPU能耗: 23.13 kJ
- max_iter: 100 (原配置10000，已调整)
- 评价: CPU密集型，时间较长

### 4. VulBERTa/mlp ✅
- 训练时长: 380.1秒 (6.3分钟)
- CPU能耗: 12.05 kJ
- GPU能耗: 86.73 kJ (最高)
- 评价: GPU密集型，能耗高

### 5. VulBERTa/cnn ✅
- 训练时长: 3.7秒 ⚡
- CPU能耗: 0.08 kJ
- GPU能耗: 0.07 kJ
- 评价: **最快的模型！** 可能训练配置极简

---

## 观察和发现

### 1. VulBERTa/cnn异常快速
- 仅3.7秒完成，需要验证是否正常训练
- 可能原因：数据集很小，或训练提前终止
- 建议：检查日志确认训练是否完整

### 2. CPU vs GPU能耗
- bug-localization主要消耗CPU (60.74 kJ vs 23.13 kJ)
- VulBERTa/mlp主要消耗GPU (12.05 kJ vs 86.73 kJ)
- 不同任务类型的能耗特征明显不同

### 3. epoch=1的效果
- 大部分模型能在10分钟内完成
- 适合快速验证和初步能耗评估
- 对于后续变异实验，可以考虑更多epochs获得更准确的能耗数据

---

## 下一步行动

### 1. 继续监控
等待剩余7个模型训练完成，预计16:59完成。

### 2. 验证VulBERTa/cnn
检查训练日志，确认3.7秒的训练是否正常：
```bash
tail -100 results/training_VulBERTa_cnn_20251110_163137.log
```

### 3. 完成后分析
- 生成完整的12模型性能-能耗对比报告
- 识别最节能和最耗能的模型
- 为后续变异实验提供基线数据

---

## 实时监控命令

```bash
# 检查进度
/tmp/check_progress.sh

# 查看当前训练日志
tail -f results/training_*20251110_16*.log | tail -20

# 统计完成数量
ls -1 results/*20251110_16*.json | wc -l
```

---

**更新**: 此报告将在训练完成后生成最终版本。
