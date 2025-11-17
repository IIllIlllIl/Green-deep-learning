# 性能度量分析总结

**生成时间**: 2025-11-09

---

## ❓ 核心问题

**项目中的12个模型提供了哪些性能度量？是否存在公共的性能度量？**

---

## ✅ 快速答案

### 1. 是否存在公共的性能度量？

**全局公共度量**: ❌ **不存在**

**按任务类型的公共度量**: ✅ **存在**

### 2. 具体的公共度量是什么？

#### 分类任务（8个模型）
- **公共度量**: **Accuracy（准确率）**
- **覆盖率**: 100%
- **包含模型**:
  - MRT-OAST
  - pytorch_resnet_cifar10（4个变体）
  - VulBERTa（2个模型）
  - examples（4个模型）

#### 检索任务（4个模型）
- **公共度量**: **mAP (Mean Average Precision)**
- **覆盖率**: 100%
- **包含模型**:
  - bug-localization-by-dnn-and-rvsm
  - Person_reID_baseline_pytorch（3个模型）

---

## 📊 详细分布

### 任务类型分布

```
分类任务: ████████████████████████████████ 66.7% (8个模型)
检索任务: ████████████████ 33.3% (4个模型)
```

### 性能度量支持情况

| 度量类型 | 支持模型数 | 占比 |
|---------|-----------|------|
| **Accuracy** | 8 | 66.7% |
| **mAP** | 4 | 33.3% |
| F1-score | 3 | 25.0% |
| Rank@N | 3 | 25.0% |
| Loss | 4 | 33.3% |
| Precision | 1 | 8.3% |
| Recall | 1 | 8.3% |
| Top-N | 1 | 8.3% |

---

## 💡 推荐方案

### **采用分层度量策略**

```
┌─────────────────────────────────────────┐
│           12个模型                       │
├─────────────────┬───────────────────────┤
│  分类任务(8个)   │   检索任务(4个)        │
├─────────────────┼───────────────────────┤
│  主要度量:       │   主要度量:            │
│  Accuracy       │   mAP                 │
│  覆盖率: 100%   │   覆盖率: 100%        │
└─────────────────┴───────────────────────┘
```

### 实施步骤

1. **配置文件增强**: 为每个仓库添加 `task_type` 和 `primary_metric` 字段
2. **代码修改**: 在 `mutation.py` 中添加统一的度量提取方法
3. **结果增强**: 在实验结果JSON中添加 `primary_performance_metric` 字段

---

## 📋 12个模型清单

| # | 仓库 | 模型 | 任务类型 | 主要度量 |
|---|------|------|---------|---------|
| 1 | MRT-OAST | default | 分类 | Accuracy |
| 2 | bug-localization | default | 检索 | mAP |
| 3 | pytorch_resnet | resnet20/32/44/56 | 分类 | Test Accuracy |
| 4 | VulBERTa | mlp | 分类 | Accuracy |
| 5 | VulBERTa | cnn | 分类 | Accuracy |
| 6 | Person_reID | densenet121 | 检索 | mAP |
| 7 | Person_reID | hrnet18 | 检索 | mAP |
| 8 | Person_reID | pcb | 检索 | mAP |
| 9 | examples | mnist_cnn | 分类 | Test Accuracy |
| 10 | examples | mnist_rnn | 分类 | Test Accuracy |
| 11 | examples | mnist_forward_forward | 分类 | Test Accuracy |
| 12 | examples | siamese | 分类 | Test Accuracy |

---

## 📚 详细文档

完整分析请参考：
- **详细报告**: [performance_metrics_analysis.md](performance_metrics_analysis.md)
- **快速参考**: [performance_metrics_summary.md](performance_metrics_summary.md)

---

**结论**: 虽然不存在适用于所有12个模型的单一公共度量，但在各自的任务类型内，所有模型都支持公共的性能度量（分类任务用Accuracy，检索任务用mAP），覆盖率均为100%。
