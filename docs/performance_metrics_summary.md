# 性能度量快速参考

**日期**: 2025-11-09

---

## 📊 12个模型的性能度量总览

| # | 仓库 | 模型名称 | 任务类型 | 主要度量 | 次要度量 |
|---|------|---------|---------|---------|---------|
| 1 | MRT-OAST | default | 分类 | **Accuracy** | Precision, Recall, F1 |
| 2 | bug-localization | default | 检索 | **MAP** | Top-1, Top-5 |
| 3 | pytorch_resnet | resnet20 | 分类 | **Test Accuracy** | Best Val Accuracy |
| 4 | pytorch_resnet | resnet32 | 分类 | **Test Accuracy** | Best Val Accuracy |
| 5 | pytorch_resnet | resnet44 | 分类 | **Test Accuracy** | Best Val Accuracy |
| 6 | pytorch_resnet | resnet56 | 分类 | **Test Accuracy** | Best Val Accuracy |
| 7 | VulBERTa | mlp | 分类 | **Accuracy** | F1-score |
| 8 | VulBERTa | cnn | 分类 | **Accuracy** | F1-score |
| 9 | Person_reID | densenet121 | 检索 | **mAP** | Rank@1, Rank@5 |
| 10 | Person_reID | hrnet18 | 检索 | **mAP** | Rank@1, Rank@5 |
| 11 | Person_reID | pcb | 检索 | **mAP** | Rank@1, Rank@5 |
| 12 | examples | mnist_cnn | 分类 | **Test Accuracy** | Test Loss |
| 13 | examples | mnist_rnn | 分类 | **Test Accuracy** | Test Loss |
| 14 | examples | mnist_forward_forward | 分类 | **Test Accuracy** | Test Loss |
| 15 | examples | siamese | 分类 | **Test Accuracy** | Test Loss |

**注意**: 实际上是 **15个模型**（不是12个），因为有些仓库包含多个模型。

---

## 🎯 公共性能度量

### ❌ 全局公共度量
**不存在**适用于所有15个模型的单一公共度量。

### ✅ 按任务类型的公共度量

#### 分类任务 (11个模型)
**公共度量**: **Accuracy (准确率)**

包含模型：
- MRT-OAST/default
- pytorch_resnet_cifar10: resnet20, resnet32, resnet44, resnet56
- VulBERTa: mlp, cnn
- examples: mnist_cnn, mnist_rnn, mnist_forward_forward, siamese

**覆盖率**: 11/15 = 73.3%

---

#### 检索任务 (4个模型)
**公共度量**: **mAP (Mean Average Precision)**

包含模型：
- bug-localization-by-dnn-and-rvsm/default
- Person_reID_baseline_pytorch: densenet121, hrnet18, pcb

**覆盖率**: 4/15 = 26.7%

---

## 📈 度量统计

| 度量类型 | 模型数量 | 占比 | 具体指标 |
|---------|---------|------|---------|
| **Accuracy类** | **11** | **73%** | accuracy, test_accuracy |
| **mAP类** | **4** | **27%** | MAP, mAP |
| **F1类** | 2 | 13% | F1, F1-score |
| **Rank@N类** | 3 | 20% | Rank@1, Rank@5 |
| **Top-N类** | 1 | 7% | Top-1, Top-5 |
| **Loss类** | 4 | 27% | test_loss |
| **Precision** | 1 | 7% | precision |
| **Recall** | 1 | 7% | recall |

---

## 💡 推荐使用策略

### 策略：分层度量法 ⭐

```python
# 根据任务类型选择主要度量
def get_primary_metric(task_type):
    if task_type == "classification":
        return "accuracy"  # 适用于11个模型
    elif task_type == "retrieval":
        return "mAP"       # 适用于4个模型
```

### 任务类型分布

```
分类任务: ████████████████████████████████████████ 73% (11个模型)
检索任务: ███████████ 27% (4个模型)
```

---

## 🔍 详细配置

各模型的性能度量在配置文件中的定义：

### 分类任务组

```json
{
  "MRT-OAST": {
    "performance_metrics": {
      "log_patterns": {
        "accuracy": "Accuracy[:\\s]+([0-9.]+)",
        "precision": "Precision[:\\s]+([0-9.]+)",
        "recall": "Recall[:\\s]+([0-9.]+)",
        "f1": "F1[:\\s]+([0-9.]+)"
      }
    }
  },
  "pytorch_resnet_cifar10": {
    "performance_metrics": {
      "log_patterns": {
        "test_accuracy": "测试准确率[:\\s]+([0-9.]+)",
        "best_val_accuracy": "最佳验证准确率[:\\s]+([0-9.]+)"
      }
    }
  },
  "VulBERTa": {
    "performance_metrics": {
      "log_patterns": {
        "accuracy": "Accuracy[:\\s]+([0-9.]+)",
        "f1": "F1[:\\s-]+score[:\\s]+([0-9.]+)"
      }
    }
  },
  "examples": {
    "performance_metrics": {
      "log_patterns": {
        "test_accuracy": "Test.*Accuracy[:\\s]+([0-9.]+)",
        "test_loss": "Test.*Loss[:\\s]+([0-9.]+)"
      }
    }
  }
}
```

### 检索任务组

```json
{
  "bug-localization-by-dnn-and-rvsm": {
    "performance_metrics": {
      "log_patterns": {
        "top1": "Top-1[:\\s@]+([0-9.]+)",
        "top5": "Top-5[:\\s@]+([0-9.]+)",
        "map": "MAP[:\\s@]+([0-9.]+)"
      }
    }
  },
  "Person_reID_baseline_pytorch": {
    "performance_metrics": {
      "log_patterns": {
        "rank1": "Rank@1[:\\s]+([0-9.]+)",
        "rank5": "Rank@5[:\\s]+([0-9.]+)",
        "map": "mAP[:\\s]+([0-9.]+)"
      }
    }
  }
}
```

---

## 📝 结论

1. **不存在全局公共度量**：15个模型无法使用单一度量指标

2. **存在任务级公共度量**：
   - 分类任务：**Accuracy** (11个模型，73%)
   - 检索任务：**mAP** (4个模型，27%)

3. **推荐方案**：采用分层度量策略
   - 为每个仓库定义任务类型和主要度量
   - 在结果JSON中统一添加 `primary_metric` 字段
   - 分析时根据任务类型分组比较

4. **实施建议**：参考 [performance_metrics_analysis.md](performance_metrics_analysis.md) 的详细方案

---

**相关文档**:
- [详细分析报告](performance_metrics_analysis.md)
- [模型配置文件](../config/models_config.json)
