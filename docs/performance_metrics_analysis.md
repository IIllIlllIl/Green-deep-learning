# 12个模型性能度量分析报告

**日期**: 2025-11-09
**作者**: Green
**项目**: Mutation-Based Training Energy Profiler

---

## 📊 概述

本文档分析项目中**12个模型/模型组**的性能度量指标，并探讨是否存在公共的性能度量。

**模型计数说明**:
- pytorch_resnet_cifar10 算作1个模型组（内含4个变体：resnet20/32/44/56，使用相同的度量）
- 其他带多个变体的按实际模型数计算

---

## 🔍 12个模型的性能度量详情

### 1. MRT-OAST

**模型数**: 1

**性能度量**:
- ✅ **Accuracy** (准确率)
- ✅ **Precision** (精确率)
- ✅ **Recall** (召回率)
- ✅ **F1** (F1分数)

**任务类型**: 分类任务（软件缺陷预测）

---

### 2. bug-localization-by-dnn-and-rvsm

**模型数**: 1

**性能度量**:
- ✅ **Top-1** (Top-1准确率)
- ✅ **Top-5** (Top-5准确率)
- ✅ **MAP** (Mean Average Precision)

**任务类型**: 检索/排序任务（缺陷定位）

---

### 3. pytorch_resnet_cifar10

**模型数**: 1组（包含4个变体，共用度量）

**性能度量**:
- ✅ **Test Accuracy** (测试准确率)
- ✅ **Best Val Accuracy** (最佳验证准确率)

**任务类型**: 图像分类（CIFAR-10）

**备注**: resnet20, resnet32, resnet44, resnet56 使用相同的性能度量

---

### 4. VulBERTa (2个模型)

**模型**: mlp, cnn

**性能度量**:
- ✅ **Accuracy** (准确率)
- ✅ **F1** (F1-score)

**任务类型**: 二分类（漏洞检测）

---

### 5. Person_reID_baseline_pytorch (3个模型)

**模型**: densenet121, hrnet18, pcb

**性能度量**:
- ✅ **Rank@1** (Rank-1准确率)
- ✅ **Rank@5** (Rank-5准确率)
- ✅ **mAP** (mean Average Precision)

**任务类型**: 检索任务（行人重识别）

---

### 6. examples (4个模型)

**模型**: mnist_cnn, mnist_rnn, mnist_forward_forward, siamese

**性能度量**:
- ✅ **Test Accuracy** (测试准确率)
- ✅ **Test Loss** (测试损失)

**任务类型**: 图像分类（MNIST）

---

## 📈 性能度量汇总表

| 模型/模型组 | Accuracy | F1 | Precision | Recall | Top-N | Rank@N | mAP | Loss |
|------------|----------|----|-----------| -------|-------|--------|-----|------|
| **MRT-OAST** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **bug-localization** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **pytorch_resnet** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **VulBERTa (×2)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Person_reID (×3)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **examples (×4)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

**度量支持统计**:
- **Accuracy类**: 4/6 仓库 (覆盖9个实际模型)
- **mAP类**: 2/6 仓库 (覆盖4个实际模型)
- **F1类**: 2/6 仓库 (覆盖3个实际模型)
- **Rank@N类**: 1/6 仓库 (覆盖3个实际模型)

---

## 🎯 公共性能度量分析

### ❌ 全局公共度量

**结论**: **不存在适用于所有12个模型的单一公共性能度量**

**原因分析**:

1. **任务类型差异**:
   - **分类任务**: MRT-OAST, pytorch_resnet, VulBERTa (×2), examples (×4) = 8个模型
   - **检索/排序任务**: bug-localization, Person_reID (×3) = 4个模型

2. **度量方式不同**:
   - 分类任务通常使用 Accuracy, F1, Precision, Recall
   - 检索任务通常使用 Rank@N, mAP, Top-N

3. **输出形式差异**:
   - 分类模型输出类别概率分布
   - 检索模型输出排序列表

---

## 📊 按模型种类的公共度量

### 种类1: 分类任务 ✅

**包含模型** (8个):
1. MRT-OAST/default
2. pytorch_resnet_cifar10 组（4个变体）
3. VulBERTa/mlp
4. VulBERTa/cnn
5. examples/mnist_cnn
6. examples/mnist_rnn
7. examples/mnist_forward_forward
8. examples/siamese

**公共度量**: ✅ **Accuracy (准确率)**

**度量覆盖情况**:

| 模型 | Accuracy字段名 | 支持状态 |
|------|---------------|---------|
| MRT-OAST | `accuracy` | ✅ |
| pytorch_resnet | `test_accuracy` | ✅ |
| VulBERTa (×2) | `accuracy` | ✅ |
| examples (×4) | `test_accuracy` | ✅ |

**覆盖率**: **8/8 = 100%** ✅

**备注**: 虽然字段名略有不同（`accuracy` vs `test_accuracy`），但本质相同，都是测试集准确率。

---

### 种类2: 检索/排序任务 ✅

**包含模型** (4个):
1. bug-localization-by-dnn-and-rvsm/default
2. Person_reID_baseline_pytorch/densenet121
3. Person_reID_baseline_pytorch/hrnet18
4. Person_reID_baseline_pytorch/pcb

**公共度量**: ✅ **mAP (Mean Average Precision)**

**度量覆盖情况**:

| 模型 | mAP字段名 | 支持状态 |
|------|----------|---------|
| bug-localization | `map` | ✅ |
| Person_reID (×3) | `map` | ✅ |

**覆盖率**: **4/4 = 100%** ✅

**次要公共度量**:
- **Top-1 / Rank@1**: 两个仓库都支持
- **Top-5 / Rank@5**: 两个仓库都支持

---

## 💡 推荐方案：分层度量策略 ⭐

### 方案概述

根据任务类型使用不同的公共度量：

```json
{
  "classification_models": {
    "primary_metric": "accuracy",
    "task_type": "classification",
    "models": [
      "MRT-OAST",
      "pytorch_resnet_cifar10",
      "VulBERTa/mlp",
      "VulBERTa/cnn",
      "examples/mnist_cnn",
      "examples/mnist_rnn",
      "examples/mnist_forward_forward",
      "examples/siamese"
    ],
    "count": 8,
    "coverage": "100%"
  },
  "retrieval_models": {
    "primary_metric": "mAP",
    "task_type": "retrieval",
    "models": [
      "bug-localization-by-dnn-and-rvsm",
      "Person_reID_baseline_pytorch/densenet121",
      "Person_reID_baseline_pytorch/hrnet18",
      "Person_reID_baseline_pytorch/pcb"
    ],
    "count": 4,
    "coverage": "100%"
  }
}
```

### 优势分析

✅ **完整覆盖**: 每类模型都有100%的公共度量覆盖
✅ **符合最佳实践**: 使用领域内公认的性能指标
✅ **便于比较**: 同类模型间可直接比较
✅ **易于实施**: 字段映射简单，维护成本低

### 劣势分析

⚠️ **跨类比较困难**: 无法直接比较分类模型和检索模型
⚠️ **需要分组处理**: 分析时需要按任务类型分组

---

## 🔧 实施方案

### 步骤1: 配置文件增强

在 `config/models_config.json` 中为每个仓库添加任务类型和主要度量：

```json
{
  "models": {
    "MRT-OAST": {
      "task_type": "classification",
      "primary_metric": "accuracy",
      "metric_mapping": {
        "primary": "accuracy"
      },
      // ... 其他配置
    },
    "pytorch_resnet_cifar10": {
      "task_type": "classification",
      "primary_metric": "test_accuracy",
      "metric_mapping": {
        "primary": "test_accuracy"
      },
      // ... 其他配置
    },
    "bug-localization-by-dnn-and-rvsm": {
      "task_type": "retrieval",
      "primary_metric": "map",
      "metric_mapping": {
        "primary": "map"
      },
      // ... 其他配置
    }
  }
}
```

### 步骤2: 添加统一度量提取方法

在 `mutation.py` 中添加：

```python
def get_primary_metric(self, repo: str, metrics: Dict[str, float]) -> Optional[float]:
    """
    获取主要性能度量

    Args:
        repo: 仓库名称
        metrics: 提取的性能度量字典

    Returns:
        主要性能度量值
    """
    repo_config = self.config["models"][repo]
    primary_metric_name = repo_config.get("primary_metric")

    if primary_metric_name and primary_metric_name in metrics:
        return metrics[primary_metric_name]

    return None

def get_task_type(self, repo: str) -> str:
    """
    获取模型任务类型

    Args:
        repo: 仓库名称

    Returns:
        任务类型 (classification 或 retrieval)
    """
    return self.config["models"][repo].get("task_type", "unknown")
```

### 步骤3: 结果JSON格式增强

修改 `save_results()` 方法，在结果中添加：

```json
{
  "experiment_id": "20251109_123456_repo_model",
  "timestamp": "2025-11-09T12:34:56",
  "repository": "VulBERTa",
  "model": "mlp",
  "task_type": "classification",           // 新增
  "hyperparameters": { ... },
  "performance_metrics": {
    "accuracy": 0.856,
    "f1": 0.843
  },
  "primary_performance_metric": 0.856,     // 新增
  "primary_metric_name": "accuracy",       // 新增
  "energy_metrics": { ... },
  "training_success": true
}
```

---

## 📉 各指标详细说明

### Accuracy (准确率)

- **定义**: 正确预测的样本数 / 总样本数
- **公式**: Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **范围**: 0.0 - 1.0 (或 0% - 100%)
- **适用场景**: 分类任务
- **优点**:
  - ✅ 直观易懂
  - ✅ 计算简单
  - ✅ 广泛使用
- **缺点**:
  - ⚠️ 类别不平衡时可能误导
  - ⚠️ 不能反映各类别的性能差异

### mAP (Mean Average Precision)

- **定义**: 所有查询的平均精度的平均值
- **公式**: mAP = (1/Q) Σ AP(q)，其中Q为查询数
- **范围**: 0.0 - 1.0
- **适用场景**: 检索/排序任务
- **优点**:
  - ✅ 综合评估排序质量
  - ✅ 考虑了排序位置
  - ✅ 信息检索领域标准度量
- **缺点**:
  - ⚠️ 计算复杂
  - ⚠️ 不如Accuracy直观

### F1-score

- **定义**: Precision和Recall的调和平均数
- **公式**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **范围**: 0.0 - 1.0
- **适用场景**: 分类任务（特别是不平衡数据）
- **优点**:
  - ✅ 综合考虑精确率和召回率
  - ✅ 适合不平衡数据集
- **缺点**:
  - ⚠️ 计算相对复杂
  - ⚠️ 不如Accuracy直观

### Rank@N

- **定义**: Top-N结果中包含正确答案的查询比例
- **公式**: Rank@N = (正确答案在前N的查询数) / (总查询数)
- **范围**: 0.0 - 1.0
- **适用场景**: 检索任务
- **优点**:
  - ✅ 直观易理解
  - ✅ 符合用户体验（用户通常只看前N个结果）
- **缺点**:
  - ⚠️ 只考虑Top-N，忽略排序细节
  - ⚠️ 需要选择合适的N值

---

## 🎯 结论与建议

### 主要发现

1. ❌ **不存在全局公共度量**: 12个模型无法用单一指标统一衡量

2. ✅ **存在任务级公共度量**:
   - **分类任务 (8个模型)**: 100% 支持 **Accuracy**
   - **检索任务 (4个模型)**: 100% 支持 **mAP**

3. 📊 **任务分布**:
   - 分类任务: 8/12 = 66.7%
   - 检索任务: 4/12 = 33.3%

### 最终推荐

✅ **采用分层度量策略**

**实施要点**:
1. 为每个模型标注任务类型（classification / retrieval）
2. 定义各任务类型的主要度量（Accuracy / mAP）
3. 在实验结果中添加统一的 `primary_performance_metric` 字段
4. 分析时按任务类型分组，使用相应的主要度量

**预期效果**:
- ✅ 同类模型可直接比较
- ✅ 保持度量的准确性和可解释性
- ✅ 符合机器学习领域最佳实践
- ✅ 易于实施和维护

---

## 📚 相关文档

- [性能度量快速参考](performance_metrics_summary.md)
- [模型配置文件](../config/models_config.json)
- [超参数支持矩阵](hyperparameter_support_matrix.md)
- [主项目文档](../README.md)

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**最后更新**: 2025-11-09
