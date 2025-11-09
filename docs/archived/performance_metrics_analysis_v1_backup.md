# 性能度量分析报告

**日期**: 2025-11-09
**作者**: Green
**项目**: Mutation-Based Training Energy Profiler

---

## 📊 概述

本文档分析项目中12个模型提供的性能度量指标，并探讨是否存在公共的性能度量。

---

## 🔍 各模型性能度量详情

### 1. MRT-OAST (1个模型)

**模型**: default

**性能度量**:
- ✅ **Accuracy** (准确率)
- ✅ **Precision** (精确率)
- ✅ **Recall** (召回率)
- ✅ **F1** (F1分数)

**任务类型**: 分类任务（软件缺陷预测）

---

### 2. bug-localization-by-dnn-and-rvsm (1个模型)

**模型**: default

**性能度量**:
- ✅ **Top-1** (Top-1准确率)
- ✅ **Top-5** (Top-5准确率)
- ✅ **MAP** (Mean Average Precision)

**任务类型**: 排序/检索任务（缺陷定位）

---

### 3. pytorch_resnet_cifar10 (4个模型)

**模型**: resnet20, resnet32, resnet44, resnet56

**性能度量**:
- ✅ **Test Accuracy** (测试准确率)
- ✅ **Best Val Accuracy** (最佳验证准确率)

**任务类型**: 图像分类（CIFAR-10）

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

| 仓库 | 模型数 | Accuracy | F1 | Precision | Recall | Top-N | Rank@N | mAP | Loss |
|------|--------|----------|----|-----------| -------|-------|--------|-----|------|
| **MRT-OAST** | 1 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **bug-localization** | 1 | ❌ | ❌ | ❌ | ❌ | ✅ (Top-1, Top-5) | ❌ | ✅ | ❌ |
| **pytorch_resnet** | 4 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **VulBERTa** | 2 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Person_reID** | 3 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (Rank@1, Rank@5) | ✅ | ❌ |
| **examples** | 4 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **总计** | **12** | **9** | **2** | **1** | **1** | **1** | **3** | **2** | **1** |

---

## 🎯 公共性能度量分析

### 全局公共度量

**结论**: ❌ **不存在适用于所有12个模型的单一公共性能度量**

**原因分析**:
1. **任务类型差异**:
   - 分类任务 (MRT-OAST, pytorch_resnet, VulBERTa, examples)
   - 检索/排序任务 (bug-localization, Person_reID)

2. **度量方式不同**:
   - 分类任务通常使用 Accuracy, F1, Precision, Recall
   - 检索任务通常使用 Rank@N, mAP, Top-N

3. **输出形式差异**:
   - 有些模型输出概率分布（分类）
   - 有些模型输出排序列表（检索）

---

## 📊 按模型种类的公共度量

### 种类1: 分类任务模型 (9个模型)

**包含**:
- MRT-OAST (1个): default
- pytorch_resnet_cifar10 (4个): resnet20, resnet32, resnet44, resnet56
- VulBERTa (2个): mlp, cnn
- examples (4个): mnist_cnn, mnist_rnn, mnist_forward_forward, siamese

**公共度量**: ✅ **Accuracy (准确率)**

**度量情况**:
| 仓库 | 模型数 | Accuracy度量 |
|------|--------|--------------|
| MRT-OAST | 1 | ✅ accuracy |
| pytorch_resnet | 4 | ✅ test_accuracy |
| VulBERTa | 2 | ✅ accuracy |
| examples | 4 | ✅ test_accuracy |
| **总计** | **11** | **全部支持** |

**备注**: 虽然字段名称略有不同（`accuracy` vs `test_accuracy`），但本质上都是测试准确率。

**次要公共度量**:
- **F1-score**: 2/9 模型支持（MRT-OAST, VulBERTa）
- **Loss**: 1/9 模型支持（examples）

---

### 种类2: 检索/排序任务模型 (3个模型)

**包含**:
- bug-localization-by-dnn-and-rvsm (1个): default
- Person_reID_baseline_pytorch (3个): densenet121, hrnet18, pcb

**公共度量**: ✅ **mAP (Mean Average Precision)**

**度量情况**:
| 仓库 | 模型数 | mAP度量 |
|------|--------|---------|
| bug-localization | 1 | ✅ MAP |
| Person_reID | 3 | ✅ mAP |
| **总计** | **4** | **全部支持** |

**次要公共度量**:
- **Rank@1 / Top-1**: 2/2 仓库支持
- **Rank@5 / Top-5**: 2/2 仓库支持

---

## 💡 推荐方案

### 方案1: 分层度量策略 ⭐ **推荐**

根据任务类型使用不同的公共度量：

```json
{
  "classification_models": {
    "primary_metric": "accuracy",
    "models": [
      "MRT-OAST/default",
      "pytorch_resnet_cifar10/*",
      "VulBERTa/*",
      "examples/*"
    ],
    "count": 11
  },
  "retrieval_models": {
    "primary_metric": "mAP",
    "models": [
      "bug-localization-by-dnn-and-rvsm/default",
      "Person_reID_baseline_pytorch/*"
    ],
    "count": 4
  }
}
```

**优点**:
- ✅ 每类模型都有明确的公共度量
- ✅ 度量指标符合各任务类型的最佳实践
- ✅ 便于跨模型比较和分析

**缺点**:
- ⚠️ 无法直接比较不同类型的模型

---

### 方案2: 归一化性能分数

创建统一的"性能分数"（Performance Score），将不同度量归一化到0-1范围：

```python
def get_unified_performance_score(model_type, metrics):
    """
    获取统一的性能分数

    分类任务: 使用 accuracy 作为分数
    检索任务: 使用 mAP 作为分数（通常已是0-1范围）
    """
    if model_type in ["MRT-OAST", "pytorch_resnet_cifar10", "VulBERTa", "examples"]:
        # 分类任务
        return metrics.get("accuracy") or metrics.get("test_accuracy")

    elif model_type in ["bug-localization-by-dnn-and-rvsm", "Person_reID_baseline_pytorch"]:
        # 检索任务
        return metrics.get("map") or metrics.get("MAP")

    return None
```

**优点**:
- ✅ 提供了统一的性能表示
- ✅ 便于跨模型类型比较
- ✅ 易于实现和理解

**缺点**:
- ⚠️ 可能掩盖任务特定的性能特征
- ⚠️ 不同任务的"100%"含义不同

---

### 方案3: 多指标聚合

为每个模型计算多个度量，并在分析时根据需要选择：

```python
# 配置文件中定义主要度量
{
  "MRT-OAST": {
    "primary_metric": "accuracy",
    "secondary_metrics": ["f1", "precision", "recall"]
  },
  "pytorch_resnet_cifar10": {
    "primary_metric": "test_accuracy",
    "secondary_metrics": ["best_val_accuracy"]
  },
  "bug-localization-by-dnn-and-rvsm": {
    "primary_metric": "map",
    "secondary_metrics": ["top1", "top5"]
  }
  // ... 其他模型
}
```

**优点**:
- ✅ 保留所有性能信息
- ✅ 灵活性高，可根据分析需求选择度量
- ✅ 不丢失任何细节

**缺点**:
- ⚠️ 复杂度较高
- ⚠️ 需要更多的数据处理和分析工作

---

## 🔧 实施建议

### 推荐实施方案1（分层度量策略）

#### 步骤1: 在配置文件中添加任务类型标记

修改 `config/models_config.json`，为每个仓库添加任务类型：

```json
{
  "models": {
    "MRT-OAST": {
      "task_type": "classification",
      "primary_metric": "accuracy",
      // ... 其他配置
    },
    "pytorch_resnet_cifar10": {
      "task_type": "classification",
      "primary_metric": "test_accuracy",
      // ... 其他配置
    },
    "bug-localization-by-dnn-and-rvsm": {
      "task_type": "retrieval",
      "primary_metric": "map",
      // ... 其他配置
    }
    // ... 其他模型
  }
}
```

#### 步骤2: 修改 mutation.py 中的度量提取逻辑

添加统一的性能分数提取方法：

```python
def get_primary_performance_metric(self, repo: str, metrics: Dict[str, float]) -> Optional[float]:
    """获取主要性能度量

    Args:
        repo: 仓库名称
        metrics: 提取的性能度量字典

    Returns:
        主要性能度量值，如果不存在则返回None
    """
    repo_config = self.config["models"][repo]
    primary_metric = repo_config.get("primary_metric")

    if primary_metric and primary_metric in metrics:
        return metrics[primary_metric]

    return None
```

#### 步骤3: 在结果JSON中添加统一字段

修改 `save_results()` 方法，添加主要性能度量：

```json
{
  "experiment_id": "...",
  "performance_metrics": {
    "accuracy": 85.0,
    "f1": 0.83,
    // ... 其他度量
  },
  "primary_performance_metric": 85.0,  // 新增：主要性能度量
  "task_type": "classification"        // 新增：任务类型
}
```

---

## 📉 各指标详细说明

### Accuracy (准确率)
- **定义**: 正确预测样本数 / 总样本数
- **范围**: 0-100% 或 0.0-1.0
- **适用**: 分类任务
- **优点**: 直观易懂
- **缺点**: 在类别不平衡时可能误导

### F1-score
- **定义**: Precision和Recall的调和平均数
- **范围**: 0.0-1.0
- **适用**: 二分类/多分类任务
- **优点**: 综合考虑精确率和召回率
- **缺点**: 计算相对复杂

### mAP (Mean Average Precision)
- **定义**: 所有查询的平均精度的平均值
- **范围**: 0.0-1.0
- **适用**: 检索/排序任务
- **优点**: 综合评估排序质量
- **缺点**: 计算复杂，不直观

### Rank@N
- **定义**: Top-N结果中包含正确答案的查询比例
- **范围**: 0.0-1.0 或 0-100%
- **适用**: 检索任务
- **优点**: 直观，易于理解
- **缺点**: 只考虑Top-N，忽略排序细节

### Loss (损失)
- **定义**: 模型预测与真实值的差异
- **范围**: 0-∞（越小越好）
- **适用**: 所有任务
- **优点**: 直接反映优化目标
- **缺点**: 不直观，不同模型不可比

---

## 📋 性能度量字段映射表

| 仓库 | 配置中的字段名 | 统一名称建议 | 数值范围 |
|------|---------------|-------------|----------|
| MRT-OAST | `accuracy` | `accuracy` | 0.0-1.0 |
| pytorch_resnet | `test_accuracy` | `accuracy` | 0.0-1.0 |
| VulBERTa | `accuracy` | `accuracy` | 0.0-1.0 |
| examples | `test_accuracy` | `accuracy` | 0.0-1.0 |
| bug-localization | `map` | `mAP` | 0.0-1.0 |
| Person_reID | `map` | `mAP` | 0.0-1.0 |

---

## 🎯 结论

### 主要发现

1. **不存在全局公共度量**: 12个模型使用不同类型的性能度量，无法找到单一的公共指标。

2. **按任务类型存在公共度量**:
   - **分类任务 (11个模型)**: 全部支持 **Accuracy**
   - **检索任务 (4个模型)**: 全部支持 **mAP**

3. **度量分布**:
   - Accuracy: 9/12 模型（75%）
   - mAP: 2/6 仓库（涉及4个模型）
   - F1: 2/12 模型（17%）
   - Rank@N: 3个模型（25%）

### 推荐方案

✅ **采用方案1：分层度量策略**

**理由**:
1. 符合机器学习领域的最佳实践
2. 保持度量的准确性和可解释性
3. 实施简单，维护成本低
4. 便于未来扩展新模型

**实施步骤**:
1. 为配置文件添加 `task_type` 和 `primary_metric` 字段
2. 修改 `mutation.py` 添加统一的度量提取方法
3. 在结果JSON中添加 `primary_performance_metric` 字段
4. 更新文档说明度量策略

---

## 📚 相关文档

- [模型配置文件](../config/models_config.json)
- [超参数支持矩阵](hyperparameter_support_matrix.md)
- [主项目文档](../README.md)

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**最后更新**: 2025-11-09
