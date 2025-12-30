# v2修复测试结果报告

**日期**: 2025-12-26
**测试任务**: image_classification_resnet
**状态**: ⚠️ **v2修复失败 - 仍然0条因果边**

---

## 📊 测试结果摘要

### v2修复参数
- **数据标准化**: ✅ StandardScaler (mean=0, std=1)
- **Alpha**: 0.5 (从0.1增加)
- **n_steps**: 10000 (从3000增加)
- **n_particles**: 20 (从10增加)
- **threshold**: 0.3

### 测试结果
- **运行时间**: 5.7分钟
- **因果边数**: **0条** ❌
- **图矩阵**: **完全为0** (max=0.0, min=0.0, std=0.0)
- **非零元素**: 0个

---

## 🔍 深度诊断

### 1. Adult成功案例对比 ⭐⭐⭐

**关键发现**: Adult数据集的样本/变量比**更低**，但成功检测到6条因果边！

| 维度 | Adult（成功） | Resnet（失败） | 差异 |
|------|---------------|---------------|------|
| 样本数 | 10 | 39 | Resnet **更多** ✅ |
| 变量数 | 24 | 13 | Adult 更多 |
| 样本/变量比 | **0.42** | **3.00** | Resnet **更高** ✅ |
| 方差为0的列 | 5个 (Width, D_SPD等) | 1个 (l2_reg) | 相似 |
| 相关性（平均） | ? | 0.387 | - |
| 高相关对（>0.7） | ? | 22对 | - |
| 缺失率 | 0% | 15.98% | Resnet 更高 ⚠️ |
| DiBS参数 | alpha=0.1, n_steps=3000 | alpha=0.5, n_steps=10000 | Resnet **更强** ✅ |
| **结果** | **6条边** ✅ | **0条边** ❌ | - |

**结论**: 参数强度不是问题（Resnet的参数已经更强），样本量也不是问题（Resnet更多）。

### 2. 可能的根本原因

#### 原因1: Alpha仍然不够 ⚠️⚠️⚠️

**论文建议**: Alpha = **0.9**（我们只用了0.5）

**Alpha的作用**:
- **Alpha太小（如0.1）**: 惩罚弱 → 倾向于学习**空图**（0条边）
- **Alpha适中（如0.5）**: 平衡，但可能仍然不足
- **Alpha较大（如0.9）**: 惩罚强 → 倾向于学习**稠密图**（更多边）

**建议**: 尝试 alpha=0.9

#### 原因2: 数据中真的没有明显因果关系 ⚠️⚠️

**能耗数据特性**:
- 高相关性（最高0.986），但**相关≠因果**
- 可能是**共同因（confounders）**导致的相关
- 例如：gpu_watts 和 cpu_ratio 高度相关（0.986），但可能是因为都受到"训练强度"的影响

**Adult数据特性**:
- 训练指标（Tr_Acc, Tr_F1）和测试指标（Te_Acc, Te_F1）之间有**明确的因果方向**
- 例如：训练F1↗ → 测试准确率↘（过拟合）

**能耗数据可能的情况**:
- 变量间是**相互独立的**（如seed和能耗）
- 或者是**共同受其他隐变量影响**（DiBS无法检测到）

#### 原因3: DiBS算法不适用 ⚠️

**DiBS的假设**:
- 线性高斯模型
- 变量间存在因果关系
- 数据满足马尔可夫性质

**能耗数据可能违反**:
- **非线性关系**：能耗和超参数可能是非线性的
- **隐变量**：真实的因果关系可能通过隐变量传递

#### 原因4: 缺失值问题 ⚠️

Resnet数据:
- training_duration: 46%缺失
- hyperparam_learning_rate: 46%缺失
- hyperparam_seed: 69%缺失

**DiBS对缺失值的处理**: 可能不够鲁棒

---

## 🛠️ 建议方案

### 方案A: 尝试 alpha=0.9（最后一搏）⭐⭐⭐

**目标**: 使用论文建议的alpha值

**修改**:
```python
# demo_single_task_dibs_v2.py:
ALPHA = 0.9  # 从0.5增加到0.9（论文建议值）
```

**预期**:
- 如果成功：检测到5-20条边
- 如果失败：彻底放弃DiBS，换其他算法

**时间**: 10分钟（单任务测试）

---

### 方案B: 移除缺失值严重的列 ⭐⭐

**目标**: 只保留缺失率<30%的列

**实施**:
```python
# 在standardization之前添加
causal_data = causal_data.dropna(thresh=len(causal_data)*0.7, axis=1)
```

**预期**:
- 变量数: 13 → 约9个
- 样本质量提高

---

### 方案C: 尝试其他因果发现算法 ⭐⭐

**备选算法**:
1. **PC算法**（Peter-Clark）- 基于条件独立性测试
2. **NOTEARS**（No Tears）- 基于连续优化的因果发现
3. **GES**（Greedy Equivalence Search）- 贪婪搜索

**优点**:
- 不同的假设
- 可能更适合能耗数据

**缺点**:
- 需要额外实现
- 耗时更长

---

### 方案D: 放弃因果发现，改用相关性分析 ⚠️

**理由**: 如果数据中真的没有因果关系，强行使用DiBS无意义

**替代方案**:
1. **相关性分析**: 使用Pearson/Spearman相关系数
2. **回归分析**: 使用多元线性回归
3. **主成分分析（PCA）**: 降维和特征提取

**优点**:
- 简单快速
- 结果可解释

**缺点**:
- 无法区分因果方向
- 无法发现权衡模式

---

## 📝 推荐行动

### 🎯 立即行动：方案A（alpha=0.9）+ 方案B（移除缺失值严重列）

**步骤1**: 创建v3版本
```bash
# 修改demo_single_task_dibs_v2.py → v3:
# 1. ALPHA = 0.9
# 2. 添加缺失值过滤
```

**步骤2**: 运行测试（examples组，样本最多）
```bash
python3 scripts/demos/demo_single_task_dibs_v3.py \
    --task image_classification_examples \
    --input data/energy_research/processed/training_data_image_classification_examples.csv \
    --output results/energy_research/6groups_v3/image_classification_examples \
    --verbose
```

**预期时间**: 15-20分钟

**决策点**:
- **如果检测到边（>0）**: ✅ v3成功，全量运行
- **如果仍然0边**: ❌ 彻底放弃DiBS，汇报给用户决定下一步

---

## 📚 相关文件

- 诊断报告: `docs/reports/6GROUPS_DIBS_ZERO_EDGES_DIAGNOSIS_20251226.md`
- v2脚本: `scripts/demos/demo_single_task_dibs_v2.py`
- v2结果: `results/energy_research/6groups_v2/image_classification_resnet/`

---

**报告时间**: 2025-12-26 18:10
**下一步**: 等待用户决定是否尝试alpha=0.9
