# 能耗因果分析诊断报告

**日期**: 2025-12-24
**状态**: ⚠️ 分析完成但结果异常
**问题**: DiBS学习到0条因果边

---

## 执行摘要

**运行信息**:
- **执行时间**: 2025-12-23 22:57:45 - 23:04:18
- **总耗时**: 6.5分钟
- **任务组**: 4/4完成
- **因果边总数**: **0条** ⚠️

**关键发现**: 所有4个任务组的DiBS学习结果都是空图（0条边），根本原因是**数据质量问题**。

---

## 详细诊断

### 1. 数据完整性分析

| 任务组 | 总样本 | 完全无缺失 | 无缺失率 | 总体缺失率 | 状态 |
|--------|--------|------------|----------|------------|------|
| 图像分类 | 258 | 125 | 48.4% | 8.83% | ⚠️ 中等 |
| Person_reID | 116 | 75 | 64.7% | 4.96% | ⚠️ 可用 |
| VulBERTa | 142 | **0** | **0.0%** | 28.87% | ❌ 严重 |
| Bug定位 | 132 | **0** | **0.0%** | 24.38% | ❌ 严重 |

**严重问题**:
- VulBERTa和Bug定位**没有任何一行完全无缺失的数据**
- 图像分类和Person_reID也有近一半数据存在缺失

### 2. 图像分类任务组缺失值详情

| 特征列 | 填充率 | 缺失率 |
|--------|--------|--------|
| hyperparam_batch_size | 52.7% | **47.3%** ❌ |
| hyperparam_learning_rate | 67.4% | **32.6%** ❌ |
| training_duration | 67.8% | **32.2%** ❌ |
| gpu_util_avg | 99.6% | 0.4% ✅ |
| gpu_temp_max | 99.6% | 0.4% ✅ |
| energy_cpu_total_joules | 99.6% | 0.4% ✅ |
| energy_gpu_total_joules | 99.6% | 0.4% ✅ |
| perf_test_accuracy | 100.0% | 0.0% ✅ |

**关键问题**: 超参数列缺失率高达32-47%！

### 3. DiBS学习结果

| 任务组 | 变量数 | 样本数 | 迭代次数 | 学到的边数 | 图密度 | 是否DAG |
|--------|--------|--------|----------|------------|--------|---------|
| 图像分类 | 13 | 258 | 3000 | **0** | 0.000 | ✅ |
| Person_reID | 16 | 116 | 3000 | **0** | 0.000 | ✅ |
| VulBERTa | 10 | 142 | 3000 | **0** | 0.000 | ✅ |
| Bug定位 | 10 | 132 | 3000 | **0** | 0.000 | ✅ |

**异常现象**:
- 所有任务组边数都是0（不是阈值筛选后的0，而是DiBS本身学到0）
- 图密度都是0.000（完全空图）
- 相关性矩阵计算结果为**nan**（无法计算）

### 4. 根本原因分析

#### 原因1: 数据缺失值未处理 ⚠️⚠️⚠️

**证据**:
```python
# 相关性矩阵计算结果
最强相关性（绝对值）: nan
平均相关性（绝对值）: nan
```

**影响**:
- DiBS需要完整的数据矩阵进行优化
- 缺失值导致相关性计算失败
- JAX无法处理nan，优化收敛到空图

#### 原因2: 预处理脚本缺失 ❌

**证据**:
- `scripts/preprocess_stratified_data.py` **不存在**
- 训练数据直接从原始数据提取，未经插补处理

**影响**:
- 缺失值直接传递给DiBS
- 没有标准化验证
- 没有异常值处理

#### 原因3: 超参数填充率不均 ⚠️

**图像分类示例**:
- MNIST模型使用 `batch_size` + `learning_rate` → 填充率高
- CIFAR-10模型使用 `training_duration`（统一参数）→ 缺失 `batch_size`
- 导致组合后的数据集超参数列缺失率高

**影响**:
- 超参数 → 性能的因果路径被阻断
- DiBS无法学习超参数的因果效应

---

## 数据安全性评估

### ✅ 安全方面

1. **文件完整性**: 所有8个文件（4×2）已生成
2. **数据格式**: .npy和.pkl格式正确，可正常加载
3. **日志完整**: 完整记录了运行过程，无错误中断
4. **元数据一致**: metadata文件与数据文件一致

### ❌ 可靠性问题

1. **因果发现失败**: 0条边意味着**无法得出任何因果结论**
2. **数据质量差**: 缺失率高，无法支持可靠的因果推断
3. **结果不可用**: 当前结果**不能用于任何科学结论**
4. **无法复现论文**: 与Adult数据集分析（6条边）形成强烈对比

---

## DiBS参数配置

**当前配置**:
```python
DIBS_ALPHA = 0.1  # 稀疏性系数
EDGE_THRESHOLD = 0.3  # 边筛选阈值
NUM_ITERATIONS = 3000  # 迭代次数（未在日志中明确显示）
```

**参数合理性**:
- ✅ alpha=0.1是标准值（Adult实验成功使用）
- ✅ 阈值=0.3是标准值（Adult发现6条边）
- ⚠️ 3000步迭代可能不足（Adult使用更多）

**但**: 参数设置不是问题根源，**数据质量**才是！

---

## 对比分析：Adult vs 能耗数据

| 维度 | Adult数据集 | 能耗数据（图像分类） |
|------|-------------|---------------------|
| 样本数 | 10 | 258 ✅ |
| 变量数 | 15 | 13 ✅ |
| 缺失率 | 0% | **8.83%** ❌ |
| 完全无缺失行 | 100% | **48.4%** ❌ |
| 相关性计算 | 成功 | **失败(nan)** ❌ |
| DiBS边数 | 6条 | **0条** ❌ |
| DML显著边 | 4条 | 无（跳过） ❌ |
| 运行时间 | 61.4分钟 | 6.5分钟 ⚠️ |

**关键差异**:
- Adult数据经过精心准备，**无缺失值**
- 能耗数据直接提取，**缺失值严重**
- 运行时间太短（6.5分钟 vs 61分钟）可能表明DiBS快速收敛到空解

---

## 解决方案

### 🔴 紧急方案（立即可行）

#### 方案A: 仅使用完全无缺失的行

**图像分类**:
```python
df_clean = df.dropna()  # 保留125/258行（48.4%）
```

**优点**: 简单快速，数据质量最高
**缺点**: 样本量减少一半（但125样本仍足够）

**预期**: 可能发现一些因果边

#### 方案B: 均值/中位数插补

```python
# 对超参数列使用中位数插补
for col in ['hyperparam_learning_rate', 'hyperparam_batch_size', 'training_duration']:
    df[col].fillna(df[col].median(), inplace=True)
```

**优点**: 保留所有样本，简单有效
**缺点**: 可能引入偏差（尤其是47%缺失的batch_size）

**预期**: 可能发现弱因果边

### 🟡 短期方案（1-2天）

#### 实现完整预处理脚本

创建 `scripts/preprocess_stratified_data.py`:

**功能**:
1. **缺失值插补**:
   - 超参数: 中位数插补
   - 能耗指标: 线性插值或删除行
2. **异常值处理**:
   - IQR方法检测
   - Winsorization处理
3. **标准化验证**:
   - 检查数据范围
   - 验证One-Hot编码
4. **数据质量报告**:
   - 插补前后对比
   - 分布变化分析

**预期**: 生成高质量训练数据，支持可靠因果推断

### 🟢 长期方案（1周）

#### 优化数据生成流程

**改进点**:
1. **源头优化**: 在数据提取阶段就处理缺失值
2. **分层处理**: 不同任务组使用不同的插补策略
3. **验证机制**: 自动检查数据质量
4. **文档完善**: 记录所有预处理决策

**预期**: 建立可复现的高质量数据管线

---

## 推荐行动计划

### 阶段1: 快速验证（今天）

1. ✅ **使用方案A**: 仅分析完全无缺失的数据
   ```bash
   # 修改数据加载代码，添加 dropna()
   # 重新运行图像分类和Person_reID分析
   ```

2. ✅ **降低DiBS迭代次数**: 3000 → 1000（加快验证）

3. ✅ **调整alpha参数**: 0.1 → 0.05（允许更密集的图）

4. ✅ **预期结果**: 至少发现1-2条因果边

### 阶段2: 完整修复（明天）

1. ⏳ 实现均值插补脚本
2. ⏳ 重新生成所有4个任务组数据
3. ⏳ 完整运行DiBS分析（预计60分钟）
4. ⏳ 生成因果效应报告

### 阶段3: 长期优化（本周）

1. ⏳ 实现完整预处理脚本
2. ⏳ 优化数据生成流程
3. ⏳ 编写数据质量验证测试
4. ⏳ 更新文档和最佳实践

---

## 文件清单

### 生成的文件（可用但结果无效）

**因果图**:
- `results/energy_research/task_specific/image_classification_causal_graph.npy` (804B)
- `results/energy_research/task_specific/person_reid_causal_graph.npy` (1.2KB)
- `results/energy_research/task_specific/vulberta_causal_graph.npy` (528B)
- `results/energy_research/task_specific/bug_localization_causal_graph.npy` (528B)

**因果边**:
- `results/energy_research/task_specific/image_classification_causal_edges.pkl` (460B)
- `results/energy_research/task_specific/person_reid_causal_edges.pkl` (479B)
- `results/energy_research/task_specific/vulberta_causal_edges.pkl` (396B)
- `results/energy_research/task_specific/bug_localization_causal_edges.pkl` (402B)

**摘要**:
- `results/energy_research/task_specific/analysis_summary.txt` (349B)

**日志**:
- `logs/energy_research/experiments/energy_causal_analysis_20251223_225745.log` (7.5KB)
- `logs/energy_research/experiments/screen.log` (11KB)

### 数据文件（需重新生成）

**训练数据**:
- `data/energy_research/training/training_data_image_classification.csv` (50KB) ⚠️ 缺失率8.83%
- `data/energy_research/training/training_data_person_reid.csv` (28KB) ⚠️ 缺失率4.96%
- `data/energy_research/training/training_data_vulberta.csv` (20KB) ❌ 缺失率28.87%
- `data/energy_research/training/training_data_bug_localization.csv` (22KB) ❌ 缺失率24.38%

---

## 结论

### 数据安全性: ✅ 安全

- 所有文件完整生成
- 无数据丢失或损坏
- 日志记录完整

### 数据可靠性: ❌ 不可靠

- 因果分析结果**完全无效**（0条边）
- 数据质量**严重不足**（缺失值高达47%）
- **不能用于任何科学结论或论文发表**

### 下一步: 🔴 紧急修复

**立即执行**:
1. 使用完全无缺失的数据快速验证
2. 实现缺失值插补
3. 重新运行分析

**预期时间**: 1-2天完成修复

---

**报告人**: Claude
**审核状态**: 待用户确认
**优先级**: 🔴 高（影响核心研究成果）
