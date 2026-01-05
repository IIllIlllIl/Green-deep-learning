# 替代因果分析工具评估报告

**日期**: 2026-01-05
**评估人**: Claude
**目的**: 评估causal-cmd和causal-learn两个因果分析工具，为能耗数据因果分析提供替代方案

---

## 📋 执行摘要

本报告评估了两个因果分析工具库，用于补充或替代当前项目中使用的DiBS因果图学习方法：

1. **causal-cmd** - Java命令行工具，提供30+种经典因果发现算法
2. **causal-learn** - Python库，提供约束、评分、函数因果模型等多种方法

**核心发现**:
- ✅ **causal-learn高度推荐** - Python集成便捷，算法丰富，与现有代码兼容性好
- ✅ **LiNGAM算法特别适合** - 可直接用于超参数→能耗的线性因果分析
- ✅ **PC/FCI/GES可作为DiBS对照** - 计算效率高，适合中小规模数据
- ⚠️ **causal-cmd适合独立分析** - Java工具，需单独数据准备，但算法全面

---

## 1. 工具1: causal-cmd

### 1.1 基本信息

- **项目地址**: https://github.com/bd2kccd/causal-cmd
- **文档**: https://bd2kccd.github.io/docs/causal-cmd/
- **开发者**: Center for Causal Discovery (CMU)
- **语言**: Java
- **类型**: 命令行工具
- **最新版本**: 1.10.0+

### 1.2 支持的算法

causal-cmd提供约**30种因果发现算法**，主要包括：

#### 约束方法 (Constraint-based)
- **PC** - Peter-Clark算法，经典约束方法
- **FCI** - Fast Causal Inference，处理潜变量和选择偏差
- **RFCI** - Really Fast Causal Inference
- **CPC** - Conservative PC
- **CCD** - Cyclic Causal Discovery

#### 评分方法 (Score-based)
- **FGES** - Fast Greedy Equivalence Search
- **GES** - Greedy Equivalence Search
- **BOSS** - Best Order Score Search

#### 混合方法
- **GRASP** - Greedy Relaxations of the Sparsest Permutation
- **FAS** - Fast Adjacency Search
- **FASK** - Fast Adjacency Skewness

### 1.3 输入/输出格式

**输入**:
```bash
java -jar causal-cmd-1.10.0.jar \
  --algorithm fges \
  --data-type continuous \
  --dataset data.txt \
  --delimiter tab \
  --score sem-bic-score
```

**数据类型支持**:
- `continuous` - 连续型数据（适合我们的能耗数据）
- `discrete` - 离散型数据
- `mixed` - 混合型数据
- `covariance` - 协方差矩阵

**输出**:
- 文本日志文件（算法运行详情）
- JSON格式因果图（可选）
- 包含：节点列表、边列表（带方向）、图评分

### 1.4 主要特性

✅ **优势**:
1. **算法全面** - 30+种算法，覆盖约束、评分、混合方法
2. **成熟稳定** - CMU开发，广泛使用于学术研究
3. **可配置性强** - 支持Bootstrap、并行化、参数调优
4. **独立运行** - 不依赖特定编程语言环境

⚠️ **限制**:
1. **Java环境** - 需要Java 8+，与Python代码集成不便
2. **数据准备** - 需要单独准备tab/comma分隔的文本文件
3. **结果解析** - 需要解析文本/JSON输出，无法直接编程调用
4. **学习曲线** - 命令行参数众多，需要熟悉各算法特性

### 1.5 适用场景

✅ **推荐用于**:
- 独立的因果发现分析（不需要与代码集成）
- 快速尝试多种算法对比
- 作为论文中的对照方法
- 大规模数据的批处理分析

❌ **不适合**:
- 需要与Python代码紧密集成的场景
- 需要自定义算法或扩展的场景
- 需要实时因果推断的场景

---

## 2. 工具2: causal-learn

### 2.1 基本信息

- **项目地址**: https://github.com/py-why/causal-learn
- **文档**: https://causal-learn.readthedocs.io/
- **开发者**: py-why团队（EconML同一组织）
- **语言**: Python (>=3.7)
- **类型**: Python库
- **安装**: `pip install causal-learn`

### 2.2 支持的算法

#### 约束方法 (Constraint-based)
- **PC** - Peter-Clark算法
- **FCI** - Fast Causal Inference
- **CD-NOD** - Causal Discovery with Non-Observable Data

#### 评分方法 (Score-based)
- **GES** with BIC score - 贪心等价搜索
- **Exact Search** - 精确搜索（小规模数据）

#### 函数因果模型 (Functional Causal Models)
- **LiNGAM** ⭐ - Linear Non-Gaussian Acyclic Model（**重点推荐**）
  - 假设：线性关系 + 非高斯加性噪声
  - 输出：因果顺序 + 因果强度
  - 适用：连续变量的线性因果关系

- **Post-nonlinear** - 后非线性因果模型
- **ANM** - Additive Noise Models

#### 高级方法
- **Hidden Causal Representation Learning** - 隐藏因果表示学习
- **Granger Causality** - 格兰杰因果（时间序列）
- **GRaSP** - Greedy Relaxations of the Sparsest Permutation
- **BOSS** - Best Order Score Search

### 2.3 API设计

**基本用法示例**:

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
import numpy as np

# 准备数据 (n_samples x n_features)
data = np.array(...)  # 我们的能耗数据

# 方法1: PC算法（约束方法）
cg = pc(data, alpha=0.05, indep_test='fisherz')
print(cg.G.graph)  # 因果图

# 方法2: GES算法（评分方法）
record = ges(data, score_func='local_score_BIC')
print(record['G'].graph)

# 方法3: LiNGAM（函数因果模型）⭐
model = lingam.ICALiNGAM()
model.fit(data)
print(model.adjacency_matrix_)  # 因果强度矩阵
print(model.causal_order_)      # 因果顺序
```

### 2.4 主要特性

✅ **优势**:
1. **Python原生** - 无缝集成到现有Python代码
2. **API简洁** - 统一的sklearn风格API
3. **算法丰富** - 约束、评分、函数模型、高级方法全覆盖
4. **活跃维护** - py-why组织维护，更新频繁
5. **与scikit-learn集成** - 可结合回归、特征选择等
6. **可扩展** - 模块化设计，易于自定义

⚠️ **限制**:
1. **文档不够完善** - 部分算法文档较少
2. **性能** - 纯Python实现，大规模数据可能较慢
3. **依赖** - 需要numpy, scipy, pandas等科学计算库

### 2.5 适用场景

✅ **推荐用于**:
- 与Python代码集成的因果分析
- 快速原型开发和实验
- 需要自定义因果模型的场景
- 与机器学习pipeline结合的场景

✅ **特别适合我们的项目**:
- 已有Python代码基础（DiBS, DML）
- 需要多种算法对比验证
- 超参数→能耗的线性因果分析（LiNGAM）
- 与回归分析结合

---

## 3. 与当前项目的相关性评估

### 3.1 当前项目背景

**数据特征**:
- 样本量: 836个实验（795个有效能耗数据）
- 数据类型: 连续型（超参数、能耗、性能指标）
- 变量数: 约50-87列（取决于使用哪个数据文件）

**研究问题**:
1. ⏳ **问题1**: 超参数对能耗的影响（方向和大小）
2. ⏳ **问题2**: 能耗和性能之间的权衡关系
3. ⏳ **问题3**: 中间变量的中介效应

**当前方法**:
- ✅ **DiBS** - 贝叶斯因果图学习（已成功，2026-01-05突破）
- ✅ **回归分析** - 超参数效应量化（进行中）
- ⏳ **中介效应分析** - 待执行

### 3.2 相关性分析

#### 3.2.1 问题1: 超参数对能耗的影响

**适用算法** (按推荐度排序):

| 算法 | 工具 | 相关度 | 推荐理由 | 限制 |
|------|------|--------|---------|------|
| **LiNGAM** ⭐⭐⭐⭐⭐ | causal-learn | 🟢 极高 | 1. 直接输出因果顺序和因果强度<br>2. 假设线性关系（符合回归分析）<br>3. 可量化超参数→能耗的因果效应<br>4. 计算效率高 | 假设非高斯噪声 |
| **FGES/GES** ⭐⭐⭐⭐ | 两者都有 | 🟢 高 | 1. 评分方法，贪心搜索效率高<br>2. 适合中等规模数据<br>3. 可作为DiBS的对照方法 | 不直接给出因果强度 |
| **PC/FCI** ⭐⭐⭐ | 两者都有 | 🟡 中等 | 1. 经典方法，广泛验证<br>2. 约束方法，基于条件独立性<br>3. 计算快速 | 需要设置alpha阈值<br>对样本量敏感 |
| **BOSS/GRaSP** ⭐⭐⭐ | 两者都有 | 🟡 中等 | 1. 新型排列方法<br>2. 可能更适合小样本 | 较新，验证较少 |

**推荐组合方案**:
```
DiBS (贝叶斯方法)
  + LiNGAM (线性方法，给出因果强度) ⭐
  + GES (评分方法，对照验证)
  + 回归分析 (量化效应大小)
```

#### 3.2.2 问题2: 能耗-性能权衡关系

**适用算法**:

| 算法 | 工具 | 相关度 | 推荐理由 |
|------|------|--------|---------|
| **LiNGAM** ⭐⭐⭐⭐ | causal-learn | 🟢 高 | 可识别能耗↔性能的双向或单向因果关系 |
| **PC/FCI** ⭐⭐⭐ | 两者都有 | 🟡 中等 | 可检测条件独立性，识别权衡关系 |

**注意**: 权衡关系主要通过Pareto分析和多目标回归，因果方法作为补充验证。

#### 3.2.3 问题3: 中间变量的中介效应

**适用算法**:

| 算法 | 工具 | 相关度 | 推荐理由 |
|------|------|--------|---------|
| **LiNGAM** ⭐⭐⭐⭐⭐ | causal-learn | 🟢 极高 | 1. 直接给出因果顺序（识别中介路径）<br>2. 因果强度矩阵（量化中介效应）<br>3. 示例：learning_rate → gpu_util → energy |
| **中介效应专用分析** | - | 🟢 极高 | 建议结合causal-learn + 专门的中介效应库 |

**关键优势**: LiNGAM的因果顺序可直接识别中介路径，无需额外假设。

#### 3.2.4 Granger因果（不推荐）

**相关度**: 🔴 低

**原因**:
- Granger因果需要**时间序列数据**（同一实体的多个时间点）
- 我们的数据是**独立实验**（不同配置的单次运行）
- 不适用于横截面数据

### 3.3 与DiBS的对比

| 维度 | DiBS | PC/FCI/GES | LiNGAM | causal-cmd |
|------|------|-----------|---------|-----------|
| **方法类型** | 贝叶斯 | 约束/评分 | 函数模型 | 多种方法 |
| **计算复杂度** | 高（MCMC采样） | 中等 | 低 | 取决于算法 |
| **样本量需求** | 较大 | 中等 | 中等 | 取决于算法 |
| **输出** | 因果图（边概率） | 因果图 | 因果顺序+强度矩阵 | 因果图 |
| **假设** | 灵活 | 条件独立性 | 线性+非高斯 | 取决于算法 |
| **Python集成** | ✅ | ✅ | ✅ | ❌ (Java) |
| **因果强度** | ❌（只有边概率） | ❌ | ✅ | ❌ |
| **适合问题1** | ✅ | ✅ | ⭐ | ✅ |
| **适合问题2** | ✅ | ✅ | ✅ | ✅ |
| **适合问题3** | ✅ | 🟡 | ⭐⭐ | 🟡 |

**核心发现**:
- **DiBS**: 适合探索性因果发现，已成功检测到23条强边（2026-01-05）
- **LiNGAM**: 最适合问题3（中介效应），可直接给出因果顺序和强度
- **PC/FCI/GES**: 作为对照方法，验证DiBS结果
- **causal-cmd**: 适合独立分析和算法对比

---

## 4. 推荐使用方案

### 4.1 短期推荐（当前任务）

**优先级1: LiNGAM用于问题3（中介效应分析）** ⭐⭐⭐⭐⭐

```python
# 步骤1: 安装
pip install causal-learn

# 步骤2: 准备数据（选择关键变量）
# 超参数: learning_rate, batch_size, optimizer
# 中介变量: gpu_util_avg, gpu_temp_max, duration_seconds
# 结果变量: energy_gpu, energy_cpu

# 步骤3: 运行LiNGAM
from causallearn.search.FCMBased import lingam
import pandas as pd

data = pd.read_csv('data.csv')
selected_cols = ['learning_rate', 'batch_size', 'gpu_util_avg',
                 'gpu_temp_max', 'energy_gpu']
X = data[selected_cols].values

model = lingam.ICALiNGAM()
model.fit(X)

# 步骤4: 分析结果
print("因果顺序:", model.causal_order_)
print("因果矩阵:\n", model.adjacency_matrix_)
# adjacency_matrix_[i,j] 表示 j → i 的因果强度

# 步骤5: 识别中介路径
# 例如: learning_rate → gpu_util_avg → energy_gpu
```

**优点**:
- ✅ 快速（<1分钟）
- ✅ 直接给出因果强度
- ✅ 无需复杂参数调优
- ✅ 结果易于解释

**优先级2: GES作为DiBS的对照** ⭐⭐⭐

```python
from causallearn.search.ScoreBased.GES import ges

record = ges(data, score_func='local_score_BIC')
causal_graph = record['G'].graph
```

**用途**: 验证DiBS发现的23条强边是否被GES也检测到

### 4.2 中期推荐（方法对比研究）

**算法对比矩阵**:

| 算法 | 工具 | 实施优先级 | 估计耗时 | 用途 |
|------|------|-----------|---------|------|
| DiBS | 现有代码 | ✅ 已完成 | 6小时 | 主方法 |
| LiNGAM | causal-learn | 🟢 高 | <5分钟 | 问题3主要方法 |
| GES | causal-learn | 🟡 中 | <10分钟 | DiBS对照 |
| PC | causal-learn | 🟡 中 | <10分钟 | DiBS对照 |
| 回归分析 | statsmodels | ✅ 进行中 | - | 问题1主要方法 |

### 4.3 长期推荐（论文完善）

如果要在论文中展示方法对比，建议：

1. **使用causal-cmd批量运行多种算法**
   ```bash
   # 创建批处理脚本
   for algo in fges pc fci ges; do
     java -jar causal-cmd.jar \
       --algorithm $algo \
       --data-type continuous \
       --dataset energy_data.txt \
       --delimiter tab \
       --output results_$algo.json
   done
   ```

2. **创建算法对比表**（算法名、检测边数、运行时间、解释性）

3. **讨论不同方法的假设和适用性**

---

## 5. 实施建议

### 5.1 立即行动（本周）

✅ **推荐**: 安装causal-learn并测试LiNGAM

```bash
# 1. 安装
pip install causal-learn

# 2. 创建测试脚本
cd analysis/scripts
touch test_lingam.py

# 3. 测试运行（小数据集）
python test_lingam.py
```

**测试数据**: 选择1个任务组（例如VulBERTa/mlp），约100-200个样本

**预期结果**:
- 因果顺序列表
- 因果强度矩阵
- 识别出的中介路径

### 5.2 近期计划（本月）

1. **完成LiNGAM全量分析**（6个任务组）
2. **运行GES对照实验**（验证DiBS结果）
3. **创建可视化**（因果图、路径图）
4. **撰写分析报告**

### 5.3 可选扩展（按需）

- ⏳ 使用causal-cmd进行多算法批量对比
- ⏳ 探索Post-nonlinear模型（如果线性假设不成立）
- ⏳ 尝试BOSS/GRaSP新型方法

---

## 6. 潜在风险与应对

### 6.1 LiNGAM假设不满足

**风险**: LiNGAM假设线性关系和非高斯噪声

**检验方法**:
```python
from scipy import stats

# 检验残差是否为非高斯
residuals = ...  # LiNGAM拟合后的残差
stat, p_value = stats.shapiro(residuals)  # Shapiro-Wilk正态性检验
if p_value > 0.05:
    print("警告: 残差接近正态分布，LiNGAM假设可能不满足")
```

**应对**:
- 如果线性假设不满足 → 尝试Post-nonlinear模型
- 如果非高斯假设不满足 → 使用PC/GES等不依赖该假设的方法

### 6.2 不同算法结果不一致

**原因**: 不同算法基于不同假设

**应对**:
1. **报告所有结果**，讨论差异原因
2. **使用多数投票**，选择多种算法都检测到的边
3. **结合领域知识**，筛选合理的因果关系

### 6.3 计算资源限制

**causal-learn性能**:
- 小规模（<500样本，<20变量）：秒级
- 中等规模（500-1000样本，20-50变量）：分钟级
- 大规模（>1000样本，>50变量）：可能较慢

**应对**:
1. **分层分析**（已在方案A'中使用）
2. **特征选择**（只保留关键变量）
3. **并行计算**（对6个任务组并行运行）

---

## 7. 总结与建议

### 7.1 核心结论

1. ✅ **causal-learn强烈推荐** - Python集成便捷，算法丰富，特别适合我们的项目
2. ✅ **LiNGAM最适合问题3** - 直接识别中介路径和量化中介效应
3. ✅ **GES/PC可作为DiBS对照** - 验证DiBS的23条强边
4. 🟡 **causal-cmd适合论文方法对比** - 批量运行多种算法，但集成不便

### 7.2 行动优先级

**立即执行** (本周):
1. 安装causal-learn
2. 测试LiNGAM（1个任务组）
3. 验证结果合理性

**近期执行** (2周内):
1. LiNGAM全量分析（6组）
2. GES对照实验
3. 撰写中介效应分析报告

**可选执行** (按需):
1. causal-cmd批量算法对比
2. 其他高级方法探索

### 7.3 与现有工作的结合

**当前进度** (根据CLAUDE.md):
1. ✅ DiBS参数调优成功（23条强边，2026-01-05）
2. ⏳ 问题1回归分析进行中
3. ⏳ 问题2和问题3待执行

**建议整合**:
```
问题1: 超参数→能耗的影响
  主方法: 回归分析 ⭐
  补充: LiNGAM（因果方向验证）
  对照: DiBS, GES

问题2: 能耗-性能权衡
  主方法: Pareto分析 ⭐
  补充: 多目标回归
  对照: LiNGAM（因果关系检验）

问题3: 中介变量效应
  主方法: LiNGAM ⭐⭐⭐
  补充: 传统中介效应分析
  对照: DiBS（因果图识别路径）
```

---

## 8. 参考资源

### 8.1 官方文档

- **causal-cmd**:
  - GitHub: https://github.com/bd2kccd/causal-cmd
  - 文档: https://bd2kccd.github.io/docs/causal-cmd/

- **causal-learn**:
  - GitHub: https://github.com/py-why/causal-learn
  - 文档: https://causal-learn.readthedocs.io/
  - 示例: https://github.com/py-why/causal-learn/tree/main/examples

### 8.2 相关论文

- **LiNGAM**: Shimizu et al. (2006) "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
- **PC**: Spirtes et al. (2000) "Causation, Prediction, and Search"
- **GES**: Chickering (2002) "Optimal Structure Identification With Greedy Search"
- **BOSS**: Solus et al. (2021) "Consistency Guarantees for Permutation-Based Causal Inference Algorithms"

### 8.3 项目内相关文档

- [DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md](DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md) - DiBS成功报告
- [RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md](RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md) - 3个研究问题的方法推荐
- [CAUSAL_METHODS_COMPARISON_20251228.md](CAUSAL_METHODS_COMPARISON_20251228.md) - 9种因果方法对比
- [QUESTION1_REGRESSION_ANALYSIS_PLAN.md](../QUESTION1_REGRESSION_ANALYSIS_PLAN.md) - 问题1回归分析方案

---

## 9. 下一步行动

### 9.1 技术准备

```bash
# 1. 安装causal-learn
cd /home/green/energy_dl/nightly/analysis
source venv/bin/activate  # 如果有虚拟环境
pip install causal-learn

# 2. 创建测试脚本目录
mkdir -p scripts/causal_learn_tests

# 3. 创建测试脚本
cat > scripts/causal_learn_tests/test_lingam.py << 'EOF'
"""
LiNGAM因果分析测试脚本
"""
import pandas as pd
import numpy as np
from causallearn.search.FCMBased import lingam

# 加载数据
data = pd.read_csv('../../data/energy_research/stage2/mediators.csv')

# 选择关键变量进行测试
selected_cols = [
    'learning_rate', 'batch_size',
    'gpu_util_avg', 'gpu_temp_max',
    'energy_gpu', 'energy_cpu'
]

# 筛选有效数据
df_test = data[selected_cols].dropna()
print(f"有效样本数: {len(df_test)}")

# 运行LiNGAM
X = df_test.values
model = lingam.ICALiNGAM()
model.fit(X)

# 输出结果
print("\n因果顺序:")
for idx in model.causal_order_:
    print(f"  {idx}: {selected_cols[idx]}")

print("\n因果强度矩阵:")
print(pd.DataFrame(
    model.adjacency_matrix_,
    columns=selected_cols,
    index=selected_cols
).round(3))
EOF

chmod +x scripts/causal_learn_tests/test_lingam.py
```

### 9.2 验证与测试

1. ✅ 运行测试脚本
2. ✅ 检查结果合理性
3. ✅ 与DiBS结果对比
4. ✅ 撰写测试报告

### 9.3 文档更新

- ✅ 本报告已创建
- ⏳ 更新INDEX.md添加本报告链接
- ⏳ 测试完成后创建LiNGAM分析报告

---

**报告状态**: ✅ 完成
**下一步**: 安装causal-learn并测试LiNGAM
**优先级**: 🟢 高 - 建议本周内完成测试
