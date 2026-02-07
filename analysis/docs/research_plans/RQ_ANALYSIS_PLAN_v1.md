# Energy DL 分层因果分析研究方案

**版本**: v1.0
**日期**: 2026-02-07
**状态**: 草案，待审查

---

## 一、研究背景与目标

### 1.1 研究背景

Energy DL项目研究深度学习训练中超参数对能耗和性能的因果影响。本研究基于已完成的分层因果分析结果，按训练场景（并行vs非并行）分层，研究不同场景下因果结构的异质性。

### 1.2 研究问题

| 研究问题 | 核心内容 | 子问题 |
|---------|---------|--------|
| **RQ1** | 超参数对能耗的影响力方向和大小 | a) 主效应 b) 交互项调节 |
| **RQ2** | 利用中间变量解释因果发现 | a) 中介路径 b) 交互项中介对比 |
| **RQ3** | 能耗和性能的权衡关系 | a) 权衡检测 b) 场景差异 |

---

## 二、数据源设计

### 2.1 数据概览

| 数据集 | 位置 | 样本数 | 特征数 | 用途 |
|-------|------|-------|-------|------|
| 全局标准化数据 | `data/energy_research/6groups_global_std/` | 6组，共818样本 | 35-50列 | RQa主分析 |
| 分层数据 | `data/energy_research/stratified/` | 4层 | 26-28列 | RQb验证 |

### 2.2 RQ数据分配

| RQ | RQa 数据源 | RQb 数据源 |
|----|-----------|-----------|
| RQ1 | 全局标准化（排除交互项） | 全局标准化（仅交互项）+ 分层数据（case study） |
| RQ2 | 全局标准化（排除交互项） | 全局标准化（仅交互项）+ 分层数据（case study） |
| RQ3 | 全局标准化（排除交互项） | 全局标准化（仅交互项）+ 分层数据（case study） |

### 2.3 变量分类

| 类别 | 变量 | 角色 |
|-----|------|------|
| **超参数 (hyperparam)** | `hyperparam_batch_size`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_dropout`, `hyperparam_l2_regularization`, `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter` | 处理变量 |
| **交互项 (interaction)** | `hyperparam_*_x_is_parallel` | 调节效应 |
| **中间变量 (mediator)** | `energy_gpu_temp_avg_celsius`, `energy_gpu_temp_max_celsius`, `energy_gpu_util_avg_percent`, `energy_gpu_util_max_percent`, `energy_gpu_avg_watts`, `energy_gpu_min_watts`, `energy_gpu_max_watts` | 中介变量 |
| **能耗结果 (energy)** | `energy_cpu_pkg_joules`, `energy_cpu_ram_joules`, `energy_gpu_total_joules`, `energy_cpu_total_joules` | 结果变量 |
| **性能结果 (performance)** | `perf_test_accuracy`, `perf_map`, `perf_rank1`, `perf_rank5` | 结果变量 |
| **控制变量 (control)** | `model_mnist_ff`, `model_mnist_rnn`, `model_siamese`, `model_hrnet18`, `model_pcb` | 协变量 |
| **模式变量 (mode)** | `is_parallel` | 分层变量 |

**注**: 运行时间变量（`duration_seconds`）存在于原始数据但未包含在当前分析数据中，后续可补充作为中间变量。

---

## 三、方法论

### 3.1 因果图学习

- **方法**: DiBS (Differentiable Bayesian Structure Learning)
- **敏感性分析**: 多次随机运行，识别稳定边
- **边强度阈值**: 0.3（参考文献标准）

### 3.2 因果效应估计

- **方法**: DML (Double Machine Learning)
- **输出**: ATE (Average Treatment Effect) 及置信区间
- **显著性校正**: BH-FDR (α=0.10)

### 3.3 路径分析

- **直接边**: source → target，强度≥0.3
- **间接路径**: source → mediator → target，每段强度≥0.3
- **路径深度**: 2-hop和3-hop

### 3.4 权衡检测

- **方法**: Algorithm 1 (opposite signs规则)
- **条件**: 同一干预变量对两个结果的ATE符号相反且均显著

---

## 四、RQ1：超参数对能耗的影响

### 4.1 研究目标

量化各超参数对4种能耗指标（CPU_pkg, CPU_ram, GPU_total, CPU_total）的因果影响方向和大小。

### 4.2 RQ1a：超参数主效应

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除所有 `*_x_is_parallel` 交互项
- 阈值: 边强度 ≥ 0.3

#### 分析方法
1. 提取 `hyperparam_* → energy_*` 直接边
2. 提取 `hyperparam_* → mediator → energy_*` 间接路径（仅统计可达性）
3. 按 (超参数, 能耗类型) 汇总统计

#### 输出指标
- **边计数 (n)**: 有多少组存在该直接边
- **ATE统计**: 均值 ± 标准差
- **间接路径数 (I)**: 存在多少条间接路径
- **显著性比例**: n条边中有多少通过FDR校正

#### 输出表格格式

**表1: 超参数对能耗的因果影响汇总（RQ1a）**

| 超参数 | CPU_pkg | CPU_ram | GPU_total | CPU_total |
|-------|---------|---------|-----------|-----------|
| batch_size | D:n=k, ATE=μ±σ; I:m | ... | ... | ... |
| epochs | ... | ... | ... | ... |
| learning_rate | ... | ... | ... | ... |
| seed | ... | ... | ... | ... |
| dropout | ... | ... | ... | ... |

单元格格式: `D:n=k, ATE=均值±标准差; I:m`
- D = 直接边 (Direct)
- I = 间接路径数 (Indirect，仅计数，不计算间接ATE)
- `—` 表示无边

### 4.3 RQ1b：交互项调节效应

#### 数据与过滤
- 主要数据: 全局标准化6组，仅保留 `*_x_is_parallel` 边
- 验证数据: 分层数据 (group1_parallel vs group1_non_parallel)

#### 分析方法
1. 提取 `*_x_is_parallel → energy_*` 直接边
2. 提取 `*_x_is_parallel → mediator → energy_*` 间接路径
3. 与RQ1a相同格式汇总

#### 输出表格格式

**表2: 交互项对能耗的调节效应（RQ1b）**

| 交互项 | CPU_pkg | CPU_ram | GPU_total | CPU_total |
|-------|---------|---------|-----------|-----------|
| batch_size×parallel | D:n=?, ATE=...; I:? | ... | ... | ... |
| epochs×parallel | ... | ... | ... | ... |
| learning_rate×parallel | ... | ... | ... | ... |
| dropout×parallel | ... | ... | ... | ... |

#### 分层验证 (Case Study)

**表3: 分层ATE对比验证**

| 超参数 | Parallel ATE | Non-Parallel ATE | Δ ATE | 与交互项一致性 |
|-------|-------------|-----------------|-------|--------------|
| batch_size→GPU_total | ... | ... | ... | ✓/✗ |
| epochs→GPU_total | ... | ... | ... | ... |

验证逻辑: 交互项ATE的符号和大小应与分层ATE差异一致。

### 4.4 间接边ATE处理策略

**采用保守策略**:
- RQ1中仅报告间接路径的存在性和计数
- 不计算间接ATE（需要"无交互假设"，可能不成立）
- 间接效应的详细分析在RQ2中进行

---

## 五、RQ2：中间变量解释

### 5.1 研究目标

识别超参数如何通过中间变量（GPU温度、利用率、功率等）影响最终能耗，解释因果机制。

### 5.2 RQ2a：因果路径与中介分析

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除交互项

#### 分析方法

**Step 1: 路径提取**
```
输入: DiBS邻接矩阵（阈值0.3）
输出: 所有 hyperparam → ... → energy 路径

算法:
1. 构建有向图 G
2. 对每个 (hyperparam, energy) 对:
   - BFS搜索所有路径（深度限制2-3）
   - 记录路径和各边强度
3. 计算路径强度 = min(边强度) 或 Π(边强度)
4. 按路径强度排序
```

**Step 2: 中介频率统计**
- 统计各中间变量作为中介的出现次数
- 记录涉及的超参数和能耗变量

**Step 3: 路径ATE估计（可选）**
- 方法: 乘积法 ATE_path = ATE₁ × ATE₂
- 局限性: 假设线性传递，无交互效应

#### 输出表格

**表4: 中介变量频率统计（RQ2a）**

| 中间变量 | 作为中介次数 | 涉及超参数 | 涉及能耗 | 典型路径示例 |
|---------|-------------|-----------|---------|-------------|
| gpu_temp_avg | ? | batch_size, epochs | gpu_total, cpu_total | batch_size→temp→gpu_total |
| gpu_util_avg | ? | batch_size | gpu_total | batch_size→util→gpu_total |
| gpu_avg_watts | ? | ... | ... | ... |
| gpu_min_watts | ? | ... | ... | ... |
| gpu_max_watts | ? | ... | ... | ... |
| (duration_seconds) | TBD | ... | ... | 待补充数据后分析 |

**表5: 主要因果路径（RQ2a）**

| 路径 | 组覆盖 | 路径强度 | 路径ATE估计 | 显著性 |
|-----|-------|---------|------------|--------|
| batch_size → gpu_temp_avg → gpu_total | ?/6 | ? | 可选 | ... |
| epochs → gpu_util_avg → gpu_total | ?/6 | ? | 可选 | ... |

### 5.3 RQ2b：交互项中介路径对比

#### 数据与过滤
- 主要数据: 全局标准化交互项路径
- 验证数据: 分层数据

#### 分析方法
1. 提取 `*_x_is_parallel → mediator → energy` 路径
2. 与RQ2a的主效应路径对比中介变量差异
3. 分析并行模式如何改变中介机制

#### 输出表格

**表6: 主效应 vs 交互项中介路径对比（RQ2b）**

| 超参数 | 主效应中介路径 | 交互项中介路径 | 中介差异 | 解释 |
|-------|--------------|--------------|---------|------|
| batch_size | →gpu_temp→ | →gpu_temp→, →gpu_util→ | 交互项增加gpu_util | 并行模式下利用率更敏感 |
| epochs | →gpu_util→ | →gpu_min→ | 中介变量改变 | ... |

#### 分层Case Study

**表7: 并行vs非并行中介模式对比**

| 场景 | 主要中介变量 | 中介频率Top3 | 解释 |
|-----|-------------|-------------|------|
| Parallel | gpu_temp_avg, gpu_util_avg | ... | 温度和利用率并重 |
| Non-Parallel | gpu_avg_watts | ... | 功率主导 |

---

## 六、RQ3：能耗与性能权衡

### 6.1 研究目标

检测并分析超参数对能耗和性能产生相反影响的权衡关系，量化权衡强度。

### 6.2 权衡检测方法

**Algorithm 1: 权衡检测**
```
输入: 所有显著ATE边
输出: 权衡关系列表

条件:
1. 同一干预变量 X
2. 两个结果变量 Y1 (能耗), Y2 (性能)
3. ATE(X→Y1) 和 ATE(X→Y2) 符号相反
4. 两个ATE均通过FDR显著性检验

权衡强度: |ATE1| + |ATE2| 或 |ATE1| × |ATE2|
```

### 6.3 RQ3a：权衡统计（主效应）

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除交互项
- 目标变量: 能耗 (energy_*) vs 性能 (perf_*)

#### 输出表格

**表8: 权衡关系汇总（RQ3a）**

| 干预变量 | 总权衡数 | 能耗vs性能 | 代表性权衡 | 权衡强度 |
|---------|---------|-----------|-----------|---------|
| batch_size | ? | ? | batch_size↑: gpu_total↑, accuracy↓ | ? |
| epochs | ? | ? | epochs↑: gpu_min↑, accuracy↑ | ? |
| learning_rate | ? | ? | ... | ... |
| model_* | ? | ? | ... | ... |

**表9: 能耗vs性能权衡详情（RQ3a）**

| 干预变量 | 能耗变量 | 性能变量 | ATE_能耗 | ATE_性能 | 权衡方向 |
|---------|---------|---------|---------|---------|---------|
| batch_size | gpu_total | accuracy | +0.15 | -0.08 | 增大batch_size: 耗能↑, 精度↓ |
| ... | ... | ... | ... | ... | ... |

### 6.4 RQ3b：交互项权衡与场景差异

#### 数据与过滤
- 主要数据: 全局标准化交互项
- 验证数据: 分层数据

#### 分析方法
1. 检测交互项导致的权衡关系
2. 对比主效应权衡 vs 交互项权衡
3. 分析并行模式如何影响权衡

#### 输出表格

**表10: 交互项权衡（RQ3b）**

| 交互项 | 总权衡数 | 能耗vs性能 | 代表性权衡 |
|-------|---------|-----------|-----------|
| batch_size×parallel | ? | ? | ... |
| epochs×parallel | ? | ? | ... |

**表11: 主效应 vs 交互项权衡对比（RQ3b）**

| 超参数 | 主效应权衡数 | 交互项权衡数 | 新增权衡 | 消失权衡 |
|-------|------------|------------|---------|---------|
| batch_size | ? | ? | ... | ... |
| epochs | ? | ? | ... | ... |

#### 分层Case Study

**表12: 并行vs非并行权衡对比**

| 场景 | 总权衡数 | 能耗vs性能 | 主要发现 |
|-----|---------|-----------|---------|
| group1_parallel | 15 | 2 | 权衡丰富 |
| group1_non_parallel | 0 | 0 | 无显著权衡 |
| group3_parallel | 0 | 0 | ... |
| group3_non_parallel | 0 | 0 | ... |

**关键发现**: 并行训练场景存在更多权衡关系，可能原因：
1. 资源竞争加剧
2. 同步开销引入新的权衡维度
3. ...

---

## 七、技术实现

### 7.1 脚本清单

| 脚本 | 功能 | 输入 | 输出 |
|-----|------|-----|------|
| `extract_hyperparam_energy_edges.py` | RQ1边提取 | DiBS图, ATE | CSV |
| `extract_causal_paths.py` | RQ2路径提取 | DiBS图 | 路径列表 |
| `compute_mediator_frequency.py` | RQ2中介统计 | 路径列表 | 频率表 |
| `detect_tradeoffs.py` | RQ3权衡检测 | ATE结果 | 权衡列表 |
| `generate_rq_tables.py` | 表格生成 | 各RQ结果 | Markdown/LaTeX |
| `visualize_results.py` | 可视化 | 各RQ结果 | 图表 |

### 7.2 可视化方案

| RQ | 图表类型 | 说明 |
|----|---------|------|
| RQ1 | 热力图 | 超参数×能耗类型，颜色=ATE方向和大小 |
| RQ2 | Sankey图/路径图 | 超参数→中介→能耗的流向 |
| RQ3 | 散点图 | ATE_能耗 vs ATE_性能，按干预变量着色 |
| RQ3 | 条形图 | 各干预变量的权衡数量 |

---

## 八、局限性与注意事项

### 8.1 样本量限制
- 各组样本量较小（93-178），样本/变量比处于边界条件
- 结果解释需谨慎，建议报告置信区间

### 8.2 间接ATE估计
- 乘积法假设线性传递，无交互效应
- 如使用，需明确标注局限性

### 8.3 稳定性
- group1_parallel的DiBS稳定边为0，使用平均因果图强边
- 可靠性相对较低，需在结果中说明

### 8.4 待补充数据
- 运行时间变量（`duration_seconds`）未包含在当前分析
- 后续可补充作为重要中间变量

---

## 九、待确认问题

1. **间接ATE**: RQ2中是否计算路径ATE（乘积法），还是仅报告路径存在性？
2. **路径深度**: 2-hop还是2-3-hop？
3. **权衡强度**: 使用 |ATE1|+|ATE2| 还是 |ATE1|×|ATE2|？
4. **运行时间**: 是否需要将 `duration_seconds` 加入数据作为中间变量？

---

## 十、参考文献

- DiBS: Lorch et al., "DiBS: Differentiable Bayesian Structure Learning"
- DML: Chernozhukov et al., "Double/Debiased Machine Learning"
- 权衡检测参考: [待补充论文信息]

---

**文档历史**:
- v1.0 (2026-02-07): 初始版本，待审查
