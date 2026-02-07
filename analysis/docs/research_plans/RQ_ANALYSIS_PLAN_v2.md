# Energy DL 分层因果分析研究方案

**版本**: v2.0
**日期**: 2026-02-07
**状态**: 审查修订版

---

## 一、研究背景与目标

### 1.1 研究背景

Energy DL项目研究深度学习训练中超参数对能耗和性能的因果影响。本研究基于已完成的分层因果分析结果，按训练场景（并行vs非并行）分层，研究不同场景下因果结构的异质性。

### 1.2 研究问题

| 研究问题 | 核心内容 | 子问题 |
|---------|---------|--------|
| **RQ1** | 超参数对能耗的影响力方向和大小 | a) 主效应 b) 调节效应对比 |
| **RQ2** | 利用中间变量解释因果发现 | a) 中介路径 b) 调节效应下的中介对比 |
| **RQ3** | 能耗和性能的权衡关系 | a) 权衡检测 b) 调节效应下的权衡差异 |

---

## 二、数据源设计

### 2.1 数据概览

| 数据集 | 位置 | 样本数 | 特征数 | 用途 |
|-------|------|-------|-------|------|
| 全局标准化数据 | `data/energy_research/6groups_global_std/` | 6组，共818样本 | 35-50列 | RQa/RQb主分析 |
| 分层数据 | `data/energy_research/stratified/` | 4层 | 26-28列 | Case Study验证 |

### 2.2 RQ数据分配

| RQ | RQa 数据源 | RQb 数据源 |
|----|-----------|-----------|
| RQ1 | 全局标准化（排除交互项） | 分层数据对比（调节效应 = ATE_parallel - ATE_non_parallel） |
| RQ2 | 全局标准化（排除交互项） | 分层数据对比（中介路径差异） |
| RQ3 | 全局标准化（排除交互项） | 分层数据对比（权衡差异） |

**注**: 分层数据仅覆盖group1和group3的部分场景，作为Case Study使用，不作为主要结论依据。

### 2.3 变量分类（修订版）

| 类别 | 变量 | 角色 | 说明 |
|-----|------|------|------|
| **超参数 (hyperparam)** | `hyperparam_batch_size`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_dropout`, `hyperparam_l2_regularization`, `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter` | 处理变量 | 可操控的训练配置 |
| **交互项 (interaction)** | `hyperparam_*_x_is_parallel` | 调节项 | 用于检验调节效应 |
| **中间变量 (mediator)** | `energy_gpu_temp_avg_celsius`, `energy_gpu_temp_max_celsius`, `energy_gpu_util_avg_percent`, `energy_gpu_util_max_percent` | 机制中介 | GPU物理状态变量 |
| **能耗指标 (energy)** | `energy_cpu_pkg_joules`, `energy_cpu_ram_joules`, `energy_gpu_total_joules`, `energy_cpu_total_joules`, `energy_gpu_avg_watts`, `energy_gpu_min_watts`, `energy_gpu_max_watts` | 结果变量 | 能耗测量（含功率指标） |
| **性能结果 (performance)** | `perf_test_accuracy`, `perf_map`, `perf_rank1`, `perf_rank5` | 结果变量 | 模型性能指标 |
| **控制变量 (control)** | `model_mnist_ff`, `model_mnist_rnn`, `model_siamese`, `model_hrnet18`, `model_pcb` | 协变量 | 模型类型指示变量 |
| **模式变量 (mode)** | `is_parallel` | 分层变量 | 并行/非并行模式 |

**修订说明**:
- 功率变量（`energy_gpu_*_watts`）从"中间变量"重新归类为"能耗指标"
- 中间变量仅保留真正的物理状态变量（温度、利用率）
- 运行时间变量（`duration_seconds`）待后续补充

---

## 三、方法论

### 3.1 因果图学习

- **方法**: DiBS (Differentiable Bayesian Structure Learning)
- **敏感性分析**: 多次随机运行，识别稳定边
- **边强度阈值**: 0.3（参考Ke et al., 2023）

### 3.2 因果效应估计

- **方法**: DML (Double Machine Learning)
- **输出**: ATE (Average Treatment Effect) 及置信区间
- **显著性校正**: BH-FDR (α=0.10)

### 3.3 调节效应估计

- **定义**: 调节效应 = ATE(parallel) - ATE(non-parallel)
- **方法**: 基于分层数据分别估计ATE，计算差值
- **验证**: 与交互项ATE进行一致性检验

### 3.4 路径分析

- **直接边**: source → target，强度≥0.3
- **间接路径**: source → mediator → target，每段强度≥0.3
- **路径深度**: 2-hop和3-hop
- **策略**: 仅报告路径存在性，不估计间接ATE（保守策略）

### 3.5 权衡检测

- **方法**: Algorithm 1 (opposite signs规则，参考相关文献)
- **条件**: 同一干预变量对两个结果的ATE符号相反且均显著
- **阈值**: 边强度≥0.3（参考文献标准）

---

## 四、RQ1：超参数对能耗的影响

### 4.1 研究目标

量化各超参数对能耗指标的**直接**因果影响方向和大小。

### 4.2 RQ1a：超参数主效应

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除所有 `*_x_is_parallel` 交互项
- 阈值: 边强度 ≥ 0.3

#### 分析方法
1. 提取 `hyperparam_* → energy_*` **直接边**
2. 按 (超参数, 能耗类型) 汇总统计
3. **不计算间接路径**（间接路径在RQ2中分析）

#### 输出指标
- **边计数 (n)**: 有多少组存在该直接边
- **ATE统计**: 均值 ± 标准差
- **显著性比例**: n条边中有多少通过FDR校正
- **组覆盖**: 哪些组存在该边

#### 输出表格

**表1: 超参数对能耗的直接因果影响汇总（RQ1a）**

| 超参数 | CPU_pkg | CPU_ram | GPU_total | CPU_total | GPU功率指标 |
|-------|---------|---------|-----------|-----------|------------|
| batch_size | n=k, ATE=μ±σ | ... | ... | ... | ... |
| epochs | ... | ... | ... | ... | ... |
| learning_rate | ... | ... | ... | ... | ... |
| seed | ... | ... | ... | ... | ... |
| dropout | ... | ... | ... | ... | ... |

单元格格式: `n=k, ATE=均值±标准差`
- n = 直接边计数（存在该边的组数）
- `—` 表示无组发现该直接边

#### 补充图表

**图1: 超参数-能耗因果效应热力图**
- 6组分别展示，便于比较任务间差异
- 颜色 = ATE方向和大小
- 标记显著性

**图2: 6组任务间ATE异质性**
- 森林图(Forest Plot)展示同一超参数在6组中的ATE及置信区间
- 便于识别任务特异性效应

### 4.3 RQ1b：调节效应（并行vs非并行）

#### 数据与方法
- 数据源: 分层数据 (group1_parallel vs group1_non_parallel, group3_parallel vs group3_non_parallel)
- 调节效应计算: **Δ ATE = ATE(parallel) - ATE(non_parallel)**

#### 分析步骤
1. 从分层数据提取 `hyperparam_* → energy_*` 直接边
2. 计算同一边在parallel和non_parallel场景下的ATE差值
3. 检验差值的统计显著性

#### 输出表格

**表2: 调节效应分析（RQ1b）- Case Study**

| 超参数 | 能耗变量 | ATE (Parallel) | ATE (Non-Parallel) | Δ ATE | 调节方向 |
|-------|---------|----------------|-------------------|-------|---------|
| batch_size | GPU_total | ... | ... | ... | 并行增强/减弱 |
| epochs | GPU_total | ... | ... | ... | ... |

**注**: 分层数据仅覆盖部分组，结论适用范围有限。

#### 补充图表

**图3: 调节效应对比图**
- 配对柱状图：Parallel vs Non-Parallel ATE
- 误差线显示置信区间

---

## 五、RQ2：中间变量解释

### 5.1 研究目标

识别超参数如何通过中间变量（GPU温度、利用率）影响最终能耗，解释因果机制。

### 5.2 RQ2a：因果路径与中介分析

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除交互项
- 中间变量: 仅 `gpu_temp_*`, `gpu_util_*`（物理状态变量）

#### 分析方法

**Step 1: 路径提取**
```
输入: DiBS邻接矩阵（阈值0.3）
输出: 所有 hyperparam → mediator → energy 路径

算法:
1. 构建有向图 G
2. 对每个 (hyperparam, energy) 对:
   - BFS搜索所有经过mediator的路径（深度2-3）
   - 记录路径和各边强度
3. 计算路径强度 = min(边强度)
4. 按路径强度排序
```

**Step 2: 中介频率统计**
- 统计各中间变量作为中介的出现次数
- 记录涉及的超参数和能耗变量

**Step 3: 路径报告（保守策略）**
- 仅报告路径存在性和路径强度
- **不计算间接ATE**（避免乘积法的强假设）

#### 输出表格

**表3: 中介变量频率统计（RQ2a）**

| 中间变量 | 作为中介次数 | 涉及超参数 | 涉及能耗 | 组覆盖 |
|---------|-------------|-----------|---------|-------|
| gpu_temp_avg | ? | batch_size, epochs | gpu_total, cpu_total | ?/6 |
| gpu_temp_max | ? | ... | ... | ?/6 |
| gpu_util_avg | ? | batch_size | gpu_total | ?/6 |
| gpu_util_max | ? | ... | ... | ?/6 |

**表4: 主要因果路径（RQ2a）**

| 路径 | 组覆盖 | 路径强度 | 典型任务 |
|-----|-------|---------|---------|
| batch_size → gpu_temp_avg → gpu_total | ?/6 | min(s1,s2) | group1, group3 |
| epochs → gpu_util_avg → gpu_total | ?/6 | min(s1,s2) | ... |

#### 补充图表

**图4: 因果路径Sankey图**
- 展示超参数→中介→能耗的主要流向
- 宽度 = 路径强度或频率

**图5: 6组任务中介模式对比**
- 雷达图或热力图
- 展示不同任务的中介变量使用差异

### 5.3 RQ2b：调节效应下的中介对比

#### 数据与方法
- 数据源: 分层数据
- 对比: Parallel vs Non-Parallel 场景下的中介路径差异

#### 分析步骤
1. 分别提取 parallel 和 non_parallel 场景的因果路径
2. 对比中介变量的使用频率差异
3. 识别场景特异性中介机制

#### 输出表格

**表5: 并行vs非并行中介模式对比（RQ2b）- Case Study**

| 中间变量 | Parallel中介次数 | Non-Parallel中介次数 | 差异 | 解释 |
|---------|-----------------|---------------------|------|------|
| gpu_temp_avg | ? | ? | Δ=? | ... |
| gpu_util_avg | ? | ? | Δ=? | ... |

**表6: 场景特异性路径（RQ2b）**

| 场景 | 特有路径 | 路径强度 | 可能解释 |
|-----|---------|---------|---------|
| Parallel | batch_size→temp→gpu_total | ... | 并行加剧热效应 |
| Non-Parallel | epochs→util→gpu_total | ... | 串行下利用率主导 |

---

## 六、RQ3：能耗与性能权衡

### 6.1 研究目标

检测并分析超参数对能耗和性能产生相反影响的权衡关系，量化权衡强度。

### 6.2 权衡检测方法

**Algorithm 1: 权衡检测**（参考Ke et al., 2023）
```
输入: 所有显著ATE边（边强度≥0.3）
输出: 权衡关系列表

条件:
1. 同一干预变量 X
2. 两个结果变量 Y1 (能耗), Y2 (性能)
3. ATE(X→Y1) 和 ATE(X→Y2) 符号相反
4. 两个ATE均通过FDR显著性检验
5. 两条边强度均≥0.3

权衡强度: |ATE1| + |ATE2|
```

### 6.3 RQ3a：权衡统计（主效应）

#### 数据与过滤
- 数据源: 全局标准化6组
- 过滤: 排除交互项
- 目标: 能耗 (energy_*) vs 性能 (perf_*)

#### 输出表格

**表7: 权衡关系汇总（RQ3a）**

| 干预变量 | 总权衡数 | 能耗vs性能 | 代表性权衡 | 权衡强度 |
|---------|---------|-----------|-----------|---------|
| batch_size | ? | ? | batch_size↑: gpu_total↑, accuracy↓ | |ATE1|+|ATE2| |
| epochs | ? | ? | ... | ... |
| learning_rate | ? | ? | ... | ... |

**表8: 能耗vs性能权衡详情（RQ3a）**

| 干预变量 | 能耗变量 | 性能变量 | ATE_能耗 | ATE_性能 | 权衡方向 | 组覆盖 |
|---------|---------|---------|---------|---------|---------|-------|
| batch_size | gpu_total | accuracy | +0.15 | -0.08 | 增大→耗能↑精度↓ | ?/6 |
| ... | ... | ... | ... | ... | ... | ... |

#### 补充图表

**图6: 权衡散点图**
- X轴 = ATE_能耗, Y轴 = ATE_性能
- 按干预变量着色
- 权衡区域 = 第二、四象限

**图7: 6组任务权衡模式对比**
- 堆叠柱状图：各组的权衡数量分布
- 展示任务间权衡异质性

### 6.4 RQ3b：调节效应下的权衡差异

#### 数据与方法
- 数据源: 分层数据
- 对比: Parallel vs Non-Parallel 场景下的权衡差异

#### 分析步骤
1. 分别检测 parallel 和 non_parallel 场景的权衡
2. 计算权衡数量差异
3. 识别场景特异性权衡

#### 输出表格

**表9: 并行vs非并行权衡对比（RQ3b）- Case Study**

| 场景 | 总权衡数 | 能耗vs性能 | 主要发现 |
|-----|---------|-----------|---------|
| group1_parallel | 15 | 2 | 权衡丰富 |
| group1_non_parallel | 0 | 0 | 无显著权衡 |
| group3_parallel | 0 | 0 | ... |
| group3_non_parallel | 0 | 0 | ... |

**表10: 场景特异性权衡（RQ3b）**

| 权衡 | 仅Parallel | 仅Non-Parallel | 两者共有 |
|-----|-----------|---------------|---------|
| batch_size: energy↔perf | ✓ | — | — |
| epochs: energy↔perf | ✓ | — | — |

**关键发现**: 并行训练场景存在更多权衡关系

**可能解释**:
1. 资源竞争加剧：多进程共享GPU导致温度和利用率波动
2. 同步开销：数据并行的同步点引入额外能耗
3. 批量效应：并行场景下有效batch size更大，影响收敛和能耗

---

## 七、技术实现

### 7.1 脚本清单

| 脚本 | 功能 | 输入 | 输出 |
|-----|------|-----|------|
| `extract_direct_edges.py` | RQ1直接边提取 | DiBS图, ATE | CSV |
| `compute_moderation_effect.py` | RQ1b调节效应 | 分层ATE | CSV |
| `extract_causal_paths.py` | RQ2路径提取 | DiBS图 | 路径列表 |
| `compute_mediator_frequency.py` | RQ2中介统计 | 路径列表 | 频率表 |
| `detect_tradeoffs.py` | RQ3权衡检测 | ATE结果 | 权衡列表 |
| `generate_rq_tables.py` | 表格生成 | 各RQ结果 | Markdown/LaTeX |
| `visualize_results.py` | 可视化 | 各RQ结果 | 图表 |

### 7.2 图表清单

| 图编号 | 图表类型 | RQ | 说明 |
|-------|---------|-----|------|
| 图1 | 热力图 | RQ1a | 超参数×能耗，6组分面展示 |
| 图2 | 森林图 | RQ1a | 6组ATE异质性对比 |
| 图3 | 配对柱状图 | RQ1b | Parallel vs Non-Parallel ATE |
| 图4 | Sankey图 | RQ2a | 因果路径流向 |
| 图5 | 雷达图/热力图 | RQ2a | 6组中介模式对比 |
| 图6 | 散点图 | RQ3a | ATE_能耗 vs ATE_性能 |
| 图7 | 堆叠柱状图 | RQ3a | 6组权衡数量分布 |

---

## 八、局限性与注意事项

### 8.1 样本量限制
- 各组样本量较小（93-178），样本/变量比处于边界条件
- 结果解释需谨慎，所有结论均报告置信区间

### 8.2 分层数据覆盖
- 分层数据仅覆盖group1和group3
- RQb结论作为Case Study，不代表所有任务

### 8.3 间接效应
- 本研究采用保守策略，仅报告间接路径存在性
- 不计算间接ATE，避免乘积法的强假设偏误

### 8.4 稳定性
- group1_parallel的DiBS稳定边为0，使用平均因果图强边
- 可靠性相对较低，已在结果中标注

### 8.5 外部效度
- 结果基于特定硬件环境（单GPU服务器）
- 推广到其他硬件配置需谨慎

---

## 九、参考文献

- DiBS: Lorch et al., "DiBS: Differentiable Bayesian Structure Learning", NeurIPS 2021
- DML: Chernozhukov et al., "Double/Debiased Machine Learning for Treatment and Structural Parameters", The Econometrics Journal, 2018
- 权衡检测与0.3阈值: Ke et al., 2023 (arXiv:2305.13057)

---

## 十、修订历史

| 版本 | 日期 | 修订内容 |
|-----|------|---------|
| v1.0 | 2026-02-07 | 初始版本 |
| v2.0 | 2026-02-07 | 根据同行评审修订：1)功率变量归类为能耗指标 2)采用保守策略不计算间接ATE 3)明确调节效应计算方法 4)增加6组异质性图表 5)分层数据定位为Case Study |

---

**文档状态**: 审查修订完成，待实施
