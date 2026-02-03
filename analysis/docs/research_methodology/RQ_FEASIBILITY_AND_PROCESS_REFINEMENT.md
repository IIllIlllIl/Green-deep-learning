# 研究问题可行性评估与流程细化

**版本**: v1.0 | **创建**: 2026-02-03 | **依据**: 数据探索结果 + RQ_RESEARCH_METHODOLOGY.md

> **目标**: 基于实际数据现状，评估四个研究问题的可行性，并细化具体的数据分析流程

---

## 📊 可行性评估摘要

| RQ | 问题 | 可行性 | 关键依据 | 建议 |
|----|------|--------|----------|------|
| **RQ1** | 超参数对能耗的影响力方向和大小 | ⚠️ **中等偏低** | 仅3条超参数→能耗直接因果边，其中1条ATE不显著 | 结合交互项分析或考虑间接路径 |
| **RQ2** | 利用中间变量解释发现 | ⚠️ **中等** | 能耗中间变量丰富，但缺乏训练时间等传统中间变量 | 聚焦能耗变量间的中介作用 |
| **RQ3** | 能耗和性能的权衡关系 | ✅ **高** | 14个显著的能耗vs性能权衡，数据质量好 | 优先进行深入分析 |
| **RQ4** | 并行/非并行场景差异 | ✅ **高** | 交互项数据丰富，多个显著交互项因果边 | 重点挖掘并行模式调节效应 |

**总体建议**: 优先完成RQ3和RQ4，对RQ1和RQ2采用调整后的分析方法。

---

## 🔍 详细数据发现

### 1. 全局标准化数据结构（818样本×50列）

**变量分类**:
- **超参数 (17个)**: `hyperparam_batch_size`, `hyperparam_learning_rate`, `hyperparam_epochs`, `hyperparam_seed`, `hyperparam_l2_regularization`, `hyperparam_dropout`, `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter` 等
- **能耗指标 (11个)**: `energy_cpu_pkg_joules`, `energy_cpu_ram_joules`, `energy_cpu_total_joules`, `energy_gpu_avg_watts`, `energy_gpu_max_watts`, `energy_gpu_min_watts`, `energy_gpu_total_joules`, `energy_gpu_temp_avg_celsius`, `energy_gpu_temp_max_celsius`, `energy_gpu_util_avg_percent`, `energy_gpu_util_max_percent`
- **性能指标 (15个)**: `perf_test_accuracy`, `perf_eval_loss`, `perf_final_training_loss`, `perf_eval_samples_per_second`, `perf_map`, `perf_rank1`, `perf_rank5`, `perf_top1_accuracy`, `perf_top5_accuracy`, `perf_top10_accuracy`, `perf_top20_accuracy`, `perf_accuracy`, `perf_precision`, `perf_recall`, `perf_best_val_accuracy`
- **交互项 (8个)**: `hyperparam_batch_size_x_is_parallel`, `hyperparam_learning_rate_x_is_parallel` 等

### 2. 因果边与ATE分析

**关键发现**:
- **超参数→能耗直接边**: 仅3条（group1: 2条, group3: 1条）
- **交互项→能耗边**: 丰富且显著，强度0.3-0.95，ATE完整
- **ATE数据问题**: 许多边的ATE值为空，需要验证计算方法

### 3. 权衡检测结果

**统计摘要**:
- 总因果边数: 217条
- 检测到的权衡: 37个（17.1%）
- **能耗vs性能权衡**: 14个（占权衡的37.8%）
- 主要分布在group1（12个）、group3（1个）、group5（1个）

### 4. 交互项数据

**丰富性**: 多个超参数×is_parallel交互项到能耗指标的直接因果边
**质量**: 强度高（0.3-0.95），ATE值完整且显著

---

## 📋 细化数据分析流程

### 总体调整原则

1. **优先级调整**: RQ3和RQ4优先，RQ1和RQ2采用调整方法
2. **方法适应性**: 根据数据现状调整分析方法
3. **验证机制**: 增加数据质量检查和交叉验证步骤
4. **输出标准化**: 统一输出格式和可视化风格

### Phase 0: 数据准备与验证（1天）

**目标**: 验证数据质量，创建分析环境

**任务**:
1. **数据质量检查**:
   ```bash
   # 验证ATE数据完整性
   python3 analysis/scripts/data/verify_ate_completeness.py

   # 检查权衡数据一致性
   python3 analysis/scripts/data/verify_tradeoff_consistency.py
   ```
2. **环境准备**:
   ```bash
   # 创建输出目录
   mkdir -p analysis/results/energy_research/rq_analysis/{rq1,rq2,rq3,rq4}

   # 准备分析配置文件
   cp analysis/docs/templates/rq_analysis_config.yaml analysis/config/rq_analysis/
   ```

### Phase 1: RQ1调整分析 - 超参数效应（1.5天）

**调整方案**: 结合直接效应和交互项效应分析

**分析步骤**:
1. **直接因果边提取**:
   - 从6组ATE文件中提取所有超参数→能耗的边
   - 筛选强度>0.3且ATE显著的边
   - 生成直接效应表格

2. **交互项效应分析**:
   - 提取交互项（hyperparam_x_is_parallel）→能耗的边
   - 计算交互项效应的强度
   - 生成交互项效应表格

3. **综合效应评估**:
   - 合并直接效应和交互项效应
   - 按效应强度排序，识别Top 10关键效应
   - 生成综合效应报告

**输出文件**:
- `rq1_direct_effects.csv`: 直接因果边列表
- `rq1_interaction_effects.csv`: 交互项效应列表
- `rq1_combined_effects_ranked.csv`: 综合效应排序
- `rq1_top_effects_visualization.png`: Top 10效应可视化

### Phase 2: RQ2调整分析 - 能耗变量中介作用（1.5天）

**调整方案**: 聚焦能耗变量间的中介作用

**分析步骤**:
1. **能耗变量网络分析**:
   - 提取所有能耗变量间的因果边（energy_→energy_）
   - 构建能耗变量因果网络
   - 识别关键中介节点（高中心性节点）

2. **超参数通过能耗网络的路径分析**:
   - 查找超参数→能耗变量A→能耗变量B的路径
   - 计算路径强度和ATE传递
   - 识别重要传递路径

3. **中介重要性评估**:
   - 统计每个能耗变量在中介路径中的出现频率
   - 计算中介贡献度
   - 生成中介重要性排序

**输出文件**:
- `rq2_energy_network_edges.csv`: 能耗变量因果边
- `rq2_mediation_paths.csv`: 中介路径列表
- `rq2_mediator_importance.csv`: 中介变量重要性排序
- `rq2_energy_network_graph.png`: 能耗因果网络图

### Phase 3: RQ3分析 - 权衡关系（1.5天）

**分析步骤**:
1. **权衡分类与整理**:
   - 读取37个已识别权衡
   - 按类型分类：能耗vs性能、能耗vs能耗、性能vs性能
   - 计算每个权衡的强度（|ATE1 - ATE2|）

2. **典型案例分析**:
   - 选择3-5个强度最高的能耗vs性能权衡
   - 深入分析权衡的机制和含义
   - 生成案例分析报告

3. **跨任务组一致性分析**:
   - 检查相同超参数在不同任务组中的权衡模式
   - 识别一致性和差异性
   - 生成一致性矩阵

**输出文件**:
- `rq3_tradeoff_classification.csv`: 权衡分类表
- `rq3_tradeoff_strength_ranking.csv`: 权衡强度排序
- `rq3_case_studies_report.md`: 典型案例分析报告
- `rq3_cross_group_consistency_matrix.csv`: 跨组一致性矩阵

### Phase 4: RQ4分析 - 并行模式调节效应（1.5天）

**分析步骤**:
1. **交互项效应提取**:
   - 提取所有交互项（hyperparam_x_is_parallel）→能耗/性能的边
   - 计算调节强度（交互项边的强度）
   - 识别显著调节效应

2. **主效应vs调节效应对比**:
   - 比较超参数主效应和交互项调节效应
   - 计算调节比例（交互效应/主效应）
   - 识别受场景显著调节的超参数

3. **场景差异总结**:
   - 总结并行vs非并行场景的关键差异
   - 识别场景特定的优化建议
   - 生成场景差异报告

**输出文件**:
- `rq4_interaction_effects.csv`: 交互项效应列表
- `rq4_moderation_strength_ranking.csv`: 调节强度排序
- `rq4_main_vs_interaction_comparison.csv`: 主效应vs交互效应对比
- `rq4_scenario_differences_report.md`: 场景差异报告

### Phase 5: 整合与撰写（2天）

**任务**:
1. **结果整合**:
   - 统一所有RQ结果的格式和风格
   - 创建综合结果表格
   - 生成整合可视化图表

2. **研究报告撰写**:
   - 撰写综合研究报告
   - 总结主要发现和贡献
   - 提出实践建议和未来方向

3. **代码和文档整理**:
   - 整理分析代码，确保可复现性
   - 更新项目文档
   - 创建分析流程文档

**输出文件**:
- `comprehensive_rq_analysis_report.md`: 综合研究报告
- `all_rq_results_summary.csv`: 所有RQ结果摘要
- `analysis_workflow_documentation.md`: 分析流程文档

---

## ⚠️ 数据质量与风险管控

### 已知问题
1. **ATE空值问题**: 部分因果边的ATE值为空
   - **缓解**: 使用强度(strength)作为替代指标，增加数据质量检查
2. **样本分布不均**: 某些任务组样本量较小
   - **缓解**: 聚焦大样本组（group1: 304, group3: 206），增加统计检验
3. **中介变量限制**: 缺乏传统中间变量（如训练时间）
   - **缓解**: 聚焦能耗变量间的中介作用

### 验证机制
1. **交叉验证**: 使用DiBS因果图和ATE结果相互验证
2. **统计显著性检查**: 所有报告效应必须统计显著（p<0.05）
3. **敏感性分析**: 对关键发现进行敏感性分析

### 成功标准（调整后）

| RQ | 必须完成（P0） | 加分项（P1） |
|----|---------------|-------------|
| **RQ1** | 识别至少5个显著超参数效应（直接或交互项） | 发现效应强度的任务组模式 |
| **RQ2** | 识别至少3个关键能耗中介变量 | 量化中介路径的贡献比例 |
| **RQ3** | 完成14个能耗vs性能权衡的深入分析 | 发现权衡强度的系统性模式 |
| **RQ4** | 识别至少3个受场景显著调节的超参数 | 提供场景特定的优化建议 |

---

## 🗓️ 时间安排（总计8.5天）

| 阶段 | 任务 | 天数 | 状态 |
|------|------|------|------|
| **Phase 0** | 数据准备与验证 | 1.0 | 待开始 |
| **Phase 1** | RQ1调整分析 | 1.5 | 待开始 |
| **Phase 2** | RQ2调整分析 | 1.5 | 待开始 |
| **Phase 3** | RQ3分析 | 1.5 | 待开始 |
| **Phase 4** | RQ4分析 | 1.5 | 待开始 |
| **Phase 5** | 整合与撰写 | 2.0 | 待开始 |
| **总计** | | **8.5天** | |

---

## 📁 文件结构

### 输入数据（已完成）
```
analysis/data/energy_research/6groups_global_std/          # 全局标准化数据
analysis/results/energy_research/data/global_std/         # DiBS因果图
analysis/results/energy_research/data/global_std_dibs_ate/ # ATE结果
analysis/results/energy_research/tradeoff_detection/      # 权衡检测结果
analysis/results/energy_research/data/interaction/        # 交互项数据
```

### 输出结果（待生成）
```
analysis/results/energy_research/rq_analysis/
├── rq1_hyperparam_effects/          # RQ1结果
├── rq2_mediator_analysis/           # RQ2结果
├── rq3_tradeoff_analysis/           # RQ3结果
├── rq4_scenario_analysis/           # RQ4结果
└── integrated_results/              # 整合结果
```

### 分析脚本（待创建）
```
analysis/scripts/rq_analysis/
├── phase0_data_preparation.py
├── phase1_rq1_analysis.py
├── phase2_rq2_analysis.py
├── phase3_rq3_analysis.py
├── phase4_rq4_analysis.py
└── phase5_integration.py
```

---

## 🔧 技术实现要点

### 关键分析函数
1. **ATE数据提取与清洗**: 处理空值，验证统计显著性
2. **因果路径枚举**: 支持长度为2-3的路径搜索
3. **网络分析**: 计算节点中心性，识别关键节点
4. **交互项效应计算**: 分离主效应和交互效应
5. **权衡强度计算**: 基于ATE差值计算权衡强度

### 可视化需求
1. **效应强度柱状图**: 显示Top 10效应
2. **因果网络图**: 可视化能耗变量网络
3. **权衡矩阵热图**: 显示权衡关系
4. **交互效应对比图**: 主效应vs交互效应对比

### 质量保证
1. **单元测试**: 每个分析函数都有对应的单元测试
2. **集成测试**: 验证整个分析流程的可复现性
3. **结果验证**: 人工抽样检查关键结果

---

## 📞 下一步行动

### 立即行动
1. **创建分析脚本框架**: 按上述文件结构创建脚本目录
2. **编写Phase 0脚本**: 实现数据质量检查和环境准备
3. **开始Phase 1分析**: 从RQ1调整分析开始

### 中长期计划
1. **按阶段顺序完成所有分析**
2. **定期检查进展和质量**
3. **根据初步结果调整后续分析**

---

**维护者**: Green | **版本**: v1.0 | **创建**: 2026-02-03

**版本历史**:
- v1.0 (2026-02-03): 基于数据探索结果创建，细化分析流程