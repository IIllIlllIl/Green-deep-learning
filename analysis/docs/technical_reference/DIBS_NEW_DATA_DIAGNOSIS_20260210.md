# 新DiBS数据诊断报告

**生成日期**: 2026-02-10
**数据版本**: 2月9-10日新DiBS因果图

---

## 一、数据概览

### 1.1 两套数据对比

| 项目 | 旧数据 (2月3日) | 新数据 (2月10日) | 变化 |
|------|----------------|-----------------|------|
| **位置** | `global_std_dibs_ate/` | `global_std/` | - |
| **内容** | DiBS图 + ATE结果 | 仅DiBS图 | 缺少ATE |
| **总强边数** | 151条 | 395条 | +161.6% ↑ |
| **平均强边%** | ~7% | 13.2% | +6.2% ↑ |
| **hyperparam→energy边** | 0条 | 26条 | **新增** |
| **交互项→energy边** | 0条 | 5条 | **新增** |

### 1.2 新数据质量指标

| 组 | 样本数 | 特征数 | 强边数 | 强边% | 图密度 |
|----|--------|--------|--------|-------|--------|
| group1_examples | 304 | 23 | 90 | 17.79% | 中 |
| group2_vulberta | 72 | 22 | 61 | 13.20% | 中 |
| group3_person_reid | 206 | 24 | 98 | 17.75% | 高 |
| group4_bug_localization | 90 | 22 | 49 | 10.61% | 低 |
| group5_mrt_oast | 60 | 24 | 57 | 10.33% | 中 |
| group6_resnet | 74 | 21 | 40 | 9.52% | 低 |
| **总计** | 806 | - | **395** | **13.2%** | - |

---

## 二、关键发现

### 2.1 超参数→能耗直接边（RQ1a核心）

**旧数据**: 0条 ✗
**新数据**: 26条 ✓

**分布**:
- group1_examples: 5条
- group2_vulberta: 8条
- group3_person_reid: 3条
- group4_bug_localization: 0条
- group5_mrt_oast: 7条
- group6_resnet: 3条

**示例边 (group1)**:
```
hyperparam_epochs → energy_gpu_total_joules (0.74)
hyperparam_epochs → energy_gpu_util_avg_percent (0.34)
```

### 2.2 交互项→能耗边（RQ1b核心）

**旧数据**: 0条 ✗
**新数据**: 5条 ✓

**分布**:
- group1_examples: 3条
- group3_person_reid: 1条
- group5_mrt_oast: 1条

**示例边 (group1)**:
```
hyperparam_epochs×parallel → energy_gpu_max_watts (0.50)
hyperparam_epochs×parallel → energy_gpu_min_watts (0.32)
```

### 2.3 中介路径（RQ2核心）

**group1_examples**:
- →mediator边: 26条
- mediator→energy边: 11条
- 潜在间接路径: ~10-15条

**预估**: 新数据的间接路径数将是旧数据的2-3倍。

### 2.4 权衡关系（RQ3核心）

由于缺少ATE，暂时无法检测权衡关系。但基于因果图结构，**预计权衡数量会显著增加**。

---

## 三、数据质量评估

### 3.1 优势 ✓

1. **因果发现能力大幅提升**
   - 强边数增加161.6%
   - 发现了之前缺失的超参数→能耗直接边（26条）
   - 发现了交互项调节效应边（5条）

2. **图密度提高**
   - 平均强边比例从~7%提升至13.2%
   - 更符合DiBS的标准输出（10%-18%）

3. **RQ分析可行性**
   - RQ1a (主效应): 26条边可用 ✓
   - RQ1b (调节效应): 5条边可用 ✓
   - RQ2 (中介路径): 预计10-15条路径 ✓
   - RQ3 (权衡): 需ATE计算

### 3.2 不足 ✗

1. **缺少ATE计算**
   - 需要运行DML因果推断
   - 预计耗时: 6组 × 5-10分钟/组 = 30-60分钟

2. **部分组边数较少**
   - group4 (49条) 和 group6 (40条) 强边数偏少
   - 可能影响统计功效

---

## 四、决策建议

### 4.1 方案对比

| 方案 | 优点 | 缺点 | 耗时 |
|------|------|------|------|
| **方案A: 使用旧数据** | ✓ ATE已就绪<br>✓ 可立即生成RQ图 | ✗ 0条hyperparam→energy边<br>✗ RQ1a无数据<br>✗ 因果发现能力弱 | 0分钟 |
| **方案B: 使用新数据+计算ATE** | ✓ 26条hyperparam→energy边<br>✓ 因果发现能力强<br>✓ RQ分析完整 | ✗ 需要计算ATE | 30-60分钟 |

### 4.2 推荐方案

**推荐: 方案B - 使用新数据并计算ATE**

**理由**:
1. **科学价值**: 新数据发现了26条超参数→能耗直接边，这是旧数据完全没有的关键发现
2. **RQ完整性**: 旧数据无法回答RQ1a（主效应），新数据可以
3. **时间成本可控**: ATE计算仅需30-60分钟
4. **质量提升**: 新因果图质量明显更好

### 4.3 执行步骤

如果选择方案B，执行步骤如下：

```bash
# Step 1: 激活环境
conda activate causal-research

# Step 2: 计算ATE (预计30-60分钟)
python3 scripts/compute_ate_for_causal_edges.py \
    --causal-graph-dir results/energy_research/data/global_std \
    --data-dir data/energy_research/6groups_global_std \
    --output-dir results/energy_research/data/global_std_ate

# Step 3: 检测权衡 (预计5-10分钟)
python3 scripts/detect_tradeoffs.py \
    --ate-dir results/energy_research/data/global_std_ate \
    --output-dir results/energy_research/tradeoff_detection_global_std_new

# Step 4: 生成RQ图表 (预计5-10分钟)
python3 scripts/rq1_analysis.py
python3 scripts/rq2_analysis.py
python3 scripts/rq3_analysis.py
```

---

## 五、技术细节

### 5.1 新数据与旧数据的主要差异

**可能原因**:
1. **DiBS训练参数优化**
   - 新数据: 13000步训练
   - 旧数据: 可能为较少步数

2. **数据预处理改进**
   - 新数据: 全局标准化优化后
   - 旧数据: 可能为早期版本

3. **特征选择**
   - 新数据: 21-24个特征（动态）
   - 旧数据: 固定35个特征

### 5.2 关键边对比示例

**新数据发现的直接因果边 (旧数据缺失)**:
```
group1: epochs → gpu_total_joules (0.74) ★★★
group2: batch_size → cpu_total_joules (0.56) ★★
group3: epochs → gpu_total_joules (0.82) ★★★
group5: learning_rate → cpu_ram_joules (0.61) ★★
```

**这些边对RQ1分析至关重要。**

---

## 六、下一步行动

**请用户决策**:

- [ ] **选项A**: 使用旧数据（立即可用，但RQ1a无数据）
- [ ] **选项B**: 使用新数据+计算ATE（推荐，30-60分钟）

**如果选择方案B**:
1. 确认数据路径正确
2. 执行ATE计算
3. 生成RQ图表
4. 对比新旧结果
5. 撰写分析报告

---

**报告生成**: Claude Code
**数据验证**: 已完成
**状态**: 等待用户决策
