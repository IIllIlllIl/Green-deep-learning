# 全局标准化修复对话背景

**创建日期**: 2026-01-26
**用途**: 下一个对话的背景介绍，包含所有必要的上下文信息
**项目**: Energy DL - 深度学习训练超参数对能耗和性能影响的因果分析

---

## 🎯 项目背景

### 研究目标
本研究通过**深度学习训练超参数变异实验**，分析超参数对能耗和性能的因果影响。核心研究问题：
1. **哪些超参数对能耗影响最大？**（因果识别）
2. **并行模式如何调节超参数效应？**（交互效应分析）
3. **哪些因果效应是跨模型稳健的？**（跨模型可比性）

### 当前实验状态
- ✅ **836个实验**（795个有效能���数据，95.1%完整性）
- ✅ **11个模型**：覆盖并行/非并行模式
- ✅ **DiBS因果发现**：已完成6个组的因果图学习
- ✅ **ATE计算**：84.1%的边成功计算，100%统计显著
- ⚠️ **核心问题**：**组内标准化破坏跨组可比性**，无法回答跨模型比较问题

---

## 🚨 核心问题发现（P0优先级）

### 问题1：标准化方法错误 ⭐⭐⭐⭐⭐
**症状**：采用组���独立标准化，导致ATE跨组不可比

**影响**：
- group1: "1个单位" = 72.97 watts (energy_gpu_avg_watts标准差)
- group2: "1个单位" = 8.10 watts (**9.0倍差异**)
- ❌ 无法回答："哪个模型的超参数对能耗影响最大？"
- ❌ 无法判断：并行模式的调节效应是否跨模型一致

**根本原因**：
- 当前实现：`generate_interaction_terms_data.py:120-174` 对每组独立标准化
- 参考论文：`CTF_original/src/discovery.py:136-137` 使用全局标准化

### 问题2：缺失值处理策略不一致 ⭐⭐⭐⭐
**CTF策略**（参考论文）：
- DiBS阶段：`dropna()` 删除所有缺失行（L118-119）
- ATE阶段：`fillna(0)` 用0填充（L115-116）
- 标准化时机：**先处理缺失值，后标准化**

**当前实现问题**：
- 统一使用中位数填充（未区分阶段）
- 标准化时机不明确
- 未评估缺失值对标准化参数的影响

### 问题3：数据异常
- group2和group6的`batch_size`标准差为0.0 → 交互项方差为0

### 问题4：DiBS与CTF实现差距
- DiBS类型：JointDiBS vs MarginalDiBS
- 似然模型：LinearGaussian vs BGe
- 参数设置：n_particles=20 vs 50, steps=5000 vs 13000
- 先验知识：无干预掩码 vs CTF有干预掩码

---

## 📁 关键文件路径

### 数据文件
```
/home/green/energy_dl/nightly/analysis/data/energy_research/
├── 6groups_interaction/              # ⭐ 当前使用的数据集（组内标准化）
│   ├── group1_examples_interaction.csv
│   ├── group2_vulberta_interaction.csv
│   ├── ...
│   └── standardization_params.json   # ⭐ 各组独立标准化参数
└── 6groups_final/                    # 基础数据集（无交互项）
```

### 核心代码文件
```
/home/green/energy_dl/nightly/analysis/
├── scripts/
│   ├── generate_interaction_terms_data.py    # ⭐ 数据标准化和交互项创建（需修改）
│   ├── run_dibs_6groups_interaction.py       # DiBS主运行脚本
│   ├── compute_ate_whitelist.py              # ATE计算脚本
│   └── config_energy.py                       # 配置文件
├── utils/
│   ├── causal_discovery.py                   # DiBS封装类
│   └── causal_inference.py                   # CTF风格DML实现
└── results/energy_research/data/interaction/whitelist/
    ├── group1_whitelist.csv                  # 白名单文件（含ATE）
    └── ...
```

### 参考论文代码
```
/home/green/energy_dl/nightly/analysis/CTF_original/
├── src/discovery.py              # ⭐ DiBS实现（L118-119: dropna, L136-137: 全局标准化）
├── src/inf.py                    # ⭐ ATE计算（L115-116: fillna(0)）
└── src/load_data.py              # 数据加载和预处理
```

### 关键文档
```
/home/green/energy_dl/nightly/analysis/docs/technical_reference/
└── STANDARDIZATION_AND_DIBS_ISSUE_SUMMARY.md    # ⭐⭐⭐ 完整问题总结（1.1版本）
```

---

## 🎯 修复目标与任务

### 主要目标
**实施全局标准化，恢复跨组可比性**

### 阶段1：全局标准化实施（2-3天）

#### 任务1.1：诊断缺失值模式
```bash
# 脚本：scripts/diagnose_missing_patterns.py
# 目标：分析各组缺失比例、模式、机制
# 输出：缺失值诊断报告
```

**关键检查点**：
- 各组能耗数据缺失比例（预期~40%）
- 缺失是否与关键变量相关（batch_size, is_parallel等）
- 完全可用记录数（用于评估dropna的影响）

#### 任务1.2：实现全局标准化数据生成
```bash
# 脚本：scripts/create_global_standardized_data.py
# 参数：--strategy {conservative|ctf_drop|ctf_fill}
# 目标：合并6组 → 缺失值处理 → 全局标准化 → 重建交互项
```

**实现要求**：
1. **合并所有6组数据**：`pd.concat([group1, ..., group6])`
2. **缺失值处理**：
   - 保守策略：基于全局均值/中位数填充
   - CTF删除策略：`dropna()`（用于敏感性分析）
3. **全局标准化**：`StandardScaler().fit_transform(all_data)`
4. **重建交互项**：`标准化超参数 × is_parallel`
5. **保存标准化参数**：用于反标准化和解释

#### 任务1.3：修复group2 batch_size标准差为0
```bash
# 脚本：scripts/fix_group2_batchsize_issue.py
# 目标：检查batch_size数据，修复零方差问题
```

#### 任务1.4：重新计算全局标准化ATE
```bash
# 脚本：scripts/compute_ate_whitelist_global.py
# 输入：全局标准化数据集
# 输出：更新的白名单文件（含全局标准化ATE）
```

#### 任务1.5：敏感性分析
```bash
# 脚本：scripts/sensitivity_analysis_missing_strategies.py
# 目标：对比保守填充 vs CTF删除的ATE差异
```

### 阶段2：算法对齐（3-4天，可选）

#### 任务2.1：测试MarginalDiBS + BGe组合
- 对比当前JointDiBS + LinearGaussian的结果差异

#### 任务2.2：实现干预掩码
- 参考CTF的`interv_mask`逻辑
- 对超参数变量进行特殊处理

---

## 📊 成功标准

### 必须达成（P0）
1. ✅ 全局标准化数据集成功生成
2. ✅ ATE表示"1个全局标准差变化"
3. ✅ 跨组ATE可直接比较（相同尺度）
4. ✅ 所有交互项边成功计算ATE
5. ✅ 交互项ATE尺度一致（仅因并行比例p有合理差异）

### 期望达成（P1）
1. 敏感性分析完成（保守填充 vs CTF删除）
2. group2 batch_size标准差问题修复
3. 数据完整性报告生成

### 可选达成（P2）
1. DiBS算法对齐测试完成
2. 先验知识注入机制实现

---

## 🔍 关键技术细节

### 标准化流程（CTF风格）
```python
# 1. 替换无穷值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. 缺失值处理（DiBS阶段）
df.dropna(inplace=True)

# 3. 数据打乱
df = df.sample(frac=1).reset_index(drop=True)

# 4. 全局标准化
scaler = StandardScaler()
data_std = scaler.fit_transform(df)

# 5. 保存标准化参数
standardization_params = {
    'mean': scaler.mean_.tolist(),
    'std': scaler.scale_.tolist(),
    'columns': df.columns.tolist()
}
```

### 交互项重建公式
```python
# 基于全局标准化后的超参数
interaction_col = f"{hp}_x_is_parallel"
df[interaction_col] = df[hp] * df['is_parallel']  # 标准化超参数 × 0/1
```

### ATE解释变化
| 标准化方法 | ATE解释 | 跨组比较 |
|----------|---------|----------|
| **组内标准化**（当前） | 处理变量变化1个**组内**标准差 | ❌ 不可比 |
| **全局标准化**（目标） | 处理变量变化1个**全局**标准差 | ✅ 可比 |

---

## ⚠️ 风险与注意事项

1. **样本量风险**：
   -能耗数据缺失~40%，dropna可能导致样本不足
   - 必须先诊断缺失模式，评估dropna的影响

2. **数据质量风险**：
   - 保守填充可能引入偏差
   - 需进行敏感性分析验证稳健性

3. **交互项尺度差异**：
   - 即使全局标准化，不同组的并行比例p仍会导致差异
   - 这是**真实的数据特性**，不应消除

4. **group2 batch_size标准差为0**：
   - 可能是数据bug（常数或单一值）
   - 需在全局标准化前修复

---

## 📋 执行检查清单

### 准备阶段
- [ ] 阅读完整问题总结：`STANDARDIZATION_AND_DIBS_ISSUE_SUMMARY.md`
- [ ] 检查当前数据集状态（6groups_interaction/）
- [ ] 理解CTF的标准化流程（`CTF_original/src/discovery.py:118-137`）

### 执行阶段
- [ ] 任务1.1：诊断缺失值模式
- [ ] 任务1.2：实现全局标准化数据生成
- [ ] 任务1.3：修复group2 batch_size问题
- [ ] 任务1.4：重新计算ATE
- [ ] 任务1.5：敏感性分析

### 验证阶段
- [ ] 检查全局标准化参数（均值、标准差）
- [ ] 验证交互项尺度一致性
- [ ] 对比组内vs全局标准化ATE差异
- [ ] 生成修复报告

---

## 🎬 快速开始命令

```bash
# 激活环境
conda activate causal-research

# 进入项目目录
cd /home/green/energy_dl/nightly/analysis

# 阅读完整问题总结
less docs/technical_reference/STANDARDIZATION_AND_DIBS_ISSUE_SUMMARY.md

# 开始第一步：诊断缺失值
# python scripts/diagnose_missing_patterns.py
```

---

## 📞 支持资源

**核心文档**：
- [STANDARDIZATION_AND_DIBS_ISSUE_SUMMARY.md](docs/technical_reference/STANDARDIZATION_AND_DIBS_ISSUE_SUMMARY.md) - 完整问题总结（v1.1）

**相关文档**：
- [DIBS_PARAMETER_TUNING_ANALYSIS.md](docs/technical_reference/DIBS_PARAMETER_TUNING_ANALYSIS.md) - DiBS参数调优
- [INTERACTION_TERMS_TRANSFORMATION_PLAN.md](docs/current_plans/INTERACTION_TERMS_TRANSFORMATION_PLAN.md) - 交互项方案
- [STAGE2_CODE_COMPARISON_20260125.md](docs/current_plans/STAGE2_CODE_COMPARISON_20260125.md) - 代码对比

**维护者**: Green
**创建日期**: 2026-01-26
**预期对话主题**: 实施全局标准化修复，恢复跨组可比性
