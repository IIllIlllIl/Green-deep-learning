# DiBS因果边白名单过滤结果

**生成日期**: 2026-01-20
**过滤脚本**: `scripts/filter_causal_edges_by_whitelist.py`
**白名单版本**: v1.1
**输入目录**: `../threshold/`

---

## 📁 目录内容

本目录包含使用白名单规则过滤后的DiBS因果边数据，共6个组：

| 文件名 | 边数 | 原始边数 | 保留率 |
|--------|------|---------|--------|
| `group1_examples_causal_edges_whitelist.csv` | 43 | 96 | 44.8% |
| `group2_vulberta_causal_edges_whitelist.csv` | 35 | 82 | 42.7% |
| `group3_person_reid_causal_edges_whitelist.csv` | 50 | 108 | 46.3% |
| `group4_bug_localization_causal_edges_whitelist.csv` | 40 | 85 | 47.1% |
| `group5_mrt_oast_causal_edges_whitelist.csv` | 40 | 104 | 38.5% |
| `group6_resnet_causal_edges_whitelist.csv` | 19 | 64 | 29.7% |
| **总计** | **227** | **539** | **42.1%** |

---

## 🎯 白名单规则概览

过滤使用了16条白名单规则，分为4组：

### 规则组1: 超参数主效应 (Q1)
- ✅ hyperparam → energy
- ✅ hyperparam → mediator
- ✅ hyperparam → performance

### 规则组2: 交互项调节效应 (Q1, Q2)
- ✅ interaction → energy
- ✅ interaction → mediator
- ✅ interaction → performance

### 规则组3: 中间变量中介效应 (Q2/Q3)
- ✅ mediator → energy
- ✅ mediator → mediator
- ✅ **mediator → performance** ⭐ (v1.1新增)
- ✅ energy → energy

### 规则组4: 控制变量影响
- ✅ control → energy/mediator/performance
- ✅ mode → energy/mediator/performance

**禁止的边**: 反因果（如 performance → hyperparam）、自循环、实验设计变量作为结果等

---

## 📊 过滤结果统计

### 按研究问题分类

| 研究问题 | 边数 | 占比 | 说明 |
|---------|------|------|------|
| Q1超参数主效应 | 16条 | 7.0% | 直接效应 |
| Q1交互项调节 | 25条 | 11.0% | 调节效应 |
| Q2性能效应 | 29条 | 12.8% | 直接+间接路径 |
| Q3中介效应 | 116条 | 51.1% | 中介和能耗分解 |
| 控制变量效应 | 23条 | 10.1% | 模型和模式影响 |

### 各组研究问题分布

| 组名 | Q1主效应 | Q1交互项 | Q2性能 | Q3中介 | 控制变量 |
|------|---------|---------|--------|--------|---------|
| group1_examples | 1 | 7 | 2 | 21 | 7 |
| group2_vulberta | 2 | 7 | 5 | 16 | 1 |
| group3_person_reid | 3 | 2 | 1 | 29 | 15 |
| group4_bug_localization | 1 | 3 | 11 | 21 | 0 |
| group5_mrt_oast | 6 | 5 | 9 | 15 | 0 |
| group6_resnet | 3 | 1 | 1 | 14 | 0 |

---

## 🚀 快速使用

### 读取数据

```python
import pandas as pd

# 读取单个组
df = pd.read_csv('group1_examples_causal_edges_whitelist.csv')

# 筛选Q1相关边
q1_edges = df[df['question_relevance'].str.contains('Q1')]

# 筛选强边（强度 >= 0.5）
strong_edges = df[df['strength'] >= 0.5]
```

### 合并所有组

```python
import glob
import pandas as pd

# 读取所有whitelist文件
files = glob.glob('*_whitelist.csv')
all_edges = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print(f"总边数: {len(all_edges)}")
```

---

## 📖 详细文档

- [白名单过滤结果总结](../../../docs/CAUSAL_EDGE_WHITELIST_SUMMARY.md) - 完整统计报告 ⭐⭐⭐
- [白名单设计方案](../../../docs/CAUSAL_EDGE_WHITELIST_DESIGN.md) - 设计文档
- [过滤脚本源码](../../../../scripts/filter_causal_edges_by_whitelist.py)

---

## 🔍 数据质量验证

所有过滤后的边都经过以下验证：
- ✅ 符合16条白名单规则
- ✅ 无反因果边（如 performance → hyperparam）
- ✅ 无自循环边
- ✅ 强度阈值 >= 0.3
- ✅ source_category 和 target_category 正确标注

---

**生成命令**:
```bash
cd ~/energy_dl/nightly/analysis
/home/green/miniconda3/envs/causal-research/bin/python scripts/filter_causal_edges_by_whitelist.py \
  --input-dir results/energy_research/data/interaction/threshold/ \
  --output-dir results/energy_research/data/interaction/whitelist/
```

**维护者**: Claude
**最后更新**: 2026-01-20
