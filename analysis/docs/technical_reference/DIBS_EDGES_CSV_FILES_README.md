# DiBS因果边CSV文件说明

**创建日期**: 2026-01-16
**最后更新**: 2026-01-16
**数据来源**: DiBS 6分组因果分析 (20260116_004323)

---

## 📋 快速概览

本目录包含从DiBS因果分析结果提取的CSV文件，方便后续数据分析和可视化。

### 文件列表

| 文件名 | 大小 | 数据量 | 用途 |
|--------|------|--------|------|
| `dibs_direct_edges.csv` | 18KB | 114条 | 直接因果边 |
| `dibs_indirect_paths.csv` | 234KB | 759条 | 间接因果路径 |
| `dibs_all_edges_summary.csv` | 1.3KB | 6组 | 按任务组汇总统计 |

### 数据质量

✅ **所有验证项通过**
- 数据完整性: 100% ✅
- 提取正确性: 100% ✅
- 文档准确性: 100% ✅

---

## 🚀 快速开始

### 读取数据

```python
import pandas as pd

# 读取直接因果边
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')

# 读取间接因果路径
paths = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_indirect_paths.csv')

# 读取汇总统计
summary = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_all_edges_summary.csv')
```

### 常见查询

```python
# 1. 识别对能耗影响最大的超参数
top_hyperparams = edges[edges['edge_type'] == 'hyperparam_to_energy'].nlargest(10, 'strength')

# 2. 查找完全中介路径
full_mediation = paths[paths['mediation_type'] == 'full']

# 3. 分析特定任务组
group1_edges = edges[edges['task_group'] == 'group1_examples']

# 4. 统计各类边的数量
edge_type_counts = edges['edge_type'].value_counts()
```

---

## 📊 数据统计概览

### 直接因果边 (114条)

| 边类型 | 数量 | 占比 |
|--------|------|------|
| 超参数 → 能耗 | 57 | 50.0% |
| 性能 → 能耗 | 46 | 40.4% |
| 超参数 → 性能 | 11 | 9.6% |

### 间接因果路径 (759条)

| 路径类型 | 数量 | 占比 |
|---------|------|------|
| 多步路径 (≥4节点) | 278 | 36.6% |
| 超参数 → 中介 → 能耗 | 266 | 35.0% |
| 性能 → 中介 → 能耗 | 200 | 26.3% |
| 超参数 → 中介 → 性能 | 15 | 2.0% |

### 按任务组统计

| 任务组 | 直接边 | 间接路径 | 总因果关系 | 样本数 |
|--------|--------|----------|------------|--------|
| examples | 11 | 92 | 103 | 276 |
| VulBERTa | 20 | 114 | 134 | 142 |
| Person_reID | 13 | 192 | 205 | 157 |
| bug-localization | 27 | 142 | 169 | 67 |
| MRT-OAST | 21 | 102 | 123 | 65 |
| pytorch_resnet | 22 | 117 | 139 | 111 |
| **总计** | **114** | **759** | **873** | **818** |

---

## 📚 相关文档

### 使用指南

- **[DIBS_EDGES_CSV_USAGE_GUIDE.md](DIBS_EDGES_CSV_USAGE_GUIDE.md)** - 详细使用指南 ⭐⭐⭐
  - 文件结构说明
  - 列定义
  - 使用示例
  - 常见查询

- **[DIBS_RESULTS_CONTENT_GUIDE.md](DIBS_RESULTS_CONTENT_GUIDE.md)** - DiBS结果内容说明
  - 结果结构
  - 数据追溯
  - 解读方法

### 验证报告

- **[DIBS_EDGES_CSV_QUALITY_VERIFICATION.md](DIBS_EDGES_CSV_QUALITY_VERIFICATION.md)** - 质量验证报告 ⭐⭐⭐⭐⭐
  - 数据完整性验证
  - 提取脚本正确性验证
  - 文档准确性验证
  - 总体质量评估

### 相关资源

- **提取脚本**: `scripts/extract_dibs_edges_to_csv.py`
- **原始结果**: `results/energy_research/dibs_6groups_final/20260116_004323/`
- **DiBS分析报告**: `docs/reports/QUESTIONS_2_3_DIBS_COMPLETE_REPORT_20260105.md`

---

## 🔧 重新生成CSV文件

如果需要重新生成CSV文件（例如更改阈值）：

```bash
# 使用默认阈值 0.3
/home/green/miniconda3/envs/causal-research/bin/python \
  scripts/extract_dibs_edges_to_csv.py

# 使用自定义阈值
/home/green/miniconda3/envs/causal-research/bin/python \
  scripts/extract_dibs_edges_to_csv.py \
  --threshold 0.5

# 指定结果目录和输出目录
/home/green/miniconda3/envs/causal-research/bin/python \
  scripts/extract_dibs_edges_to_csv.py \
  --result-dir results/energy_research/dibs_6groups_final \
  --output-dir results/energy_research/dibs_edges_csv_custom
```

---

## 💡 分析建议

### 推荐分析流程

1. **全局理解**: 使用 `dibs_all_edges_summary.csv` 了解各任务组的因果关系复杂度
2. **识别关键因子**: 使用 `dibs_direct_edges.csv` 找出对能耗/性能影响最大的超参数
3. **探索传导机制**: 使用 `dibs_indirect_paths.csv` 理解变量间的传导路径和中介效应

### 研究问题对应

| 研究问题 | 推荐使用的文件 | 关键列 |
|---------|---------------|--------|
| **问题1**: 超参数对能耗的影响 | `dibs_direct_edges.csv` | `edge_type == 'hyperparam_to_energy'` |
| **问题2**: 能耗-性能权衡 | `dibs_direct_edges.csv` + `dibs_indirect_paths.csv` | `edge_type == 'performance_to_energy'` |
| **问题3**: 中介变量效应 | `dibs_indirect_paths.csv` | `mediation_type`, `indirect_strength` |

### 可视化建议

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')

# 1. 边强度分布
plt.figure(figsize=(10, 6))
sns.histplot(edges['strength'], bins=20)
plt.title('Distribution of Edge Strengths')
plt.xlabel('Strength')
plt.ylabel('Count')
plt.savefig('edge_strength_distribution.png')

# 2. 任务组对比
task_counts = edges.groupby('task_group')['edge_type'].value_counts().unstack(fill_value=0)
task_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Edge Types by Task Group')
plt.xlabel('Task Group')
plt.ylabel('Edge Count')
plt.legend(title='Edge Type')
plt.savefig('task_group_comparison.png')

# 3. 超参数影响热力图
hp_to_energy = edges[edges['edge_type'] == 'hyperparam_to_energy']
pivot = hp_to_energy.pivot_table(
    values='strength',
    index='source',
    columns='task_group',
    aggfunc='mean'
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Hyperparameter Effects on Energy Across Task Groups')
plt.savefig('hyperparam_heatmap.png')
```

---

## ⚠️ 使用注意事项

### 数据解读

1. **边强度**: 值域 [0.3, 1.0]，0.3为阈值下限，1.0表示完美因果关系
2. **间接效应**: 通过路径各步边强度的乘积计算，值较小是正常的
3. **任务组差异**: 不同任务组的因果结构可能差异很大，需分组分析

### 常见误区

❌ **错误**: 直接比较不同任务组的边强度绝对值
✅ **正确**: 在任务组内比较边强度的相对大小

❌ **错误**: 认为间接效应一定弱于直接效应
✅ **正确**: 多个中介变量可能形成强间接效应

❌ **错误**: 忽略完全中介路径
✅ **正确**: 完全中介意味着所有效应都通过中介变量传导

---

## 📞 支持与反馈

如有疑问或发现问题，请：

1. 查阅 [DIBS_EDGES_CSV_USAGE_GUIDE.md](DIBS_EDGES_CSV_USAGE_GUIDE.md) 获取详细使用说明
2. 查阅 [DIBS_EDGES_CSV_QUALITY_VERIFICATION.md](DIBS_EDGES_CSV_QUALITY_VERIFICATION.md) 了解数据质量保证
3. 检查 `scripts/extract_dibs_edges_to_csv.py` 了解数据提取逻辑

---

## 📝 版本信息

- **CSV文件版本**: v1.0
- **生成日期**: 2026-01-16 13:09
- **DiBS结果版本**: 20260116_004323
- **边强度阈值**: 0.3
- **数据来源**: 6分组最终DiBS分析

---

**维护者**: Analysis Team
**最后验证**: 2026-01-16
**验证状态**: ✅ 所有质量检查通过
