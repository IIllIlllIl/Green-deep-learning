# DiBS因果分析可读结果

**生成时间**: 2026-01-17 15:10:48
**原始数据**: ../20260117_000522/
**文件数量**: 24个CSV文件（6组 × 4文件）

---

## 文件说明

每个任务组生成4个CSV文件：

1. **{group}_causal_edges_all.csv** (529行)
   - 所有直接因果边（无筛选）
   - 包括强度=0的边
   - 用于验证"边不存在"

2. **{group}_causal_paths.csv** (约100-200行)
   - 所有间接因果路径（2步和3步）
   - 筛选: path_strength > 0.05
   - 用于发现中介效应

3. **{group}_causal_edges_0.3.csv** (约42-78行)
   - 强直接边（strength > 0.3）
   - 快速查看最重要的因果关系

4. **{group}_causal_paths_0.3.csv** (约0-5行)
   - 强间接路径（path_strength > 0.3）
   - 发现极强的中介效应

---

## 使用示例

### Excel快速查看

1. 打开 `group1_examples_causal_edges_0.3.csv`
2. 查看所有强因果关系（一目了然）

### 查看调节效应

```bash
cat group1_examples_causal_edges_all.csv | grep "moderation"
```

### 查看间接效应

```bash
cat group1_examples_causal_paths.csv | grep "batch_size"
```

### Python分析

```python
import pandas as pd

# 读取所有边
edges = pd.read_csv('group1_examples_causal_edges_all.csv')

# 筛选调节效应
moderation = edges[edges['edge_type'] == 'moderation']
print(moderation[['source', 'target', 'strength', 'interpretation']])

# 读取间接路径
paths = pd.read_csv('group1_examples_causal_paths.csv')

# 查找batch_size相关的路径
batch_paths = paths[paths['source'].str.contains('batch_size')]
print(batch_paths[['path', 'path_strength', 'interpretation']])
```

---

## 原始数据位置

完整的DiBS输出（包括.npy矩阵和.json结果）位于：
`../20260117_000522/`

---

## 数据字典

### causal_edges_all.csv 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| source | string | 源变量（因） |
| target | string | 目标变量（果） |
| strength | float | 边强度（0-1） |
| edge_type | string | 边类型（main_effect/moderation/mediator等） |
| is_significant | string | 是否显著（yes: >0.1, no: ≤0.1） |
| strength_level | string | 强度等级（very_strong/strong/moderate/weak/zero） |
| source_category | string | 源变量类别 |
| target_category | string | 目标变量类别 |
| question_relevance | string | 相关研究问题（Q1/Q2/Q3） |
| interpretation | string | 人类可读解释 |

### causal_paths.csv 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| path_id | string | 路径唯一ID |
| path_length | int | 路径步数（2或3） |
| source | string | 起点变量 |
| target | string | 终点变量 |
| path | string | 完整路径（简化变量名） |
| path_strength | float | 路径总强度（各步相乘） |
| step1_strength | float | 第1步强度 |
| step2_strength | float | 第2步强度 |
| step3_strength | float | 第3步强度（2步路径为空） |
| path_type | string | 路径类型 |
| question_relevance | string | 相关研究问题 |
| interpretation | string | 人类可读解释 |

---

**生成脚本**: `scripts/convert_dibs_to_csv.py`
**方案文档**: `docs/DIBS_RESULTS_REPRESENTATION_FINAL.md`
