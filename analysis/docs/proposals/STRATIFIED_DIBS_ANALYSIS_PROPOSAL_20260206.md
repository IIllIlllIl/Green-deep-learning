# 分层DiBS因果图学习方案

**版本**: 1.1 (审查修订版)
**日期**: 2026-02-06
**作者**: 资深科研工作者 (Claude Code)
**状态**: 已通过审查，待执行

---

## 审查修订说明 (v1.0 → v1.1)

**审查评分**: 3.5/5 分（可条件性执行）

**P0问题修订**:
| ID | 原问题 | 修订内容 |
|----|--------|----------|
| P0-1 | ELBO历史不可获取 | ✅ 替换为基于一致性的收敛检查（方案B） |
| P0-2 | 数据路径硬编码 | ✅ 明确路径修改方案 |
| P0-3 | 缺少敏感性分析函数 | ✅ 添加完整实现设计 |
| P0-4 | 随机种子未参数化 | ✅ 确认CausalGraphLearner已支持random_seed |
| P0-5 | V3验收标准无法实现 | ✅ 改为基于一致性的收敛评估 |

**P1问题修订**:
- P1-1: ✅ 一致性阈值改为统计显著性测试（二项式检验）
- P1-2: ✅ group3_non_parallel增加到7次运行
- P1-3: ✅ 优化小样本配置参数
- P1-4: ✅ 添加--output-dir命令行参数

---

## 1. 方案概述

### 1.1 目标

在分层数据（按is_parallel分割）上执行DiBS因果发现，生成4个独立的因果图，用于后续ATE计算和跨场景对比分析。

### 1.2 分层数据概况（已验证）

| 分层ID | 数据文件 | 样本数 | 特征数 | 样本/特征比 | 敏感性运行次数 |
|--------|----------|--------|--------|-------------|----------------|
| group1_parallel | group1_parallel.csv | 178 | 26 | 6.8 | 3次 |
| group1_non_parallel | group1_non_parallel.csv | 126 | 26 | 4.8 | 5次 |
| group3_parallel | group3_parallel.csv | 113 | 28 | 4.0 | 5次 |
| group3_non_parallel | group3_non_parallel.csv | 93 | 28 | 3.3 | **7次** |

**关键变化**：
- 分层后无交互项列（`*_x_is_parallel`已移除）
- 无`is_parallel`列（分层后为常量）
- 数据无缺失值（来自`6groups_dibs_ready`，已验证）
- 存在常量列（7-8列），DiBS会自动处理

### 1.3 复用策略

**核心复用脚本**: `scripts/run_dibs_6groups_global_std.py` (456行)

**复用内容（约70%代码复用）**:
- `OPTIMAL_CONFIG` 参数配置
- `load_task_group_data()` 数据加载逻辑（需修改路径）
- `run_dibs_for_group()` DiBS运行逻辑（需添加random_seed参数）
- `CausalGraphLearner` 调用方式（已支持random_seed）
- 输出文件格式（邻接矩阵CSV、边列表、摘要JSON）

**需修改内容**:
- 数据路径：`6groups_dibs_ready` → `stratified/`
- 任务组配置：6组 → 4个分层
- 新增：敏感性分析函数
- 新增：基于一致性的收敛评估（替代ELBO检查）

---

## 2. 技术方案

### 2.1 DiBS配置

```python
# 基础配置（与全局分析一致）
DIBS_CONFIG = {
    "alpha_linear": 0.05,
    "beta_linear": 0.1,
    "n_particles": 20,
    "tau": 1.0,
    "n_steps": 5000,
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32
}

# 小样本层适配（group3_non_parallel: 93样本）- 已优化
SMALL_SAMPLE_CONFIG = {
    "alpha_linear": 0.08,    # 增加DAG惩罚（审查建议）
    "beta_linear": 0.15,     # 增加无环约束
    "n_particles": 15,       # 降低粒子数（计算效率）
    "tau": 1.2,              # 略微增加温度（探索多样性）
    "n_steps": 10000,        # 增加步数（更充分收敛）
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32
}
```

### 2.2 分层任务配置

```python
STRATIFIED_TASKS = [
    {
        "id": "group1_parallel",
        "name": "group1_examples (并行)",
        "csv_file": "group1_examples/group1_parallel.csv",
        "expected_samples": 178,
        "expected_features": 26,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 3  # 样本充足，3次足够
    },
    {
        "id": "group1_non_parallel",
        "name": "group1_examples (非并行)",
        "csv_file": "group1_examples/group1_non_parallel.csv",
        "expected_samples": 126,
        "expected_features": 26,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 5
    },
    {
        "id": "group3_parallel",
        "name": "group3_person_reid (并行)",
        "csv_file": "group3_person_reid/group3_parallel.csv",
        "expected_samples": 113,
        "expected_features": 28,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 5
    },
    {
        "id": "group3_non_parallel",
        "name": "group3_person_reid (非并行)",
        "csv_file": "group3_person_reid/group3_non_parallel.csv",
        "expected_samples": 93,
        "expected_features": 28,
        "use_small_sample_config": True,  # 最小样本层
        "n_sensitivity_runs": 7  # 增加运行次数（审查建议）
    }
]

# 随机种子配置
RANDOM_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033]  # 7个种子供选择
```

### 2.3 敏感性分析（审查后改进）

**目的**: 评估DiBS结果在小样本下的稳定性

**方法**:
1. 根据分层样本量使用不同数量的随机种子运行
2. 对每个分层独立运行
3. 使用统计显著性测试计算边检测稳定性

**改进的一致性计算（使用二项式检验）**:
```python
from scipy.stats import binom_test

def compute_edge_consistency(graphs, threshold=0.3, alpha=0.05):
    """
    计算边在多次运行中的检测一致性（使用统计显著性）

    参数:
        graphs: n个邻接矩阵列表
        threshold: 边检测阈值
        alpha: 显著性水平

    返回:
        consistency_matrix: 每条边被检测到的次数
        stable_edges: 统计显著的稳定边列表
        stability_report: 详细稳定性报告
    """
    n_runs = len(graphs)
    n_vars = graphs[0].shape[0]

    # 统计每条边被检测到的次数
    consistency_matrix = np.zeros((n_vars, n_vars))
    for graph in graphs:
        consistency_matrix += (graph > threshold).astype(int)

    # 使用二项式检验确定稳定边
    # H0: 边被检测的概率 = 0.5 (随机)
    # Ha: 边被检测的概率 > 0.5 (稳定)
    stable_edges = []
    stability_report = []

    for i in range(n_vars):
        for j in range(n_vars):
            detections = int(consistency_matrix[i, j])
            if detections > 0:
                # 单尾二项式检验
                p_value = binom_test(detections, n_runs, 0.5, alternative='greater')
                is_stable = p_value < alpha
                stability_report.append({
                    "source_idx": i,
                    "target_idx": j,
                    "detections": detections,
                    "n_runs": n_runs,
                    "detection_rate": detections / n_runs,
                    "p_value": p_value,
                    "is_stable": is_stable
                })
                if is_stable:
                    stable_edges.append((i, j, detections, p_value))

    return consistency_matrix, stable_edges, stability_report
```

**输出**:
- 平均邻接矩阵（多次运行平均）
- 稳定边列表（统计显著，p<0.05）
- 敏感性分析报告（每条边的检测次数和p值）

### 2.4 收敛性评估（替代方案）

**⚠️ 重要变更**: 原方案中的ELBO收敛检查无法实现（CausalGraphLearner不返回ELBO历史）

**替代方案: 基于一致性的隐含收敛检查**

```python
def evaluate_convergence_via_consistency(consistency_matrix, n_runs,
                                         convergence_threshold=0.7):
    """
    通过敏感性分析的一致性评估收敛性

    原理：如果DiBS未收敛，不同种子会产生高方差的不同图
          如果已收敛，不同种子应产生相似的图（高一致性）

    参数:
        consistency_matrix: 边检测次数矩阵
        n_runs: 运行次数
        convergence_threshold: 一致性阈值

    返回:
        converged: 是否收敛
        consistency_score: 一致性评分 (0-1)
        report: 详细报告
    """
    # 计算所有被检测到的边的平均一致性
    detected_edges = consistency_matrix > 0
    if np.sum(detected_edges) == 0:
        return False, 0.0, {"error": "no edges detected"}

    # 一致性评分 = 平均检测率
    consistency_scores = consistency_matrix[detected_edges] / n_runs
    avg_consistency = np.mean(consistency_scores)

    # 高一致性边的比例
    high_consistency_ratio = np.mean(consistency_scores >= convergence_threshold)

    # 判断收敛：如果≥50%的边有≥70%的一致性，认为收敛
    converged = high_consistency_ratio >= 0.5

    report = {
        "avg_consistency": float(avg_consistency),
        "high_consistency_ratio": float(high_consistency_ratio),
        "n_detected_edges": int(np.sum(detected_edges)),
        "convergence_threshold": convergence_threshold,
        "converged": converged
    }

    return converged, avg_consistency, report
```

---

## 3. 实现方案

### 3.1 脚本结构

创建新脚本 `scripts/stratified/run_dibs_stratified.py`：

```python
#!/usr/bin/env python3
"""
分层DiBS因果图学习脚本

在分层数据上执行DiBS因果发现，包含敏感性分析。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 1. 复用现有配置
from scripts.run_dibs_6groups_global_std import OPTIMAL_CONFIG

# 2. 复用CausalGraphLearner
from utils.causal_discovery import CausalGraphLearner

# 3. 分层任务配置
STRATIFIED_TASKS = [...]  # 如2.2节所示

# 4. 敏感性分析函数
def run_sensitivity_analysis(task_config, output_dir, seeds):
    """运行多次DiBS并分析一致性"""
    ...

# 5. 一致性计算（统计显著性）
def compute_edge_consistency(graphs, threshold=0.3, alpha=0.05):
    ...

# 6. 收敛性评估
def evaluate_convergence_via_consistency(consistency_matrix, n_runs):
    ...

# 7. 主函数
def main():
    ...
```

### 3.2 数据流

```
输入:
  data/energy_research/stratified/
  ├── group1_examples/
  │   ├── group1_parallel.csv (178×26)
  │   └── group1_non_parallel.csv (126×26)
  └── group3_person_reid/
      ├── group3_parallel.csv (113×28)
      └── group3_non_parallel.csv (93×28)

输出:
  results/energy_research/stratified/dibs/
  ├── group1_parallel/
  │   ├── run_seed_42/
  │   │   ├── causal_graph.csv
  │   │   ├── edges_threshold_0.3.csv
  │   │   └── summary.json
  │   ├── run_seed_123/
  │   ├── run_seed_456/
  │   ├── averaged_causal_graph.csv      # 3次平均
  │   ├── stable_edges_threshold_0.3.csv # 统计显著的稳定边
  │   ├── sensitivity_analysis.json      # 敏感性报告
  │   └── convergence_report.json        # 收敛性报告
  ├── group1_non_parallel/               # 5次运行
  ├── group3_parallel/                   # 5次运行
  ├── group3_non_parallel/               # 7次运行
  └── stratified_dibs_summary.json       # 总报告
```

### 3.3 命令行接口（已完善）

```bash
# Dry run（检查数据）
python scripts/stratified/run_dibs_stratified.py --dry-run

# 运行所有分层
python scripts/stratified/run_dibs_stratified.py

# 运行特定分层
python scripts/stratified/run_dibs_stratified.py --layer group1_parallel

# 指定输出目录（审查建议添加）
python scripts/stratified/run_dibs_stratified.py --output-dir results/custom/

# 指定随机种子数量（调试用）
python scripts/stratified/run_dibs_stratified.py --n-seeds 1

# 跳过敏感性分析（快速测试）
python scripts/stratified/run_dibs_stratified.py --skip-sensitivity

# 详细输出
python scripts/stratified/run_dibs_stratified.py --verbose
```

---

## 4. 验收标准（已修订）

### 4.1 必须满足

| 编号 | 验收项 | 标准 | 备注 |
|------|--------|------|------|
| V1 | 4个分层全部成功运行 | 无错误退出 | |
| V2 | 无缺失值 | 数据加载时检查通过 | 已验证 |
| V3 | **收敛性（修订）** | **一致性评分≥0.5** | 替代ELBO检查 |
| V4 | 边检测合理 | 每个因果图至少5条强边(>0.3) | |
| V5 | 敏感性分析完成 | 按配置运行次数完成 | |
| V6 | 稳定边识别 | 统计显著边数≥3 | 新增 |

### 4.2 预期输出（已修订）

| 分层 | 运行次数 | 预期强边数(>0.3) | 预期稳定边比例 |
|------|----------|------------------|----------------|
| group1_parallel | 3 | 10-50 | ≥70% |
| group1_non_parallel | 5 | 10-50 | ≥70% |
| group3_parallel | 5 | 10-50 | ≥60% |
| group3_non_parallel | 7 | 5-40 | **≥50%** |

*注：group3_non_parallel稳定边比例预期调整为≥50%（审查建议，更现实）*

---

## 5. 风险与缓解（已更新）

| 风险 | 概率 | 影响 | 缓解措施 | 状态 |
|------|------|------|----------|------|
| ELBO历史不可用 | 确定 | 中 | 已替换为一致性收敛检查 | ✅ 已解决 |
| group3_non_parallel不收敛 | 中 | 高 | 使用优化的小样本配置，增加到7次运行 | ✅ 已缓解 |
| 敏感性分析时间过长 | 中 | 低 | 差异化运行次数(3/5/5/7)，总计20次 | ✅ 已优化 |
| 边检测过少 | 低 | 中 | 降低阈值到0.2，或调整beta参数 | 备选方案 |
| 稳定边比例过低 | 中 | 中 | 使用统计显著性检验，更科学的阈值 | ✅ 已改进 |

---

## 6. 时间估算（已修订）

| 分层 | 运行次数 | 单次时间 | 小计 |
|------|----------|----------|------|
| group1_parallel | 3 | 4分钟 | 12分钟 |
| group1_non_parallel | 5 | 5分钟 | 25分钟 |
| group3_parallel | 5 | 5分钟 | 25分钟 |
| group3_non_parallel | 7 | 7分钟 | **49分钟** |
| 后处理和报告 | - | - | 5分钟 |
| **总计** | **20次** | - | **约116分钟（~2小时）** |

*注：group3_non_parallel使用小样本配置(n_steps=10000)，单次运行时间增加*

---

## 7. 已确认问题

| 问题 | 确认结果 | 解决方案 |
|------|----------|----------|
| ELBO历史可用性 | ❌ 不可用 | 使用一致性收敛检查 |
| 随机种子设置 | ✅ CausalGraphLearner支持random_seed参数 | 直接传参 |
| GPU内存 | ✅ 充分（<100MB） | 无需特殊处理 |
| 常量列处理 | ✅ DiBS自动处理 | 警告但不阻塞 |

---

## 8. 代码复用清单

### 8.1 直接复用（无修改）

- `OPTIMAL_CONFIG` 配置块
- 因果图统计计算（min/max/mean/std/edges计数）
- JSON/CSV保存逻辑
- 边列表提取逻辑

### 8.2 轻微修改（<20行）

```python
# load_task_group_data() - 修改数据路径
# 原: data_dir = Path(...) / "6groups_dibs_ready"
# 新: data_dir = Path(...) / "stratified"

# run_dibs_for_group() - 添加random_seed参数
def run_dibs_for_group(task_config, output_dir, config, random_seed=42, verbose=True):
    # ...
    learner = CausalGraphLearner(
        n_vars=len(feature_names),
        random_seed=random_seed,  # 新增
        # ... 其他参数
    )
```

### 8.3 新增代码（约150行）

- `run_sensitivity_analysis()` 函数
- `compute_edge_consistency()` 函数（使用二项式检验）
- `evaluate_convergence_via_consistency()` 函数
- 命令行参数扩展
- 分层任务配置

---

## 9. 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-02-06 | 初始方案 |
| v1.1 | 2026-02-06 | 审查修订：替换ELBO检查、优化敏感性分析、改进小样本配置、完善命令行接口 |

---

**方案制定时间**: 2026-02-06
**审查状态**: ✅ 已通过（评分3.5/5，P0问题已解决）
**计划执行时间**: 待批准后开始

**签字**:
```
[方案制定者] - 资深科研工作者 ✓
[同行评审] - 已审查，评分3.5/5，条件性通过 ✓
[待批准] - 项目负责人
```
