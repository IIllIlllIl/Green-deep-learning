# DiBS后续工作流快速指南

**创建**: 2026-02-10 | **状态**: 审查完成

---

## 当前进度

| 阶段 | 状态 | 输出位置 |
|------|------|----------|
| DiBS训练 | ✅ 完成 | `results/energy_research/data/global_std/group*/` |
| ATE计算 | ✅ 完成 | `results/energy_research/data/global_std_dibs_ate/` |
| 权衡检测 | ❌ 待执行 | `results/energy_research/tradeoff_detection/` |
| 可视化 | ❌ 待执行 | `results/energy_research/tradeoff_detection/figures/` |

---

## 步骤1: 权衡检测

**脚本**: `scripts/run_algorithm1_tradeoff_detection_global_std.py`

**输入**:
- ATE文件: `results/energy_research/data/global_std_dibs_ate/*_dibs_global_std_ate.csv`

**输出**:
- 权衡文件: `results/energy_research/tradeoff_detection/*_tradeoffs.csv`

**核心逻辑**:
1. 加载ATE数据，筛选显著边 (`is_significant=True`)
2. 使用CTF论文算法1检测权衡 (共享源节点，相反效应)
3. 应用规则系统判断改善方向

**执行**:
```bash
conda activate causal-research
python3 scripts/run_algorithm1_tradeoff_detection_global_std.py
```

**验证标准**:
- 预期权衡: 30-60个
- 能耗vs性能: 至少5个
- 统计可靠性: 100%基于显著ATE

---

## 步骤2: 可视化

**脚本**: `scripts/visualize_dibs_causal_graphs.py`

**⚠️ 问题**: 路径指向旧目录，需要更新

**修复**:
- 旧路径: `questions_2_3_dibs/20260105_212940`
- 新路径: `global_std/`

**建议**: 创建新脚本 `visualize_dibs_global_std.py` 或更新现有脚本

---

## 关键验收标准

1. **DiBS结果**: 6组完整，强边比例2%-18% ✅
2. **ATE覆盖**: >95%的边成功计算 ✅
3. **权衡检测**: 30-60个显著权衡 ⏳
4. **可视化**: 6组因果图+汇总图 ⏳

---

## 快速执行

```bash
# 1. 权衡检测
conda activate causal-research
cd /home/green/energy_dl/nightly/analysis
python3 scripts/run_algorithm1_tradeoff_detection_global_std.py

# 2. 可视化（需要先修复路径）
# python3 scripts/visualize_dibs_causal_graphs.py
```

---

## 下一步

1. 执行权衡检测脚本
2. 修复可视化脚本路径
3. 生成可视化报告
4. 创建验收报告
