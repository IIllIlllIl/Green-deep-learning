# DiBS因果分析尝试归档

**归档日期**: 2025-12-30
**归档原因**: DiBS因果图学习在能耗数据上失败（0条因果边），转向回归分析方法

---

## 📋 归档内容

本目录包含2025-12-21至2025-12-29期间DiBS因果分析的所有尝试文件。

### 失败原因总结

DiBS在能耗数据上失败的根本原因：**数据聚合丢失了时间因果信息**

- **问题**: 每个实验将整个训练过程的能耗聚合为单一数值（总能耗）
- **后果**: 无法学习"超参数 → 训练动态 → 能耗"的时间因果关系
- **详细分析**: 查看 `../docs/reports/DIBS_FINAL_FAILURE_REPORT_20251226.md`

### 替代方案

转向使用回归分析、随机森林、因果森林等方法：
- **文档**: `../docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md`
- **状态**: 2025-12-30方案确认

---

## 📁 归档目录结构

```
archived_dibs_attempts/
├── scripts/          # Stage0-7数据处理脚本、DiBS演示脚本
├── data/             # Stage0-7中间数据、DiBS训练数据
├── results/          # DiBS实验结果（6groups v1/v2/v3）
├── docs/             # 数据处理流程文档（非失败分析报告）
└── README.md         # 本文件
```

---

## ⚠️ 重要说明

1. **不要使用归档的脚本**: 这些脚本是为DiBS设计的，与新的回归分析方案不兼容
2. **不要使用归档的数据**: 这些数据经过了DiBS特定的预处理（如归一化、One-Hot编码）
3. **保留作为历史记录**: 归档内容仅供参考，了解DiBS失败的原因

---

## ✅ 当前活跃内容（未归档）

**脚本**:
- 无（新方案脚本待创建）

**数据**:
- `data/energy_research/raw/energy_data_original.csv` - 原始数据（726行，56列）

**文档**:
- `docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md` - 回归分析方案
- `docs/reports/DIBS_FINAL_FAILURE_REPORT_20251226.md` - DiBS失败分析（保留）
- `docs/reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md` - 新旧数据对比

**工具**（可能以后使用）:
- `utils/causal_discovery.py` - DiBS核心工具
- `utils/causal_inference.py` - DML核心工具

---

**维护者**: Green + Claude
**最后更新**: 2025-12-30
