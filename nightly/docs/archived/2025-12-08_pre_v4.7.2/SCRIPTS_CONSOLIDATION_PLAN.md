# Scripts 整合计划

**日期**: 2025-12-06
**版本**: 1.0
**状态**: 待执行

---

## 📋 整合目标

1. **消除重复功能** - 合并3个实验分析脚本为统一工具
2. **归档临时脚本** - 移动8个Stage7调试脚本到archived/
3. **改进可维护性** - 减少脚本数量，提高代码复用

---

## 🔍 脚本分类分析

### **类别1: 实验分析脚本（重复度高 ⚠️）**

| 脚本名称 | 功能 | 代码行数 | 重复度 |
|---------|------|---------|--------|
| `analyze_from_csv.py` | 从CSV分析实验完成情况 | 169行 | 90% |
| `analyze_from_json.py` | 从JSON分析实验完成情况 | 179行 | 90% |
| `analyze_missing_experiments.py` | 分析缺失实验 | 173行 | 85% |

**共同功能**:
- 统计参数-模式组合的唯一值数量
- 生成完成度报告
- 列出缺失的参数-模式组合
- 估算需要补充的实验数

**唯一差异**:
- 数据源：CSV vs JSON文件
- 输出格式略有不同

**整合方案**: 创建统一的 `analyze_experiments.py`

---

### **类别2: Stage7调试脚本（临时性 📋）**

| 脚本名称 | 功能 | 状态 | 建议 |
|---------|------|------|------|
| `analyze_stage7_results.py` | Stage7结果分析 | ✓完成 | 归档 |
| `analyze_stage7_mutation_attempts.py` | Stage7变异调试 | 调试 | 归档 |
| `check_stage7_before_state.py` | Stage7状态检查 | 调试 | 归档 |
| `reproduce_stage7_exact.py` | Stage7问题复现 | 调试 | 归档 |
| `track_mutate_calls.py` | Stage7调试工具 | 调试 | 归档 |
| `locate_defect.py` | Stage7缺陷定位 | 调试 | 归档 |
| `exact_simulation.py` | Stage7模拟 | 调试 | 归档 |
| `analyze_mutation_retry_mechanism.py` | 变异重试机制分析 | 调试 | 归档 |

**建议**: 移至 `scripts/archived/stage7_debug/`

---

### **类别3: Stage特定分析脚本（已完成任务 ✓）**

| 脚本名称 | 功能 | 状态 | 建议 |
|---------|------|------|------|
| `analyze_stage2_results.py` | Stage2结果分析 | ✓完成 | 归档 |
| `merge_stage3_stage4.py` | 合并Stage3-4配置 | ✓完成 | 归档 |

**建议**: 移至 `scripts/archived/stage_specific/`

---

### **类别4: 配置工具脚本（保留 ✅）**

| 脚本名称 | 功能 | 状态 |
|---------|------|------|
| `generate_mutation_config.py` | 生成变异配置 | 活跃 |
| `validate_mutation_config.py` | 验证配置文件 | 活跃 |
| `analyze_stage_configs.py` | 分析Stage配置统计 | 活跃 |

**建议**: 保持独立，功能清晰

---

### **类别5: 数据处理脚本（保留 ✅）**

| 脚本名称 | 功能 | 状态 |
|---------|------|------|
| `aggregate_csvs.py` | CSV聚合（历史） | 完成 |
| `analyze_baseline.py` | 基线结果分析 | 活跃 |
| `download_pretrained_models.py` | 下载预训练模型 | 工具 |

**建议**: 保持独立

---

## 🎯 整合方案详细设计

### **方案1: 创建统一的实验分析工具**

**新脚本**: `scripts/analyze_experiments.py`

**功能设计**:
```python
#!/usr/bin/env python3
"""
统一的实验分析工具

支持多种数据源:
- CSV文件 (summary_all.csv)
- JSON文件 (遍历experiment.json)

功能:
1. 统计参数-模式组合的唯一值数量
2. 生成完成度报告
3. 列出缺失的参数-模式组合
4. 估算需要补充的实验数
"""

使用示例:
  # 从CSV分析
  python3 scripts/analyze_experiments.py --source csv --file results/summary_all.csv

  # 从JSON分析
  python3 scripts/analyze_experiments.py --source json --dir results/

  # 仅显示缺失组合
  python3 scripts/analyze_experiments.py --source csv --missing-only

  # 导出报告
  python3 scripts/analyze_experiments.py --source csv --output report.md
```

**优势**:
- 统一接口，减少维护成本
- 支持多种数据源
- 灵活的输出格式
- 减少60%代码量（从521行→约200行）

---

### **方案2: 归档临时和完成的脚本**

**目录结构**:
```
scripts/
├── archived/
│   ├── stage7_debug/              # Stage7调试脚本
│   │   ├── analyze_stage7_mutation_attempts.py
│   │   ├── check_stage7_before_state.py
│   │   ├── reproduce_stage7_exact.py
│   │   ├── track_mutate_calls.py
│   │   ├── locate_defect.py
│   │   ├── exact_simulation.py
│   │   ├── analyze_mutation_retry_mechanism.py
│   │   └── analyze_stage7_results.py
│   ├── stage_specific/            # 其他Stage特定脚本
│   │   ├── analyze_stage2_results.py
│   │   └── merge_stage3_stage4.py
│   └── legacy_analysis/           # 旧版分析脚本（整合后）
│       ├── analyze_from_csv.py
│       ├── analyze_from_json.py
│       └── analyze_missing_experiments.py
├── analyze_experiments.py         # 新的统一工具
├── analyze_baseline.py            # 保留
├── generate_mutation_config.py    # 保留
├── validate_mutation_config.py    # 保留
├── analyze_stage_configs.py       # 保留
├── aggregate_csvs.py              # 保留
└── download_pretrained_models.py  # 保留
```

**README_ARCHIVE.md** 内容:
```markdown
# 归档脚本说明

## stage7_debug/
Stage7实验调试过程中使用的临时脚本。

归档原因:
- Stage7已于2025-12-06完成
- 这些脚本为调试特定问题而创建，不具通用性
- 保留用于历史参考和问题溯源

## stage_specific/
特定Stage的分析和配置脚本。

归档原因:
- Stage2已完成（2025-12-04）
- Stage3-4已合并完成（2025-12-05）
- 功能已完成，不再需要

## legacy_analysis/
旧版实验分析脚本。

归档原因:
- 功能重复（90%代码重复）
- 已被analyze_experiments.py替代
- 保留用于向后兼容和参考
```

---

## 📊 整合效果预期

### **代码量减少**
- **整合前**: 19个脚本，约4500行代码
- **整合后**: 8个活跃脚本 + 11个归档脚本
- **减少**: 3个重复脚本合并为1个（减少60%代码量）

### **可维护性提升**
- 统一接口，减少学习成本
- 单一数据源，减少bug
- 清晰的脚本分类

### **文件组织改进**
- 活跃脚本: 8个（清晰、易查找）
- 归档脚本: 11个（保留历史，不干扰）

---

## ✅ 执行步骤

### **步骤1: 创建统一分析工具**
1. 创建 `scripts/analyze_experiments.py`
2. 实现CSV数据源支持
3. 实现JSON数据源支持
4. 添加命令行参数解析
5. 测试功能完整性

### **步骤2: 创建归档目录结构**
```bash
mkdir -p scripts/archived/stage7_debug
mkdir -p scripts/archived/stage_specific
mkdir -p scripts/archived/legacy_analysis
```

### **步骤3: 移动脚本到归档目录**

**Stage7调试脚本**:
```bash
mv scripts/analyze_stage7_mutation_attempts.py scripts/archived/stage7_debug/
mv scripts/check_stage7_before_state.py scripts/archived/stage7_debug/
mv scripts/reproduce_stage7_exact.py scripts/archived/stage7_debug/
mv scripts/track_mutate_calls.py scripts/archived/stage7_debug/
mv scripts/locate_defect.py scripts/archived/stage7_debug/
mv scripts/exact_simulation.py scripts/archived/stage7_debug/
mv scripts/analyze_mutation_retry_mechanism.py scripts/archived/stage7_debug/
mv scripts/analyze_stage7_results.py scripts/archived/stage7_debug/
```

**Stage特定脚本**:
```bash
mv scripts/analyze_stage2_results.py scripts/archived/stage_specific/
mv scripts/merge_stage3_stage4.py scripts/archived/stage_specific/
```

**旧版分析脚本**（整合后）:
```bash
mv scripts/analyze_from_csv.py scripts/archived/legacy_analysis/
mv scripts/analyze_from_json.py scripts/archived/legacy_analysis/
mv scripts/analyze_missing_experiments.py scripts/archived/legacy_analysis/
```

### **步骤4: 创建归档说明**
```bash
# 为每个归档目录创建README_ARCHIVE.md
touch scripts/archived/stage7_debug/README_ARCHIVE.md
touch scripts/archived/stage_specific/README_ARCHIVE.md
touch scripts/archived/legacy_analysis/README_ARCHIVE.md
```

### **步骤5: 测试新工具**
```bash
# 测试CSV数据源
python3 scripts/analyze_experiments.py --source csv --file results/summary_all.csv

# 测试JSON数据源
python3 scripts/analyze_experiments.py --source json --dir results/

# 对比输出与旧脚本一致性
```

### **步骤6: 更新文档**
- 更新 `CLAUDE.md` - 反映新的脚本结构
- 更新 `README.md` - 更新脚本使用说明
- 创建 `docs/SCRIPTS_USAGE_GUIDE.md` - 脚本使用指南

---

## 🔄 回滚计划

如果整合后发现问题:

1. **保留旧脚本**: 所有旧脚本仍在 `archived/` 中
2. **快速恢复**:
   ```bash
   cp scripts/archived/legacy_analysis/*.py scripts/
   ```
3. **向后兼容**: 新工具支持旧脚本的所有功能

---

## 📝 待办清单

- [ ] 创建 `analyze_experiments.py`
- [ ] 实现CSV数据源
- [ ] 实现JSON数据源
- [ ] 添加单元测试
- [ ] 创建归档目录结构
- [ ] 移动脚本到归档目录
- [ ] 创建归档说明文件
- [ ] 测试新工具功能
- [ ] 更新项目文档
- [ ] 验证所有功能正常

---

**维护者**: Green
**创建日期**: 2025-12-06
**状态**: 设计阶段 - 待执行
