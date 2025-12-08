# 归档文件说明

本目录包含已被新配置替代的过时配置文件。

## 归档日期: 2025-12-02

## 归档文件列表

### 1. mutation_2x_supplement系列 (2025-12-01)
这些配置用于补充现有模型的实验，以达到每个参数5个唯一值的目标。

- **mutation_2x_supplement.json** - 初始补充配置
- **mutation_2x_supplement_complete.json** - 完整补充配置
- **mutation_2x_supplement_existing_models.json** - 针对现有模型的补充
- **mutation_2x_supplement_full_backup_20251201.json** - 备份版本
- **mutation_2x_supplement_remaining.json** - 剩余实验配置
- **MUTATION_2X_SUPPLEMENT_README.md** - 补充实验说明文档

**归档原因**: 这些配置仅运行了18/44个计划实验（由于去重机制）。后续发现需要更全面的实验计划来达到100%完成度。

### 2. mutation_final_completion系列 (2025-12-02)
完整的实验补全配置，包含所有剩余的非并行和并行实验。

- **mutation_final_completion.json** - 单一完整配置文件（57个实验配置）
- **mutation_final_completion_README.md** - 执行计划和说明文档

**归档原因**: 用户要求将实验分割成更小的阶段（每阶段2-3天），以便更好地管理和监控。该配置预计运行时间过长（230-350小时连续），不便于实际操作。

## 替代方案

上述归档文件的功能已被以下**分阶段配置**完全替代：

### 分阶段实验配置 (2025-12-02)
将完整实验计划分割成7个可管理的阶段：

1. **stage1_nonparallel_completion.json** - 阶段1: 非并行补全 (30小时)
2. **stage2_fast_models_parallel.json** - 阶段2: 快速模型并行 (20小时)
3. **stage3_medium_models_parallel.json** - 阶段3: 中速模型并行 (46小时)
4. **stage4_vulberta_parallel.json** - 阶段4: VulBERTa并行 (40小时)
5. **stage5_densenet121_parallel.json** - 阶段5: densenet121并行 (40小时)
6. **stage6_hrnet18_parallel.json** - 阶段6: hrnet18并行 (40小时)
7. **stage7_pcb_parallel.json** - 阶段7: pcb并行 (40小时)

### 配套文档
- **STAGED_EXECUTION_PLAN.md** - 详细的分阶段执行计划
- **QUICK_REFERENCE.md** - 快速参考指南

## 分阶段方案的优势

1. **时间可控**: 每个阶段20-46小时，便于安排
2. **便于监控**: 每阶段结束可以验证结果
3. **灵活暂停**: 可以在阶段间休息或检查
4. **风险分散**: 不会一次性运行10天+
5. **渐进式完成**: 可以看到逐步的进展

## 实验范围

### 归档配置的实验范围
- **mutation_2x_supplement系列**: 44个实验配置（实际只运行了18个）
- **mutation_final_completion**: 57个实验配置（未运行）

### 分阶段配置的实验范围
- **总实验配置数**: 57个（与mutation_final_completion相同）
- **总实验数**: 269个（44非并行 + 225并行）
- **预计总时间**: 256小时（约10.7天）
- **分成**: 7个阶段，每阶段20-46小时

## 数据完整性

### CSV追加bug修复 (v4.4.0)
在创建分阶段配置之前，修复了关键的数据完整性问题：

**问题**: `mutation/runner.py:_append_to_summary_all()` 调用 `aggregate_csvs.py`，该脚本使用 'w' 模式覆盖整个 `summary_all.csv` 文件。

**影响**: mutation_2x_supplement_existing_models.json 运行时导致90个历史实验数据丢失（从301行降至211行）。

**修复**:
- 重写 `_append_to_summary_all()` 方法，直接使用CSV模块的 'a' 模式追加
- 移除对 `aggregate_csvs.py` 的调用
- 创建11个单元测试和4个手动测试验证修复

**当前状态**: 所有分阶段配置使用修复后的代码，确保数据安全追加。

## 去重机制

所有分阶段配置都启用了去重机制：
```json
{
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"]
}
```

- 自动跳过已存在于历史CSV中的实验
- 为需要多个唯一值的参数自动生成新的随机值
- 允许中断后重新运行，自动继续未完成的实验

## 当前实验状态 (2025-12-02)

### summary_all.csv统计
- **总实验数**: 319个
  - 非并行: 214个
  - 并行: 105个

### 完成度分析
- **非并行完成度**: 73.3% (33/45参数达到5个唯一值)
- **并行完成度**: 0% (0/45参数达到5个唯一值)
- **整体完成度**: 0% (需要两种模式都达到5个唯一值)

### 剩余工作
- **非并行实验**: 44个
- **并行实验**: 225个
- **总计**: 269个实验
- **预计时间**: 256小时（通过7个阶段完成）

## 使用建议

1. **不要使用归档文件**: 这些配置已过时或被替代
2. **使用分阶段配置**: 参考 `../QUICK_REFERENCE.md` 和 `../STAGED_EXECUTION_PLAN.md`
3. **顺序执行**: 按照阶段1→2→3→4→5→6→7的顺序执行
4. **验证中间结果**: 每个阶段完成后检查数据完整性

## 参考文档

- [分阶段执行计划](../STAGED_EXECUTION_PLAN.md) - 详细的执行指南
- [快速参考](../QUICK_REFERENCE.md) - 所有阶段的命令列表
- [主README](../../README.md) - 项目总体说明

---

**归档人**: Green
**最后更新**: 2025-12-02
