# Stage3和Stage4配置文件归档说明

**归档日期**: 2025-12-04
**归档原因**: Stage3和Stage4配置文件已合并为单个配置文件

## 📁 归档文件

| 文件 | 原始位置 | 归档原因 |
|------|----------|----------|
| `stage3_optimized_mnist_ff_and_medium_parallel.json` | `settings/` | 已合并到`stage3_4_merged_optimized_parallel.json` |
| `stage4_optimized_vulberta_densenet121_parallel.json` | `settings/` | 已合并到`stage3_4_merged_optimized_parallel.json` |

## 🔄 合并详情

### 合并统计

**实验项数量**:
- Stage3: 17个实验项
- Stage4: 8个实验项
- 合并后: 25个实验项

**预期实验数量** (runs_per_config总和):
- Stage3: 41个预期实验
- Stage4: 16个预期实验
- 合并后: 57个预期实验

**时间预估**:
- 原始预估: Stage3 (36小时) + Stage4 (32小时) = 68小时
- 基于Stage2经验重新预估: 57.1小时 (基于38.5%完成率)
- 时间减少: 16.0%

### 合并配置内容

**新配置文件**: `settings/stage3_4_merged_optimized_parallel.json`

**包含内容**:
1. 原Stage3所有实验项 (17个)
   - examples/mnist_ff 并行补充 (4个实验项)
   - bug-localization 并行 (4个实验项)
   - pytorch_resnet_cifar10 并行 (4个实验项)
   - MRT-OAST 并行 (5个实验项)

2. 原Stage4所有实验项 (8个)
   - VulBERTa/mlp 并行 (4个实验项)
   - Person_reID/densenet121 并行 (4个实验项)

**配置特性**:
- ✅ 启用去重机制 (`use_deduplication: true`)
- ✅ 使用历史CSV文件 (`historical_csvs: ["results/summary_all.csv"]`)
- ✅ 基于Stage2经验重新预估时间
- ✅ 添加合并说明注释

## 📊 项目状态更新

### 阶段合并后的状态

**当前阶段结构**:
1. ✅ Stage1: 非并行补全 (已完成)
2. ✅ Stage2: 非并行补充 + 快速模型并行 (已完成)
3. ⏳ **Stage3-4合并**: mnist_ff剩余 + 中速模型 + VulBERTa + densenet121 (待执行)
4. ⏳ Stage5: hrnet18并行 (待执行)
5. ⏳ Stage6: pcb并行 (待执行)

**预计总时间调整**:
- 原计划: 140-144小时 (6个阶段)
- 合并后: 减少1个阶段，总时间相应调整
- 建议重新计算剩余阶段总时间

## 🚀 执行建议

### 使用合并配置
```bash
# 执行合并后的Stage3-4
sudo -E python3 mutation.py -ec settings/stage3_4_merged_optimized_parallel.json
```

### 监控要点
1. **去重效果**: 监控实际完成的实验数量 (预计21个，基于38.5%完成率)
2. **时间验证**: 验证实际运行时间与预估57.1小时的差异
3. **参数达标**: 确保所有目标参数达到5个唯一值

## 📋 恢复流程

如需恢复独立配置文件:
1. 从本目录复制所需配置文件到 `settings/` 目录
2. 更新 `README.md` 和 `CLAUDE.md` 中的配置引用
3. 删除或归档合并配置文件

## ✅ 质量检查

### 归档前检查
- [x] 确认配置文件已成功合并
- [x] 验证合并配置文件格式正确
- [x] 更新相关文档中的配置引用
- [x] 创建归档说明

### 合并配置文件验证
- [x] JSON格式正确 (可通过 `python -m json.tool` 验证)
- [x] 所有实验项完整包含
- [x] 去重机制保持启用
- [x] 时间预估基于实际经验

---

**归档操作者**: Claude Code
**归档时间**: 2025-12-04
**状态**: ✅ 归档完成 - Stage3和Stage4已成功合并

> 提示：执行合并配置后，根据实际完成率更新后续阶段的时间预估。
