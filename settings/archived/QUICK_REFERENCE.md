# 分阶段实验快速参考

**更新日期**: 2025-12-03
**状态**: ✅ 已修复runs_per_config问题

---

## ⚠️ 重要更新

**Stage2配置文件已更新**:
- 旧文件: ~~stage2_fast_models_parallel.json~~ (runs_per_config=1, 有问题)
- **新文件**: `stage2_nonparallel_supplement_and_fast_parallel.json` (runs_per_config=6, 已修复)
- 新配置包含stage1未完成的8个非并行实验

**Stage3-7已修复**: 所有`runs_per_config`从1改为6

---

## 执行顺序

```bash
# 阶段1 (已完成) - 非并行补全
# sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json
# 状态: ✅ 已完成 (12/44实验，27.3%)

# 阶段2 (23小时) - 补充stage1 + 快速模型并行 ⭐ 使用新配置
sudo -E python3 mutation.py -ec settings/stage2_nonparallel_supplement_and_fast_parallel.json

# 阶段3 (46小时) - 中速模型并行
sudo -E python3 mutation.py -ec settings/stage3_medium_models_parallel.json

# 阶段4 (40小时) - VulBERTa并行
sudo -E python3 mutation.py -ec settings/stage4_vulberta_parallel.json

# 阶段5 (40小时) - densenet121并行
sudo -E python3 mutation.py -ec settings/stage5_densenet121_parallel.json

# 阶段6 (40小时) - hrnet18并行
sudo -E python3 mutation.py -ec settings/stage6_hrnet18_parallel.json

# 阶段7 (40小时) - pcb并行
sudo -E python3 mutation.py -ec settings/stage7_pcb_parallel.json
```

---

## 阶段概览

| 阶段 | 配置文件 | runs_per_config | 时间 | 实验配置数 | 内容 |
|-----|----------|----------------|------|----------|------|
| 1 | stage1_nonparallel_completion.json | 1 | 30h | 12 | 非并行补全（部分完成） |
| 2 | **stage2_nonparallel_supplement_and_fast_parallel.json** | **6** | 23h | 24 | **补充stage1 + 快速模型并行** ⭐ |
| 3 | stage3_medium_models_parallel.json | **6** | 46h | 13 | 中速模型并行 |
| 4 | stage4_vulberta_parallel.json | **6** | 40h | 4 | VulBERTa并行 |
| 5 | stage5_densenet121_parallel.json | **6** | 40h | 4 | densenet121并行 |
| 6 | stage6_hrnet18_parallel.json | **6** | 40h | 4 | hrnet18并行 |
| 7 | stage7_pcb_parallel.json | **6** | 40h | 4 | pcb并行 |
| **总计** | **7个文件** | - | **259h** | **65** | **全部完成** |

---

## 每阶段完成后验证

```bash
# 检查实验数量
wc -l results/summary_all.csv

# 查看最新session
ls -lht results/run_* | head -1

# 查看最新session的summary
tail -20 results/run_*/summary.csv
```

---

## 预期进度

| 阶段完成后 | 总实验数 | 非并行完成度 | 并行完成度 | 整体完成度 |
|-----------|---------|------------|----------|----------|
| 初始状态 | 319 | 73.3% | 0% | 0% |
| **Stage1完成** | **331** | **80.0%** | **0%** | **0%** |
| Stage2完成 | ~410 | **100%** ✓ | 17.8% | 17.8% |
| Stage3完成 | ~475 | 100% ✓ | 42.2% | 42.2% |
| Stage4完成 | ~495 | 100% ✓ | 51.1% | 51.1% |
| Stage5完成 | ~515 | 100% ✓ | 60.0% | 60.0% |
| Stage6完成 | ~535 | 100% ✓ | 68.9% | 68.9% |
| Stage7完成 | ~555 | 100% ✓ | **100%** ✓ | **100%** ✓ |

---

## Stage1实际完成情况

### 数据统计
- 预期实验数: 44
- 实际完成: 12 (27.3%)
- 未完成原因: `runs_per_config=1` 配置问题

### 未完成部��（已合并到Stage2）
1. hrnet18 - learning_rate (需要1个值)
2. hrnet18 - dropout (需要1个值)
3. hrnet18 - seed (需要1个值)
4. pcb - learning_rate (需要1个值)
5. pcb - seed (需要1个值)
6. mnist_ff - batch_size (需要2个值)
7. mnist_ff - learning_rate (需要1个值)
8. mnist_ff - seed (需要1个值)

**总缺口**: 10个实验

---

## 注意事项

1. **必须顺序执行**: 不要并行运行多个阶段
2. **每个阶段完成后**: 验证数据正确性再继续
3. **如果中断**: 重新运行相同配置文件即可（去重机制会跳过已完成的）
4. **预计总时间**: 约10.8天（259小时）连续运行
5. **runs_per_config=6**: 确保每个参数生成足够的唯一值

---

## 修复说明

### 问题
原配置的`runs_per_config=1`导致每个实验配置只运行1次，无法生成足够的唯一值。

### 解决方案
1. ✅ 将Stage2-7的`runs_per_config`从1改为6
2. ✅ 创建新的Stage2配置，包含Stage1未完成部分
3. ✅ 去重机制会自动跳过重复值和已达标的参数

### 为什么是6？
- 目标: 每个参数5个唯一值
- 余量: +1提供缓冲
- 去重机制会在达到5个后自动停止

---

## 详细说明

- **完整执行计划**: `settings/STAGED_EXECUTION_PLAN.md`
- **修复报告**: `settings/STAGE_CONFIG_FIX_REPORT.md`
- **Stage1分析**: `results/STAGE1_SUMMARY.md`

---

**维护者**: Green (Claude Code)
**最后更新**: 2025-12-03
