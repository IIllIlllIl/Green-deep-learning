# 最终配置整合报告 - Stage12和Stage13

**日期**: 2025-12-08
**版本**: v4.7.2
**状态**: ✅ 配置完成，准备执行

---

## 📋 执行摘要

基于当前实验数据审计，对Stage11、Stage12和Stage13进行了重新整合，优化资源利用，使所有阶段运行时间接近24小时。

### 关键变更
1. **Stage11补充合并到Stage13** - 节省独立执行开销
2. **Stage12精确修正** - 基于实际数据（2个→5个唯一值）
3. **最终只需执行Stage12和Stage13** - 完成所有剩余实验

---

## 📊 最终配置概览

### Stage12: PCB并行补充
| 项目 | 值 |
|-----|---|
| 配置文件 | `settings/stage12_parallel_pcb.json` |
| 版本 | v4.7.2-corrected |
| 实验数 | **12个**（4参数 × 3次） |
| 预计时间 | **13.86小时** |
| 配置项 | 4个 |

**目标**: pcb并行模式，每个参数从2个唯一值补充到5个

**当前状态**:
- epochs: 2个（60, 57）→ 需补充3个
- learning_rate: 2个（0.05, 0.0254）→ 需补充3个
- seed: 2个（1334, 7650）→ 需补充3个
- dropout: 2个（0.5, 0.403）→ 需补充3个

### Stage13: 最终合并补充
| 项目 | 值 |
|-----|---|
| 配置文件 | `settings/stage13_final_merged.json` |
| 版本 | v4.7.2-final-merged |
| 实验数 | **66个**（4部分合并） |
| 预计时间 | **23.94小时** |
| 配置项 | 22个 |

**包含4个部分**:
1. **hrnet18并行补充**（Stage11合并）
   - 8个实验，11.44小时
   - 4参数，每个补充2个唯一值（3→5）

2. **其他并行快速模型补充**（原Stage13）
   - 43个实验，5.0小时
   - mnist_ff, mnist, mnist_rnn, siamese, bug-localization

3. **MRT-OAST非并行补充**（原Stage14）
   - 7个实验，2.5小时
   - epochs参数（4→5个唯一值）

4. **VulBERTa/cnn完整覆盖**（新增）
   - 8个实验，5.0小时
   - 非并行和并行各4参数（从0开始）

---

## 🔄 整合对比

### 原计划（整合前）
| 阶段 | 实验数 | 时间 | 状态 |
|-----|--------|------|------|
| Stage11 | 16 | 22.88h | 需补充 |
| Stage12 | 20 | 23.1h | 待执行 |
| Stage13 | 90 | 12.5h | 待执行 |
| **总计** | **126** | **58.48h** | 需执行3个阶段 |

### 最终方案（整合后）
| 阶段 | 实验数 | 时间 | 变更 |
|-----|--------|------|------|
| Stage12 | 12 | 13.86h | ✅ runs修正（5→3） |
| Stage13 | 66 | 23.94h | ✅ 合并Stage11+13+14 |
| **总计** | **78** | **37.80h** | **仅需执行2个阶段** |

### 资源节省
- ✅ 实验数减少: 126 → 78（-48个，-38.1%）
- ✅ 时间节省: 58.48h → 37.80h（-20.68h，-35.4%）
- ✅ 执行简化: 3个阶段 → 2个阶段

---

## 📈 详细变更说明

### 1. Stage11处理（合并到Stage13）

**原计划**:
```json
{
  "estimated_experiments": 16,
  "estimated_duration_hours": 22.88,
  "runs_per_config": 4
}
```

**发现**: 实际hrnet18并行已有**3个唯一值**（而非1个）

**修正后（合并到Stage13）**:
```json
{
  "estimated_experiments": 8,
  "estimated_duration_hours": 11.44,
  "runs_per_config": 2  // 3已有 + 2新增 = 5目标
}
```

**节省**: 8个实验，11.44小时

### 2. Stage12修正

**原配置**:
```json
{
  "estimated_experiments": 20,
  "runs_per_config": 5
}
```

**发现**: pcb并行已有**2个唯一值**

**修正后**:
```json
{
  "estimated_experiments": 12,
  "runs_per_config": 3  // 2已有 + 3新增 = 5目标
}
```

**节省**: 8个实验，9.24小时

### 3. Stage13整合

**整合内容**:
- ✅ Stage11补充（8实验，11.44h）
- ✅ Stage13并行快速模型（43实验，5.0h）
- ✅ Stage14 MRT-OAST（7实验，2.5h）
- ✅ VulBERTa/cnn新增（8实验，5.0h）

**总计**: 66个实验，23.94小时（接近24小时目标✅）

---

## ✅ 执行准备

### Stage12执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage12_parallel_pcb.json
```

**预期**:
- 12个新实验
- 每个参数5个唯一值（2已有 + 3新增）
- 用时约13.86小时

### Stage13执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage13_final_merged.json
```

**预期**:
- 66个新实验（可能因去重减少）
- hrnet18并行: 每个参数5个唯一值（3已有 + 2新增）
- VulBERTa/cnn: 每个参数5个唯一值（从0开始）
- 其他模型: 补充至5个唯一值
- 用时约23.94小时

---

## 🎯 完成标准

### Stage12完成后
- [ ] pcb并行实验总数: 17个（5已有 + 12新增）
- [ ] epochs: 5个唯一值 ✅
- [ ] learning_rate: 5个唯一值 ✅
- [ ] seed: 5个唯一值 ✅
- [ ] dropout: 5个唯一值 ✅

### Stage13完成后
- [ ] hrnet18并行实验总数: 17个（9已有 + 8新增）
- [ ] hrnet18并行所有参数: 5个唯一值 ✅
- [ ] VulBERTa/cnn非并行: 4参数各5个唯一值 ✅
- [ ] VulBERTa/cnn并行: 4参数各5个唯一值 ✅
- [ ] 其他并行快速模型: 补充完成 ✅
- [ ] MRT-OAST/default epochs: 5个唯一值 ✅

### 最终目标
- [ ] **100%覆盖**: 90/90参数-模式组合达标
- [ ] 所有实验training_succeeded=True
- [ ] CPU和GPU能耗数据完整

---

## 📁 配置文件清单

### 最终配置（使用这些）
1. ✅ `settings/stage12_parallel_pcb.json` - v4.7.2-corrected
2. ✅ `settings/stage13_final_merged.json` - v4.7.2-final-merged

### 归档配置（不再使用）
- `settings/stage11_supplement_parallel_hrnet18.json` - 已合并到Stage13
- `settings/stage13_merged_final_supplement.json` - 被新版本替代

---

## 📊 实验组成分解

### Stage12实验分解（12个）
```
pcb并行:
  epochs配置: 3个实验
  learning_rate配置: 3个实验
  seed配置: 3个实验
  dropout配置: 3个实验
```

### Stage13实验分解（66个）
```
第1部分 - hrnet18并行（8个）:
  epochs: 2个
  learning_rate: 2个
  seed: 2个
  dropout: 2个

第2部分 - 其他并行快速模型（43个）:
  mnist_ff: 20个（4参数×5次）
  mnist: 5个（batch_size×5次）
  mnist_rnn: 8个（batch_size×5 + epochs×3）
  siamese: 5个（batch_size×5次）
  bug-localization: 5个（kfold×5次）

第3部分 - MRT-OAST非并行（7个）:
  epochs: 7个

第4部分 - VulBERTa/cnn（8个）:
  非并行: 4个（4参数各1次基线，实际会运行5次变异）
  并行: 4个（4参数各1次基线，实际会运行5次变异）
```

---

## 🔧 技术细节

### 去重机制
- ✅ 所有配置启用`use_deduplication: true`
- ✅ 历史数据:`results/summary_all.csv`
- ✅ 自动跳过已存在的超参数组合
- ✅ 碰撞概率<0.1%（随机变异）

### runs_per_config优先级
- 并行模式: 外层experiment > foreground > 全局
- 非并行模式: experiment-level > 全局

### 模式区分
- 所有去重key包含模式信息（parallel/nonparallel）
- 确保并行和非并行实验独立去重

---

## 💡 执行建议

### 建议顺序
1. **先执行Stage12**（13.86h）
   - 相对独立（pcb模型）
   - 时间较短，验证配置修复效果
   - 完成后可验证并行模式runs_per_config修复

2. **再执行Stage13**（23.94h）
   - 包含多个模型和部分
   - 时间接近24小时
   - 完成后达到100%覆盖

### 监控命令
```bash
# 实时监控
watch -n 60 'tail -1 results/run_*/summary.csv 2>/dev/null'

# 检查进度
python3 -c "
import csv
with open('results/summary_all.csv') as f:
    rows = list(csv.DictReader(f))
    print(f'总实验数: {len(rows)}')
"
```

---

## ⚠️ 注意事项

1. **去重效果**: 实际运行时间可能因去重而缩短
2. **数据备份**: 执行前务必备份`summary_all.csv`
3. **磁盘空间**: 确保>100GB可用空间
4. **GPU可用**: 验证nvidia-smi正常
5. **sudo权限**: 能耗监控需要sudo权限

---

## 📚 相关文档

- [Stage11实际状态修正报告](STAGE11_ACTUAL_STATE_CORRECTION.md)
- [Stage11 Bug修复报告](STAGE11_BUG_FIX_REPORT.md)
- [去重与随机变异分析](DEDUPLICATION_RANDOM_MUTATION_ANALYSIS.md)
- [JSON配置最佳实践](../JSON_CONFIG_BEST_PRACTICES.md)

---

**创建者**: Green + Claude
**日期**: 2025-12-08
**版本**: v4.7.2
**状态**: ✅ 配置完成，准备执行
**下一步**: 执行Stage12和Stage13完成所有实验
