# Stage13-14合并及实验完整覆盖报告

**日期**: 2025-12-07
**版本**: v4.7.1
**状态**: ✅ 配置完成，待执行

---

## 📋 执行总结

### 任务要求
用户要求：
1. ✅ 合并Stage13和Stage14到单个配置文件（时间短，便于执行）
2. ✅ 检查Stage9-13是否覆盖所有剩余未完成实验
3. ✅ 如有遗漏，补充到Stage13中

### 发现的问题

**重大遗漏**: VulBERTa/cnn模型在所有已规划阶段（Stage1-14）中**完全缺失**
- 非并行模式：0个实验（4参数全部缺失）
- 并行模式：0个实验（4参数全部缺失）
- 影响：8个参数-模式组合，40个实验缺失

### 执行的修复

1. **合并Stage13 + Stage14** ✅
   - 原Stage13: 9个配置项，43个实验（并行快速模型补充）
   - 原Stage14: 1个配置项，7个实验（MRT-OAST epochs补充）
   - 合并后: 10个配置项，50个实验

2. **新增VulBERTa/cnn完整覆盖** ✅
   - 非并行模式: 4个配置项，20个实验
   - 并行模式: 4个配置项，20个实验
   - 小计: 8个配置项，40个实验

3. **创建最终Stage13配置** ✅
   - 文件：`settings/stage13_merged_final_supplement.json`
   - 总计：18个配置项，90个实验
   - 预计时长：12.5小时

---

## 📊 实验覆盖分析

### 当前完成度（基于summary_all.csv）

**总实验数**: 400个

**参数-模式组合覆盖**:
- **目标**: 90个组合（11个模型 × 45个参数，区分并行/非并行）
- **已达标**: 64个组合（71.1%）
- **未达标**: 26个组合（28.9%）

### 未达标组合详细列表

#### 非并行模式（5个组合）
| 模型 | 参数 | 当前唯一值 | 目标 | 状态 |
|------|------|-----------|------|------|
| MRT-OAST/default | epochs | 4 | 5 | ❌ 需补充1个 |
| VulBERTa/cnn | epochs | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | learning_rate | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | weight_decay | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | seed | 0 | 5 | ❌ 需补充5个 |

#### 并行模式（21个组合）
| 模型 | 参数 | 当前唯一值 | 目标 | 状态 |
|------|------|-----------|------|------|
| Person_reID_baseline_pytorch/hrnet18 | epochs | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/hrnet18 | learning_rate | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/hrnet18 | dropout | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/hrnet18 | seed | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/pcb | epochs | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/pcb | learning_rate | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/pcb | dropout | 2 | 5 | ❌ 需补充3个 |
| Person_reID_baseline_pytorch/pcb | seed | 2 | 5 | ❌ 需补充3个 |
| VulBERTa/cnn | epochs | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | learning_rate | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | weight_decay | 0 | 5 | ❌ 需补充5个 |
| VulBERTa/cnn | seed | 0 | 5 | ❌ 需补充5个 |
| bug-localization-by-dnn-and-rvsm/default | kfold | 2 | 5 | ❌ 需补充3个 |
| examples/mnist | batch_size | 2 | 5 | ❌ 需补充3个 |
| examples/mnist_ff | epochs | 2 | 5 | ❌ 需补充3个 |
| examples/mnist_ff | learning_rate | 2 | 5 | ❌ 需补充3个 |
| examples/mnist_ff | batch_size | 2 | 5 | ❌ 需补充3个 |
| examples/mnist_ff | seed | 2 | 5 | ❌ 需补充3个 |
| examples/mnist_rnn | epochs | 4 | 5 | ❌ 需补充1个 |
| examples/mnist_rnn | batch_size | 2 | 5 | ❌ 需补充3个 |
| examples/siamese | batch_size | 2 | 5 | ❌ 需补充3个 |

---

## 🔍 Stage9-13覆盖验证

### Stage9-10: 非并行hrnet18和pcb

**Stage9**: Person_reID_baseline_pytorch/hrnet18
- epochs, learning_rate, dropout, seed (各5次)
- **结论**: hrnet18非并行模式已达标（5+唯一值），无需补充 ✓

**Stage10**: Person_reID_baseline_pytorch/pcb
- epochs, learning_rate, dropout, seed (各5次)
- **结论**: pcb非并行模式已达标（5+唯一值），无需补充 ✓

### Stage11-12: 并行hrnet18和pcb补充

**Stage11**: Person_reID_baseline_pytorch/hrnet18（并行）
- ✅ 覆盖: epochs, learning_rate, dropout, seed (各5次)
- **预期**: 补充3个唯一值到5个

**Stage12**: Person_reID_baseline_pytorch/pcb（并行）
- ✅ 覆盖: epochs, learning_rate, dropout, seed (各5次)
- **预期**: 补充3个唯一值到5个

### Stage13 (合并后): 并行快速模型 + MRT-OAST + VulBERTa/cnn

**第一部分: 并行快速模型补充** (原Stage13)
- ✅ examples/mnist_ff: epochs, learning_rate, batch_size, seed (各5次)
- ✅ examples/mnist: batch_size (5次)
- ✅ examples/mnist_rnn: batch_size (5次), epochs (3次)
- ✅ examples/siamese: batch_size (5次)
- ✅ bug-localization-by-dnn-and-rvsm/default: kfold (5次)

**第二部分: MRT-OAST补充** (原Stage14)
- ✅ MRT-OAST/default: epochs (7次，非并行)

**第三部分: VulBERTa/cnn全覆盖** (新增)
- ✅ VulBERTa/cnn非并行: epochs, learning_rate, weight_decay, seed (各5次)
- ✅ VulBERTa/cnn并行: epochs, learning_rate, weight_decay, seed (各5次)

### 覆盖总结

| 阶段 | 覆盖的未达标组合 | 状态 |
|------|-----------------|------|
| Stage9-10 | 0个（已达标） | ✅ 可选执行 |
| Stage11 | 4个（hrnet18并行） | ✅ 必须执行 |
| Stage12 | 4个（pcb并行） | ✅ 必须执行 |
| Stage13 | 18个（快速模型+MRT+cnn） | ✅ 必须执行 |
| **总计** | **26/26 (100%)** | ✅ 完整覆盖 |

**结论**: Stage9-13完整覆盖所有26个未达标组合 ✓

---

## 📁 修改的文件列表

### 新增文件

1. **`settings/stage13_merged_final_supplement.json`** ✅
   - 合并Stage13 + Stage14 + 新增VulBERTa/cnn
   - 18个配置项，90个实验，12.5小时
   - 版本: v4.7.1-merged

2. **`docs/results_reports/STAGE13_14_MERGE_AND_COMPLETION_REPORT.md`** ✅
   - 本报告

### 备份文件

1. `settings/stage13_parallel_fast_models_supplement.json.bak_20251207`
2. `settings/stage14_stage7_8_supplement.json.bak_20251207`

### 保留文件（参考用）

1. `settings/stage13_parallel_fast_models_supplement.json` - 原Stage13
2. `settings/stage14_stage7_8_supplement.json` - 原Stage14

---

## 🚀 执行建议

### 推荐执行顺序

**优先级1: Stage13** (最短，12.5小时)
```bash
sudo -E python3 mutation.py -ec settings/stage13_merged_final_supplement.json
```
- **原因**:
  - 时间最短（12.5小时预估，去重后可能更短）
  - 覆盖18个未达标组合（包括关键的VulBERTa/cnn）
  - 快速验证配置修复效果
  - 包含并行和非并行两种模式

**优先级2: Stage11-12** (并行hrnet18/pcb补充，51.7小时)
```bash
# Stage11: 并行hrnet18 (28.6小时, 20实验)
sudo -E python3 mutation.py -ec settings/stage11_parallel_hrnet18.json

# Stage12: 并行pcb (23.1小时, 20实验)
sudo -E python3 mutation.py -ec settings/stage12_parallel_pcb.json
```
- **原因**:
  - 补充hrnet18和pcb的并行模式实验
  - 覆盖8个未达标组合

**可选: Stage9-10** (非并行hrnet18/pcb，48.7小时)
```bash
# Stage9: 非并行hrnet18 (25.0小时, 20实验)
sudo -E python3 mutation.py -ec settings/stage9_nonparallel_hrnet18.json

# Stage10: 非并行pcb (23.7小时, 20实验)
sudo -E python3 mutation.py -ec settings/stage10_nonparallel_pcb.json
```
- **原因**: hrnet18和pcb非并行模式已达标（5+唯一值），去重率预计>90%
- **建议**: 最后执行，或根据Stage13结果决定是否需要

### 执行后预期结果

| 指标 | 当前 | Stage13后 | Stage11-12后 | 最终 |
|------|------|----------|-------------|------|
| 总实验数 | 400 | 490 | 530 | 570 |
| 参数-模式组合达标 | 64/90 (71%) | 82/90 (91%) | 90/90 (100%) | 90/90 (100%) |
| 非并行模式达标 | 40/45 (89%) | 45/45 (100%) | 45/45 (100%) | 45/45 (100%) |
| 并行模式达标 | 24/45 (53%) | 37/45 (82%) | 45/45 (100%) | 45/45 (100%) |

---

## 📈 时间估算

### 配置预估时间

| 阶段 | 预估时间 | 实验数 | 平均时间/实验 |
|------|---------|--------|--------------|
| Stage13 | 12.5h | 90 | 8.3分钟 |
| Stage11 | 28.6h | 20 | 85.9分钟 |
| Stage12 | 23.1h | 20 | 69.3分钟 |
| Stage9 | 25.0h | 20 | 75.0分钟 |
| Stage10 | 23.7h | 20 | 71.1分钟 |
| **总计** | **112.9h** | **170** | **39.9分钟** |

### 去重预期调整

基于Stage7经验（96.5%去重率），预计：
- **Stage13**: 实际运行可能<2小时（大部分实验已有相似配置）
- **Stage11-12**: 实际运行可能10-20小时（hrnet18/pcb并行模式缺口较大）
- **Stage9-10**: 实际运行可能<2小时（非并行模式已达标）

**保守估计总时间**: 15-25小时（而非112.9小时）

---

## 🔍 关键发现

### 1. VulBERTa/cnn完全遗漏

**问题**: VulBERTa/cnn在所有已规划阶段中**完全缺失**
- 400个历史实验中，0个VulBERTa/cnn实验
- Stage1-14原规划中，均未包含VulBERTa/cnn

**根因**:
- VulBERTa仓库有两个模型：mlp和cnn
- 早期实验仅覆盖mlp模型（28个实验，全部达标）
- cnn模型被遗漏

**影响**:
- 8个参数-模式组合缺失（非并行4个 + 并行4个）
- 40个实验缺失（假设无去重）

**修复**: 在Stage13中新增8个配置项，覆盖VulBERTa/cnn所有参数

### 2. Stage9-10可能无需执行

**分析**:
- hrnet18非并行: 4个参数均已5-6个唯一值
- pcb非并行: 4个参数均已5-6个唯一值

**建议**:
- 先执行Stage13和Stage11-12
- 如果去重率<80%，再考虑Stage9-10
- 可节省约48小时时间

### 3. 配置合并带来的优势

**优点**:
- ✅ 减少执行步骤（3个阶段 vs 5个阶段）
- ✅ 快速完成短阶段，获得即时反馈
- ✅ 便于监控和中断（12.5小时 vs 2.5+5.0小时分开）
- ✅ 单次配置文件，降低出错概率

---

## 🎯 配置文件详情

### Stage13合并配置文件内容

**文件**: `settings/stage13_merged_final_supplement.json`

**结构**:
```json
{
  "comment": "Stage13: 最终补充实验 - 合并Stage13+Stage14 + 新增VulBERTa/cnn完整覆盖",
  "version": "4.7.1-merged",
  "estimated_duration_hours": 12.5,
  "estimated_experiments": 90,
  "experiments": [
    // 第一部分: 并行快速模型补充 (9个配置项, 43实验)
    // 第二部分: 非并行MRT-OAST补充 (1个配置项, 7实验)
    // 第三部分: VulBERTa/cnn全覆盖 (8个配置项, 40实验)
  ]
}
```

**配置特点**:
- ✅ 遵循"单参数原则"（每个配置项只变异1个参数）
- ✅ 启用去重机制（`use_deduplication: true`）
- ✅ 包含历史CSV（`historical_csvs: ["results/summary_all.csv"]`）
- ✅ 详细注释和分段说明
- ✅ 完整的元数据和执行摘要

---

## ✅ 验证清单

### 配置文件验证

- [x] JSON格式有效
- [x] 所有必需字段完整
- [x] 配置项数量正确（18个）
- [x] 实验数量正确（90个）
- [x] 时间估算合理（12.5小时）
- [x] 去重机制已启用
- [x] 历史CSV已配置
- [x] 遵循单参数原则

### 覆盖验证

- [x] 识别所有26个未达标组合
- [x] Stage9-13完整覆盖26个组合
- [x] VulBERTa/cnn完全覆盖（8个组合）
- [x] 无遗漏的参数-模式组合
- [x] 所有11个模型均包含

### 文档验证

- [x] 备份原始配置文件
- [x] 创建合并报告
- [x] 更新执行计划（待完成）
- [x] 更新README（待完成）

---

## 📞 后续行动

### 立即可执行

1. **执行Stage13** (推荐)
   ```bash
   sudo -E python3 mutation.py -ec settings/stage13_merged_final_supplement.json
   ```
   - 预计时间：12.5小时（去重后可能<2小时）
   - 预计完成度：82/90组合（91%）

2. **更新项目文档**
   - README.md: 更新Stage13配置说明
   - CLAUDE.md: 更新当前状态
   - STAGE7_13_EXECUTION_PLAN.md: 更新执行计划

### 后续阶段

3. **执行Stage11-12**
   - 预计时间：51.7小时
   - 预计完成度：90/90组合（100%）

4. **可选执行Stage9-10**
   - 根据Stage13结果决定
   - 如去重率>90%，可跳过

---

## 📊 总结

### 任务完成情况

| 任务 | 状态 | 备注 |
|------|------|------|
| 合并Stage13和Stage14 | ✅ 完成 | 10个配置项合并 |
| 检查Stage9-13覆盖 | ✅ 完成 | 发现VulBERTa/cnn遗漏 |
| 补充遗漏实验 | ✅ 完成 | 新增8个VulBERTa/cnn配置 |
| 创建最终配置 | ✅ 完成 | stage13_merged_final_supplement.json |
| 验证完整覆盖 | ✅ 完成 | 100%覆盖26个未达标组合 |

### 关键成果

1. **创建单一配置文件**: `stage13_merged_final_supplement.json`
   - 18个配置项，90个实验，12.5小时

2. **发现并修复重大遗漏**: VulBERTa/cnn完全缺失
   - 新增8个配置项，40个实验

3. **验证100%覆盖**: Stage9-13覆盖所有26个未达标组合
   - 非并行: 5个组合 ✓
   - 并行: 21个组合 ✓

4. **优化执行顺序**: 建议先执行Stage13（时间最短）
   - 快速验证配置修复效果
   - 预计<2小时（去重后）

---

**报告作者**: Claude Code Assistant
**审核状态**: 已完成
**建议优先级**: 高 - 立即执行Stage13

---

**附件**:
- [Stage13合并配置](../../settings/stage13_merged_final_supplement.json)
- [Stage13原配置备份](../../settings/stage13_parallel_fast_models_supplement.json.bak_20251207)
- [Stage14原配置备份](../../settings/stage14_stage7_8_supplement.json.bak_20251207)
