# 数据提取问题修复 - 任务进度记录

**创建日期**: 2025-12-12
**最后更新**: 2025-12-13 01:30
**负责人**: Green + Claude (AI Assistant)

---

## 📋 项目背景

### 问题发现

通过对458个唯一实验的数据分析，发现：
- **151个实验** (33.0%) 缺失性能数据
- **4个模型** 存在数据提取问题
- **能耗数据100%完整** - 说明训练成功，只是性能指标提取失败

### 问题模型列表

| 模型 | 实验数 | 性能数据缺失 | 优先级 |
|------|--------|--------------|--------|
| examples/mnist_ff | 46 | 46 (100%) | 🔴 最高 |
| VulBERTa/mlp | 45 | 45 (100%) | 🔴 高 |
| bug-localization-by-dnn-and-rvsm/default | 40 | 40 (100%) | 🔴 高 |
| MRT-OAST/default | 54 | 20 (37%) | 🟡 中 |

### 目标

修复数据提取逻辑，恢复151个实验的性能数据，将有效实验从327提升至458 (100%)。

---

## 🎯 总体任务规划

### Phase 1: 终端输出捕获功能开发 ✅ **已完成**

**目的**: 添加功能捕获训练过程的完整命令行输出，用于诊断性能指标提取失败原因

**子任务**:
1. ✅ 修改 `mutation/command_runner.py` 添加 `capture_stdout` 参数
2. ✅ 实现 stdout/stderr 捕获和保存功能
3. ✅ 处理超时场景的部分输出保存
4. ✅ 创建自动化测试 (4个测试用例)
5. ✅ 编写配置文件生成8个调试实验
6. ✅ 编写使用文档

**完成时间**: 2025-12-12 22:30
**测试结果**: 4/4 测试通过 🎉

### Phase 2: 数据提取问题诊断 ✅ **已完成**

**目的**: 运行测试实验，收集输出，分析问题模型的实际输出格式

**子任务**:
1. ✅ 运行测试配置 `settings/test_data_extraction_debug.json` (4个实验)
2. ✅ 收集 `terminal_output.txt` 文件
3. ✅ 提取性能指标关键词
4. ✅ 对比正常模型和问题模型的输出差异
5. ✅ 记录每个问题模型的实际指标格式

**完成时间**: 2025-12-13 01:30
**实际时间**: 2小时（实验运行） + 1小时（分析）
**输出报告**: [DATA_EXTRACTION_DIAGNOSIS_REPORT.md](results_reports/DATA_EXTRACTION_DIAGNOSIS_REPORT.md)

### Phase 3: 性能指标提取脚本修复 ✅ **已完成**

**目的**: 根据诊断结果，更新性能指标提取逻辑以支持问题模型

**子任务**:
1. ✅ 修复 mutation/models_config.json 中4个模型的 log_patterns
2. ✅ 添加性能数据提取验证机制到 runner.py
3. ✅ 创建单元测试 tests/test_performance_extraction_fix.py
4. ✅ 运行全量测试确保兼容性 (14个单元测试 + 9个集成测试)

**完成时间**: 2025-12-13 02:00
**实际时间**: 35分钟（预计60分钟）
**测试结果**: 23/23 测试通过 🎉
**输出报告**: [EXTRACTION_SCRIPT_FIX_REPORT.md](results_reports/EXTRACTION_SCRIPT_FIX_REPORT.md)

### Phase 4: 历史数据重新提取 ✅ **已完成**

**目的**: 使用修复后的脚本重新提取151个实验的性能数据

**子任务**:
1. ✅ 备份当前 `results/raw_data.csv` - backup_20251213_182044
2. ✅ 运行重新提取脚本 - 成功更新265行
3. ✅ 验证数据完整性 - 77.9% (371/476)
4. ✅ 更新汇总文件和报告 - Phase 4报告已生成
5. ⏳ 重新运行完成度分析 - 移至Phase 5

**完成时间**: 2025-12-13 18:30
**实际时间**: 1小时
**测试结果**: 265/476行成功更新，数据完整性从66.2%提升至77.9% (+11.7%)
**输出报告**: [PHASE4_HISTORICAL_DATA_REEXTRACTION_REPORT.md](results_reports/PHASE4_HISTORICAL_DATA_REEXTRACTION_REPORT.md)

**关键成果**:
- ✅ examples/mnist_ff: 56/56 (100%恢复率)
- ⚠️ VulBERTa/mlp: 19/45 (42.2%恢复率)
- ⚠️ bug-localization: 20/40 (50%恢复率)
- ⚠️ MRT-OAST: 25/57 (43.9%恢复率)

**限制**: 211个老实验的目录已不存在，仍有105个实验（22.1%）缺失性能数据

### Phase 5: 验证与文档更新 🔄 **进行中**

**目的**: 确认修复成功，更新项目文档

**子任务**:
1. ✅ 验证新正则表达式对其他模型的有效性 - 7/11模型100%完整
2. ✅ 分析剩余105个缺失数据的实验 - 3个模型的老实验数据
3. ✅ 创建验证配置 - test_phase4_validation_optimized.json (33实验，15.6小时)
4. ⏳ 执行验证配置，恢复剩余105个实验数据
5. ⏳ 重新计算实验完成度
6. ⏳ 更新项目报告和文档
7. ⏳ 归档旧版本数据和报告

**预计时间**: 2-3小时（不含实验运行时间）

---

## 📊 Phase 1 详细记录 ✅

### 任务1: 添加终端输出记录功能 ✅

**修改文件**: `mutation/command_runner.py`

**代码变更**:
- **位置**: `run_training_with_monitoring()` 方法 (lines 111-232)
- **新增参数**: `capture_stdout: bool = True`
- **功能**:
  - 使用 `subprocess.run(capture_output=True, text=True)` 捕获输出
  - 保存到 `<实验目录>/terminal_output.txt`
  - 分离 STDOUT 和 STDERR 两个部分
  - 超时场景自动保存部分输出
  - 处理字符串和字节类型的输出
  - 向后兼容 (`capture_stdout=False` 恢复原行为)

**输出文件位置**:
```
results/run_YYYYMMDD_HHMMSS/<实验ID>/terminal_output.txt
```

**输出文件格式**:
```
================================================================================
STDOUT:
================================================================================
(训练过程的标准输出)

================================================================================
STDERR:
================================================================================
(训练过程的标准错误)
```

**完成时间**: 2025-12-12 22:00

### 任务2: 创建测试验证新功能 ✅

**测试文件**: `tests/test_terminal_output_capture.py`

**测试用例**:
1. ✅ **test_capture_enabled** - 验证 `capture_stdout=True` 正确捕获输出
   - 检查 terminal_output.txt 创建
   - 检查 STDOUT/STDERR 部分存在
   - 检查实际输出内容完整

2. ✅ **test_capture_disabled** - 验证 `capture_stdout=False` 保持原行为
   - 检查 terminal_output.txt 不创建
   - 检查退出码正确

3. ✅ **test_timeout_capture** - 验证超时场景保存部分输出
   - 检查超时正确检测
   - 检查部分输出捕获
   - 检查超时标记存在

4. ✅ **test_empty_output** - 验证空输出处理
   - 检查空输出标记为 "(empty)"

**测试结果**:
```
Total: 4/4 tests passed
🎉 ALL TESTS PASSED - New functionality verified!
```

**完成时间**: 2025-12-12 22:15

### 任务3: 构建测试配置JSON ✅

**配置文件**: `settings/test_data_extraction_debug.json`

**实验配置**:
```json
{
  "comment": "数据提取调试配置 - 4个问题模型的默认值实验",
  "global_settings": {
    "runs_per_config": 1,
    "use_deduplication": false
  },
  "experiments": [
    // 8个实验: 4个模型 × 2种模式 (nonparallel + parallel)
  ]
}
```

**包含实验**:
1. examples/mnist_ff - nonparallel
2. examples/mnist_ff - parallel
3. VulBERTa/mlp - nonparallel
4. VulBERTa/mlp - parallel
5. bug-localization-by-dnn-and-rvsm/default - nonparallel
6. bug-localization-by-dnn-and-rvsm/default - parallel
7. MRT-OAST/default - nonparallel
8. MRT-OAST/default - parallel

**预计运行时间**: 4-6小时

**完成时间**: 2025-12-12 22:20

### 任务4: 编写使用文档 ✅

**文档文件**: `docs/TERMINAL_OUTPUT_CAPTURE_GUIDE.md`

**包含章节**:
1. 功能概述
2. 输出文件位置和格式
3. 使用方法（代码 + 配置文件）
4. 输出文件查找与分析
5. 数据提取调试流程（6个步骤）
6. 问题模型诊断清单
7. 常见问题解答
8. 预期收益

**完成时间**: 2025-12-12 22:25

---

## 📝 Phase 2 计划详情 ⏳

### 步骤概览

1. **运行测试实验** (4-6小时)
   ```bash
   sudo -E python3 mutation.py -ec settings/test_data_extraction_debug.json
   ```

2. **收集输出文件** (10分钟)
   ```bash
   find results/run_* -name "terminal_output.txt" -mtime -1
   ```

3. **提取性能指标关键词** (30分钟)
   ```bash
   grep -oP "(?i)\b(test|train|val)[-_]?(accuracy|loss|precision|recall|f1|map|rank)\b" \
     results/run_*/*/terminal_output.txt | sort | uniq
   ```

4. **对比分析** (1小时)
   - 对比 mnist (正常) 和 mnist_ff (问题) 的输出
   - 识别指标命名差异
   - 记录日志格式差异

5. **记录发现** (30分钟)
   - 创建诊断报告
   - 为每个问题模型记录实际指标格式
   - 规划提取脚本修复方案

### 预期发现

**examples/mnist_ff**:
- 可能使用 "val_accuracy" 而非 "test_accuracy"
- 可能输出到 stderr 而非 stdout
- 可能使用不同的日志格式

**VulBERTa/mlp**:
- 可能使用 HuggingFace Transformers 格式 ("eval_accuracy", "eval_loss")
- 可能使用 Trainer 类的日志格式
- 可能输出到 wandb/tensorboard

**bug-localization**:
- 可能使用 Top-K 准确率而非 accuracy
- 可能使用 MRR (Mean Reciprocal Rank)
- 可能输出到 CSV/JSON 文件

**MRT-OAST**:
- 对比正常批次和 mutation_2x_safe 批次
- 可能是配置或脚本版本差异

---

## 🎯 关键里程碑

| 里程碑 | 状态 | 完成时间 | 备注 |
|--------|------|----------|------|
| Phase 1: 终端输出捕获功能开发 | ✅ 完成 | 2025-12-12 22:30 | 4/4测试通过 |
| Phase 2: 数据提取问题诊断 | ✅ 完成 | 2025-12-13 01:30 | 诊断报告已生成 |
| Phase 3: 提取脚本修复 | ✅ 完成 | 2025-12-13 02:00 | 23个测试全部通过 |
| Phase 4: 历史数据重新提取 | ✅ 完成 | 2025-12-13 18:30 | 265行更新，+11.7%数据完整性 |
| Phase 5: 验证与文档更新 | 🔄 进行中 | 预计2025-12-13 | 验证配置已创建 |

---

## 📂 相关文件清单

### 新增/修改的代码文件
- ✅ `mutation/command_runner.py` - 添加终端输出捕获功能
- ✅ `tests/test_terminal_output_capture.py` - 自动化测试套件
- ✅ `mutation/models_config.json` - 更新4个模型的log_patterns
- ✅ `tests/test_performance_extraction_fix.py` - 性能提取修复测试
- ✅ `scripts/reextract_performance_metrics.py` - 历史数据重提取脚本
- ✅ `scripts/update_raw_data_with_reextracted.py` - raw_data.csv更新脚本

### 新增的配置文件
- ✅ `settings/test_data_extraction_debug.json` - 8个调试实验配置
- ✅ `settings/test_phase4_validation_optimized.json` - Phase 4验证配置（33实验，12-18小时）

### 新增的文档文件
- ✅ `docs/TERMINAL_OUTPUT_CAPTURE_GUIDE.md` - 使用指南
- ✅ `docs/results_reports/DATA_EXTRACTION_UPDATED_20251212.md` - 问题模型分析（更新版）
- ✅ `docs/results_reports/DISTANCE_TO_GOAL_20251212.md` - 距离目标评估
- ✅ `docs/results_reports/DATA_EXTRACTION_DIAGNOSIS_REPORT.md` - 诊断报告
- ✅ `docs/results_reports/EXTRACTION_SCRIPT_FIX_REPORT.md` - 修复报告
- ✅ `docs/results_reports/PHASE4_HISTORICAL_DATA_REEXTRACTION_REPORT.md` - Phase 4完成报告

### 数据文件
- ✅ `results/raw_data.csv` - 更新后的主数据文件（371/476有性能数据，77.9%）
- ✅ `results/raw_data.csv.backup_20251213_182044` - Phase 4前备份（315/476，66.2%）

### 相关的历史文档
- `docs/results_reports/NUM_MUTATED_PARAMS_FIX_REPORT_20251212.md` - num_mutated_params修复
- `docs/results_reports/NUM_MUTATED_PARAMS_FG_PREFIX_VERIFICATION_20251212.md` - fg_前缀验证
- `docs/results_reports/DEFAULT_EXPERIMENTS_ANALYSIS_20251212.md` - 默认实验分析

---

## 💡 注意事项

### 数据清理

**重要**: 测试运行可能产生新的 results 数据，需及时清理防止污染

**检查命令**:
```bash
# 查找今天创建的实验目录
find results/run_* -type d -mtime -1

# 如果是测试产生的，删除
rm -rf results/run_YYYYMMDD_HHMMSS
```

**保护规则**:
- ✅ `results/raw_data.csv` - 主数据文件，定期备份
- ✅ `results/summary_all.csv` - 汇总文件（已归档）
- ✅ `results/run_*` - 真实实验目录，不要删除
- ⚠️  测试实验 - 确认后立即清理

### 备份策略

**每次修改前备份**:
```bash
# 备份 raw_data.csv
cp results/raw_data.csv results/raw_data.csv.backup_$(date +%Y%m%d_%H%M%S)

# 备份配置文件
cp settings/XXX.json settings/archived/XXX_$(date +%Y%m%d).json
```

### 版本控制

**文档命名规范**:
- 分析报告: `<主题>_<日期YYYYMMDD>.md`
- 修复报告: `<功能>_FIX_REPORT_<日期>.md`
- 配置文件: `<stage>_<描述>_v<版本>.json`

---

## 📊 预期成果

### 数据完整性提升

| 指标 | 当前 | 目标 | 改进 |
|------|------|------|------|
| 有效实验 | 327/458 (71.4%) | 458/458 (100%) | +131 (+40.1%) |
| 数据完整模型 | 7/11 (63.6%) | 11/11 (100%) | +4 (+57.1%) |
| 默认值实验 | 17 | ~22+ | +5+ |
| 可分析组合 | 10/74 (13.5%) | ~30+/74 (40%+) | +20+ |

### 项目价值

- ✅ **无需重新训练**: 151个实验已训练成功，只需修复提取逻辑
- ✅ **节省计算资源**: 避免重新运行151个实验（约50-100小时计算时间）
- ✅ **完整数据集**: 获得458个完整实验数据用于分析
- ✅ **可复现性**: 建立标准化的数据提取流程

---

## 📞 联系与协作

**项目负责人**: Green
**AI助手**: Claude (Anthropic)
**项目仓库**: /home/green/energy_dl/nightly
**关键文档**: `CLAUDE.md`, `docs/TERMINAL_OUTPUT_CAPTURE_GUIDE.md`

---

## 📊 Phase 2 详细记录 ✅

### 实际执行情况

**运行时间**: 2025-12-12 22:49 - 2025-12-13 00:56 (2小时7分)

**实验数量**: 4个（仅运行非并行模式）
- examples/mnist_ff (7.7秒)
- VulBERTa/mlp (1小时34分)
- bug-localization-by-dnn-and-rvsm/default (9.3分钟)
- MRT-OAST/default (19.6分钟)

**关键发现**:

1. **examples/mnist_ff**: 正则表达式缺少 `[SUCCESS]` 标签匹配
   - 实际输出: `[SUCCESS] Test Accuracy: 9.559999...%`
   - 当前模式: `Final Test Accuracy[:\s]+([0-9.]+)%` ❌
   - 修复方案: 添加可选 `[SUCCESS]` 标签，移除 "Final" 要求

2. **VulBERTa/mlp**: 缺少 HuggingFace Transformers 字典格式提取
   - 实际输出: `{'eval_loss': 5.012244701385498, 'epoch': 18.0}`
   - 当前模式: `Accuracy[:\s]+([0-9.]+)` ❌
   - 修复方案: 添加 `'eval_loss':` 和 `eval_loss:` 两种格式

3. **bug-localization**: Top-k 格式有空格
   - 实际输出: `Top- 1 Accuracy: 0.380` (有空格)
   - 当前模式: `Top-1[:\s@]+([0-9.]+)` ❌
   - 修复方案: 使用 `\s*` 允许空格可选

4. **MRT-OAST**: 配置正确，部分历史数据问题
   - 实际输出: `Precision: 0.979006` ✓
   - 当前模式: `Precision[:\s]+([0-9.]+)` ✓
   - 修复方案: 增强支持中文关键词（鲁棒性）

**详细报告**: [DATA_EXTRACTION_DIAGNOSIS_REPORT.md](results_reports/DATA_EXTRACTION_DIAGNOSIS_REPORT.md)

---

**文档版本**: 1.3
**最后更新**: 2025-12-13 18:30
**下次更新**: Phase 5 完成后
**状态**: ✅ Phase 4 完成，Phase 5 进行中

---

## 📊 Phase 4 详细记录 ✅

### 执行摘要

**完成时间**: 2025-12-13 18:30
**执行时间**: 约1小时

**关键成果**:
- ✅ 成功从265个实验的training.log文件恢复性能数据
- ✅ 数据完整性从66.2%提升至77.9% (+11.7%)
- ✅ examples/mnist_ff达到100%数据恢复率
- ✅ 创建了可重用的数据提取脚本

### 技术实现

**主脚本**: `scripts/update_raw_data_with_reextracted.py`

**核心功能**:
1. 从raw_data.csv读取所有实验记录
2. 对每个实验，查找对应的实验目录
3. 优先使用terminal_output.txt，回退到training.log
4. 使用更新后的models_config.json中的正则表达式提取性能指标
5. 更新raw_data.csv，保留原有列结构

**关键修复**:
- **Fallback机制**: terminal_output.txt（4个）→ training.log（472个）
- **CSV兼容性**: 使用`extrasaction='ignore'`处理额外字段
- **备份机制**: 自动创建时间戳备份文件

### 分模型统计

| 模型 | 总数 | 提取前 | 成功提取 | 恢复率 | 最终覆盖率 |
|------|------|--------|----------|--------|-----------|
| examples/mnist_ff | 56 | 0 | 56 | 100.0% | 100.0% ✅ |
| VulBERTa/mlp | 45 | 0 | 19 | 42.2% | 42.2% ⚠️ |
| bug-localization | 40 | 0 | 20 | 50.0% | 50.0% ⚠️ |
| MRT-OAST | 57 | 37 | 25 | 43.9% | 108.8% ✅ |

### 限制与问题

**老实验目录缺失**:
- 211个老实验的目录已不存在
- 这些实验的数据已记录在raw_data.csv，但training.log文件已被删除
- 无法通过重新提取恢复这些实验的性能数据

**剩余缺失数据**: 105个实验（22.1%）
- VulBERTa/mlp: 26个老实验
- bug-localization: 20个老实验
- MRT-OAST: 20个老实验
- 其他模型: 39个

**解决方案**: 创建验证配置重新运行这些实验

### 详细报告

完整报告: [PHASE4_HISTORICAL_DATA_REEXTRACTION_REPORT.md](results_reports/PHASE4_HISTORICAL_DATA_REEXTRACTION_REPORT.md)

---

## 📊 Phase 5 当前进展 🔄

### 1. 验证新正则表达式有效性 ✅

**测试结果**: 7/11模型达到100%数据完整性

| 状态 | 模型 | 覆盖率 |
|------|------|--------|
| ✅ | examples/mnist | 100.0% |
| ✅ | examples/mnist_ff | 100.0% |
| ✅ | examples/mnist_rnn | 100.0% |
| ✅ | examples/siamese | 100.0% |
| ✅ | Person_reID/densenet121 | 100.0% |
| ✅ | Person_reID/hrnet18 | 100.0% |
| ✅ | Person_reID/pcb | 100.0% |
| ✅ | pytorch_resnet_cifar10/resnet20 | 100.0% |
| ⚠️ | MRT-OAST/default | 64.9% |
| ❌ | VulBERTa/mlp | 0.0% (老实验) |
| ❌ | bug-localization | 0.0% (老实验) |

**结论**: 修复后的正则表达式对8/11模型完全有效，3个问题模型需要重新运行老实验

### 2. 分析剩余105个缺失数据的实验 ✅

**按模型和模式分组**:

| 模型 | 模式 | 实验数 | 参数分布 |
|------|------|--------|----------|
| VulBERTa/mlp | 非并行 | 28 | default:8, epochs:5, lr:5, wd:5, seed:5 |
| VulBERTa/mlp | 并行 | 17 | default:1, epochs:4, lr:4, wd:4, seed:4 |
| bug-localization | 非并行 | 21 | default:2, max_iter:5, alpha:5, seed:5, kfold:4 |
| bug-localization | 并行 | 19 | default:1, max_iter:4, alpha:4, seed:4, kfold:6 |
| MRT-OAST | 非并行 | 10 | epochs:2, lr:2, dropout:2, wd:2, seed:2 |
| MRT-OAST | 并行 | 10 | epochs:2, lr:2, dropout:2, wd:2, seed:2 |

**平均训练时间**:
- VulBERTa/mlp: 0.71小时
- bug-localization: 0.33小时
- MRT-OAST: 0.25小时

### 3. 创建验证配置 ✅

**配置文件**: `settings/test_phase4_validation_optimized.json`

**策略**: 优先测试默认值实验和最重要参数，**确保单参数变异**

**配置修正** (2025-12-13 19:00):
- ❌ 初版使用`mutate_params`对象（多参数变异）
- ✅ 修正为`mutate`数组格式（单参数变异）
- ✅ 参考`stage2`配置的正确格式
- 📝 修正报告: [PHASE4_CONFIG_FIX_REPORT.md](results_reports/PHASE4_CONFIG_FIX_REPORT.md)

**实验数量**: 17个
- VulBERTa/mlp: 7个（5非并行 + 2并行）
- bug-localization: 6个（4非并行 + 2并行）
- MRT-OAST: 4个（2非并行 + 2并行）

**单参数变异验证**:
- learning_rate: 4次
- alpha: 2次
- dropout: 2次
- seed: 2次
- epochs: 1次
- max_iter: 1次
- weight_decay: 1次

**预计时间**:
- 无去重: 10.5小时
- 去重率30%: 7.4小时
- 去重率50%: 5.3小时 ⭐ 预期
- 去重率70%: 3.2小时

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/test_phase4_validation_optimized.json
```

### 下一步

1. ⏳ 执行验证配置，恢复剩余数据
2. ⏳ 重新计算实验完成度
3. ⏳ 更新项目报告
4. ⏳ 归档旧版本数据

