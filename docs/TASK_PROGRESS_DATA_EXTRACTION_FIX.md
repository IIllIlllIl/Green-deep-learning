# 数据提取问题修复 - 任务进度记录

**创建日期**: 2025-12-12
**最后更新**: 2025-12-12 22:30
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

### Phase 2: 数据提取问题诊断 ⏳ **待开始**

**目的**: 运行测试实验，收集输出，分析问题模型的实际输出格式

**子任务**:
1. ⏳ 运行测试配置 `settings/test_data_extraction_debug.json` (8个实验)
2. ⏳ 收集 `terminal_output.txt` 文件
3. ⏳ 提取性能指标关键词
4. ⏳ 对比正常模型和问题模型的输出差异
5. ⏳ 记录每个问题模型的实际指标格式

**预计时间**: 4-6小时（实验运行） + 2小时（分析）
**下一步**: 运行 `sudo -E python3 mutation.py -ec settings/test_data_extraction_debug.json`

### Phase 3: 性能指标提取脚本修复 ⏳ **待开始**

**目的**: 根据诊断结果，更新性能指标提取逻辑以支持问题模型

**子任务**:
1. ⏳ 定位性能指标提取代码位置
2. ⏳ 为 examples/mnist_ff 添加提取规则
3. ⏳ 为 VulBERTa/mlp 添加提取规则
4. ⏳ 为 bug-localization 添加提取规则
5. ⏳ 为 MRT-OAST 修复特定批次问题
6. ⏳ 在少量实验上测试修复效果

**预计时间**: 2-4小时

### Phase 4: 历史数据重新提取 ⏳ **待开始**

**目的**: 使用修复后的脚本重新提取151个实验的性能数据

**子任务**:
1. ⏳ 备份当前 `results/raw_data.csv`
2. ⏳ 运行重新提取脚本
3. ⏳ 验证数据完整性 (458个实验全部有性能数据)
4. ⏳ 更新汇总文件和报告
5. ⏳ 重新运行完成度分析

**预计时间**: 1-2小时

### Phase 5: 验证与文档更新 ⏳ **待开始**

**目的**: 确认修复成功，更新项目文档

**子任务**:
1. ⏳ 验证有效实验从327提升至458 (100%)
2. ⏳ 验证数据完整模型从7提升至11 (100%)
3. ⏳ 重新计算实验完成度
4. ⏳ 更新项目报告和文档
5. ⏳ 归档旧版本数据和报告

**预计时间**: 1-2小时

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
| Phase 2: 数据提取问题诊断 | ⏳ 待开始 | 预计2025-12-13 | 需运行4-6小时实验 |
| Phase 3: 提取脚本修复 | ⏳ 待开始 | 预计2025-12-13 | 依赖Phase 2结果 |
| Phase 4: 历史数据重新提取 | ⏳ 待开始 | 预计2025-12-13 | 依赖Phase 3完成 |
| Phase 5: 验证与文档更新 | ⏳ 待开始 | 预计2025-12-13 | 最终验证 |

---

## 📂 相关文件清单

### 新增/修改的代码文件
- ✅ `mutation/command_runner.py` - 添加终端输出捕获功能
- ✅ `tests/test_terminal_output_capture.py` - 自动化测试套件

### 新增的配置文件
- ✅ `settings/test_data_extraction_debug.json` - 8个调试实验配置

### 新增的文档文件
- ✅ `docs/TERMINAL_OUTPUT_CAPTURE_GUIDE.md` - 使用指南
- ✅ `docs/results_reports/DATA_EXTRACTION_UPDATED_20251212.md` - 问题模型分析（更新版）
- ✅ `docs/results_reports/DISTANCE_TO_GOAL_20251212.md` - 距离目标评估
- ⏳ `docs/results_reports/DATA_EXTRACTION_DIAGNOSIS_REPORT.md` - 诊断报告（待创建）
- ⏳ `docs/results_reports/EXTRACTION_SCRIPT_FIX_REPORT.md` - 修复报告（待创建）

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

**文档版本**: 1.0
**最后更新**: 2025-12-12 22:30
**下次更新**: Phase 2 完成后
**状态**: ✅ Phase 1 完成，等待启动 Phase 2
