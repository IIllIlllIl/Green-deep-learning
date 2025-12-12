# Stage11 Execution Report & Parallel Mode runs_per_config Bug Fix

**日期**: 2025-12-08
**版本**: v4.7.2
**阶段**: Stage11 - 并行模式hrnet18补充实验
**状态**: ⚠️ 发现并修复runs_per_config读取bug

---

## 📋 执行概况

### Stage11配置信息
- **配置文件**: `settings/stage11_parallel_hrnet18.json`
- **目标**: 补充hrnet18并行模式，4个参数各达到5个唯一值
- **配置项数**: 4个（epochs, learning_rate, seed, dropout）
- **预期runs_per_config**: 5次/参数
- **预期总实验数**: 20个（4参数 × 5次）
- **预期运行时间**: 28.6小时

### 实际执行结果
- **实际实验数**: **4个** ❌
- **实际运行时间**: 5.42小时
- **缺失实验数**: 16个（80%）
- **session目录**: `results/run_20251207_155242/`
- **运行时间**: 2025-12-07 17:25 ~ 22:50

---

## 🔴 发现的Bug

### Bug描述
**并行模式runs_per_config读取逻辑错误** - 代码仅从`foreground_config`读取`runs_per_config`，忽略了外层`experiment`级别的定义。

### 根本原因

在`mutation/runner.py`的并行模式处理中（第1007行和第1025行）：

```python
# 旧代码（错误）:
exp_runs_per_config = foreground_config.get("runs_per_config", runs_per_config)
```

**问题**:
- Stage11配置将`runs_per_config: 5`定义在**外层experiment**，而非`foreground`内部
- 代码直接从`foreground_config`读取，找不到时fallback到全局`runs_per_config`（默认值1）
- 导致每个配置项只运行1次而非5次

### 配置结构对比

**Stage11实际结构**:
```json
{
  "experiments": [
    {
      "mode": "parallel",
      "runs_per_config": 5,  // ← 在这里！（外层）
      "foreground": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "hrnet18",
        "mode": "mutation",
        "mutate": ["epochs"]
        // 没有runs_per_config
      },
      "background": {...}
    }
  ]
}
```

**旧代码期望的结构**:
```json
{
  "experiments": [
    {
      "mode": "parallel",
      "foreground": {
        "runs_per_config": 5,  // ← 旧代码期望在这里
        "repo": "...",
        "model": "...",
        "mutate": [...]
      },
      "background": {...}
    }
  ]
}
```

### Bug影响范围

**受影响的配置文件**:
1. ✅ `stage11_parallel_hrnet18.json` - 确认受影响（已完成，但不完整）
2. ⚠️ `stage12_parallel_pcb.json` - 同样结构，未运行
3. ⚠️ `stage13_merged_final_supplement.json` - 需检查并行部分

**Stage11影响**:
- 预期: 20个实验（4参数 × 5次）
- 实际: 4个实验（4参数 × 1次）
- 缺失: 16个实验（80%）
- 时间浪费: 5.4小时（实际）vs 28.6小时（预期） - 节省了23.2小时但目标未达成

---

## ✅ Bug修复

### 修复方案

修改`mutation/runner.py`的两处（并行模式mutation和default分支）：

**修复后代码**:
```python
# 新代码（正确）- 三级fallback:
# 1. 优先检查外层experiment的runs_per_config
# 2. 其次检查foreground_config的runs_per_config
# 3. 最后fallback到全局runs_per_config
exp_runs_per_config = exp.get("runs_per_config",
                              foreground_config.get("runs_per_config", runs_per_config))
```

### 修复位置
- `mutation/runner.py:1010-1011` - 并行模式mutation分支
- `mutation/runner.py:1030-1031` - 并行模式default分支

### 向后兼容性
✅ **完全兼容** - 修复后支持三种配置方式：
1. **外层定义**（推荐）: `exp["runs_per_config"]`
2. **foreground定义**: `foreground_config["runs_per_config"]`
3. **全局定义**: 配置文件顶层`runs_per_config`

优先级: 外层 > foreground > 全局

---

## 🧪 测试验证

### 测试文件
`tests/test_parallel_runs_per_config_fix.py`

### 测试用例
1. ✅ **外层定义**（Stage11场景）: `exp["runs_per_config"] = 5` → 正确返回5
2. ✅ **foreground定义**: `foreground_config["runs_per_config"] = 3` → 正确返回3
3. ✅ **两处都定义**: 外层=7, foreground=3 → 正确返回7（外层优先）
4. ✅ **都未定义**: 正确fallback到全局默认值1
5. ✅ **Stage11配置解析**: 确认20个实验（4×5）

### 测试结果
```
================================================================================
✅ All tests passed!
================================================================================
Total expected experiments: 20
Expected per Stage11: 20 (4 params × 5 runs)
✅ PASS - Stage11 would now run 20 experiments
```

---

## 📊 Stage11实验数据分析

### 实际运行的4个实验

| ID | 参数变异 | 超参数值 | 运行时间 | 性能指标 |
|----|---------|---------|---------|---------|
| hrnet18_001_parallel | epochs | 53 | 92.2分钟 | mAP: 0.759 |
| hrnet18_002_parallel | learning_rate | 0.0302 | 104.4分钟 | mAP: 0.771 |
| hrnet18_003_parallel | seed | 6621 | 111.7分钟 | mAP: 0.740 |
| hrnet18_004_parallel | dropout | 0.5436 | 104.3分钟 | mAP: 0.746 |

**特点**:
- ✅ 每个参数成功运行1次实验
- ✅ 100%训练成功率（4/4）
- ✅ 所有实验完整的能耗和性能数据
- ❌ 缺失每个参数的后续4次重复实验

### 缺失的16个实验

每个参数应该有5个唯一值，但目前只有1个：
- **epochs**: 1/5个唯一值（缺失4个）
- **learning_rate**: 1/5个唯一值（缺失4个）
- **seed**: 1/5个唯一值（缺失4个）
- **dropout**: 1/5个唯一值（缺失4个）

**总缺失**: 16个实验

---

## 🔄 重新执行计划

### 补充策略

**选项1: 重新运行完整Stage11** ⭐推荐
```bash
sudo -E python3 mutation.py -ec settings/stage11_parallel_hrnet18.json
```
- 去重机制会跳过已完成的4个实验
- 只需补充16个新实验
- 预计时间: ~23小时（28.6h × 16/20）

**选项2: 创建补充配置**
创建`stage11_supplement.json`，仅包含缺失的实验：
- 4个配置项，每个`runs_per_config: 4`
- 预计时间: ~23小时

**选项3: 与Stage12合并运行**
由于Stage12也受同样bug影响且未运行，可以：
1. 修复后运行Stage12（完整）
2. 补充Stage11剩余实验

### 推荐执行顺序

1. **立即执行**: 重新运行Stage11（修复后）
   - 使用去重机制，自动跳过4个已完成实验
   - 补充16个缺失实验

2. **后续执行**: Stage12和Stage13
   - Stage12: 与Stage11类似结构，20个实验
   - Stage13: 需验证配置中并行部分是否有同样问题

---

## 📈 summary_all.csv更新

### 当前状态
- **运行前总数**: 400个实验
- **Stage11新增**: 4个实验
- **当前总数**: 404个实验
- **文件**: `results/summary_all.csv`

### hrnet18并行模式统计
```sql
grep 'hrnet18.*parallel' results/summary_all.csv | wc -l
```
**结果**: 9个实验

**组成**:
- 历史实验: 5个
- Stage11新增: 4个
- **目标**: 20个（仍需补充11个）

---

## 🎯 关键发现总结

### Bug本质
**v4.7.0的"per-experiment runs_per_config修复"不完整**:
- ✅ 修复了非并行模式（mutation、default）
- ❌ 未修复并行模式
- 原因: 并行模式的配置结构嵌套更复杂（foreground/background）

### 之前的修复（v4.7.0）
仅修复了mutation和default模式，读取逻辑为：
```python
exp_runs_per_config = exp.get("runs_per_config", runs_per_config)
```

### 新的修复（v4.7.2）
补充修复了parallel模式，读取逻辑为：
```python
exp_runs_per_config = exp.get("runs_per_config",
                              foreground_config.get("runs_per_config", runs_per_config))
```

### 教训
1. **配置灵活性**: 同一参数在不同嵌套级别的定义需要明确优先级
2. **测试覆盖**: 之前测试未覆盖并行模式
3. **文档规范**: 需在配置文档中明确推荐的配置方式

---

## 📚 相关文档更新

### 需要更新的文档
1. ✅ `tests/test_parallel_runs_per_config_fix.py` - 新增测试
2. ⏳ `docs/FEATURES_OVERVIEW.md` - 添加v4.7.2版本说明
3. ⏳ `CLAUDE.md` - 更新当前状态和Stage11进度
4. ⏳ `README.md` - 更新版本号和Stage11状态
5. ⏳ `docs/SETTINGS_CONFIGURATION_GUIDE.md` - 明确runs_per_config优先级

### 建议新增文档
- `docs/CONFIG_PARAMETER_PRIORITY.md` - 配置参数优先级完整说明

---

## ✅ 行动清单

- [x] 1. 识别Stage11实验数不符的问题
- [x] 2. 分析根本原因（runs_per_config读取bug）
- [x] 3. 修复runner.py并行模式逻辑
- [x] 4. 编写测试验证修复
- [x] 5. 生成详细执行报告
- [ ] 6. 重新执行Stage11补充16个实验
- [ ] 7. 验证Stage12配置是否受影响
- [ ] 8. 更新项目文档
- [ ] 9. 更新版本号至v4.7.2

---

## 📝 版本信息

**修复版本**: v4.7.2
**修复文件**: `mutation/runner.py`
**测试文件**: `tests/test_parallel_runs_per_config_fix.py`
**报告日期**: 2025-12-08
**作者**: Green + Claude

---

## 🔗 相关资源

- [Stage11配置](../../settings/stage11_parallel_hrnet18.json)
- [v4.7.0修复报告](../../docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md)
- [配置最佳实践](../../docs/JSON_CONFIG_BEST_PRACTICES.md)
- [修复测试代码](../../tests/test_parallel_runs_per_config_fix.py)

---

**结论**: Stage11发现了v4.7.0修复的遗漏点。修复后需要重新运行Stage11补充16个实验，并检查Stage12/13是否受同样影响。修复已通过完整测试验证，确保向后兼容性。
