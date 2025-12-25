# Stage7 调试脚本归档

**归档日期**: 2025-12-06
**归档原因**: Stage7实验已完成，调试脚本不再需要

---

## 归档脚本列表

### Stage7特定分析脚本
- `analyze_stage7_results.py` - Stage7结果分析
- `analyze_stage7_mutation_attempts.py` - Stage7变异尝试分析

### Stage7调试工具
- `check_stage7_before_state.py` - Stage7执行前状态检查
- `reproduce_stage7_exact.py` - Stage7问题精确复现
- `track_mutate_calls.py` - 跟踪变异函数调用
- `locate_defect.py` - 定位Stage7缺陷
- `exact_simulation.py` - Stage7行为精确模拟
- `analyze_mutation_retry_mechanism.py` - 变异重试机制分析

---

## 历史背景

Stage7实验（2025-12-06）遇到了per-experiment runs_per_config配置读取问题。这些脚本是为了调试和定位该问题而创建的。

**问题**: runner.py第881行全局runs_per_config默认为1，导致per-experiment配置无法生效
**修复**: 在mutation、parallel、default三种模式中添加per-experiment值读取逻辑
**测试**: 创建5个测试用例全部通过（tests/test_runs_per_config_fix.py）

---

## 用途说明

这些脚本为临时调试工具，具有以下特点：
- **针对性强**: 专门为Stage7问题设计
- **不具通用性**: 硬编码Stage7特定参数
- **已完成使命**: Stage7问题已修复，实验已完成

---

## 保留原因

1. **历史参考**: 记录问题诊断过程
2. **问题溯源**: 如遇类似问题可参考
3. **调试示例**: 展示调试方法和工具编写

---

**不建议**: 在新实验中使用这些脚本
**建议**: 使用通用工具 `scripts/analyze_experiments.py`
