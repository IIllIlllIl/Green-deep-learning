# Stage特定分析脚本归档

**归档日期**: 2025-12-06
**归档原因**: 相关Stage实验已完成

---

## 归档脚本列表

| 脚本名称 | 功能 | 关联Stage | 完成日期 |
|---------|------|----------|---------|
| `analyze_stage2_results.py` | Stage2结果分析 | Stage2 | 2025-12-04 |
| `merge_stage3_stage4.py` | 合并Stage3-4配置 | Stage3-4 | 2025-12-05 |

---

## 脚本说明

### analyze_stage2_results.py
**功能**: 分析Stage2实验执行结果
- 统计实验数量和运行时长
- 分析参数唯一值数量变化
- 检查数据完整性
- 对比预期vs实际完成情况

**Stage2背景**:
- 实验名称: 非并行补充 + 快速模型并行
- 实际运行: 7.3小时（预期20-24小时）
- 实验数量: 25个（预期44个）
- 去重效果: 61.5%跳过率

### merge_stage3_stage4.py
**功能**: 合并Stage3和Stage4配置文件
- 合并两个JSON配置文件
- 重新计算实验数量和时间预估
- 生成合并后的配置文件
- 提供归档建议

**合并结果**:
- 合并后配置: `stage3_4_merged_optimized_parallel.json`
- 实验项: 37个
- 预期实验数: 57个（基于Stage2完成率重估）
- 实际完成: 25个（2025-12-05）

---

## 保留原因

1. **历史记录**: 保留Stage执行分析过程
2. **配置参考**: 记录配置合并方法
3. **经验总结**: Stage2-4的去重率和完成率数据

---

## 替代工具

**不再需要stage-specific分析脚本**，推荐使用：
- `scripts/analyze_experiments.py` - 统一的实验分析工具
- 直接查看执行报告: `docs/results_reports/STAGE*_EXECUTION_REPORT.md`

---

**注意**: 这些脚本为特定Stage定制，不适用于其他Stage实验。
