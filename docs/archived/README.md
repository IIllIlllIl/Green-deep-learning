# 归档文档说明

**归档日期**: 2025-11-09

本目录包含已完成任务或已被替代的历史文档。这些文档保留用于参考，但不再作为活跃文档维护。

---

## 📋 归档文档清单

### 历史分析文档（问题已解决）

1. **failed_models_analysis.md**
   - 内容：第一轮模型训练失败分析
   - 归档原因：问题已全部修复
   - 替代文档：FIXES_AND_TESTING.md

2. **remaining_failures_investigation.md**
   - 内容：第二轮剩余失败调查
   - 归档原因：问题已全部修复
   - 替代文档：FIXES_AND_TESTING.md

3. **hyperparameter_analysis.md**
   - 内容：初期超参数可行性分析
   - 归档原因：内容已整合到其他文档
   - 替代文档：hyperparameter_support_matrix.md

4. **original_hyperparameter_defaults.md**
   - 内容：原始超参数默认值记录
   - 归档原因：信息已整合到配置文件
   - 替代文档：config/models_config.json

---

### 开发过程文档（已完成）

5. **CONVERSATION_SUMMARY.md**
   - 内容：开发对话总结
   - 归档原因：开发阶段已完成
   - 参考价值：了解项目演化过程

6. **REORGANIZATION_SUMMARY.md**
   - 内容：代码重组总结
   - 归档原因：重组已完成
   - 参考价值：了解重构决策

7. **MUTATION_UNIQUENESS_AND_RENAME.md**
   - 内容：变异唯一性功能实现
   - 归档原因：功能已稳定运行
   - 参考价值：了解实现细节

8. **CONFIG_FILE_FEATURE.md**
   - 内容：配置文件功能开发记录
   - 归档原因：功能已完成
   - 替代文档：SETTINGS_CONFIGURATION_GUIDE.md

9. **COMPARISON_MUTATION_VS_SHELL.md**
   - 内容：mutation.py vs shell脚本方案对比
   - 归档原因：方案已确定
   - 参考价值：了解设计决策

---

### 实施指南文档（已完成）

10. **code_modification_patterns.md**
    - 内容：代码修改模式和规范
    - 归档原因：已整合到 code_modifications_log.md
    - 替代文档：code_modifications_log.md

11. **stage2_3_modification_guide.md**
    - 内容：阶段性修改指南
    - 归档原因：修改已完成
    - 替代文档：code_modifications_log.md

---

### 备份文档

12. **performance_metrics_analysis_v1_backup.md**
    - 内容：旧版性能度量分析
    - 归档原因：已被 v2 版本替代
    - 替代文档：performance_metrics_analysis.md

---

## 🔍 如何使用归档文档

### 查看归档文档

```bash
# 列出所有归档文档
ls /home/green/energy_dl/nightly/docs/archived/

# 查看特定归档文档
cat /home/green/energy_dl/nightly/docs/archived/CONVERSATION_SUMMARY.md
```

### 何时参考归档文档

1. **了解历史问题**: 查看 failed_models_analysis.md 等
2. **理解设计决策**: 查看 COMPARISON_MUTATION_VS_SHELL.md
3. **追溯项目演化**: 查看 CONVERSATION_SUMMARY.md
4. **研究实现细节**: 查看 MUTATION_UNIQUENESS_AND_RENAME.md

### 注意事项

⚠️ **归档文档可能包含过时信息**：
- 问题描述可能已不适用
- 解决方案可能已被更新
- 配置示例可能已改变

✅ **使用建议**：
- 优先参考活跃文档
- 归档文档仅用于历史参考
- 如有疑问，查看对应的替代文档

---

## 📊 归档统计

- **总归档文档**: 12个
- **归档时间**: 2025-11-09
- **归档触发**: 文档清理和精简
- **活跃文档**: 15个

---

## 🔗 相关文档

- [活跃文档索引](../README.md)
- [项目主文档](../../README.md)

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
