# 项目文档索引

**最后更新**: 2025-11-09

---

## 📚 核心文档

### 使用指南

#### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐
**快速参考卡片**
- mutation.py 参数缩写速查表
- 常用命令和模型速查
- 日常使用最佳选择

#### [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)
**修复与测试指南**
- Sudo环境问题修复记录
- 快速验证测试说明
- 故障排查指南

---

### 配置说明

#### [CONFIG_EXPLANATION.md](CONFIG_EXPLANATION.md)
**模型配置详解**
- models_config.json 结构说明
- 参数映射规则
- 添加新模型指南

#### [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md)
**实验配置指南**
- settings/ 目录配置文件使用
- 预设配置说明
- 自定义配置方法

---

### 性能度量

#### [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md) ⭐
**性能度量分析结论**
- 12个模型度量总览
- 公共度量快速答案
- 推荐方案可视化

#### [performance_metrics_analysis.md](performance_metrics_analysis.md)
**详细分析报告**
- 深入度量分析
- 分层度量策略
- 实施指南

#### [performance_metrics_summary.md](performance_metrics_summary.md)
**快速参考表**
- 模型度量对照表
- 配置示例

---

### 参数说明

#### [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) ⭐ **NEW**
**变异范围速查表**
- 各参数变异范围和方式
- 模型特定建议
- 性能下界阈值
- **适合**: 设计变异实验时快速查询

#### [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) ⭐ **NEW**
**变异策略详细指南**
- 科学的变异范围设计
- 对数/均匀分布选择
- 完整理论依据和参考文献
- **适合**: 深入理解变异策略

#### [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md)
**参数缩写手册**
- 所有参数详细说明
- 缩写定义
- 使用示例

#### [PARAMETER_ABBREVIATIONS_SUMMARY.md](PARAMETER_ABBREVIATIONS_SUMMARY.md)
**缩写功能总结**
- 功能实现细节
- 设计原理
- 使用建议

#### [hyperparameter_support_matrix.md](hyperparameter_support_matrix.md)
**超参数支持矩阵**
- 各模型支持的超参数
- 参数范围说明

---

### 技术文档

#### [energy_monitoring_improvements.md](energy_monitoring_improvements.md)
**能耗监控改进**
- v2.0 直接包装方法
- 精度提升分析
- 技术实现细节

#### [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
**使用示例集**
- 常见研究场景
- 命令示例
- 批量实验方法

#### [code_modifications_log.md](code_modifications_log.md)
**代码修改日志**
- 训练脚本修改记录
- 兼容性修复

#### [full_test_run_guide.md](full_test_run_guide.md)
**完整测试指南**
- 全模型测试流程
- 监控和分析

---

## 🗂️ 文档分类

### 按用途分类

**快速上手**:
- QUICK_REFERENCE.md
- PERFORMANCE_METRICS_CONCLUSION.md

**深入理解**:
- CONFIG_EXPLANATION.md
- performance_metrics_analysis.md
- energy_monitoring_improvements.md

**问题解决**:
- FIXES_AND_TESTING.md
- code_modifications_log.md

**参考查询**:
- hyperparameter_support_matrix.md
- performance_metrics_summary.md
- mutation_parameter_abbreviations.md

---

## 🔍 快速查找

### 我想要...

#### 快速使用命令
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

#### 了解性能度量
→ [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md)

#### 设计变异实验
→ [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - 快速查询
→ [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) - 深入理解

#### 查找参数缩写
→ [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md)

#### 配置实验
→ [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md)

#### 排查问题
→ [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)

#### 添加新模型
→ [CONFIG_EXPLANATION.md](CONFIG_EXPLANATION.md)

#### 查看超参数
→ [hyperparameter_support_matrix.md](hyperparameter_support_matrix.md)

#### 了解能耗监控
→ [energy_monitoring_improvements.md](energy_monitoring_improvements.md)

---

## 📦 已归档文档

以下文档已完成任务并归档至 `archived/` 目录：

### 历史分析文档
- `failed_models_analysis.md` - 第一轮失败分析（已修复）
- `remaining_failures_investigation.md` - 第二轮失败调查（已修复）
- `hyperparameter_analysis.md` - 初期超参数分析（已整合）
- `original_hyperparameter_defaults.md` - 原始默认值记录（已整合）

### 开发过程文档
- `CONVERSATION_SUMMARY.md` - 对话总结（已完成）
- `REORGANIZATION_SUMMARY.md` - 重组总结（已完成）
- `MUTATION_UNIQUENESS_AND_RENAME.md` - 变异唯一性实现（已完成）
- `CONFIG_FILE_FEATURE.md` - 配置文件功能开发（已完成）
- `COMPARISON_MUTATION_VS_SHELL.md` - 方案对比（已完成）

### 实施指南文档
- `code_modification_patterns.md` - 代码修改模式（已整合到 log）
- `stage2_3_modification_guide.md` - 阶段性修改指南（已完成）

### 备份文档
- `performance_metrics_analysis_v1_backup.md` - 旧版度量分析

**查看归档**: `ls docs/archived/`

---

## 📊 文档统计

**当前活跃文档**: 15个
- 快速参考: 2个
- 配置说明: 2个
- 性能度量: 3个
- 参数说明: 3个
- 技术文档: 4个
- 索引文件: 1个

**已归档文档**: 12个

---

## 📝 文档编写规范

### 命名规范
- **核心文档**: 大写字母 + 下划线 (如 `QUICK_REFERENCE.md`)
- **详细文档**: 小写字母 + 下划线 (如 `performance_metrics_analysis.md`)

### 内容规范
1. 包含最后更新日期
2. 清晰的文档用途说明
3. 适当的交叉引用
4. 代码示例和用法说明

### 维护规范
1. 定期更新内容
2. 及时归档过时文档
3. 保持索引同步
4. 添加状态标识

---

## 🗑️ 归档策略

### 何时归档

文档符合以下条件之一时应归档：
1. ✅ 任务已完成且不再需要参考
2. ✅ 内容已被新文档取代
3. ✅ 属于临时分析或开发过程记录
4. ✅ 历史问题已解决且无参考价值

### 归档方法

```bash
# 移动到归档目录
mv docs/old_document.md docs/archived/

# 更新 README.md 索引
# 从活跃文档列表移除，添加到归档说明
```

---

## 📞 需要帮助？

1. **快速查找**: 使用上方"快速查找"部分
2. **问题排查**: [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)
3. **主项目文档**: [../README.md](../README.md)
4. **查看归档**: `docs/archived/` 目录

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**文档版本**: v3.0 - Streamlined Edition
