# 项目文档索引

**最后更新**: 2025-11-11

本目录包含Mutation-Based Training Energy Profiler项目的完整文档。

---

## 🚀 快速开始

### 新手推荐阅读顺序
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考卡片
2. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 实验配置指南
3. [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) - 变异策略

---

## 📚 核心文档

### 使用指南

#### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐
**快速参考卡片** - 日常使用首选
- mutation.py参数缩写速查表
- 常用命令示例
- 支持的模型列表

#### [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
**使用示例集**
- 常见研究场景
- 完整命令示例
- 批量实验脚本

#### [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)
**问题排查与测试**
- 常见问题解决方案
- 测试验证方法
- 环境兼容性修复

---

### 配置说明

#### [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) ⭐
**实验配置完整指南**
- settings/目录配置文件使用
- 预设配置说明（default.json, all.json等）
- 自定义配置方法

#### [CONFIG_EXPLANATION.md](CONFIG_EXPLANATION.md)
**模型配置详解**
- models_config.json结构说明
- 参数映射规则
- 添加新模型指南

---

### 超参数与变异

#### [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) ⭐
**变异策略详细指南**
- 科学的变异范围设计
- 对数/均匀分布选择理论
- 完整理论依据和参考文献

#### [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) ⭐
**变异范围速查表**
- 各超参数变异范围一览
- 模型特定建议
- 性能下界阈值

#### [hyperparameter_support_matrix.md](hyperparameter_support_matrix.md)
**超参数支持矩阵**
- 各模型支持的超参数完整列表
- 默认值和范围说明

#### [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md)
**参数缩写手册**
- 所有命令行参数详细说明
- 缩写定义
- 使用示例

#### [PARAMETER_ABBREVIATIONS_SUMMARY.md](PARAMETER_ABBREVIATIONS_SUMMARY.md)
**缩写功能总结**
- 功能实现细节
- 设计原理

---

### 性能度量

#### [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md) ⭐
**性能度量分析结论**
- 12个模型度量总览
- 公共度量识别
- 推荐监控方案

#### [performance_metrics_analysis.md](performance_metrics_analysis.md)
**详细分析报告**
- 深入度量分析
- 分层度量策略
- 实施建议

#### [performance_metrics_summary.md](performance_metrics_summary.md)
**快速参考表**
- 模型度量对照表
- 配置示例

---

### 技术实现

#### [energy_monitoring_improvements.md](energy_monitoring_improvements.md)
**能耗监控改进（v2.0）**
- 直接包装（Direct Wrapping）方法
- 精度提升分析（误差<2%）
- CPU/GPU监控技术细节

#### [code_modifications_log.md](code_modifications_log.md)
**代码修改日志**
- 训练脚本修改记录
- 兼容性修复历史
- Sudo环境适配说明

---

## 🔍 按需查找

### 我想要...

| 需求 | 推荐文档 |
|------|---------|
| 快速使用命令 | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| 设计变异实验 | [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) |
| 理解变异策略 | [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) |
| 配置实验 | [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) |
| 排查问题 | [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md) |
| 添加新模型 | [CONFIG_EXPLANATION.md](CONFIG_EXPLANATION.md) |
| 查看支持的超参数 | [hyperparameter_support_matrix.md](hyperparameter_support_matrix.md) |
| 了解能耗监控 | [energy_monitoring_improvements.md](energy_monitoring_improvements.md) |
| 了解性能度量 | [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md) |
| 查找参数缩写 | [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md) |

---

## 📦 已归档文档

以下文档已完成历史使命，归档至`archived/`目录：

### 2025-11-11归档
- **边界测试系列**（5个）- 测试已完成，范围已确定
- **修复报告系列**（2个）- 问题已解决
- **进度报告系列**（3个）- 历史记录
- **实施总结系列**（3个）- 开发已完成
- **临时文档**（4个）- 任务已完成

### 早期归档（2025-11-09及之前）
- 失败分析文档（问题已修复）
- 超参数初期分析（已整合）
- 开发过程文档（已完成）
- 备份文档

查看归档: `ls docs/archived/`

---

## 📊 文档统计

**当前活跃文档**: 16个
- 使用指南: 3个
- 配置说明: 2个
- 超参数与变异: 5个
- 性能度量: 3个
- 技术实现: 2个
- 索引文件: 1个

**已归档文档**: 37个

---

## 📝 文档维护

### 命名规范
- **核心文档**: 大写字母+下划线 (如`QUICK_REFERENCE.md`)
- **详细文档**: 小写字母+下划线 (如`performance_metrics_analysis.md`)

### 归档标准
文档符合以下条件之一时应归档：
1. ✅ 任务已完成且不再需要参考
2. ✅ 内容已被新文档取代
3. ✅ 属于临时分析或开发过程记录
4. ✅ 历史问题已解决且无参考价值

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**文档版本**: v4.0 - Streamlined & Clean
