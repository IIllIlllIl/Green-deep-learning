# 项目文档索引

**最后更新**: 2025-11-15
**版本**: v6.0 - Unified Hyperparameter Ranges

本目录包含Mutation-Based Training Energy Profiler项目的完整文档。

---

## 🚀 快速开始

### 新手推荐阅读顺序
1. [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) - **功能特性总览** ⭐ 了解所有功能
2. [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - **超参数范围** ⭐⭐⭐ 必读
3. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 实验配置指南

---

## 📚 核心文档

### 超参数变异 (v4.1.0 重大更新)

#### [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) ⭐⭐⭐
**统一超参数范围快速参考** - 必读文档
- 所有超参数的统一变异范围公式
- Epochs: [0.5×, 1.5×] Uniform
- Learning Rate: [0.5×, 2.0×] Log-Uniform
- Weight Decay: [0.00001, 0.01] Log-Uniform (100%)
- Dropout: [0.0, 0.4] Uniform
- 默认值排除和重复检查机制

#### [MUTATION_MECHANISMS_DETAILED.md](MUTATION_MECHANISMS_DETAILED.md) ⭐⭐
**变异机制详解**
- 超参数变异算法详细说明
- Uniform vs Log-Uniform分布
- 默认值排除机制实现
- 代码示例和配置说明

#### [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) ⭐⭐
**变异策略理论基础**
- 科学的变异范围设计
- 对数/均匀分布选择理论
- 完整理论依据和参考文献

---

### 实验结果与验证

#### [BOUNDARY_TEST_V2_FINAL_SUMMARY.md](BOUNDARY_TEST_V2_FINAL_SUMMARY.md) ⭐⭐⭐
**边界测试V2最终总结** - 重要参考
- 12个边界测试实验完整结果
- 性能影响分析（DenseNet121崩溃、MRT-OAST最优等）
- 统一范围公式的实验依据
- 运行时间：5.46小时

#### [MINIMAL_VALIDATION_SUMMARY.md](MINIMAL_VALIDATION_SUMMARY.md) ⭐⭐
**最小验证实验总结**
- 14个验证实验设计（LR + Dropout + WD边界）
- 新范围的安全性和有效性验证
- 预期结果和验证目标
- 预计时间：6.1小时

#### [PARALLEL_FEASIBILITY_TEST_SUMMARY.md](PARALLEL_FEASIBILITY_TEST_SUMMARY.md) ⭐⭐
**并行训练可行性测试**
- 12种GPU内存组合测试设计
- 并行训练配置示例
- 快速验证（1 epoch per experiment）

#### [COMPLETE_PERFORMANCE_TABLE.md](COMPLETE_PERFORMANCE_TABLE.md) ⭐
**完整性能数据表**
- 所有boundary_test_v2实验性能数据
- 按模型分组的性能对比
- 训练时长和能耗数据

---

### 功能总览

#### [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) ⭐⭐
**功能特性总览**
- 所有核心功能概览
- 功能状态矩阵
- 版本历史
- 快速开始指南

---

### 使用指南

#### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐⭐
**快速参考卡片** - 日常使用首选
- mutation.py参数速查表
- 常用命令示例
- 支持的模型列表

#### [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) ⭐
**使用示例集**
- 常见研究场景
- 完整命令示例
- 批量实验脚本

#### [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) ⭐⭐
**实验配置完整指南**
- settings/目录配置文件使用
- 预设配置说明（default.json, all.json等）
- 并行训练配置
- 自定义配置方法

---

### 输出与分析

#### [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) ⭐
**输出结构快速参考**
- 分层目录结构说明
- CSV总结格式
- experiment.json详细数据说明
- 使用方法

#### [BOUNDARY_TEST_DATA_OUTPUT.md](BOUNDARY_TEST_DATA_OUTPUT.md) ⭐
**边界测试数据输出说明**
- boundary_test_v2输出结构
- 数据文件说明

---

### 并行训练

#### [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) ⭐⭐
**并行训练使用指南**
- 并行训练机制说明
- 配置方法
- 两种模式对比
- 使用示例

#### [PARALLEL_FEASIBILITY_TEST_DESIGN.md](PARALLEL_FEASIBILITY_TEST_DESIGN.md) ⭐
**并行可行性测试设计文档**
- 12种GPU内存组合详细设计
- 测试目标和预期结果

---

### 技术实现

#### [energy_monitoring_improvements.md](energy_monitoring_improvements.md) ⭐⭐
**能耗监控改进（v2.0）**
- 直接包装（Direct Wrapping）方法
- 精度提升分析（误差<2%）
- CPU/GPU监控技术细节

#### [HYPERPARAMS_MECHANISM_ANALYSIS.md](HYPERPARAMS_MECHANISM_ANALYSIS.md) ⭐
**超参数变异机制分析**
- hyperparams.py实现分析
- 连续随机采样 vs 离散边界测试
- 实现细节说明

---

### 测试与验证

#### [BOUNDARY_TEST_PERFORMANCE_TIME_ANALYSIS.md](BOUNDARY_TEST_PERFORMANCE_TIME_ANALYSIS.md) ⭐⭐
**边界测试性能和时长分析**
- 各模型训练时长详细数据
- 性能影响分析
- 失败案例分析

#### [BOUNDARY_TEST_V2_RESULTS.md](BOUNDARY_TEST_V2_RESULTS.md) ⭐
**边界测试V2详细结果**
- 原始实验结果数据
- 逐个实验分析

#### [MINIMAL_VALIDATION_DESIGN.md](MINIMAL_VALIDATION_DESIGN.md) ⭐
**最小验证设计文档**
- 14个实验的详细设计过程
- 实验选择理由
- 预期结果推导

---

## 🔍 按需查找

### 我想要...

| 需求 | 推荐文档 | 优先级 |
|------|---------|--------|
| 了解超参数范围 | [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) | ⭐⭐⭐ |
| 查看边界测试结果 | [BOUNDARY_TEST_V2_FINAL_SUMMARY.md](BOUNDARY_TEST_V2_FINAL_SUMMARY.md) | ⭐⭐⭐ |
| 了解所有功能 | [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) | ⭐⭐ |
| 快速使用命令 | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | ⭐⭐ |
| 理解变异机制 | [MUTATION_MECHANISMS_DETAILED.md](MUTATION_MECHANISMS_DETAILED.md) | ⭐⭐ |
| 理解变异策略理论 | [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) | ⭐⭐ |
| 配置实验 | [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) | ⭐⭐ |
| 使用并行训练 | [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) | ⭐⭐ |
| 了解输出结构 | [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) | ⭐ |
| 查看最小验证设计 | [MINIMAL_VALIDATION_SUMMARY.md](MINIMAL_VALIDATION_SUMMARY.md) | ⭐⭐ |
| 查看并行测试设计 | [PARALLEL_FEASIBILITY_TEST_SUMMARY.md](PARALLEL_FEASIBILITY_TEST_SUMMARY.md) | ⭐⭐ |
| 了解能耗监控 | [energy_monitoring_improvements.md](energy_monitoring_improvements.md) | ⭐⭐ |
| 分析性能时长 | [BOUNDARY_TEST_PERFORMANCE_TIME_ANALYSIS.md](BOUNDARY_TEST_PERFORMANCE_TIME_ANALYSIS.md) | ⭐⭐ |
| 查看完整性能数据 | [COMPLETE_PERFORMANCE_TABLE.md](COMPLETE_PERFORMANCE_TABLE.md) | ⭐ |

---

## 📦 已归档文档

以下文档已完成历史使命，归档至`archive/`目录：

### 2025-11-15归档（v6.0超参数更新）
- **Bug修复文档**（3个）- 功能签名、路径重复、run_training修复
- **配置迁移文档**（3个）- 配置迁移、配置更新、脚本迁移
- **重构文档**（3个）- 重构完成、重构摘要、任务完成总结
- **旧超参数范围分析**（4个）- 旧的范围对比、范围验证、变异分析、公式建议
- **旧分析文档**（3个）- CLI测试覆盖、脚本分析、脚本复用

### 2025-11-12归档（v5.0清理）
- **输出结构详细实施文档**（4个）- 功能已完成，保留快速参考
- **代码质量详细分析**（4个）- 优化已完成，保留快速参考
- **脚本复用详细实施**（2个）- 功能已完成，保留快速参考
- **并行训练详细设计**（4个）- 功能已完成，保留使用指南
- **采样策略文档**（2个）- 研究已完成
- **设计决策文档**（1个）- 历史记录
- **超参数详细范围**（2个）- 已整合到策略文档
- **边界测试快速参考**（1个）- 已有更新版本

查看归档: `ls docs/archive/`

---

## 📊 文档统计

**当前活跃文档**: 25个
- 超参数变异: 3个 ⭐⭐⭐
- 实验结果与验证: 4个
- 功能总览: 1个
- 使用指南: 3个
- 输出与分析: 2个
- 并行训练: 2个
- 技术实现: 2个
- 测试与验证: 3个
- 分析文档: 4个
- 索引文件: 1个

**已归档文档**: 77个（2025-11-15: +16个）

---

## 🎯 快速导航

### 我是新手
1. [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - 理解超参数范围
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 学习基本命令
3. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 配置第一个实验

### 我要做研究
1. [BOUNDARY_TEST_V2_FINAL_SUMMARY.md](BOUNDARY_TEST_V2_FINAL_SUMMARY.md) - 查看边界测试结果
2. [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) - 理解变异策略
3. [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) - 最大化GPU利用率
4. [energy_monitoring_improvements.md](energy_monitoring_improvements.md) - 确保测量精度

### 我要验证新范围
1. [MINIMAL_VALIDATION_SUMMARY.md](MINIMAL_VALIDATION_SUMMARY.md) - 最小验证实验设计
2. [PARALLEL_FEASIBILITY_TEST_SUMMARY.md](PARALLEL_FEASIBILITY_TEST_SUMMARY.md) - 并行训练测试

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**文档版本**: v6.0 - Unified Hyperparameter Ranges
**状态**: ✅ Production Ready
