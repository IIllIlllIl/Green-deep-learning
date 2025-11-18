# 项目文档索引

**最后更新**: 2025-11-18
**版本**: v4.3.0 - Enhanced Parallel Experiments & Offline Training

本目录包含Mutation-Based Training Energy Profiler项目的完整文档。

---

## 🚀 快速开始

### 新手推荐阅读顺序
1. [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) - **功能特性总览** ⭐ 了解所有功能
2. [CHANGELOG_20251118.md](CHANGELOG_20251118.md) - **v4.3.0更新日志** 🆕 了解最新改进
3. [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - **超参数范围** ⭐⭐⭐ 必读
4. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 实验配置指南

---

## 📚 核心文档

### 更新日志

#### [CHANGELOG_20251118.md](CHANGELOG_20251118.md) 🆕
**v4.3.0 更新日志** - 今日完成
- 并行实验JSON结构增强 - 完整记录前景+背景模型信息
- 离线训练环境完善 - HF_HUB_OFFLINE=1 完全离线运���
- 快速验证配置 - 1-epoch版本，15-20分钟验证全模型
- 实验数据完整性改进 - 目录结构优化

---

### 超参数变异

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

---

### 使用指南

#### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐⭐
**快速参考卡片** - 日常使用首选
- mutation.py参数速查表
- 常用命令示例
- 支持的模型列表

#### [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) ⭐⭐
**实验配置完整指南**
- settings/目录配置文件使用
- 预设配置说明（default.json, all.json等）
- 并行训练配置
- 自定义配置方法

#### [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) ⭐
**使用示例集**
- 常见研究场景
- 完整命令示例
- 批量实验脚本

---

### 功能特性

#### [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) ⭐⭐
**功能特性总览**
- 所有核心功能概览
- 功能状态矩阵
- 版本历史
- 快速开始指南

#### [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) ⭐⭐
**并行训练使用指南**
- 并行训练机制说明
- 配置方法
- 两种模式对比
- 使用示例

---

### 输出与分析

#### [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) ⭐
**输出结构快速参考**
- 分层目录结构说明
- CSV总结格式
- experiment.json详细数据说明
- 使用方法

---

### 技术实现

#### [energy_monitoring_improvements.md](energy_monitoring_improvements.md) ⭐⭐
**能耗监控改进（v2.0）**
- 直接包装（Direct Wrapping）方法
- 精度提升分析（误差<2%）
- CPU/GPU监控技术细节

---

## 🔍 按需查找

| 需求 | 推荐文档 | 优先级 |
|------|---------|--------|
| 了解最新改进 | [CHANGELOG_20251118.md](CHANGELOG_20251118.md) | ⭐⭐⭐ 🆕 |
| 了解超参数范围 | [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) | ⭐⭐⭐ |
| 了解所有功能 | [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) | ⭐⭐ |
| 快速使用命令 | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | ⭐⭐ |
| 理解变异机制 | [MUTATION_MECHANISMS_DETAILED.md](MUTATION_MECHANISMS_DETAILED.md) | ⭐⭐ |
| 配置实验 | [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) | ⭐⭐ |
| 使用并行训练 | [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) | ⭐⭐ |
| 了解输出结构 | [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) | ⭐ |
| 了解能耗监控 | [energy_monitoring_improvements.md](energy_monitoring_improvements.md) | ⭐⭐ |

---

## 📦 已归档文档

已完成的测试、实验和设计文档已归档至 `archive/` 目录：

### 2025-11-18归档 🆕
- **问题修复**: 并行目录结构修复、hrnet18 SSL证书问题
- **环境配置**: 离线训练环境设置指南和完成报告
- **相关文档**: FIX_SUMMARY_20251118.md, HRNET18_SSL_FIX.md, OFFLINE_SETUP_COMPLETION_REPORT.md, OFFLINE_TRAINING_SETUP.md

### 2025-11-17归档
- **已完成的测试**: 并行可行性测试V3设计文档（测试已完成）

### 2025-11-16归档
- **已完成的测试**: 边界测试V2、最小验证测试、并行可行性测试V1/V2
- **设计和分析**: 超参数变异策略、性能分析、完整性能表
- **文档重组记录**: 2025-11-15文档重组日志

### 2025-11-15归档
- **Bug修复文档**: 功能签名、路径重复、run_training修复
- **配置迁移文档**: 配置迁移、配置更新、脚本迁移
- **重构文档**: 重构完成、重构摘要、任务完成总结

查看归档: `ls -R docs/archive/`

---

## 📊 文档统计

**当前活跃文档**: 11个
- 更新日志: 1个 🆕
- 超参数变异: 2个
- 使用指南: 3个
- 功能特性: 2个
- 输出与分析: 1个
- 技术实现: 1个
- 索引文件: 1个

**已归档文档**: 94+个

---

## 🎯 快速导航

### 我是新手
1. [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) - 了解系统功能
2. [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - 理解超参数范围
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 学习基本命令
4. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 配置第一个实验

### 我要做研究
1. [MUTATION_MECHANISMS_DETAILED.md](MUTATION_MECHANISMS_DETAILED.md) - 理解变异机制
2. [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) - 最大化GPU利用率
3. [energy_monitoring_improvements.md](energy_monitoring_improvements.md) - 确保测量精度
4. [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) - 分析输出数据

### 我要进行并行训练
1. [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) - 并行训练使用指南
2. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 配置并行训练实验

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**文档版本**: v4.3.0
**状态**: ✅ Production Ready
