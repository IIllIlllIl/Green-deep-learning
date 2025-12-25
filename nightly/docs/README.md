# 项目文档索引

**最后更新**: 2025-11-19

---

## 📋 文件重要级别说明

### 核心文件（必须保留）
- **mutation.py** - 主程序
- **config/models_config.json** - 模型配置
- **settings/*.json** - 实验配置
- **scripts/run.sh** - 训练包装脚本

### 核心文档（日常参考）
- **QUICK_REFERENCE.md** - 快速参考
- **MUTATION_RANGES_QUICK_REFERENCE.md** - 变异范围
- **SETTINGS_CONFIGURATION_GUIDE.md** - 配置指南

### 参考文档（需要时查阅）
- 其他docs/目录下的文档

### 临时文件（完成后归档）
- 修复报告、进度报告
- 测试设计文档
- 工作总结

### 归档规则
**应立即归档**:
1. 问题已解决的修复文档
2. 测试已完成的设计文档
3. 带日期后缀的临时报告
4. 过于细节的技术分析

---

## 📚 当前活跃文档

### 使用指南
| 文档 | 说明 |
|------|------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速参考卡片 |
| [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) | 实验配置指南 |
| [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) | 使用示例 |

### 超参数变异
| 文档 | 说明 |
|------|------|
| [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) | 变异范围速查 |

### 功能特性
| 文档 | 说明 |
|------|------|
| [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) | 功能总览 |
| [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) | 并行训练指南 |

### 模型信息
| 文档 | 说明 |
|------|------|
| [11_MODELS_OVERVIEW.md](11_MODELS_OVERVIEW.md) | 11个模型概览 |
| [REPOSITORIES_LINKS.md](REPOSITORIES_LINKS.md) | 仓库链接 |

### 技术参考
| 文档 | 说明 |
|------|------|
| [energy_monitoring_improvements.md](energy_monitoring_improvements.md) | 能耗监控改进 |
| [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) | 输出结构参考 |
| [QUICKSTART_BASELINE.md](QUICKSTART_BASELINE.md) | 基线快速开始 |

---

## 🔍 按需查找

| 需求 | 文档 |
|------|------|
| 快速使用命令 | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| 配置实验 | [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) |
| 了解变异范围 | [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) |
| 使用并行训练 | [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) |
| 了解模型 | [11_MODELS_OVERVIEW.md](11_MODELS_OVERVIEW.md) |
| 了解能耗监控 | [energy_monitoring_improvements.md](energy_monitoring_improvements.md) |

---

## 📦 已归档文档

临时报告、已解决问题文档、细节分析已归档至 `archived/` 目录。

### 本次归档 (2025-11-19)
- 带日期后缀的临时报告 (4个)
- 已解决问题文档 (3个)
- 过于细节的技术文档 (4个)

查看归档: `ls docs/archived/`

---

## 📝 文档维护规范

### 写作原则
1. **简洁优先** - 细节不必写入文档
2. **面向使用** - 只写用户需要的信息
3. **及时归档** - 临时文档完成后立即归档

### 命名规范
- 核心文档: 大写字母 (QUICK_REFERENCE.md)
- 临时报告: 加日期后缀 (REPORT_20251119.md)

---

## 📊 文档统计

**活跃文档**: 12个
**已归档文档**: 50+个

---

**项目**: Mutation-Based Training Energy Profiler
**版本**: v4.3.0
