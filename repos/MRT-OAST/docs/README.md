# MRT-OAST 文档中心

本目录包含MRT-OAST项目的所有配置和使用文档。

## 📚 文档索引

### 快速开始
- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南
  - 一键测试命令
  - 常用训练配置
  - 快速参考手册

### 环境配置
- **[SETUP_CN.md](SETUP_CN.md)** - 环境配置说明
  - Conda环境安装
  - 依赖包配置
  - 环境维护指南

### 训练脚本
- **[SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** - 训练脚本详细指南
  - train_and_evaluate.sh 使用说明
  - quick_train.sh 快速训练
  - evaluate_model.sh 模型评估
  - 完整参数说明和示例

### 项目状态
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - 项目完整状态报告
  - 环境配置状态
  - 数据集详情
  - 训练配置示例
  - 常见问题解答

## 🚀 快速导航

### 我是新手，如何开始？
1. 阅读 [SETUP_CN.md](SETUP_CN.md) 配置环境
2. 查看 [QUICKSTART.md](QUICKSTART.md) 运行第一个训练
3. 参考 [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) 了解详细用法

### 我想了解训练脚本的所有功能
直接查看 [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)，包含：
- 30+ 可配置参数说明
- 多种训练场景示例
- 性能监控方法
- 常见问题解决方案

### 我想快速测试
参考 [QUICKSTART.md](QUICKSTART.md)，一行命令开始：
```bash
./quick_train.sh
```

### 我想查看项目完整配置
查看 [PROJECT_STATUS.md](PROJECT_STATUS.md) 了解：
- 已配置的环境详情
- 数据集统计信息
- 推荐的训练配置
- 项目文件结构

## 📖 文档概览

| 文档 | 大小 | 描述 | 适用对象 |
|------|------|------|----------|
| QUICKSTART.md | ~3KB | 快速开始，最常用命令 | 所有用户 |
| SETUP_CN.md | ~3KB | 环境配置详细说明 | 首次使用 |
| SCRIPTS_GUIDE.md | ~10KB | 训练脚本完整指南 | 深度使用 |
| PROJECT_STATUS.md | ~6KB | 项目状态和配置报告 | 参考查阅 |

## 🎯 常见任务快速指引

### 任务1: 首次配置环境
1. [SETUP_CN.md](SETUP_CN.md) - 安装conda环境
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - 验证数据集

### 任务2: 运行第一次训练
1. [QUICKSTART.md](QUICKSTART.md) - 一键快速测试
2. [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - 理解参数含义

### 任务3: 调整训练参数
1. [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - 查看所有可用参数
2. [QUICKSTART.md](QUICKSTART.md) - 参考配置示例

### 任务4: 评估已训练模型
1. [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - 评估脚本使用方法
2. [QUICKSTART.md](QUICKSTART.md) - 快速评估命令

### 任务5: 监控训练过程
1. [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - 日志管理和TensorBoard
2. [QUICKSTART.md](QUICKSTART.md) - 监控命令速查

## 🔧 脚本文件位置

训练脚本位于项目根目录：
- `../train_and_evaluate.sh` - 主训练脚本
- `../quick_train.sh` - 快速训练脚本
- `../evaluate_model.sh` - 模型评估脚本

## 📝 更新记录

- 2025-10-13: 创建文档中心，整合所有说明文档
- 2025-10-13: 添加训练脚本系统
- 2025-10-13: 完成环境配置和数据集准备

## ❓ 需要帮助？

1. 查看 [QUICKSTART.md](QUICKSTART.md) 快速解决常见问题
2. 查看 [PROJECT_STATUS.md](PROJECT_STATUS.md) 的故障排除部分
3. 查看 [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) 的常见问题章节

## 🌟 推荐阅读顺序

**首次使用：**
```
SETUP_CN.md → QUICKSTART.md → 开始训练
```

**深入了解：**
```
QUICKSTART.md → SCRIPTS_GUIDE.md → PROJECT_STATUS.md
```

**快速参考：**
```
QUICKSTART.md （收藏此文件）
```

---

**提示**: 所有文档都支持Markdown格式，建议使用支持Markdown预览的编辑器查看。
