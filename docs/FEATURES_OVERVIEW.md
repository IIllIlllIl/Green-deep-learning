# 功能特性总览

**最后更新**: 2025-12-03
**版本**: v4.5.0 - Production Ready

本文档提供 Mutation-Based Training Energy Profiler 所有核心功能的概览。

---

## 🎯 核心功能

### 1. 分层目录结构与CSV总结 ✅

**功能**: 自动组织实验结果，生成汇总CSV

**目录结构**:
```
results/
└── run_20251112_150000/              # Session目录（单次运行）
    ├── summary.csv                   # 总结CSV
    ├── pytorch_resnet_cifar10_resnet20_001/
    │   ├── experiment.json
    │   ├── training.log
    │   └── energy/
    └── pytorch_resnet_cifar10_resnet20_002_parallel/
        ├── experiment.json
        ├── training.log
        ├── energy/
        └── background_logs/          # 并行训练背景日志
```

**CSV内容**:
- 实验元数据（ID、时间戳、模型）
- 动态超参数列（自动适配）
- 性能指标
- 能耗数据

**使用**: 自动生成，无需配置

**详细文档**: [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md)

---

### 2. 并行训练与冷却机制 ✅

**功能**: 在前景训练间隙运行背景训练，最大化GPU利用率

**工作原理**:
```
前景训练 → 60秒冷却（背景训练循环） → 前景训练 → ...
```

**两种模式**:

1. **无限循环模式** (默认):
   ```bash
   sudo python3 mutation.py -ec settings/parallel_example.json
   ```

2. **脚本复用模式** (推荐):
   ```bash
   sudo python3 mutation.py -ec settings/parallel_with_script_reuse.json
   ```
   - 复用同一训练脚本实例
   - 减少启动开销
   - 能耗测量更准确

**背景日志位置**: `{foreground_exp_dir}/background_logs/`

**详细文档**:
- [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md)
- [SCRIPT_REUSE_QUICKREF.md](SCRIPT_REUSE_QUICKREF.md)

---

### 3. 高精度能耗监控 ✅

**方法**: 直接包装（Direct Wrapping）

**精度提升**:
| 维度 | 旧方法 | 新方法 | 改进 |
|------|--------|--------|------|
| CPU能耗误差 | 5-10% | <2% | **提升3-5倍** |
| 时间边界误差 | 1-3秒 | 0秒 | **零误差** |
| GPU指标数量 | 1项 | 5项 | **5倍信息** |

**监控指标**:
- **CPU**: Package能耗、RAM能耗（通过perf stat）
- **GPU**: 功耗（平均/最大/最小）、温度、利用率

**使用**: 自动集成到训练流程，无需额外配置

**详细文档**: [energy_monitoring_improvements.md](energy_monitoring_improvements.md)

---

### 4. 超参数变异系统 ✅

**支持的超参数**:
- `epochs` - 训练轮次
- `learning_rate` - 学习率
- `dropout` - Dropout率
- `weight_decay` - 权重衰减
- `seed` - 随机种子

**变异范围** (科学设计):
| 超参数 | 范围 | 分布 | 依据 |
|--------|------|------|------|
| Learning Rate | [0.25×, 4×] | 对数 | 文献共识 |
| Dropout | d-0.2 到 d+0.1 | 均匀 | 边界测试验证 |
| Weight Decay | [0.1×, 10×] | 对数 | 正则化理论 |
| Epochs | [0.5×, 2×] | 均匀 | 训练时长平衡 |

**使用示例**:
```bash
# 变异单个超参数
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt learning_rate -n 10

# 变异所有超参数
python3 mutation.py -r VulBERTa -m mlp -mt all -n 20
```

**详细文档**:
- [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md)
- [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md)

---

### 5. 配置文件系统 ✅

**功能**: 批量实验配置，支持复杂实验设计

**预设配置**:
- `default.json` - 基线实验（原始超参数）
- `all.json` - 全面变异所有模型
- `parallel_example.json` - 并行训练示例
- 自定义配置 - 灵活定义实验

**配置结构**:
```json
{
  "experiment_name": "my_experiment",
  "mode": "parallel",  // 或 "default"
  "governor": "performance",
  "runs_per_config": 3,
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "hyperparameters": {"epochs": 200, "learning_rate": 0.1}
    }
  ],
  "parallel_config": {
    "background_repo": "...",
    "background_model": "...",
    "background_hyperparams": {...}
  }
}
```

**使用**:
```bash
sudo python3 mutation.py -ec settings/my_experiment.json
```

**详细文档**: [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md)

---

### 6. 自动重试与错误恢复 ✅

**功能**: 自动重试失败的训练，确保实验可靠性

**特性**:
- 默认最多重试2次（可配置）
- 重试使用相同实验目录（覆盖日志）
- 记录重试次数和错误信息
- 超时保护（可配置）

**使用**:
```bash
# 设置最大重试次数
python3 mutation.py ... --max-retries 3
```

**日志位置**: `{exp_dir}/training.log`

---

### 7. 性能指标提取 ✅

**支持的指标**:
- **分类任务**: Accuracy, Loss, F1-Score
- **检测任务**: mAP, Rank-1, Rank-5
- **ReID任务**: mAP, CMC scores
- **其他**: 根据模型自动提取

**提取方法**: 正则表达式匹配训练日志

**配置**: `config/models_config.json` 中定义提取模式

**详细文档**: [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md)

---

### 8. Governor CPU频率控制 ✅

**功能**: 设置CPU频率调度器，减少实验干扰

**支持的模式**:
- `performance` - 固定最高频率（推荐用于实验）
- `powersave` - 节能模式
- `ondemand` - 按需调节

**使用**:
```bash
# 自动设置（实验结束后恢复）
sudo python3 mutation.py ... --governor performance

# 手动设置
sudo ./governor.sh performance
```

---

### 9. 参数缩写系统 ✅

**功能**: 所有命令行参数支持简短缩写

**常用缩写**:
```bash
--repo           → -r
--model          → -m
--mutate         → -mt
--runs           → -n
--governor       → -g
--max-retries    → -mr
--experiment-config → -ec
--config         → -c
--seed           → -s
--list           → -l
```

**示例**:
```bash
# 完整形式
python3 mutation.py --repo VulBERTa --model mlp --mutate all --runs 5

# 缩写形式（等效）
python3 mutation.py -r VulBERTa -m mlp -mt all -n 5
```

**详细文档**: [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md)

---

### 10. 代码质量优化 ✅

**优化项目**:
- ✅ **消除魔法数字**: 定义常量
- ✅ **DRY原则**: 提取可复用方法
- ✅ **超时配置**: 统一超时管理
- ✅ **进程清理**: 自动cleanup，防止僵尸进程
- ✅ **错误处理**: 完善的异常处理

**测试覆盖**:
- 32个核心测试全部通过
- 14个输出结构测试通过
- 代码质量评分: 4.86/5.0

**详细文档**: [CODE_QUALITY_QUICKREF.md](CODE_QUALITY_QUICKREF.md)

---

## 📊 功能状态矩阵

| 功能 | 状态 | 版本 | 文档 |
|------|------|------|------|
| 分层目录结构 | ✅ Production | v3.0 | [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) |
| CSV总结生成 | ✅ Production | v3.0 | [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) |
| 并行训练 | ✅ Production | v2.5 | [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md) |
| 脚本复用 | ✅ Production | v2.5 | [SCRIPT_REUSE_QUICKREF.md](SCRIPT_REUSE_QUICKREF.md) |
| 高精度能耗监控 | ✅ Production | v2.0 | [energy_monitoring_improvements.md](energy_monitoring_improvements.md) |
| 超参数变异 | ✅ Production | v1.0 | [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md) |
| 配置文件系统 | ✅ Production | v1.5 | [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) |
| 自动重试 | ✅ Production | v1.0 | - |
| 性能指标提取 | ✅ Production | v1.0 | [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md) |
| Governor控制 | ✅ Production | v1.0 | - |
| 参数缩写 | ✅ Production | v1.5 | [mutation_parameter_abbreviations.md](mutation_parameter_abbreviations.md) |
| 代码质量优化 | ✅ Production | v2.5 | [CODE_QUALITY_QUICKREF.md](CODE_QUALITY_QUICKREF.md) |

---

## 🚀 快速开始指南

### 场景1: 基线实验
```bash
sudo python3 mutation.py -ec settings/default.json
```

### 场景2: 超参数变异研究
```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt learning_rate -n 10
```

### 场景3: 并行训练最大化GPU利用率
```bash
sudo python3 mutation.py -ec settings/parallel_with_script_reuse.json
```

### 场景4: 全面变异所有模型
```bash
sudo python3 mutation.py -ec settings/all.json
```

---

## 📖 文档导航

### 新手推荐
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考
2. [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 配置指南
3. [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 使用示例

### 深入学习
- 超参数变异: [HYPERPARAMETER_MUTATION_STRATEGY.md](HYPERPARAMETER_MUTATION_STRATEGY.md)
- 并行训练: [PARALLEL_TRAINING_USAGE.md](PARALLEL_TRAINING_USAGE.md)
- 能耗监控: [energy_monitoring_improvements.md](energy_monitoring_improvements.md)
- 性能度量: [PERFORMANCE_METRICS_CONCLUSION.md](PERFORMANCE_METRICS_CONCLUSION.md)

### 问题排查
- [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)

---

## 版本历史

### v4.5.0 (2025-12-03)
- ✅ CSV列不匹配修复
  - 修复_append_to_summary_all()使用错误的fieldnames问题
  - 修复summary_all.csv第321-332行列数问题（29列→37列）
  - 所有331行现在都有37列，GitHub可正常解析
- ✅ 参数精确优化（runs_per_config v2.0）
  - 每个参数使用精确的runs_per_config值
  - 资源利用率从26.9%提升到>90%
  - 减少无效尝试：从390次降低到105-145次（节省63%）

### v4.4.0 (2025-12-02)
- ✅ CSV追加bug修复
  - 修复_append_to_summary_all()方法导致的数据覆盖问题
  - 直接使用CSV模块的'a'模式安全追加数据
  - 创建11个单元测试和4个手动测试验证修复
- ✅ 去重机制
  - 支持基于历史CSV文件的实验去重
  - 自动跳过已存在的超参数组合，生成新的随机值
- ✅ 分阶段实验计划
  - 大规模实验分割成7个可管理的阶段
  - 总计256小时实验分割为20-46小时/阶段

### v4.3.0 (2025-11-19)
- ✅ 11个模型完整支持（基线+变异）
- ✅ 动态变异系统（log-uniform/uniform分布）
- ✅ 并行训练（前景+背景GPU同时利用）
- ✅ 离线训练（HF_HUB_OFFLINE=1）
- ✅ 高精度能耗监控（CPU误差<2%）
- ✅ Dropout范围优化：统一采用d-0.2到d+0.1策略

### v3.0 (2025-11-12)
- ✅ 分层目录结构
- ✅ CSV总结生成
- ✅ Bug修复（timeout TypeError）

### v2.5 (2025-11-11)
- ✅ 并行训练机制
- ✅ 脚本复用优化
- ✅ 代码质量全面提升

### v2.0 (2025-11-10)
- ✅ 高精度能耗监控
- ✅ 所有模型验证通过

### v1.5 (2025-11-09)
- ✅ 配置文件系统
- ✅ 参数缩写功能

### v1.0 (2025-11-08)
- ✅ 核心超参数变异功能
- ✅ 基础能耗监控

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**状态**: ✅ Production Ready
