# Person_reID Dropout 边界值测试设计

**文档创建**: 2025-11-19
**实验状态**: ⏳ 待运行
**配置文件**: `settings/person_reid_dropout_boundary_test.json`

---

## 📋 实验概述

### 目标

验证 Person_reID 模型使用 **default±0.2 dropout 策略** 的有效性，通过测试边界值 [0.3, 0.7] 的性能影响，确定该范围是否合适。

### 背景

在超参数变异实验中，dropout 参数的变异范围需要合理设定：
- **MRT-OAST**: default=0.2, range=[0.0, 0.4] ✅ 符合 default±0.2
- **Person_reID**: default=0.5, range=[0.0, 0.4] ❌ 不符合 default±0.2

Person_reID 的当前配置存在问题：
1. 默认值 0.5 超出范围 [0.0, 0.4]
2. 无法对称探索 default±0.2 的效果

**建议范围**: [0.3, 0.7] (default±0.2)

---

## 🎯 实验设计

### 测试点

| Dropout 值 | 类型 | 说明 |
|-----------|------|------|
| **0.3** | 下边界 | default - 0.2，较低正则化 |
| **0.5** | 默认值 | 当前默认值，基线对比 |
| **0.7** | 上边界 | default + 0.2，较高正则化 |

### 测试模型

Person_reID 的 3 个模型全部测试：
1. **densenet121** - DenseNet架构
2. **hrnet18** - High-Resolution Network
3. **pcb** - Part-based Convolutional Baseline

### 实验参数

```json
{
  "epochs": 60,
  "learning_rate": 0.05,
  "seed": 1334,
  "dropout": [0.3, 0.5, 0.7]  // 唯一变量
}
```

**控制变量**: 除 dropout 外，所有参数保持一致，确保单一变量对比。

### 实验配置

- **总配置数**: 9 (3个模型 × 3个dropout值)
- **每配置运行次数**: 3 (计算均值和标准差)
- **总训练运行数**: 27
- **CPU Governor**: performance

---

## ⏱️ 预计运行时间

### 完整运行 (runs_per_config=3)

| GPU配置 | 单次运行 | 总时间(27次) | 天数 |
|---------|---------|-------------|------|
| **高性能GPU** (RTX 4090, A100) | 1.0小时 | **27.3小时** | 1.1天 |
| **中等GPU** (RTX 2080Ti, V100) | 1.5小时 | **40.8小时** | 1.7天 ⭐ |
| **低性能GPU** (GTX 1080Ti) | 2.5小时 | **67.8小时** | 2.8天 |

### 快速验证 (runs_per_config=1)

| GPU配置 | 总时间(9次) |
|---------|------------|
| 中等GPU | **13.5小时** |

**推荐策略**: 两阶段
1. **阶段1**: runs_per_config=1，快速验证趋势 (~13.5小时)
2. **阶段2**: 如有意义，runs_per_config=3 获取统计数据 (~40.8小时)

---

## 🔬 预期结果

### 判断标准

通过对比 3 个 dropout 值的性能（Rank@1, Rank@5, mAP），可以判断：

| 观察结果 | 结论 | 建议行动 |
|---------|------|---------|
| **0.5 最优** | default±0.2 策略合适 | 采用 [0.3, 0.7] 范围 ✅ |
| **0.3 最优** | 下边界可能需扩展 | 考虑 [0.0, 0.5] 或 [0.2, 0.6] |
| **0.7 最优** | 上边界可能需扩展 | 考虑 [0.5, 0.9] |
| **三者相近** | dropout 影响较小 | 可使用更宽范围如 [0.0, 0.7] |

### 性能曲线预期

```
Rank@1
  ↑
  │     可能的曲线形状：
  │     1. U型：存在最优dropout值
  │     2. 单调：需要调整范围
  │     3. 平坦：dropout影响不大
  │
  └────────────────────→ Dropout
     0.3   0.5   0.7
```

---

## 📂 相关文件

### 配置文件
- `settings/person_reid_dropout_boundary_test.json` - 实验配置

### 脚本和工具
- `scripts/dropout_analysis.py` - dropout 配置分析
- `scripts/dropout_strategy_analysis.py` - default±0.2 策略分析
- `scripts/estimate_dropout_test_time.py` - 运行时间估算
- `scripts/estimate_mutation_runtime.py` - mutation.py 运行时间估算
- `scripts/validate_dropout_boundary_config.py` - 配置验证脚本

### 测试文件
- `tests/test_mutation_verification.py` - 变异方法验证测试

---

## 🚀 运行命令

### 验证配置

```bash
# 验证配置文件格式
python3 scripts/validate_dropout_boundary_config.py

# 估算运行时间
python3 scripts/estimate_dropout_test_time.py
python3 scripts/estimate_mutation_runtime.py
```

### 执行实验

```bash
# 完整运行 (推荐使用 tmux/screen)
sudo -E python3 mutation.py -ec settings/person_reid_dropout_boundary_test.json

# 或者使用环境变量
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/person_reid_dropout_boundary_test.json -g performance
```

### 快速验证版本

如需快速验证，修改配置文件：
```json
"runs_per_config": 1  // 从 3 改为 1
```

---

## 📊 预期输出

### 结果目录结构

```
results/run_YYYYMMDD_HHMMSS/
├── summary.csv                                      # 27次实验汇总
├── Person_reID_baseline_pytorch_densenet121_001/
│   ├── experiment.json                             # dropout=0.3, run 1
│   ├── training.log
│   └── energy/
├── Person_reID_baseline_pytorch_densenet121_002/   # dropout=0.3, run 2
├── Person_reID_baseline_pytorch_densenet121_003/   # dropout=0.3, run 3
├── Person_reID_baseline_pytorch_densenet121_004/   # dropout=0.5, run 1
...
└── Person_reID_baseline_pytorch_pcb_009/           # dropout=0.7, run 3
```

### 关键指标

每个实验的 `experiment.json` 包含：
```json
{
  "hyperparameters": {
    "dropout": 0.3,  // or 0.5, 0.7
    ...
  },
  "performance_metrics": {
    "rank1": 0.85,
    "rank5": 0.95,
    "map": 0.75
  },
  "energy_consumption": {
    "total_joules": 12345.67,
    ...
  }
}
```

---

## 📈 数据分析

### 收集数据

```bash
# 提取所有实验的性能指标
grep -r "rank1" results/run_*/Person_reID*/experiment.json

# 或使用 Python 分析
python3 -c "
import json
import glob
for f in glob.glob('results/run_*/Person_reID*/experiment.json'):
    with open(f) as fp:
        data = json.load(fp)
        print(f'{data[\"hyperparameters\"][\"dropout\"]},{data[\"performance_metrics\"][\"rank1\"]}')
"
```

### 计算统计

对每个 dropout 值计算：
- 均值 (mean)
- 标准差 (std)
- 最大/最小值

### 可视化建议

绘制 dropout vs 性能曲线：
- X轴: dropout (0.3, 0.5, 0.7)
- Y轴: Rank@1, Rank@5, mAP
- 误差棒: ±1 std (基于3次运行)
- 分别为 3 个模型绘制

---

## ⚠️ 重要提示

### 运行前检查

1. **GPU可用性**: `nvidia-smi` 确认GPU空闲
2. **磁盘空间**: 至少 10-15GB 可用空间
3. **数据集**: Market-1501 数据集是否已下载（首次需10分钟）
4. **会话管理**: 使用 `tmux` 或 `screen` 避免SSH断开

### 运行中监控

```bash
# 监控GPU使用
watch -n 5 nvidia-smi

# 查看实时日志
tail -f results/run_*/Person_reID*/training.log

# 检查进度
ls -lt results/run_*/ | head -20
```

### 失败处理

- **配置**: `max_retries: 2` 自动重试失败的训练
- **中断恢复**: 需要重新运行（暂不支持断点续传）
- **日志检查**: 查看 `training.log` 了解失败原因

---

## 🔗 相关文档

- [MUTATION_RANGES_QUICK_REFERENCE.md](MUTATION_RANGES_QUICK_REFERENCE.md) - 超参数范围参考
- [11_MODELS_OVERVIEW.md](11_MODELS_OVERVIEW.md) - Person_reID 模型详情
- [SETTINGS_CONFIGURATION_GUIDE.md](SETTINGS_CONFIGURATION_GUIDE.md) - 配置文件指南
- [OUTPUT_STRUCTURE_QUICKREF.md](OUTPUT_STRUCTURE_QUICKREF.md) - 输出结构说明

---

## 📝 后续工作

实验完成后：

1. **分析结果** - 比较 3 个 dropout 值的性能
2. **更新配置** - 根据结果修改 `mutation/models_config.json`
3. **文档更新** - 更新超参数范围文档
4. **创建报告** - 生成实验报告（参考 `DEFAULT_BASELINE_REPORT_20251118.md`）

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**文档版本**: v4.3.0
**状态**: ⏳ 实验待运行
