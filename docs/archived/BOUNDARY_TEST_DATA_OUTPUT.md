# 边界测试输出数据清单

**生成时间**: 2025-11-14 19:45
**测试会话**: run_20251114_160919
**配置文件**: settings/boundary_test_v2.json
**测试版本**: v4.0.5

---

## 1. 输出数据汇总

### 1.1 已完成实验数据 (5/12)

| 实验ID | 模型 | 超参数配置 | 训练日志 | CPU能耗 | GPU监控 | 总大小 |
|--------|------|-----------|---------|---------|---------|--------|
| 001 | DenseNet121 | lr=0.05, dropout=0.5 (Baseline) | 5.07 MB | 288 B | 109.8 KB | 5.18 MB |
| 002 | DenseNet121 | lr=0.0125, dropout=0.5 (LR下界) | 5.07 MB | 288 B | 109.1 KB | 5.17 MB |
| 003 | DenseNet121 | lr=0.2, dropout=0.5 (LR上界) | 5.29 MB | 288 B | 111.2 KB | 5.40 MB |
| 004 | DenseNet121 | lr=0.05, dropout=0.0 (Dropout下界) | 5.07 MB | 288 B | 111.4 KB | 5.18 MB |
| 005 | DenseNet121 | lr=0.05, dropout=0.4 (Dropout上界) | 5.07 MB | 288 B | 111.3 KB | 5.18 MB |
| **总计** | **5个实验** | - | **25.57 MB** | **1.4 KB** | **552.8 KB** | **26.11 MB** |

### 1.2 运行中实验 (2/12)

| 实验ID | 模型 | 超参数配置 | 状态 | 进度 |
|--------|------|-----------|------|------|
| 006 | MRT-OAST | lr=0.0001, dropout=0.2 (Baseline) | 🔄 运行中 | ~80% |
| 007 | MRT-OAST | lr=0.000025, dropout=0.2 (LR下界) | 🔄 运行中 | ~26% |

### 1.3 待运行实验 (5/12)

| 实验ID | 模型 | 超参数配置 | 状态 |
|--------|------|-----------|------|
| 008 | MRT-OAST | lr=0.0004, dropout=0.2 (LR上界) | ⏳ 待运行 |
| 009 | MRT-OAST | lr=0.0001, dropout=0.0 (Dropout下界) | ⏳ 待运行 |
| 010 | ResNet20 | lr=0.1, wd=0.0001 (Baseline) | ⏳ 待运行 |
| 011 | ResNet20 | lr=0.025, wd=0.0001 (LR下界) | ⏳ 待运行 |
| 012 | ResNet20 | lr=0.4, wd=0.0001 (LR上界) | ⏳ 待运行 |

---

## 2. 数据类型详细说明

### 2.1 文件结构

每个实验的输出文件结构：

```
results/run_20251114_160919/
└── {experiment_id}/
    ├── training.log              # 训练日志
    └── energy/                   # 能耗监控数据
        ├── cpu_energy_raw.txt    # CPU能耗原始数据
        ├── cpu_energy.txt        # CPU能耗汇总数据
        ├── gpu_power.csv         # GPU功率时序数据
        ├── gpu_temperature.csv   # GPU温度时序数据
        └── gpu_utilization.csv   # GPU利用率时序数据
```

### 2.2 数据类型表

| 数据类型 | 文件名 | 描述 | 大小 | 格式 | 采样频率 |
|---------|--------|------|------|------|---------|
| 训练日志 | `training.log` | 完整的训练输出日志，包含epoch进度、损失、准确率等 | ~5 MB/实验 | 文本 | 实时 |
| CPU能耗(原始) | `energy/cpu_energy_raw.txt` | `perf stat` 原始输出 | ~323 B | 文本 | 训练结束 |
| CPU能耗(解析) | `energy/cpu_energy.txt` | 解析后的焦耳值（Package + RAM） | ~288 B | 文本 | 训练结束 |
| GPU功率 | `energy/gpu_power.csv` | GPU功率消耗（瓦特） | ~37 KB | CSV | 1秒 |
| GPU温度 | `energy/gpu_temperature.csv` | GPU温度（摄氏度） | ~37 KB | CSV | 1秒 |
| GPU利用率 | `energy/gpu_utilization.csv` | GPU利用率（百分比） | ~35 KB | CSV | 1秒 |

**注**: GPU监控数据约2122个数据点/实验（对应~35分钟训练时间）

---

## 3. 数据格式详解

### 3.1 训练日志 (training.log)

**格式**: 纯文本

**内容示例**:
```
---------------------------------------
Start training...
Epoch: 0001 | LR: 0.050000
Train Loss: 2.8234 | Train Acc: 32.45%
Val Loss: 2.1234 | Val Acc: 45.67%
Best Acc: 45.67%
...
Model saved at: ./model/ft_DenseNet121/
Training curve: ./model/ft_DenseNet121/train.jpg
=========================================
```

**包含信息**:
- Epoch进度
- 学习率变化
- 训练/验证损失
- 训练/验证准确率
- 最佳准确率
- 模型保存路径

---

### 3.2 CPU能耗数据 (cpu_energy.txt)

**格式**: 文本摘要

**内容示例**:
```
CPU Energy Consumption Summary
==============================
Package Energy (Joules): 72727.90
RAM Energy (Joules): 5666.43
Total CPU Energy (Joules): 78394.33

Note: Measured using perf stat with direct command wrapping
```

**数据说明**:
- **Package Energy**: CPU包能耗（焦耳）
- **RAM Energy**: 内存能耗（焦耳）
- **Total CPU Energy**: 总CPU能耗（焦耳）
- 测量方法: `perf stat` 直接包装命令

**示例数据** (DenseNet121 Baseline):
- Package: 72,727.90 J = 20.20 Wh
- RAM: 5,666.43 J = 1.57 Wh
- Total: 78,394.33 J = 21.78 Wh

---

### 3.3 GPU功率数据 (gpu_power.csv)

**格式**: CSV (timestamp, power_draw_w)

**列说明**:
- `timestamp`: Unix时间戳（秒）
- `power_draw_w`: GPU功率消耗（瓦特）

**内容示例**:
```csv
timestamp,power_draw_w
1763107759,68.54
1763107760,68.63
1763107762,68.60
1763107764,77.02
1763107765,101.82
1763107766,187.79
1763107767,247.32
1763107768,248.97
1763107769,249.50
1763107770,250.68
```

**数据特征**:
- 采样间隔: ~1秒
- 数据点数: ~2122点/实验
- 功率范围: 68-252瓦（训练过程）
- 来源: `nvidia-smi --query-gpu=power.draw`

---

### 3.4 GPU温度数据 (gpu_temperature.csv)

**格式**: CSV (timestamp, temperature_c)

**列说明**:
- `timestamp`: Unix时间戳（秒）
- `temperature_c`: GPU温度（摄氏度）

**数据特征**:
- 采样间隔: ~1秒
- 数据点数: ~2122点/实验
- 温度范围: 约50-80°C
- 来源: `nvidia-smi --query-gpu=temperature.gpu`

---

### 3.5 GPU利用率数据 (gpu_utilization.csv)

**格式**: CSV (timestamp, utilization_percent)

**列说明**:
- `timestamp`: Unix时间戳（秒）
- `utilization_percent`: GPU利用率（百分比）

**数据特征**:
- 采样间隔: ~1秒
- 数据点数: ~2122点/实验
- 利用率范围: 0-100%
- 来源: `nvidia-smi --query-gpu=utilization.gpu`

---

## 4. 数据收集方法

### 4.1 训练日志

**收集方式**:
```bash
python train.py [args] > training.log 2>&1
```

**特点**:
- 捕获所有标准输出和标准错误
- 实时写入文件
- 包含完整的训练过程

---

### 4.2 CPU能耗

**收集方式**:
```bash
perf stat -e power/energy-pkg/,power/energy-ram/ -o cpu_energy_raw.txt python train.py [args]
```

**特点**:
- 使用Linux `perf` 工具
- 直接包装训练命令
- 测量整个训练过程的累计能耗
- 比间隔采样更准确

**解析**:
- 从 `cpu_energy_raw.txt` 提取焦耳值
- 保存到 `cpu_energy.txt`
- 脚本: `mutation/energy.py::parse_energy_metrics()`

---

### 4.3 GPU监控数据

**收集方式**:
```bash
# 后台循环采样
while true; do
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >> gpu_power.csv
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits >> gpu_temperature.csv
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >> gpu_utilization.csv
    sleep 1
done
```

**特点**:
- 1秒采样间隔
- 后台并行运行
- CSV格式便于分析
- 包含时间戳便于对齐

**实现**: `mutation/run.sh` 中的能耗监控循环

---

## 5. 数据使用场景

### 5.1 性能分析

**使用文件**: `training.log`

**可提取指标**:
- 训练准确率 (Train Acc)
- 验证准确率 (Val Acc)
- 最佳准确率 (Best Acc)
- 损失值 (Loss)
- 学习率变化 (LR)

**用途**: 评估超参数变化对模型性能的影响

---

### 5.2 能耗分析

**使用文件**:
- `cpu_energy.txt`
- `gpu_power.csv`

**可计算指标**:
- 总能耗 (焦耳/千瓦时)
- 平均功率 (瓦特)
- 峰值功率 (瓦特)
- 能耗效率 (焦耳/准确率%)

**用途**: 评估超参数变化对训练能耗的影响

---

### 5.3 系统监控

**使用文件**:
- `gpu_temperature.csv`
- `gpu_utilization.csv`

**可分析内容**:
- 温度稳定性
- 是否过热
- GPU利用率
- 是否存在瓶颈

**用途**: 确保训练过程稳定，硬件正常工作

---

## 6. 数据质量验证

### 6.1 已验证项

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 文件完整性 | ✅ | 所有5个已完成实验都有完整的文件 |
| 训练日志大小 | ✅ | ~5 MB/实验，大小一致 |
| CPU能耗数据 | ✅ | 所有实验都有焦耳值 |
| GPU监控数据点数 | ✅ | ~2122点/实验，对应~35分钟 |
| 文件格式 | ✅ | CSV格式正确，可解析 |
| 时间戳连续性 | ✅ | GPU数据时间戳连续，间隔~1秒 |

### 6.2 数据示例 (实验001 - Baseline)

```
训练时长: ~36分钟
训练日志: 5.07 MB, 包含完整epoch信息
CPU能耗: 78,394.33 J (21.78 Wh)
GPU数据点: 2122个
GPU功率范围: 68-252 W
GPU温度范围: (待提取)
GPU利用率范围: (待提取)
```

---

## 7. 数据缺失项

### 7.1 当前未收集的数据

| 数据类型 | 状态 | 说明 |
|---------|------|------|
| experiment.json | ❌ 缺失 | 实验元数据和结果汇总 |
| 性能指标解析 | ⏳ 未完成 | 需要从training.log提取 |
| 能耗汇总 | ⏳ 未完成 | 需要从GPU CSV计算总能耗 |
| 训练曲线图 | ❓ 未知 | 需检查模型保存目录 |

**原因**: 实验尚未完成最终的结果保存步骤

**计划**: 等待所有实验完成后，运行结果汇总脚本

---

### 7.2 experiment.json 预期格式

```json
{
  "experiment_id": "Person_reID_baseline_pytorch_densenet121_001",
  "timestamp": "2025-11-14T16:09:00",
  "repository": "Person_reID_baseline_pytorch",
  "model": "densenet121",
  "hyperparameters": {
    "epochs": 60,
    "learning_rate": 0.05,
    "dropout": 0.5
  },
  "duration_seconds": 2164,
  "training_success": true,
  "performance_metrics": {
    "best_acc": 0.8523,
    "final_train_acc": 0.8634,
    "final_val_acc": 0.8523
  },
  "energy_metrics": {
    "cpu_energy_joules": 78394.33,
    "gpu_energy_joules": 321234.56,
    "total_energy_joules": 399628.89,
    "avg_power_watts": 184.5
  },
  "retries": 0,
  "error_message": ""
}
```

---

## 8. 下一步数据处理

### 8.1 待执行任务

1. **等待所有实验完成** (预计明早)
2. **提取性能指标** - 从training.log解析准确率
3. **计算能耗汇总** - 从GPU CSV计算总能耗
4. **生成experiment.json** - 汇总所有数据
5. **生成summary.csv** - 创建所有实验的汇总表
6. **数据可视化** - 绘制性能vs能耗图表

### 8.2 分析计划

**主要分析目标**:
1. 超参数边界对模型性能的影响
2. 超参数边界对训练能耗的影响
3. 性能-能耗权衡分析
4. 最优超参数范围建议

---

## 9. 总结

### 9.1 当前数据状态

✅ **已收集**:
- 5个DenseNet121实验的完整数据
- 训练日志 (25.57 MB)
- CPU能耗数据 (1.4 KB)
- GPU监控数据 (552.8 KB)

🔄 **收集中**:
- 2个MRT-OAST实验正在运行

⏳ **待收集**:
- 2个MRT-OAST实验
- 3个ResNet20实验

### 9.2 数据完整性

**总体评估**: ✅ 良好

- 所有已完成实验都有完整的文件
- 能耗监控正常工作
- 数据格式正确
- 无路径重复问题（Bug #3已修复）

### 9.3 预期最终数据量

**预估总计**:
- 训练日志: ~80-100 MB (12个实验)
- CPU能耗: ~3.5 KB
- GPU监控: ~2-3 MB
- **总计**: ~85-105 MB

---

**文档版本**: 1.0
**最后更新**: 2025-11-14 19:45
**状态**: 🔄 实验进行中 (5/12完成)
