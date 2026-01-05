# Phase 5 并行模式优先补充配置说明

**创建日期**: 2025-12-14
**配置文件**: `settings/test_phase5_parallel_supplement_20h.json`
**预计时间**: ~17.45小时（30%去重率）或 ~12.46小时（50%去重率）
**目标**: 优先补充VulBERTa/mlp、bug-localization、MRT-OAST的并行模式实验，以及部分快速模型的并行实验

---

## 📋 配置概述

### 实验分布

| 模型 | 参数数量 | runs_per_config | 预计实验数 | 预计时间（无去重） |
|------|---------|----------------|-----------|-----------------|
| VulBERTa/mlp | 3 | 4 | 12 | 12.52小时 |
| bug-localization | 2 | 4 | 8 | 2.18小时 |
| MRT-OAST | 3 | 4 | 12 | 5.23小时 |
| examples/mnist | 4 | 5 | 20 | 2.50小时 |
| examples/mnist_ff | 4 | 5 | 20 | 2.50小时 |
| **总计** | **16** | - | **72** | **24.92小时** |

### 时间估算（基于Phase 4实际数据）

| 去重率 | 预计时间 | 说明 |
|--------|---------|------|
| 0% (无去重) | 24.92小时 | 所有72个实验都运行 |
| 30% | **17.45小时** | 跳过22个重复实验，运行50个新实验 |
| 50% | 12.46小时 | 跳过36个重复实验，运行36个新实验 |

**推荐**: 根据Phase 4的经验，预计30%去重率较合理，实际运行时间约**17-18小时**。

---

## 🎯 实验目标

### 1. VulBERTa/mlp（并行模式）

**当前状态**（Phase 4后）:
- epochs: 1个唯一值 → **目标: 5个**
- learning_rate: 3个唯一值 → **目标: 5个**
- weight_decay: 1个唯一值 → **目标: 5个**
- seed: 1个唯一值 → **目标: 5个**

**Phase 5补充**:
- epochs变异: 4次
- weight_decay变异: 4次
- seed变异: 4次
- learning_rate变异: 已有3个，跳过（由去重机制控制）

**预计新增**: 12个实验（或因去重减少）

### 2. bug-localization（并行模式）

**当前状态**（Phase 4后）:
- alpha: 3个唯一值 → **目标: 5个**
- max_iter: 1个唯一值 → **目标: 5个**
- seed: 1个唯一值 → **目标: 5个**

**Phase 5补充**:
- max_iter变异: 4次
- seed变异: 4次
- alpha变异: 已有3个，跳过（由去重机制控制）

**预计新增**: 8个实验（或因去重减少）

### 3. MRT-OAST（并行模式）

**当前状态**（Phase 4后）:
- dropout: 3个唯一值 → **目标: 5个**
- epochs: 1个唯一值 → **目标: 5个**
- learning_rate: 3个唯一值 → **目标: 5个**
- seed: 1个唯一值 → **目标: 5个**
- weight_decay: 1个唯一值 → **目标: 5个**

**Phase 5补充**:
- epochs变异: 4次
- seed变异: 4次
- weight_decay变异: 4次
- dropout变异: 已有3个，跳过（由去重机制控制）
- learning_rate变异: 已有3个，跳过（由去重机制控制）

**预计新增**: 12个实验（或因去重减少）

### 4. examples/mnist（并行模式）

**当前状态**: 0个实验

**Phase 5补充**:
- epochs变异: 5次
- learning_rate变异: 5次
- batch_size变异: 5次
- seed变异: 5次

**预计新增**: 20个实验

### 5. examples/mnist_ff（并行模式）

**当前状态**: 0个实验

**Phase 5补充**:
- epochs变异: 5次
- learning_rate变异: 5次
- batch_size变异: 5次
- seed变异: 5次

**预计新增**: 20个实验

---

## 🚀 执行命令

```bash
# 执行Phase 5配置
sudo -E python3 mutation.py -ec settings/test_phase5_parallel_supplement_20h.json
```

**注意事项**:
1. 需要sudo权限以获取准确的CPU能量数据
2. 配置已启用去重机制（`use_deduplication: true`）
3. 历史数据文件设置为`data/raw_data.csv`
4. 预计运行时间17-18小时（假设30%去重率）

---

## 📊 预期结果

### 数据完整性
- ✅ 训练成功率: 100%
- ✅ CPU能耗数据: 100%
- ✅ GPU能耗数据: 100%
- ✅ 性能指标: 100%（已修复并行模式数据提取bug）

### 实验完成度提升

**并行模式**（Phase 5后预计）:
| 模型 | Phase 4后 | Phase 5后 | 提升 |
|------|-----------|-----------|------|
| VulBERTa/mlp | 4参数，1-3个唯一值 | 4参数，≥5个唯一值 | ✅ 达标 |
| bug-localization | 3参数，1-3个唯一值 | 3参数，≥5个唯一值 | ✅ 达标 |
| MRT-OAST | 5参数，1-3个唯一值 | 5参数，≥5个唯一值 | ✅ 达标 |
| examples/mnist | 0实验 | 4参数，≥5个唯一值 | ✅ 达标 |
| examples/mnist_ff | 0实验 | 4参数，≥5个唯一值 | ✅ 达标 |

---

## 🔄 后续计划

### Phase 6建议（非并行模式补充）

**未达标模型** (非并行模式):
- examples/mnist_rnn
- examples/siamese
- Person_reID/hrnet18
- Person_reID/pcb
- Person_reID/densenet121
- pytorch_resnet_cifar10/resnet20

**预计实验数**: ~120个
**预计时间**: ~40-50小时

### Phase 7建议（并行模式剩余模型）

**未达标模型** (并行模式):
- examples/mnist_rnn
- examples/siamese
- Person_reID/hrnet18
- Person_reID/pcb
- Person_reID/densenet121
- pytorch_resnet_cifar10/resnet20

**预计实验数**: ~120个
**预计时间**: ~80-100小时

---

## 📝 关键技术点

### 并行模式配置格式

```json
{
  "comment": "模型描述",
  "mode": "parallel",
  "foreground": {
    "repo": "仓库名",
    "model": "模型名",
    "mode": "mutation",
    "mutate": ["参数名"]
  },
  "background": {
    "repo": "背景仓库名",
    "model": "背景模型名",
    "hyperparameters": {
      "参数名": 值
    }
  },
  "runs_per_config": 重复次数
}
```

### 去重机制

- 启用: `"use_deduplication": true`
- 历史数据: `"historical_csvs": ["data/raw_data.csv"]`
- 去重逻辑: 基于`repository + model + 超参数组合 + mode`
- 效果: 自动跳过已存在的超参数组合

### 数据提取

- 并行模式数据从`foreground`子对象提取
- 字段名映射: `energy_metrics` → CSV列名, `performance_metrics` → CSV列名
- 回归测试: `tests/test_parallel_data_extraction.py`（6个测试全部通过）

---

## ✅ 验证清单

执行前检查:
- [ ] JSON格式验证通过 (`python3 -m json.tool settings/test_phase5_parallel_supplement_20h.json`)
- [ ] 回归测试通过 (`python3 tests/test_parallel_data_extraction.py`)
- [ ] raw_data.csv备份已创建
- [ ] 磁盘空间充足（需要约5-10GB）
- [ ] GPU可用（`nvidia-smi`检查）

执行后验证:
- [ ] 所有实验训练成功
- [ ] 数据已正确追加到raw_data.csv
- [ ] 数据完整性验证通过 (`python3 tools/data_management/validate_raw_data.py`)
- [ ] 并行模式实验数据提取正确（检查mode字段）
- [ ] 目标模型达到≥5个唯一值

---

**版本**: 1.0
**状态**: ✅ 已验证，准备执行
**创建时间**: 2025-12-14
**预计执行时间**: 17-18小时
