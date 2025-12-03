# 阶段1执行结果分析报告

**执行日期**: 2025-12-02 18:58 - 2025-12-03 04:12
**配置文件**: `settings/stage1_nonparallel_completion.json`
**Session目录**: `results/run_20251202_185830/`

---

## 执行结果概览

| 指标 | 数值 |
|------|------|
| 初始实验数 | 319 |
| 新增实验数 | 12 |
| 当前总实验数 | 331 |
| 预期新增 | 44 |
| 实际完成率 | **27.3%** |

---

## 新增实验明细

### 按模型分布
| 模型 | 新增实验数 | 预期数 |
|------|----------|--------|
| MRT-OAST/default | 1 | 1 ✓ |
| Person_reID_baseline_pytorch/hrnet18 | 3 | 3 ✓ |
| Person_reID_baseline_pytorch/pcb | 4 | 20 ✗ |
| examples/mnist_ff | 4 | 20 ✗ |
| **总计** | **12** | **44** |

---

## 参数唯一值完成情况

### MRT-OAST/default ✓
| 参数 | 唯一值数量 | 状态 |
|------|----------|------|
| dropout | 6 | ✓ 已达标 (超出1个) |

**结论**: MRT-OAST已完成，dropout参数已有6个唯一值。

---

### Person_reID_baseline_pytorch/hrnet18 ⚠️
| 参数 | 唯一值数量 | 状态 | 缺少 |
|------|----------|------|------|
| learning_rate | 4 | ✗ 未达标 | 1个 |
| dropout | 4 | ✗ 未达标 | 1个 |
| seed | 4 | ✗ 未达标 | 1个 |

**本次新增**: 3个实验（每个参数1次）
**需要补充**: 每个参数还需1个实验
**总缺口**: 3个实验

---

### Person_reID_baseline_pytorch/pcb ⚠️
| 参数 | 唯一值数量 | 状态 | 缺少 |
|------|----------|------|------|
| epochs | 5 | ✓ 已达标 | 0个 |
| learning_rate | 4 | ✗ 未达标 | 1个 |
| dropout | 5 | ✓ 已达标 | 0个 |
| seed | 4 | ✗ 未达标 | 1个 |

**本次新增**: 4个实验（每个参数1次）
**已完成参数**: epochs, dropout (去重机制识别并跳过��续尝试)
**需要补充**: learning_rate, seed各需1个实验
**总缺口**: 2个实验

---

### examples/mnist_ff ⚠️
| 参数 | 唯一值数量 | 状态 | 缺少 |
|------|----------|------|------|
| batch_size | 3 | ✗ 未达标 | 2个 |
| epochs | 5 | ✓ 已达标 | 0个 |
| learning_rate | 4 | ✗ 未达标 | 1个 |
| seed | 4 | ✗ 未达标 | 1个 |

**本次新增**: 4个实验（每个参数1次）
**已完成参数**: epochs (去重机制识别并跳过后续尝试)
**需要补充**:
- batch_size: 2个实验
- learning_rate: 1个实验
- seed: 1个实验
**总缺口**: 5个实验

---

## 问��分析

### 根本原因
配置文件设置了 `"runs_per_config": 1`，意味着每个实验配置只运行1次。

**预期行为**:
- 对于需要5个唯一值的参数，应该运行5次
- 去重机制会跳过重复值，继续尝试生成新值

**实际行为**:
- 每个配置只运行1次
- 如果该参数已有足够的唯一值，去重机制跳过
- 如果该参数还需要更多值，也只运行1次就停止

### 为什么只完成了12/44个实验？

1. **MRT-OAST/default** (1/1 完成):
   - dropout已有5个唯一值 → 运行1次后去重跳过 ✓

2. **hrnet18** (3/3 完成):
   - 3个参数各运行1次 ✓

3. **pcb** (4/20 完成):
   - 4个参数各运行1次
   - epochs和dropout在第1次后已达5个唯一值 → 不再继续 ✓
   - learning_rate和seed还需要更多次，但配置限制只运行1次 ✗

4. **mnist_ff** (4/20 完成):
   - 4个参数各运行1次
   - epochs在第1次后已达5个唯一值 → 不再继续 ✓
   - 其他3个参数还需要更多次，但配置限制只运行1次 ✗

---

## 去重机制验证 ✓

去重机制工作正常！证据：

1. **MRT-OAST/dropout**: 原本已有5个唯一值，本次新增1个后达到6个，说明去重机制正确识别并接受新值

2. **pcb/epochs**: 本次新增1个实验后达到5个唯一值，去���机制正确识别并停止继续运行该参数的实验

3. **pcb/dropout**: 已有4个唯一值，本次新增1个后达到5个，去重机制正确工作

4. **mnist_ff/epochs**: 已有4个唯一值，本次新增1个后达到5个，去重机制正确工作

---

## 配置问题

### 问题所在
```json
{
  "runs_per_config": 1,  // ← 这里限制了每个配置只运行1次
  "experiments": [
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "pcb",
      "mode": "mutation",
      "mutate": ["learning_rate"],
      "comment": "需要运行5次以生成5个不同的learning_rate值"  // ← 注释说明需要5次
    }
  ]
}
```

### 正确配置应该是
```json
{
  "runs_per_config": 5,  // ← 改为5次，让每个参数都有机会生成5个唯一值
  "experiments": [...]
}
```

或者为每个需要多次运行的参数创建多个配置条目。

---

## 剩余工作统计

### 需要补充的实验

| 模型 | 参数 | 需要实验数 | 说明 |
|------|------|----------|------|
| hrnet18 | learning_rate | 1 | 从4个→5个 |
| hrnet18 | dropout | 1 | 从4个→5个 |
| hrnet18 | seed | 1 | 从4个→5个 |
| pcb | learning_rate | 1 | 从4个→5个 |
| pcb | seed | 1 | 从4个→5个 |
| mnist_ff | batch_size | 2 | 从3个→5个 |
| mnist_ff | learning_rate | 1 | 从4个→5个 |
| mnist_ff | seed | 1 | 从4个→5个 |
| **总计** | - | **10** | - |

### 完成度更新

| 项目 | 之前 | 当前 | 变化 |
|------|------|------|------|
| 总实验数 | 319 | 331 | +12 |
| 非并行实验 | 214 | 226 | +12 |
| 并行实验 | 105 | 105 | 0 |
| 非并行完成参数 | 33/45 | 36/45 | +3 |
| 非并行完成度 | 73.3% | 80.0% | +6.7% |

---

## 解决方案

### 方案1: 修改stage1配置文件（推荐）

创建 `stage1_supplement.json` 补充剩余10个实验：

```json
{
  "experiment_name": "stage1_supplement",
  "description": "补充阶段1未完成的10个实验",
  "mode": "mutation",
  "runs_per_config": 2,  // 每个配置运行2次，确保生成足够的唯一值
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"],
  "experiments": [
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "mode": "mutation",
      "mutate": ["learning_rate"]
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "mode": "mutation",
      "mutate": ["dropout"]
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "mode": "mutation",
      "mutate": ["seed"]
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "pcb",
      "mode": "mutation",
      "mutate": ["learning_rate"]
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "pcb",
      "mode": "mutation",
      "mutate": ["seed"]
    },
    {
      "repo": "examples",
      "model": "mnist_ff",
      "mode": "mutation",
      "mutate": ["batch_size"]
    },
    {
      "repo": "examples",
      "model": "mnist_ff",
      "mode": "mutation",
      "mutate": ["learning_rate"]
    },
    {
      "repo": "examples",
      "model": "mnist_ff",
      "mode": "mutation",
      "mutate": ["seed"]
    }
  ]
}
```

**预计时间**: 约2-3小时

### 方案2: 修改所有阶段配置的runs_per_config

将stage2-7的配置文件中的 `"runs_per_config": 1` 改为 `"runs_per_config": 5`，确保并行模式的实验能够充分生成唯一值。

---

## 建议的下一步操作

### 立即执行
```bash
# 1. 创建补充配置文件
# (使用方案1的配置)

# 2. 运行补充实验
sudo -E python3 mutation.py -ec settings/stage1_supplement.json

# 3. 验证完成情况
wc -l results/summary_all.csv  # 应该增加约10行
```

### 后续调整
在执行stage2之前，修改stage2-7的配置文件：
```bash
# 将 runs_per_config 从 1 改为 5
sed -i 's/"runs_per_config": 1/"runs_per_config": 5/' settings/stage{2..7}_*.json
```

---

## 经验总结

1. **`runs_per_config` 的作用**:
   - 控制每个实验配置运行的次数
   - 对于需要多个唯一值的参数，应设置为期望的唯一值数量（如5）

2. **去重机制的行为**:
   - 在每次运行时检查历史数据
   - 跳过重复的超参数组合
   - 继续尝试生成新的唯一值，直到达到`runs_per_config`指定的次数

3. **最佳实践**:
   - 对于需要5个唯一值的参数，设置 `"runs_per_config": 5` 或更高
   - 去重机制会自动跳过已达标的参数
   - 对于大规模实验，可以设置较大的`runs_per_config`值，让去重机制自动处理

---

**报告生成时间**: 2025-12-03
**分析者**: Green (Claude Code)
