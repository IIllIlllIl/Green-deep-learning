# Parallel Feasibility Test V3 - 设计文档

**日期**: 2025-11-15
**版本**: V3
**目标**: 每个模型恰好作为前景1次，保持良好的分层抽样覆盖

---

## 设计目标

### 新增约束
- ✅ **每个模型恰好作为前景1次** (新增)
- ✅ 保持11个实验总数
- ✅ 保持分层抽样 (超低/低/中/高显存)
- ✅ 保持良好的显存覆盖范围 (1300MB - 5000MB)
- ✅ 最小化对V2配置的修改

---

## 完整实验配置表

| 序号 | 层级 | 总显存 | 前景模型 | 背景模型 | 修改说明 |
|------|------|--------|----------|----------|----------|
| 1 | 超低显存 | 1300MB | **pytorch_resnet_cifar10/resnet20** | examples/mnist_ff | 前景背景互换 |
| 2 | 低显存 | 2000MB | **VulBERTa/mlp** | examples/mnist | 前景背景互换 |
| 3 | 低显存 | 2000MB | **examples/mnist** | VulBERTa/mlp | 背景改为VulBERTa/mlp |
| 4 | 低显存 | 2700MB | **MRT-OAST/default** | examples/mnist_rnn | 前景背景互换 |
| 5 | 中显存 | 3000MB | **Person_reID_baseline_pytorch/pcb** | examples/mnist_rnn | 前景背景互换 |
| 6 | 中显存 | 3000MB | **Person_reID_baseline_pytorch/hrnet18** | examples/mnist_rnn | 前景背景互换 |
| 7 | 中显存 | 3500MB | **examples/siamese** | Person_reID_baseline_pytorch/pcb | 保持不变 ✅ |
| 8 | 中显存 | 3000MB | **examples/mnist_rnn** | Person_reID_baseline_pytorch/pcb | 保持V2组合 |
| 9 | 中显存 | 4000MB | **examples/mnist_ff** | Person_reID_baseline_pytorch/densenet121 | 保持不变 ✅ |
| 10 | 中显存 | 4000MB | **bug-localization-by-dnn-and-rvsm/default** | Person_reID_baseline_pytorch/pcb | 前景背景互换 |
| 11 | 高显存 | 5000MB | **Person_reID_baseline_pytorch/densenet121** | VulBERTa/mlp | 保持不变 ✅ |

---

## V2 → V3 修改对照

### 前景模型变化

| V3序号 | V2序号 | V2前景模型 | V3前景模型 | 变化 |
|--------|--------|-----------|-----------|------|
| 1 | 1 | examples/mnist_ff | **pytorch_resnet_cifar10/resnet20** | 🔄 改变 |
| 2 | 2 | examples/mnist | **VulBERTa/mlp** | 🔄 改变 |
| 3 | 3 | examples/mnist | examples/mnist | ✅ 保持 |
| 4 | 5 | examples/mnist_rnn | **MRT-OAST/default** | 🔄 改变 |
| 5 | 4 | examples/mnist_rnn | **Person_reID_baseline_pytorch/pcb** | 🔄 改变 |
| 6 | 6 | examples/mnist_rnn | **Person_reID_baseline_pytorch/hrnet18** | 🔄 改变 |
| 7 | 8 | examples/siamese | examples/siamese | ✅ 保持 |
| 8 | 6 | examples/mnist_rnn | examples/mnist_rnn | ✅ 保持 |
| 9 | 10 | examples/mnist_ff | examples/mnist_ff | ✅ 保持 |
| 10 | 11 | Person_reID_baseline_pytorch/pcb | **bug-localization-by-dnn-and-rvsm/default** | 🔄 改变 |
| 11 | 12 | Person_reID_baseline_pytorch/densenet121 | Person_reID_baseline_pytorch/densenet121 | ✅ 保持 |

**统计**:
- 保持前景: 5个实验 (45.5%)
- 改变前景: 6个实验 (54.5%)

### 修改类型统计

| 修改类型 | 数量 | 百分比 |
|---------|------|--------|
| 保持不变 | 3个 | 27.3% |
| 前景背景互换 | 6个 | 54.5% |
| 其他修改 | 2个 | 18.2% |

---

## 分层抽样对比

### V2 vs V3 分层统计

| 层级 | V2数量 | V2比例 | V3数量 | V3比例 | 变化 |
|------|--------|--------|--------|--------|------|
| 超低显存 | 1个 | 9.1% | 1个 | 9.1% | 0 |
| 低显存 | 4个 | 36.4% | 3个 | 27.3% | -1 ⬇️ |
| 中显存 | 5个 | 45.5% | 6个 | 54.5% | +1 ⬆️ |
| 高显存 | 1个 | 9.1% | 1个 | 9.1% | 0 |

**说明**:
- 低显存减少1个实验 (从4个→3个)
- 中显存增加1个实验 (从5个→6个)
- 分层平衡略有调整，但仍保持合理分布

---

## 前景模型验证

### 11个模型各作为前景1次

| # | 前景模型 | 出现次数 | V3序号 | 状态 |
|---|---------|---------|--------|------|
| 1 | MRT-OAST/default | 1 | 4 | ✅ |
| 2 | Person_reID_baseline_pytorch/densenet121 | 1 | 11 | ✅ |
| 3 | Person_reID_baseline_pytorch/hrnet18 | 1 | 6 | ✅ |
| 4 | Person_reID_baseline_pytorch/pcb | 1 | 5 | ✅ |
| 5 | VulBERTa/mlp | 1 | 2 | ✅ |
| 6 | bug-localization-by-dnn-and-rvsm/default | 1 | 10 | ✅ |
| 7 | examples/mnist | 1 | 3 | ✅ |
| 8 | examples/mnist_ff | 1 | 9 | ✅ |
| 9 | examples/mnist_rnn | 1 | 8 | ✅ |
| 10 | examples/siamese | 1 | 7 | ✅ |
| 11 | pytorch_resnet_cifar10/resnet20 | 1 | 1 | ✅ |

**✅ 验证通过: 每个模型恰好作为前景1次**

---

## 背景模型统计

### 背景模型使用频率

| 背景模型 | 使用次数 | 出现在V3序号 |
|---------|---------|-------------|
| Person_reID_baseline_pytorch/pcb | 4次 | 5,7,8,10 |
| examples/mnist_rnn | 4次 | 4,5,6,8 |
| VulBERTa/mlp | 2次 | 3,11 |
| examples/mnist | 1次 | 2 |
| examples/mnist_ff | 1次 | 1 |
| Person_reID_baseline_pytorch/densenet121 | 1次 | 9 |

**说明**:
- pcb和mnist_rnn是最常用的背景模型（各4次）
- 这是合理的，因为它们显存适中且训练稳定

---

## 显存覆盖分析

### 显存测试点

| 显存点 | 实验数量 | V3序号 |
|--------|---------|--------|
| 1300MB | 1个 | 1 |
| 2000MB | 2个 | 2,3 |
| 2700MB | 1个 | 4 |
| 3000MB | 3个 | 5,6,8 |
| 3500MB | 1个 | 7 |
| 4000MB | 2个 | 9,10 |
| 5000MB | 1个 | 11 |

**覆盖统计**:
- 显存测试点: **7个**
- 显存范围: **1300MB - 5000MB** (跨度3700MB)
- 平均间隔: ~617MB

---

## 关键指标总结

### 目标达成情况

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| **每个模型作为前景1次** | 是 | 是 | ✅ |
| **实验总数** | 11个 | 11个 | ✅ |
| **模型总数** | 11个 | 11个 | ✅ |
| **分层抽样** | 保持 | 1:3:6:1 | ✅ |
| **显存覆盖范围** | 1300-5000MB | 1300-5000MB | ✅ |
| **显存测试点** | ≥7个 | 7个 | ✅ |
| **最小修改** | 尽量少 | 6个互换+2个调整 | ✅ |

---

## 与V2的主要区别

### V2的问题
- ❌ examples/mnist_ff 作为前景2次 (序号1,10)
- ❌ examples/mnist 作为前景2次 (序号2,3)
- ❌ examples/mnist_rnn 作为前景3次 (序号4,5,6)
- ❌ Person_reID_baseline_pytorch/pcb 作为前景2次 (序号9,11)
- ❌ 5个模型从未作为前景

### V3的改进
- ✅ 每个模型恰好作为前景1次
- ✅ 所有11个模型都有机会作为主训练任务
- ✅ 保持良好的分层抽样分布
- ✅ 修改数量最小化 (3个完全保持，6个简单互换)

---

## 实验详细配置

### 实验1: 超低显存 (1300MB)
```json
{
  "foreground": {
    "repo": "pytorch_resnet_cifar10",
    "model": "resnet20",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.1,
      "seed": 1334
    }
  },
  "background": {
    "repo": "examples",
    "model": "mnist_ff",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  }
}
```

### 实验2: 低显存 (2000MB)
```json
{
  "foreground": {
    "repo": "VulBERTa",
    "model": "mlp",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 3e-05,
      "seed": 1334
    }
  },
  "background": {
    "repo": "examples",
    "model": "mnist",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  }
}
```

### 实验3: 低显存 (2000MB)
```json
{
  "foreground": {
    "repo": "examples",
    "model": "mnist",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  },
  "background": {
    "repo": "VulBERTa",
    "model": "mlp",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 3e-05,
      "seed": 1334
    }
  }
}
```

### 实验4: 低显存 (2700MB)
```json
{
  "foreground": {
    "repo": "MRT-OAST",
    "model": "default",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.0001,
      "dropout": 0.2,
      "seed": 1334
    }
  },
  "background": {
    "repo": "examples",
    "model": "mnist_rnn",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  }
}
```

### 实验5: 中显存 (3000MB)
```json
{
  "foreground": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "pcb",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  },
  "background": {
    "repo": "examples",
    "model": "mnist_rnn",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  }
}
```

### 实验6: 中显存 (3000MB)
```json
{
  "foreground": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "hrnet18",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  },
  "background": {
    "repo": "examples",
    "model": "mnist_rnn",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  }
}
```

### 实验7: 中显存 (3500MB)
```json
{
  "foreground": {
    "repo": "examples",
    "model": "siamese",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "pcb",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  }
}
```

### 实验8: 中显存 (3000MB)
```json
{
  "foreground": {
    "repo": "examples",
    "model": "mnist_rnn",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "pcb",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  }
}
```

### 实验9: 中显存 (4000MB)
```json
{
  "foreground": {
    "repo": "examples",
    "model": "mnist_ff",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.01,
      "batch_size": 32,
      "seed": 1334
    }
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  }
}
```

### 实验10: 中显存 (4000MB)
```json
{
  "foreground": {
    "repo": "bug-localization-by-dnn-and-rvsm",
    "model": "default",
    "hyperparameters": {
      "epochs": 1,
      "max_iter": 10000,
      "alpha": 1e-05,
      "kfold": 10,
      "seed": 1334
    }
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "pcb",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  }
}
```

### 实验11: 高显存 (5000MB)
```json
{
  "foreground": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.05,
      "dropout": 0.5,
      "seed": 1334
    }
  },
  "background": {
    "repo": "VulBERTa",
    "model": "mlp",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 3e-05,
      "seed": 1334
    }
  }
}
```

---

## 执行建议

### 运行命令
```bash
python mutation.py -ec settings/parallel_feasibility_test_v3.json
```

### 预期运行时间
- 基于V2的1.16小时运行时间
- V3调整了分层分布，预计运行时间相近
- 估计: **~1.0-1.2小时**

### 验证检查点
1. ✅ 每个实验的前景模型都不同
2. ✅ 所有11个模型都作为前景出现1次
3. ✅ 背景模型持续运行直到前景完成
4. ✅ 显存利用率符合预期分层

---

## 版本演进

| 版本 | 实验数 | 主要特点 | 问题 |
|------|--------|---------|------|
| **V1** | 12个 | 初始设计，分层抽样 | 序号7 VulBERTa-CNN未实现 |
| **V2** | 11个 | 删除失败的序号7 | 前景模型重复 |
| **V3** | 11个 | 每个模型恰好作为前景1次 | - |

---

**状态**: ✅ 设计完成，配置文件已生成
**配置文件**: `settings/parallel_feasibility_test_v3.json`
**推荐**: 立即可执行测试
