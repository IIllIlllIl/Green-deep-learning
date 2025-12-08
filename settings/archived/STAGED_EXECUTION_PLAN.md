# 分阶段实验执行计划

## 概述
为了便于管理和监控,将所有剩余实验分成7个阶段执行,每个阶段运行时间控制在20-46小时之间(约1-2天)。

## 当前状态
- **已完成实验**: 319个 (214非并行 + 105并行)
- **待完成实验**: 约270个 (44非并行 + 225并行)
- **总预计时间**: 约235小时 (约10天)

## 7个阶段详细说明

### 阶段1: 非并行模式补全
**配置文件**: `settings/stage1_nonparallel_completion.json`
**预计时间**: 30小时 (约1.25天)
**实验数量**: 44个

#### 实验内容
| 模型 | 参数 | 需要值数 | 实验数 | 预计时间 |
|------|------|---------|--------|---------|
| MRT-OAST | dropout | 1 | 1 | 20分钟 |
| Person_reID_hrnet18 | learning_rate, dropout, seed | 3 | 3 | 3小时 |
| Person_reID_pcb | epochs, learning_rate, dropout, seed | 20 | 20 | 23小时 |
| mnist_ff | batch_size, epochs, learning_rate, seed | 20 | 20 | 3.3小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json
```

#### 预期结果
- 非并行模式从73.3%完成度提升到100%
- 所有模型的非并行实验达到5个唯一值目标

---

### 阶段2: 快速模型并行实验
**配置文件**: `settings/stage2_fast_models_parallel.json`
**预计时间**: 20小时 (约0.83天)
**实验数量**: 80个 (4个模型 × 4个参数 × 5次)

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| examples/mnist | 4 | 5 | 20 | 5小时 |
| examples/mnist_rnn | 4 | 5 | 20 | 5小时 |
| examples/siamese | 4 | 5 | 20 | 5小时 |
| examples/mnist_ff | 4 | 5 | 20 | 5小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage2_fast_models_parallel.json
```

#### 预期结果
- 4个快速模型的并行模式从0%完成到100%
- 每个模型4个参数都达到5个唯一值

---

### 阶段3: 中速模型并行实验
**配置文件**: `settings/stage3_medium_models_parallel.json`
**预计时间**: 46小时 (约1.92天)
**实验数量**: 65个

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| MRT-OAST | 5 | 5 | 25 | 12.5小时 |
| bug-localization | 4 | 5 | 20 | 17小时 |
| pytorch_resnet_cifar10 | 4 | 5 | 20 | 17小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage3_medium_models_parallel.json
```

#### 预期结果
- MRT-OAST从20%完成到100%
- bug-localization和pytorch_resnet_cifar10从0%完成到100%

---

### 阶段4: VulBERTa并行实验
**配置文件**: `settings/stage4_vulberta_parallel.json`
**预计时间**: 40小时 (约1.67天)
**实验数量**: 20个

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| VulBERTa/mlp | 4 | 5 | 20 | 40小时 |

每个实验约2小时(包含能量监控和重试机制)。

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage4_vulberta_parallel.json
```

#### 预期结果
- VulBERTa的并行模式从0%完成到100%

---

### 阶段5: Person_reID_densenet121并行实验
**配置文件**: `settings/stage5_densenet121_parallel.json`
**预计时间**: 40小时 (约1.67天)
**实验数量**: 20个

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| Person_reID_densenet121 | 4 | 5 | 20 | 40小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage5_densenet121_parallel.json
```

#### 预期结果
- Person_reID_densenet121的并行模式从0%完成到100%

---

### 阶段6: Person_reID_hrnet18并行实验
**配置文件**: `settings/stage6_hrnet18_parallel.json`
**预计时间**: 40小时 (约1.67天)
**实验数量**: 20个

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| Person_reID_hrnet18 | 4 | 5 | 20 | 40小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage6_hrnet18_parallel.json
```

#### 预期结果
- Person_reID_hrnet18的并行模式从0%完成到100%
- 该模型的非并行和并行都达到100%完成度

---

### 阶段7: Person_reID_pcb并行实验
**配置文件**: `settings/stage7_pcb_parallel.json`
**预计时间**: 40小时 (约1.67天)
**实验数量**: 20个

#### 实验内容
| 模型 | 参数数 | 每参数需要值数 | 总实验数 | 预计时间 |
|------|--------|--------------|---------|---------|
| Person_reID_pcb | 4 | 5 | 20 | 40小时 |

#### 执行命令
```bash
sudo -E python3 mutation.py -ec settings/stage7_pcb_parallel.json
```

#### 预期结果
- Person_reID_pcb的并行模式从0%完成到100%
- 该模型的非并行和并行都达到100%完成度
- **所有11个模型全部完成!**

---

## 执行时间表

| 阶段 | 内容 | 预计时间 | 累计时间 | 实验数 |
|-----|------|---------|---------|--------|
| 1 | 非并行补全 | 30小时 | 30小时 | 44 |
| 2 | 快速模型并行 | 20小时 | 50小时 | 80 |
| 3 | 中速模型并行 | 46小时 | 96小时 | 65 |
| 4 | VulBERTa并行 | 40小时 | 136小时 | 20 |
| 5 | densenet121并行 | 40小时 | 176小时 | 20 |
| 6 | hrnet18并行 | 40小时 | 216小时 | 20 |
| 7 | pcb并行 | 40小时 | 256小时 | 20 |
| **合计** | **7个阶段** | **256小时** | **(约10.7天)** | **269** |

## 执行策略建议

### 推荐方案: 顺序执行
按照阶段1→2→3→4→5→6→7的顺序依次执行。

**优点**:
1. 先完成短时间任务,快速看到进展
2. 非并行模式优先完成,便于中期验证
3. 逐步过渡到长时间任务,降低风险
4. 每个阶段结束后可以验证进度

**执行步骤**:
```bash
# 阶段1 (30小时)
sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json

# 等待完成后,检查结果
wc -l results/summary_all.csv  # 应该增加约44行

# 阶段2 (20小时)
sudo -E python3 mutation.py -ec settings/stage2_fast_models_parallel.json

# 等待完成后,检查结果
wc -l results/summary_all.csv  # 应该增加约80行

# 以此类推...
```

### 替代方案: 并行执行(不推荐)
理论上可以同时运行多个阶段,但**不推荐**,原因:
- GPU资源竞争会导致实验结果不准确
- 能量监控数据会混淆
- 去重机制可能产生冲突

## 阶段间的验证

每个阶段完成后,建议执行以下验证:

### 1. 检查实验数量
```bash
# 查看总实验数
wc -l results/summary_all.csv

# 查看最新session的实验数
ls -lh results/run_*/summary.csv
```

### 2. 检查数据完整性
```bash
# 验证CSV格式正确
python3 -c "
import csv
with open('results/summary_all.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Total experiments: {len(rows)}')
    print(f'Columns: {len(reader.fieldnames)}')
"
```

### 3. 检查去重效果
观察运行日志中的去重统计:
```
使用去重机制: 启用
Historical mutations loaded: XXX
Deduplication set size: XXX
```

### 4. 阶段性完成度分析
可以创建一个简单的分析脚本,检查当前完成度:
```python
import pandas as pd

df = pd.read_csv('results/summary_all.csv')

# 分离并行和非并行
non_parallel = df[~df['experiment_id'].str.contains('_parallel')]
parallel = df[df['experiment_id'].str.contains('_parallel')]

print(f"非并行实验数: {len(non_parallel)}")
print(f"并行实验数: {len(parallel)}")

# 按模型统计
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    print(f"{model}: {len(model_df)}个实验")
```

## 注意事项

### 1. 去重机制
- 所有配置都启用了 `use_deduplication: true`
- 每个阶段都引用 `results/summary_all.csv` 作为历史数据
- 系统会自动跳过已存在的实验,生成新的随机值

### 2. 失败处理
- 每个实验有 `max_retries: 2`,失败后自动重试2次
- 如果某个实验彻底失败,系统会继续下一个实验
- 可以通过日志查看失败原因

### 3. 数据安全
- 修复后的 `_append_to_summary_all()` 方法使用追加模式
- 每次实验完成后立即追加到 `summary_all.csv`
- 不会覆盖已有数据

### 4. 中断恢复
如果某个阶段中断:
1. 检查 `summary_all.csv` 的当前行数
2. 重新运行相同的配置文件
3. 去重机制会自动跳过已完成的实验
4. 只会运行缺失的实验

### 5. 系统资源
- 确保GPU可用且无其他训练任务
- 确保磁盘空间充足(每个阶段约10-50GB)
- 确保系统稳定,避免意外重启

## 完成后的最终验证

当所有7个阶段都完成后,执行完整验证:

```bash
# 1. 检查总实验数
wc -l results/summary_all.csv
# 预期: 319(初始) + 269(新增) = 588行(+1行header)

# 2. 验证每个模型的参数唯一值数量
python3 scripts/analyze_completion.py results/summary_all.csv

# 预期输出:
# 非并行完成度: 45/45 (100%)
# 并行完成度: 45/45 (100%)
# 整体完成度: 45/45 (100%)
# 完全达标模型: 11/11 (100%)
```

## 时间优化建议

如果需要加快进度:

1. **增加 `runs_per_config`**:
   - 将配置文件中的 `runs_per_config: 1` 改为 `2` 或 `3`
   - 这样每个参数配置会运行多次,更快达到5个唯一值
   - 但可能增加重复实验的概率

2. **减少重试次数**:
   - 将 `max_retries: 2` 改为 `1` 或 `0`
   - 减少失败实验的重试时间
   - 但可能降低成功率

3. **调整sleep时间**:
   - 在 `mutation/runner.py` 中减少 `RUN_SLEEP_SECONDS` 和 `CONFIG_SLEEP_SECONDS`
   - 但可能影响能量测量的准确性

## 开始执行

所有配置文件已准备完毕,可以开始执行阶段1:

```bash
cd /home/green/energy_dl/nightly
sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json
```

预计30小时后完成,届时会看到:
```
ALL EXPERIMENTS COMPLETED!
Results successfully appended to results/summary_all.csv
```

然后可以继续执行阶段2。

---

**祝实验顺利! 🚀**
