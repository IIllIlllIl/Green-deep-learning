# Stage配置文件修复报告

**修复日期**: 2025-12-03
**问题**: 所有stage2-7配置文件的`runs_per_config`设置为1，导致无法生成足够的唯一值

---

## 📋 修复内容

### 1. 检查结果

| 文件 | 原始值 | 修复后 | 状态 |
|------|--------|--------|------|
| stage1_nonparallel_completion.json | 1 | 1 | ⚠️ 已执行(保持) |
| stage1_supplement.json | 2 | 2 | ✓ 已正确设置 |
| stage2_fast_models_parallel.json | 1 | 1 | ⚠️ 旧文件(已替换) |
| **stage2_nonparallel_supplement_and_fast_parallel.json** | - | **6** | ✅ **新建** |
| stage3_medium_models_parallel.json | 1 | **6** | ✅ 已修复 |
| stage4_vulberta_parallel.json | 1 | **6** | ✅ 已修复 |
| stage5_densenet121_parallel.json | 1 | **6** | ✅ 已修复 |
| stage6_hrnet18_parallel.json | 1 | **6** | ✅ 已修复 |
| stage7_pcb_parallel.json | 1 | **6** | ✅ 已修复 |

---

## 🎯 关键改进

### 1. 新建Stage2配置文件 ✨

**文件名**: `stage2_nonparallel_supplement_and_fast_parallel.json`

**内容**:
- ✅ 包含stage1未完成的8个非并行实验
- ✅ 包含原有的16个并行实验配置（4个模型 × 4个参数）
- ✅ 设置 `runs_per_config: 6`
- ✅ 总计24个实验配置

**Stage1未完成部分（8个配置）**:
1. hrnet18 - learning_rate (4→5个唯一值)
2. hrnet18 - dropout (4→5个唯一值)
3. hrnet18 - seed (4→5个唯一值)
4. pcb - learning_rate (4→5个唯一值)
5. pcb - seed (4→5个唯一值)
6. mnist_ff - batch_size (3→5个唯一值，需要2个)
7. mnist_ff - learning_rate (4→5个唯一值)
8. mnist_ff - seed (4→5个唯一值)

**并行实验部分（16个配置）**:
- examples/mnist: 4个参数
- examples/mnist_rnn: 4个参数
- examples/siamese: 4个参数
- examples/mnist_ff: 4个参数

### 2. 修复Stage3-7 ✅

所有文件的`runs_per_config`从**1**改为**6**。

**为什么是6而不是5？**
- 目标: 每个参数5个唯一值
- 余量: +1提供缓冲，确保达标
- 去重机制会自动跳过重复值和已达标的参数

---

## 📊 预期效果

### 旧配置 vs 新配置

| 阶段 | 旧runs_per_config | 新runs_per_config | 预期改进 |
|------|-----------------|-----------------|---------|
| Stage2 | 1 | 6 | **6倍运行次数** ✓ |
| Stage3 | 1 | 6 | **6倍运行次数** ✓ |
| Stage4 | 1 | 6 | **6倍运行次数** ✓ |
| Stage5 | 1 | 6 | **6倍运行次数** ✓ |
| Stage6 | 1 | 6 | **6倍运行次数** ✓ |
| Stage7 | 1 | 6 | **6倍运行次数** ✓ |

### 实际运行效果预测

#### 旧配置（runs_per_config=1）
```
16个配置 × 1次运行 = 16次尝试
去重后实际有效: ~12-16个实验
完成度: 20-30% ✗
```

#### 新配置（runs_per_config=6）
```
16个配置 × 6次运行 = 96次尝试
去重机制自动跳过重复和已达标
预期实际有效: ~70-80个实验
完成度: 接近100% ✓
```

---

## 🔧 技术细节

### runs_per_config的工作原理

```python
# 伪代码说明
for config in experiments:
    successful_count = 0

    while successful_count < runs_per_config:
        mutation = generate_unique_mutation(config)

        # 去重检查
        if mutation in historical_data:
            print("重复，跳过")
            continue

        # 达标检查
        if parameter_has_5_unique_values(config.mutate):
            print("已达标，停止该配置")
            break

        # 运行实验
        run_experiment(mutation)
        successful_count += 1
```

### 去重机制保障

1. **历史数据去重**: 通过`historical_csvs`加载已有实验
2. **会话内去重**: 同一运行中不会重复生成相同值
3. **达标自动停止**: 参数达到5个唯一值后自动跳过
4. **最大尝试次数**: MAX_MUTATION_ATTEMPTS = 1000

---

## 📂 文件变更总结

### 新建文件
- ✅ `settings/stage2_nonparallel_supplement_and_fast_parallel.json`

### 修改文件
- ✅ `settings/stage3_medium_models_parallel.json`
- ✅ `settings/stage4_vulberta_parallel.json`
- ✅ `settings/stage5_densenet121_parallel.json`
- ✅ `settings/stage6_hrnet18_parallel.json`
- ✅ `settings/stage7_pcb_parallel.json`

### 保留文件（不再使用）
- ⚠️ `settings/stage2_fast_models_parallel.json` (已被新配置替代)

---

## 🚀 下一步执行计划

### 更新后的执行顺序

```bash
# 阶段2（新）: 非并行补充 + 快速模型并行 (预计23小时)
sudo -E python3 mutation.py -ec settings/stage2_nonparallel_supplement_and_fast_parallel.json

# 阶段3: 中速模型并行 (预计46小时)
sudo -E python3 mutation.py -ec settings/stage3_medium_models_parallel.json

# 阶段4: VulBERTa并行 (预计40小时)
sudo -E python3 mutation.py -ec settings/stage4_vulberta_parallel.json

# 阶段5: densenet121并行 (预计40小时)
sudo -E python3 mutation.py -ec settings/stage5_densenet121_parallel.json

# 阶段6: hrnet18并行 (预计40小时)
sudo -E python3 mutation.py -ec settings/stage6_hrnet18_parallel.json

# 阶段7: pcb并行 (预计40小时)
sudo -E python3 mutation.py -ec settings/stage7_pcb_parallel.json
```

### 预期完成度

| 阶段完成 | 非并行完成度 | 并行完成度 | 整体完成度 |
|---------|------------|----------|----------|
| Stage1完成 | 80.0% | 0% | 0% |
| **Stage2完成** | **100%** ✓ | **17.8%** | **17.8%** |
| Stage3完成 | 100% ✓ | 42.2% | 42.2% |
| Stage4完成 | 100% ✓ | 51.1% | 51.1% |
| Stage5完成 | 100% ✓ | 60.0% | 60.0% |
| Stage6完成 | 100% ✓ | 68.9% | 68.9% |
| Stage7完成 | 100% ✓ | **100%** ✓ | **100%** ✓ |

---

## ⚠️ 注意事项

### 1. Stage1 Supplement已过时
由于已将未完成部分合并到Stage2，`stage1_supplement.json`可以归档或删除。

### 2. 旧的Stage2配置已被替代
`stage2_fast_models_parallel.json`已被`stage2_nonparallel_supplement_and_fast_parallel.json`替代。

### 3. 运行时间可能更长
由于`runs_per_config`增加到6，每个阶段的实际运行时间可能比原估计略长，但去重机制会跳过已达标的参数，实际影响有限。

### 4. 监控建议
每个阶段完成后，检查：
```bash
# 查看总实验数
wc -l results/summary_all.csv

# 查看最新session
ls -lht results/run_* | head -1

# 检查是否有警告信息
grep -i "warning" results/run_*/logs/*.log
```

---

## ✅ 验证清单

- [x] 检查所有stage2-7的runs_per_config值
- [x] 修复stage3-7为runs_per_config=6
- [x] 创建新的stage2配置文件
- [x] 合并stage1未完成的8个非并行实验到stage2
- [x] 保留stage2原有的16个并行实验配置
- [x] 验证所有修改后的配置文件
- [x] 创建修复报告文档

---

## 📊 修复前后对比

### 问题配置（修复前）
```json
{
  "runs_per_config": 1,  // ❌ 只运行1次
  "experiments": [
    {
      "mutate": ["learning_rate"],
      "comment": "需要5个唯一值"  // ❌ 与配置不符
    }
  ]
}
```

### 正确配置（修复后）
```json
{
  "runs_per_config": 6,  // ✅ 运���6次（5个+余量）
  "use_deduplication": true,  // ✅ 启用去重
  "historical_csvs": ["results/summary_all.csv"],  // ✅ 加载历史
  "experiments": [
    {
      "mutate": ["learning_rate"],
      "comment": "需要5个唯一值"  // ✅ 与配置一致
    }
  ]
}
```

---

## 🎉 修复完成

所有stage配置文件已修复，现在可以开始执行更新后的stage2配置：

```bash
sudo -E python3 mutation.py -ec settings/stage2_nonparallel_supplement_and_fast_parallel.json
```

**预期结果**:
- 补充stage1未完成的10个实验
- 完成快速模型的并行实验
- 非并行模式达到100%
- 并行模式达到17.8%

---

**修复者**: Green (Claude Code)
**修复状态**: ✅ 完成
**影响范围**: Stage2-7所有配置文件
