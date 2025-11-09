# 全量测试运行文档

## 概述

本文档说明如何运行完整的模型训练测试，以验证所有仓库脚本的功能性和鲁棒性。

## 配置文件

**文件**: `settings/full_test_run.json`

**目的**: 依次完整训练每个仓库中的一个代表性模型,使用默认参数

### 选择的模型

每个仓库选择一个模型进行测试,优先选择训练时间较短的模型:

| 顺序 | 仓库 | 模型 | Epochs | 预估时间 | 选择原因 |
|------|------|------|---------|----------|----------|
| 1 | examples | mnist_cnn | 10 | ~5-10分钟 | 最快的模型,用于初步验证 |
| 2 | MRT-OAST | default | 10 | ~20-30分钟 | 轻量级模型 |
| 3 | bug-localization-by-dnn-and-rvsm | default | 10 | ~20-30分钟 | 中等复杂度 |
| 4 | VulBERTa | mlp | 10 | ~30-60分钟 | MLP比CNN快 |
| 5 | Person_reID_baseline_pytorch | densenet121 | 60 | ~2-4小时 | 较长训练时间 |
| 6 | pytorch_resnet_cifar10 | resnet20 | 200 | ~8-12小时 | ResNet家族最小模型,训练时间最长 |

**总预估时间**: 约12-18小时

### 测试顺序逻辑

1. **快速验证** (mnist_cnn): 先运行最快的模型,确保基础设施正常
2. **中等复杂度** (MRT-OAST, bug-localization, VulBERTa): 逐步增加复杂度
3. **长时间训练** (Person_reID, ResNet20): 最后运行耗时最长的模型

这种顺序可以:
- 快速发现明显问题
- 避免在开始就运行长时间任务
- 如果中途失败,至少完成了部分测试

## 运行方法

### 方法1: 使用便捷脚本(推荐)

```bash
# 在screen会话中运行
screen -r test
cd /home/green/energy_dl/nightly
sudo ./scripts/run_full_test.sh
```

脚本会:
- 检查配置文件是否存在
- 显示将要训练的模型列表
- 运行完整的训练流程
- 报告成功/失败状态

### 方法2: 直接使用mutation.py

```bash
# 在screen会话中运行
screen -r test
cd /home/green/energy_dl/nightly
sudo python3 mutation.py --experiment-config settings/full_test_run.json
```

### 为什么使用screen?

由于预期运行时间很长(12-18小时),使用screen可以:
- 允许SSH连接断开后继续运行
- 可以随时重新连接查看进度
- 避免意外中断

## 监控进度

### 查看screen会话

```bash
# 重新连接到screen会话
screen -r test

# 断开连接(保持程序运行)
# 按 Ctrl+A 然后按 D
```

### 查看运行日志

训练过程中会显示:
- 当前训练的模型
- 超参数配置
- 训练进度
- 能耗监控数据

### 检查结果文件

```bash
# 查看最新的结果
ls -lht results/*.json | head -10

# 查看特定实验的结果
cat results/<experiment_id>.json | jq '.'

# 查看成功/失败状态
cat results/<experiment_id>.json | jq '.training_success'
```

## 预期输出

每个模型训练完成后会生成:

1. **JSON结果文件** (`results/<experiment_id>.json`):
   - 实验元数据
   - 超参数配置
   - 性能指标
   - 能耗数据
   - 训练成功/失败状态

2. **能耗监控数据** (`results/energy_<experiment_id>/`):
   - CPU能耗总结
   - GPU功耗时间序列
   - GPU温度数据
   - GPU利用率数据

## 验证项目

此测试验证以下功能:

### 1. 脚本功能性
- ✓ 各仓库的train.sh脚本能正确执行
- ✓ 超参数正确传递给训练脚本
- ✓ 性能指标正确提取

### 2. 能耗监控
- ✓ CPU能耗监控正常工作
- ✓ GPU能耗监控正常工作
- ✓ 数据正确保存和解析

### 3. 鲁棒性
- ✓ 自动重试机制工作正常
- ✓ 错误处理正确
- ✓ 结果文件格式正确

### 4. Governor控制
- ✓ CPU频率调度器设置为performance模式
- ✓ 减少能耗测量干扰

## 常见问题

### 训练失败怎么办?

1. 查看错误信息:
   ```bash
   cat results/<experiment_id>.json | jq '.error_message'
   ```

2. 框架会自动重试(最多2次)

3. 如果持续失败,检查:
   - 仓库的训练脚本是否正确
   - 超参数配置是否合理
   - 数据集是否可用

### 如何暂停/恢复?

- **暂停**: 按 Ctrl+C (会中断当前训练)
- **恢复**: 重新运行命令(会从下一个模型开始)
- **注意**: 中断的训练不会保存结果

### 如何跳过某个模型?

编辑 `settings/full_test_run.json`,删除不想运行的实验条目

### 如何查看实时进度?

```bash
# 方法1: 连接到screen会话
screen -r test

# 方法2: 监控进程
watch -n 5 'ps aux | grep python3 | grep mutation'

# 方法3: 查看最新结果
watch -n 30 'ls -lt results/*.json | head -5'
```

## 结果分析

### 检查所有训练是否成功

```bash
# 统计成功的训练数量
cat results/*.json | jq -s '[.[] | select(.training_success == true)] | length'

# 列出失败的训练
cat results/*.json | jq -s '.[] | select(.training_success == false) | .experiment_id'
```

### 提取性能和能耗数据

```bash
# 生成CSV格式的汇总
cat results/*.json | jq -r '[.repository, .model, .duration_seconds, .energy_metrics.cpu_energy_total_joules, .energy_metrics.gpu_energy_total_joules, .performance_metrics] | @csv'
```

### 比对不同模型

```bash
# 比较能耗
cat results/*.json | jq -s 'sort_by(.energy_metrics.cpu_energy_total_joules) | .[] | {repo: .repository, model: .model, cpu_energy: .energy_metrics.cpu_energy_total_joules}'
```

## 下一步

测试完成后:

1. **验证结果**:
   - 检查所有6个模型是否都成功训练
   - 验证性能指标是否合理
   - 检查能耗数据是否完整

2. **问题修复**:
   - 如有失败,查看错误日志并修复
   - 更新配置或脚本

3. **准备变异实验**:
   - 测试通过后,可以开始运行超参数变异实验
   - 使用 `settings/all.json` 等配置文件

## 技术细节

### CPU Governor设置

配置文件中设置 `"governor": "performance"`,确保:
- CPU运行在最高频率
- 减少能耗测量的波动
- 提高实验可重复性

### 重试机制

配置 `"max_retries": 2`:
- 训练失败时自动重试
- 最多重试2次
- 减少临时问题导致的失败

### 防干扰休眠

每次训练之间自动休眠60秒:
- 让系统冷却
- 减少前一个训练对下一个的能耗干扰
- 确保能耗测量准确性

## 联系与支持

如遇到问题:
1. 检查本文档的"常见问题"部分
2. 查看主README.md中的故障排除章节
3. 检查test/README.md中的测试指南
