# Bug Localization Training Guide

## 概述

本项目实现了基于DNN和RVSM的Bug定位模型。训练脚本`train.sh`支持在screen会话中运行，并���供详细的训练报告。

## 快速开始

### 基本用法

```bash
# 训练RVSM模型
./train.sh -n rvsm 2>&1 | tee training_rvsm.log

# 训练DNN模型（使用默认参数）
./train.sh -n dnn 2>&1 | tee training_dnn.log

# 在screen会话中训练
screen -dmS bug_training bash -c "./train.sh -n dnn 2>&1 | tee training.log"
```

### 自定义参数

```bash
# 自定义DNN超参数
./train.sh -n dnn \
  --kfold 10 \
  --hidden_sizes 300 \
  --alpha 1e-5 \
  --max_iter 10000 \
  --n_iter_no_change 30 \
  --solver sgd

# 快速测试（较小参数）
./train.sh -n dnn --kfold 2 --hidden_sizes 100 --max_iter 1000
```

## 训练结果

### RVSM基线模型

**训练时间:** 4.92秒 (0.08分钟)

**性能指标:**
- Top-1 Accuracy: 31.3%
- Top-5 Accuracy: 57.8%
- Top-10 Accuracy: 70.4%
- Top-20 Accuracy: 80.8%

### DNN模型（10-fold交叉验证）

#### 配置1: hidden_sizes=200 (GPU优化)
**训练时间:** 729.60秒 (12.16分钟)

**超参数:**
- K-fold: 10
- Hidden layer sizes: (200,)
- Alpha (L2 penalty): 1e-05
- Max iterations: 10000
- Early stopping patience: 30
- Solver: sgd

**性能指标:**
- Top-1 Accuracy: 38.1% (+6.8% vs RVSM)
- Top-5 Accuracy: 62.9% (+5.1% vs RVSM)
- Top-10 Accuracy: 73.9% (+3.5% vs RVSM)
- Top-20 Accuracy: 82.9% (+2.1% vs RVSM)

#### 配置2: hidden_sizes=100 (快速测试，2-fold)
**训练时间:** 80.35秒 (1.34分钟)

**性能指标:**
- Top-1 Accuracy: 38.8%
- Top-5 Accuracy: 63.0%
- Top-10 Accuracy: 74.2%
- Top-20 Accuracy: 83.1%

## 性能对比

| 模型 | Top-1 | Top-5 | Top-10 | Top-20 | 训练时间 |
|------|-------|-------|--------|--------|----------|
| RVSM | 31.3% | 57.8% | 70.4%  | 80.8%  | 5秒      |
| DNN (200) | 38.1% | 62.9% | 73.9%  | 82.9%  | 12分钟   |
| 原始论文 | - | - | - | ~85% | - |
| 仓库实现 | - | - | - | ~79% | - |

## 训练报告格式

训练完成后，脚本会自动输出包含以下信息的报告：
- 模型名称和配置
- 训练开始和结束时间
- 总训练时长
- Top-k准确率（k=1,5,10,20）
- 详细��所有k值结果（k=1到20）
- 任何训练过程中的错误或警告

## GPU性能优化建议

考虑到GPU性能限制，推荐以下参数调整：

1. **减小隐藏层大小:** `--hidden_sizes 200` (默认300)
2. **减少交叉验证折数:** `--kfold 5` (默认10)
3. **减少最大迭代次数:** `--max_iter 5000` (默认10000)
4. **并行作业数:** `--n_jobs -2` (默认，使用除一个核心外的所有核心)

## 技术细节

### 环境配置
- Conda环境: `dnn_rvsm`
- Python路径: `/home/green/miniconda3/envs/dnn_rvsm/bin/python`
- 脚本自动使用conda环境的Python，无需手动激活

### 数据路径
- 特征文件: `data/features.csv`
- Bug报告: `data/Eclipse_Platform_UI.txt`

### 模型架构
- DNN: MLPRegressor (sklearn)
- 输入特征: 5维 (rVSM_similarity, collab_filter, classname_similarity, bug_recency, bug_frequency)
- 输出: 相关性得分

## 故障排查

### 常见问题

1. **ModuleNotFoundError:** 确保在正确的conda环境中运行
2. **Worker停止警告:** 正常情况，训练会继续进行
3. **训练时间过长:** 考虑减小参数或使用GPU

### 调试建议

```bash
# 检查训练进程
ps aux | grep train_wrapper

# 监控训练日志
tail -f training.log

# 在screen会话中查看
screen -r bug_training
```

## 参考

- 原始论文: "Bug Localization with Combination of Deep Learning and Information Retrieval"
- 数据集: Eclipse UI Platform
- GitHub: https://github.com/eclipse/eclipse.platform.ui
