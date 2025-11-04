# Bug Localization Training Scripts

## 文件说明

本项目包含以下训练相关文件：

- `train.sh` - 主训练Shell脚本
- `train_wrapper.py` - Python训练包装器
- `docs/TRAINING_GUIDE.md` - 详细训练指南和结果分析

## 快速使用

### 1. RVSM基线模型训练（快速，5秒）

```bash
./train.sh -n rvsm 2>&1 | tee training_rvsm.log
```

### 2. DNN模型训练（约12分钟，10-fold）

```bash
./train.sh -n dnn 2>&1 | tee training_dnn.log
```

### 3. 在screen会话中运行

如果screen会话"test"已存在：
```bash
screen -S test -X stuff $'cd /path/to/bug-localization-by-dnn-and-rvsm\n'
screen -S test -X stuff $'./train.sh -n dnn 2>&1 | tee training.log\n'
```

或创建新的专用screen会话：
```bash
screen -dmS bug_training bash -c "cd $(pwd) && ./train.sh -n dnn 2>&1 | tee training.log"
screen -r bug_training  # 查看训练进度
```

## 参数说明

### 必需参数

- `-n, --model_name` - 模型类型：`dnn` 或 `rvsm`

### 可选参数（仅DNN）

- `--kfold N` - 交叉验证折数（默认: 10）
- `--hidden_sizes SIZES` - 隐藏层大小（默认: 300，GPU优化可用200）
- `--alpha ALPHA` - L2正则化参数（默认: 1e-5）
- `--max_iter N` - 最大迭代次数（默认: 10000）
- `--n_iter_no_change N` - 早停耐心值（默认: 30）
- `--solver SOLVER` - 优化器（默认: sgd，可选: adam, lbfgs）
- `--n_jobs N` - 并行作业数（默认: -2，使用所有核心除一个）

## 使用示例

### GPU性能优化配置

```bash
./train.sh -n dnn --hidden_sizes 200 --kfold 5 2>&1 | tee training_opt.log
```

### 快速测试

```bash
./train.sh -n dnn --kfold 2 --hidden_sizes 100 --max_iter 1000 2>&1 | tee training_test.log
```

### 完整10-fold训练（推荐用于最终评估）

```bash
./train.sh -n dnn --kfold 10 --hidden_sizes 300 2>&1 | tee training_full.log
```

## 训练报告示例

训练完成后会自动输出详细报告，包括：

```
================================================================================
TRAINING REPORT
================================================================================
Model: DNN
Start time: 2025-11-03 14:54:10
End time: 2025-11-03 15:06:19
Total duration: 729.60 seconds (12.16 minutes)

MODEL PERFORMANCE (Top-k Accuracy):
--------------------------------------------------------------------------------
  Top- 1 Accuracy: 0.381 (38.1%)
  Top- 5 Accuracy: 0.629 (62.9%)
  Top-10 Accuracy: 0.739 (73.9%)
  Top-20 Accuracy: 0.829 (82.9%)

Detailed Results (All k values):
  Top- 1: 0.381
  Top- 2: 0.482
  ...
  Top-20: 0.829
================================================================================
```

## 验证结果

所有模型已成功在本地和screen会话中测试：

### RVSM模型
- ✅ 训练成功
- ✅ 生成完整报告
- ⏱️ 训练时间: ~5秒
- 📊 Top-20准确率: 80.8%

### DNN模型
- ✅ 训练成功（hidden_sizes=200, 10-fold）
- ✅ 训练成功（hidden_sizes=300, 2-fold）
- ✅ 在screen会话中成功运行
- ✅ 生成完整报告
- ⏱️ 训练时间: ~12分钟（10-fold）
- 📊 Top-20准确率: 82.9%

## 技术特性

1. **自动环境管理** - 直接使用conda环境的Python，无需手动激活
2. **详细报告** - 自动记录训练时间、性能指标、错误信息
3. **并行训练** - 使用joblib并行执行K-fold交叉验证
4. **参数灵活** - 支持所有主要超参数的命令行配置
5. **Screen兼容** - 可在screen会话中稳定运行长时间训练

## 故障排查

如果遇到问题，请查看 `docs/TRAINING_GUIDE.md` 中的详细故障排查指南。

## 更多信息

详细的训练结果分析、性能对比和技术细节，请参阅：
- `docs/TRAINING_GUIDE.md` - 完整训练指南
- 原始README: `README.md`
