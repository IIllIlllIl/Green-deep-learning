# 问题修复与测试指南

**最后更新**: 2025-11-09
**状态**: ✅ 所有已知问题已修复

## 概述

本文档记录了在sudo环境下运行训练脚本时发现的问题及其修复方案。所有修复已完成并准备测试。

---

## 修复清单

### ✅ 1. bug-localization - 参数名称不匹配

**文件**: `config/models_config.json` (行48-85)

**问题**: 配置文件中的参数名称与训练脚本不匹配
- 配置使用: `--epochs`, `--lr`
- 脚本实际: `--max_iter`, `--alpha`, `--kfold`

**修复**:
```json
{
  "supported_hyperparams": {
    "max_iter": {"flag": "--max_iter", "default": 10000, "range": [1000, 20000]},
    "kfold": {"flag": "--kfold", "default": 10},
    "alpha": {"flag": "--alpha", "default": 0.00001}
  }
}
```

**测试状态**: ✅ 已验证成功 (2025-11-08)

---

### ✅ 2. Person_reID - $HOME变量问题

**文件**: `repos/Person_reID_baseline_pytorch/train.sh` (行70)

**问题**: sudo环境下$HOME变量指向/root而非/home/green

**修复前**:
```bash
PYTHON_PATH="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
```

**修复后**:
```bash
PYTHON_PATH="/home/green/miniconda3/envs/$CONDA_ENV/bin/python"
```

**原因**: sudo改变环境变量，需要使用绝对路径

---

### ✅ 3. Person_reID - StepLR除零错误

**文件**: `repos/Person_reID_baseline_pytorch/train.py` (行628)

**问题**: 当`total_epoch < 3`时，学习率调度器的`step_size`计算结果为0

**错误信息**:
```
ZeroDivisionError: integer division or modulo by zero
  File ".../torch/optim/lr_scheduler.py", line 538, in get_lr
    if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
```

**修复前**:
```python
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft,
    step_size=opt.total_epoch*2//3,  # epochs=1时为0
    gamma=0.1
)
```

**修复后**:
```python
# Fix: Ensure step_size is at least 1 to avoid ZeroDivisionError
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft,
    step_size=max(1, opt.total_epoch*2//3),
    gamma=0.1
)
```

**测试状态**: ⏳ 待验证

---

### ✅ 4. MRT-OAST - conda激活方式

**文件**: `repos/MRT-OAST/train.sh` (行346)

**问题**: sudo非交互式shell下`conda activate`不工作

**修复前**:
```bash
eval "$(conda shell.bash hook)"
conda activate mrt-oast
```

**修复后**:
```bash
source /home/green/miniconda3/bin/activate mrt-oast
```

**原因**: conda hook初始化在sudo下挂起，直接source环境激活脚本更可靠

---

### ✅ 5. MRT-OAST - conda检查命令

**文件**: `repos/MRT-OAST/train.sh` (行304-307)

**问题**: conda命令在sudo环境下不在PATH中

**错误信息**:
```
./train.sh: line 304: conda: command not found
```

**修复**:
```bash
# 检查conda环境（在sudo下跳过，因为conda命令不可用）
# if ! conda info --envs | grep -q "mrt-oast"; then
#     print_error "conda环境 'mrt-oast' 不存在..."
#     exit 1
# fi
```

**原因**:
- sudo下`/home/green/miniconda3/bin`不在PATH中
- conda命令找不到，但环境实际存在且可通过绝对路径激活
- 检查是冗余的，激活失败时会自然报错

**测试状态**: ⏳ 待验证

---

## 测试配置文件

### 快速验证测试

**文件**: `settings/failed_models_quick_test.json`

**用途**: 快速验证3个之前失败的模型（10-15分钟）

**配置**:
```json
{
  "experiment_name": "failed_models_quick_test",
  "experiments": [
    {"repo": "bug-localization-by-dnn-and-rvsm", "hyperparameters": {"max_iter": 1000, "kfold": 2}},
    {"repo": "Person_reID_baseline_pytorch", "model": "densenet121", "hyperparameters": {"epochs": 1}},
    {"repo": "MRT-OAST", "hyperparameters": {"epochs": 1}}
  ]
}
```

**运行命令**:
```bash
cd /home/green/energy_dl/nightly
sudo python3 mutation.py --experiment-config settings/failed_models_quick_test.json
```

**预期结果**: 3/3 成功

---

### 完整测试

**文件**: `settings/full_test_run.json`

**用途**: 测试全部6个模型（数小时）

**包含模型**:
1. examples / mnist_cnn (10 epochs)
2. MRT-OAST / default (10 epochs)
3. bug-localization-by-dnn-and-rvsm / default (10000 iterations)
4. VulBERTa / mlp (10 epochs)
5. Person_reID_baseline_pytorch / densenet121 (60 epochs)
6. pytorch_resnet_cifar10 / resnet20 (200 epochs)

**运行命令**:
```bash
cd /home/green/energy_dl/nightly
sudo python3 mutation.py --experiment-config settings/full_test_run.json
```

**预期结果**: 6/6 成功

---

## 修复历史

### 第一轮修复 (2025-11-08)

**修复**:
- ✅ bug-localization参数映射
- ✅ Person_reID $HOME路径
- ✅ MRT-OAST conda激活

**测试结果**: 1/3 成功 (bug-localization)

**剩余问题**:
- ❌ Person_reID: Python Traceback错误
- ❌ MRT-OAST: conda命令找不到

---

### 第二轮修复 (2025-11-09)

**修复**:
- ✅ Person_reID StepLR除零错误
- ✅ MRT-OAST conda检查注释

**预期结果**: 3/3 成功

---

## 检查修复状态

### 查看最近测试结果

```bash
# 查看最新的3个结果文件
ls -lht /home/green/energy_dl/nightly/results/*.json | head -3

# 检查成功率
for f in $(ls -t /home/green/energy_dl/nightly/results/202511*.json | head -6); do
  echo "$(basename $f): $(jq -r 'if .training_success then "✓" else "✗" end' $f)"
done
```

### 查看训练日志

```bash
# 查看最新的训练日志
ls -lht /home/green/energy_dl/nightly/results/training_*.log | head -3

# 检查特定模型的错误
grep -i "error\|traceback\|failed" results/training_Person*.log | tail -20
```

---

## 已知限制

### 1. 硬编码路径

**位置**:
- `repos/Person_reID_baseline_pytorch/train.sh:70`
- `repos/MRT-OAST/train.sh:346`

**问题**: 使用硬编码的`/home/green`路径

**影响**: 如果在其他用户环境下运行需要修改

**解决方案**: 可以使用`$(eval echo ~green)`替代硬编码

---

### 2. Sudo环境要求

**原因**: perf需要root权限监控CPU能耗

**影响**:
- $HOME变量改变
- PATH不包含用户conda路径
- 环境变量需要特殊处理

**解决方案**:
- 使用绝对路径
- 避免依赖环境变量
- 使用`source`而非`conda activate`

---

### 3. 数据集要求

**Person_reID需要Market-1501数据集**

位置: `repos/Person_reID_baseline_pytorch/Market/Market-1501-v15.09.15/pytorch`

**如果缺失**: 训练会失败并报告数据集不存在

---

## 故障排查

### 问题: Training failed: Log file too small

**可能原因**:
- 训练脚本立即退出
- conda环境激活失败
- Python路径错误

**检查方法**:
```bash
# 查看训练日志
cat results/training_REPO_MODEL_TIMESTAMP.log

# 手动测试训练脚本
cd repos/REPO_NAME
sudo bash train.sh [参数]
```

---

### 问题: Error pattern found: Traceback

**可能原因**:
- Python运行时错误
- 依赖包缺失
- 参数错误

**检查方法**:
```bash
# 查找Traceback详情
grep -A 10 "Traceback" results/training_*.log
```

---

### 问题: conda: command not found

**解决方案**: 已通过注释检查和使用绝对路径激活解决

**验证**:
```bash
# 测试conda环境激活
sudo bash -c "source /home/green/miniconda3/bin/activate ENV_NAME && python --version"
```

---

## 文件修改记录

| 文件 | 行号 | 修改内容 | 状态 |
|------|------|---------|------|
| `config/models_config.json` | 48-85 | bug-localization参数映射 | ✅ 已验证 |
| `repos/Person_reID_baseline_pytorch/train.sh` | 70 | $HOME硬编码 | ✅ 已修复 |
| `repos/Person_reID_baseline_pytorch/train.py` | 628 | StepLR step_size修复 | ⏳ 待验证 |
| `repos/MRT-OAST/train.sh` | 304-307 | 注释conda检查 | ⏳ 待验证 |
| `repos/MRT-OAST/train.sh` | 346 | conda激活方式 | ✅ 已修复 |
| `settings/full_test_run.json` | 38-42 | bug-localization参数更新 | ✅ 已更新 |
| `settings/failed_models_quick_test.json` | - | 创建快速测试配置 | ✅ 已创建 |

---

## 下一步

1. **立即执行**: 运行快速验证测试
   ```bash
   cd /home/green/energy_dl/nightly
   sudo python3 mutation.py --experiment-config settings/failed_models_quick_test.json
   ```

2. **验证成功后**: 运行完整测试
   ```bash
   sudo python3 mutation.py --experiment-config settings/full_test_run.json
   ```

3. **分析结果**: 查看能耗和性能数据
   ```bash
   # 查看结果JSON
   cat results/LATEST_RESULT.json | jq .

   # 查看能耗指标
   cat results/LATEST_RESULT.json | jq '.energy_metrics'
   ```

---

## 参考文档

- [完整测试运行指南](full_test_run_guide.md)
- [失败模型分析报告](remaining_failures_investigation.md) (已过时)
- [配置文件说明](../settings/README.md)
- [快速参考](quick_reference.md)
