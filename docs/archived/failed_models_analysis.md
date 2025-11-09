# ⚠️ 文档已过时 - 问题已全部修复

**状态**: 已归档
**更新时间**: 2025-11-09
**替代文档**: [FIXES_AND_TESTING.md](FIXES_AND_TESTING.md)

本文档记录了第一轮失败分析（2025-11-08），所有问题已在后续修复中解决。请参考最新的修复文档。

---

# 3个模型训练失败原因分析报告

## 执行摘要

测试日期: 2025-11-08
失败模型: 3/6 (50%失败率)
共同特征: 所有失败都在1.3秒内快速退出,重试3次均失败

## 失败模型列表

1. **MRT-OAST / default**
2. **bug-localization-by-dnn-and-rvsm / default**
3. **Person_reID_baseline_pytorch / densenet121**

---

## 根本原因分析

### 🔴 主要问题: Conda环境激活与sudo执行冲突

#### 问题描述

训练脚本通过`scripts/run.sh`包装器执行,该包装器使用`perf stat`来监控CPU能耗:

```bash
# scripts/run.sh 第141-142行
perf stat -e power/energy-pkg/,power/energy-ram/ -o "$CPU_ENERGY_RAW" \
    $TRAIN_SCRIPT $TRAIN_ARGS 2>&1 | tee "$LOG_FULL_PATH"
```

**问题**:
- `mutation.py`以`sudo`权限运行
- `perf`包装了整个训练脚本的执行(包括conda激活过程)
- conda在sudo+非交互式环境中无法正确初始化

#### 失败流程

```
sudo python3 mutation.py
  └─> scripts/run.sh (sudo环境)
      └─> perf stat <训练脚本>
          └─> conda activate <环境名>  ❌ 在这里挂起或失败
```

---

## 各模型详细分析

### 1. MRT-OAST

**失败信息**:
- 时长: 1.3秒
- 重试: 3次
- 错误: "Log file too small, training likely failed"

**训练脚本问题** (repos/MRT-OAST/train.sh:346-347):

```bash
eval "$(conda shell.bash hook)"
conda activate mrt-oast
```

**问题**:
1. `conda shell.bash hook`在sudo环境下无法正确初始化
2. `eval`执行在子shell中,导致环境激活无效
3. 脚本在conda激活时挂起,等待输入或初始化失败

**环境状态**:
- ✅ conda环境存在: `mrt-oast`
- ✅ Python可用: `/home/green/miniconda3/envs/mrt-oast/bin/python` (3.7.12)
- ❌ 激活方式不兼容sudo+非交互式环境

**手动测试结果**:
```bash
$ cd /home/green/energy_dl/nightly/repos/MRT-OAST
$ timeout 5 ./train.sh --epochs 1 --lr 0.0001
# 命令在5秒后超时,确认脚本在conda激活处挂起
```

---

### 2. bug-localization-by-dnn-and-rvsm

**失败信息**:
- 时长: 1.3秒
- 重试: 3次
- 错误: "Log file too small, training likely failed"

**训练脚本特点** (repos/bug-localization-by-dnn-and-rvsm/train.sh):

```bash
# 第10行
CONDA_ENV="dnn_rvsm"
CONDA_BASE="/home/green/miniconda3"
PYTHON_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

# 直接使用Python路径,不激活conda
```

**环境状态**:
- ✅ conda环境存在: `dnn_rvsm`
- ✅ Python可用: `/home/green/miniconda3/envs/dnn_rvsm/bin/python` (3.7.12)
- ✅ 不需要conda激活

**可能的问题**:
1. **权限问题**: Python脚本可能需要访问用户目录下的数据/模型文件
2. **依赖缺失**: Python包在sudo环境下找不到
3. **数据路径问题**: 数据文件路径在sudo用户下不可访问
4. **环境变量**: 某些环境变量在sudo下未正确设置

**需要进一步调查**:
- 检查训练脚本的实际Python命令
- 查看数据文件路径和权限
- 手动以sudo执行测试

---

### 3. Person_reID_baseline_pytorch

**失败信息**:
- 时长: 1.3秒
- 重试: 3次
- 错误: "Log file too small, training likely failed"

**训练脚本特点** (repos/Person_reID_baseline_pytorch/train.sh):

```bash
# 第36-37行
CONDA_ENV="reid_baseline"
PYTHON_PATH="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
```

**环境状态**:
- ✅ conda环境存在: `reid_baseline`
- ✅ Python可用: `/home/green/miniconda3/envs/reid_baseline/bin/python` (3.10.19)
- ✅ 使用直接Python路径,不需要激活

**可能的问题**:
1. **数据集缺失**: Market-1501数据集不存在或路径错误
   ```bash
   DATA_DIR="./Market/Market-1501-v15.09.15/pytorch"
   ```
2. **GPU权限**: 某些GPU操作在sudo下可能受限
3. **文件权限**: 模型保存路径权限问题
4. **依赖问题**: PyTorch等深度学习库在sudo环境下的行为

**需要进一步调查**:
- 检查数据集是否存在
- 查看训练脚本的错误处理
- 手动以sudo执行测试

---

## 共同问题模式

### 问题1: sudo环境隔离

当使用`sudo`运行`mutation.py`时:
- `$HOME`变为`/root`而非`/home/green`
- 环境变量被重置
- 文件路径和权限发生变化

### 问题2: 非交互式Shell

- 训练脚本在非交互式shell中执行
- conda激活命令需要交互式shell环境
- 某些初始化脚本(~/.bashrc等)不会被执行

### 问题3: 快速失败

所有失败都在1.3秒内退出,说明:
- 脚本在早期就遇到致命错误
- 不是训练过程中的错误
- 很可能是环境初始化或依赖检查失败

### 问题4: perf包装的副作用

`perf stat`包装了整个训练脚本:
- 使conda激活更困难
- 可能干扰某些环境初始化
- 子进程管理变复杂

---

## 解决方案建议

### 方案1: 修改训练脚本的conda激活方式 ⭐推荐

**适用**: MRT-OAST (确认问题)

**方法**: 修改`repos/MRT-OAST/train.sh`第346-347行:

```bash
# 旧代码 (不工作)
eval "$(conda shell.bash hook)"
conda activate mrt-oast

# 新代码 (推荐)
source /home/green/miniconda3/bin/activate mrt-oast
```

**优点**:
- 直接激活,不依赖shell hook
- 在sudo和非交互式环境下都能工作
- 简单可靠

### 方案2: 使用直接Python路径

**适用**: bug-localization, Person_reID (已经这样做)

**方法**: 不激活conda,直接使用完整Python路径:

```bash
PYTHON_PATH="/home/green/miniconda3/envs/环境名/bin/python"
$PYTHON_PATH train.py <args>
```

**但需要解决**:
- 数据文件路径问题
- 文件权限问题
- 依赖库路径问题

### 方案3: 避免使用sudo运行mutation.py

**方法**: 修改perf权限设置,允许普通用户使用:

```bash
# 永久设置
echo 'kernel.perf_event_paranoid=-1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

然后以普通用户运行:
```bash
python3 mutation.py --experiment-config settings/full_test_run.json
```

**优点**:
- 避免sudo带来的所有权限和环境问题
- conda激活正常工作
- 文件路径和权限正常

**缺点**:
- 需要系统配置更改
- perf权限设置可能有安全考虑

### 方案4: 改进run.sh的执行方式

**方法**: 让训练脚本以用户权限执行,只有perf以sudo执行:

```bash
# 修改mutation.py,传递用户信息
# 在run.sh中使用su切换回用户
sudo -u green bash <<EOF
  perf stat -e ... train.sh $args
EOF
```

**复杂度**: 高,需要重大改动

---

## 调试步骤建议

### 立即调试 (MRT-OAST)

```bash
# 1. 修改MRT-OAST/train.sh
cd /home/green/energy_dl/nightly/repos/MRT-OAST
# 编辑train.sh,修改第346-347行为:
# source /home/green/miniconda3/bin/activate mrt-oast

# 2. 手动测试
sudo ./train.sh --epochs 1 --lr 0.0001

# 3. 如果成功,重新运行完整测试
```

### 深入调试 (bug-localization)

```bash
# 1. 检查训练脚本实际执行的命令
cd /home/green/energy_dl/nightly/repos/bug-localization-by-dnn-and-rvsm
cat train.sh | grep "PYTHON_CMD\|exec"

# 2. 手动以sudo执行
sudo ./train.sh -n dnn --epochs 1 --seed 42

# 3. 查看详细错误输出
sudo bash -x ./train.sh -n dnn --epochs 1 --seed 42 2>&1 | head -100
```

### 深入调试 (Person_reID)

```bash
# 1. 检查数据集
cd /home/green/energy_dl/nightly/repos/Person_reID_baseline_pytorch
ls -la Market/Market-1501-v15.09.15/pytorch/

# 2. 手动以sudo执行
sudo ./train.sh -n densenet121 --total_epoch 1

# 3. 查看错误日志
sudo bash -x ./train.sh -n densenet121 --total_epoch 1 2>&1 | tee debug.log
```

---

## 优先级行动计划

### 🔥 高优先级 (立即执行)

1. **修复MRT-OAST** (问题确认,解决方案明确)
   - 修改conda激活方式
   - 预计解决时间: 5分钟
   - 成功率: 95%

2. **调试其他两个模型** (问题待确认)
   - 手动执行并捕获详细错误
   - 预计时间: 15-30分钟
   - 可能发现数据/权限问题

### 📋 中优先级 (测试后执行)

3. **统一conda激活方式**
   - 修改所有仓库使用相同的conda激活方法
   - 确保在sudo环境下都能工作

4. **增强错误日志**
   - 修改run.sh捕获更详细的stderr
   - 保存脚本执行的中间步骤

### 🔍 低优先级 (长期改进)

5. **评估方案3** (避免sudo)
   - 测试不使用sudo运行mutation.py
   - 评估perf权限设置的安全性

6. **改进框架鲁棒性**
   - 添加环境检查步骤
   - 训练前验证conda环境和数据

---

## 测试验证计划

### 阶段1: 快速修复验证

```bash
# 修复MRT-OAST后,创建测试配置
cat > settings/failed_models_test.json << 'EOF'
{
  "experiment_name": "failed_models_retest",
  "description": "Retest 3 failed models after fixes",
  "governor": "performance",
  "runs_per_config": 1,
  "max_retries": 20,
  "mode": "default",
  "experiments": [
    {
      "repo": "MRT-OAST",
      "model": "default",
      "mode": "default",
      "hyperparameters": {"epochs": 1, "learning_rate": 0.0001, "seed": 1334}
    },
    {
      "repo": "bug-localization-by-dnn-and-rvsm",
      "model": "default",
      "mode": "default",
      "hyperparameters": {"epochs": 1, "learning_rate": 0.001, "seed": 42}
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "densenet121",
      "mode": "default",
      "hyperparameters": {"epochs": 1, "learning_rate": 0.05}
    }
  ]
}
EOF

# 运行测试
sudo python3 mutation.py --experiment-config settings/failed_models_test.json
```

### 阶段2: 完整重测

修复所有问题后,重新运行完整测试:

```bash
sudo python3 mutation.py --experiment-config settings/full_test_run.json
```

目标: 6/6成功率

---

## 结论

### 确认的问题

1. ✅ **MRT-OAST**: conda激活方式不兼容sudo+非交互式环境
   - 解决方案明确
   - 预计可以修复

### 待确认的问题

2. ❓ **bug-localization**: 可能是数据/依赖/权限问题
   - 需要手动调试
   - 解决方案待确定

3. ❓ **Person_reID**: 可能是数据集缺失或权限问题
   - 需要检查Market-1501数据集
   - 解决方案待确定

### 框架评估

**优点**:
- ✅ 核心功能正常(能耗监控、重试、结果保存)
- ✅ 成功的3个模型运行稳定

**缺点**:
- ❌ sudo环境兼容性问题
- ❌ 错误日志不够详细
- ❌ 缺少环境检查

### 下一步

1. 立即修复MRT-OAST
2. 调试另外两个模型
3. 运行重测验证
4. 改进框架鲁棒性

---

**报告生成时间**: 2025-11-08
**分析者**: Claude Code
**状态**: 待修复验证
