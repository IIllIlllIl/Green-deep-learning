# MRT-OAST 快速开始指南

## 一键快速测试

```bash
# 1. 激活环境
conda activate mrt-oast

# 2. 快速训练（3分钟快速测试）
./quick_train.sh OJClone 2

# 3. 查看结果
cat logs/metrics_*.txt
```

## 标准训练流程

### 1. 训练GCJ数据集（默认配置）
```bash
./train_and_evaluate.sh --dataset GCJ --epochs 10
```

### 2. 训练OJClone数据集（小数据集）
```bash
./train_and_evaluate.sh --dataset OJClone --epochs 5
```

### 3. 训练BCB数据集（大数据集）
```bash
./train_and_evaluate.sh --dataset BCB --epochs 20 --batch-size 32
```

## 自定义参数示例

### 调整学习率和批次大小
```bash
./train_and_evaluate.sh \
    --dataset GCJ \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0005
```

### 大模型配置
```bash
./train_and_evaluate.sh \
    --dataset BCB \
    --layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --heads 16 \
    --batch-size 32
```

### 内存受限配置
```bash
./train_and_evaluate.sh \
    --dataset OJClone \
    --batch-size 16 \
    --max-len 128 \
    --d-model 64
```

## 模型评估

### 评估已训练的模型
```bash
# 自动检测数据集
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST

# 使用自定义阈值
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_OJClone_OAST --threshold 0.85

# 详细评估
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_BCB_OAST --detailed
```

## 查看训练结果

### 训练日志
```bash
# 最新的训练日志
ls -lt logs/train_*.log | head -1

# 查看日志
cat logs/train_GCJ_20251013_180000.log
```

### 性能报告
```bash
# 最新的性能报告
ls -lt logs/metrics_*.txt | head -1

# 查看报告
cat logs/metrics_GCJ_20251013_180000.txt
```

### TensorBoard
```bash
# 启动TensorBoard
conda activate mrt-oast
tensorboard --logdir model/

# 浏览器访问：http://localhost:6006
```

## 帮助信息

```bash
# 查看训练脚本帮助
./train_and_evaluate.sh --help

# 查看评估脚本帮助
./evaluate_model.sh --help
```

## 常见配置组合

### 配置1：快速验证（2-5分钟）
```bash
./quick_train.sh OJClone 2
```

### 配置2：标准训练（2-4小时）
```bash
./train_and_evaluate.sh --dataset GCJ --epochs 10
```

### 配置3：高性能训练（8-16小时）
```bash
./train_and_evaluate.sh \
    --dataset BCB \
    --epochs 20 \
    --layers 4 \
    --d-model 256 \
    --batch-size 32
```

## 文档索引

- **SCRIPTS_GUIDE.md** - 详细的脚本使用指南
- **PROJECT_STATUS.md** - 项目状态和配置说明
- **SETUP_CN.md** - 环境配置说明
- **README.md** - 项目原始说明

## 监控命令

```bash
# GPU使用情况
watch -n 1 nvidia-smi

# 实时查看训练日志
tail -f logs/train_*.log

# 磁盘空间
df -h .
```

---

快速问题？直接运行：`./quick_train.sh`
