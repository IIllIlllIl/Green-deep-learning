# MRT-OAST 训练脚本使用指南

本项目提供了三个Shell脚本来简化模型的训练和评估过程。

## 脚本概览

1. **train_and_evaluate.sh** - 主训练和评估脚本（功能完整，可自定义所有参数）
2. **quick_train.sh** - 快速训练脚本（预设配置，适合快速测试）
3. **evaluate_model.sh** - 模型评估脚本（评估已训练的模型）

---

## 1. 主训练和评估脚本 (train_and_evaluate.sh)

### 功能
- 完整的训练流程
- 自动评估模型性能
- 支持所有超参数的命令行配置
- 自动记录训练日志
- 生成详细的性能报告

### 基本使用

```bash
# 查看帮助信息
./train_and_evaluate.sh --help

# 使用默认参数训练GCJ数据集
./train_and_evaluate.sh

# 训练OJClone数据集（小数据集，适合快速测试）
./train_and_evaluate.sh --dataset OJClone --epochs 5

# 训练BCB数据集（大数据集）
./train_and_evaluate.sh --dataset BCB --epochs 20
```

### 常用参数

#### 数据集选项
```bash
--dataset DATASET       # 数据集名称 (OJClone|GCJ|BCB)
--ast-type TYPE         # AST类型 (AST|OAST)
```

#### 训练参数
```bash
--epochs N              # 训练轮数
--batch-size N          # 批次大小
--lr RATE               # 学习率
--dropout RATE          # Dropout率
--seed N                # 随机种子
--valid-step N          # 验证步数（0表示每epoch验证）
```

#### 模型参数
```bash
--max-len N             # 最大序列长度
--layers N              # Transformer层数
--d-model N             # 模型维度
--d-ff N                # 前馈网络维度
--heads N               # 注意力头数
--output-dim N          # 输出维度
```

#### 其他选项
```bash
--quick                 # 快速模式（减少训练量用于测试）
--no-cuda               # 不使用GPU
--tag TAG               # 模型标签
--save-dir DIR          # 模型保存目录
--log-dir DIR           # 日志保存目录
```

### 使用示例

#### 示例1：快速测试（小数据集，少量epoch）
```bash
./train_and_evaluate.sh \
    --dataset OJClone \
    --epochs 3 \
    --batch-size 32 \
    --max-len 128 \
    --quick
```

#### 示例2：完整训练（默认配置）
```bash
./train_and_evaluate.sh \
    --dataset GCJ \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.0001
```

#### 示例3：大模型训练
```bash
./train_and_evaluate.sh \
    --dataset BCB \
    --epochs 20 \
    --batch-size 32 \
    --layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --heads 16
```

#### 示例4：自定义学习率和序列长度
```bash
./train_and_evaluate.sh \
    --dataset GCJ \
    --epochs 15 \
    --lr 0.0005 \
    --max-len 512
```

#### 示例5：内存受限配置
```bash
./train_and_evaluate.sh \
    --dataset BCB \
    --batch-size 16 \
    --max-len 128 \
    --d-model 64
```

### 输出说明

训练完成后会生成以下文件：
- `model/BTransfrom_MRT_*/model.pt` - 训练好的模型
- `logs/train_*.log` - 完整的训练日志
- `logs/metrics_*.txt` - 性能指标报告
- `model/BTransfrom_MRT_*/log_*` - TensorBoard日志

---

## 2. 快速训练脚本 (quick_train.sh)

### 功能
- 预设的快速训练配置
- 适合快速测试和验证
- 自动使用较小的批次大小和序列长度

### 基本使用

```bash
# 使用默认配置（OJClone, 3 epochs）
./quick_train.sh

# 训练GCJ数据集
./quick_train.sh GCJ

# 训练BCB数据集，5个epoch
./quick_train.sh BCB 5
```

### 预设配置
- 批次大小：32（更小，更快）
- 最大序列长度：128（更短，更快）
- 训练轮数：3（默认）
- 其他参数：使用默认值

---

## 3. 模型评估脚本 (evaluate_model.sh)

### 功能
- 评估已训练的模型
- 生成详细的性能报告
- 支持快速评估和详细评估两种模式

### 基本使用

```bash
# 查看帮助信息
./evaluate_model.sh --help

# 评估模型（自动检测数据集）
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST

# 使用自定义阈值
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_OJClone_OAST --threshold 0.85

# 详细评估模式（更慢但更详细）
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_BCB_OAST --detailed
```

### 参数说明

```bash
MODEL_PATH              # 模型目录路径（必需）
--dataset DATASET       # 数据集名称（自动检测）
--ast-type TYPE         # AST类型（默认: OAST）
--max-len N             # 最大序列长度（默认: 256）
--threshold FLOAT       # 测试阈值（默认: 0.9）
--no-cuda               # 不使用GPU
--detailed              # 详细评估模式
--output FILE           # 输出报告文件路径
```

### 评估模式

#### 快速评估（默认）
- 使用 `--quick_test` 模式
- 速度快，适合快速查看性能
- 输出主要指标（准确率、F1等）

#### 详细评估
- 使用 `--detailed` 参数
- 生成更详细的评估结果
- 可能包含混淆矩阵、分类报告等

---

## 完整训练流程示例

### 流程1：从零开始训练

```bash
# 1. 快速测试（验证环境和脚本）
./quick_train.sh OJClone 2

# 2. 正式训练
./train_and_evaluate.sh \
    --dataset GCJ \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.0001

# 3. 重新评估（如需要）
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST
```

### 流程2：参数调优

```bash
# 训练多个配置进行对比
./train_and_evaluate.sh --dataset GCJ --epochs 10 --lr 0.0001 --tag exp1
./train_and_evaluate.sh --dataset GCJ --epochs 10 --lr 0.0005 --tag exp2
./train_and_evaluate.sh --dataset GCJ --epochs 15 --lr 0.0002 --tag exp3

# 评估并对比
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST_exp1
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST_exp2
./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST_exp3
```

### 流程3：多数据集训练

```bash
# 训练三个数据集
./train_and_evaluate.sh --dataset OJClone --epochs 10
./train_and_evaluate.sh --dataset GCJ --epochs 10
./train_and_evaluate.sh --dataset BCB --epochs 10

# 批量评估
for dataset in OJClone GCJ BCB; do
    ./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_${dataset}_OAST
done
```

---

## 性能监控

### 查看训练日志
```bash
# 实时查看训练日志
tail -f logs/train_GCJ_*.log

# 查看历史日志
ls -lt logs/train_*.log
cat logs/train_GCJ_20251013_143000.log
```

### 使用TensorBoard
```bash
# 启动TensorBoard
conda activate mrt-oast
tensorboard --logdir model/

# 在浏览器访问
# http://localhost:6006
```

### GPU监控
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 查看当前GPU状态
nvidia-smi
```

---

## 常见问题

### 1. GPU内存不足
```bash
# 减小批次大小
./train_and_evaluate.sh --batch-size 16

# 减小序列长度
./train_and_evaluate.sh --max-len 128

# 减小模型维度
./train_and_evaluate.sh --d-model 64 --d-ff 256
```

### 2. 训练速度慢
```bash
# 确保使用GPU
./train_and_evaluate.sh --dataset GCJ  # 不要加 --no-cuda

# 检查GPU是否被占用
nvidia-smi

# 使用快速训练模式测试
./quick_train.sh
```

### 3. 脚本权限问题
```bash
# 添加执行权限
chmod +x train_and_evaluate.sh
chmod +x quick_train.sh
chmod +x evaluate_model.sh
```

### 4. Conda环境问题
```bash
# 确保环境已创建
conda env list

# 手动激活环境测试
conda activate mrt-oast
python main_batch.py --help
```

---

## 推荐配置

### 配置1：快速测试（适合验证环境）
```bash
./quick_train.sh OJClone 2
```

### 配置2：标准训练（平衡性能和速度）
```bash
./train_and_evaluate.sh \
    --dataset GCJ \
    --epochs 10 \
    --batch-size 64 \
    --max-len 256 \
    --lr 0.0001
```

### 配置3：高性能训练（追求最佳效果）
```bash
./train_and_evaluate.sh \
    --dataset BCB \
    --epochs 20 \
    --batch-size 32 \
    --max-len 512 \
    --layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --lr 0.00005
```

### 配置4：内存受限（GPU内存不足）
```bash
./train_and_evaluate.sh \
    --dataset OJClone \
    --epochs 10 \
    --batch-size 16 \
    --max-len 128 \
    --d-model 64 \
    --d-ff 256
```

---

## 输出文件说明

### 目录结构
```
MRT-OAST/
├── model/                          # 模型保存目录
│   └── BTransfrom_MRT_*_*/
│       ├── model.pt                # 训练好的模型
│       └── log_*/                  # TensorBoard日志
├── logs/                           # 日志目录
│   ├── train_*.log                 # 训练日志
│   ├── metrics_*.txt               # 性能报告
│   └── evaluation_*.txt            # 评估报告
```

### 性能报告示例
```
======================================
MRT-OAST 模型性能报告
======================================

训练配置:
  数据集: GCJ
  AST类型: OAST
  训练轮数: 10
  批次大小: 64
  学习率: 0.0001
  序列长度: 256
  模型层数: 2

性能指标:
  准确率 (Accuracy): 0.9234
  精确率 (Precision): 0.9156
  召回率 (Recall): 0.9312
  F1分数 (F1-Score): 0.9233

模型位置: model/BTransfrom_MRT_len256_batch64_GCJ_OAST/model.pt
完成时间: 2025-10-13 14:30:00
======================================
```

---

## 总结

- 使用 `quick_train.sh` 进行快速测试
- 使用 `train_and_evaluate.sh` 进行完整训练
- 使用 `evaluate_model.sh` 重新评估已训练模型
- 所有日志和报告自动保存在 `logs/` 目录
- 使用 TensorBoard 监控训练过程
- 根据GPU内存调整批次大小和序列长度
