# 阶段2&3修改指南：添加weight_decay和precision支持

**生成日期**: 2025-11-05
**当前进度**: 阶段1完成（Seed支持100%✅），阶段2&3待完成

---

## 📊 当前状态

### ✅ 已完成（阶段1）
- pytorch_resnet_cifar10: seed支持 ✅
- Person_reID_baseline_pytorch: seed支持 ✅
- bug-localization-by-dnn-and-rvsm: seed支持 ✅
- **总计: 12个模型，seed支持率100%**

### ⏳ 待完成（阶段2）
需要添加weight_decay支持的7个模型：
1. MRT-OAST
2. VulBERTa-MLP
3. VulBERTa-CNN
4. examples-MNIST CNN
5. examples-MNIST RNN
6. examples-MNIST FF
7. examples-Siamese

### ⏳ 待完成（阶段3-可选）
需要添加/增强precision支持的6个模型：
1. MRT-OAST (添加fp16/bf16)
2. pytorch_resnet_cifar10 (添加bf16，已有fp16)
3. examples-MNIST CNN (添加fp16/bf16)
4. examples-MNIST RNN (添加fp16/bf16)
5. examples-MNIST FF (添加fp16/bf16)
6. examples-Siamese (添加fp16/bf16)

---

## 🎯 阶段2详细修改步骤

### 1. MRT-OAST - 添加weight_decay支持

**文件路径**: `/home/green/energy_dl/nightly/models/MRT-OAST/`

#### 1.1 修改 `main_batch.py`

**步骤1**: 添加argparse参数（在Line 197附近）
```python
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0,
                    help="weight decay (L2 penalty, default: 0.0)")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="gamma")
```

**步骤2**: 修改optimizer创建（在Line 105附近）
```python
# 修改前:
optimizer = optim.Adam(model.parameters(), lr=1.0)

# 修改后:
optimizer = optim.Adam(model.parameters(), lr=1.0, weight_decay=args.weight_decay)
```

#### 1.2 修改 `train.sh`

**步骤1**: 添加默认值（在Line 93附近）
```bash
DROPOUT=0.2
SEED=1334
WEIGHT_DECAY=0.0  # 新增
VALID_STEP=1750
```

**步骤2**: 添加usage说明（在Line 46附近）
```bash
    --dropout RATE          Dropout率 (默认: 0.2)
    --weight-decay DECAY    权重衰减 (默认: 0.0)
    --seed N                随机种子 (默认: 1334)
```

**步骤3**: 添加参数解析（在Line 136附近）
```bash
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
```

**步骤4**: 添加配置显示（在Line 290附近）
```bash
echo "Dropout: $DROPOUT"
echo "Weight Decay: $WEIGHT_DECAY"
echo "随机种子: $SEED"
```

**步骤5**: 添加到训练命令（在Line 319附近）
```bash
TRAIN_CMD="python main_batch.py \
    ...
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    ..."
```

#### 1.3 验证
```bash
cd /home/green/energy_dl/nightly/models/MRT-OAST
python main_batch.py --help | grep "weight.decay"
# 应该显示: --weight_decay WEIGHT_DECAY

# 测试默认值
./train.sh --dataset OJClone --epochs 1 --quick 2>&1 | grep "Weight"
# 应该显示: Weight Decay: 0.0
```

---

### 2. VulBERTa (MLP & CNN) - 添加weight_decay支持

**文件路径**: `/home/green/energy_dl/nightly/models/VulBERTa/`

#### 2.1 查找训练脚本
```bash
cd /home/green/energy_dl/nightly/models/VulBERTa
find . -name "train*.py" -o -name "main*.py"
```

#### 2.2 修改模式（参考）

**Python文件修改**:
```python
# 1. 添加argparse参数
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')

# 2. 修改optimizer（通常是AdamW）
# 修改前:
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# 修改后:
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

**Shell脚本修改**:
```bash
# 1. 添加默认值
WEIGHT_DECAY=0.0

# 2. 添加参数解析
--weight-decay)
    WEIGHT_DECAY="$2"
    shift 2
    ;;

# 3. 添加到命令
python train.py --weight_decay $WEIGHT_DECAY
```

---

### 3. examples (4个模型) - 添加weight_decay支持

**文件路径**: `/home/green/energy_dl/nightly/models/examples/`

这4个模型共享同一个训练脚本：
- MNIST CNN
- MNIST RNN
- MNIST FF
- Siamese Network

#### 3.1 查找脚本结构
```bash
cd /home/green/energy_dl/nightly/models/examples
ls -la
```

#### 3.2 修改每个模型的主Python文件

**MNIST CNN (使用SGD)**:
```python
# main.py中
parser.add_argument('--weight-decay', type=float, default=0.0)

# optimizer修改
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
```

**MNIST RNN, FF, Siamese (使用Adam)**:
```python
# main.py中
parser.add_argument('--weight-decay', type=float, default=0.0)

# optimizer修改
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
```

#### 3.3 修改train.sh（如果存在）
```bash
# 添加默认值
WEIGHT_DECAY=0.0

# 添加参数
--weight-decay)
    WEIGHT_DECAY="$2"
    shift 2
    ;;

# 添加到命令
--weight-decay $WEIGHT_DECAY
```

---

## 🎯 阶段3详细修改步骤（可选）

### Precision支持模式

#### PyTorch混合精度训练模板
```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 1. 添加argparse
parser.add_argument('--fp16', action='store_true', help='use fp16')
parser.add_argument('--bf16', action='store_true', help='use bf16')

# 2. 设置dtype
if args.fp16:
    dtype = torch.float16
elif args.bf16:
    dtype = torch.bfloat16
else:
    dtype = torch.float32

# 3. 创建GradScaler（fp16需要）
scaler = GradScaler() if args.fp16 else None

# 4. 训练循环中使用
for data, target in train_loader:
    if args.fp16 or args.bf16:
        with autocast(device_type='cuda', dtype=dtype):
            output = model(data)
            loss = criterion(output, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## 📝 完整修改清单

### 阶段2: Weight Decay

| 模型 | Python文件 | Shell文件 | 优化器 | 默认值 | 状态 |
|------|-----------|----------|--------|--------|------|
| MRT-OAST | main_batch.py | train.sh | Adam | 0.0 | ⏳ |
| VulBERTa-MLP | train_vulberta.py | train.sh | AdamW | 0.0 | ⏳ |
| VulBERTa-CNN | train_vulberta.py | train.sh | AdamW | 0.0 | ⏳ |
| examples-MNIST CNN | main.py | train.sh | SGD | 0.0 | ⏳ |
| examples-MNIST RNN | main.py | train.sh | Adam | 0.0 | ⏳ |
| examples-MNIST FF | main.py | train.sh | Adam | 0.0 | ⏳ |
| examples-Siamese | main.py | train.sh | Adam | 0.0 | ⏳ |

**预计时间**: 1-1.5小时
**代码行数**: ~70行总计（每个模型~10行）

### 阶段3: Precision（可选）

| 模型 | 当前支持 | 需要添加 | 代码行数 | 状态 |
|------|---------|---------|----------|------|
| MRT-OAST | 无 | fp16/bf16 | ~30行 | ⏳ |
| pytorch_resnet_cifar10 | fp16 | bf16 | ~5行 | ⏳ |
| examples-MNIST CNN | 无 | fp16/bf16 | ~30行 | ⏳ |
| examples-MNIST RNN | 无 | fp16/bf16 | ~30行 | ⏳ |
| examples-MNIST FF | 无 | fp16/bf16 | ~30行 | ⏳ |
| examples-Siamese | 无 | fp16/bf16 | ~30行 | ⏳ |

**预计时间**: 2-3小时
**代码行数**: ~155行总计

---

## 🔍 验证方法

### Weight Decay验证
对于每个修改的模型：

```bash
# 1. 检查参数帮助
python main.py --help | grep "weight"

# 2. 测试默认值
python main.py --epochs 1 2>&1 | grep -i "weight"

# 3. 测试自定义值
python main.py --epochs 1 --weight-decay 0.001
```

### Precision验证
```bash
# 1. 检查参数
python main.py --help | grep -E "fp16|bf16"

# 2. 测试fp16
python main.py --epochs 1 --fp16

# 3. 测试bf16
python main.py --epochs 1 --bf16
```

---

## 🎯 预期最终结果

完成所有修改后：

| 超参数 | 当前支持率 | 完成后支持率 | 提升 |
|--------|----------|-------------|------|
| epochs | 100% | 100% | - |
| learning_rate | 83% | 83% | - |
| **seed** | **100%** ✅ | **100%** ✅ | - |
| precision | 42% | **92%** | +50% |
| dropout | 25% | 25% | - |
| **weight_decay** | 58% | **92%** | +34% |
| **平均支持率** | **68%** | **83%** | **+15%** 🎉 |

---

## 💡 修改技巧和注意事项

### 1. Weight Decay默认值选择
- **新添加的weight_decay**: 使用`default=0.0`
- **原因**: 0.0表示不使用正则化，与原始代码行为一致
- **常用值**: 1e-5 (小), 1e-4 (中), 1e-3 (大)

### 2. Optimizer选择
- **SGD**: 直接在参数中添加`weight_decay`
- **Adam/AdamW**: AdamW更适合weight decay，但Adam也支持
- **验证**: 查看optimizer创建行，添加`weight_decay=args.weight_decay`

### 3. Precision实现要点
- **fp16**: 需要GradScaler防止梯度下溢
- **bf16**: 不需要GradScaler，范围更大但精度略低
- **检测GPU支持**: 检查GPU是否支持bf16（Ampere及以上架构）

### 4. 验证要点
- ✅ 不传参数时默认值为0.0（weight_decay）或None（precision）
- ✅ 传参数时正确应用
- ✅ 训练仍能正常运行
- ✅ 原有参数默认值未改变

---

## 🚀 下一步行动

### 选项1: 继续由AI完成
如果您希望我继续完成所有修改：
```
请继续完成阶段2：为7个模型添加weight_decay支持
```

### 选项2: 自行完成
按照本指南逐个模型修改，每完成一个模型后：
1. 运行验证命令
2. 记录修改日志
3. 更新超参数支持矩阵

### 选项3: 混合方式
我完成部分模型（如MRT-OAST, VulBERTa），您完成剩余的examples模型。

---

## 📚 参考文档

- [PyTorch Optimizer文档](https://pytorch.org/docs/stable/optim.html)
- [PyTorch混合精度训练](https://pytorch.org/docs/stable/amp.html)
- [Weight Decay解释](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

---

**文档版本**: 1.0
**生成时间**: 2025-11-05
**预计完成时间**: 3-4.5小时（全部完成）
