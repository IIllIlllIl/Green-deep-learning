# 代码修改日志

## 修改目标
为各仓库添加超参数支持，确保默认值保持原始训练行为不变。

---

## ✅ 已完成的修改

### 1. pytorch_resnet_cifar10 - 添加seed支持

**修改日期**: 2025-11-05
**修改文件**:
1. `trainer.py`
2. `train.sh`

**修改内容**:

#### trainer.py (3处修改)

**修改1**: 添加seed参数（第58-59行）
```python
parser.add_argument('--seed', type=int, default=None,
                    help='random seed for reproducibility (default: None, uses non-deterministic training)')
```

**修改2**: 添加seed设置逻辑（第67-78行）
```python
# Set random seed for reproducibility
if args.seed is not None:
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"=> Using seed: {args.seed} (deterministic mode)")
else:
    print("=> No seed set - using non-deterministic training (original behavior)")
```

**修改3**: 条件设置cudnn模式（第103-108行）
```python
# Set cudnn behavior based on whether seed is set
if args.seed is not None:
    cudnn.deterministic = True
    cudnn.benchmark = False
else:
    cudnn.benchmark = True  # Original behavior for faster training
```

#### train.sh (4处修改)

**修改1**: 添加SEED默认值（第41行）
```bash
SEED=""  # 空字符串表示不设置seed（保持原始随机行为）
```

**修改2**: 更新usage说明（第61行）
```bash
--seed SEED                 随机种子 (默认: 不设置，保持原始随机行为)
```

**修改3**: 添加seed参数解析（第123-126行）
```bash
--seed)
    SEED="$2"
    shift 2
    ;;
```

**修改4**: 添加seed到训练配置显示（第186行）
```bash
echo "  随机种子: $([ -n "$SEED" ] && echo "$SEED" || echo '未设置（原始随机行为）')"
```

**修改5**: 添加seed到训练命令（第221行）
```bash
$([ -n "$SEED" ] && echo "--seed=$SEED") \
```

**默认值保证**:
- ✅ `default=None` 确保不传参数时不设置seed
- ✅ 条件判断确保只有明确指定seed时才激活
- ✅ cudnn.benchmark保持原始行为（True）当没有seed时

**验证方法**:
```bash
# 测试1: 不传seed（应保持原始随机行为）
cd /home/green/energy_dl/nightly/models/pytorch_resnet_cifar10
./train.sh -e 1 --dry-run
# 预期输出: "No seed set - using non-deterministic training"

# 测试2: 传seed（应使用确定性模式）
./train.sh -e 1 --dry-run --seed 42
# 预期输出: "Using seed: 42 (deterministic mode)"
```

**状态**: ✅ 完成，待验证

---

## 📋 待完成的修改

### 阶段1: Seed支持（剩余4个仓库）

#### 2. Person_reID_baseline_pytorch
- **需要修改**: `train.py`, `train.sh`
- **难度**: 🟢 简单（15行代码）
- **优先级**: 高

#### 3. bug-localization-by-dnn-and-rvsm
- **需要修改**: `train_wrapper.py`, `train.sh`
- **难度**: 🟢 简单（10行代码）
- **注意**: sklearn的MLPClassifier需要使用`random_state`参数
- **优先级**: 中

#### 4. MRT-OAST
- **需要修改**: 无需修改（已支持seed）
- **状态**: ✅ 已有seed支持（默认1334）

#### 5. VulBERTa (2个模型)
- **需要修改**: 无需修改（已支持seed）
- **状态**: ✅ 已有seed支持

#### 6. examples (4个模型)
- **需要修改**: 无需修改（已支持seed）
- **状态**: ✅ 已有seed支持（通过train.sh）

### 阶段2: Weight Decay支持（7个模型）

#### 需要添加的仓库:
1. MRT-OAST (main_batch.py)
2. VulBERTa-MLP (train_vulberta.py)
3. VulBERTa-CNN (train_vulberta.py)
4. examples-MNIST CNN (main.py)
5. examples-MNIST RNN (main.py)
6. examples-MNIST FF (main.py)
7. examples-Siamese (main.py)

**原始默认值**:
- MRT-OAST: 0 (Adam未设置)
- VulBERTa: 0 (未设置)
- examples: 0 (未设置)

### 阶段3: Precision支持（6个模型）

#### 需要添加/增强的仓库:
1. MRT-OAST - 添加fp16/bf16支持
2. pytorch_resnet_cifar10 - 添加bf16支持（已有fp16）
3. examples-MNIST CNN - 添加fp16/bf16支持
4. examples-MNIST RNN - 添加fp16/bf16支持
5. examples-MNIST FF - 添加fp16/bf16支持
6. examples-Siamese - 添加fp16/bf16支持

---

## 📊 当前进度

| 阶段 | 总数 | 已完成 | 进度 |
|------|-----|-------|-----|
| 阶段1: Seed | 5个仓库 | 1/5 | 20% |
| 阶段2: Weight Decay | 7个模型 | 0/7 | 0% |
| 阶段3: Precision | 6个模型 | 0/6 | 0% |
| **总计** | **18个修改项** | **1/18** | **5.6%** |

---

## 🔍 验证清单

### pytorch_resnet_cifar10
- [ ] 验证不传seed时的行为（应输出"No seed set"）
- [ ] 验证传seed时的行为（应输出"Using seed: XX"）
- [ ] 验证两次不传seed的结果不同（随机性）
- [ ] 验证两次传相同seed的结果相同（可重复性）
- [ ] 验证默认训练性能与baseline接近

### Person_reID_baseline_pytorch
- [ ] 待添加

### bug-localization-by-dnn-and-rvsm
- [ ] 待添加

---

## 📝 修改模式总结

### Seed添加模式（适用于所有PyTorch模型）

**Python代码**:
```python
# 1. argparse
parser.add_argument('--seed', type=int, default=None)

# 2. seed设置
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
```

**Bash代码**:
```bash
# 1. 默认值
SEED=""

# 2. 参数解析
--seed)
    SEED="$2"
    shift 2
    ;;

# 3. 训练命令
$([ -n "$SEED" ] && echo "--seed=$SEED")
```

---

## ⚠️ 注意事项

### 1. 原始默认值必须保持不变
- pytorch_resnet_cifar10: `seed=None` ✅
- 其他PyTorch模型: `seed=None`
- sklearn模型: `random_state=None`

### 2. 特殊情况处理
- **MRT-OAST**: 已有seed支持（默认1334），但仍建议改为None以匹配原始随机行为
- **VulBERTa**: 已有seed支持，无需修改
- **examples**: 已通过train.sh支持seed

### 3. Sklearn模型（bug-localization）
需要使用 `random_state` 参数：
```python
MLPClassifier(..., random_state=args.seed if args.seed is not None else None)
```

---

## 🎯 下一步行动

根据当前进度，建议：

### 选项A: 继续完成所有seed修改
- 修改Person_reID (15行代码)
- 修改bug-localization (10行代码)
- 预计时间：30分钟
- 完成后seed支持率：100%

### 选项B: 先验证已完成的修改
- 运行pytorch_resnet_cifar10的验证测试
- 确保修改正确后再继续其他仓库
- 预计时间：15分钟

### 选项C: 开始weight_decay修改
- 跳过剩余seed修改
- 开始添加weight_decay支持
- 预计时间：1小时

---

**最后更新**: 2025-11-05
**修改者**: Claude Code
**状态**: 进行中
