# 超参数添加模式：保持原始默认值

## 📌 核心原则

**在添加新的超参数支持时，必须确保：**
1. ✅ 不传参数时 = 原始训练行为
2. ✅ 传参数时 = 启用新功能
3. ✅ 默认值必须与原始代码行为完全一致

---

## 🎯 标准模式

### 模式1: 添加原始不存在的参数（如seed）

#### ❌ 错误做法（会改变原始行为）
```python
# train.py
parser.add_argument('--seed', type=int, default=42, help='random seed')

# 训练代码
args = parser.parse_args()
torch.manual_seed(args.seed)  # ❌ 即使不传--seed，也会设置seed=42
```

**问题**: 原始代码没有seed，现在默认使用42，改变了原始随机性！

---

#### ✅ 正确做法1（使用None作为默认值）
```python
# train.py
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None, uses random behavior)')

# 训练代码
args = parser.parse_args()

# 只有明确传入seed时才设置
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 可选：设置确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using seed: {args.seed}")
else:
    print("No seed set - using non-deterministic training (original behavior)")
```

**效果**:
- `./train.sh` → 不设置seed（原始行为）✅
- `./train.sh --seed 42` → 设置seed=42（新功能）✅

---

#### ✅ 正确做法2（推荐用于实验）
```python
# train.py
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')

# 训练代码
args = parser.parse_args()

# 如果实验框架要求所有实验可重复，可以在这里设置默认值
if args.seed is None:
    args.seed = 42  # 为实验设置默认值
    print(f"No seed specified, using default seed: {args.seed} for reproducibility")

# 设置seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
```

**适用场景**: 当实验可重复性比完全保持原始行为更重要

---

### 模式2: 添加原始不存在的precision参数

#### ❌ 错误做法
```python
# train.py
parser.add_argument('--precision', type=str, default='fp16',
                    choices=['fp16', 'bf16', 'fp32'])

# 训练循环
if args.precision == 'fp16':
    with torch.cuda.amp.autocast():  # ❌ 默认就启用fp16了！
        ...
```

**问题**: 原始代码使用fp32，现在默认fp16，性能会不同！

---

#### ✅ 正确做法（MRT-OAST示例）
```python
# main_batch.py
parser.add_argument('--precision', type=str, default=None,
                    choices=['fp16', 'bf16', 'fp32', None],
                    help='Mixed precision training (default: None, uses fp32)')

# 或者更明确的方式
parser.add_argument('--fp16', action='store_true',
                    help='Use fp16 mixed precision')
parser.add_argument('--bf16', action='store_true',
                    help='Use bf16 mixed precision')

# 训练循环
args = parser.parse_args()

# 确定使用的精度
use_amp = False
amp_dtype = torch.float32

if args.fp16:
    use_amp = True
    amp_dtype = torch.float16
elif args.bf16:
    use_amp = True
    amp_dtype = torch.bfloat16

# 创建GradScaler（只在使用混合精度时）
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# 训练循环
for data, target in train_loader:
    if use_amp:
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # 原始训练方式（fp32）
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**效果**:
- `./train.sh` → 使用fp32（原始行为）✅
- `./train.sh --fp16` → 使用fp16混合精度 ✅
- `./train.sh --bf16` → 使用bf16混合精度 ✅

---

### 模式3: 添加原始不存在的weight_decay参数

#### ❌ 错误做法（MRT-OAST）
```python
# main_batch.py
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')

# 优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)  # ❌ 原始代码是0，现在是1e-4！
```

---

#### ✅ 正确做法
```python
# main_batch.py
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay (default: 0, matches original code)')

# 优化器（与原始代码保持一致）
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)  # 默认为0 ✅
```

**关键**: 查看原始优化器配置，确保默认值一致！

原始MRT-OAST代码：
```python
# 原始 main_batch.py:105
optimizer = optim.Adam(model.parameters(), lr=1.0)
# 没有weight_decay参数 → 默认值是0
```

---

### 模式4: 修改已有参数但改变默认值（谨慎！）

#### ❌ 绝对禁止
```python
# pytorch_resnet_cifar10/trainer.py 原始代码
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)

# ❌ 错误修改
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
# 改变了默认值！会影响原始性能！
```

---

#### ✅ 正确做法（保持原始默认值）
```python
# pytorch_resnet_cifar10/trainer.py
# 保持原始默认值不变
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# ✅ 默认值仍然是1e-4，与原始代码一致
```

---

## 📚 各仓库具体修改示例

### 示例1: pytorch_resnet_cifar10 添加seed

#### 修改文件: `trainer.py`

```python
# 在argparse部分添加（第57行附近）
parser.add_argument('--seed', type=int, default=None,
                    help='random seed for reproducibility (default: None)')

# 在main函数中，模型创建之前添加
def main():
    global args, best_prec1
    args = parser.parse_args()

    # === 新增：设置seed ===
    if args.seed is not None:
        import random
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # 注意：原始代码有 cudnn.benchmark = True (line 89)
        # 设置seed时需要覆盖它
        cudnn.deterministic = True
        cudnn.benchmark = False
        print(f"Using seed: {args.seed} (deterministic mode)")
    else:
        # 保持原始行为
        cudnn.benchmark = True  # 原始代码第89行
        print("No seed set - using non-deterministic training (original behavior)")
    # === 新增结束 ===

    # 检查save_dir...（原始代码继续）
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # ...
```

#### train.sh修改

```bash
# 在参数解析部分添加（第56行附近）
--seed)
    SEED="$2"
    shift 2
    ;;

# 在构建训练命令时（第200行附近）
TRAIN_CMD="$PYTHON -u trainer.py \
    --arch=$MODEL_NAME \
    --epochs=$EPOCHS \
    ...
    $([ -n "$SEED" ] && echo "--seed=$SEED") \
    $USE_HALF"
```

---

### 示例2: MRT-OAST 添加weight_decay

#### 修改文件: `main_batch.py`

```python
# 在argparse部分添加（第201行附近，seed参数之后）
parser.add_argument("--weight_decay", type=float, default=0,
                    help="weight decay (L2 penalty) (default: 0)")

# 修改优化器部分（第105行）
# 原始代码：
# optimizer = optim.Adam(model.parameters(), lr=1.0)

# 修改为：
optimizer = optim.Adam(model.parameters(), lr=1.0,
                       weight_decay=args.weight_decay)  # 添加这个参数
```

#### train.sh修改

```bash
# 在默认参数部分添加（第92行附近）
WEIGHT_DECAY=0  # 与代码默认值一致

# 在参数解析添加
--weight-decay)
    WEIGHT_DECAY="$2"
    shift 2
    ;;

# 在训练命令构建添加
TRAIN_CMD="python main_batch.py \
    ...
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    ...
```

---

### 示例3: Person_reID_baseline_pytorch 添加seed

#### 修改文件: `train.py`

```python
# 在argparse部分添加（第86行warm_epoch之后）
parser.add_argument('--seed', default=None, type=int,
                    help='random seed for reproducibility (default: None)')

# 在开始训练前添加（找到GPU设置部分后）
# 原始代码大约在第200行有use_gpu相关代码
# 在那之后添加：

args = parser.parse_args()

# === 新增seed设置 ===
if args.seed is not None:
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using seed: {args.seed}")
# === 新增结束 ===
```

#### train.sh修改（已经有模板了）

```bash
# 在默认参数部分添加
SEED=""  # 空字符串表示不设置

# 在参数解析添加
--seed)
    SEED="$2"
    shift 2
    ;;

# 在构建训练命令时
build_train_command() {
    local cmd="$PYTHON_PATH train.py"
    ...
    [ -n "$SEED" ] && cmd="$cmd --seed $SEED"
    ...
    echo "$cmd"
}
```

---

### 示例4: VulBERTa 添加weight_decay

#### 修改文件: `train_vulberta.py`

```python
# 在argparse添加（第167行附近，fp16之后）
parser.add_argument('--weight_decay', type=float, default=None,
                    help='Weight decay (default: 0 for both MLP and CNN)')

# 在设置模型默认值时（第188-197行）
if args.model_name == 'mlp':
    if args.batch_size is None:
        args.batch_size = 2
    ...
    if args.weight_decay is None:
        args.weight_decay = 0  # 新增
else:  # cnn
    if args.batch_size is None:
        args.batch_size = 128
    ...
    if args.weight_decay is None:
        args.weight_decay = 0  # 新增

# 在TrainingArguments中添加（第296-309行）
training_args = TrainingArguments(
    output_dir=output_dir,
    ...
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,  # 新增这一行
    fp16=args.fp16,
    ...
)
```

#### train.sh修改（已经很简单，直接传参即可）

```bash
# VulBERTa的train.sh已经直接传所有参数给Python
# 只需要在帮助信息中添加说明
show_help() {
    cat << EOF
...
Optional arguments:
    ...
    --weight_decay DECAY  Weight decay (default: 0)
...
EOF
}
```

---

## ✅ 验证清单

在修改完代码后，运行以下测试验证默认行为：

### 测试1: 不传参数（验证原始默认行为）
```bash
# 应该使用所有原始默认值
./train.sh

# 检查输出日志，确认：
# - seed: None (或未设置seed相关日志)
# - weight_decay: [原始默认值]
# - precision: fp32 (无混合精度日志)
# - 其他参数使用原始默认值
```

### 测试2: 传入新参数（验证新功能）
```bash
# 测试seed
./train.sh --seed 42
# 应该看到: "Using seed: 42" 或类似日志

# 测试precision
./train.sh --fp16
# 应该看到: 混合精度训练相关日志

# 测试weight_decay
./train.sh --weight_decay 0.001
# 检查优化器配置日志
```

### 测试3: 对比结果（可选）
```bash
# baseline（使用默认值）
./train.sh 2>&1 | tee baseline.log

# 使用之前保存的原始训练日志对比
# 验证loss曲线、准确率是否一致（允许小幅波动）
```

---

## ⚠️ 常见陷阱

### 陷阱1: 忘记检查原始优化器配置
```python
# ❌ 假设原始有weight_decay
parser.add_argument('--weight_decay', default=1e-4)

# ✅ 先查看原始代码
# 如果原始optimizer没有weight_decay参数，默认值应该是0！
parser.add_argument('--weight_decay', default=0)
```

### 陷阱2: 修改了cudnn.benchmark设置
```python
# 原始代码
cudnn.benchmark = True  # 使用非确定性算法加速

# ❌ 错误：总是设置deterministic
cudnn.deterministic = True
cudnn.benchmark = False

# ✅ 正确：只在设置seed时修改
if args.seed is not None:
    cudnn.deterministic = True
    cudnn.benchmark = False
else:
    cudnn.benchmark = True  # 保持原始设置
```

### 陷阱3: precision参数的互斥性
```python
# ❌ 错误：允许同时设置fp16和bf16
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
# 没有互斥检查！

# ✅ 正确：添加互斥检查
args = parser.parse_args()
if args.fp16 and args.bf16:
    raise ValueError("Cannot use both --fp16 and --bf16")
```

### 陷阱4: 不同模型的不同默认值
```python
# Person_reID中不同模型有不同的lr
# ❌ 错误：所有模型用同一个默认值
LR = 0.05

# ✅ 正确：根据模型设置默认值
case "$MODEL_NAME" in
    "pcb")
        LR=0.02  # PCB特殊的lr
        ;;
    *)
        LR=0.05  # 其他模型默认lr
        ;;
esac
```

---

## 📝 修改后的文档更新

每次修改代码后，必须更新以下文档：

1. `original_hyperparameter_defaults.md`
   - 更新对应仓库的参数默认值表
   - 标注修改日期和修改内容

2. `hyperparameter_mutation_analysis.md`
   - 更新支持情况统计
   - 更新"需要修改的文件"清单

3. README或train.sh的help信息
   - 添加新参数的说明
   - 标注默认值

---

## 🎓 最佳实践总结

1. **Always use `None` for new optional parameters**
   - 便于区分"未设置"和"设置为默认值"

2. **查看原始优化器配置**
   - 确认参数是否存在，默认值是什么

3. **保留原始的cudnn.benchmark设置**
   - 只在明确需要时修改

4. **添加清晰的日志**
   - 打印实际使用的超参数值
   - 标注是默认值还是用户指定

5. **编写验证测试**
   - 不传参数 = 原始行为
   - 传参数 = 新功能启用

6. **文档同步更新**
   - 代码、文档、配置文件保持一致

---

**文档版本**: 1.0
**最后更新**: 2025-11-05
**作者**: Claude Code

**记住**: 实验可重复性很重要，但保持原始baseline行为同样重要！
