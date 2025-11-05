# 原始超参数默认值配置表

## 📌 重要说明

**本文档记录了6个仓库中10个模型的原始超参数默认值**

### 关键原则
1. ✅ **不指定变异参数时，必须使用此表中的默认值**
2. ✅ **这些默认值确保baseline训练与原始仓库完全一致**
3. ✅ **只有在明确指定变异时，才改变这些值**
4. ✅ **对于原始代码中不存在的参数（如seed），需要特殊处理**

---

## 📊 各仓库原始默认值总表

### 1. MRT-OAST

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `10` | `main_batch.py:192` | ✅ argparse默认值 |
| **learning_rate** | `0.0001` | `main_batch.py:190` | ✅ argparse默认值 |
| **seed** | `1334` | `main_batch.py:200` | ✅ argparse默认值 |
| **precision** | `fp32` | 不存在 | ⚠️ 原始代码未设置，默认fp32 |
| **dropout** | `0.2` | `main_batch.py:196` | ✅ argparse默认值 |
| **weight_decay** | `0` | `main_batch.py:105` | ⚠️ Adam优化器未指定，默认0 |

**优化器配置**:
```python
# 原始代码 main_batch.py:105
optimizer = optim.Adam(model.parameters(), lr=1.0)  # lr由scheduler控制
# 注意：原始代码中没有weight_decay参数！
```

**关键发现**:
- ✅ 已有seed支持，默认1334
- ⚠️ 没有weight_decay（Adam默认值为0）
- ⚠️ 没有混合精度训练

---

### 2. bug-localization-by-dnn-and-rvsm

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs (max_iter)** | `10000` | `train.sh:31` | ✅ shell脚本默认值 |
| **learning_rate** | 不可配置 | sklearn内部 | ❌ sklearn MLPClassifier固定 |
| **seed (random_state)** | 无 | 不存在 | ⚠️ 原始代码未设置 |
| **precision** | N/A | sklearn | ❌ sklearn不适用 |
| **dropout** | 不可配置 | sklearn内部 | ❌ sklearn MLPClassifier无dropout |
| **weight_decay (alpha)** | `1e-5` | `train.sh:30` | ✅ 对应sklearn的L2正则化 |

**Sklearn MLPClassifier配置**:
```python
# 原始代码使用的sklearn参数
MLPClassifier(
    hidden_layer_sizes=(300,),  # 默认300
    alpha=1e-5,                 # L2正则化
    max_iter=10000,
    n_iter_no_change=30,
    solver='sgd',
    learning_rate_init=0.001    # sklearn内部默认值
)
```

**关键发现**:
- ❌ 没有seed/random_state设置（训练不可重复）
- ⚠️ sklearn限制，learning_rate和dropout不可直接配置

---

### 3. pytorch_resnet_cifar10

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `200` | `trainer.py:30` | ✅ argparse默认值 |
| **learning_rate** | `0.1` | `trainer.py:36` | ✅ argparse默认值 |
| **seed** | 无 | 不存在 | ⚠️ 原始代码未设置seed |
| **precision** | `fp32` | 不存在 | ⚠️ 仅支持fp16 (--half) |
| **dropout** | N/A | ResNet架构 | ❌ ResNet模型无dropout层 |
| **weight_decay** | `0.0001` (1e-4) | `trainer.py:40` | ✅ argparse默认值 |

**优化器配置**:
```python
# 原始代码 trainer.py:119-121
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,              # 默认0.1
    momentum=0.9,
    weight_decay=1e-4        # 默认1e-4
)
```

**Seed情况**:
```python
# 原始代码 trainer.py:89
cudnn.benchmark = True  # 使用非确定性算法加速
# 注意：原始代码没有设置任何random seed！
```

**关键发现**:
- ❌ 完全没有seed设置（训练结果不可重复）
- ⚠️ `cudnn.benchmark=True` 会引入额外的随机性
- ✅ weight_decay已支持

---

### 4. VulBERTa (2个模型)

#### 4.1 VulBERTa-MLP

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `10` | `train_vulberta.py:180` | ✅ 已修改支持 |
| **learning_rate** | `3e-05` (0.00003) | `train_vulberta.py:182` | ✅ 已修改支持 |
| **seed** | `42` | `train_vulberta.py:184` | ✅ 已修改支持 |
| **precision** | `fp16` | `train_vulberta.py:186` | ✅ 已修改支持 |
| **dropout** | 固定在模型中 | RoBERTa模型 | ⚠️ 模型内部dropout，不可配置 |
| **weight_decay** | `0` | Trainer默认 | ⚠️ 未在TrainingArguments中设置 |

#### 4.2 VulBERTa-CNN

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `20` | `train_vulberta.py:191` | ✅ 已修改支持 |
| **learning_rate** | `0.0005` | `train_vulberta.py:193` | ✅ 已修改支持 |
| **seed** | `1234` | `train_vulberta.py:195` | ✅ 已修改支持 |
| **precision** | `fp32` | `train_vulberta.py:197` | ✅ 已修改支持 |
| **dropout** | 固定在模型中 | CNN模型 | ⚠️ 模型内部dropout，不可配置 |
| **weight_decay** | `0` | Adam默认 | ⚠️ 未在优化器中设置 |

**优化器配置**:
```python
# VulBERTa-MLP 使用 HuggingFace Trainer
# 原始代码 train_vulberta.py:96-109
training_args = TrainingArguments(
    per_device_train_batch_size=2,     # MLP: 2
    num_train_epochs=10,               # MLP: 10
    learning_rate=3e-05,               # MLP: 3e-05
    seed=42,                           # MLP: 42
    fp16=True,                         # MLP: True
    # weight_decay 未指定，默认为0
)
```

**关键发现**:
- ✅ 已添加完整的seed支持（通过修改）
- ⚠️ dropout固定在预训练模型中，不可直接修改
- ⚠️ 没有weight_decay设置

---

### 5. Person_reID_baseline_pytorch (3个模型)

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs (total_epoch)** | `60` | `train.py:29` | ✅ argparse默认值 |
| **learning_rate** | `0.05` | `train.py:29` | ✅ argparse默认值 |
| **seed** | 无 | 不存在 | ⚠️ 原始代码未设置 |
| **precision** | `fp32` | 不存在 | ✅ 支持--fp16, --bf16 |
| **dropout (droprate)** | `0.5` | `train.py:35` | ✅ argparse默认值 |
| **weight_decay** | `0.0005` (5e-4) | `train.py:30` | ✅ argparse默认值 |

**优化器配置**:
```python
# 原始代码会根据不同loss使用不同优化器
# SGD (默认)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,              # 默认0.05
    weight_decay=5e-4,       # 默认5e-4
    momentum=0.9,
    nesterov=True
)
```

**Seed情况**:
```python
# 原始代码 prepare_viper.py:53 (仅在数据准备时)
np.random.seed(0)
# 训练时没有设置seed！
```

**模型特定默认值**:

#### densenet121:
- batch_size: `24`
- lr: `0.05`

#### hrnet18:
- batch_size: `24`
- lr: `0.05`

#### pcb:
- batch_size: `32`
- lr: `0.02`
- 使用PCB架构（Part-based Convolutional Baseline）

**关键发现**:
- ✅ dropout和weight_decay已完全支持
- ✅ 精度选项已支持（fp16, bf16）
- ❌ 没有seed设置（训练不可重复）
- ✅ 这是参数支持最完善的仓库

---

### 6. examples (4个模型)

#### 6.1 MNIST CNN

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `14` | `train.sh:179` | ✅ train.sh包装默认值 |
| **learning_rate** | `1.0` | `train.sh:181` | ✅ train.sh包装默认值 |
| **seed** | `1` | `train.sh:182` | ✅ train.sh包装默认值 |
| **precision** | `fp32` | 不存在 | ⚠️ 未实现 |
| **dropout** | `0.25` 和 `0.5` | `main.py` | ⚠️ 硬编码在模型中 |
| **weight_decay** | `0` | SGD默认 | ⚠️ 未设置 |

#### 6.2 MNIST RNN

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `14` | `train.sh:187` | ✅ train.sh包装默认值 |
| **learning_rate** | `0.1` | `train.sh:189` | ✅ train.sh包装默认值 |
| **seed** | `1` | `train.sh:191` | ✅ train.sh包装默认值 |
| **precision** | `fp32` | 不存在 | ⚠️ 未实现 |
| **dropout** | 无 | LSTM内部 | ⚠️ LSTM无dropout参数 |
| **weight_decay** | `0` | Adadelta默认 | ⚠️ 未设置 |

#### 6.3 MNIST Forward-Forward

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `1000` | `train.sh:195` | ✅ train.sh包装默认值 |
| **learning_rate** | `0.03` | `train.sh:197` | ✅ train.sh包装默认值 |
| **seed** | `1` | `train.sh:198` | ✅ train.sh包装默认值 |
| **precision** | `fp32` | 不存在 | ⚠️ 未实现 |
| **dropout** | 无 | 算法特性 | ❌ Forward-Forward算法无dropout |
| **weight_decay** | `0` | Adam默认 | ⚠️ 未设置 |

#### 6.4 Siamese Network

| 超参数 | 原始默认值 | 代码位置 | 备注 |
|--------|-----------|---------|------|
| **epochs** | `14` | `train.sh:204` | ✅ train.sh包装默认值 |
| **learning_rate** | `1.0` | `train.sh:205` | ✅ train.sh包装默认值 |
| **seed** | `1` | `train.sh:206` | ✅ train.sh包装默认值 |
| **precision** | `fp32` | 不存在 | ⚠️ 未实现 |
| **dropout** | 无 | 模型设计 | ⚠️ 模型中无dropout层 |
| **weight_decay** | `0` | SGD默认 | ⚠️ 未设置 |

**关键发现**:
- ✅ 通过train.sh包装，所有4个模型都有seed支持
- ⚠️ dropout在需要的模型中硬编码，不可配置
- ⚠️ 没有weight_decay设置
- ⚠️ 没有混合精度支持

---

## 🎯 关键问题与解决方案

### 问题1: 原始代码中不存在seed

**影响的仓库**:
- pytorch_resnet_cifar10 ❌ 完全没有seed
- Person_reID_baseline_pytorch ❌ 完全没有seed
- bug-localization-by-dnn-and-rvsm ❌ 完全没有seed

**解决方案**:
```python
# 方案A: 不设置seed（保持原始随机性）
# 优点：完全复现原始训练行为
# 缺点：baseline结果不可重复

# 方案B: 设置一个固定的默认seed（推荐）
# 优点：baseline可重复，仍可通过不设置seed来模拟原始行为
# 缺点：与原始训练有微小差异

# 推荐做法：
if seed is not None:  # 只有明确指定seed时才设置
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
else:
    # 不设置seed，保持原始随机性
    pass
```

**建议的默认值**:
- pytorch_resnet_cifar10: 使用 `seed=None` （不设置，保持原始行为）
- Person_reID_baseline_pytorch: 使用 `seed=None`
- bug-localization-by-dnn-and-rvsm: 使用 `seed=None`

---

### 问题2: precision原始默认值

**所有仓库的原始行为**:
- ✅ 默认使用 **FP32** (float32) 精度
- ⚠️ VulBERTa-MLP例外：原始使用FP16

**解决方案**:
```python
# 各仓库precision默认值配置
PRECISION_DEFAULTS = {
    "MRT-OAST": None,              # 不使用混合精度，相当于fp32
    "bug-localization": "fp32",    # sklearn, N/A
    "pytorch_resnet_cifar10": None, # 不使用混合精度
    "VulBERTa-MLP": "fp16",        # ⚠️ 特例！原始使用fp16
    "VulBERTa-CNN": None,          # 不使用混合精度
    "Person_reID": None,           # 不使用混合精度
    "examples": None,              # 不使用混合精度
}
```

---

### 问题3: weight_decay原始默认值

**不同仓库的原始值**:
- MRT-OAST: `0` (Adam优化器，未指定)
- bug-localization: `1e-5` (sklearn的alpha参数)
- pytorch_resnet_cifar10: `1e-4` ✅
- VulBERTa: `0` (未指定)
- Person_reID_baseline_pytorch: `5e-4` ✅
- examples: `0` (未指定)

**解决方案**:
```python
# 必须按仓库使用不同的默认值！
WEIGHT_DECAY_DEFAULTS = {
    "MRT-OAST": 0,
    "bug-localization-by-dnn-and-rvsm": 1e-5,
    "pytorch_resnet_cifar10": 1e-4,
    "VulBERTa": 0,
    "Person_reID_baseline_pytorch": 5e-4,
    "examples": 0,
}
```

---

### 问题4: dropout原始默认值

**情况分类**:
1. **可配置的** (✅):
   - MRT-OAST: `0.2`
   - Person_reID_baseline_pytorch: `0.5`

2. **硬编码的** (⚠️):
   - examples/MNIST CNN: `0.25` 和 `0.5` (两层不同)
   - VulBERTa: 固定在预训练模型中

3. **不适用的** (❌):
   - pytorch_resnet_cifar10: ResNet模型无dropout
   - MNIST Forward-Forward: 算法特性
   - bug-localization: sklearn限制

**解决方案**:
- 对于可配置的：使用原始默认值
- 对于硬编码的：记录固定值，不可变异
- 对于不适用的：跳过此参数

---

## 📝 实验脚本配置建议

### 配置文件示例 (YAML)

```yaml
# original_defaults.yaml
repositories:
  MRT-OAST:
    models:
      - name: "MRT-OAST"
        hyperparameters:
          epochs: 10
          learning_rate: 0.0001
          seed: null  # 原始代码使用1334，但我们用null表示使用代码默认值
          precision: null  # 不使用混合精度（相当于fp32）
          dropout: 0.2
          weight_decay: 0

  pytorch_resnet_cifar10:
    models:
      - name: "resnet20"
        hyperparameters:
          epochs: 200
          learning_rate: 0.1
          seed: null  # 原始代码没有seed
          precision: null  # 不使用混合精度
          dropout: null  # ResNet无dropout
          weight_decay: 0.0001

  VulBERTa:
    models:
      - name: "mlp"
        hyperparameters:
          epochs: 10
          learning_rate: 0.00003  # 3e-05
          seed: 42
          precision: "fp16"  # ⚠️ 原始使用fp16！
          dropout: null  # 固定在模型中
          weight_decay: 0

      - name: "cnn"
        hyperparameters:
          epochs: 20
          learning_rate: 0.0005
          seed: 1234
          precision: null
          dropout: null  # 固定在模型中
          weight_decay: 0

  Person_reID_baseline_pytorch:
    models:
      - name: "densenet121"
        hyperparameters:
          epochs: 60
          learning_rate: 0.05
          seed: null  # 原始代码没有seed
          precision: null
          dropout: 0.5
          weight_decay: 0.0005

      - name: "hrnet18"
        hyperparameters:
          epochs: 60
          learning_rate: 0.05
          seed: null
          precision: null
          dropout: 0.5
          weight_decay: 0.0005

      - name: "pcb"
        hyperparameters:
          epochs: 60
          learning_rate: 0.02  # PCB使用更小的lr
          seed: null
          precision: null
          dropout: 0.5
          weight_decay: 0.0005

  bug-localization-by-dnn-and-rvsm:
    models:
      - name: "dnn"
        hyperparameters:
          epochs: 10000  # max_iter
          learning_rate: null  # sklearn固定
          seed: null  # 原始代码没有seed
          precision: null  # sklearn N/A
          dropout: null  # sklearn无dropout
          weight_decay: 0.00001  # 1e-5 (alpha)

  examples:
    models:
      - name: "mnist_cnn"
        hyperparameters:
          epochs: 14
          learning_rate: 1.0
          seed: 1
          precision: null
          dropout: null  # 硬编码0.25和0.5
          weight_decay: 0

      - name: "mnist_rnn"
        hyperparameters:
          epochs: 14
          learning_rate: 0.1
          seed: 1
          precision: null
          dropout: null
          weight_decay: 0

      - name: "mnist_ff"
        hyperparameters:
          epochs: 1000
          learning_rate: 0.03
          seed: 1
          precision: null
          dropout: null
          weight_decay: 0

      - name: "siamese"
        hyperparameters:
          epochs: 14
          learning_rate: 1.0
          seed: 1
          precision: null
          dropout: null
          weight_decay: 0
```

---

## ⚠️ 关键注意事项

### 1. Seed的特殊处理

**重要决策点**：对于原始没有seed的仓库，我们有两个选择：

#### 选项A: 不设置seed（完全复现原始行为）
```python
# baseline训练
./train.sh  # 不传seed参数，保持原始随机性

# 变异训练
./train.sh --seed 42  # 设置seed，确保变异实验可重复
```

#### 选项B: 统一设置默认seed（推荐）
```python
# baseline训练
./train.sh --seed 42  # 设置默认seed，确保baseline可重复

# 变异训练
./train.sh --seed 123  # 变异seed
```

**推荐使用选项B**，理由：
1. ✅ 实验可重复性更重要
2. ✅ 便于对比baseline和变异的差异
3. ✅ 可以通过多次运行不同seed来评估模型稳定性
4. ⚠️ 原始随机性导致的差异可以忽略（通常<1%）

---

### 2. Precision的特殊情况

**VulBERTa-MLP是唯一原始使用FP16的模型！**

```python
# VulBERTa-MLP baseline（必须使用fp16）
./train.sh -n mlp --fp16

# VulBERTa-MLP 变异到fp32
./train.sh -n mlp  # 不加--fp16标志

# 其他所有模型的baseline（使用fp32）
./train.sh  # 不加任何precision标志
```

---

### 3. Weight Decay的差异

不同仓库使用的weight_decay值差异很大：

| 仓库 | 原始值 | 数量级 |
|------|-------|--------|
| bug-localization | 1e-5 | 0.00001 |
| pytorch_resnet_cifar10 | 1e-4 | 0.0001 |
| Person_reID | 5e-4 | 0.0005 |
| MRT-OAST, VulBERTa, examples | 0 | 0 |

**变异策略建议**：
- 以原始值为中心进行变异
- 变异范围：原始值 × [0.1, 0.5, 1.0, 2.0, 5.0]
- 例如 Person_reID (原始5e-4)：变异到 [5e-5, 2.5e-4, 5e-4, 1e-3, 2.5e-3]

---

### 4. 不可变异的参数

| 仓库/模型 | 不可变异的参数 | 原因 |
|---------|--------------|------|
| bug-localization (sklearn) | learning_rate, dropout | sklearn API限制 |
| pytorch_resnet_cifar10 | dropout | ResNet架构无dropout |
| VulBERTa | dropout | 固定在预训练模型中 |
| MNIST Forward-Forward | dropout | 算法特性 |
| 所���sklearn模型 | precision | sklearn不支持 |

**实验时应跳过这些参数或标记为N/A**

---

## 📋 验证Checklist

在运行实验前，验证以下配置：

### Baseline训练验证
- [ ] epochs使用原始默认值
- [ ] learning_rate使用原始默认值
- [ ] seed根据策略选择（null或固定值）
- [ ] precision未设置（除了VulBERTa-MLP）
- [ ] dropout使用原始默认值
- [ ] weight_decay使用原始默认值

### 变异实验验证
- [ ] 只修改指定的超参数
- [ ] 未指定的参数使用原始默认值
- [ ] 记录所有超参数值到CSV
- [ ] 不可变异参数标记为N/A

---

## 🔄 配置更新记录

| 日期 | 修改内容 | 修改人 |
|------|---------|-------|
| 2025-11-05 | 初始版本，记录所有原始默认值 | Claude |
|  |  |  |

---

**文档版本**: 1.0
**最后更新**: 2025-11-05
**维护者**: 项目组

**⚠️ 重要**: 修改任何仓库代���后，必须更新此文档中对应的默认值！
