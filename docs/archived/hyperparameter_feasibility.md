# 6个超参数变异可行性调研报告

生成时间: 2025-11-04
目标超参数: epochs, learning_rate, seed, dropout, weight_decay, 精度(fp16/bf16)

---

## 执行摘要

针对6个仓库12个模型的6个目标超参数进行了详细调研。主要发现：

### 整体支持情况

| 超参数 | 完全支持 | 部分支持 | 需要修改 | 不适用 |
|-------|---------|---------|---------|--------|
| **epochs** | 12/12 (100%) | 0 | 0 | 0 |
| **learning_rate** | 12/12 (100%) | 0 | 0 | 0 |
| **seed** | 7/12 (58%) | 0 | 5/12 (42%) | 0 |
| **dropout** | 5/12 (42%) | 0 | 6/12 (50%) | 1/12 (8%) |
| **weight_decay** | 4/12 (33%) | 1/12 (8%) | 7/12 (58%) | 0 |
| **精度 (fp16/bf16)** | 5/12 (42%) | 0 | 7/12 (58%) | 0 |

**核心结论**:
- ✅ **epochs和learning_rate**: 所有模型完全支持，无需任何修改
- ⭐ **seed**: 大部分支持，少数需要简单修改
- ⚠️ **dropout, weight_decay, 精度**: 需要不同程度的代码扩展

---

## 1. 各仓库详细调研结果

### 1.1 MRT-OAST (1个模型)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|-------|---------|------|
| **epochs** | ✅ 完全支持 | `--epochs` | N/A | 默认10 |
| **learning_rate** | ✅ 完全支持 | `--lr` | N/A | 默认0.0001 |
| **seed** | ✅ 完全支持 | `--seed` | N/A | 默认1334 |
| **dropout** | ✅ 完全支持 | `--dropout` | N/A | 默认0.2 |
| **weight_decay** | ❌ 不支持 | - | ⭐⭐ 容易 | optimizer硬编码为Adam(lr=1.0)，无weight_decay |
| **精度 (fp16/bf16)** | ❌ 不支持 | - | ⭐⭐ 容易 | 需要添加autocast |

**代码位置**:
- train.sh: models/MRT-OAST/train.sh
- 训练代码: models/MRT-OAST/main_batch.py:105
  ```python
  optimizer = optim.Adam(model.parameters(), lr=1.0)  # 第105行
  ```

**修改建议**:

1. **weight_decay** (约5-10行代码):
   ```python
   # 在main_batch.py中修改
   parser.add_argument("--weight_decay", type=float, default=0.0)

   # 修改optimizer初始化
   optimizer = optim.Adam(model.parameters(), lr=1.0, weight_decay=args.weight_decay)
   ```

2. **fp16/bf16** (约30-40行代码):
   ```python
   # 添加参数
   parser.add_argument("--fp16", action="store_true")
   parser.add_argument("--bf16", action="store_true")

   # 训练循环中添加autocast
   if args.fp16 or args.bf16:
       dtype = torch.float16 if args.fp16 else torch.bfloat16
       scaler = torch.cuda.amp.GradScaler()
       with torch.cuda.amp.autocast(dtype=dtype):
           current_loss = train(model, training_batch, args, optimizer, criterion)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

---

### 1.2 bug-localization-by-dnn-and-rvsm (1个模型 - DNN)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|-------|---------|------|
| **epochs** | ⚠️ 等效支持 | `--max_iter` | ⭐ 极易 | sklearn的max_iter相当于epochs |
| **learning_rate** | ⚠️ 内置 | - | ⭐⭐⭐ 中等 | sklearn内置，难以直接控制 |
| **seed** | ❌ 不支持 | - | ⭐ 极易 | sklearn的MLPRegressor有random_state参数 |
| **dropout** | ❌ 不适用 | - | N/A | sklearn的MLPRegressor不支持dropout |
| **weight_decay** | ✅ 等效支持 | `--alpha` | N/A | alpha是L2正则化系数（等效weight_decay） |
| **精度 (fp16/bf16)** | ❌ 不适用 | - | N/A | sklearn不支持混合精度 |

**特殊说明**:
- 这是基于sklearn的MLPRegressor，不是PyTorch模型
- learning_rate在sklearn中与adaptive学习率策略绑定，不建议单独修改
- 不支持GPU和混合精度

**修���建议**:

1. **seed** (约3行代码):
   ```python
   # 在train_wrapper.py中修改MLPRegressor初始化
   clf = MLPRegressor(
       solver=solver,
       alpha=alpha,
       hidden_layer_sizes=hidden_sizes,
       random_state=args.seed,  # 添加这一行
       max_iter=max_iter,
       n_iter_no_change=n_iter_no_change,
   )
   ```

2. **learning_rate** (不推荐修改):
   - sklearn的learning_rate_init参数可以设置初始学习率
   - 但会与adaptive策略冲突，可能影响收敛
   - 如果必须修改，需要同时调整learning_rate策略

---

### 1.3 pytorch_resnet_cifar10 (1个模型 - ResNet20)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|-------|---------|------|
| **epochs** | ✅ 完全支持 | `--epochs` / `-e` | N/A | 默认200 |
| **learning_rate** | ✅ 完全支持 | `--lr` | N/A | 默认0.1 |
| **seed** | ❌ 不支持 | - | ⭐ 极易 | 代码中有torch.manual_seed，但未参数化 |
| **dropout** | ❌ 不支持 | - | ⭐⭐⭐ 中等 | ResNet默认无dropout，需修改模型定义 |
| **weight_decay** | ✅ 完全支持 | `--wd` / `--weight-decay` | N/A | 默认0.0001 |
| **精度 (fp16)** | ✅ 完全支持 | `--half` | N/A | 支持fp16 |

**代码位置**:
- train.sh: models/pytorch_resnet_cifar10/train.sh
- 训练代码: models/pytorch_resnet_cifar10/trainer.py:232
  ```python
  torch.manual_seed(my_args.seed)  # 但seed未作为参数
  ```

**修改建议**:

1. **seed** (约10行代码):
   ```python
   # 在trainer.py中添加参数
   parser.add_argument('--seed', default=1, type=int, help='random seed')

   # 在main()中设置seed
   torch.manual_seed(args.seed)
   torch.cuda.manual_seed(args.seed)
   np.random.seed(args.seed)
   ```

2. **dropout** (约50-80行代码):
   ```python
   # 需要修���resnet.py中的BasicBlock类
   class BasicBlock(nn.Module):
       def __init__(self, ..., dropout=0.0):
           ...
           if dropout > 0:
               self.dropout = nn.Dropout(dropout)
           else:
               self.dropout = None

       def forward(self, x):
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
           if self.dropout:
               out = self.dropout(out)
           ...
   ```

---

### 1.4 VulBERTa (2个模型 - MLP + CNN)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|---------|---------|------|
| **epochs** | ✅ 完全支持 | `--epochs` | N/A | MLP默认10, CNN默认20 |
| **learning_rate** | ✅ 完全支持 | `--learning_rate` | N/A | MLP默认3e-05, CNN默认0.0005 |
| **seed** | ✅ 完全支持 | `--seed` | N/A | MLP默认42, CNN默认1234 |
| **dropout** | ❌ 不支持 | - | ⚠️ 困难 | RoBERTa预训练模型内置，难以修改 |
| **weight_decay** | ⚠️ 隐式支持 | - | ⭐⭐ 容易 | Transformers TrainingArguments默认有 |
| **精度 (fp16)** | ✅ 完全支持 | `--fp16` | N/A | MLP默认True, CNN默认False |

**代码位置**:
- train.sh: models/VulBERTa/train.sh
- 训练代码: models/VulBERTa/train_vulberta.py:295

**修改建议**:

1. **weight_decay** (约3行代码):
   ```python
   # 在train_vulberta.py中添加参数
   parser.add_argument('--weight_decay', type=float, default=0.0)

   # 在TrainingArguments中添加
   training_args = TrainingArguments(
       ...
       weight_decay=args.weight_decay,  # 添加这一行
   )
   ```

2. **dropout** (不推荐修改):
   - RoBERTa模型的dropout是在预训练时固定的
   - 修改需要重新配置模型config
   - 影响预训练权重的使用
   - 如果必须修改，需要在model加载时设置config:
     ```python
     config = RobertaConfig.from_pretrained('./models/VulBERTa/')
     config.hidden_dropout_prob = args.dropout
     model = RobertaForSequenceClassification.from_pretrained(
         './models/VulBERTa/', config=config
     )
     ```

---

### 1.5 Person_reID_baseline_pytorch (3个模型 - densenet121, hrnet18, pcb)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|-------|---------|------|
| **epochs** | ✅ 完全支持 | `--total_epoch` | N/A | 默认60 |
| **learning_rate** | ✅ 完全支持 | `--lr` | N/A | 默认0.05 |
| **seed** | ❌ 不支持 | - | ⭐ 极易 | 代码中无随机种子设置 |
| **dropout** | ✅ 完全支持 | `--droprate` | N/A | 默认0.5 |
| **weight_decay** | ✅ 完全支持 | (内置) | N/A | train.py中硬编码5e-4 |
| **精度 (fp16/bf16)** | ✅ 完全支持 | `--fp16` / `--bf16` | N/A | 两者都支持 |

**代码位置**:
- train.sh: models/Person_reID_baseline_pytorch/train.sh
- 训练代码: models/Person_reID_baseline_pytorch/train.py:48

**修改建议**:

1. **seed** (约15行代码):
   ```python
   # 在train.py中添加参数
   parser.add_argument('--seed', default=42, type=int, help='random seed')

   # 在训练开始前设置seed
   import random
   random.seed(opt.seed)
   np.random.seed(opt.seed)
   torch.manual_seed(opt.seed)
   torch.cuda.manual_seed_all(opt.seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. **weight_decay参数化** (约8行代码):
   ```python
   # 当前weight_decay硬编码在train.py中
   # 添加参���使其可配置
   parser.add_argument('--weight_decay', default=5e-4, type=float)

   # 在optimizer初始化时使用
   optimizer_ft = optim.SGD([...], weight_decay=opt.weight_decay, ...)
   ```

---

### 1.6 examples (4个模型 - mnist, mnist_rnn, mnist_ff, siamese)

**当前支持情况**:

| 超参数 | 支持状态 | 参数名 | 修改难度 | 说明 |
|-------|---------|-------|---------|------|
| **epochs** | ✅ 完全支持 | `-e` / `--epochs` | N/A | 各模型默认值不同 |
| **learning_rate** | ✅ 完全支持 | `-l` / `--lr` | N/A | 各模型默认值不同 |
| **seed** | ✅ 完全支持 | `--seed` | N/A | 默认1 |
| **dropout** | ⚠️ 部分支持 | - | ⭐⭐ 容易 | 模型中有dropout，但未参数化 |
| **weight_decay** | ❌ 不支持 | - | ⭐ 极易 | 优化器中可直接添加 |
| **精度 (fp16/bf16)** | ❌ 不支持 | - | ⭐⭐ 容易 | 需要添加autocast |

**代码位置**:
- train.sh: models/examples/train.sh
- 训练代码: models/examples/mnist/main.py, models/examples/mnist_rnn/main.py, etc.

**修改建议**:

1. **dropout参数化** (约10行代码/模型):
   ```python
   # 在各个main.py中添加参数
   parser.add_argument('--dropout', type=float, default=0.25)

   # 在模型定义中使用
   class Net(nn.Module):
       def __init__(self, dropout=0.25):
           ...
           self.dropout1 = nn.Dropout(dropout)
   ```

2. **weight_decay** (约3行代码/模型):
   ```python
   # 在optimizer初始化中添加
   parser.add_argument('--weight-decay', type=float, default=0.0)

   optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   ```

3. **fp16/bf16** (约30行代码/模型):
   ```python
   parser.add_argument('--fp16', action='store_true')

   if args.fp16:
       scaler = torch.cuda.amp.GradScaler()
       with torch.cuda.amp.autocast():
           output = model(data)
           loss = F.nll_loss(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

---

## 2. 汇总统计

### 2.1 按超参数的支持情况

#### epochs (训练轮数)
- ✅ **完全支持**: 12/12 (100%)
- **无需修改**: 所有模型都支持
- **参数名变化**:
  - 大部分: `--epochs` 或 `-e`
  - Person_reID: `--total_epoch`
  - bug-localization: `--max_iter` (等效)

#### learning_rate (学习率)
- ✅ **完全支持**: 12/12 (100%)
- **无需修改**: 所有模型都支持
- **参数名变化**:
  - 大部分: `--lr`
  - VulBERTa: `--learning_rate`
  - bug-localization: 内置在sklearn中，不建议单独修改

#### seed (随机种子)
- ✅ **完全支持**: 7/12 (58%)
  - MRT-OAST, VulBERTa (2个), examples (4个)
- ❌ **需要简单修改**: 5/12 (42%)
  - bug-localization (1个) - ⭐ 极易
  - pytorch_resnet_cifar10 (1个) - ⭐ 极易
  - Person_reID (3个) - ⭐ 极易

**修改工作量**: 每个模型约5-15行代码

#### dropout
- ✅ **完全支持**: 5/12 (42%)
  - MRT-OAST, Person_reID (3个), examples部分
- ❌ **需要修改**: 6/12 (50%)
  - pytorch_resnet_cifar10 (1个) - ⭐⭐⭐ 中等
  - examples (4个) - ⭐⭐ 容易
  - VulBERTa (1个) - ⚠️ 困难（不推荐）
- ⚠️ **不适用**: 1/12 (8%)
  - bug-localization (sklearn模型)

**修改工作量**:
- 简单参数化: 10-20行代码/模型
- 修改模型结构: 50-80行代码/模型

#### weight_decay (权重衰减)
- ✅ **完全支持**: 4/12 (33%)
  - pytorch_resnet_cifar10, Person_reID (3个)
- ⚠️ **等效支持**: 1/12 (8%)
  - bug-localization (alpha参数)
- ❌ **需要修改**: 7/12 (58%)
  - MRT-OAST (1个) - ⭐⭐ 容易
  - VulBERTa (2个) - ⭐⭐ 容易
  - examples (4个) - ⭐ 极易

**修改工作量**: 每个模型约3-10行代码

#### 精度 (fp16/bf16)
- ✅ **完全支持**: 5/12 (42%)
  - pytorch_resnet_cifar10 (fp16), VulBERTa (2个, fp16), Person_reID (3个, fp16+bf16)
- ❌ **需要修改**: 7/12 (58%)
  - MRT-OAST (1个) - ⭐⭐ 容易
  - examples (4个) - ⭐⭐ 容易
- ⚠️ **不适用**: bug-localization (sklearn不支持)

**修改工作量**: 每个模型约30-40行代码

---

### 2.2 按仓库的总体修改难度

| 仓库 | 模型数 | 需修改超参数数 | 总体难度 | 估计工作量 |
|------|-------|--------------|---------|-----------|
| **MRT-OAST** | 1 | 2 (weight_decay, 精度) | ⭐⭐ 容易 | 0.5-1小时 |
| **bug-localization** | 1 | 1 (seed) | ⭐ 极易 | 10-15分钟 |
| **pytorch_resnet_cifar10** | 1 | 2 (seed, dropout) | ⭐⭐⭐ 中等 | 1-2小时 |
| **VulBERTa** | 2 | 1 (weight_decay) | ⭐⭐ 容易 | 10-20分钟 |
| **Person_reID** | 3 | 1 (seed) | ⭐ 极易 | 15-20分钟 |
| **examples** | 4 | 3 (dropout, weight_decay, 精度) | ⭐⭐ 容易 | 2-3小时 |

**总计**: 约4-7小时工作量可以使所有模型支持6个超参数

---

## 3. 推荐的实施策略

### 3.1 优先级划分

#### 第一优先级：无需修改（立即可用）
✅ **epochs**: 所有12个模型
✅ **learning_rate**: 所有12个模型

**可立即开始变异实验**。

#### 第二优先级：简单修改��1-2小时总工作量）
这些修改简单且影响大，建议优先实施：

1. **seed** (5个模型需要修改):
   - bug-localization (10分钟)
   - pytorch_resnet_cifar10 (10分钟)
   - Person_reID ×3 (20分钟)
   - **总计**: 约40分钟

2. **weight_decay** (VulBERTa):
   - VulBERTa ×2 (20分钟)

#### 第三优先级：中等修改（2-4小时总工作量）
需要一定代码修改，但价值高：

3. **weight_decay** (其他模型):
   - MRT-OAST (30分钟)
   - examples ×4 (40分钟)

4. **精度 (fp16/bf16)**:
   - MRT-OAST (1小时)
   - examples ×4 (2小时)

#### 第四优先级：复杂修改（可选）
修改复杂或影响有限，可根据需要选择：

5. **dropout** (examples参数化):
   - examples ×4 (1小时)

6. **dropout** (ResNet结构修改):
   - pytorch_resnet_cifar10 (1-2小时)
   - **注意**: ResNet默认无dropout是设计决策

7. **dropout** (VulBERTa):
   - **不推荐修改**：会影响预训练权重的使用

---

### 3.2 渐进式实施方案

#### 阶段1: 立即开始（0小时）
**可变超参数**: epochs, learning_rate
**适用模型**: 全部12个
**实验数量**: ~200次（如果使用标准方案）

**优势**:
- 无需任何代码修改
- 可以立即开始收集数据
- 验证实验流程

#### 阶段2: 快速扩展（1小时工作量）
**新增超参数**: seed, weight_decay (部分)
**修改模型**: 7个模型
- seed: bug-localization, pytorch_resnet_cifar10, Person_reID ×3
- weight_decay: VulBERTa ×2

**可变超参数**: epochs, learning_rate, seed, weight_decay (9/12模型)
**实验数量**: ~300次

**优势**:
- 工作量小（1小时）
- 覆盖75%的模型
- 增加重要的正则化参数

#### 阶段3: 全面覆盖（再3-4小时）
**新增超参数**: weight_decay (剩余), fp16/bf16
**修改模型**: 剩余5个模型

**可变超参数**: epochs, learning_rate, seed, dropout (部分), weight_decay, fp16/bf16
**实验数量**: ~400-500次

**优势**:
- 所有模型支持5-6个超参数
- 包含节能关键参数（fp16/bf16）
- 数据完整性好

#### 阶段4: 精细化（可选，再2-3小时）
**新增超参数**: dropout (参数化)
**修改模型**: examples ×4, pytorch_resnet_cifar10

**可变超参数**: 全部6个超参数完全覆盖

---

### 3.3 最小可行方案（推荐用于快速启动）

如果时间非常紧张，推荐这个最小方案：

**修改内容**:
1. seed支持 (3个仓库, 40分钟)
   - bug-localization
   - pytorch_resnet_cifar10
   - Person_reID

**结果**:
- 4个超参数可用: **epochs, learning_rate, seed, dropout (部分), weight_decay (部分), fp16/bf16 (部分)**
- 覆盖模型: 12/12 (epochs, lr, seed支持全部)
- 实验数量: 约200-300次
- 总工作量: **40分钟**

---

## 4. 修改难度分级说明

### ⭐ 极易 (10-20分钟/模型)
- 只需添加命令行参数
- 修改1-3处代码
- 不涉及训练逻辑
- 例如: 添加seed支持, weight_decay参数化

### ⭐⭐ 容易 (30分钟-1小时/模型)
- 需要修改训练逻辑
- 添加10-40行代码
- 不涉及模型结构
- 例如: 添加混合精度支持, weight_decay到Adam

### ⭐⭐⭐ 中等 (1-2小时/模型)
- 需要修改模型定义
- 添加50-100行代码
- 可能需要理解模型结构
- 例如: 为ResNet添加dropout

### ⭐⭐⭐⭐ 困难 (2-4小时/模型)
- 需要深入修改
- 可能影响预训练权重
- 需要仔细测试
- 例如: 修改预训练模型的dropout

---

## 5. 特殊注意事项

### 5.1 bug-localization (sklearn模型)

**限制**:
- 不支持GPU和混合精度
- learning_rate修改需谨慎
- dropout不适用

**建议**:
- 只变异: epochs (max_iter), weight_decay (alpha), seed
- 如果必须变异learning_rate，添加learning_rate_init参数

### 5.2 VulBERTa (预训练模型)

**限制**:
- dropout内置在预训练模型中，修改会影响性能
- weight_decay在TrainingArguments中默认存在

**建议**:
- 变异: epochs, learning_rate, seed, weight_decay, fp16
- **不要修改**: dropout

### 5.3 pytorch_resnet_cifar10

**设计决策**:
- ResNet原始设计不包含dropout
- 添加dropout可能不符合原始架构

**建议**:
- 如果需要研究正则化，使用weight_decay
- 如果必须添加dropout，在BasicBlock的conv层之后添加

### 5.4 Person_reID

**已有的正则化**:
- 默认dropout=0.5 (较大)
- 默认weight_decay=5e-4
- 支持多种loss (circle, triplet等)

**建议**:
- 这个模型对正则化已经配置较完善
- 重点变异: epochs, learning_rate, seed, dropout, fp16/bf16
- weight_decay已经可用，只需参数化

---

## 6. 推荐的修改顺序

基于修改难度和影响大小，推荐按以下顺序进行修改：

### 顺序1: seed支持 (40分钟)
**修改**:
- bug-localization
- pytorch_resnet_cifar10
- Person_reID ×3

**原因**:
- 修改最简单
- 对实验可重复性影响大
- 为后续实验建立基线

### 顺序2: weight_decay扩展 (1小时)
**修改**:
- MRT-OAST
- VulBERTa ×2
- examples ×4

**原因**:
- 修改简单
- 重要的正则化参数
- 对性能影响明显

### 顺序3: 精度支持 (3小时)
**修改**:
- MRT-OAST
- examples ×4

**原因**:
- 对能耗研究至关重要
- 修改相对标准化
- 性能影响可控

### 顺序4: dropout参数化 (1.5小时)
**修改**:
- examples ×4

**原因**:
- 已有dropout，只需参数化
- 修改较简单

### 顺序5: ResNet dropout (可选, 2小时)
**修改**:
- pytorch_resnet_cifar10

**原因**:
- 修改复杂
- 可能偏离原始设计
- 可以用weight_decay替代

---

## 7. 测试建议

修改后务必测试：

### 7.1 功能测试
```bash
# 测试每个超参数都可以成功修改
./train.sh --epochs 5 --lr 0.01 --seed 42 --dropout 0.3 \
           --weight-decay 0.001 --fp16

# 测试训练可以正常完成
# ��试性能指标可以正常输出
```

### 7.2 一致性测试
```bash
# 使用相同seed，多次运行应该得到相同结果
./train.sh --seed 42 > run1.log
./train.sh --seed 42 > run2.log
diff run1.log run2.log  # 应该基本一致
```

### 7.3 性能测试
```bash
# 修改前后性能对比
# 确保修改没有引入性能回归
```

---

## 8. 附录：代码模板

### A. 添加seed支持模板
```python
# 在参数解析中添加
parser.add_argument('--seed', default=42, type=int, help='random seed')

# 在训练开始前设置
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用
set_seed(args.seed)
```

### B. 添加weight_decay模板
```python
# 在参数解析中添加
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

# 在optimizer初始化时使用
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# 或
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
```

### C. 添加混合精度支持模板
```python
# 在参数解析中添加
parser.add_argument('--fp16', action='store_true', help='use fp16')
parser.add_argument('--bf16', action='store_true', help='use bf16')

# 在训练循环中
if args.fp16 or args.bf16:
    from torch.cuda.amp import autocast, GradScaler
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = GradScaler()

    for data, target in train_loader:
        optimizer.zero_grad()

        with autocast(dtype=dtype):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
else:
    # 原有训练代码
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### D. 添加dropout参数化模板
```python
# 在模型定义中
class Net(nn.Module):
    def __init__(self, dropout=0.25):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        ...

# 在主程序中
parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
model = Net(dropout=args.dropout)
```

---

## 9. 总结与建议

### 核心发现

1. **epochs和learning_rate** 已经完全支持，可以立即开始实验
2. **seed** 大部分支持，少数需要简单修改（40分钟）
3. **dropout, weight_decay, 精度** 需要不同程度的扩展，但都是可行的

### 推荐方案

**对于时间紧张的情况**:
- 立即使用 epochs 和 learning_rate 开始实验
- 花40分钟添加seed支持
- 使用已支持的dropout和weight_decay (部分模型)
- 总计: 40分钟准备 + 立即开始实验

**对于完整研究**:
- 分3个阶段，总计4-7小时
- 最终实现所有12个模型支持6个超参数
- 可以进行全面的超参数变异研究

### 成本效益分析

| 阶段 | 工作量 | 新增超参数 | 新增实验 | 性价比 |
|------|-------|-----------|---------|-------|
| 阶段1 | 0小时 | 2个 | ~200次 | ⭐⭐⭐⭐⭐ 极高 |
| 阶段2 | 1小时 | +2个 | ~300次 | ⭐⭐⭐⭐⭐ 极高 |
| 阶段3 | +4小时 | +1个关键 | ~500次 | ⭐⭐⭐⭐ 高 |
| 阶段4 | +3小时 | +1个 | ~650次 | ⭐⭐⭐ 中 |

**建议**: 先完成阶段1和2（1小时），根据实验进展决定是否继续阶段3和4。

---

**报告版本**: 1.0
**生成时间**: 2025-11-04
**建议后续行动**: 选择实施方案并开始代码修改
