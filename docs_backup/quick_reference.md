# 超参数变异快速参考指南

**完整报告**: [hyperparameter_analysis.md](./hyperparameter_analysis.md)

---

## 一、核心发现

### 1.1 完全通用超参数（12/12模型）

| 超参数 | 说明 | 修改难度 |
|-------|------|---------|
| `epochs` | 训练轮数 | ⭐ 极易 |
| `batch_size` | 批次大小 | ⭐ 极易 |
| `learning_rate` | 学习率 | ⭐ 极易 |
| `seed` | 随机种子 | ⭐ 极易 |

### 1.2 高频通用超参数

| 超参数 | 当前支持 | 扩展后支持 | 修改难度 |
|-------|---------|-----------|---------|
| `dropout` | 5/12 | 12/12 | ⭐⭐ 容易 |
| `weight_decay` | 4/12 | 12/12 | ⭐⭐ 容易 |
| `fp16/bf16` | 5/12 | 12/12 | ⭐⭐ 容易 |
| `momentum` | 2/12 | 2/12 | ⭐ 极易 |

---

## 二、推荐的变异超参数

### 2.1 第一优先级（必须包含）

#### epochs
```python
variants = {
    'MRT-OAST': [5, 7, 10, 15, 20],
    'pytorch_resnet_cifar10': [100, 150, 200, 250, 300],
    'VulBERTa-MLP': [5, 8, 10, 15, 20],
    'Person_reID': [30, 45, 60, 80, 100],
    'examples-mnist': [7, 10, 14, 20],
}
```
- **性能影响**: 高
- **能耗影响**: 高（线性相关）
- **推荐原因**: 最直接影响性能和能耗的超参数

#### batch_size
```python
variants = {
    'MRT-OAST': [32, 48, 64, 96, 128],
    'pytorch_resnet_cifar10': [64, 96, 128, 192],
    'VulBERTa-MLP': [1, 2, 4],  # 显存限制
    'Person_reID': [12, 18, 24, 32],
    'examples-mnist': [16, 32, 64, 128],
}
```
- **性能影响**: 中
- **能耗影响**: 中（复杂关系）
- **注意**: 修改batch_size建议同步调整learning_rate

#### learning_rate
```python
# 对数尺度变异
variants = {
    'MRT-OAST': [5e-5, 1e-4, 2e-4, 5e-4],  # Adam
    'pytorch_resnet_cifar10': [0.05, 0.075, 0.1, 0.15, 0.2],  # SGD
    'VulBERTa-MLP': [1e-5, 3e-5, 5e-5, 1e-4],  # AdamW
    'Person_reID': [0.02, 0.035, 0.05, 0.08],  # SGD
}
```
- **性能影响**: 高
- **能耗影响**: 中
- **警告**: 最敏感的超参数，不当值可能导致训练失败

#### fp16/bf16（混合精度）
```python
variants = ['fp32', 'fp16', 'bf16']  # bf16更稳定
```
- **性能影响**: 低（-0.5%到+0.5%）
- **能耗影响**: 高（节省15-40%）
- **强烈推荐**: 对能耗研究非常重要

### 2.2 第二优先级（推荐包含）

#### dropout
```python
variants = [0.0, 0.1, 0.2, 0.3, 0.5]
```
- **性能影响**: 中到高
- **能耗影响**: 低

#### weight_decay
```python
variants = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]  # 对数尺度
```
- **性能影响**: 中
- **能耗影响**: 极低

#### seed
```python
variants = [42, 123, 456, 789, 1024]
```
- **性能影响**: 低到中（±1-5%）
- **能耗影响**: 极低
- **用途**: 评估性能方差，验证可重复性

---

## 三、实验方案

### 3.1 最小方案（资源受限）

**推荐**: 方案B - 关键参数组合

| 配置 | 说明 | 实验数 |
|------|------|--------|
| 1. 默认配置 | 基线 | 12 |
| 2. 减少epochs + 增大batch | 快速训练 | 12 |
| 3. 增加epochs + 降低lr | 追求性能 | 12 |
| 4. 默认 + fp16 | 节能 | 12 |
| 5. 默认 + 增强正则化 | 防止过拟合 | 12 |

**总计**: 60次实验

### 3.2 标准方案（一般情况）

**第一阶段**: 单参数扫描
- epochs: 5值 × 12模型 = 60次
- batch_size: 4值 × 12模型 = 48次
- learning_rate: 4值 × 12模型 = 48次
- fp16/bf16: 3值 × 12模型 = 36次

**第二阶段**: 部分第二优先级
- dropout: 5值 × 12模型 = 60次
- weight_decay: 5值 × 12模型 = 60次

**总计**: 约312次实验

### 3.3 完整方案（资源充足）

- 第一阶段: 单参数扫描（~400次）
- 第二阶段: 最优组合验证（~150次）
- 第三阶段: 消融研究（~100-200次）

**总计**: 约650-750次实验

---

## 四、影响评估总结

### 4.1 高性能影响 + 高能耗影响（优先变异）

| 超参数 | 性能 | 能耗 | 推荐度 |
|-------|------|------|-------|
| `epochs` | 高 | 高（线性） | ⭐⭐⭐⭐⭐ |
| `learning_rate` | 高 | 中 | ⭐⭐⭐⭐ |
| `layers` (Transformer) | 高 | 高（线性） | ⭐⭐⭐⭐ |
| `d_model` (Transformer) | 高 | 高（平方） | ⭐⭐⭐⭐ |

### 4.2 低性能影响 + 高能耗节省（节能优化）

| 超参数 | 性能 | 能耗 | 推荐度 |
|-------|------|------|-------|
| `fp16/bf16` | 低 | 节省15-40% | ⭐⭐⭐⭐⭐ |
| `early_stopping` | 中 | 节省可变 | ⭐⭐⭐⭐ |

### 4.3 高性能影响 + 低能耗影响（性能调优）

| 超参数 | 性能 | 能耗 | 推荐度 |
|-------|------|------|-------|
| `dropout` | 中-高 | 低 | ⭐⭐⭐⭐ |
| `weight_decay` | 中 | 极低 | ⭐⭐⭐ |

---

## 五、12个模型快速参考

### 5.1 模型列表

| # | 仓库 | 模型 | 训练指令 |
|---|------|------|---------|
| 1 | MRT-OAST | MRT-OAST | `./train.sh` |
| 2 | bug-localization | DNN | `./train.sh` |
| 3 | pytorch_resnet_cifar10 | ResNet20 | `./train.sh` |
| 4 | VulBERTa | MLP | `./train.sh -n mlp` |
| 5 | VulBERTa | CNN | `./train.sh -n cnn` |
| 6 | Person_reID | densenet121 | `./train.sh -n densenet121` |
| 7 | Person_reID | hrnet18 | `./train.sh -n hrnet18` |
| 8 | Person_reID | pcb | `./train.sh -n pcb` |
| 9 | examples | mnist | `./train.sh -n mnist` |
| 10 | examples | mnist_rnn | `./train.sh -n mnist_rnn` |
| 11 | examples | mnist_ff | `./train.sh -n mnist_ff` |
| 12 | examples | siamese | `./train.sh -n siamese` |

### 5.2 训练命令示例

#### 变异epochs
```bash
# MRT-OAST
cd models/MRT-OAST
./train.sh --epochs 15 2>&1 | tee training.log

# pytorch_resnet_cifar10
cd models/pytorch_resnet_cifar10
./train.sh -e 150 2>&1 | tee training.log

# Person_reID
cd models/Person_reID_baseline_pytorch
./train.sh -n densenet121 --total_epoch 80 2>&1 | tee training.log
```

#### 变异batch_size
```bash
# MRT-OAST
./train.sh --batch-size 48 2>&1 | tee training.log

# VulBERTa-MLP
cd models/VulBERTa
./train.sh -n mlp -d devign --batch_size 4 2>&1 | tee training.log
```

#### 变异learning_rate
```bash
# MRT-OAST
./train.sh --lr 0.0002 2>&1 | tee training.log

# pytorch_resnet_cifar10
./train.sh --lr 0.05 2>&1 | tee training.log
```

#### 组合变异
```bash
# 快速训练配置
./train.sh --epochs 5 --batch-size 96 --lr 0.0002 2>&1 | tee training.log

# 追求性能配置
./train.sh --epochs 20 --batch-size 48 --lr 0.00005 --dropout 0.3 2>&1 | tee training.log

# 节能配置
./train.sh --bf16 --batch-size 64 2>&1 | tee training.log
```

---

## 六、代码扩展建议

### 6.1 高优先级扩展（1天工作量）

#### 扩展fp16/bf16到所有模型

**MRT-OAST示例**:
```python
# 在train.sh中添加参数
parser.add_argument("--fp16", action="store_true", help="use fp16")
parser.add_argument("--bf16", action="store_true", help="use bf16")

# 在main_batch.py中添加
if args.fp16 or args.bf16:
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环中
    with torch.cuda.amp.autocast(dtype=dtype):
        outputs = model(inputs)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 扩展dropout到所有模型

**pytorch_resnet_cifar10示例**:
```python
# 在resnet.py的BasicBlock中添加
class BasicBlock(nn.Module):
    def __init__(self, ..., dropout=0.0):
        # ...
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv1(x)
        # ...
        if self.dropout:
            out = self.dropout(out)
        return out
```

### 6.2 中优先级扩展（1-3天工作量）

#### 添加早停机制

**通用模板**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -float('inf')

    def __call__(self, metric):
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
            return False  # 不停止
        else:
            self.counter += 1
            return self.counter >= self.patience  # 达到耐心值则停止

# 使用
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    # 训练...
    val_metric = evaluate()
    if early_stopping(val_metric):
        print(f"Early stopping at epoch {epoch}")
        break
```

#### 添加学习率调度器选择

**通用模板**:
```python
def get_scheduler(optimizer, args):
    if args.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.lr_scheduler == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay_gamma)
    else:
        return None
```

---

## 七、实施检查清单

### 7.1 准备阶段

- [ ] 确认所有模型可以正常训练（使用默认参数）
- [ ] 准备足够的GPU资源
- [ ] 设置实验管理系统（如TensorBoard、WandB）
- [ ] 创建实验结果记录表格
- [ ] 准备自动化脚本

### 7.2 执行阶段

- [ ] 运行基线实验（默认参数）
- [ ] 执行第一优先级单参数扫描
- [ ] 记录每次实验的性能和能耗数据
- [ ] 定期备份实验结果
- [ ] 分析初步结果，调整实验计划

### 7.3 分析阶段

- [ ] 汇总所有实验数据
- [ ] 绘制性能-能耗曲线
- [ ] 识别最优超参数配置
- [ ] 分析超参数交互效应
- [ ] 撰写实验报告

---

## 八、常见问题

### Q1: 为什么不同模型的learning_rate差异这么大？

**A**: learning_rate高度依赖于：
1. **优化器类型**: SGD通常需要较大lr（0.01-0.1），Adam需要较小lr（1e-5到1e-3）
2. **模型大小**: 大模型通常需要更小的lr
3. **batch_size**: 大batch通常需要更大的lr
4. **数据集**: 不同数据集的scale不同

### Q2: 如何确定每个超参数的搜索范围？

**A**: 建议策略：
1. 从默认值开始
2. 对数尺度搜索: 0.1x, 0.5x, 1x, 2x, 5x, 10x
3. 观察训练曲线判断方向
4. 逐步缩小范围

### Q3: batch_size修改后需要同步调整learning_rate吗？

**A**: 建议调整。经验法则：
```
new_lr = old_lr * sqrt(new_batch / old_batch)
```
例如：batch从32变为64，lr应从0.01变为0.0141

### Q4: fp16训练会损失性能吗？

**A**:
- 现代混合精度实现（torch.cuda.amp）通常性能损失<1%
- bf16比fp16更稳定，推荐优先使用
- 某些模型可能完全不受影响

### Q5: seed的影响有多大？

**A**:
- 小数据集: 性能波动±2-5%
- 大数据集: 性能波动±0.5-2%
- 建议使用3-5个不同seed，报告均值±标准差

---

## 九、联系和支持

**完整报告**: `docs/hyperparameter_analysis.md`

**问题反馈**: 请在项目issue中提出

**更新日志**:
- v1.0 (2025-11-04): 初始版本

---

**祝实验顺利！**
