# 训练优化建议

## 训练时长估算结果

基于NVIDIA GeForce RTX 3080 (10GB)的测试结果：

| 模型 | 批次时间 | Epoch时间 | 总训练时间(200 epochs) | 状态 |
|------|---------|----------|----------------------|------|
| ResNet20 | 0.008s | 0.1分钟 | 0.2小时 | ✓ 可正常训练 |
| ResNet32 | 0.013s | 0.1分钟 | 0.3小时 | ✓ 可正常训练 |
| ResNet44 | 0.017s | 0.1分钟 | 0.4小时 | ✓ 可正常训练 |
| ResNet56 | 0.021s | 0.1分钟 | 0.5小时 | ✓ 可正常训练 |
| ResNet110 | 0.042s | 0.3分钟 | 0.9小时 | ✓ 可正常训练 |
| ResNet1202 | N/A | N/A | N/A | ✗ 显存不足 (OOM) |

## 关键发现

### 1. 训练时长
所有能够成功运行的模型（ResNet20-110）训练时间均**小于1小时**，远低于2小时的阈值。因此：
- **不需要采取任何训练加速措施**
- 可以直接使用原始配置进行训练

### 2. ResNet1202的显存问题

**问题描述：**
ResNet1202在当前设备上无法训练，出现CUDA OOM错误：
- 模型参数量：19.4M
- 需要显存：约16GB（根据原始论文）
- 当前显存：10GB (RTX 3080)
- 批次大小：128（默认）

**解决方案：**

#### 方案1：减小批次大小（推荐）
修改trainer.py的批次大小参数，或运行时指定：

```bash
# 方法A：修改run.sh
python -u trainer.py --arch=resnet1202 --batch-size=32 --save-dir=save_resnet1202

# 方法B：尝试更小的批次
python -u trainer.py --arch=resnet1202 --batch-size=16 --save-dir=save_resnet1202

# 方法C：如果还不够，继续减小
python -u trainer.py --arch=resnet1202 --batch-size=8 --save-dir=save_resnet1202
```

**影响分析：**
- 减小批次大小会增加训练时间（每个epoch需要更多批次）
- 可能影响收敛性和最终精度（但通常影响不大）
- 估算训练时间（batch_size=32）：约2-3小时
- 估算训练时间（batch_size=16）：约4-5小时

#### 方案2：使用梯度累积
在保持有效批次大小的同时减少显存使用：

```python
# 修改trainer.py的train函数
# 设置累积步数
accumulation_steps = 4  # 有效批次=32*4=128

for i, (input, target) in enumerate(train_loader):
    output = model(input_var)
    loss = criterion(output, target_var)
    loss = loss / accumulation_steps  # 缩放loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 方案3：使用半精度训练（FP16）
启用混合精度训练可以减少显存使用：

```bash
python -u trainer.py --arch=resnet1202 --half --batch-size=64 --save-dir=save_resnet1202
```

**注意：** 原代码已支持--half参数（trainer.py:50-51, 112-114）

#### 方案4：梯度检查点（需要修改代码）
在resnet.py中添加梯度检查点以牺牲计算时间换取显存：

```python
import torch.utils.checkpoint as checkpoint

# 在BasicBlock的forward中使用
def forward(self, x):
    return checkpoint.checkpoint(self._forward, x)

def _forward(self, x):
    # 原来的forward逻辑
    ...
```

#### 方案5：跳过ResNet1202
如果上述方案都不可行，建议：
- 专注于训练ResNet20-110（都能正常运行）
- 使用仓库提供的预训练ResNet1202模型进行评估
- 预训练模型下载地址：https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202-f3b1deed.th

## 推荐的训练策略

### 对于ResNet20-110
直接使用原始run.sh脚本：
```bash
chmod +x run.sh
./run.sh
```

或单独训练某个模型：
```bash
python -u trainer.py --arch=resnet56 --save-dir=save_resnet56
```

### 对于ResNet1202
推荐尝试顺序：
1. **首选：** 使用batch_size=32 + 半精度
   ```bash
   python -u trainer.py --arch=resnet1202 --batch-size=32 --half --save-dir=save_resnet1202
   ```

2. **备选：** 仅减小batch_size
   ```bash
   python -u trainer.py --arch=resnet1202 --batch-size=16 --save-dir=save_resnet1202
   ```

3. **最后：** 使用预训练模型
   ```bash
   # 下载预训练模型
   wget https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202-f3b1deed.th

   # 评估模型
   python trainer.py --arch=resnet1202 --resume=resnet1202-f3b1deed.th --evaluate
   ```

## 额外优化建议（可选）

虽然当前训练时间已经很短，但如果需要进一步加速：

### 1. 减少epoch数量
对于快速验证，可以减少epoch数：
```bash
python -u trainer.py --arch=resnet56 --epochs=100 --save-dir=save_resnet56_100ep
```

### 2. 调整学习率调度
原配置在epoch 100和150降低学习率。如果减少epoch数，需要相应调整。

### 3. 使用更大的批次大小（对于小模型）
如果显存充足，可以增加批次大小以提高GPU利用率：
```bash
python -u trainer.py --arch=resnet20 --batch-size=256 --save-dir=save_resnet20_bs256
```

### 4. 启用cuDNN自动优化
代码已启用（trainer.py:86）：
```python
cudnn.benchmark = True
```

## 训练验证脚本

已创建的辅助脚本：
- `scripts/estimate_training_time.py`: 估算训练时间
- `scripts/verify_environment.py`: 验证环境配置（待创建）

## 总结

- **ResNet20-110**: 无需优化，训练时间<1小时
- **ResNet1202**: 需要调整配置（减小batch size或使用半精度）才能在10GB显存上运行
- **推荐**: 先训练ResNet20-110以验证环境，再按推荐策略尝试ResNet1202
