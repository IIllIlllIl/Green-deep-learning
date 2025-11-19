# 离线训练模式设置指南

## 概述

本指南说明如何在无网络环境下运行训练实验。主要包括：
1. 提前下载所有预训练模型权重
2. 配置离线模式
3. 验证离线训练可用性

## 为什么需要离线模式？

能耗测量实验通常在隔离的测试环境中运行，这些环境可能：
- 无互联网连接
- 有防火墙限制（如hrnet18的SSL证书问题）
- 需要可重复的实验环境（避免网络波动）

## 预下载模型权重

### 步骤1：在有网络的环境中下载模型

```bash
cd /home/green/energy_dl/nightly

# 确保使用正确的conda环境
conda activate reid_baseline

# 运行下载脚本
python3 scripts/download_pretrained_models.py
```

脚本会下载以下模型：

| 模型类型 | 模型名称 | 用途 | 大小 |
|---------|---------|------|------|
| timm | hrnet_w18 | Person_reID_baseline_pytorch/hrnet18 | ~300 MB |
| torchvision | resnet50 | Person_reID_baseline_pytorch/resnet50 | ~100 MB |
| torchvision | densenet121 | Person_reID_baseline_pytorch/densenet121 | ~30 MB |

**总大小**：约 500 MB - 1 GB（含依赖）

### 步骤2：验证下载完成

下载完成后，脚本会自动验证并测试离线加载。检查输出确认：

```
✅ Successfully loaded hrnet_w18 in offline mode
✅ Successfully loaded resnet50 in offline mode
✅ Successfully loaded densenet121 in offline mode
```

### 步骤3：备份缓存目录（可选）

如果需要在不同机器间传输：

```bash
# 打包HuggingFace缓存
cd ~/.cache
tar czf huggingface_cache.tar.gz huggingface/

# 打包PyTorch缓存
tar czf torch_cache.tar.gz torch/

# 传输到目标机器
scp huggingface_cache.tar.gz torch_cache.tar.gz target_machine:~/.cache/

# 在目标机器上解压
cd ~/.cache
tar xzf huggingface_cache.tar.gz
tar xzf torch_cache.tar.gz
```

## 配置离线模式

### 方法1：环境变量（推荐）

在运行实验前设置环境变量：

```bash
# 启用HuggingFace离线模式
export HF_HUB_OFFLINE=1

# 禁用HuggingFace更新检查
export HF_HUB_DISABLE_TELEMETRY=1

# 运行实验
sudo -E python3 mutation.py settings/your_config.json
```

注意：使用`sudo -E`保留环境变量。

### 方法2：修改代码

在`repos/Person_reID_baseline_pytorch/model.py`文件顶部添加：

```python
import os

# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 其余导入...
import timm
import torch
```

### 方法3：创建离线运行脚本

创建`scripts/run_offline.sh`:

```bash
#!/bin/bash

# 设置离线模式环境变量
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1

# 禁用pip更新检查
export PIP_NO_INDEX=1

# 运行实验
sudo -E python3 mutation.py "$@"
```

使用：

```bash
chmod +x scripts/run_offline.sh
./scripts/run_offline.sh settings/your_config.json
```

## 验证离线模式

### 测试1：运行hrnet18单次训练

创建测试配置`settings/test_offline_hrnet18.json`:

```json
{
  "experiment_name": "offline_hrnet18_test",
  "description": "Test hrnet18 in offline mode",
  "governor": "performance",
  "runs_per_config": 1,
  "max_retries": 0,
  "experiments": [
    {
      "mode": "default",
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "hyperparameters": {
        "epochs": 1,
        "batch_size": 24,
        "learning_rate": 0.05,
        "dropout": 0.5,
        "seed": 1334
      }
    }
  ]
}
```

运行测试：

```bash
# 设置离线模式
export HF_HUB_OFFLINE=1

# 运行测试
sudo -E python3 mutation.py settings/test_offline_hrnet18.json

# 检查日志
tail -100 results/run_*/Person_reID_baseline_pytorch_hrnet18_*/training.log
```

**成功标志**：
- 日志中显示 "Using seed: 1334"
- 没有SSL证书错误
- 模型成功加载："model_ft = timm.create_model('hrnet_w18', pretrained=True)"

**失败标志**：
- 出现 "HTTPError" 或 "ConnectError"
- 提示 "Offline mode is enabled" 但找不到缓存
- SSL证书验证失败

### 测试2：完全断网测试（可选）

```bash
# 临时禁用网络接口（需要root权限）
sudo ip link set <interface> down

# 运行实验
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py settings/test_offline_hrnet18.json

# 恢复网络
sudo ip link set <interface> up
```

## 常见问题

### Q1: 离线模式下仍然报SSL错书错误

**原因**：代码中可能有其他网络请求（如数据集下载）

**解决**：
1. 检查日志确认是哪个操作需要网络
2. 提前下载数据集到本地
3. 修改代码使用本地数据集路径

### Q2: 提示找不到缓存的模型

**原因**：
- 缓存路径不正确
- 用户权限问题（sudo运行时使用root的缓存）

**解决**：
```bash
# 检查缓存位置
echo $HOME/.cache/huggingface
echo $HOME/.cache/torch

# 如果使用sudo，确保缓存在正确位置
sudo ls -la ~/.cache/huggingface
sudo ls -la /home/green/.cache/huggingface

# 如果需要，复制缓存到root用户
sudo cp -r /home/green/.cache/huggingface /root/.cache/
sudo cp -r /home/green/.cache/torch /root/.cache/
```

### Q3: 下载脚本失败

**原因**：网络问题或依赖包未安装

**解决**：
```bash
# 检查网络连接
ping huggingface.co

# 检查SSL证书
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"

# 更新pip和依赖
pip install --upgrade pip
pip install --upgrade timm torchvision torch
```

### Q4: 模型版本不匹配

**原因**：下载环境和运行环境的库版本不同

**解决**：
```bash
# 在两个环境中使用相同的依赖版本
pip freeze > requirements.txt

# 在目标环境中安装相同版本
pip install -r requirements.txt
```

## 最佳实践

1. **在联网环境中**：
   - 运行`scripts/download_pretrained_models.py`下载所有模型
   - 备份缓存目录
   - 验证离线加载成功

2. **在离线环境中**：
   - 恢复缓存目录
   - 设置`HF_HUB_OFFLINE=1`
   - 使用`sudo -E`保留环境变量
   - 在测试配置上先验证

3. **长期维护**：
   - 定期更新预训练模型（如有新版本）
   - 记录所有模型的版本号
   - 保持下载环境和运行环境的依赖一致

## 自动化脚本

### 一键设置离线环境

创建`scripts/setup_offline.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Setting up offline training environment"
echo "=========================================="

# 1. 下载预训练模型
echo "Step 1: Downloading pretrained models..."
python3 scripts/download_pretrained_models.py

# 2. 备份缓存
echo "Step 2: Backing up cache..."
cd ~/.cache
tar czf ~/pretrained_models_backup.tar.gz huggingface/ torch/

# 3. 测试离线模式
echo "Step 3: Testing offline mode..."
export HF_HUB_OFFLINE=1
python3 -c "import timm; m=timm.create_model('hrnet_w18', pretrained=True); print('✅ Offline test passed')"

echo "=========================================="
echo "✅ Offline environment setup complete!"
echo "=========================================="
echo "Backup location: ~/pretrained_models_backup.tar.gz"
echo "To use: export HF_HUB_OFFLINE=1"
```

### 一键恢复离线环境

创建`scripts/restore_offline.sh`:

```bash
#!/bin/bash
set -e

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "=========================================="
echo "Restoring offline training environment"
echo "=========================================="

# 解压缓存
echo "Extracting cache from: $BACKUP_FILE"
cd ~/.cache
tar xzf "$BACKUP_FILE"

# 验证
echo "Verifying cache..."
ls -lh ~/.cache/huggingface/
ls -lh ~/.cache/torch/

echo "=========================================="
echo "✅ Offline environment restored!"
echo "=========================================="
```

## 相关文档

- [HRNet18 SSL修复方案](HRNET18_SSL_FIX.md)
- [预训练模型下载脚本](../scripts/download_pretrained_models.py)
- [HuggingFace离线模式文档](https://huggingface.co/docs/huggingface_hub/guides/offline)

## 支持的模型列表

当前支持离线训练的模型：

✅ **Person_reID_baseline_pytorch**
- resnet50
- densenet121
- hrnet18
- pcb (基于resnet50)

✅ **其他模型**
- MRT-OAST (无需预训练权重)
- bug-localization-by-dnn-and-rvsm (无需预训练权重)
- pytorch_resnet_cifar10 (无需预训练权重)
- VulBERTa (需要BERT权重，待添加)
- examples/* (无需预训练权重)

如需添加其他模型，请修改`scripts/download_pretrained_models.py`。
