# HRNet18 SSL证书验证失败修复方案

## 问题描述

hrnet18模型训练时，在下载HuggingFace预训练权重时遇到SSL证书验证失败：

```
httpcore.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1017)
```

**失败位置**：`Person_reID_baseline_pytorch/model.py:229`
```python
model_ft = timm.create_model('hrnet_w18', pretrained=True)
```

**影响范围**：
- Person_reID_baseline_pytorch/hrnet18 模型
- 所有需要从HuggingFace下载预训练权重的模型

**测试结果**：
- ���败次数：2/22 (9.1%)
- 失败实验ID：hrnet18_006, hrnet18_023

## 根本原因

系统中存在自签名SSL证书（可能是企业代理或防火墙），导致Python的`httpx`库在连接HuggingFace时无法验证SSL证书。

## 解决方案

### 方��1：禁用SSL验证（快速但不推荐）

**优点**：快速解决问题，无需额外配置
**缺点**：降低安全性，不适合生产环境

#### 方法1A：环境变量

在运行训练前设置环境变量：

```bash
# 在run.sh中添加
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
```

或在Python环境中：

```python
# 在model.py顶部添加
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 或者
import os
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['REQUESTS_CA_BUNDLE'] = ""
```

#### 方法1B：修改huggingface_hub配置

```python
# 在model.py中，在导入timm之前添加
import os
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

import timm
```

#### 方法1C：修改httpx客户端

在`Person_reID_baseline_pytorch/model.py`中修改：

```python
# 在文件顶部添加
import httpx
import ssl

# 创建不验证SSL的httpx客户端
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 在timm.create_model之前设置
import timm
# timm内部会使用httpx，需要在全局设置
```

### 方案2：配置系统SSL证书（推荐）

**优点**：安全，一次配置永久有效
**缺点**：需要系统管理员权限

#### 步骤：

1. **查找自签名证书**：
```bash
# 检查系统证书
ls /etc/ssl/certs/
```

2. **更新CA证书包**：
```bash
sudo apt-get update
sudo apt-get install ca-certificates
sudo update-ca-certificates
```

3. **导出证书（如果有代理）**：
```bash
# 从浏览器导出代理证书，然后：
sudo cp proxy-cert.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

4. **配置Python使用系统证书**：
```bash
# 在conda环境中
conda install -c conda-forge certifi
python -m pip install --upgrade certifi
```

### 方案3：使用本地镜像或预下载模型

**优点**：完全避免网络问题，速度更快
**缺点**：需要手动下载和管理模型文件

#### 步骤：

1. **手动下载HRNet18预训练权重**：

在有网络访问的机器上：
```bash
# 使用huggingface-cli下载
pip install huggingface_hub
huggingface-cli download timm/hrnet_w18.ms_in1k
```

或直接访问：https://huggingface.co/timm/hrnet_w18.ms_in1k/tree/main

2. **将模型文件复制到本地缓存**：

```bash
# 创建本地缓存目录
mkdir -p ~/.cache/huggingface/hub/models--timm--hrnet_w18.ms_in1k

# 复制下载的模型文件
cp -r downloaded_model/* ~/.cache/huggingface/hub/models--timm--hrnet_w18.ms_in1k/
```

3. **修改代码使用本地模型**：

```python
# 在model.py中修改：
# 原代码：
model_ft = timm.create_model('hrnet_w18', pretrained=True)

# 修改为：
model_ft = timm.create_model('hrnet_w18', pretrained=True,
                             checkpoint_path='/path/to/local/checkpoint.pth')
```

### 方案4：使用国内镜像源（针对中国用户）

如果是在中国，可以配置HuggingFace镜像：

```bash
# 设置环境变量使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在Python代码中
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 推荐实施方案

**短期快速修复**（用于完成当前测试）：
使用**方案1B**，在`Person_reID_baseline_pytorch/model.py`中添加环境变量禁用SSL验证。

**长期稳定方案**（用于生产环境）：
使用**方案3**，预下载所有需要的预训练模型到本地缓存。

## 实施步骤（短期修复）

1. **修改model.py**：

```bash
cd /home/green/energy_dl/nightly/repos/Person_reID_baseline_pytorch
```

在`model.py`文件顶部（在导入timm之前）添加：

```python
import os
# Disable SSL verification for HuggingFace downloads
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
```

2. **测试修复**：

```bash
cd /home/green/energy_dl/nightly
sudo python3 mutation.py settings/test_hrnet18_fix.json
```

其中`test_hrnet18_fix.json`应包含：

```json
{
  "experiment_name": "hrnet18_ssl_fix_test",
  "description": "Test hrnet18 with SSL fix",
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

3. **验证结果**：

检查训练日志确认模型成功加载：
```bash
tail -f results/run_*/Person_reID_baseline_pytorch_hrnet18_*/training.log
```

## 注意事项

1. **安全警告**：禁用SSL验证会降低安全性，仅在可信网络环境中使用
2. **权限问题**：修改系���证书需要sudo权限
3. **环境隔离**：建议在conda环境中进行修改，避免影响系统Python
4. **文档记录**：在项目README中记录SSL配置，方便其他开发者

## 相关文件

- 失败日志：`results/run_20251117_182512/Person_reID_baseline_pytorch_hrnet18_006/training.log`
- 模型代码：`repos/Person_reID_baseline_pytorch/model.py`
- 训练脚本：`repos/Person_reID_baseline_pytorch/train.py`

## 参考资料

- HuggingFace SSL配置：https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- Python SSL文档：https://docs.python.org/3/library/ssl.html
- timm库文档：https://huggingface.co/docs/timm/index
