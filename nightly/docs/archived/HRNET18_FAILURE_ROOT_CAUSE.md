# 上次实验失败原因分析

## 问题回顾

**失败**: 2个hrnet18实验（实验6和17）
**成功**: 20个其他实验

---

## 失败的根本原因

上次运行很可能是这样的：

```bash
# 在screen session中运行
sudo python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json
```

### 两个关键问题：

#### ❌ 问题1: 没有设置环境变量

```bash
# 没有设置这些环境变量
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

**结果**: timm尝试从HuggingFace Hub下载模型

#### ❌ 问题2: 即使设置了，也没有用 `-E`

即使你设置了环境变量：
```bash
export HF_HUB_OFFLINE=1
sudo python3 mutation.py ...  # 没有-E
```

**结果**: `sudo` 会丢弃环境变量，Python脚本看不到 `HF_HUB_OFFLINE=1`

---

## 为什么其他模型成功了？

### ✅ 成功的模型

| 模型 | 预训练权重来源 | 为什么成功 |
|------|--------------|----------|
| mnist系列 | 无需预训练 | 从头训练 |
| resnet20 | 无需预训练 | 从头训练（CIFAR-10专用） |
| densenet121 | torchvision | torchvision有更好的缓存fallback机制 |
| pcb | torchvision (ResNet-50) | torchvision有更好的缓存fallback机制 |
| VulBERTa_mlp | 无需预训练/已在conda环境 | 可能使用本地模型或从头训练 |
| MRT-OAST | 无需预训练 | 从头训练 |
| bug-localization | 无需预训练 | 从头训练 |

### ❌ 失败的模型

| 模型 | 预训练权重来源 | 为什么失败 |
|------|--------------|----------|
| hrnet18 | timm (HuggingFace Hub) | timm**必须**明确设置离线模式，否则会强制联网 |

---

## 关键差异：torchvision vs timm

### torchvision (densenet, resnet-50)

```python
model = torchvision.models.densenet121(pretrained=True)
```

**行为**:
1. 先尝试联网下载最新版本
2. **如果失败，自动fallback到本地缓存** ✅
3. 如果本地也没有，再报错

**结果**: 即使没有 `HF_HUB_OFFLINE=1` 也能使用缓存

### timm (hrnet18)

```python
model = timm.create_model('hrnet_w18', pretrained=True)
```

**行为**:
1. 先尝试联网验证版本
2. **如果失败，直接报错** ❌
3. **不会自动使用本地缓存**（除非设置了 `HF_HUB_OFFLINE=1`）

**结果**: 没有 `HF_HUB_OFFLINE=1` 必定失败

---

## 完整的因果链

```
没有设置 HF_HUB_OFFLINE=1
           ↓
或者：设置了但使用 sudo（没有-E）
           ↓
环境变量被 sudo 丢弃
           ↓
Python脚本看不到 HF_HUB_OFFLINE=1
           ↓
timm 尝试联网下载 hrnet_w18
           ↓
遇到网络问题（SSL timeout / Connection reset）
           ↓
timm 不会fallback到本地缓存
           ↓
抛出 LocalEntryNotFoundError
           ↓
实验失败 ❌
```

---

## 正确的运行方式

### 方式1: 设置环境变量 + sudo -E（推荐）

```bash
# 1. 设置环境变量
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# 2. 使用 -E 保留环境变量
sudo -E python3 mutation.py -ec settings/config.json
     ↑
     关键！确保环境变量传递给Python
```

### 方式2: 内联设置（也可以）

```bash
sudo HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
     python3 mutation.py -ec settings/config.json
```

**注意**: 这种方式某些系统可能不工作（sudo的安全策略）

### 方式3: 使用脚本（最简单）

```bash
./scripts/fix_hrnet18.sh
```

脚本内部已经正确设置了环境变量和 `-E` 参数。

---

## 总结

**上次失败的原因**:

1. ❌ **主要原因**: 没有设置 `HF_HUB_OFFLINE=1` 环境变量
2. ❌ **次要原因**: 即使设置了，使用 `sudo` 而不是 `sudo -E` 会丢失环境变量
3. ⚠️ **加重因素**: timm 库不会自动fallback到本地缓存（不像torchvision）
4. 🌐 **触发因素**: 尝试联网时遇到网络问题（SSL/连接错误）

**修复方案**:

```bash
# 正确方式
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
sudo -E python3 mutation.py -ec settings/config.json
     ↑
     这个-E很关键！
```

**最简单**: 直接运行修复脚本
```bash
./scripts/fix_hrnet18.sh
```

---

**文档创建时间**: 2025-11-18 19:30
**版本**: v1.0
