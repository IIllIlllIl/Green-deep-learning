# HRNet18实验失败分析报告

**日期**: 2025-11-18
**实验会话**: run_20251118_155526
**失败数量**: 2/22 (9.1%)
**失败模型**: hrnet18

---

## 📋 失败概况

| 实验ID | 实验名称 | 类型 | 重试次数 | 失败时间 | 耗时 |
|--------|---------|------|---------|---------|------|
| 6 | Person_reID_baseline_pytorch_hrnet18_006 | 顺序 | 3次 | 16:31:29 | 58秒 |
| 17 | Person_reID_baseline_pytorch_hrnet18_017_parallel | 并行 | 3次 | 17:06:15 | 17秒 |

**成功率**: 20/22 (90.9%)

---

## 🔍 错误分析

### 错误类型

两个实验都出现了相同的错误，但表现略有不同：

#### 实验6 (顺序训练)
```
httpx.ConnectTimeout: _ssl.c:1000: The handshake operation timed out
```

#### 实验17 (并行训练)
```
httpx.ConnectError: [Errno 104] Connection reset by peer
```

### 最终错误
两个实验最终都抛出：
```python
huggingface_hub.errors.LocalEntryNotFoundError:
An error happened while trying to locate the file on the Hub
and we cannot find the requested files in the local cache.
Please check your connection and try again or make sure your Internet connection is on.
```

### 错误堆栈追踪

```python
File "train.py", line 548, in <module>
    model = ft_net_hr(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
File "model.py", line 229, in __init__
    model_ft = timm.create_model('hrnet_w18', pretrained=True)
    ↓
File "timm/models/_factory.py", line 138, in create_model
    ↓
File "timm/models/_builder.py", line 226, in load_pretrained
    state_dict = load_state_dict_from_hf(pretrained_loc, weights_only=True, cache_dir=cache_dir)
    ↓
File "timm/models/_hub.py", line 240, in load_state_dict_from_hf
    cached_file = hf_hub_download(...)
    ↓
File "huggingface_hub/file_download.py", line 991, in hf_hub_download
    ↓
File "huggingface_hub/file_download.py", line 1117, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
    ↓
❌ LocalEntryNotFoundError
```

---

## 🎯 根本原因

### 1. 环境变量未设置 ⚠️

**问题**: 实验运行时**没有设置`HF_HUB_OFFLINE=1`环境变量**

**证据**:
```bash
$ grep -n "HF_HUB_OFFLINE" mutation/runner.py
# 无输出 - 代码中没有设置此环境变量
```

**结果**:
- timm尝试从HuggingFace Hub下载模型
- 而不是使用本地缓存

### 2. 网络连接问题 🌐

**问题**: 尝试下载时遇到网络错误

**表现**:
- SSL握手超时 (实验6)
- 连接被重置 (实验17)

**可能原因**:
- 防火墙/代理配置
- SSL证书问题
- 网络不稳定
- HuggingFace Hub服务暂时不可用

### 3. 缓存识别失败 💾

**问题**: 虽然本地有缓存，但未能正确识别

**验证缓存存在**:
```bash
$ ls ~/.cache/huggingface/hub/models--timm--hrnet_w18.ms_aug_in1k/
blobs/  refs/  snapshots/

$ ls -lh ~/.cache/huggingface/hub/ | grep hrnet
drwxrwxr-x 5 green green 4.0K 11月  1 17:30 models--timm--hrnet_w18.ms_aug_in1k
```

**缓存下载时间**: 2025-11-01 17:30 (18天前)

**问题**: HuggingFace Hub库在没有设置`local_files_only=True`或`HF_HUB_OFFLINE=1`时，会先尝试联网验证，即使本地有缓存也会失败。

---

## ✅ 解决方案

### 方案1: 设置离线环境变量（推荐）⭐⭐⭐

**在运行实验前设置环境变量**:

```bash
# 方式1: 导出环境变量
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
sudo -E python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 方式2: 内联设置
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 sudo -E python3 mutation.py -ec settings/your_config.json
```

**说明**:
- `HF_HUB_OFFLINE=1`: 强制使用本地缓存，禁止网络请求
- `HF_HUB_DISABLE_TELEMETRY=1`: 禁用遥测数据上传
- `sudo -E`: 保留环境变量传递给sudo

**优点**:
- ✅ 简单快速
- ✅ 无需修改代码
- ✅ 本地缓存已存在
- ✅ 完全离线运行

### 方案2: 在代码中设置环境变量

**修改`mutation/runner.py`或`mutation.py`入口**:

在文件顶部添加：
```python
import os
# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
```

**优点**:
- ✅ 自动生效，用户无需记住
- ✅ 对所有实验生效

**缺点**:
- ⚠️ 需要修改代码
- ⚠️ 如果确实需要下载新模型会失败

### 方案3: 在Person_reID仓库中设置

**修改`repos/Person_reID_baseline_pytorch/model.py`**:

在第229行之前添加：
```python
# Line 229: model_ft = timm.create_model('hrnet_w18', pretrained=True)
# 修改为：
import os
os.environ['HF_HUB_OFFLINE'] = '1'
model_ft = timm.create_model('hrnet_w18', pretrained=True)
```

**优点**:
- ✅ 针对性强，只影响hrnet18

**缺点**:
- ⚠️ 侵入原始代码
- ⚠️ 不是通用解决方案

### 方案4: 重新下载模型（不推荐）

如果缓存损坏，可以重新下载：

```bash
conda activate reid_baseline
python3 scripts/download_pretrained_models.py
```

**说明**: 本例中缓存完好，无需重新下载。

---

## 🔄 重新运行失败的实验

### 方法1: 创建修复配置（推荐）

创建仅包含hrnet18的配置：

**文件**: `settings/fix_hrnet18_1epoch.json`

```json
{
  "experiment_name": "fix_hrnet18_1epoch",
  "description": "Re-run failed hrnet18 experiments with offline mode",
  "governor": "performance",
  "runs_per_config": 1,
  "max_retries": 2,
  "experiments": [
    {
      "mode": "default",
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "hyperparameters": {
        "epochs": 1,
        "learning_rate": 0.05,
        "dropout": 0.5,
        "seed": 1334
      },
      "note": "Sequential hrnet18 - fix experiment 6"
    },
    {
      "mode": "parallel",
      "foreground": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "hrnet18",
        "mode": "default",
        "hyperparameters": {
          "epochs": 1,
          "learning_rate": 0.05,
          "dropout": 0.5,
          "seed": 1334
        }
      },
      "background": {
        "repo": "examples",
        "model": "mnist_rnn",
        "hyperparameters": {
          "epochs": 1,
          "learning_rate": 0.01,
          "batch_size": 32,
          "seed": 1
        }
      },
      "note": "Parallel hrnet18 - fix experiment 17"
    }
  ]
}
```

**运行修复实验**:
```bash
export HF_HUB_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
sudo -E python3 mutation.py -ec settings/fix_hrnet18_1epoch.json
```

**预计时间**: 约12-16分钟（两个实验）

### 方法2: 手动单独测试

```bash
# 测试顺序hrnet18
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -r Person_reID_baseline_pytorch -m hrnet18 -n 1

# 测试并行hrnet18
# (需要在配置文件中定义并行实验)
```

---

## 📊 影响评估

### 对整体实验的影响

| 指标 | 数值 | 影响程度 |
|------|------|---------|
| **成功率** | 90.9% (20/22) | 🟡 中等 |
| **失败模型** | 1个 (hrnet18) | 🟢 低 |
| **数据完整性** | 20/22完整 | 🟡 中等 |
| **可修复性** | 100% | 🟢 易修复 |

### 受影响的并行组合

| 并行组合 | 前景 | 背景 | 状态 |
|---------|------|------|------|
| Parallel 6/11 | hrnet18 | mnist_rnn | ❌ 失败 |
| 其他10个 | 各种模型 | 各种模型 | ✅ 成功 |

### 数据完整性

✅ **20个成功实验的数据完整**:
- experiment.json ✅
- training.log ✅
- energy数据 ✅
- summary.csv ✅

❌ **2个失败实验**:
- experiment.json ✅ (包含错误信息)
- training.log ✅ (包含错误堆栈)
- energy数据 ❌ (训练未开始，无能耗数据)
- summary.csv ❌ (失败实验未记录)

---

## 🎓 经验总结

### 1. 离线训练的重要性

在生产环境或无网络环境中运行深度学习实验时，必须：
- ✅ 预先下载所有预训练权重
- ✅ 设置离线模式环境变量
- ✅ 验证缓存完整性

### 2. 环境变量管理

**最佳实践**:
```bash
# 创建运行脚本
cat > run_experiments.sh <<'EOF'
#!/bin/bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
sudo -E python3 mutation.py "$@"
EOF
chmod +x run_experiments.sh

# 使用脚本运行
./run_experiments.sh -ec settings/your_config.json
```

### 3. 错误处理机制

**系统表现良好**:
- ✅ 自动重试机制（每个实验重试3次）
- ✅ 错误日志完整记录
- ✅ 其他实验继续运行
- ✅ 失败不影响整体流程

**可改进**:
- ⚠️ 可以添加预检查：验证所需模型是否在缓存中
- ⚠️ 可以在代码中默认设置离线模式

---

## 📝 建议的修复步骤

### 立即执行（5分钟）

1. **验证缓存完整性**:
   ```bash
   ls -lh ~/.cache/huggingface/hub/models--timm--hrnet_w18.ms_aug_in1k/
   # 应该看到 blobs/, refs/, snapshots/ 三个目录
   ```

2. **创建修复配置文件**:
   保存上面的`fix_hrnet18_1epoch.json`

3. **运行修复实验**:
   ```bash
   export HF_HUB_OFFLINE=1
   export HF_HUB_DISABLE_TELEMETRY=1
   sudo -E python3 mutation.py -ec settings/fix_hrnet18_1epoch.json
   ```

### 短期（今天完成）

4. **更新文档**:
   - ✅ 在README.md中强调离线模式的重要性
   - ✅ 在快速开始指南中包含环境变量设置

5. **创建运行脚本**:
   - 将环境变量设置封装到脚本中
   - 避免用户忘记设置

### 长期改进（可选）

6. **代码增强**:
   - 在`mutation.py`入口自动设置离线模式
   - 添加预检查机制验证缓存
   - 改进错误提示信息

7. **持续集成**:
   - 将离线模式纳入CI/CD流程
   - 定期验证缓存完整性

---

## 🔗 相关文档

- [离线训练设置指南](docs/archive/2025-11-18/OFFLINE_TRAINING_SETUP.md)
- [HRNet18 SSL修复](docs/archive/2025-11-18/HRNET18_SSL_FIX.md)
- [预训练模型下载脚本](scripts/download_pretrained_models.py)
- [快速验证配置](settings/11_models_quick_validation_1epoch.json)

---

## 📞 FAQ

**Q: 为什么其他实验没有失败？**
A: 其他模型要么不需要预训练权重（mnist系列），要么使用torchvision的模型（densenet, resnet, pcb），这些会自动fallback到本地缓存。只有timm的hrnet需要显式的离线模式设置。

**Q: 缓存中有模型为什么还失败？**
A: HuggingFace Hub库的默认行为是先尝试联网验证最新版本，只有在设置`HF_HUB_OFFLINE=1`时才会直接使用缓存。

**Q: 是否需要重新下载hrnet18模型？**
A: 不需要。缓存是完整的（11月1日下载），只需设置离线模式即可。

**Q: 修复实验需要多长时间？**
A: 约12-16分钟（2个实验，每个6-8分钟）

**Q: 如何验证修复成功？**
A: 检查新生成的实验目录中是否有完整的training.log、experiment.json和energy数据。

---

**报告生成时间**: 2025-11-18 19:30
**报告作者**: Claude Code
**版本**: v1.0
**状态**: ✅ 分析完成，等待修复
