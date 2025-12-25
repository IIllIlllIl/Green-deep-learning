# Environment Files Summary

## 📁 已创建的文件

### Conda环境配置文件 (.yml)

✅ **mutation_runner.yml** (435 bytes)
   - 用于运行 mutation_runner.py 主程序
   - Python 3.10 + 标准库
   - 无特殊依赖

✅ **pytorch_resnet_cifar10.yml** (1.5K)
   - 用于训练 ResNet CIFAR-10 模型
   - Python 3.10 + PyTorch + TorchVision

✅ **vulberta.yml** (4.1K)
   - 用于训练 VulBERTa 漏洞检测模型
   - Python 3.9 + Transformers + PyTorch

✅ **reid_baseline.yml** (2.5K)
   - 用于训练 Person Re-ID 模型
   - Python 3.9 + PyTorch + TorchVision

✅ **mrt-oast.yml** (4.4K)
   - 用于训练 MRT-OAST 代码克隆检测模型
   - Python 3.8 + PyTorch + Transformers

✅ **dnn_rvsm.yml** (874 bytes)
   - 用于训练 DNN+RVSM Bug定位模型
   - Python 3.7 + TensorFlow/Keras

✅ **pytorch_examples.yml** (200 bytes)
   - 用于运行 PyTorch 基础示例
   - Python 3.10 + PyTorch

### 配置与映射文件

✅ **environment_mapping.yml**
   - 仓库与环境的对应关系
   - 包含每个环境的Python版本和描述

### 工具脚本

✅ **setup_environments.sh** (可执行)
   - 批量创建所有环境
   - 支持选择性创建
   - 支持强制重建

✅ **check_environments.sh** (可执行)
   - 检查环境安装状态
   - 显示已安装/缺失的环境
   - 提供安装提示

### 文档

✅ **README.md** (8.9K)
   - 完整的环境设置指南
   - 故障排除
   - 最佳实践

✅ **QUICK_REFERENCE.md** (1.7K)
   - 快速参考卡片
   - 常用命令
   - 使用示例

## 🎯 环境状态

当前系统环境安装情况（通过 check_environments.sh 查看）:

```
✓ pytorch_resnet_cifar10 - 已安装
✓ vulberta - 已安装
✓ reid_baseline - 已安装
✓ mrt-oast - 已安装
✓ dnn_rvsm - 已安装
✗ mutation_runner - 缺失（需创建）
✗ pytorch_examples - 缺失（需创建）
```

## 🚀 在新机器上的设置步骤

### 步骤1: 复制整个environment目录到新机器

```bash
# 在旧机器上打包
tar -czf environments.tar.gz environment/

# 在新机器上解压
tar -xzf environments.tar.gz
cd environment
```

### 步骤2: 检查conda安装

```bash
conda --version
# 如果没有conda，安装Miniconda:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
```

### 步骤3: 创建所有环境

```bash
# 方法1: 使用自动化脚本（推荐）
./setup_environments.sh --all

# 方法2: 手动创建单个环境
conda env create -f mutation_runner.yml
conda env create -f pytorch_resnet_cifar10.yml
# ... 依次创建其他环境
```

### 步骤4: 验证环境

```bash
# 检查所有环境状态
./check_environments.sh

# 验证特定环境
conda activate mutation_runner
python --version
python -c "import sys; print('Python paths:', sys.path)"
```

### 步骤5: 配置CUDA（如果需要）

```bash
# 检查CUDA版本
nvidia-smi

# 为PyTorch环境安装对应的CUDA版本
conda activate pytorch_resnet_cifar10
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 📊 文件大小汇总

| 类型 | 数量 | 总大小 |
|------|------|--------|
| .yml环境文件 | 7 | ~14KB |
| .sh脚本 | 2 | ~9KB |
| .md文档 | 2 | ~11KB |
| **总计** | **11** | **~34KB** |

## 💾 环境安装后的磁盘空间

预计每个环境安装后的大小：

- mutation_runner: ~200MB
- pytorch_resnet_cifar10: ~3GB
- vulberta: ~5GB
- reid_baseline: ~4GB
- mrt-oast: ~5GB
- dnn_rvsm: ~2GB
- pytorch_examples: ~3GB

**总计约 22GB**（实际大小取决于依赖版本）

## 🔍 验证清单

在新机器上完成设置后，验证以下内容：

### 1. 环境创建
- [ ] 所有7个环境已创建
- [ ] 无创建失败或错误

### 2. Python版本
- [ ] mutation_runner: Python 3.10
- [ ] pytorch_resnet_cifar10: Python 3.10
- [ ] vulberta: Python 3.9
- [ ] reid_baseline: Python 3.9
- [ ] mrt-oast: Python 3.8
- [ ] dnn_rvsm: Python 3.7
- [ ] pytorch_examples: Python 3.10

### 3. 核心包
- [ ] PyTorch环境可以 `import torch`
- [ ] Transformers环境可以 `import transformers`
- [ ] TensorFlow环境可以 `import tensorflow`

### 4. CUDA支持（如果有GPU）
```bash
conda activate pytorch_resnet_cifar10
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. 训练脚本
- [ ] mutation_runner.py 可以运行 `--list`
- [ ] 各仓库的 train.sh 可以找到正确的Python环境

## 🔗 相关文件

- 主项目README: `../README.md`
- 配置说明: `../docs/CONFIG_EXPLANATION.md`
- 使用示例: `../docs/USAGE_EXAMPLES.md`
- 测试文档: `../test/README.md`

## 📝 维护建议

### 定期更新环境文件

当修改任何环境后，重新导出：

```bash
# 导出更新的环境
conda env export -n pytorch_resnet_cifar10 --no-builds > pytorch_resnet_cifar10.yml

# 提交到版本控制
git add pytorch_resnet_cifar10.yml
git commit -m "Update pytorch_resnet_cifar10 environment"
```

### 版本控制

建议将 environment/ 目录添加到 git：

```bash
cd /home/green/energy_dl/nightly
git add environment/
git commit -m "Add conda environment configurations"
```

### 备份

定期备份环境配置：

```bash
# 打包所有环境文件
tar -czf environments_backup_$(date +%Y%m%d).tar.gz environment/

# 移动到备份目录
mv environments_backup_*.tar.gz ~/backups/
```

## ❓ 常见问题

**Q: 为什么有些环境文件比较大？**

A: 大的环境文件（如vulberta 4.1K）包含更多依赖包。使用 `--no-builds` 导出已经去掉了build字符串，这是已经精简过的版本。

**Q: 可以在Windows上使用这些.yml文件吗？**

A: 部分可以。纯Python包可以跨平台，但包含系统特定依赖（���CUDA）的可能需要调整。建议在Windows上重新导出环境文件。

**Q: 环境创建失败怎么办？**

A: 查看 README.md 中的"故障排除"章节，或尝试：
   1. 使用 mamba 代替 conda
   2. 编辑.yml文件，将精确版本改为版本范围
   3. 创建最小环境后手动安装包

## 🎉 完成状态

✅ 已导出 5 个现有环境
✅ 已创建 2 个新环境配置
✅ 已创建自动化设置脚本
✅ 已创建环境检查工具
✅ 已编写完整文档
✅ 已测试环境检查脚本

**所有环境配置文件已准备就绪，可以在新机器上使用！** 🚀
