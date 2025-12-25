# Quick Reference - Conda Environments

## 🚀 快速创建所有环境

```bash
cd environment
./setup_environments.sh --all
```

## 📋 环境对照表

| 仓库 | 环境名 | 文件 |
|------|--------|------|
| mutation_runner.py | `mutation_runner` | mutation_runner.yml |
| MRT-OAST | `mrt-oast` | mrt-oast.yml |
| bug-localization | `dnn_rvsm` | dnn_rvsm.yml |
| pytorch_resnet_cifar10 | `pytorch_resnet_cifar10` | pytorch_resnet_cifar10.yml |
| VulBERTa | `vulberta` | vulberta.yml |
| Person_reID | `reid_baseline` | reid_baseline.yml |
| examples | `pytorch_examples` | pytorch_examples.yml |

## 🔧 常用命令

### 创建环境
```bash
conda env create -f <环境文件>.yml
```

### 激活环境
```bash
conda activate <环境名>
```

### 检查环境状态
```bash
cd environment
./check_environments.sh
```

### 删除环境
```bash
conda env remove -n <环境名>
```

### 更新环境
```bash
conda env update -n <环境名> -f <环境文件>.yml --prune
```

### 导出环境
```bash
conda env export -n <环境名> --no-builds > <环境文件>.yml
```

## 🎯 使用示例

### 运行mutation_runner
```bash
conda activate mutation_runner
python3 mutation_runner.py --list
```

### 训练ResNet
```bash
conda activate pytorch_resnet_cifar10
cd repos/pytorch_resnet_cifar10
./train.sh
```

### 训练VulBERTa
```bash
conda activate vulberta
cd repos/VulBERTa
./train.sh -n mlp -d d2a
```

## 💡 提示

- 使用 `mamba` 代替 `conda` 可以更快创建环境
- 首次创建环境可能需要较长时间下载依赖
- 推荐先创建 `mutation_runner` 环境
- GPU环境需要预先安装NVIDIA驱动

## 📚 详细文档

完整文档请参考: [README.md](README.md)
