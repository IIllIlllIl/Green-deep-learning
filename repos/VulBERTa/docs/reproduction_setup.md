# VulBERTa 复现环境搭建文档

## 1. 仓库训练代码检查结果

### 训练类型
- **本地训练**: 是
- **训练框架**: PyTorch 1.7.0 + Transformers 4.4.1
- **训练方式**: Jupyter Notebook

### 训练流程
1. **Tokenizer训练** (`Tokenizer_training.ipynb`)
   - 使用clang进行自定义tokenization
   - BPE tokenizer，vocab_size=50000
   - 已有预训练tokenizer在 `tokenizer/` 目录

2. **预训练** (`Pretraining_VulBERTa.ipynb`)
   - 数据集: DrapGH dataset
   - 任务: Masked Language Modeling (MLM)
   - MLM概率: 15%
   - 配置选项: small, medium, base, large

3. **微调**
   - VulBERTa-MLP (`Finetuning_VulBERTa-MLP.ipynb`)
   - VulBERTa-CNN (`Finetuning+evaluation_VulBERTa-CNN.ipynb`)

### 自定义模块
- `custom.py`: 自定义数据collator和masking逻辑
- `models.py`: VulBERTa模型架构定义
  - VulBERTa_Vanilla (MLP)
  - VulBERTa_Extend (深层MLP)
  - VulBERTa_CNN (卷积神经网络)
  - VulBERTa_LSTM (LSTM)

## 2. Conda环境配置

### 环境信息
- **环境名**: vulberta
- **Python版本**: 3.8.20
- **CUDA版本**: 10.1

### 已安装的核心包
- PyTorch: 1.7.0+cu101
- Transformers: 4.4.1
- Tokenizers: 0.10.1
- libclang: 12.0.0
- pandas: 2.0.3
- scikit-learn: 1.3.2
- jupyter/jupyterlab: 已安装

### 硬件配置
- **GPU**: NVIDIA GeForce RTX 3080 10GB VRAM
- **Driver Version**: 535.183.01
- **CUDA Version**: 12.2 (兼容CUDA 10.1运行时)

### 注意事项
- packaging降级到20.4以解决transformers依赖问题
- JupyterLab可能有版本冲突警告，但不影响训练

## 3. 数据和模型下载

### 数据下载
来源: OneDrive
- **数据链接**: https://1drv.ms/u/s!AueKnGqzBuIVkq4B9ESELGQ-VtjIYA?e=f0moEm
- **包含目录**:
  - `finetune/` - 微调数据集
  - `pretrain/` - 预训练数据集 (DrapGH)
  - `tokenizer/` - Tokenizer训练数据

### 模型下载
来源: OneDrive

**预训练模型 (VulBERTa)**:
- **链接**: https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p
- **目录**: `VulBERTa/`

**微调模型 (VulBERTa-MLP, VulBERTa-CNN)**:
- **链接**: https://1drv.ms/u/s!AueKnGqzBuIVkq4DAleeVbhSzuB87w?e=jdI83b
- **目录**:
  - `VB-MLP_{dataset-name}` (6个数据集)
  - `VB-CNN_{dataset-name}` (6个数据集)
  - 共12个文件夹

## 4. 激活环境命令

```bash
conda activate vulberta
```
