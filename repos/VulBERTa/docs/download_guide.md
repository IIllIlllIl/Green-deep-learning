# VulBERTa 数据和模型下载指南

## 重要提示
由于OneDrive的访问限制，自动下载脚本可能无法正常工作。请按照以下步骤手动下载。

## 下载步骤

### 1. 下载数据集
**必需 - 用于训练tokenizer和预训练**

1. 访问链接: https://1drv.ms/u/s!AueKnGqzBuIVkq4B9ESELGQ-VtjIYA?e=f0moEm
2. 点击"下载"按钮
3. 将下载的 `data.zip` 保存到项目根目录 `/home/green/energy_dl/test/VulBERTa/`
4. 解压文件:
   ```bash
   cd /home/green/energy_dl/test/VulBERTa
   unzip data.zip -d data/
   rm data.zip  # 解压后删除zip文件
   ```

解压后应该包含以下目录:
- `data/finetune/` - 微调数据集
- `data/pretrain/` - 预训练数据集 (drapgh.pkl)
- `data/tokenizer/` - Tokenizer训练数据 (drapgh.txt)

### 2. 下载预训练模型 (可选)
**如果需要直接使用预训练模型进行微调**

1. 访问链接: https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p
2. 点击"下载"按钮
3. 将下载的 `pretraining_model.zip` 保存到项目根目录
4. 解压文件:
   ```bash
   unzip pretraining_model.zip -d models/
   rm pretraining_model.zip
   ```

解压后应该包含:
- `models/VulBERTa/` - 预训练的VulBERTa模型

### 3. 下载微调模型 (可选)
**如果需要直接评估微调后的模型**

1. 访问链接: https://1drv.ms/u/s!AueKnGqzBuIVkq4DAleeVbhSzuB87w?e=jdI83b
2. 点击"下载"按钮
3. 将下载的 `finetuning_models.zip` 保存到项目根目录
4. 解压文件:
   ```bash
   unzip finetuning_models.zip -d models/
   rm finetuning_models.zip
   ```

解压后应该包含12个目录:
- `models/VB-MLP_*` (6个数据集)
- `models/VB-CNN_*` (6个数据集)

## 验证下载

运行以下命令验证文件结构:

```bash
# 检查数据目录
ls -lh data/pretrain/
ls -lh data/tokenizer/
ls -lh data/finetune/

# 检查模型目录 (如果下载了预训练模型)
ls -lh models/VulBERTa/
```

或使用验证脚本:

```bash
python scripts/verify_data.py
```

## 文件大小参考

- `data.zip`: 约 XXX MB (需要实际下载后确认)
- `pretraining_model.zip`: 约 XXX MB
- `finetuning_models.zip`: 约 XXX MB

## 常见问题

### Q: 下载速度很慢怎么办?
A: OneDrive在国内访问可能较慢，建议使用代理或在网络较好时下载。

### Q: 解压时提示空间不足?
A: 确保磁盘有足够空间。数据集和模型可能需要几GB的空间。

### Q: 我只想复现部分实验，需要下载所有文件吗?
A:
- **仅预训练**: 只需要 `data.zip` 中的 `data/pretrain/` 和 `data/tokenizer/`
- **仅微调**: 需要 `data.zip` 中的 `data/finetune/` 和 `pretraining_model.zip`
- **仅评估**: 需要 `data.zip` 中的 `data/finetune/` 和 `finetuning_models.zip`
- **完整复现**: 需要所有文件

## 备选下载方法

如果OneDrive链接无法访问，可以尝试:
1. 联系论文作者获取数据集
2. 查看GitHub Issues是否有其他下载源
3. 使用浏览器的下载管理器进行断点续传
