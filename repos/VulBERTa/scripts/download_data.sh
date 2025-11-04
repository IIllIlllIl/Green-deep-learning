#!/bin/bash

# VulBERTa数据和模型下载脚本

set -e

echo "======================================"
echo "VulBERTa 数据和模型下载脚本"
echo "======================================"
echo ""

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "项目目录: $PROJECT_DIR"
echo ""

# 下载数据集
echo "1. 下载数据集..."
echo "请从以下链接手动下载数据集（约需几分钟）:"
echo "https://1drv.ms/u/s!AueKnGqzBuIVkq4B9ESELGQ-VtjIYA?e=f0moEm"
echo ""
echo "下载后，将data.zip放到当前目录，然后按Enter继续..."
read -p "按Enter继续..."

if [ -f "data.zip" ]; then
    echo "找到data.zip，正在解压..."
    unzip -q data.zip -d data/
    echo "数据集解压完成！"
    rm data.zip
    echo "已删除data.zip"
else
    echo "警告: 未找到data.zip，跳过数据集解压"
fi

echo ""
echo "2. 下载预训练模型..."
echo "请从以下链接手动下载预训练模型:"
echo "https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p"
echo ""
echo "下载后，将pretraining_model.zip放到当前目录，然后按Enter继续..."
read -p "按Enter继续..."

if [ -f "pretraining_model.zip" ]; then
    echo "找到pretraining_model.zip，正在解压..."
    unzip -q pretraining_model.zip -d models/
    echo "预训练模型解压完成！"
    rm pretraining_model.zip
    echo "已删除pretraining_model.zip"
else
    echo "警告: 未找到pretraining_model.zip，跳过预训练模型解压"
fi

echo ""
echo "3. 下载微调模型（可选）..."
echo "请从以下链接手动下载微调模型:"
echo "https://1drv.ms/u/s!AueKnGqzBuIVkq4DAleeVbhSzuB87w?e=jdI83b"
echo ""
echo "下载后，将finetuning_models.zip放到当前目录，然后按Enter继续..."
echo "如果不需要，直接按Enter跳过"
read -p "按Enter继续..."

if [ -f "finetuning_models.zip" ]; then
    echo "找到finetuning_models.zip，正在解压..."
    unzip -q finetuning_models.zip -d models/
    echo "微调模型解压完成！"
    rm finetuning_models.zip
    echo "已删除finetuning_models.zip"
else
    echo "跳过微调模型下载"
fi

echo ""
echo "======================================"
echo "下载和解压完成！"
echo "======================================"
echo ""
echo "请验证以下目录结构:"
echo "data/"
echo "  ├── finetune/"
echo "  ├── pretrain/"
echo "  └── tokenizer/"
echo ""
echo "models/"
echo "  └── VulBERTa/"
echo ""
