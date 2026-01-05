#!/bin/bash

# HRNet18 修复实验运行脚本
# 用于重新运行失败的hrnet18实验（实验6和17）
# 日期: 2025-11-18

echo "========================================"
echo "HRNet18 修复实验"
echo "========================================"
echo ""
echo "此脚本将重新运行2个失败的hrnet18实验："
echo "  - 实验6: Person_reID_baseline_pytorch_hrnet18 (顺序)"
echo "  - 实验17: Person_reID_baseline_pytorch_hrnet18_parallel (并行)"
echo ""
echo "预计时间: 12-16分钟"
echo ""

# 设置离线模式环境变量
echo "设置离线模式环境变量..."
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1

echo "✓ HF_HUB_OFFLINE=1"
echo "✓ HF_HUB_DISABLE_TELEMETRY=1"
echo "✓ TRANSFORMERS_OFFLINE=1"
echo ""

# 验证缓存存在
echo "验证HuggingFace缓存..."
if [ -d "$HOME/.cache/huggingface/hub/models--timm--hrnet_w18.ms_aug_in1k" ]; then
    echo "✓ HRNet18模型缓存存在"
else
    echo "✗ 警告: HRNet18模型缓存不存在"
    echo "  请先运行: python3 scripts/download_pretrained_models.py"
    exit 1
fi
echo ""

# 运行修复实验
echo "开始运行修复实验..."
echo "配置文件: settings/fix_hrnet18_1epoch.json"
echo ""
echo "========================================"
echo ""

# 使用sudo -E保留环境变量
sudo -E python3 mutation.py -ec settings/fix_hrnet18_1epoch.json

echo ""
echo "========================================"
echo "修复实验完成"
echo "========================================"
echo ""
echo "请检查结果目录："
echo "  ls -ltr results/"
echo ""
echo "查看最新的运行："
echo "  ls results/run_*/Person_reID_baseline_pytorch_hrnet18_*/"
echo ""
