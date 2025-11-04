#!/usr/bin/env python3
"""
验证VulBERTa数据和模型是否已正确下载和解压
"""

import os
from pathlib import Path

def check_path(path, description, required=True):
    """检查路径是否存在"""
    exists = Path(path).exists()
    status = "✓" if exists else ("✗" if required else "○")
    req_str = "必需" if required else "可选"
    print(f"{status} [{req_str}] {description}: {path}")
    return exists

def check_file_size(path):
    """检查文件大小"""
    if Path(path).exists():
        if Path(path).is_file():
            size = Path(path).stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"    文件大小: {size_mb:.2f} MB")
        elif Path(path).is_dir():
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            file_count = sum(1 for f in Path(path).rglob('*') if f.is_file())
            print(f"    目录大小: {size_mb:.2f} MB, 文件数: {file_count}")

def main():
    print("=" * 60)
    print("VulBERTa 数据和模型验证")
    print("=" * 60)
    print()

    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    all_required_ok = True

    # 检查tokenizer (已存在)
    print("1. Tokenizer文件检查:")
    check_path("tokenizer/drapgh-vocab.json", "Tokenizer词表", required=True)
    check_path("tokenizer/drapgh-merges.txt", "Tokenizer合并规则", required=True)
    print()

    # 检查数据集
    print("2. 数据集检查:")
    has_data = check_path("data/pretrain", "预训练数据目录", required=True)
    if has_data:
        check_file_size("data/pretrain")
        check_path("data/pretrain/drapgh.pkl", "DrapGH数据集", required=True)

    has_tokenizer_data = check_path("data/tokenizer", "Tokenizer训练数据目录", required=True)
    if has_tokenizer_data:
        check_file_size("data/tokenizer")
        check_path("data/tokenizer/drapgh.txt", "Tokenizer训练文本", required=True)

    has_finetune = check_path("data/finetune", "微调数据目录", required=True)
    if has_finetune:
        check_file_size("data/finetune")
    print()

    # 检查预训练模型 (可选)
    print("3. 预训练模型检查 (可选):")
    has_pretrained = check_path("models/VulBERTa", "预训练VulBERTa模型", required=False)
    if has_pretrained:
        check_file_size("models/VulBERTa")
        check_path("models/VulBERTa/config.json", "模型配置", required=False)
        check_path("models/VulBERTa/pytorch_model.bin", "模型权重", required=False)
    print()

    # 检查微调模型 (可选)
    print("4. 微调模型检查 (可选):")
    mlp_count = len(list(Path("models").glob("VB-MLP_*")))
    cnn_count = len(list(Path("models").glob("VB-CNN_*")))
    print(f"  发现 VB-MLP 模型: {mlp_count} 个")
    print(f"  发现 VB-CNN 模型: {cnn_count} 个")
    print()

    # 总结
    print("=" * 60)
    print("验证总结:")
    print("=" * 60)

    required_items = [
        ("Tokenizer", Path("tokenizer/drapgh-vocab.json").exists()),
        ("预训练数据", Path("data/pretrain").exists()),
        ("微调数据", Path("data/finetune").exists()),
    ]

    optional_items = [
        ("预训练模型", Path("models/VulBERTa").exists()),
        ("微调模型", mlp_count > 0 or cnn_count > 0),
    ]

    print("\n必需项:")
    all_required_ok = True
    for name, exists in required_items:
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            all_required_ok = False

    print("\n可选项:")
    for name, exists in optional_items:
        status = "✓" if exists else "○"
        print(f"  {status} {name}")

    print()
    if all_required_ok:
        print("✓ 所有必需的数据文件都已就绪!")
        print("  可以开始训练流程。")
    else:
        print("✗ 部分必需文件缺失!")
        print("  请参考 docs/download_guide.md 下载缺失的文件。")

    if not Path("models/VulBERTa").exists():
        print("\n提示: 未找到预训练模型。")
        print("  - 如果要从头预训练，无需下载")
        print("  - 如果要直接微调，请下载预训练模型")

    print()

if __name__ == "__main__":
    main()
