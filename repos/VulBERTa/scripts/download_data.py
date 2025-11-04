#!/usr/bin/env python3
"""
VulBERTa数据和模型自动下载脚本
使用requests库从OneDrive下载文件
"""

import os
import sys
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_file(url, output_path, description=""):
    """下载文件并显示进度条"""
    print(f"正在下载: {description}")
    print(f"URL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ 下载完成: {output_path}")
        return True

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False


def unzip_file(zip_path, extract_to):
    """解压zip文件"""
    print(f"正在解压: {zip_path} 到 {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ 解压完成")
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def main():
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    print("=" * 50)
    print("VulBERTa 数据和模型自动下载脚本")
    print("=" * 50)
    print()

    # OneDrive直接下载链接
    # 注意: 这些链接可能需要转换为直接下载链接
    downloads = [
        {
            "name": "数据集 (data.zip)",
            "url": "https://onedrive.live.com/download?cid=15E206B36A9C8AE7&resid=15E206B36A9C8AE7%21300801&authkey=AMLXq2nFAmlQYAw",
            "output": "data.zip",
            "extract_to": "data"
        },
        {
            "name": "预训练模型 (pretraining_model.zip)",
            "url": "https://onedrive.live.com/download?cid=15E206B36A9C8AE7&resid=15E206B36A9C8AE7%21300802&authkey=AFnlaFjZFQCCp5w",
            "output": "pretraining_model.zip",
            "extract_to": "models"
        }
    ]

    # 询问是否下载微调模型
    print("是否下载微调模型? (y/n, 默认n): ", end="")
    download_finetuned = input().strip().lower() == 'y'

    if download_finetuned:
        downloads.append({
            "name": "微调模型 (finetuning_models.zip)",
            "url": "https://onedrive.live.com/download?cid=15E206B36A9C8AE7&resid=15E206B36A9C8AE7%21300803&authkey=AIj7B7HPzR0lljI",
            "output": "finetuning_models.zip",
            "extract_to": "models"
        })

    # 下载和解压
    for item in downloads:
        print(f"\n{'=' * 50}")
        print(f"处理: {item['name']}")
        print(f"{'=' * 50}")

        # 检查是否已存在
        if Path(item['output']).exists():
            print(f"文件已存在: {item['output']}")
            print("是否重新下载? (y/n, 默认n): ", end="")
            if input().strip().lower() != 'y':
                print("跳过下载")
                continue

        # 下载
        if download_file(item['url'], item['output'], item['name']):
            # 解压
            if unzip_file(item['output'], item['extract_to']):
                # 删除zip文件
                os.remove(item['output'])
                print(f"✓ 已删除: {item['output']}")
        else:
            print(f"\n⚠ 警告: 自动下载失败")
            print(f"请手动从以下链接下载:")
            print(f"原始链接: {item['url'].replace('onedrive.live.com/download', 'onedrive.live.com')}")
            print(f"下载后放到项目根目录，命名为: {item['output']}")
            print(f"然后运行: unzip {item['output']} -d {item['extract_to']}")

    print("\n" + "=" * 50)
    print("下载完成检查")
    print("=" * 50)

    # 验证目录结构
    expected_dirs = [
        "data/finetune",
        "data/pretrain",
        "data/tokenizer",
        "models/VulBERTa"
    ]

    print("\n验证目录结构:")
    all_ok = True
    for dir_path in expected_dirs:
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n✓ 所有必需的目录都已创建!")
    else:
        print("\n⚠ 部分目录缺失，请检查下载是否成功")

    print("\n完成!")


if __name__ == "__main__":
    main()
