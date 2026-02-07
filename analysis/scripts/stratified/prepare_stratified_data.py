#!/usr/bin/env python3
"""
分层因果分析 - 数据分层脚本

按is_parallel变量将全局标准化数据分割为并行/非并行子集。

输入 (使用DiBS就绪数据，无缺失值):
  - data/energy_research/6groups_dibs_ready/group1_examples_dibs_ready.csv
  - data/energy_research/6groups_dibs_ready/group3_person_reid_dibs_ready.csv

输出:
  - data/energy_research/stratified/group1_examples/group1_parallel.csv
  - data/energy_research/stratified/group1_examples/group1_non_parallel.csv
  - data/energy_research/stratified/group3_person_reid/group3_parallel.csv
  - data/energy_research/stratified/group3_person_reid/group3_non_parallel.csv
  - data/energy_research/stratified/stratification_report.json

使用方法:
    # Dry run (测试模式)
    python prepare_stratified_data.py --dry-run

    # 实际运行
    python prepare_stratified_data.py

    # 强制覆盖已存在的文件
    python prepare_stratified_data.py --force

依赖:
    - pandas
    - numpy
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


# ============================================================================
# 配置
# ============================================================================

# 分层配置 (使用DiBS就绪数据)
STRATIFIED_CONFIG = {
    "group1_examples": {
        "input_file": "group1_examples_dibs_ready.csv",
        "expected_parallel": 178,
        "expected_non_parallel": 126,
        "tolerance": 0.05  # 5%容差
    },
    "group3_person_reid": {
        "input_file": "group3_person_reid_dibs_ready.csv",
        "expected_parallel": 113,
        "expected_non_parallel": 93,
        "tolerance": 0.05
    }
}

# 要移除的列模式
COLUMNS_TO_REMOVE = [
    "is_parallel",      # 分层后为常量
    "timestamp",        # 非因果变量
]

# 交互项列后缀（分层后冗余）
INTERACTION_SUFFIX = "_x_is_parallel"


# ============================================================================
# 辅助函数
# ============================================================================

def get_columns_to_remove(df: pd.DataFrame) -> List[str]:
    """获取需要移除的列列表"""
    cols_to_remove = []

    for col in df.columns:
        # 精确匹配
        if col in COLUMNS_TO_REMOVE:
            cols_to_remove.append(col)
        # 交互项列
        elif col.endswith(INTERACTION_SUFFIX):
            cols_to_remove.append(col)

    return cols_to_remove


def check_data_quality(df: pd.DataFrame, layer_name: str) -> Dict:
    """检查数据质量"""
    quality_report = {
        "layer": layer_name,
        "n_samples": len(df),
        "n_features": len(df.columns),
        "missing_values": {},
        "constant_columns": [],
        "issues": []
    }

    # 检查缺失值
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        quality_report["missing_values"] = missing_cols.to_dict()
        quality_report["issues"].append(f"发现 {len(missing_cols)} 列包含缺失值")

    # 检查常量列
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality_report["constant_columns"].append(col)

    if quality_report["constant_columns"]:
        quality_report["issues"].append(
            f"发现 {len(quality_report['constant_columns'])} 个常量列"
        )

    return quality_report


def validate_sample_count(actual: int, expected: int, tolerance: float) -> Tuple[bool, str]:
    """验证样本数是否在预期范围内"""
    lower = int(expected * (1 - tolerance))
    upper = int(expected * (1 + tolerance))

    if lower <= actual <= upper:
        return True, f"✅ 样本数 {actual} 在预期范围 [{lower}, {upper}] 内"
    else:
        return False, f"❌ 样本数 {actual} 不在预期范围 [{lower}, {upper}] 内"


# ============================================================================
# 主函数
# ============================================================================

def stratify_data(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    force: bool = False
) -> Dict:
    """
    执行数据分层

    参数:
        input_dir: 输入目录 (6groups_global_std)
        output_dir: 输出目录 (stratified)
        dry_run: 是否只检查不执行
        force: 是否强制覆盖

    返回:
        分层报告字典
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "groups": {},
        "summary": {
            "total_input_samples": 0,
            "total_output_samples": 0,
            "validation_passed": True
        }
    }

    print("=" * 70)
    print("分层因果分析 - 数据分层")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模式: {'Dry Run (仅检查)' if dry_run else '实际执行'}")
    print()

    for group_name, config in STRATIFIED_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"处理组: {group_name}")
        print(f"{'='*60}")

        group_report = {
            "input_file": config["input_file"],
            "layers": {},
            "columns_removed": [],
            "validation": {}
        }

        # 1. 读取数据
        input_file = input_dir / config["input_file"]
        if not input_file.exists():
            print(f"❌ 输入文件不存在: {input_file}")
            group_report["error"] = f"输入文件不存在: {input_file}"
            report["groups"][group_name] = group_report
            report["summary"]["validation_passed"] = False
            continue

        print(f"读取文件: {input_file}")
        df = pd.read_csv(input_file)
        print(f"   原始数据: {len(df)} 行 × {len(df.columns)} 列")
        report["summary"]["total_input_samples"] += len(df)

        # 2. 检查is_parallel列
        if "is_parallel" not in df.columns:
            print(f"❌ 缺少is_parallel列")
            group_report["error"] = "缺少is_parallel列"
            report["groups"][group_name] = group_report
            report["summary"]["validation_passed"] = False
            continue

        # 3. 分层
        # 处理is_parallel列的不同表示方式
        if df["is_parallel"].dtype == bool:
            parallel_mask = df["is_parallel"]
        else:
            # 可能是字符串 "True"/"False"
            parallel_mask = df["is_parallel"].astype(str).str.lower() == "true"

        df_parallel = df[parallel_mask].copy()
        df_non_parallel = df[~parallel_mask].copy()

        print(f"   并行样本: {len(df_parallel)}")
        print(f"   非并行样本: {len(df_non_parallel)}")

        # 4. 验证样本数
        valid_p, msg_p = validate_sample_count(
            len(df_parallel), config["expected_parallel"], config["tolerance"]
        )
        valid_np, msg_np = validate_sample_count(
            len(df_non_parallel), config["expected_non_parallel"], config["tolerance"]
        )

        print(f"   {msg_p}")
        print(f"   {msg_np}")

        group_report["validation"]["parallel"] = {
            "actual": len(df_parallel),
            "expected": config["expected_parallel"],
            "passed": valid_p
        }
        group_report["validation"]["non_parallel"] = {
            "actual": len(df_non_parallel),
            "expected": config["expected_non_parallel"],
            "passed": valid_np
        }

        if not (valid_p and valid_np):
            report["summary"]["validation_passed"] = False

        # 5. 移除列
        cols_to_remove = get_columns_to_remove(df_parallel)
        print(f"\n   移除列 ({len(cols_to_remove)} 个):")
        for col in cols_to_remove[:5]:
            print(f"      - {col}")
        if len(cols_to_remove) > 5:
            print(f"      ... 等 {len(cols_to_remove) - 5} 个")

        group_report["columns_removed"] = cols_to_remove

        # 实际移除列
        df_parallel_clean = df_parallel.drop(columns=cols_to_remove, errors='ignore')
        df_non_parallel_clean = df_non_parallel.drop(columns=cols_to_remove, errors='ignore')

        print(f"\n   清理后数据:")
        print(f"      并行: {len(df_parallel_clean)} 行 × {len(df_parallel_clean.columns)} 列")
        print(f"      非并行: {len(df_non_parallel_clean)} 行 × {len(df_non_parallel_clean.columns)} 列")

        # 6. 数据质量检查
        quality_p = check_data_quality(df_parallel_clean, f"{group_name}_parallel")
        quality_np = check_data_quality(df_non_parallel_clean, f"{group_name}_non_parallel")

        group_report["layers"]["parallel"] = {
            "n_samples": len(df_parallel_clean),
            "n_features": len(df_parallel_clean.columns),
            "quality": quality_p
        }
        group_report["layers"]["non_parallel"] = {
            "n_samples": len(df_non_parallel_clean),
            "n_features": len(df_non_parallel_clean.columns),
            "quality": quality_np
        }

        report["summary"]["total_output_samples"] += len(df_parallel_clean) + len(df_non_parallel_clean)

        if quality_p["issues"]:
            print(f"\n   ⚠️ 并行层质量问题:")
            for issue in quality_p["issues"]:
                print(f"      - {issue}")

        if quality_np["issues"]:
            print(f"\n   ⚠️ 非并行层质量问题:")
            for issue in quality_np["issues"]:
                print(f"      - {issue}")

        # 7. 保存数据
        if not dry_run:
            group_output_dir = output_dir / group_name
            group_output_dir.mkdir(parents=True, exist_ok=True)

            # 短名称 (去掉group1_/group3_前缀)
            short_name = group_name.split("_", 1)[0]  # group1 or group3

            parallel_file = group_output_dir / f"{short_name}_parallel.csv"
            non_parallel_file = group_output_dir / f"{short_name}_non_parallel.csv"

            if parallel_file.exists() and not force:
                print(f"\n   ⚠️ 文件已存在，跳过: {parallel_file}")
                print(f"      使用 --force 强制覆盖")
            else:
                df_parallel_clean.to_csv(parallel_file, index=False)
                print(f"\n   ✅ 已保存: {parallel_file}")

            if non_parallel_file.exists() and not force:
                print(f"   ⚠️ 文件已存在，跳过: {non_parallel_file}")
            else:
                df_non_parallel_clean.to_csv(non_parallel_file, index=False)
                print(f"   ✅ 已保存: {non_parallel_file}")

            group_report["output_files"] = {
                "parallel": str(parallel_file),
                "non_parallel": str(non_parallel_file)
            }

        report["groups"][group_name] = group_report

    # 8. 保存报告
    if not dry_run:
        report_file = output_dir / "stratification_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 报告已保存: {report_file}")

    # 9. 打印总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"输入样本总数: {report['summary']['total_input_samples']}")
    print(f"输出样本总数: {report['summary']['total_output_samples']}")
    print(f"验证状态: {'✅ 全部通过' if report['summary']['validation_passed'] else '❌ 存在问题'}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="分层因果分析 - 数据分层脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run模式，只检查数据不实际保存"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的文件"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入目录路径 (默认: data/energy_research/6groups_dibs_ready)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: data/energy_research/stratified)"
    )

    args = parser.parse_args()

    # 确定路径
    script_dir = Path(__file__).parent.absolute()
    analysis_dir = script_dir.parent.parent  # analysis/

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = analysis_dir / "data" / "energy_research" / "6groups_dibs_ready"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = analysis_dir / "data" / "energy_research" / "stratified"

    # 检查输入目录
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 执行分层
    report = stratify_data(
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
        force=args.force
    )

    # 返回状态
    if report["summary"]["validation_passed"]:
        print("\n🎉 数据分层完成！")
        sys.exit(0)
    else:
        print("\n⚠️ 数据分层完成，但存在验证问题，请检查报告")
        sys.exit(1)


if __name__ == "__main__":
    main()
