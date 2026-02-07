#!/usr/bin/env python3
"""
分层分析与全局分析对标脚本

对比分层ATE结果与全局ATE结果，验证一致性

对标要求（来自方案）:
- 因果边一致性检查：分层结果应保留全局分析中的核心强边（≥70%保留率）
- ATE符号一致性检查：≥80%的边ATE符号一致
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# 路径设置
ANALYSIS_DIR = Path(__file__).parent.parent.parent

# 全局ATE目录
GLOBAL_ATE_DIR = ANALYSIS_DIR / "results/energy_research/data/global_std_dibs_ate"

# 分层ATE目录
STRATIFIED_ATE_DIR = ANALYSIS_DIR / "results/energy_research/stratified/ate"

# 输出目录
OUTPUT_DIR = ANALYSIS_DIR / "results/energy_research/stratified/benchmark"


def load_global_ate(group_name: str) -> pd.DataFrame:
    """加载全局ATE结果"""
    file_path = GLOBAL_ATE_DIR / f"{group_name}_dibs_global_std_ate.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    # 标准化列名
    df = df.rename(columns={
        'ate_global_std': 'ate',
        'ate_global_std_ci_lower': 'ci_lower',
        'ate_global_std_ci_upper': 'ci_upper',
        'ate_global_std_is_significant': 'is_significant'
    })
    df['edge'] = df['source'] + '->' + df['target']
    return df


def load_stratified_ate(layer_id: str) -> pd.DataFrame:
    """加载分层ATE结果"""
    file_path = STRATIFIED_ATE_DIR / layer_id / f"{layer_id}_ate_raw.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['edge'] = df['source'] + '->' + df['target']
    return df


def compare_edges(global_df: pd.DataFrame, stratified_df: pd.DataFrame) -> dict:
    """对比因果边"""
    global_edges = set(global_df['edge'].tolist())
    stratified_edges = set(stratified_df['edge'].tolist())

    common_edges = global_edges & stratified_edges
    global_only = global_edges - stratified_edges
    stratified_only = stratified_edges - global_edges

    # 计算保留率
    if len(global_edges) > 0:
        retention_rate = len(common_edges) / len(global_edges)
    else:
        retention_rate = 0

    return {
        "global_edges": len(global_edges),
        "stratified_edges": len(stratified_edges),
        "common_edges": len(common_edges),
        "global_only": len(global_only),
        "stratified_only": len(stratified_only),
        "retention_rate": retention_rate,
        "common_edge_list": list(common_edges),
        "global_only_list": list(global_only)[:10],  # 只保留前10个
        "stratified_only_list": list(stratified_only)[:10]
    }


def compare_ate_signs(global_df: pd.DataFrame, stratified_df: pd.DataFrame, common_edges: list) -> dict:
    """对比ATE符号一致性"""
    if not common_edges:
        return {"consistent_count": 0, "total_count": 0, "consistency_rate": 0}

    consistent = 0
    inconsistent = 0
    details = []

    for edge in common_edges:
        global_row = global_df[global_df['edge'] == edge]
        stratified_row = stratified_df[stratified_df['edge'] == edge]

        if len(global_row) == 0 or len(stratified_row) == 0:
            continue

        global_ate = global_row['ate'].values[0]
        stratified_ate = stratified_row['ate'].values[0]

        if pd.isna(global_ate) or pd.isna(stratified_ate):
            continue

        global_sign = np.sign(global_ate)
        stratified_sign = np.sign(stratified_ate)

        if global_sign == stratified_sign:
            consistent += 1
        else:
            inconsistent += 1
            details.append({
                "edge": edge,
                "global_ate": float(global_ate),
                "stratified_ate": float(stratified_ate),
                "global_sign": "+" if global_sign > 0 else "-",
                "stratified_sign": "+" if stratified_sign > 0 else "-"
            })

    total = consistent + inconsistent
    rate = consistent / total if total > 0 else 0

    return {
        "consistent_count": consistent,
        "inconsistent_count": inconsistent,
        "total_count": total,
        "consistency_rate": rate,
        "inconsistent_details": details[:5]  # 只保留前5个
    }


def run_benchmark(layer_id: str, global_group: str) -> dict:
    """运行单个分层的对标"""
    print(f"\n{'='*60}")
    print(f"对标: {layer_id} vs {global_group}")
    print(f"{'='*60}")

    # 加载数据
    global_df = load_global_ate(global_group)
    stratified_df = load_stratified_ate(layer_id)

    if len(global_df) == 0:
        print(f"  ❌ 全局ATE数据不存在")
        return {"success": False, "error": "全局ATE数据不存在"}

    if len(stratified_df) == 0:
        print(f"  ❌ 分层ATE数据不存在")
        return {"success": False, "error": "分层ATE数据不存在"}

    print(f"  全局边数: {len(global_df)}")
    print(f"  分层边数: {len(stratified_df)}")

    # 边对比
    edge_comparison = compare_edges(global_df, stratified_df)
    print(f"\n  边对比:")
    print(f"    共有边: {edge_comparison['common_edges']}")
    print(f"    全局特有: {edge_comparison['global_only']}")
    print(f"    分层特有: {edge_comparison['stratified_only']}")
    print(f"    保留率: {edge_comparison['retention_rate']:.1%}")

    # 验收标准: ≥70%保留率
    retention_pass = edge_comparison['retention_rate'] >= 0.70
    print(f"    验收(≥70%): {'✅' if retention_pass else '❌'}")

    # ATE符号对比
    ate_comparison = compare_ate_signs(
        global_df, stratified_df, edge_comparison['common_edge_list']
    )
    print(f"\n  ATE符号一致性:")
    print(f"    一致: {ate_comparison['consistent_count']}/{ate_comparison['total_count']}")
    print(f"    一致率: {ate_comparison['consistency_rate']:.1%}")

    # 验收标准: ≥80%一致率
    ate_pass = ate_comparison['consistency_rate'] >= 0.80
    print(f"    验收(≥80%): {'✅' if ate_pass else '❌'}")

    if ate_comparison['inconsistent_details']:
        print(f"\n  不一致的边:")
        for detail in ate_comparison['inconsistent_details']:
            print(f"    {detail['edge']}: 全局{detail['global_sign']} vs 分层{detail['stratified_sign']}")

    return {
        "success": True,
        "layer_id": layer_id,
        "global_group": global_group,
        "edge_comparison": edge_comparison,
        "ate_comparison": ate_comparison,
        "retention_pass": retention_pass,
        "ate_pass": ate_pass,
        "overall_pass": retention_pass and ate_pass
    }


def main():
    print("=" * 60)
    print("分层分析与全局分析对标")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 对标配置
    benchmarks = [
        ("group1_parallel", "group1_examples"),
        ("group1_non_parallel", "group1_examples"),
        ("group3_parallel", "group3_person_reid"),
        ("group3_non_parallel", "group3_person_reid"),
    ]

    results = []

    for layer_id, global_group in benchmarks:
        try:
            result = run_benchmark(layer_id, global_group)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ 对标失败: {e}")
            results.append({
                "success": False,
                "layer_id": layer_id,
                "global_group": global_group,
                "error": str(e)
            })

    # 生成报告
    print(f"\n{'='*60}")
    print("对标总结")
    print(f"{'='*60}")

    successful = [r for r in results if r.get("success")]
    passed = [r for r in successful if r.get("overall_pass")]

    print(f"\n成功对标: {len(successful)}/{len(benchmarks)}")
    print(f"通过验收: {len(passed)}/{len(successful)}")

    for result in successful:
        layer = result['layer_id']
        edge_pass = '✅' if result['retention_pass'] else '❌'
        ate_pass = '✅' if result['ate_pass'] else '❌'
        overall = '✅ PASS' if result['overall_pass'] else '❌ FAIL'

        print(f"\n  {layer}:")
        print(f"    边保留率: {result['edge_comparison']['retention_rate']:.1%} {edge_pass}")
        print(f"    ATE一致率: {result['ate_comparison']['consistency_rate']:.1%} {ate_pass}")
        print(f"    总体: {overall}")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_benchmarks": len(benchmarks),
        "successful": len(successful),
        "passed": len(passed),
        "results": results
    }

    report_file = OUTPUT_DIR / "benchmark_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ 报告已保存: {report_file}")

    return 0 if len(passed) == len(successful) else 1


if __name__ == "__main__":
    sys.exit(main())
