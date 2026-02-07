#!/usr/bin/env python3
"""
分层权衡检测脚本

基于分层ATE结果检测能耗vs性能权衡
使用Algorithm 1: opposite signs规则

数据源:
  - 分层ATE结果: results/energy_research/stratified/ate/

输出:
  - 权衡检测结果: results/energy_research/stratified/tradeoff/

使用方法:
    python scripts/stratified/run_tradeoff_stratified.py

依赖:
    - analysis/utils/tradeoff_detection.py (TradeoffDetector)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.tradeoff_detection import TradeoffDetector, ENERGY_PERF_RULES


# ============================================================================
# 配置
# ============================================================================

STRATIFIED_LAYERS = [
    "group1_parallel",
    "group1_non_parallel",
    "group3_parallel",
    "group3_non_parallel"
]


# ============================================================================
# 数据加载
# ============================================================================

def load_stratified_ate(layer_id: str, ate_dir: Path, use_fdr: bool = True) -> Dict:
    """
    加载分层ATE数据

    参数:
        layer_id: 分层ID
        ate_dir: ATE结果目录
        use_fdr: 是否使用FDR校正后的显著性

    返回:
        causal_effects: 因果效应字典
            {
                'source->target': {
                    'ate': float,
                    'ci_lower': float,
                    'ci_upper': float,
                    'is_significant': bool
                },
                ...
            }
    """
    # 优先加载FDR校正后的合并文件
    if use_fdr:
        fdr_file = ate_dir / "all_layers_ate_fdr_corrected.csv"
        if fdr_file.exists():
            df = pd.read_csv(fdr_file)
            df = df[df['layer'] == layer_id]

            causal_effects = {}
            for _, row in df.iterrows():
                edge = f"{row['source']}->{row['target']}"

                # 检查有效性
                if pd.isna(row['ate']):
                    continue

                # 使用FDR校正后的显著性
                is_significant = row.get('is_significant_fdr', False)
                if pd.isna(is_significant):
                    is_significant = False

                causal_effects[edge] = {
                    'ate': row['ate'],
                    'ci_lower': row.get('ci_lower', None),
                    'ci_upper': row.get('ci_upper', None),
                    'is_significant': bool(is_significant),
                    'p_value': row.get('p_value', None),
                    'p_value_fdr': row.get('p_value_fdr', None)
                }

            print(f"  加载FDR校正结果: {len(causal_effects)} 条边")
            return causal_effects

    # 回退到原始ATE文件
    raw_file = ate_dir / layer_id / f"{layer_id}_ate_raw.csv"
    if not raw_file.exists():
        print(f"  ❌ ATE文件不存在: {raw_file}")
        return {}

    df = pd.read_csv(raw_file)

    causal_effects = {}
    for _, row in df.iterrows():
        edge = f"{row['source']}->{row['target']}"

        if pd.isna(row['ate']):
            continue

        # 使用原始p值判断显著性
        p_value = row.get('p_value', 1.0)
        is_significant = p_value < 0.10 if pd.notna(p_value) else False

        causal_effects[edge] = {
            'ate': row['ate'],
            'ci_lower': row.get('ci_lower', None),
            'ci_upper': row.get('ci_upper', None),
            'is_significant': is_significant,
            'p_value': p_value
        }

    print(f"  加载原始ATE: {len(causal_effects)} 条边 ({sum(1 for e in causal_effects.values() if e['is_significant'])} 显著)")
    return causal_effects


# ============================================================================
# 权衡检测
# ============================================================================

def detect_tradeoffs_for_layer(
    layer_id: str,
    ate_dir: Path,
    output_dir: Path,
    use_fdr: bool = True
) -> Dict:
    """
    为单个分层检测权衡

    返回:
        results: 检测结果字典
    """
    print(f"\n{'='*60}")
    print(f"权衡检测: {layer_id}")
    print(f"{'='*60}")

    # 加载ATE数据
    causal_effects = load_stratified_ate(layer_id, ate_dir, use_fdr)

    if not causal_effects:
        print(f"  ⚠️ 无有效ATE数据")
        return {"success": False, "layer_id": layer_id, "error": "无有效ATE数据"}

    # 初始化权衡检测器
    detector = TradeoffDetector(
        rules=ENERGY_PERF_RULES,
        current_values={},
        verbose=True
    )

    # 执行权衡检测
    tradeoffs = detector.detect_tradeoffs(
        causal_effects=causal_effects,
        require_significance=True
    )

    print(f"\n  ✅ 检测到 {len(tradeoffs)} 个权衡")

    # 分析权衡类型
    energy_vs_perf = 0
    hyperparam_intervention = 0

    for t in tradeoffs:
        has_energy = ('energy' in t['metric1'].lower() or 'energy' in t['metric2'].lower())
        has_perf = ('perf' in t['metric1'].lower() or 'perf' in t['metric2'].lower())
        if has_energy and has_perf:
            energy_vs_perf += 1

        if 'hyperparam' in t['intervention'].lower():
            hyperparam_intervention += 1

    print(f"  能耗vs性能: {energy_vs_perf}")
    print(f"  超参数干预: {hyperparam_intervention}")

    # 保存结果
    layer_output_dir = output_dir / layer_id
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    # 保存权衡列表
    tradeoff_file = layer_output_dir / f"{layer_id}_tradeoffs.json"
    with open(tradeoff_file, 'w') as f:
        json.dump(tradeoffs, f, indent=2)

    # 保存为CSV
    if tradeoffs:
        tradeoff_df = pd.DataFrame(tradeoffs)
        tradeoff_csv = layer_output_dir / f"{layer_id}_tradeoffs.csv"
        tradeoff_df.to_csv(tradeoff_csv, index=False)

    # 统计信息
    stats = {
        "layer_id": layer_id,
        "total_edges": len(causal_effects),
        "significant_edges": sum(1 for e in causal_effects.values() if e['is_significant']),
        "tradeoffs_detected": len(tradeoffs),
        "energy_vs_perf": energy_vs_perf,
        "hyperparam_intervention": hyperparam_intervention,
        "timestamp": datetime.now().isoformat()
    }

    stats_file = layer_output_dir / f"{layer_id}_tradeoff_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return {
        "success": True,
        "layer_id": layer_id,
        "tradeoffs": tradeoffs,
        "stats": stats
    }


def generate_comparison_report(
    all_results: List[Dict],
    output_dir: Path
) -> None:
    """
    生成分层权衡对比报告
    """
    print(f"\n{'='*60}")
    print("生成对比报告")
    print(f"{'='*60}")

    # 合并所有权衡
    all_tradeoffs = []
    all_stats = []

    for result in all_results:
        if not result.get("success"):
            continue

        layer_id = result["layer_id"]
        tradeoffs = result.get("tradeoffs", [])
        stats = result.get("stats", {})

        all_stats.append(stats)

        for t in tradeoffs:
            t_copy = t.copy()
            t_copy["layer_id"] = layer_id
            all_tradeoffs.append(t_copy)

    # 保存合并的权衡列表
    if all_tradeoffs:
        combined_df = pd.DataFrame(all_tradeoffs)
        combined_file = output_dir / "all_layers_tradeoffs.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"  合并权衡表: {combined_file} ({len(all_tradeoffs)} 个权衡)")

    # 保存统计摘要
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_file = output_dir / "tradeoff_summary.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"  统计摘要: {stats_file}")

    # 分层对比分析
    print(f"\n分层对比:")
    print(f"{'层':<25} {'总边数':<10} {'显著边':<10} {'权衡数':<10} {'能耗vs性能':<12}")
    print("-" * 70)

    for stats in all_stats:
        print(f"{stats['layer_id']:<25} {stats['total_edges']:<10} {stats['significant_edges']:<10} "
              f"{stats['tradeoffs_detected']:<10} {stats['energy_vs_perf']:<12}")

    # 并行vs非并行对比
    print(f"\n并行vs非并行对比:")

    for group in ["group1", "group3"]:
        parallel_stats = next((s for s in all_stats if s['layer_id'] == f"{group}_parallel"), None)
        non_parallel_stats = next((s for s in all_stats if s['layer_id'] == f"{group}_non_parallel"), None)

        if parallel_stats and non_parallel_stats:
            print(f"\n  {group.upper()}:")
            print(f"    并行: {parallel_stats['tradeoffs_detected']} 权衡, {parallel_stats['energy_vs_perf']} 能耗vs性能")
            print(f"    非并行: {non_parallel_stats['tradeoffs_detected']} 权衡, {non_parallel_stats['energy_vs_perf']} 能耗vs性能")

            # 差异分析
            diff = parallel_stats['tradeoffs_detected'] - non_parallel_stats['tradeoffs_detected']
            if diff > 0:
                print(f"    → 并行场景多 {diff} 个权衡")
            elif diff < 0:
                print(f"    → 非并行场景多 {-diff} 个权衡")
            else:
                print(f"    → 权衡数量相同")

    # 保存对比报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "layers_processed": len(all_stats),
        "total_tradeoffs": len(all_tradeoffs),
        "stats_by_layer": all_stats,
        "comparison": {
            "group1": {
                "parallel": next((s for s in all_stats if s['layer_id'] == "group1_parallel"), {}),
                "non_parallel": next((s for s in all_stats if s['layer_id'] == "group1_non_parallel"), {})
            },
            "group3": {
                "parallel": next((s for s in all_stats if s['layer_id'] == "group3_parallel"), {}),
                "non_parallel": next((s for s in all_stats if s['layer_id'] == "group3_non_parallel"), {})
            }
        }
    }

    report_file = output_dir / "stratified_tradeoff_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ 对比报告: {report_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("分层权衡检测")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 路径设置
    script_dir = Path(__file__).parent.absolute()
    analysis_dir = script_dir.parent.parent

    ate_dir = analysis_dir / "results" / "energy_research" / "stratified" / "ate"
    output_dir = analysis_dir / "results" / "energy_research" / "stratified" / "tradeoff"

    print(f"\nATE目录: {ate_dir}")
    print(f"输出目录: {output_dir}")

    # 检查ATE结果是否存在
    if not ate_dir.exists():
        print(f"\n❌ ATE目录不存在: {ate_dir}")
        print("请先运行: python scripts/stratified/compute_ate_stratified.py")
        return 1

    # 检查FDR校正结果
    fdr_file = ate_dir / "all_layers_ate_fdr_corrected.csv"
    use_fdr = fdr_file.exists()
    if use_fdr:
        print(f"\n✅ 使用FDR校正后的结果")
    else:
        print(f"\n⚠️ FDR校正结果不存在，使用原始p值")

    # 处理每个分层
    all_results = []

    for layer_id in STRATIFIED_LAYERS:
        # 检查ATE结果是否存在
        layer_ate_dir = ate_dir / layer_id
        has_raw = (layer_ate_dir / f"{layer_id}_ate_raw.csv").exists()

        if not has_raw and not use_fdr:
            print(f"\n⚠️ 跳过 {layer_id}（无ATE结果）")
            continue

        try:
            result = detect_tradeoffs_for_layer(
                layer_id=layer_id,
                ate_dir=ate_dir,
                output_dir=output_dir,
                use_fdr=use_fdr
            )
            all_results.append(result)

        except Exception as e:
            print(f"\n❌ {layer_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "success": False,
                "layer_id": layer_id,
                "error": str(e)
            })

    # 生成对比报告
    successful_results = [r for r in all_results if r.get("success")]
    if len(successful_results) >= 2:
        generate_comparison_report(successful_results, output_dir)
    else:
        print(f"\n⚠️ 成功的分层数不足，无法生成对比报告 ({len(successful_results)}/4)")

    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")

    successful = len(successful_results)
    failed = len(all_results) - successful
    total_tradeoffs = sum(len(r.get("tradeoffs", [])) for r in successful_results)

    print(f"\n处理完成:")
    print(f"  成功: {successful}/{len(STRATIFIED_LAYERS)}")
    print(f"  失败: {failed}/{len(STRATIFIED_LAYERS)}")
    print(f"  总权衡数: {total_tradeoffs}")

    print(f"\n✅ 分层权衡检测完成！")
    print(f"结果保存在: {output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
