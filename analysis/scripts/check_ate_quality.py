#!/usr/bin/env python3
"""
检查白名单ATE数据质量
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def check_whitelist_quality(file_path):
    """检查单个白名单文件的ATE数据质量"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取失败: {file_path}: {e}")
        return None

    filename = os.path.basename(file_path)
    total_edges = len(df)

    # 检查ATE相关列是否存在
    ate_cols = ['ate', 'ate_ci_lower', 'ate_ci_upper', 'ate_is_significant']
    has_ate_cols = all(col in df.columns for col in ate_cols)

    if not has_ate_cols:
        print(f"❌ {filename}: 缺少ATE列")
        return {
            'file': filename,
            'total_edges': total_edges,
            'has_ate': 0,
            'has_ate_pct': 0,
            'significant': 0,
            'significant_pct': 0,
            'ate_mean': np.nan,
            'ate_std': np.nan
        }

    # 统计ATE数据
    has_ate = df['ate'].notna().sum()
    has_ate_pct = has_ate / total_edges * 100 if total_edges > 0 else 0

    # 统计显著边
    if 'ate_is_significant' in df.columns:
        # 处理可能的类型问题
        significant_col = df['ate_is_significant']
        # 转换为布尔值
        if significant_col.dtype == object:
            significant = significant_col.fillna(False).astype(bool).sum()
        else:
            significant = significant_col.fillna(False).sum()
    else:
        significant = 0

    significant_pct = significant / has_ate * 100 if has_ate > 0 else 0

    # ATE统计
    ate_values = df['ate'].dropna()
    ate_mean = ate_values.mean() if len(ate_values) > 0 else np.nan
    ate_std = ate_values.std() if len(ate_values) > 1 else np.nan

    return {
        'file': filename,
        'total_edges': total_edges,
        'has_ate': has_ate,
        'has_ate_pct': has_ate_pct,
        'significant': significant,
        'significant_pct': significant_pct,
        'ate_mean': ate_mean,
        'ate_std': ate_std
    }

def main():
    whitelist_dir = Path(__file__).parent.parent / 'results' / 'energy_research' / 'data' / 'interaction' / 'whitelist'

    if not whitelist_dir.exists():
        print(f"❌ 白名单目录不存在: {whitelist_dir}")
        return

    # 获取所有白名单CSV文件
    whitelist_files = list(whitelist_dir.glob('*.csv'))
    if not whitelist_files:
        print("❌ 未找到白名单CSV文件")
        return

    print("📊 白名单ATE数据质量检查")
    print("=" * 80)

    results = []
    for file_path in sorted(whitelist_files):
        result = check_whitelist_quality(file_path)
        if result:
            results.append(result)

    # 打印详细结果
    for result in results:
        print(f"\n📁 {result['file']}")
        print(f"   总边数: {result['total_edges']}")
        print(f"   有ATE的边: {result['has_ate']} ({result['has_ate_pct']:.1f}%)")
        if result['has_ate'] > 0:
            print(f"   显著边: {result['significant']} ({result['significant_pct']:.1f}%)")
            print(f"   ATE均值: {result['ate_mean']:.3f}")
            print(f"   ATE标准差: {result['ate_std']:.3f}")
        else:
            print(f"   ⚠ 无ATE数据")

    # 汇总统计
    print("\n" + "=" * 80)
    print("📈 汇总统计")

    total_all_edges = sum(r['total_edges'] for r in results)
    total_has_ate = sum(r['has_ate'] for r in results)
    total_significant = sum(r['significant'] for r in results)

    overall_has_ate_pct = total_has_ate / total_all_edges * 100 if total_all_edges > 0 else 0
    overall_significant_pct = total_significant / total_has_ate * 100 if total_has_ate > 0 else 0

    print(f"   总边数: {total_all_edges}")
    print(f"   总ATE计算成功: {total_has_ate} ({overall_has_ate_pct:.1f}%)")
    print(f"   总显著边: {total_significant} ({overall_significant_pct:.1f}%)")

    # 检查是否有问题
    if overall_has_ate_pct < 50:
        print(f"\n⚠️ 警告: ATE计算成功率较低 ({overall_has_ate_pct:.1f}%)")
        print("   可能原因:")
        print("   - 部分边在因果图中权重低于阈值 (0.3)")
        print("   - 存在循环依赖被跳过")
        print("   - 数据中存在NaN值")
    else:
        print(f"\n✅ ATE计算成功率良好 ({overall_has_ate_pct:.1f}%)")

    # 检查ATE值范围
    ate_values = []
    for result in results:
        if not pd.isna(result['ate_mean']):
            ate_values.append(abs(result['ate_mean']))

    if ate_values:
        max_ate = max(ate_values)
        if max_ate > 1000:
            print(f"⚠️ 警告: 部分ATE值较大 (最大绝对值: {max_ate:.1f})")
            print("   这可能表示:")
            print("   - 变量尺度差异较大 (如joules vs. accuracy)")
            print("   - 因果效应确实很强")
            print("   - 建议检查ATE单位一致性")
        else:
            print(f"✅ ATE值范围合理 (最大绝对值: {max_ate:.1f})")

if __name__ == '__main__':
    main()