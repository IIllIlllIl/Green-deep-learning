#!/usr/bin/env python3
"""
DiBS训练数据质量评估脚本

评估6组DiBS训练数据的质量，包括样本量、特征完整性、数据分布等
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_group_data(csv_path, group_info):
    """分析单个组的数据质量"""
    df = pd.read_csv(csv_path)

    analysis = {
        'group_id': group_info['group_id'],
        'group_name': group_info['group_name'],
        'n_samples': len(df),
        'n_features': len(df.columns),
    }

    # 1. 检查缺失值（DiBS严格要求无缺失）
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    analysis['missing_values'] = {
        'total_missing': int(missing_counts.sum()),
        'features_with_missing': int((missing_counts > 0).sum()),
        'missing_rate': float(missing_counts.sum() / (len(df) * len(df.columns))),
        'worst_features': dict(missing_pct[missing_pct > 0].sort_values(ascending=False).head(5))
    }

    # 2. 检查特征方差（识别零方差或近零方差特征）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    stds = df[numeric_cols].std()

    # 标准化方差（相对于均值）
    means = df[numeric_cols].mean()
    cv = (stds / means.abs()).replace([np.inf, -np.inf], np.nan)  # 变异系数

    zero_var_features = variances[variances == 0].index.tolist()
    near_zero_var_features = variances[(variances > 0) & (variances < 1e-10)].index.tolist()
    low_cv_features = cv[(cv < 0.01) & (cv > 0)].index.tolist()  # CV < 1%

    analysis['feature_variance'] = {
        'zero_variance_features': zero_var_features,
        'near_zero_variance_features': near_zero_var_features,
        'low_cv_features': low_cv_features,
        'n_constant_features': len(zero_var_features),
        'n_low_variance_features': len(near_zero_var_features) + len(low_cv_features)
    }

    # 3. 数据分布统计
    stats_dict = {}
    for col in numeric_cols:
        col_stats = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
        }
        # 检测异常值（使用IQR方法）
        iqr = col_stats['q75'] - col_stats['q25']
        lower_bound = col_stats['q25'] - 3 * iqr
        upper_bound = col_stats['q75'] + 3 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        col_stats['n_outliers'] = len(outliers)
        col_stats['outlier_rate'] = float(len(outliers) / len(df))

        stats_dict[col] = col_stats

    analysis['distribution_stats'] = stats_dict

    # 4. 特征分类统计
    hyperparams = [col for col in df.columns if col.startswith('hyperparam_')]
    energy_features = [col for col in df.columns if col.startswith('energy_')]
    perf_features = [col for col in df.columns if col.startswith('perf_')]
    control_features = [col for col in df.columns if col in ['duration_seconds', 'num_mutated_params']]

    analysis['feature_categories'] = {
        'hyperparameters': hyperparams,
        'energy_metrics': energy_features,
        'performance_metrics': perf_features,
        'control_variables': control_features,
        'n_hyperparams': len(hyperparams),
        'n_energy': len(energy_features),
        'n_performance': len(perf_features),
        'n_controls': len(control_features)
    }

    # 5. 超参数多样性检查
    hyperparam_diversity = {}
    for hp in hyperparams:
        unique_vals = df[hp].nunique()
        total_vals = len(df[hp].dropna())
        hyperparam_diversity[hp] = {
            'n_unique': int(unique_vals),
            'n_total': int(total_vals),
            'diversity_ratio': float(unique_vals / total_vals) if total_vals > 0 else 0
        }
    analysis['hyperparam_diversity'] = hyperparam_diversity

    # 6. 数据质量评级
    rating = evaluate_quality(analysis, group_info)
    analysis['quality_rating'] = rating

    return analysis, df

def evaluate_quality(analysis, group_info):
    """评估数据质量并给出评级"""
    rating = {
        'overall': 'unknown',
        'sample_size': 'unknown',
        'missing_data': 'unknown',
        'feature_coverage': 'unknown',
        'hyperparam_coverage': 'unknown',
        'dibs_ready': False,
        'recommendations': []
    }

    n_samples = analysis['n_samples']
    n_features = analysis['n_features']
    n_hyperparams = analysis['feature_categories']['n_hyperparams']
    missing_rate = analysis['missing_values']['missing_rate']

    # 样本量评级
    if n_samples >= 50:
        rating['sample_size'] = '优秀'
    elif n_samples >= 30:
        rating['sample_size'] = '可用'
        rating['recommendations'].append(f"样本量为{n_samples}，建议增加到50+以提高稳定性")
    else:
        rating['sample_size'] = '不推荐'
        rating['recommendations'].append(f"样本量仅{n_samples}，低于DiBS最低要求(30)，强烈建议增加样本")

    # 缺失值评级（DiBS严格要求）
    if missing_rate == 0:
        rating['missing_data'] = '优秀'
    elif missing_rate < 0.01:
        rating['missing_data'] = '良好'
        rating['recommendations'].append(f"存在{analysis['missing_values']['total_missing']}个缺失值，需要在DiBS训练前处理")
    else:
        rating['missing_data'] = '不可用'
        rating['recommendations'].append(f"缺失率{missing_rate:.2%}过高，必须处理后才能用于DiBS")

    # 特征数量评级
    if n_features >= 15:
        rating['feature_coverage'] = '优秀'
    elif n_features >= 10:
        rating['feature_coverage'] = '良好'
    else:
        rating['feature_coverage'] = '一般'
        rating['recommendations'].append(f"特征数仅{n_features}，可能限制因果发现能力")

    # 超参数覆盖评级
    if n_hyperparams >= 3:
        rating['hyperparam_coverage'] = '充分'
    elif n_hyperparams >= 2:
        rating['hyperparam_coverage'] = '基本'
    elif n_hyperparams >= 1:
        rating['hyperparam_coverage'] = '不足'
        rating['recommendations'].append(f"超参数仅{n_hyperparams}个，分析能力受限")
    else:
        rating['hyperparam_coverage'] = '无'
        rating['recommendations'].append("无超参数数据，只能分析能耗-性能关系，不能分析超参数影响")

    # 零方差特征警告
    n_constant = analysis['feature_variance']['n_constant_features']
    if n_constant > 0:
        rating['recommendations'].append(f"存在{n_constant}个常数特征，需要在DiBS训练前移除")

    # DiBS就绪判断
    dibs_ready_criteria = [
        n_samples >= 30,
        missing_rate == 0,
        n_constant == 0,
        n_features >= 10
    ]
    rating['dibs_ready'] = all(dibs_ready_criteria)

    # 总体评级
    if rating['dibs_ready'] and n_samples >= 50 and n_hyperparams >= 3:
        rating['overall'] = '优秀'
    elif rating['dibs_ready'] and n_samples >= 30:
        rating['overall'] = '良好'
    elif not missing_rate == 0 or n_constant > 0:
        rating['overall'] = '需要清理'
    elif n_samples < 30:
        rating['overall'] = '不推荐'
    else:
        rating['overall'] = '可用'

    return rating

def generate_markdown_report(all_analyses, stats_file):
    """生成详细的Markdown评估报告"""

    report = []
    report.append("# DiBS训练数据质量评估报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## 执行摘要")
    report.append("")

    # 总体统计
    total_samples = sum(a['n_samples'] for a in all_analyses)
    report.append(f"- **总样本数**: {total_samples}")
    report.append(f"- **分组数量**: {len(all_analyses)}")
    report.append("")

    # 快速评级总览
    report.append("### 快速评级总览")
    report.append("")
    report.append("| 组别 | 样本量 | 特征数 | 超参数 | 缺失值 | 总体评级 | DiBS就绪 |")
    report.append("|------|--------|--------|--------|--------|----------|----------|")

    for analysis in all_analyses:
        rating = analysis['quality_rating']
        group_name = analysis['group_name']
        n_samples = analysis['n_samples']
        n_features = analysis['n_features']
        n_hyperparams = analysis['feature_categories']['n_hyperparams']
        missing_rate = analysis['missing_values']['missing_rate']
        dibs_ready = "✅" if rating['dibs_ready'] else "❌"

        report.append(f"| {group_name} | {n_samples} | {n_features} | {n_hyperparams} | {missing_rate:.1%} | {rating['overall']} | {dibs_ready} |")

    report.append("")
    report.append("---")
    report.append("")

    # 详细分组分析
    report.append("## 详细分组分析")
    report.append("")

    for i, analysis in enumerate(all_analyses, 1):
        rating = analysis['quality_rating']

        report.append(f"### {i}. {analysis['group_name']}")
        report.append("")
        report.append(f"**组ID**: `{analysis['group_id']}`")
        report.append("")

        # 基本信息
        report.append("#### 基本信息")
        report.append("")
        report.append(f"- **样本数量**: {analysis['n_samples']} ({rating['sample_size']})")
        report.append(f"- **特征数量**: {analysis['n_features']} ({rating['feature_coverage']})")
        report.append(f"- **超参数数量**: {analysis['feature_categories']['n_hyperparams']} ({rating['hyperparam_coverage']})")
        report.append(f"- **能耗指标数量**: {analysis['feature_categories']['n_energy']}")
        report.append(f"- **性能指标数量**: {analysis['feature_categories']['n_performance']}")
        report.append("")

        # 数据质量
        report.append("#### 数据质量")
        report.append("")
        report.append(f"- **缺失值**: {analysis['missing_values']['total_missing']} ({rating['missing_data']})")
        report.append(f"  - 缺失率: {analysis['missing_values']['missing_rate']:.2%}")
        report.append(f"  - 有缺失的特征数: {analysis['missing_values']['features_with_missing']}")

        if analysis['missing_values']['worst_features']:
            report.append("  - 缺失最严重的特征:")
            for feat, pct in list(analysis['missing_values']['worst_features'].items())[:3]:
                report.append(f"    - `{feat}`: {pct:.1f}%")

        report.append("")
        report.append(f"- **常数特征**: {analysis['feature_variance']['n_constant_features']}")
        if analysis['feature_variance']['zero_variance_features']:
            report.append("  - 零方差特征:")
            for feat in analysis['feature_variance']['zero_variance_features']:
                report.append(f"    - `{feat}`")

        report.append(f"- **低方差特征**: {analysis['feature_variance']['n_low_variance_features']}")
        report.append("")

        # 超参数详情
        if analysis['feature_categories']['hyperparameters']:
            report.append("#### 超参数覆盖")
            report.append("")
            report.append("| 超参数 | 唯一值数 | 多样性比率 |")
            report.append("|--------|----------|------------|")
            for hp, diversity in analysis['hyperparam_diversity'].items():
                hp_name = hp.replace('hyperparam_', '')
                report.append(f"| {hp_name} | {diversity['n_unique']} | {diversity['diversity_ratio']:.2%} |")
            report.append("")
        else:
            report.append("#### 超参数覆盖")
            report.append("")
            report.append("⚠️ **无超参数数据** - 只能分析能耗-性能关系，不能研究超参数的因果影响")
            report.append("")

        # 异常值检测摘要
        outlier_features = []
        for feat, stats in analysis['distribution_stats'].items():
            if stats['n_outliers'] > 0 and stats['outlier_rate'] > 0.05:  # >5%异常值
                outlier_features.append((feat, stats['n_outliers'], stats['outlier_rate']))

        if outlier_features:
            report.append("#### 异常值检测")
            report.append("")
            report.append("存在较多异常值的特征（>5%）:")
            report.append("")
            outlier_features.sort(key=lambda x: x[2], reverse=True)
            for feat, n, rate in outlier_features[:5]:
                report.append(f"- `{feat}`: {n}个 ({rate:.1%})")
            report.append("")

        # 质量评级
        report.append("#### 质量评级")
        report.append("")
        report.append(f"- **总体评级**: **{rating['overall']}**")
        report.append(f"- **DiBS就绪**: {'✅ 是' if rating['dibs_ready'] else '❌ 否'}")
        report.append("")

        # 建议
        if rating['recommendations']:
            report.append("#### 改进建议")
            report.append("")
            for rec in rating['recommendations']:
                report.append(f"- {rec}")
            report.append("")

        report.append("---")
        report.append("")

    # 总体结论和建议
    report.append("## 总体结论")
    report.append("")

    # 按质量分组
    excellent = [a for a in all_analyses if a['quality_rating']['overall'] == '优秀']
    good = [a for a in all_analyses if a['quality_rating']['overall'] == '良好']
    usable = [a for a in all_analyses if a['quality_rating']['overall'] == '可用']
    needs_cleaning = [a for a in all_analyses if a['quality_rating']['overall'] == '需要清理']
    not_recommended = [a for a in all_analyses if a['quality_rating']['overall'] == '不推荐']

    report.append("### 数据质量分层")
    report.append("")

    if excellent:
        report.append(f"**优秀组 ({len(excellent)}个)** - 可直接用于DiBS分析:")
        for a in excellent:
            report.append(f"- {a['group_name']} ({a['n_samples']}样本, {a['feature_categories']['n_hyperparams']}超参数)")
        report.append("")

    if good:
        report.append(f"**良好组 ({len(good)}个)** - 可用于DiBS分析，略有限制:")
        for a in good:
            report.append(f"- {a['group_name']} ({a['n_samples']}样本, {a['feature_categories']['n_hyperparams']}超参数)")
        report.append("")

    if usable:
        report.append(f"**可用组 ({len(usable)}个)** - 可用但存在局限:")
        for a in usable:
            report.append(f"- {a['group_name']} ({a['n_samples']}样本, {a['feature_categories']['n_hyperparams']}超参数)")
        report.append("")

    if needs_cleaning:
        report.append(f"**需要清理组 ({len(needs_cleaning)}个)** - 需要数据预处理:")
        for a in needs_cleaning:
            issues = []
            if a['missing_values']['missing_rate'] > 0:
                issues.append("有缺失值")
            if a['feature_variance']['n_constant_features'] > 0:
                issues.append("有常数特征")
            report.append(f"- {a['group_name']} ({', '.join(issues)})")
        report.append("")

    if not_recommended:
        report.append(f"**不推荐组 ({len(not_recommended)}个)** - 样本量不足:")
        for a in not_recommended:
            report.append(f"- {a['group_name']} ({a['n_samples']}样本 < 30)")
        report.append("")

    # 关键发现
    report.append("### 关键发现")
    report.append("")

    # 1. 超参数问题
    no_hyperparam_groups = [a for a in all_analyses if a['feature_categories']['n_hyperparams'] == 0]
    if no_hyperparam_groups:
        report.append(f"⚠️ **{len(no_hyperparam_groups)}个组缺少超参数数据**:")
        for a in no_hyperparam_groups:
            report.append(f"- {a['group_name']}")
        report.append("")
        report.append("这些组只能用于能耗-性能关系分析，无法研究超参数的因果影响。")
        report.append("")

    # 2. 样本量问题
    small_sample_groups = [a for a in all_analyses if a['n_samples'] < 50]
    if small_sample_groups:
        report.append(f"⚠️ **{len(small_sample_groups)}个组样本量<50**:")
        for a in small_sample_groups:
            report.append(f"- {a['group_name']}: {a['n_samples']}样本")
        report.append("")
        report.append("DiBS在小样本上可能不够稳定，建议增加样本量或使用交叉验证。")
        report.append("")

    # 3. 缺失值问题
    missing_groups = [a for a in all_analyses if a['missing_values']['missing_rate'] > 0]
    if missing_groups:
        report.append(f"⚠️ **{len(missing_groups)}个组存在缺失值**:")
        for a in missing_groups:
            report.append(f"- {a['group_name']}: {a['missing_values']['missing_rate']:.2%}缺失率")
        report.append("")
        report.append("DiBS需要完整数据，必须在训练前处理缺失值（删除行或插补）。")
        report.append("")

    # 使用建议
    report.append("### 使用建议")
    report.append("")

    report.append("#### 推荐用于DiBS分析的组")
    report.append("")
    ready_groups = [a for a in all_analyses if a['quality_rating']['dibs_ready']]
    if ready_groups:
        report.append(f"以下{len(ready_groups)}个组已准备就绪，可直接用于DiBS因果图学习:")
        report.append("")
        for a in sorted(ready_groups, key=lambda x: x['n_samples'], reverse=True):
            n_hp = a['feature_categories']['n_hyperparams']
            hp_info = f"{n_hp}个超参数" if n_hp > 0 else "无超参数（仅能耗-性能分析）"
            report.append(f"- **{a['group_name']}**: {a['n_samples']}样本, {a['n_features']}特征, {hp_info}")
        report.append("")
    else:
        report.append("⚠️ **当前没有完全就绪的组**，需要数据清理后才能使用。")
        report.append("")

    report.append("#### 研究问题建议")
    report.append("")

    hyperparam_groups = [a for a in ready_groups if a['feature_categories']['n_hyperparams'] >= 3]
    if hyperparam_groups:
        report.append(f"**问题1: 超参数对能耗的影响** - 推荐使用以下{len(hyperparam_groups)}个组:")
        for a in hyperparam_groups:
            report.append(f"- {a['group_name']} ({a['feature_categories']['n_hyperparams']}个超参数)")
        report.append("")

    if ready_groups:
        report.append(f"**问题2: 能耗和性能的权衡关系** - 所有{len(ready_groups)}个就绪组都可用")
        report.append("")

    if hyperparam_groups:
        report.append(f"**问题3: 中间变量的中介效应** - 推荐使用高质量组:")
        for a in hyperparam_groups[:3]:  # 前3个最好的
            report.append(f"- {a['group_name']} (最完整的特征集)")
        report.append("")

    # 数据清理步骤
    if needs_cleaning or missing_groups:
        report.append("#### 数据清理步骤")
        report.append("")
        report.append("在使用DiBS前，需要执行以下清理步骤:")
        report.append("")
        report.append("1. **处理缺失值**:")
        report.append("   ```python")
        report.append("   # 方案A: 删除有缺失值的行（如果缺失较少）")
        report.append("   df_clean = df.dropna()")
        report.append("   ")
        report.append("   # 方案B: 插补（如果缺失较多）")
        report.append("   from sklearn.impute import SimpleImputer")
        report.append("   imputer = SimpleImputer(strategy='median')")
        report.append("   df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)")
        report.append("   ```")
        report.append("")
        report.append("2. **移除常数特征**:")
        report.append("   ```python")
        report.append("   # 移除方差为0的特征")
        report.append("   df_clean = df_clean.loc[:, df_clean.var() > 0]")
        report.append("   ```")
        report.append("")
        report.append("3. **标准化数据** (DiBS推荐):")
        report.append("   ```python")
        report.append("   from sklearn.preprocessing import StandardScaler")
        report.append("   scaler = StandardScaler()")
        report.append("   df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)")
        report.append("   ```")
        report.append("")

    # 附录：评估标准
    report.append("---")
    report.append("")
    report.append("## 附录：评估标准")
    report.append("")
    report.append("### 样本量评级")
    report.append("")
    report.append("- **优秀**: ≥50 样本")
    report.append("- **可用**: 30-49 样本")
    report.append("- **不推荐**: <30 样本 (DiBS最低要求)")
    report.append("")
    report.append("### 缺失值评级")
    report.append("")
    report.append("- **优秀**: 0% 缺失 (DiBS严格要求)")
    report.append("- **良好**: <1% 缺失")
    report.append("- **不可用**: ≥1% 缺失")
    report.append("")
    report.append("### 特征数评级")
    report.append("")
    report.append("- **优秀**: ≥15 特征")
    report.append("- **良好**: 10-14 特征")
    report.append("- **一般**: <10 特征")
    report.append("")
    report.append("### 超参数覆盖评级")
    report.append("")
    report.append("- **充分**: ≥3 超参数")
    report.append("- **基本**: 2 超参数")
    report.append("- **不足**: 1 超参数")
    report.append("- **无**: 0 超参数 (仅能耗-性能分析)")
    report.append("")
    report.append("### DiBS就绪条件")
    report.append("")
    report.append("满足所有以下条件:")
    report.append("- 样本量 ≥30")
    report.append("- 缺失率 = 0%")
    report.append("- 无常数特征")
    report.append("- 特征数 ≥10")
    report.append("")

    return "\n".join(report)

def main():
    """主函数"""
    base_dir = Path('/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training')

    # 读取统计文件
    with open(base_dir / 'generation_stats.json', 'r') as f:
        stats = json.load(f)

    print("=" * 70)
    print("DiBS训练数据质量评估")
    print("=" * 70)
    print()

    # 分析所有组
    all_analyses = []
    all_dataframes = {}

    for task_stat in stats['task_stats']:
        group_id = task_stat['group_id']
        csv_path = base_dir / f"{group_id}.csv"

        print(f"正在分析: {task_stat['group_name']} ({group_id})...")

        analysis, df = analyze_group_data(csv_path, task_stat)
        all_analyses.append(analysis)
        all_dataframes[group_id] = df

        # 打印简要信息
        rating = analysis['quality_rating']
        print(f"  样本数: {analysis['n_samples']}")
        print(f"  特征数: {analysis['n_features']}")
        print(f"  超参数: {analysis['feature_categories']['n_hyperparams']}")
        print(f"  缺失率: {analysis['missing_values']['missing_rate']:.2%}")
        print(f"  评级: {rating['overall']}")
        print(f"  DiBS就绪: {'是' if rating['dibs_ready'] else '否'}")
        print()

    # 生成Markdown报告
    print("生成详细评估报告...")
    report_content = generate_markdown_report(all_analyses, stats)

    report_path = base_dir / 'DATA_QUALITY_ASSESSMENT_20260115.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 报告已生成: {report_path}")
    print()

    # 保存详细分析结果为JSON
    analysis_path = base_dir / 'detailed_quality_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)

    print(f"✅ 详细分析已保存: {analysis_path}")
    print()

    # 打印总结
    print("=" * 70)
    print("评估总结")
    print("=" * 70)

    ready_groups = [a for a in all_analyses if a['quality_rating']['dibs_ready']]
    print(f"DiBS就绪组数: {len(ready_groups)}/{len(all_analyses)}")

    if ready_groups:
        print("\n就绪组:")
        for a in ready_groups:
            print(f"  - {a['group_name']} ({a['n_samples']}样本)")

    needs_work = [a for a in all_analyses if not a['quality_rating']['dibs_ready']]
    if needs_work:
        print(f"\n需要处理的组: {len(needs_work)}")
        for a in needs_work:
            print(f"  - {a['group_name']} ({a['quality_rating']['overall']})")

    print()
    print("详细报告请查看: DATA_QUALITY_ASSESSMENT_20260115.md")
    print()

if __name__ == '__main__':
    main()
