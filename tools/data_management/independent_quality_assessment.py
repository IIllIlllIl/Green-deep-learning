#!/usr/bin/env python3
"""
Independent Data Quality Assessment Script
Analyzes data/raw_data.csv without referencing existing reports
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

def load_data(filepath):
    """Load CSV data"""
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} records from {filepath}")
    return df

def analyze_completeness(df):
    """Analyze data completeness and missing values"""
    print("\n" + "="*80)
    print("DATA COMPLETENESS ANALYSIS")
    print("="*80)

    total_records = len(df)
    print(f"\nTotal Records: {total_records}")
    print(f"Total Columns: {len(df.columns)}")

    # Key field groups
    critical_fields = {
        'Identifiers': ['experiment_id', 'timestamp', 'repository', 'model'],
        'Training Status': ['training_success', 'duration_seconds', 'error_message'],
        'Energy Data': ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
                       'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
                       'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
                       'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent'],
        'Performance Metrics': ['perf_accuracy', 'perf_test_accuracy', 'perf_best_val_accuracy',
                               'perf_map', 'perf_rank1', 'perf_rank5', 'perf_precision', 'perf_recall',
                               'perf_test_loss', 'perf_eval_loss', 'perf_top1_accuracy', 'perf_top5_accuracy'],
        'Hyperparameters': ['hyperparam_batch_size', 'hyperparam_epochs', 'hyperparam_learning_rate',
                           'hyperparam_dropout', 'hyperparam_weight_decay', 'hyperparam_seed'],
    }

    print("\n--- Missing Data by Field Group ---")
    for group, fields in critical_fields.items():
        existing_fields = [f for f in fields if f in df.columns]
        if not existing_fields:
            print(f"\n{group}: NO FIELDS FOUND")
            continue

        missing_counts = df[existing_fields].isna().sum()
        missing_pct = (missing_counts / total_records * 100).round(2)

        print(f"\n{group}:")
        for field in existing_fields:
            print(f"  {field:40s}: {missing_counts[field]:5d} missing ({missing_pct[field]:6.2f}%)")

    return critical_fields

def analyze_usability(df):
    """Analyze data usability - which records are actually usable"""
    print("\n" + "="*80)
    print("DATA USABILITY ANALYSIS")
    print("="*80)

    total = len(df)

    # Define usability criteria
    print("\n--- Usability Criteria ---")
    print("A record is considered USABLE if it has:")
    print("  1. Training Success = True")
    print("  2. At least one energy metric (energy_gpu_total_joules or energy_cpu_total_joules)")
    print("  3. At least one performance metric (perf_* fields)")

    # Check training success
    has_training_success = df['training_success'].fillna(False) == True
    print(f"\n✓ Training Success: {has_training_success.sum()} / {total} ({has_training_success.sum()/total*100:.1f}%)")

    # Check energy data
    energy_fields = ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
                    'energy_gpu_total_joules', 'energy_gpu_avg_watts']
    has_energy = df[energy_fields].notna().any(axis=1)
    print(f"✓ Has Energy Data: {has_energy.sum()} / {total} ({has_energy.sum()/total*100:.1f}%)")

    # Check performance metrics
    perf_fields = [col for col in df.columns if col.startswith('perf_')]
    has_perf = df[perf_fields].notna().any(axis=1)
    print(f"✓ Has Performance Metrics: {has_perf.sum()} / {total} ({has_perf.sum()/total*100:.1f}%)")

    # Combined usability
    fully_usable = has_training_success & has_energy & has_perf
    print(f"\n{'='*80}")
    print(f"FULLY USABLE RECORDS: {fully_usable.sum()} / {total} ({fully_usable.sum()/total*100:.1f}%)")
    print(f"{'='*80}")

    # Categorize unusable records
    print("\n--- Unusable Records Breakdown ---")
    unusable = ~fully_usable
    print(f"Total Unusable: {unusable.sum()} ({unusable.sum()/total*100:.1f}%)")

    # Reasons for being unusable
    reasons = []
    for idx, row in df[unusable].iterrows():
        reason_list = []
        if not row['training_success']:
            reason_list.append('training_failed')
        if not has_energy[idx]:
            reason_list.append('no_energy_data')
        if not has_perf[idx]:
            reason_list.append('no_performance_metrics')
        reasons.append(','.join(reason_list) if reason_list else 'unknown')

    reason_counts = Counter(reasons)
    print("\nReasons for Unusability:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:50s}: {count:5d} ({count/unusable.sum()*100:5.1f}%)")

    return fully_usable, unusable

def analyze_by_model(df, fully_usable):
    """Analyze usability by model"""
    print("\n" + "="*80)
    print("MODEL-LEVEL USABILITY ANALYSIS")
    print("="*80)

    # Handle parallel mode records
    df_analysis = df.copy()
    df_analysis['model_key'] = df_analysis.apply(
        lambda row: f"{row['fg_repository']}/{row['fg_model']}"
        if pd.notna(row['mode']) and row['mode'] == 'parallel'
        else f"{row['repository']}/{row['model']}",
        axis=1
    )

    model_stats = []
    for model_key in sorted(df_analysis['model_key'].unique()):
        if pd.isna(model_key) or '/' not in str(model_key):
            continue

        mask = df_analysis['model_key'] == model_key
        total_records = mask.sum()
        usable_records = (mask & fully_usable).sum()
        usability_pct = usable_records / total_records * 100 if total_records > 0 else 0

        model_stats.append({
            'model': model_key,
            'total': total_records,
            'usable': usable_records,
            'unusable': total_records - usable_records,
            'usability_pct': usability_pct
        })

    model_df = pd.DataFrame(model_stats).sort_values('usability_pct', ascending=False)

    print("\n--- Models by Usability (sorted by usability %) ---")
    print(f"{'Model':<50s} {'Total':>8s} {'Usable':>8s} {'Unusable':>8s} {'Usability %':>12s}")
    print("-" * 100)

    for _, row in model_df.iterrows():
        quality = "⭐⭐⭐" if row['usability_pct'] == 100 else "⭐⭐" if row['usability_pct'] >= 80 else "⭐" if row['usability_pct'] >= 50 else "⚠️"
        print(f"{row['model']:<50s} {row['total']:>8d} {row['usable']:>8d} {row['unusable']:>8d} {row['usability_pct']:>11.1f}% {quality}")

    # Group by quality
    print("\n--- Models Grouped by Quality ---")
    excellent = model_df[model_df['usability_pct'] == 100]
    good = model_df[(model_df['usability_pct'] >= 80) & (model_df['usability_pct'] < 100)]
    fair = model_df[(model_df['usability_pct'] >= 50) & (model_df['usability_pct'] < 80)]
    poor = model_df[model_df['usability_pct'] < 50]

    print(f"\n⭐⭐⭐ EXCELLENT (100% usable): {len(excellent)} models, {excellent['total'].sum()} records")
    if len(excellent) > 0:
        for model in excellent['model'].values:
            total = excellent[excellent['model'] == model]['total'].values[0]
            print(f"    - {model} ({total} records)")

    print(f"\n⭐⭐ GOOD (80-99% usable): {len(good)} models, {good['total'].sum()} records")
    if len(good) > 0:
        for _, row in good.iterrows():
            print(f"    - {row['model']} ({row['usable']}/{row['total']} = {row['usability_pct']:.1f}%)")

    print(f"\n⭐ FAIR (50-79% usable): {len(fair)} models, {fair['total'].sum()} records")
    if len(fair) > 0:
        for _, row in fair.iterrows():
            print(f"    - {row['model']} ({row['usable']}/{row['total']} = {row['usability_pct']:.1f}%)")

    print(f"\n⚠️ POOR (<50% usable): {len(poor)} models, {poor['total'].sum()} records")
    if len(poor) > 0:
        for _, row in poor.iterrows():
            print(f"    - {row['model']} ({row['usable']}/{row['total']} = {row['usability_pct']:.1f}%)")

    return model_df

def identify_quality_issues(df, unusable):
    """Identify specific data quality issues"""
    print("\n" + "="*80)
    print("DATA QUALITY ISSUES IDENTIFICATION")
    print("="*80)

    issues = []

    # Issue 1: Records with missing repository/model info
    missing_model_info = df['repository'].isna() | df['model'].isna()
    if missing_model_info.sum() > 0:
        issues.append({
            'priority': 'P0',
            'issue': 'Missing Model Information',
            'count': missing_model_info.sum(),
            'description': 'Records without repository or model information cannot be properly categorized',
            'affected_records': df[missing_model_info]['experiment_id'].tolist()[:10]
        })

    # Issue 2: Training succeeded but no performance metrics
    training_ok_no_perf = (df['training_success'] == True) & (~df[[col for col in df.columns if col.startswith('perf_')]].notna().any(axis=1))
    if training_ok_no_perf.sum() > 0:
        issues.append({
            'priority': 'P1',
            'issue': 'Training Success but No Performance Metrics',
            'count': training_ok_no_perf.sum(),
            'description': 'Training completed successfully but performance metrics were not captured',
            'affected_models': df[training_ok_no_perf].groupby(['repository', 'model']).size().to_dict()
        })

    # Issue 3: Training succeeded but no energy data
    training_ok_no_energy = (df['training_success'] == True) & (~df[['energy_cpu_total_joules', 'energy_gpu_total_joules']].notna().any(axis=1))
    if training_ok_no_energy.sum() > 0:
        issues.append({
            'priority': 'P1',
            'issue': 'Training Success but No Energy Data',
            'count': training_ok_no_energy.sum(),
            'description': 'Training completed but energy monitoring failed',
            'affected_models': df[training_ok_no_energy].groupby(['repository', 'model']).size().to_dict()
        })

    # Issue 4: Duplicate experiment_ids
    dup_exp_ids = df['experiment_id'].value_counts()
    duplicates = dup_exp_ids[dup_exp_ids > 1]
    if len(duplicates) > 0:
        issues.append({
            'priority': 'P2',
            'issue': 'Duplicate Experiment IDs',
            'count': duplicates.sum() - len(duplicates),  # Extra records beyond first occurrence
            'description': 'Same experiment_id appears multiple times (may be legitimate for repeated runs)',
            'examples': duplicates.head(10).to_dict()
        })

    # Issue 5: Training failed records
    training_failed = df['training_success'] == False
    if training_failed.sum() > 0:
        issues.append({
            'priority': 'P2',
            'issue': 'Training Failed',
            'count': training_failed.sum(),
            'description': 'Training did not complete successfully',
            'affected_models': df[training_failed].groupby(['repository', 'model']).size().to_dict()
        })

    # Print issues
    for i, issue in enumerate(issues, 1):
        print(f"\n--- Issue #{i}: {issue['issue']} [{issue['priority']}] ---")
        print(f"Count: {issue['count']}")
        print(f"Description: {issue['description']}")
        if 'affected_models' in issue:
            print("Affected models (top 10):")
            sorted_models = sorted(issue['affected_models'].items(), key=lambda x: x[1], reverse=True)[:10]
            for model, count in sorted_models:
                print(f"  {model}: {count} records")
        if 'affected_records' in issue:
            print(f"Sample affected records: {issue['affected_records']}")
        if 'examples' in issue:
            print(f"Examples: {issue['examples']}")

    return issues

def provide_recommendations(df, fully_usable, issues, model_df):
    """Provide prioritized recommendations"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    total = len(df)
    usable_count = fully_usable.sum()

    print("\n--- Priority-Based Repair Recommendations ---")

    # P0 recommendations
    print("\n🔴 P0 - CRITICAL (Must Fix):")
    p0_issues = [i for i in issues if i['priority'] == 'P0']
    if p0_issues:
        for issue in p0_issues:
            print(f"  ❌ {issue['issue']}: {issue['count']} records")
            print(f"     → Action: Investigate and fix data collection pipeline")
            print(f"     → Impact: Critical for data integrity")
    else:
        print("  ✓ No critical issues found")

    # P1 recommendations
    print("\n🟡 P1 - HIGH PRIORITY (Should Fix):")
    p1_issues = [i for i in issues if i['priority'] == 'P1']
    if p1_issues:
        for issue in p1_issues:
            print(f"  ⚠️  {issue['issue']}: {issue['count']} records")
            if 'Training Success but No Performance Metrics' in issue['issue']:
                print(f"     → Action: Review metric collection code for affected models")
                print(f"     → Feasibility: May require re-running experiments if logs unavailable")
                print(f"     → Potential Gain: +{issue['count']} usable records")
            elif 'Training Success but No Energy Data' in issue['issue']:
                print(f"     → Action: Check energy monitoring setup (perf permissions, GPU drivers)")
                print(f"     → Feasibility: May be recoverable from experiment logs")
                print(f"     → Potential Gain: +{issue['count']} records with energy data")
    else:
        print("  ✓ No high priority issues found")

    # P2 recommendations
    print("\n🟢 P2 - MEDIUM PRIORITY (Nice to Have):")
    p2_issues = [i for i in issues if i['priority'] == 'P2']
    if p2_issues:
        for issue in p2_issues:
            print(f"  ℹ️  {issue['issue']}: {issue['count']} records")
            if 'Training Failed' in issue['issue']:
                print(f"     → Action: Analyze error messages, may need to re-run with fixes")
                print(f"     → Feasibility: Depends on failure reasons")
            elif 'Duplicate' in issue['issue']:
                print(f"     → Action: Verify if duplicates are intentional (repeated runs)")
                print(f"     → Feasibility: Easy to detect, may be expected behavior")

    print("\n--- Data Usage Recommendations ---")

    # Recommend high-quality subset
    excellent_models = model_df[model_df['usability_pct'] == 100]
    if len(excellent_models) > 0:
        print(f"\n⭐⭐⭐ RECOMMENDED: High-Quality Subset")
        print(f"  Models with 100% usability: {len(excellent_models)} models")
        print(f"  Total records: {excellent_models['total'].sum()}")
        print(f"  Use cases: Primary analysis, model training, publication-quality results")
        print(f"  Models:")
        for model in excellent_models['model'].values:
            total = excellent_models[excellent_models['model'] == model]['total'].values[0]
            print(f"    - {model} ({total} records)")

    # Balanced subset
    good_or_better = model_df[model_df['usability_pct'] >= 80]
    if len(good_or_better) > 0:
        print(f"\n⭐⭐ BALANCED: Good Quality Subset")
        print(f"  Models with ≥80% usability: {len(good_or_better)} models")
        print(f"  Total records: {good_or_better['usable'].sum()} usable / {good_or_better['total'].sum()} total")
        print(f"  Use cases: Extended analysis with acceptable quality")

    # All available
    print(f"\n⭐ MAXIMUM: All Available Data")
    print(f"  Total usable records: {usable_count} / {total} ({usable_count/total*100:.1f}%)")
    print(f"  Use cases: Exploratory analysis, maximizing sample size")
    print(f"  ⚠️  Note: May include models with data quality issues")

    # Expected improvements
    print("\n--- Expected Impact of Repairs ---")

    p1_total = sum(issue['count'] for issue in p1_issues)
    if p1_total > 0:
        potential_usable = usable_count + p1_total
        print(f"  Current usability: {usable_count}/{total} ({usable_count/total*100:.1f}%)")
        print(f"  After P1 fixes: ~{potential_usable}/{total} ({potential_usable/total*100:.1f}%)")
        print(f"  Improvement: +{p1_total} records (+{p1_total/total*100:.1f}%)")
    else:
        print("  ✓ No significant improvements expected from repairs")
        print("  → Focus on collecting new high-quality data for underrepresented models")

def generate_summary(df, fully_usable, unusable, model_df):
    """Generate executive summary"""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)

    total = len(df)
    usable = fully_usable.sum()
    unusable_count = unusable.sum()

    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total Records: {total}")
    print(f"  Fully Usable: {usable} ({usable/total*100:.1f}%)")
    print(f"  Unusable: {unusable_count} ({unusable_count/total*100:.1f}%)")

    print(f"\n🎯 KEY FINDINGS")

    # Energy data
    has_energy = df[['energy_cpu_total_joules', 'energy_gpu_total_joules']].notna().any(axis=1).sum()
    print(f"  1. Energy Data Coverage: {has_energy}/{total} ({has_energy/total*100:.1f}%)")

    # Performance data
    perf_fields = [col for col in df.columns if col.startswith('perf_')]
    has_perf = df[perf_fields].notna().any(axis=1).sum()
    print(f"  2. Performance Metrics Coverage: {has_perf}/{total} ({has_perf/total*100:.1f}%)")

    # Training success
    training_ok = (df['training_success'] == True).sum()
    print(f"  3. Training Success Rate: {training_ok}/{total} ({training_ok/total*100:.1f}%)")

    # Model quality
    excellent = len(model_df[model_df['usability_pct'] == 100])
    total_models = len(model_df)
    print(f"  4. High-Quality Models: {excellent}/{total_models} models with 100% usability")

    print(f"\n⚠️  TOP 3 ISSUES")
    training_ok_no_perf = ((df['training_success'] == True) &
                           (~df[perf_fields].notna().any(axis=1))).sum()
    training_ok_no_energy = ((df['training_success'] == True) &
                             (~df[['energy_cpu_total_joules', 'energy_gpu_total_joules']].notna().any(axis=1))).sum()
    training_failed = (df['training_success'] == False).sum()

    print(f"  1. Missing Performance Metrics: {training_ok_no_perf} records ({training_ok_no_perf/total*100:.1f}%)")
    print(f"  2. Missing Energy Data: {training_ok_no_energy} records ({training_ok_no_energy/total*100:.1f}%)")
    print(f"  3. Training Failures: {training_failed} records ({training_failed/total*100:.1f}%)")

    print(f"\n✅ RECOMMENDATIONS")
    if excellent > 0:
        excellent_records = model_df[model_df['usability_pct'] == 100]['total'].sum()
        print(f"  → PRIMARY: Use {excellent} high-quality models ({excellent_records} records) for main analysis")
    print(f"  → PRIORITY: Fix P1 issues to recover ~{training_ok_no_perf + training_ok_no_energy} records")
    print(f"  → FUTURE: Improve data collection pipeline to prevent metric/energy capture failures")

def main():
    """Main execution"""
    print("="*80)
    print("INDEPENDENT DATA QUALITY ASSESSMENT")
    print("Target: data/raw_data.csv")
    print("="*80)

    # Load data
    filepath = '/home/green/energy_dl/nightly/data/raw_data.csv'
    df = load_data(filepath)

    # Run analyses
    critical_fields = analyze_completeness(df)
    fully_usable, unusable = analyze_usability(df)
    model_df = analyze_by_model(df, fully_usable)
    issues = identify_quality_issues(df, unusable)
    provide_recommendations(df, fully_usable, issues, model_df)
    generate_summary(df, fully_usable, unusable, model_df)

    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
