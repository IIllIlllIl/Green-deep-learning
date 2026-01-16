#!/usr/bin/env python3
"""
Independent verification script for hyperparameter backfilling data quality.
This script performs comprehensive checks on the backfilled data.
"""

import pandas as pd
import json
import random
from pathlib import Path
from collections import defaultdict

# File paths
ORIGINAL_DATA = "/home/green/energy_dl/nightly/data/raw_data.csv"
BACKFILLED_DATA = "/home/green/energy_dl/nightly/analysis/data/energy_research/backfilled/raw_data_backfilled.csv"
MODELS_CONFIG = "/home/green/energy_dl/nightly/mutation/models_config.json"
REPORT_PATH = "/home/green/energy_dl/nightly/analysis/data/energy_research/backfilled/independent_verification_report.md"

def load_data():
    """Load all required data files."""
    print("Loading data files...")
    original_df = pd.read_csv(ORIGINAL_DATA)
    backfilled_df = pd.read_csv(BACKFILLED_DATA)

    with open(MODELS_CONFIG, 'r') as f:
        models_config = json.load(f)

    return original_df, backfilled_df, models_config

def verify_data_integrity(original_df, backfilled_df):
    """Verify basic data integrity between original and backfilled data."""
    print("\n=== 1. DATA INTEGRITY VERIFICATION ===")

    issues = []
    checks_passed = []

    # Check row count
    if len(original_df) == len(backfilled_df):
        checks_passed.append(f"Row count matches: {len(original_df)} rows")
    else:
        issues.append(f"Row count mismatch: Original={len(original_df)}, Backfilled={len(backfilled_df)}")

    # Check experiment_id consistency
    orig_ids = set(original_df['experiment_id'].values)
    back_ids = set(backfilled_df['experiment_id'].values)

    if orig_ids == back_ids:
        checks_passed.append(f"All experiment_ids preserved: {len(orig_ids)} unique IDs")
    else:
        missing = orig_ids - back_ids
        extra = back_ids - orig_ids
        if missing:
            issues.append(f"Missing experiment_ids: {missing}")
        if extra:
            issues.append(f"Extra experiment_ids: {extra}")

    # Check timestamp consistency
    if 'timestamp' in original_df.columns and 'timestamp' in backfilled_df.columns:
        orig_ts = original_df.set_index('experiment_id')['timestamp']
        back_ts = backfilled_df.set_index('experiment_id')['timestamp']

        if orig_ts.equals(back_ts):
            checks_passed.append("All timestamps preserved")
        else:
            issues.append("Timestamp mismatch detected")

    # Check that original columns are preserved
    orig_cols = set(original_df.columns)
    back_cols = set(backfilled_df.columns)

    missing_cols = orig_cols - back_cols
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    new_cols = back_cols - orig_cols
    expected_new_cols = {
        'hyperparam_epochs_source', 'hyperparam_max_iter_source',
        'hyperparam_learning_rate_source', 'hyperparam_batch_size_source',
        'hyperparam_dropout_source', 'hyperparam_weight_decay_source',
        'hyperparam_alpha_source', 'hyperparam_kfold_source',
        'hyperparam_seed_source', 'fg_hyperparam_epochs_source',
        'fg_hyperparam_max_iter_source', 'fg_hyperparam_learning_rate_source',
        'fg_hyperparam_batch_size_source', 'fg_hyperparam_dropout_source',
        'fg_hyperparam_weight_decay_source', 'fg_hyperparam_alpha_source',
        'fg_hyperparam_kfold_source', 'fg_hyperparam_seed_source'
    }

    if new_cols == expected_new_cols:
        checks_passed.append(f"All expected source columns added: {len(new_cols)} columns")
    else:
        unexpected = new_cols - expected_new_cols
        missing_expected = expected_new_cols - new_cols
        if unexpected:
            issues.append(f"Unexpected new columns: {unexpected}")
        if missing_expected:
            issues.append(f"Missing expected source columns: {missing_expected}")

    return checks_passed, issues

def verify_backfilled_values(backfilled_df, models_config, sample_size=30):
    """Verify correctness of backfilled values by sampling."""
    print(f"\n=== 2. BACKFILLED VALUES CORRECTNESS (Sample size: {sample_size}) ===")

    issues = []
    checks_passed = []

    # Hyperparameter columns to check
    hyperparam_cols = [
        'hyperparam_epochs', 'hyperparam_max_iter', 'hyperparam_learning_rate',
        'hyperparam_batch_size', 'hyperparam_dropout', 'hyperparam_weight_decay',
        'hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_seed'
    ]

    fg_hyperparam_cols = ['fg_' + col for col in hyperparam_cols]

    all_hyperparam_cols = hyperparam_cols + fg_hyperparam_cols

    # Collect all backfilled cells
    backfilled_cells = []

    for idx, row in backfilled_df.iterrows():
        repo = row['repository']
        fg_repo = row.get('fg_repository', None)

        for col in hyperparam_cols:
            source_col = col + '_source'
            if source_col in row and row[source_col] == 'backfilled':
                backfilled_cells.append({
                    'row_idx': idx,
                    'experiment_id': row['experiment_id'],
                    'repository': repo,
                    'column': col,
                    'value': row[col],
                    'is_fg': False
                })

        for col in fg_hyperparam_cols:
            source_col = col + '_source'
            if source_col in row and row[source_col] == 'backfilled':
                backfilled_cells.append({
                    'row_idx': idx,
                    'experiment_id': row['experiment_id'],
                    'repository': fg_repo,
                    'column': col,
                    'value': row[col],
                    'is_fg': True
                })

    print(f"Total backfilled cells found: {len(backfilled_cells)}")

    if len(backfilled_cells) == 0:
        issues.append("No backfilled cells found - this is unexpected")
        return checks_passed, issues

    # Sample random cells
    sample_size = min(sample_size, len(backfilled_cells))
    sampled_cells = random.sample(backfilled_cells, sample_size)

    correct_count = 0
    incorrect_count = 0

    for cell in sampled_cells:
        repo = cell['repository']
        col = cell['column']
        value = cell['value']
        is_fg = cell['is_fg']

        # Extract hyperparam name
        if is_fg:
            hyperparam_name = col.replace('fg_hyperparam_', '')
        else:
            hyperparam_name = col.replace('hyperparam_', '')

        # Get expected default value from models_config
        if repo and repo in models_config['models']:
            repo_config = models_config['models'][repo]
            supported_params = repo_config.get('supported_hyperparams', {})

            if hyperparam_name in supported_params:
                expected_default = supported_params[hyperparam_name]['default']

                # Compare values (handle float precision)
                if isinstance(expected_default, float):
                    if abs(float(value) - expected_default) < 1e-9:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                        issues.append(
                            f"INCORRECT: {cell['experiment_id']}, {col}, "
                            f"Expected: {expected_default}, Got: {value}"
                        )
                else:
                    if value == expected_default or str(value) == str(expected_default):
                        correct_count += 1
                    else:
                        incorrect_count += 1
                        issues.append(
                            f"INCORRECT: {cell['experiment_id']}, {col}, "
                            f"Expected: {expected_default}, Got: {value}"
                        )
            else:
                issues.append(
                    f"WARNING: {hyperparam_name} not in supported_hyperparams for {repo}"
                )
        else:
            issues.append(f"WARNING: Repository {repo} not found in models_config")

    checks_passed.append(f"Sampled {sample_size} backfilled cells")
    checks_passed.append(f"Correct values: {correct_count}/{sample_size} ({100*correct_count/sample_size:.1f}%)")

    if incorrect_count > 0:
        issues.append(f"Found {incorrect_count} incorrect backfilled values")

    return checks_passed, issues

def verify_source_tracking(backfilled_df):
    """Verify source column accuracy."""
    print("\n=== 3. SOURCE TRACKING VERIFICATION ===")

    issues = []
    checks_passed = []

    # Check that all source columns exist
    hyperparam_cols = [
        'hyperparam_epochs', 'hyperparam_max_iter', 'hyperparam_learning_rate',
        'hyperparam_batch_size', 'hyperparam_dropout', 'hyperparam_weight_decay',
        'hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_seed'
    ]

    fg_hyperparam_cols = ['fg_' + col for col in hyperparam_cols]
    all_cols = hyperparam_cols + fg_hyperparam_cols

    for col in all_cols:
        source_col = col + '_source'
        if source_col not in backfilled_df.columns:
            issues.append(f"Missing source column: {source_col}")

    # Count source types
    source_stats = defaultdict(lambda: {'recorded': 0, 'backfilled': 0, 'not_applicable': 0, 'empty': 0})

    for col in all_cols:
        source_col = col + '_source'
        if source_col in backfilled_df.columns:
            value_counts = backfilled_df[source_col].value_counts()
            source_stats[col]['recorded'] = value_counts.get('recorded', 0)
            source_stats[col]['backfilled'] = value_counts.get('backfilled', 0)
            source_stats[col]['not_applicable'] = value_counts.get('not_applicable', 0)
            source_stats[col]['empty'] = backfilled_df[source_col].isna().sum()

    # Verify source logic
    for col in all_cols:
        source_col = col + '_source'
        if source_col in backfilled_df.columns:
            for idx, row in backfilled_df.iterrows():
                value = row[col]
                source = row[source_col]

                # Check: if value is NaN, source should be empty or not_applicable
                if pd.isna(value):
                    if source not in ['not_applicable', None] and not pd.isna(source):
                        issues.append(
                            f"Row {idx}: {col} is NaN but source is '{source}' (should be not_applicable or empty)"
                        )
                        break  # Only report first occurrence per column

                # Check: if source is 'recorded' or 'backfilled', value should not be NaN
                elif source in ['recorded', 'backfilled']:
                    if pd.isna(value):
                        issues.append(
                            f"Row {idx}: {col} source is '{source}' but value is NaN"
                        )
                        break

    # Print statistics
    total_recorded = sum(stats['recorded'] for stats in source_stats.values())
    total_backfilled = sum(stats['backfilled'] for stats in source_stats.values())
    total_not_applicable = sum(stats['not_applicable'] for stats in source_stats.values())
    total_cells = len(backfilled_df) * len(all_cols)

    checks_passed.append(f"Total cells: {total_cells}")
    checks_passed.append(f"Recorded: {total_recorded} ({100*total_recorded/total_cells:.2f}%)")
    checks_passed.append(f"Backfilled: {total_backfilled} ({100*total_backfilled/total_cells:.2f}%)")
    checks_passed.append(f"Not applicable: {total_not_applicable} ({100*total_not_applicable/total_cells:.2f}%)")

    return checks_passed, issues, source_stats

def verify_no_overwrite(original_df, backfilled_df):
    """Verify that original values were not overwritten."""
    print("\n=== 4. ORIGINAL VALUES PRESERVATION ===")

    issues = []
    checks_passed = []

    hyperparam_cols = [
        'hyperparam_epochs', 'hyperparam_max_iter', 'hyperparam_learning_rate',
        'hyperparam_batch_size', 'hyperparam_dropout', 'hyperparam_weight_decay',
        'hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_seed'
    ]

    fg_hyperparam_cols = ['fg_' + col for col in hyperparam_cols]
    all_cols = hyperparam_cols + fg_hyperparam_cols

    overwrite_count = 0
    preserved_count = 0

    # Compare row by row using index position
    for idx in range(len(original_df)):
        orig_row = original_df.iloc[idx]
        back_row = backfilled_df.iloc[idx]

        exp_id = orig_row['experiment_id']

        for col in all_cols:
            if col in original_df.columns:
                orig_val = orig_row[col]
                back_val = back_row[col]

                # If original had a value, check if it's preserved
                if not pd.isna(orig_val):
                    if pd.isna(back_val):
                        overwrite_count += 1
                        if overwrite_count <= 10:  # Limit issue reporting
                            issues.append(
                                f"Value removed: {exp_id}, {col}, Original: {orig_val}, Backfilled: NaN"
                            )
                    elif orig_val != back_val:
                        # Handle float comparison
                        if isinstance(orig_val, float) and isinstance(back_val, float):
                            if abs(orig_val - back_val) > 1e-9:
                                overwrite_count += 1
                                if overwrite_count <= 10:  # Limit issue reporting
                                    issues.append(
                                        f"Value changed: {exp_id}, {col}, Original: {orig_val}, Backfilled: {back_val}"
                                    )
                            else:
                                preserved_count += 1
                        else:
                            overwrite_count += 1
                            if overwrite_count <= 10:  # Limit issue reporting
                                issues.append(
                                    f"Value changed: {exp_id}, {col}, Original: {orig_val}, Backfilled: {back_val}"
                                )
                    else:
                        preserved_count += 1

    checks_passed.append(f"Original values preserved: {preserved_count}")

    if overwrite_count > 0:
        issues.append(f"CRITICAL: {overwrite_count} original values were overwritten or removed")
    else:
        checks_passed.append("No original values were overwritten")

    return checks_passed, issues

def main():
    """Main verification function."""
    print("="*80)
    print("INDEPENDENT VERIFICATION OF HYPERPARAMETER BACKFILLING")
    print("="*80)

    # Load data
    original_df, backfilled_df, models_config = load_data()

    all_checks_passed = []
    all_issues = []

    # 1. Data Integrity
    checks, issues = verify_data_integrity(original_df, backfilled_df)
    all_checks_passed.extend(checks)
    all_issues.extend(issues)

    # 2. Backfilled Values Correctness
    checks, issues = verify_backfilled_values(backfilled_df, models_config, sample_size=30)
    all_checks_passed.extend(checks)
    all_issues.extend(issues)

    # 3. Source Tracking
    checks, issues, source_stats = verify_source_tracking(backfilled_df)
    all_checks_passed.extend(checks)
    all_issues.extend(issues)

    # 4. Original Values Preservation
    checks, issues = verify_no_overwrite(original_df, backfilled_df)
    all_checks_passed.extend(checks)
    all_issues.extend(issues)

    # Generate report
    generate_report(all_checks_passed, all_issues, source_stats, original_df, backfilled_df)

    print("\n" + "="*80)
    print(f"Verification complete. Report saved to: {REPORT_PATH}")
    print("="*80)

def generate_report(checks_passed, issues, source_stats, original_df, backfilled_df):
    """Generate markdown verification report."""

    report_lines = []
    report_lines.append("# Independent Verification Report")
    report_lines.append("# Hyperparameter Backfilling Data Quality")
    report_lines.append("")
    report_lines.append(f"**Verification Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Verifier**: Independent Verification Script")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(f"- **Original Data**: {len(original_df)} rows, {len(original_df.columns)} columns")
    report_lines.append(f"- **Backfilled Data**: {len(backfilled_df)} rows, {len(backfilled_df.columns)} columns")
    report_lines.append(f"- **Checks Passed**: {len(checks_passed)}")
    report_lines.append(f"- **Issues Found**: {len(issues)}")
    report_lines.append("")

    if len(issues) == 0:
        report_lines.append("**Overall Assessment**: ✅ **PASSED** - All verification checks passed successfully.")
    else:
        critical_issues = [i for i in issues if 'CRITICAL' in i or 'INCORRECT' in i]
        if critical_issues:
            report_lines.append("**Overall Assessment**: ❌ **FAILED** - Critical issues found that require immediate attention.")
        else:
            report_lines.append("**Overall Assessment**: ⚠️ **PASSED WITH WARNINGS** - Minor issues found but data quality is acceptable.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Detailed Results
    report_lines.append("## Detailed Verification Results")
    report_lines.append("")

    report_lines.append("### ✅ Checks Passed")
    report_lines.append("")
    if checks_passed:
        for check in checks_passed:
            report_lines.append(f"- {check}")
    else:
        report_lines.append("- None")
    report_lines.append("")

    report_lines.append("### ❌ Issues Found")
    report_lines.append("")
    if issues:
        for issue in issues:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("- None")
    report_lines.append("")

    # Source Statistics
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Source Tracking Statistics")
    report_lines.append("")
    report_lines.append("| Column | Recorded | Backfilled | Not Applicable | Empty |")
    report_lines.append("|--------|----------|------------|----------------|-------|")

    for col, stats in sorted(source_stats.items()):
        report_lines.append(
            f"| {col} | {stats['recorded']} | {stats['backfilled']} | "
            f"{stats['not_applicable']} | {stats['empty']} |"
        )

    report_lines.append("")

    # Recommendations
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Recommendations")
    report_lines.append("")

    if len(issues) == 0:
        report_lines.append("✅ The backfilled data has passed all verification checks. The data is ready for use in regression analysis.")
    else:
        critical_issues = [i for i in issues if 'CRITICAL' in i or 'INCORRECT' in i]
        if critical_issues:
            report_lines.append("❌ **Critical issues detected**. Please review and fix the following before proceeding:")
            report_lines.append("")
            for issue in critical_issues:
                report_lines.append(f"- {issue}")
            report_lines.append("")
            report_lines.append("**Action Required**: Re-run the backfilling script after fixing the issues.")
        else:
            report_lines.append("⚠️ Minor warnings detected. Review the issues above to determine if they require attention.")
            report_lines.append("")
            report_lines.append("The data quality is generally acceptable for analysis, but consider addressing the warnings for completeness.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Verification Methodology")
    report_lines.append("")
    report_lines.append("This independent verification performed the following checks:")
    report_lines.append("")
    report_lines.append("1. **Data Integrity**: Verified row counts, experiment IDs, and column preservation")
    report_lines.append("2. **Backfilled Values Correctness**: Randomly sampled 30 backfilled cells and verified against models_config.json")
    report_lines.append("3. **Source Tracking**: Verified all source columns exist and contain valid values")
    report_lines.append("4. **Original Values Preservation**: Ensured no original values were overwritten")
    report_lines.append("")

    # Write report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()
