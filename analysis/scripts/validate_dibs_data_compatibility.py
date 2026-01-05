#!/usr/bin/env python3
"""
验证6组DiBS训练数据的适配性

检查数据是否满足DiBS的输入要求

创建日期: 2026-01-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# DiBS输入数据要求（从代码分析得出）
DIBS_REQUIREMENTS = {
    'data_type': {
        'name': '数据类型',
        'requirement': 'DataFrame或numpy数组，可转换为float',
        'critical': True
    },
    'shape': {
        'name': '数据形状',
        'requirement': '二维数组 (n_samples, n_vars)',
        'critical': True
    },
    'non_empty': {
        'name': '非空要求',
        'requirement': '样本数 > 0, 变量数 > 0',
        'critical': True
    },
    'no_nan': {
        'name': 'NaN值检查',
        'requirement': '无NaN值（DiBS内部转换为float时会出错）',
        'critical': True
    },
    'no_inf': {
        'name': '无穷值检查',
        'requirement': '无inf/-inf值',
        'critical': True
    },
    'numeric': {
        'name': '数值型要求',
        'requirement': '所有值可转换为float',
        'critical': True
    },
    'variance': {
        'name': '方差检查',
        'requirement': '所有特征方差 > 0（避免常数列）',
        'critical': True
    },
    'sample_size': {
        'name': '样本量推荐',
        'requirement': '样本数 >= 50（推荐，非强制）',
        'critical': False
    },
    'feature_count': {
        'name': '特征数推荐',
        'requirement': '特征数 >= 3, <= 100（推荐，非强制）',
        'critical': False
    },
    'standardized': {
        'name': '标准化推荐',
        'requirement': '均值≈0，标准差≈1（推荐，非强制）',
        'critical': False
    }
}

def check_data_requirements(data_path, verbose=True):
    """
    检查单个数据文件是否满足DiBS要求

    参数:
        data_path: 数据文件路径
        verbose: 是否输出详细信息

    返回:
        results: 检查结果字典
    """
    results = {
        'file': str(data_path),
        'checks': {},
        'critical_pass': True,
        'all_pass': True,
        'warnings': [],
        'errors': []
    }

    try:
        # 加载数据
        df = pd.read_csv(data_path)

        if verbose:
            print(f"\n检查文件: {data_path.name}")
            print(f"  数据维度: {df.shape[0]}行 × {df.shape[1]}列")

        # 检查1: 数据类型
        check_key = 'data_type'
        try:
            df_array = df.values.astype(float)
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 数据可转换为float数组'
            }
        except Exception as e:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ 数据无法转换为float: {e}'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"数据类型错误: {e}")

        # 检查2: 数据形状
        check_key = 'shape'
        if len(df.shape) == 2:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 二维数据 {df.shape}'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ 数据不是二维: {df.shape}'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"数据维度错误: {df.shape}")

        # 检查3: 非空要求
        check_key = 'non_empty'
        if len(df) > 0 and len(df.columns) > 0:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 非空数据 ({len(df)}行, {len(df.columns)}列)'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ 数据为空'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append("数据为空")

        # 检查4: NaN值
        check_key = 'no_nan'
        nan_count = df.isna().sum().sum()
        if nan_count == 0:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 无NaN值'
            }
        else:
            nan_cols = df.columns[df.isna().any()].tolist()
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ 发现{nan_count}个NaN值',
                'details': f'NaN列: {nan_cols[:5]}...' if len(nan_cols) > 5 else f'NaN列: {nan_cols}'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"数据包含{nan_count}个NaN值")

        # 检查5: 无穷值
        check_key = 'no_inf'
        inf_count = np.isinf(df.values.astype(float)).sum()
        if inf_count == 0:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 无inf值'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ 发现{inf_count}个inf值'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"数据包含{inf_count}个inf值")

        # 检查6: 数值型
        check_key = 'numeric'
        non_numeric = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

        if len(non_numeric) == 0:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 所有列均为数值型'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ {len(non_numeric)}列非数值型',
                'details': f'非数值列: {non_numeric[:5]}'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"存在非数值列: {non_numeric}")

        # 检查7: 方差
        check_key = 'variance'
        variances = df.var()
        zero_var_cols = variances[variances == 0].index.tolist()

        if len(zero_var_cols) == 0:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 所有特征方差 > 0'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'✗ {len(zero_var_cols)}个零方差列',
                'details': f'零方差列: {zero_var_cols}'
            }
            results['critical_pass'] = False
            results['all_pass'] = False
            results['errors'].append(f"存在零方差列: {zero_var_cols}")

        # 检查8: 样本量推荐（非强制）
        check_key = 'sample_size'
        if len(df) >= 50:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 样本量充足 ({len(df)} >= 50)'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'⚠️  样本量偏少 ({len(df)} < 50)',
                'severity': 'warning'
            }
            results['all_pass'] = False
            results['warnings'].append(f"样本量偏少: {len(df)} < 50（推荐>=50）")

        # 检查9: 特征数推荐（非强制）
        check_key = 'feature_count'
        n_features = len(df.columns)
        if 3 <= n_features <= 100:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 特征数合理 ({n_features})'
            }
        elif n_features < 3:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'⚠️  特征数过少 ({n_features} < 3)',
                'severity': 'warning'
            }
            results['all_pass'] = False
            results['warnings'].append(f"特征数过少: {n_features} < 3")
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'⚠️  特征数较多 ({n_features} > 100)',
                'severity': 'warning'
            }
            results['all_pass'] = False
            results['warnings'].append(f"特征数较多: {n_features} > 100（可能影响性能）")

        # 检查10: 标准化推荐（非强制）
        check_key = 'standardized'
        mean_abs = abs(df.mean().mean())
        std_mean = df.std().mean()

        if mean_abs < 0.1 and 0.9 < std_mean < 1.1:
            results['checks'][check_key] = {
                'pass': True,
                'message': f'✓ 数据已标准化 (均值≈{mean_abs:.3f}, 标准差≈{std_mean:.3f})'
            }
        else:
            results['checks'][check_key] = {
                'pass': False,
                'message': f'⚠️  数据未标准化 (均值={mean_abs:.3f}, 标准差={std_mean:.3f})',
                'severity': 'info'
            }
            results['all_pass'] = False
            # 标准化只是推荐，不算警告

        # 输出结果
        if verbose:
            print(f"\n  检查结果:")
            for check_name, check_result in results['checks'].items():
                req = DIBS_REQUIREMENTS.get(check_name, {})
                is_critical = req.get('critical', False)
                marker = "[关键]" if is_critical else "[推荐]"
                print(f"    {marker} {check_result['message']}")

            if results['critical_pass']:
                print(f"\n  ✅ 所有关键检查通过")
            else:
                print(f"\n  ❌ 存在关键问题，无法用于DiBS")

            if results['warnings']:
                print(f"\n  警告:")
                for warning in results['warnings']:
                    print(f"    ⚠️  {warning}")

    except Exception as e:
        results['checks']['load_error'] = {
            'pass': False,
            'message': f'✗ 文件加载失败: {e}'
        }
        results['critical_pass'] = False
        results['all_pass'] = False
        results['errors'].append(f"文件加载失败: {e}")

    return results

def main():
    """主函数"""
    print("="*80)
    print("DiBS训练数据适配性验证")
    print("="*80)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 数据文件路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'energy_research' / 'dibs_training'

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return 1

    # 找到所有CSV文件
    csv_files = sorted(data_dir.glob('group*.csv'))

    if len(csv_files) == 0:
        print(f"❌ 未找到数据文件（group*.csv）")
        return 1

    print(f"找到 {len(csv_files)} 个数据文件\n")

    # 验证每个文件
    all_results = []
    critical_pass_count = 0
    all_pass_count = 0

    for csv_file in csv_files:
        result = check_data_requirements(csv_file, verbose=True)
        all_results.append(result)

        if result['critical_pass']:
            critical_pass_count += 1
        if result['all_pass']:
            all_pass_count += 1

    # 生成总结报告
    print(f"\n{'='*80}")
    print("验证总结")
    print(f"{'='*80}\n")

    print(f"总文件数: {len(csv_files)}")
    print(f"关键检查通过: {critical_pass_count}/{len(csv_files)} ({critical_pass_count/len(csv_files)*100:.0f}%)")
    print(f"所有检查通过: {all_pass_count}/{len(csv_files)} ({all_pass_count/len(csv_files)*100:.0f}%)")

    # 创建汇总表
    print(f"\n数据适配性汇总:")
    print(f"{'文件':<30} {'关键检查':<12} {'推荐检查':<12} {'状态':<15}")
    print("-"*80)

    for result in all_results:
        filename = Path(result['file']).name
        critical_status = "✅ 通过" if result['critical_pass'] else "❌ 失败"
        all_status = "✅ 通过" if result['all_pass'] else "⚠️  有警告"
        final_status = "可用" if result['critical_pass'] else "不可用"

        print(f"{filename:<30} {critical_status:<12} {all_status:<12} {final_status:<15}")

    # 保存详细结果JSON
    output_file = data_dir / 'validation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': len(csv_files),
            'critical_pass_count': critical_pass_count,
            'all_pass_count': all_pass_count,
            'requirements': DIBS_REQUIREMENTS,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 详细结果已保存: {output_file}")

    # 生成Markdown报告
    report_file = data_dir / 'VALIDATION_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# DiBS训练数据适配性验证报告\n\n")
        f.write(f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**验证文件数**: {len(csv_files)}\n")
        f.write(f"**关键检查通过率**: {critical_pass_count}/{len(csv_files)} ({critical_pass_count/len(csv_files)*100:.0f}%)\n\n")

        # DiBS要求说明
        f.write("## DiBS输入数据要求\n\n")
        f.write("### 关键要求（必须满足）\n\n")
        for key, req in DIBS_REQUIREMENTS.items():
            if req['critical']:
                f.write(f"- **{req['name']}**: {req['requirement']}\n")

        f.write("\n### 推荐要求（建议满足）\n\n")
        for key, req in DIBS_REQUIREMENTS.items():
            if not req['critical']:
                f.write(f"- **{req['name']}**: {req['requirement']}\n")

        # 验证结果汇总
        f.write("\n## 验证结果汇总\n\n")
        f.write("| 文件 | 样本数 | 特征数 | 关键检查 | 推荐检查 | DiBS适配性 |\n")
        f.write("|------|--------|--------|---------|----------|------------|\n")

        for result in all_results:
            filename = Path(result['file']).stem

            # 提取样本数和特征数
            shape_check = result['checks'].get('shape', {})
            if 'message' in shape_check:
                import re
                match = re.search(r'\((\d+), (\d+)\)', shape_check['message'])
                if match:
                    n_samples = match.group(1)
                    n_features = match.group(2)
                else:
                    n_samples = "N/A"
                    n_features = "N/A"
            else:
                n_samples = "N/A"
                n_features = "N/A"

            critical = "✅" if result['critical_pass'] else "❌"
            all_checks = "✅" if result['all_pass'] else "⚠️"
            usable = "✅ 可用" if result['critical_pass'] else "❌ 不可用"

            f.write(f"| {filename} | {n_samples} | {n_features} | {critical} | {all_checks} | {usable} |\n")

        # 详细检查结果
        f.write("\n## 详细检查结果\n\n")
        for result in all_results:
            filename = Path(result['file']).name
            f.write(f"### {filename}\n\n")

            for check_name, check_result in result['checks'].items():
                req = DIBS_REQUIREMENTS.get(check_name, {})
                req_name = req.get('name', check_name)
                is_critical = req.get('critical', False)
                marker = "🔴 关键" if is_critical else "🟡 推荐"

                f.write(f"**{marker} {req_name}**: {check_result['message']}\n\n")

                if 'details' in check_result:
                    f.write(f"  - {check_result['details']}\n\n")

            if result['errors']:
                f.write("**错误**:\n\n")
                for error in result['errors']:
                    f.write(f"- ❌ {error}\n")
                f.write("\n")

            if result['warnings']:
                f.write("**警告**:\n\n")
                for warning in result['warnings']:
                    f.write(f"- ⚠️  {warning}\n")
                f.write("\n")

        f.write("---\n\n")
        f.write(f"**报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✓ Markdown报告已保存: {report_file}")

    # 最终结论
    print(f"\n{'='*80}")
    print("最终结论")
    print(f"{'='*80}\n")

    if critical_pass_count == len(csv_files):
        print("🎉 所有数据文件都满足DiBS关键要求，可以安全使用！")

        if all_pass_count == len(csv_files):
            print("✨ 所有数据文件都满足DiBS推荐要求，数据质量优秀！")
        else:
            print(f"⚠️  {len(csv_files) - all_pass_count} 个文件存在轻微警告，但不影响DiBS使用")

        print("\n✅ 建议: 可以立即对所有6个任务组运行DiBS分析")
        return 0
    else:
        print(f"❌ {len(csv_files) - critical_pass_count} 个文件不满足DiBS关键要求")
        print("\n建议: 修复数据质量问题后再运行DiBS")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
