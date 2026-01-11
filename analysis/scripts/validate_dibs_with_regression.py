#!/usr/bin/env python3
"""
回归分析验证DiBS发现

目的：
1. 验证DiBS检测到的超参数→能耗因果边
2. 量化因果效应大小（回归系数）
3. 对比DiBS和回归分析的一致性

基于DiBS发现的优先验证边：
- Group1: batch_size → energy_gpu_max_watts (DiBS强度=0.2)
- Group3: epochs → energy_gpu_avg_watts (DiBS强度=0.3)
- Group3: epochs → energy_gpu_min_watts (DiBS强度=0.4)
- Group6: epochs → energy_gpu_total_joules (DiBS强度=0.3)
- Group6: epochs → energy_gpu_avg_watts (DiBS强度=0.15)

创建日期: 2026-01-05
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 数据和结果目录
DATA_DIR = Path("/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training")
DIBS_RESULT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/questions_2_3_dibs/20260105_212940")
OUTPUT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/regression_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DiBS发现的优先验证边
DIBS_EDGES_TO_VALIDATE = [
    {
        "group": "group1_examples",
        "hyperparam": "hyperparam_batch_size",
        "energy": "energy_gpu_max_watts",
        "dibs_strength": 0.2,
        "expected_direction": "positive"
    },
    {
        "group": "group3_person_reid",
        "hyperparam": "hyperparam_epochs",
        "energy": "energy_gpu_avg_watts",
        "dibs_strength": 0.3,
        "expected_direction": "positive"
    },
    {
        "group": "group3_person_reid",
        "hyperparam": "hyperparam_epochs",
        "energy": "energy_gpu_min_watts",
        "dibs_strength": 0.4,
        "expected_direction": "positive"
    },
    {
        "group": "group6_resnet",
        "hyperparam": "hyperparam_epochs",
        "energy": "energy_gpu_total_joules",
        "dibs_strength": 0.3,
        "expected_direction": "positive"
    },
    {
        "group": "group6_resnet",
        "hyperparam": "hyperparam_epochs",
        "energy": "energy_gpu_avg_watts",
        "dibs_strength": 0.15,
        "expected_direction": "positive"
    }
]


def load_group_data(group_id):
    """加载任务组数据"""
    csv_file = DATA_DIR / f"{group_id}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"\n加载数据: {group_id}")
    print(f"  样本数: {len(df)}")
    print(f"  特征数: {len(df.columns)}")

    return df


def validate_edge_with_regression(df, hyperparam, energy, controls=None):
    """
    使用回归分析验证DiBS发现的因果边

    参数:
        df: 数据框
        hyperparam: 超参数列名
        energy: 能耗指标列名
        controls: 控制变量列表

    返回:
        result: 验证结果字典
    """
    # 检查变量是否存在
    if hyperparam not in df.columns:
        return {
            "status": "failed",
            "error": f"超参数列不存在: {hyperparam}"
        }

    if energy not in df.columns:
        return {
            "status": "failed",
            "error": f"能耗列不存在: {energy}"
        }

    # 默认控制变量
    if controls is None:
        controls = []
        # 添加duration_seconds（如果存在）
        if 'duration_seconds' in df.columns:
            controls.append('duration_seconds')

    # 构建回归公式
    formula_parts = [energy, "~", hyperparam]
    if controls:
        formula_parts.extend(["+"] + [" + ".join(controls)])

    formula = " ".join(formula_parts)

    print(f"\n  回归公式: {formula}")

    # 选择数据（移除缺失值）
    vars_needed = [hyperparam, energy] + controls
    df_clean = df[vars_needed].dropna()

    print(f"  有效样本数: {len(df_clean)} / {len(df)}")

    if len(df_clean) < 10:
        return {
            "status": "failed",
            "error": f"有效样本数不足: {len(df_clean)}"
        }

    # 运行回归
    try:
        model = smf.ols(formula, data=df_clean).fit()

        # 提取超参数的系数
        coef = model.params[hyperparam]
        stderr = model.bse[hyperparam]
        tvalue = model.tvalues[hyperparam]
        pvalue = model.pvalues[hyperparam]
        ci_lower, ci_upper = model.conf_int().loc[hyperparam]

        # 判断显著性
        is_significant = pvalue < 0.05

        # 判断方向
        direction = "positive" if coef > 0 else "negative"

        # 结果
        result = {
            "status": "success",
            "n_samples": len(df_clean),
            "formula": formula,
            "coefficient": float(coef),
            "std_error": float(stderr),
            "t_value": float(tvalue),
            "p_value": float(pvalue),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "is_significant": is_significant,
            "direction": direction,
            "model_summary": str(model.summary())
        }

        # 打印结果
        print(f"\n  回归结果:")
        print(f"    系数: {coef:.6f}")
        print(f"    标准误: {stderr:.6f}")
        print(f"    t值: {tvalue:.4f}")
        print(f"    p值: {pvalue:.6f} {'***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else 'ns'}")
        print(f"    95%置信区间: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"    R²: {model.rsquared:.4f}")
        print(f"    显著性: {'✅ 显著' if is_significant else '❌ 不显著'}")
        print(f"    方向: {direction}")

        return result

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def validate_all_dibs_edges():
    """验证所有DiBS发现的边"""

    print("="*80)
    print("回归分析验证DiBS发现")
    print("="*80)

    all_results = []

    for i, edge in enumerate(DIBS_EDGES_TO_VALIDATE, 1):
        print(f"\n{'='*80}")
        print(f"验证 {i}/{len(DIBS_EDGES_TO_VALIDATE)}: {edge['group']}")
        print(f"  边: {edge['hyperparam']} → {edge['energy']}")
        print(f"  DiBS强度: {edge['dibs_strength']}")
        print(f"  预期方向: {edge['expected_direction']}")
        print(f"{'='*80}")

        # 加载数据
        try:
            df = load_group_data(edge['group'])
        except FileNotFoundError as e:
            print(f"  ❌ 错误: {e}")
            all_results.append({
                **edge,
                "regression_status": "failed",
                "error": str(e)
            })
            continue

        # 运行回归
        result = validate_edge_with_regression(
            df,
            hyperparam=edge['hyperparam'],
            energy=edge['energy']
        )

        # 对比DiBS和回归结果
        if result['status'] == 'success':
            direction_match = result['direction'] == edge['expected_direction']
            both_significant = result['is_significant']  # DiBS已经筛选过（强度>0.1）

            validation_status = (
                "✅ 一致且显著" if direction_match and both_significant else
                "⚠️ 方向一致但回归不显著" if direction_match and not both_significant else
                "❌ 方向不一致" if not direction_match else
                "⚠️ 其他"
            )

            print(f"\n  对比DiBS和回归:")
            print(f"    方向匹配: {'✅' if direction_match else '❌'}")
            print(f"    回归显著: {'✅' if both_significant else '❌'}")
            print(f"    验证状态: {validation_status}")

            all_results.append({
                **edge,
                **result,
                "direction_match": direction_match,
                "validation_status": validation_status
            })
        else:
            print(f"\n  ❌ 回归失败: {result['error']}")
            all_results.append({
                **edge,
                **result
            })

    return all_results


def generate_validation_report(results):
    """生成验证报告"""

    report_file = OUTPUT_DIR / "REGRESSION_VALIDATION_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# 回归分析验证DiBS发现报告\n\n")
        f.write(f"**分析日期**: 2026-01-05\n")
        f.write(f"**验证边数**: {len(results)}条\n\n")

        f.write("---\n\n")

        # 汇总统计
        f.write("## 📊 验证汇总\n\n")

        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']

        f.write(f"- **成功验证**: {len(successful)}/{len(results)}\n")
        f.write(f"- **失败验证**: {len(failed)}/{len(results)}\n\n")

        if successful:
            significant = [r for r in successful if r['is_significant']]
            direction_match = [r for r in successful if r['direction_match']]
            fully_validated = [r for r in successful if r['is_significant'] and r['direction_match']]

            f.write(f"### 成功验证的边详情\n\n")
            f.write(f"- **回归显著（p<0.05）**: {len(significant)}/{len(successful)}\n")
            f.write(f"- **方向与DiBS一致**: {len(direction_match)}/{len(successful)}\n")
            f.write(f"- **完全验证（显著+方向一致）**: {len(fully_validated)}/{len(successful)} ✅\n\n")

        # 详细结果表格
        f.write("## 📋 详细验证结果\n\n")
        f.write("| 任务组 | 边 | DiBS强度 | 回归系数 | p值 | R² | 验证状态 |\n")
        f.write("|--------|-----|---------|---------|-----|-----|----------|\n")

        for r in results:
            if r.get('status') == 'success':
                edge_str = f"{r['hyperparam']} → {r['energy']}"
                f.write(f"| {r['group'][:20]} | {edge_str[:30]} | {r['dibs_strength']:.3f} | "
                       f"{r['coefficient']:.6f} | {r['p_value']:.4f} | {r['r_squared']:.4f} | "
                       f"{r['validation_status']} |\n")
            else:
                edge_str = f"{r['hyperparam']} → {r['energy']}"
                f.write(f"| {r['group'][:20]} | {edge_str[:30]} | {r['dibs_strength']:.3f} | "
                       f"- | - | - | ❌ 失败: {r.get('error', 'unknown')[:20]} |\n")

        f.write("\n")

        # 完全验证的边（重点）
        if successful:
            fully_validated = [r for r in successful if r['is_significant'] and r['direction_match']]

            if fully_validated:
                f.write("## ✅ 完全验证的因果边（显著+方向一致）\n\n")

                for r in fully_validated:
                    f.write(f"### {r['group']}: {r['hyperparam']} → {r['energy']}\n\n")
                    f.write(f"**DiBS发现**:\n")
                    f.write(f"- 边强度: {r['dibs_strength']:.3f}\n")
                    f.write(f"- 预期方向: {r['expected_direction']}\n\n")

                    f.write(f"**回归验证**:\n")
                    f.write(f"- 回归系数: {r['coefficient']:.6f}\n")
                    f.write(f"- 标准误: {r['std_error']:.6f}\n")
                    f.write(f"- t值: {r['t_value']:.4f}\n")
                    f.write(f"- p值: {r['p_value']:.6f} {'***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*'}\n")
                    f.write(f"- 95%置信区间: [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]\n")
                    f.write(f"- R²: {r['r_squared']:.4f}\n")
                    f.write(f"- 样本数: {r['n_samples']}\n\n")

                    f.write(f"**解释**:\n")
                    if r['hyperparam'] == 'hyperparam_epochs':
                        f.write(f"- 训练轮数（epochs）每增加1，{r['energy']}增加约{abs(r['coefficient']):.2f}单位\n")
                    elif r['hyperparam'] == 'hyperparam_batch_size':
                        f.write(f"- 批量大小（batch_size）每增加1，{r['energy']}增加约{abs(r['coefficient']):.2f}单位\n")
                    f.write(f"- 该因果关系在统计上显著（p={r['p_value']:.4f} < 0.05）\n")
                    f.write(f"- 模型解释了{r['r_squared']*100:.1f}%的能耗变化\n\n")

        # 不一致的边（需要进一步分析）
        if successful:
            inconsistent = [r for r in successful if not r['is_significant'] or not r['direction_match']]

            if inconsistent:
                f.write("## ⚠️ 需要进一步分析的边\n\n")

                for r in inconsistent:
                    f.write(f"### {r['group']}: {r['hyperparam']} → {r['energy']}\n\n")
                    f.write(f"- DiBS强度: {r['dibs_strength']:.3f}\n")
                    f.write(f"- 回归系数: {r['coefficient']:.6f} (p={r['p_value']:.4f})\n")

                    if not r['is_significant']:
                        f.write(f"- ⚠️ **回归不显著**: DiBS可能是假阳性，或者样本量不足\n")
                    if not r['direction_match']:
                        f.write(f"- ⚠️ **方向不一致**: DiBS预期{r['expected_direction']}，回归发现{r['direction']}\n")

                    f.write("\n")

        # 结论
        f.write("## 💡 结论\n\n")

        if successful:
            fully_validated = [r for r in successful if r['is_significant'] and r['direction_match']]
            validation_rate = len(fully_validated) / len(results) * 100

            f.write(f"### DiBS验证率: {validation_rate:.1f}%\n\n")

            if validation_rate >= 80:
                f.write("✅ **DiBS发现高度可信**: 80%以上的边通过回归验证\n\n")
            elif validation_rate >= 50:
                f.write("⚠️ **DiBS发现部分可信**: 50-80%的边通过回归验证，建议结合其他方法\n\n")
            else:
                f.write("❌ **DiBS发现可信度低**: 不足50%的边通过回归验证，建议使用其他因果方法\n\n")

            f.write("### 关键发现\n\n")

            # 提取epochs的效应
            epochs_edges = [r for r in fully_validated if 'epochs' in r['hyperparam']]
            if epochs_edges:
                f.write("1. **epochs是能耗的主要驱动因素** ✅\n")
                for r in epochs_edges:
                    f.write(f"   - {r['group']}: 每增加1个epoch，{r['energy']}增加{abs(r['coefficient']):.2f}单位 (p={r['p_value']:.4f})\n")
                f.write("\n")

            # 提取batch_size的效应
            batch_edges = [r for r in fully_validated if 'batch_size' in r['hyperparam']]
            if batch_edges:
                f.write("2. **batch_size影响GPU峰值功率** ✅\n")
                for r in batch_edges:
                    f.write(f"   - {r['group']}: 每增加1个batch_size，{r['energy']}增加{abs(r['coefficient']):.2f}单位 (p={r['p_value']:.4f})\n")
                f.write("\n")

        f.write("### 后续建议\n\n")
        f.write("1. 对完全验证的边，可以在论文中作为**强证据**引用\n")
        f.write("2. 对不一致的边，建议使用因果森林或工具变量法进一步验证\n")
        f.write("3. 结合DiBS（因果发现）+ 回归（因果量化）的双重证据，结论更可信\n\n")

        f.write("---\n\n")
        f.write(f"**报告生成时间**: 2026-01-05\n")
        f.write(f"**数据来源**: DiBS分析结果 + 原始DiBS训练数据\n")

    print(f"\n✅ 验证报告已保存: {report_file}")

    return report_file


def save_results_json(results):
    """保存JSON格式的结果"""
    json_file = OUTPUT_DIR / "regression_validation_results.json"

    # 转换numpy类型为Python原生类型
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    results_converted = convert_to_python_types(results)

    with open(json_file, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"✅ JSON结果已保存: {json_file}")


def main():
    """主函数"""
    print("="*80)
    print("开始回归分析验证DiBS发现")
    print("="*80)

    # 验证所有边
    results = validate_all_dibs_edges()

    # 保存结果
    save_results_json(results)

    # 生成报告
    report_file = generate_validation_report(results)

    print("\n"+"="*80)
    print("✅ 验证完成！")
    print("="*80)
    print(f"  验证边数: {len(results)}")
    print(f"  结果目录: {OUTPUT_DIR}")
    print(f"  报告文件: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
