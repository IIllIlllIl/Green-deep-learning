#!/usr/bin/env python3
"""
中介效应分析（问题3）

目的：
使用Sobel检验分析超参数→中间变量→能耗的中介路径，
独立于DiBS的中介路径检测。

优先测试路径（基于DiBS部分发现）：
1. epochs → gpu_util_avg → energy_gpu_total_joules (group6)
2. epochs → gpu_temp_max → energy_gpu_total_joules (group6)
3. batch_size → gpu_temp_max → energy_gpu_avg_watts (group1/group3)

创建日期: 2026-01-06
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 数据目录
DATA_DIR = Path("/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training")
OUTPUT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/mediation_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 预定义的中介路径（基于DiBS和领域知识）
MEDIATION_PATHS = [
    # Group 6 (ResNet) - 最强的epochs效应
    {
        "group": "group6_resnet",
        "X": "hyperparam_epochs",
        "M": "energy_gpu_util_avg_percent",
        "Y": "energy_gpu_total_joules",
        "label": "epochs通过GPU利用率影响总能耗"
    },
    {
        "group": "group6_resnet",
        "X": "hyperparam_epochs",
        "M": "energy_gpu_temp_max_celsius",
        "Y": "energy_gpu_total_joules",
        "label": "epochs通过GPU温度影响总能耗"
    },
    {
        "group": "group6_resnet",
        "X": "hyperparam_epochs",
        "M": "energy_gpu_util_max_percent",
        "Y": "energy_gpu_total_joules",
        "label": "epochs通过GPU峰值利用率影响总能耗"
    },

    # Group 3 (Person ReID) - epochs效应
    {
        "group": "group3_person_reid",
        "X": "hyperparam_epochs",
        "M": "energy_gpu_util_avg_percent",
        "Y": "energy_gpu_avg_watts",
        "label": "epochs通过GPU利用率影响平均功率"
    },
    {
        "group": "group3_person_reid",
        "X": "hyperparam_epochs",
        "M": "energy_gpu_temp_max_celsius",
        "Y": "energy_gpu_avg_watts",
        "label": "epochs通过GPU温度影响平均功率"
    },

    # Group 1 (Examples) - batch_size效应
    {
        "group": "group1_examples",
        "X": "hyperparam_batch_size",
        "M": "energy_gpu_temp_max_celsius",
        "Y": "energy_gpu_max_watts",
        "label": "batch_size通过GPU温度影响峰值功率"
    },
    {
        "group": "group1_examples",
        "X": "hyperparam_batch_size",
        "M": "energy_gpu_util_avg_percent",
        "Y": "energy_gpu_max_watts",
        "label": "batch_size通过GPU利用率影响峰值功率"
    },
]


def load_group_data(group_id):
    """加载任务组数据"""
    csv_file = DATA_DIR / f"{group_id}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"\n加载数据: {group_id}")
    print(f"  样本数: {len(df)}")

    return df


def sobel_test_mediation(df, X, M, Y, controls=None):
    """
    Sobel检验中介效应

    中介分析的三个回归：
    1. M ~ X + controls  (路径a: X对M的效应)
    2. Y ~ X + M + controls  (路径b: M对Y的效应，控制X)
    3. Y ~ X + controls  (路径c: X对Y的总效应)

    间接效应 = a × b
    直接效应 = c' (来自模型2的X系数)
    总效应 = c

    Sobel标准误 = sqrt(b²×SE_a² + a²×SE_b²)
    """

    # 默认控制变量（在检查变量存在性之前确定）
    if controls is None:
        controls = []
        if 'duration_seconds' in df.columns:
            controls.append('duration_seconds')

    # 检查变量是否存在
    vars_needed = [X, M, Y] + controls

    missing_vars = [v for v in vars_needed if v not in df.columns]
    if missing_vars:
        return {
            "status": "failed",
            "error": f"变量不存在: {missing_vars}"
        }

    # 移除缺失值
    df_clean = df[vars_needed].dropna()

    if len(df_clean) < 10:
        return {
            "status": "failed",
            "error": f"有效样本数不足: {len(df_clean)}"
        }

    # 构建回归公式
    def build_formula(outcome, predictors):
        return f"{outcome} ~ {' + '.join(predictors)}"

    try:
        # 模型1: M ~ X + controls (路径a)
        formula_1 = build_formula(M, [X] + controls)
        model_1 = smf.ols(formula_1, data=df_clean).fit()

        a = model_1.params[X]  # 路径a系数
        se_a = model_1.bse[X]   # 路径a标准误
        p_a = model_1.pvalues[X]

        # 模型2: Y ~ X + M + controls (路径b和c')
        formula_2 = build_formula(Y, [X, M] + controls)
        model_2 = smf.ols(formula_2, data=df_clean).fit()

        b = model_2.params[M]  # 路径b系数 (M对Y的效应)
        se_b = model_2.bse[M]   # 路径b标准误
        p_b = model_2.pvalues[M]

        c_prime = model_2.params[X]  # 直接效应 (控制M后X对Y的效应)
        p_c_prime = model_2.pvalues[X]

        # 模型3: Y ~ X + controls (路径c，总效应)
        formula_3 = build_formula(Y, [X] + controls)
        model_3 = smf.ols(formula_3, data=df_clean).fit()

        c = model_3.params[X]  # 总效应
        p_c = model_3.pvalues[X]

        # 计算间接效应和Sobel检验
        indirect_effect = a * b
        sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
        sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))  # 双侧检验

        # 中介比例
        mediation_pct = (indirect_effect / c * 100) if c != 0 else 0

        # 判断中介类型
        if sobel_p < 0.05:  # 间接效应显著
            if p_c_prime < 0.05:  # 直接效应也显著
                mediation_type = "部分中介"
            else:  # 直接效应不显著
                mediation_type = "完全中介"
        else:  # 间接效应不显著
            mediation_type = "无中介"

        # 结果
        result = {
            "status": "success",
            "n_samples": len(df_clean),

            # 路径系数
            "path_a": float(a),
            "path_a_se": float(se_a),
            "path_a_p": float(p_a),

            "path_b": float(b),
            "path_b_se": float(se_b),
            "path_b_p": float(p_b),

            "path_c": float(c),
            "path_c_p": float(p_c),

            "path_c_prime": float(c_prime),
            "path_c_prime_p": float(p_c_prime),

            # 中介效应
            "indirect_effect": float(indirect_effect),
            "direct_effect": float(c_prime),
            "total_effect": float(c),

            # Sobel检验
            "sobel_se": float(sobel_se),
            "sobel_z": float(sobel_z),
            "sobel_p": float(sobel_p),

            # 中介类型
            "mediation_type": mediation_type,
            "mediation_pct": float(mediation_pct),

            # 显著性
            "is_significant": sobel_p < 0.05,

            # 模型拟合度
            "model_1_r2": float(model_1.rsquared),
            "model_2_r2": float(model_2.rsquared),
            "model_3_r2": float(model_3.rsquared),
        }

        # 打印结果
        print(f"\n  中介分析结果:")
        print(f"    路径a ({X}→{M}): {a:.4f} (p={p_a:.4f})")
        print(f"    路径b ({M}→{Y}): {b:.4f} (p={p_b:.4f})")
        print(f"    总效应 ({X}→{Y}): {c:.4f} (p={p_c:.4f})")
        print(f"    直接效应: {c_prime:.4f} (p={p_c_prime:.4f})")
        print(f"    间接效应: {indirect_effect:.4f}")
        print(f"    Sobel z: {sobel_z:.4f}, p={sobel_p:.4f}")
        print(f"    中介类型: {mediation_type} ({mediation_pct:.1f}%)")
        print(f"    显著性: {'✅ 显著' if sobel_p < 0.05 else '❌ 不显著'}")

        return result

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def analyze_all_mediation_paths():
    """分析所有预定义的中介路径"""

    print("="*80)
    print("中介效应分析（问题3）")
    print("="*80)

    all_results = []

    for i, path in enumerate(MEDIATION_PATHS, 1):
        print(f"\n{'='*80}")
        print(f"分析 {i}/{len(MEDIATION_PATHS)}: {path['group']}")
        print(f"  路径: {path['X']} → {path['M']} → {path['Y']}")
        print(f"  标签: {path['label']}")
        print(f"{'='*80}")

        # 加载数据
        try:
            df = load_group_data(path['group'])
        except FileNotFoundError as e:
            print(f"  ❌ 错误: {e}")
            all_results.append({
                **path,
                "status": "failed",
                "error": str(e)
            })
            continue

        # 运行中介分析
        result = sobel_test_mediation(
            df,
            X=path['X'],
            M=path['M'],
            Y=path['Y']
        )

        # 保存结果
        all_results.append({
            **path,
            **result
        })

    return all_results


def generate_mediation_report(results):
    """生成中介分析报告"""

    report_file = OUTPUT_DIR / "MEDIATION_ANALYSIS_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# 中介效应分析报告（问题3）\n\n")
        f.write(f"**分析日期**: 2026-01-06\n")
        f.write(f"**测试路径数**: {len(results)}条\n\n")

        f.write("---\n\n")

        # 汇总统计
        f.write("## 📊 分析汇总\n\n")

        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']

        f.write(f"- **成功分析**: {len(successful)}/{len(results)}\n")
        f.write(f"- **失败分析**: {len(failed)}/{len(results)}\n\n")

        if successful:
            significant = [r for r in successful if r['is_significant']]
            full_mediation = [r for r in successful if r['mediation_type'] == '完全中介']
            partial_mediation = [r for r in successful if r['mediation_type'] == '部分中介']
            no_mediation = [r for r in successful if r['mediation_type'] == '无中介']

            f.write(f"### 中介效应详情\n\n")
            f.write(f"- **显著中介效应**: {len(significant)}/{len(successful)}\n")
            f.write(f"- **完全中介**: {len(full_mediation)}/{len(successful)}\n")
            f.write(f"- **部分中介**: {len(partial_mediation)}/{len(successful)}\n")
            f.write(f"- **无中介**: {len(no_mediation)}/{len(successful)}\n\n")

        # 详细结果表格
        f.write("## 📋 详细结果\n\n")
        f.write("| 任务组 | 路径 | 间接效应 | Sobel p | 中介类型 | 中介比例 |\n")
        f.write("|--------|------|----------|---------|----------|----------|\n")

        for r in results:
            if r.get('status') == 'success':
                path_str = f"{r['X'][:15]}→{r['M'][:10]}→{r['Y'][:15]}"
                f.write(f"| {r['group'][:18]} | {path_str[:35]} | "
                       f"{r['indirect_effect']:.4f} | {r['sobel_p']:.4f} | "
                       f"{r['mediation_type']} | {r['mediation_pct']:.1f}% |\n")
            else:
                f.write(f"| {r['group'][:18]} | {r['label'][:35]} | - | - | ❌ 失败 | - |\n")

        f.write("\n")

        # 显著中介路径（重点）
        if successful:
            significant = [r for r in successful if r['is_significant']]

            if significant:
                f.write("## ✅ 显著中介路径\n\n")

                for r in significant:
                    f.write(f"### {r['group']}: {r['label']}\n\n")
                    f.write(f"**路径**: {r['X']} → {r['M']} → {r['Y']}\n\n")

                    f.write(f"**路径系数**:\n")
                    f.write(f"- 路径a ({r['X']}→{r['M']}): {r['path_a']:.4f} (p={r['path_a_p']:.4f})\n")
                    f.write(f"- 路径b ({r['M']}→{r['Y']}): {r['path_b']:.4f} (p={r['path_b_p']:.4f})\n")
                    f.write(f"- 总效应c: {r['total_effect']:.4f} (p={r['path_c_p']:.4f})\n")
                    f.write(f"- 直接效应c': {r['direct_effect']:.4f} (p={r['path_c_prime_p']:.4f})\n\n")

                    f.write(f"**中介效应**:\n")
                    f.write(f"- 间接效应: {r['indirect_effect']:.4f}\n")
                    f.write(f"- Sobel检验: z={r['sobel_z']:.4f}, p={r['sobel_p']:.4f}\n")
                    f.write(f"- 中介类型: **{r['mediation_type']}**\n")
                    f.write(f"- 中介比例: {r['mediation_pct']:.1f}%\n\n")

                    f.write(f"**解释**:\n")
                    if r['mediation_type'] == '完全中介':
                        f.write(f"- {r['X']}对{r['Y']}的影响**完全**通过{r['M']}实现\n")
                    elif r['mediation_type'] == '部分中介':
                        f.write(f"- {r['X']}对{r['Y']}的影响**部分**通过{r['M']}实现（{r['mediation_pct']:.1f}%）\n")
                        f.write(f"- 还存在{100-r['mediation_pct']:.1f}%的直接效应\n")
                    f.write("\n")

        # 无显著中介的路径
        if successful:
            non_significant = [r for r in successful if not r['is_significant']]

            if non_significant:
                f.write("## ⚠️ 无显著中介的路径\n\n")

                for r in non_significant:
                    f.write(f"### {r['group']}: {r['label']}\n\n")
                    f.write(f"- 路径: {r['X']} → {r['M']} → {r['Y']}\n")
                    f.write(f"- 间接效应: {r['indirect_effect']:.4f} (p={r['sobel_p']:.4f})\n")

                    # 诊断原因
                    if r['path_a_p'] >= 0.05:
                        f.write(f"- ⚠️ 路径a不显著: {r['X']}对{r['M']}无显著影响\n")
                    if r['path_b_p'] >= 0.05:
                        f.write(f"- ⚠️ 路径b不显著: {r['M']}对{r['Y']}无显著影响（控制{r['X']}后）\n")

                    f.write("\n")

        # 结论
        f.write("## 💡 结论\n\n")

        if successful:
            significant = [r for r in successful if r['is_significant']]

            if len(significant) > 0:
                sig_rate = len(significant) / len(successful) * 100
                f.write(f"### 中介效应检出率: {sig_rate:.1f}%\n\n")

                if sig_rate >= 50:
                    f.write("✅ **中间变量起到重要中介作用**\n\n")
                elif sig_rate >= 20:
                    f.write("⚠️ **部分中间变量起到中介作用**\n\n")
                else:
                    f.write("❌ **中介效应整体较弱**\n\n")

            # 关键发现
            f.write("### 关键发现\n\n")

            # GPU利用率作为中介
            gpu_util_paths = [r for r in significant if 'gpu_util' in r['M']]
            if gpu_util_paths:
                f.write("1. **GPU利用率是重要中介变量** ✅\n")
                for r in gpu_util_paths:
                    f.write(f"   - {r['group']}: {r['X']}通过{r['M']}影响{r['Y']} "
                           f"（{r['mediation_type']}，{r['mediation_pct']:.1f}%）\n")
                f.write("\n")

            # GPU温度作为中介
            gpu_temp_paths = [r for r in significant if 'gpu_temp' in r['M']]
            if gpu_temp_paths:
                f.write("2. **GPU温度是重要中介变量** ✅\n")
                for r in gpu_temp_paths:
                    f.write(f"   - {r['group']}: {r['X']}通过{r['M']}影响{r['Y']} "
                           f"（{r['mediation_type']}，{r['mediation_pct']:.1f}%）\n")
                f.write("\n")

            # GPU显存作为中介
            gpu_mem_paths = [r for r in significant if 'gpu_memory' in r['M']]
            if gpu_mem_paths:
                f.write("3. **GPU显存是重要中介变量** ✅\n")
                for r in gpu_mem_paths:
                    f.write(f"   - {r['group']}: {r['X']}通过{r['M']}影响{r['Y']} "
                           f"（{r['mediation_type']}，{r['mediation_pct']:.1f}%）\n")
                f.write("\n")

        f.write("### 对问题3的回答\n\n")
        f.write("**问题3: 训练过程中的中间变量（如GPU利用率、温度等）在超参数对能耗的影响中起到什么作用？**\n\n")

        if successful:
            significant = [r for r in successful if r['is_significant']]

            if len(significant) > 0:
                f.write("**回答**: 中间变量在超参数对能耗的影响中起到**显著中介作用**。\n\n")

                full_med = [r for r in significant if r['mediation_type'] == '完全中介']
                partial_med = [r for r in significant if r['mediation_type'] == '部分中介']

                if full_med:
                    f.write(f"- **完全中介路径** ({len(full_med)}条): 超参数对能耗的影响完全通过中间变量实现\n")
                if partial_med:
                    f.write(f"- **部分中介路径** ({len(partial_med)}条): 超参数对能耗的影响部分通过中间变量实现\n")

                f.write("\n")
                f.write("这说明：\n")
                f.write("1. 超参数不是直接影响能耗，而是通过改变GPU状态（利用率、温度、显存）来影响能耗\n")
                f.write("2. 优化能耗的关键是控制这些中间变量\n")
            else:
                f.write("**回答**: 中间变量的中介作用**不显著**。\n\n")
                f.write("可能原因：\n")
                f.write("1. 样本量不足（部分组<100）\n")
                f.write("2. 超参数直接影响能耗，不需要通过中间变量\n")
                f.write("3. 数据预处理（标准化、填充）破坏了中介关系\n")

        f.write("\n---\n\n")
        f.write(f"**报告生成时间**: 2026-01-06\n")
        f.write(f"**分析方法**: Sobel检验中介分析\n")
        f.write(f"**数据来源**: DiBS训练数据（6个任务组）\n")

    print(f"\n✅ 中介分析报告已保存: {report_file}")

    return report_file


def save_results_json(results):
    """保存JSON格式的结果"""
    json_file = OUTPUT_DIR / "mediation_analysis_results.json"

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
    print("开始中介效应分析（问题3）")
    print("="*80)

    # 分析所有路径
    results = analyze_all_mediation_paths()

    # 保存结果
    save_results_json(results)

    # 生成报告
    report_file = generate_mediation_report(results)

    print("\n"+"="*80)
    print("✅ 中介分析完成！")
    print("="*80)
    print(f"  测试路径数: {len(results)}")
    print(f"  结果目录: {OUTPUT_DIR}")
    print(f"  报告文件: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
