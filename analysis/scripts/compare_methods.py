#!/usr/bin/env python3
"""
能耗数据集方法对比测试脚本

测试多种分析方法，找出最适合能耗数据的方法：
1. 相关性分析（Pearson + Spearman）
2. 回归分析（线性回归 + 随机森林）
3. 偏相关分析（控制混淆变量）
4. PC算法（基于条件独立性的因果发现）
5. 互信息分析

日期: 2025-12-26
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def load_and_prepare_data(csv_path):
    """加载并准备数据"""
    print_section("数据加载与预处理")

    df = pd.read_csv(csv_path)
    print(f"✅ 数据加载: {len(df)}行 × {len(df.columns)}列")

    # 提取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()

    # 移除全缺失列
    df_numeric = df_numeric.dropna(axis=1, how='all')

    # 移除缺失率>30%的列
    missing_threshold = 0.30
    missing_per_col = df_numeric.isna().sum() / len(df_numeric)
    cols_to_keep = missing_per_col[missing_per_col <= missing_threshold].index.tolist()
    df_numeric = df_numeric[cols_to_keep]

    print(f"✅ 数值型变量: {len(df_numeric.columns)}个")
    print(f"✅ 有效样本: {len(df_numeric)}行")
    print(f"✅ 缺失率: {df_numeric.isna().sum().sum() / (len(df_numeric) * len(df_numeric.columns)) * 100:.2f}%")

    return df_numeric

def method1_correlation_analysis(df):
    """方法1: 相关性分析（Pearson + Spearman）"""
    print_section("方法1: 相关性分析")

    start_time = time.time()

    # Pearson相关系数
    print("\n[1.1] Pearson相关系数...")
    corr_pearson = df.corr(method='pearson')

    # Spearman秩相关系数
    print("[1.2] Spearman秩相关系数...")
    corr_spearman = df.corr(method='spearman')

    # 提取上三角（不包括对角线）
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
    upper_pearson = corr_pearson.where(mask)
    upper_spearman = corr_spearman.where(mask)

    # 统计
    pearson_values = upper_pearson.values[~np.isnan(upper_pearson.values)]
    spearman_values = upper_spearman.values[~np.isnan(upper_spearman.values)]

    print(f"\n✅ Pearson相关系数统计:")
    print(f"   平均绝对值: {np.abs(pearson_values).mean():.3f}")
    print(f"   最大绝对值: {np.abs(pearson_values).max():.3f}")
    print(f"   |r|>0.5: {(np.abs(pearson_values) > 0.5).sum()}对")
    print(f"   |r|>0.7: {(np.abs(pearson_values) > 0.7).sum()}对")
    print(f"   |r|>0.9: {(np.abs(pearson_values) > 0.9).sum()}对")

    print(f"\n✅ Spearman相关系数统计:")
    print(f"   平均绝对值: {np.abs(spearman_values).mean():.3f}")
    print(f"   最大绝对值: {np.abs(spearman_values).max():.3f}")
    print(f"   |r|>0.5: {(np.abs(spearman_values) > 0.5).sum()}对")
    print(f"   |r|>0.7: {(np.abs(spearman_values) > 0.7).sum()}对")
    print(f"   |r|>0.9: {(np.abs(spearman_values) > 0.9).sum()}对")

    # 找出最强相关对
    pearson_pairs = []
    for i in range(len(corr_pearson.columns)):
        for j in range(i+1, len(corr_pearson.columns)):
            pearson_pairs.append((
                corr_pearson.columns[i],
                corr_pearson.columns[j],
                corr_pearson.iloc[i, j]
            ))
    pearson_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n✅ Top 10最强Pearson相关对:")
    for i, (var1, var2, corr) in enumerate(pearson_pairs[:10], 1):
        print(f"   {i:2d}. {var1:30s} <-> {var2:30s}: {corr:7.3f}")

    elapsed = time.time() - start_time

    result = {
        'method': 'Correlation Analysis',
        'time_seconds': elapsed,
        'success': True,
        'pearson_mean': float(np.abs(pearson_values).mean()),
        'pearson_max': float(np.abs(pearson_values).max()),
        'pearson_strong_pairs': int((np.abs(pearson_values) > 0.7).sum()),
        'spearman_mean': float(np.abs(spearman_values).mean()),
        'spearman_max': float(np.abs(spearman_values).max()),
        'top_pairs': pearson_pairs[:10],
        'corr_matrix': corr_pearson
    }

    print(f"\n⏱️  耗时: {elapsed:.2f}秒")

    return result

def method2_regression_analysis(df):
    """方法2: 回归分析（线性回归 + 随机森林）"""
    print_section("方法2: 回归分析")

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()

    # 选择目标变量（能耗和性能）
    target_cols = [col for col in df.columns if 'energy' in col or 'perf' in col or 'accuracy' in col]
    feature_cols = [col for col in df.columns if col not in target_cols]

    print(f"\n[2.1] 特征变量: {len(feature_cols)}个")
    print(f"[2.2] 目标变量: {len(target_cols)}个")

    results = {}

    for target in target_cols[:3]:  # 只分析前3个目标（节省时间）
        print(f"\n[目标: {target}]")

        # 准备数据
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target]

        # 移除y的缺失值
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 10:
            print(f"   ⚠️  样本不足（{len(X)}个），跳过")
            continue

        print(f"   有效样本: {len(X)}个")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 线性回归
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        lr_score = lr.score(X_scaled, y)

        # 特征系数
        lr_coef = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': lr.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        print(f"   [线性回归] R²: {lr_score:.3f}")
        print(f"   Top 5特征:")
        for i, row in lr_coef.head(5).iterrows():
            print(f"      {row['feature']:30s}: {row['coefficient']:8.3f}")

        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_scaled, y)
        rf_score = rf.score(X_scaled, y)

        # 特征重要性
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"   [随机森林] R²: {rf_score:.3f}")
        print(f"   Top 5特征:")
        for i, row in rf_importance.head(5).iterrows():
            print(f"      {row['feature']:30s}: {row['importance']:8.3f}")

        results[target] = {
            'lr_r2': float(lr_score),
            'rf_r2': float(rf_score),
            'lr_top_features': lr_coef.head(5).to_dict('records'),
            'rf_top_features': rf_importance.head(5).to_dict('records')
        }

    elapsed = time.time() - start_time

    result = {
        'method': 'Regression Analysis',
        'time_seconds': elapsed,
        'success': True,
        'targets': results
    }

    print(f"\n⏱️  耗时: {elapsed:.2f}秒")

    return result

def method3_partial_correlation(df):
    """方法3: 偏相关分析"""
    print_section("方法3: 偏相关分析")

    from scipy.stats import pearsonr

    start_time = time.time()

    print("\n[3.1] 计算偏相关系数...")
    print("策略: 控制其他所有变量，计算两变量间的偏相关")

    # 只计算一些关键变量对（节省时间）
    key_vars = [col for col in df.columns if 'energy' in col or 'perf' in col or 'duration' in col][:5]

    print(f"关键变量: {key_vars}")

    partial_corrs = []

    for i in range(len(key_vars)):
        for j in range(i+1, len(key_vars)):
            var1 = key_vars[i]
            var2 = key_vars[j]

            # 准备数据
            data = df[[var1, var2]].dropna()

            if len(data) < 10:
                continue

            # 简单偏相关：控制其他变量
            # 这里使用简化方法：先对两变量做线性回归去除其他变量影响，再计算残差相关
            from sklearn.linear_model import LinearRegression

            other_vars = [v for v in key_vars if v not in [var1, var2]]
            if len(other_vars) > 0:
                X_other = df.loc[data.index, other_vars].dropna(axis=1)
                if len(X_other.columns) > 0:
                    # 对var1回归
                    lr1 = LinearRegression()
                    lr1.fit(X_other, data[var1])
                    resid1 = data[var1] - lr1.predict(X_other)

                    # 对var2回归
                    lr2 = LinearRegression()
                    lr2.fit(X_other, data[var2])
                    resid2 = data[var2] - lr2.predict(X_other)

                    # 计算残差相关
                    partial_r, _ = pearsonr(resid1, resid2)

                    partial_corrs.append((var1, var2, partial_r))

    partial_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n✅ 偏相关系数（控制其他变量后）:")
    for i, (var1, var2, pcorr) in enumerate(partial_corrs[:10], 1):
        print(f"   {i:2d}. {var1:30s} <-> {var2:30s}: {pcorr:7.3f}")

    elapsed = time.time() - start_time

    result = {
        'method': 'Partial Correlation',
        'time_seconds': elapsed,
        'success': True,
        'partial_corrs': partial_corrs[:10]
    }

    print(f"\n⏱️  耗时: {elapsed:.2f}秒")

    return result

def method4_pc_algorithm(df):
    """方法4: PC算法（基于条件独立性的因果发现）"""
    print_section("方法4: PC算法（条件独立性）")

    start_time = time.time()

    try:
        # 尝试导入PC算法（需要causal-learn库）
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        print("\n[4.1] 准备数据...")

        # 使用前10个变量（PC算法对大数据集很慢）
        selected_cols = df.columns[:10].tolist()
        data = df[selected_cols].dropna().values

        print(f"✅ 使用变量: {len(selected_cols)}个")
        print(f"✅ 有效样本: {len(data)}个")

        print("\n[4.2] 运行PC算法...")
        print("注意: 这可能需要几分钟...")

        # 运行PC算法
        cg = pc(data, alpha=0.05, indep_test=fisherz)

        # 提取因果边
        edges = []
        for i in range(cg.G.shape[0]):
            for j in range(i+1, cg.G.shape[1]):
                if cg.G[i, j] != 0:
                    edges.append((selected_cols[i], selected_cols[j], cg.G[i, j]))

        print(f"\n✅ PC算法完成")
        print(f"   检测到边数: {len(edges)}")

        if edges:
            print(f"\n   检测到的边:")
            for i, (var1, var2, edge_type) in enumerate(edges[:10], 1):
                edge_str = "→" if edge_type == -1 else ("←" if edge_type == 1 else "—")
                print(f"      {i:2d}. {var1} {edge_str} {var2}")
        else:
            print(f"   ⚠️  未检测到任何边")

        elapsed = time.time() - start_time

        result = {
            'method': 'PC Algorithm',
            'time_seconds': elapsed,
            'success': True,
            'n_edges': len(edges),
            'edges': edges[:10]
        }

    except ImportError:
        print("\n⚠️  causal-learn库未安装")
        print("   安装方法: pip install causal-learn")

        elapsed = time.time() - start_time

        result = {
            'method': 'PC Algorithm',
            'time_seconds': elapsed,
            'success': False,
            'error': 'causal-learn not installed'
        }

    except Exception as e:
        print(f"\n❌ PC算法执行失败: {str(e)}")

        elapsed = time.time() - start_time

        result = {
            'method': 'PC Algorithm',
            'time_seconds': elapsed,
            'success': False,
            'error': str(e)
        }

    print(f"\n⏱️  耗时: {elapsed:.2f}秒")

    return result

def method5_mutual_information(df):
    """方法5: 互信息分析"""
    print_section("方法5: 互信息分析")

    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()

    print("\n[5.1] 计算互信息...")

    # 选择目标变量
    target_cols = [col for col in df.columns if 'energy' in col or 'perf' in col]
    feature_cols = [col for col in df.columns if col not in target_cols]

    results = {}

    for target in target_cols[:3]:  # 只分析前3个目标
        print(f"\n[目标: {target}]")

        # 准备数据
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target]

        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 10:
            print(f"   ⚠️  样本不足，跳过")
            continue

        print(f"   有效样本: {len(X)}个")

        # 计算互信息
        mi = mutual_info_regression(X, y, random_state=42)

        mi_scores = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi
        }).sort_values('mi_score', ascending=False)

        print(f"   Top 5特征（按互信息）:")
        for i, row in mi_scores.head(5).iterrows():
            print(f"      {row['feature']:30s}: {row['mi_score']:8.3f}")

        results[target] = {
            'top_features': mi_scores.head(5).to_dict('records')
        }

    elapsed = time.time() - start_time

    result = {
        'method': 'Mutual Information',
        'time_seconds': elapsed,
        'success': True,
        'targets': results
    }

    print(f"\n⏱️  耗时: {elapsed:.2f}秒")

    return result

def generate_comparison_report(results, output_dir):
    """生成方法对比报告"""
    print_section("方法对比总结")

    report_file = output_dir / 'method_comparison_report.md'

    with open(report_file, 'w') as f:
        f.write("# 能耗数据集方法对比报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 测试方法摘要\n\n")
        f.write("| 方法 | 运行时间 | 成功 | 关键发现 |\n")
        f.write("|------|----------|------|----------|\n")

        for r in results:
            method = r['method']
            time_sec = r['time_seconds']
            success = r.get('success', True)

            # 关键发现
            if 'Correlation' in method:
                key_finding = f"强相关对: {r.get('pearson_strong_pairs', 0)}"
            elif 'Regression' in method:
                key_finding = f"分析了{len(r.get('targets', {}))}个目标"
            elif 'PC Algorithm' in method:
                key_finding = f"检测到{r.get('n_edges', 0)}条边" if success else "失败"
            else:
                key_finding = "完成"

            status = "✅" if success else "❌"
            f.write(f"| {method} | {time_sec:.1f}秒 | {status} | {key_finding} |\n")

        f.write("\n---\n\n")

        # 详细结果
        for r in results:
            f.write(f"## {r['method']}\n\n")

            if 'Correlation' in r['method'] and r.get('success', False):
                f.write(f"### 统计摘要\n\n")
                f.write(f"- Pearson平均: {r.get('pearson_mean', 0):.3f}\n")
                f.write(f"- Pearson最大: {r.get('pearson_max', 0):.3f}\n")
                f.write(f"- 强相关对(|r|>0.7): {r.get('pearson_strong_pairs', 0)}\n\n")

                f.write(f"### Top 10相关对\n\n")
                for i, (var1, var2, corr) in enumerate(r.get('top_pairs', []), 1):
                    f.write(f"{i}. {var1} <-> {var2}: {corr:.3f}\n")

            elif 'Regression' in r['method'] and r.get('success', False):
                for target, data in r.get('targets', {}).items():
                    f.write(f"### 目标: {target}\n\n")
                    f.write(f"- 线性回归 R²: {data.get('lr_r2', 0):.3f}\n")
                    f.write(f"- 随机森林 R²: {data.get('rf_r2', 0):.3f}\n\n")

            elif 'PC Algorithm' in r['method']:
                if r.get('success', False):
                    f.write(f"- 检测到边数: {r.get('n_edges', 0)}\n")
                    if r.get('edges'):
                        f.write(f"\n检测到的边:\n")
                        for var1, var2, edge_type in r['edges']:
                            f.write(f"- {var1} → {var2}\n")
                else:
                    f.write(f"- 失败原因: {r.get('error', '未知')}\n")

            f.write("\n---\n\n")

        # 推荐
        f.write("## 方法推荐\n\n")
        f.write("基于测试结果，推荐使用的方法按优先级排序：\n\n")

        # 简单评分
        scores = []
        for r in results:
            if 'Correlation' in r['method']:
                score = 5  # 相关性分析总是有用
                scores.append((r['method'], score, "快速、直观、易解释"))
            elif 'Regression' in r['method']:
                score = 4  # 回归分析很有用
                scores.append((r['method'], score, "提供特征重要性，可预测"))
            elif 'Mutual' in r['method']:
                score = 3
                scores.append((r['method'], score, "捕捉非线性关系"))
            elif 'Partial' in r['method']:
                score = 3
                scores.append((r['method'], score, "控制混淆变量"))
            elif 'PC' in r['method']:
                if r.get('success') and r.get('n_edges', 0) > 0:
                    score = 4
                    scores.append((r['method'], score, "发现因果结构"))
                else:
                    score = 1
                    scores.append((r['method'], score, "未成功"))

        scores.sort(key=lambda x: x[1], reverse=True)

        for i, (method, score, reason) in enumerate(scores, 1):
            stars = "⭐" * score
            f.write(f"{i}. **{method}** {stars}\n")
            f.write(f"   - 原因: {reason}\n\n")

    print(f"\n✅ 对比报告已保存: {report_file}")

    return report_file

def main():
    """主函数"""
    print("=" * 80)
    print("  能耗数据集方法对比测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 配置
    data_path = "data/energy_research/processed/training_data_image_classification_examples.csv"
    output_dir = Path("results/energy_research/method_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # 加载数据
    df = load_and_prepare_data(data_path)

    # 运行所有方法
    results = []

    # 方法1: 相关性分析
    try:
        result1 = method1_correlation_analysis(df)
        results.append(result1)
    except Exception as e:
        print(f"❌ 方法1失败: {e}")

    # 方法2: 回归分析
    try:
        result2 = method2_regression_analysis(df)
        results.append(result2)
    except Exception as e:
        print(f"❌ 方法2失败: {e}")

    # 方法3: 偏相关分析
    try:
        result3 = method3_partial_correlation(df)
        results.append(result3)
    except Exception as e:
        print(f"❌ 方法3失败: {e}")

    # 方法4: PC算法
    try:
        result4 = method4_pc_algorithm(df)
        results.append(result4)
    except Exception as e:
        print(f"❌ 方法4失败: {e}")

    # 方法5: 互信息
    try:
        result5 = method5_mutual_information(df)
        results.append(result5)
    except Exception as e:
        print(f"❌ 方法5失败: {e}")

    # 生成对比报告
    report_file = generate_comparison_report(results, output_dir)

    # 保存结果
    results_json = output_dir / 'results.json'
    # 创建可序列化的副本
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        # 移除不能序列化的对象
        if 'corr_matrix' in r_copy:
            del r_copy['corr_matrix']
        # 移除top_pairs（包含元组）
        if 'top_pairs' in r_copy:
            r_copy['top_pairs'] = [(str(v1), str(v2), float(c)) for v1, v2, c in r_copy['top_pairs']]
        if 'partial_corrs' in r_copy:
            r_copy['partial_corrs'] = [(str(v1), str(v2), float(c)) for v1, v2, c in r_copy['partial_corrs']]
        results_to_save.append(r_copy)

    with open(results_json, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    total_time = time.time() - total_start

    print_section("完成")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"报告文件: {report_file}")
    print(f"结果文件: {results_json}")
    print("=" * 80)

if __name__ == '__main__':
    main()
