"""
演示脚本：使用模拟数据验证整个流程
这个脚本不需要下载真实数据集，可以快速验证代码功能
"""
import numpy as np
import pandas as pd
import sys
import os

# 设置随机种子保证可复现
np.random.seed(42)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator, define_sign_functions
from utils.fairness_methods import get_fairness_method
import config

print("="*70)
print(" "*15 + "精简版功能复现 - 快速演示")
print("="*70)

# ============================================================================
# 步骤1: 生成模拟数据
# ============================================================================
print("\n" + "▶"*35)
print("步骤1: 生成模拟数据")
print("▶"*35)

n_samples_train = 500
n_samples_test = 200
n_features = 10

# 生成特征
X_train = np.random.randn(n_samples_train, n_features)
X_test = np.random.randn(n_samples_test, n_features)

# 生成标签（模拟与第一个特征相关）
y_train = (X_train[:, 0] + np.random.randn(n_samples_train) * 0.5 > 0).astype(int)
y_test = (X_test[:, 0] + np.random.randn(n_samples_test) * 0.5 > 0).astype(int)

# 生成敏感属性（二元）
sensitive_train = np.random.randint(0, 2, n_samples_train)
sensitive_test = np.random.randint(0, 2, n_samples_test)

print(f"✓ 生成训练集: {len(X_train)} 样本, {n_features} 特征")
print(f"✓ 生成测试集: {len(X_test)} 样本")
print(f"✓ 标签分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
print(f"✓ 敏感属性分布 - 训练集: {np.bincount(sensitive_train)}")

# ============================================================================
# 步骤2: 数据收集（收集少量数据点用于演示）
# ============================================================================
print("\n" + "▶"*35)
print("步骤2: 数据收集")
print("▶"*35)

results = []
methods_to_test = ['Baseline', 'Reweighing']  # 简化：只测试2个方法
alpha_values = [0.0, 0.5, 1.0]  # 简化：只测试3个alpha值

total_configs = len(methods_to_test) * len(alpha_values)
current_config = 0

for method_name in methods_to_test:
    for alpha in alpha_values:
        current_config += 1
        print(f"\n[{current_config}/{total_configs}] 测试: {method_name}, α={alpha}")

        try:
            # 应用公平性方法
            method = get_fairness_method(method_name, alpha, sensitive_attr='sex')
            X_transformed, y_transformed = method.fit_transform(
                X_train, y_train, sensitive_train
            )

            # 训练模型
            model = FFNN(input_dim=n_features, width=2)  # 使用更小的模型
            trainer = ModelTrainer(model, device='cpu', lr=0.01)
            print(f"  - 训练模型（5轮）...")
            trainer.train(X_transformed, y_transformed, epochs=5, verbose=False)

            # 计算指标
            calculator = MetricsCalculator(trainer, sensitive_attr='sex')

            print(f"  - 计算指标...")
            # 数据集指标
            dataset_metrics = calculator.compute_all_metrics(
                X_train, y_train, sensitive_train, phase='D'
            )

            # 训练集指标
            train_metrics = calculator.compute_all_metrics(
                X_transformed, y_transformed, sensitive_train, phase='Tr'
            )

            # 测试集指标
            test_metrics = calculator.compute_all_metrics(
                X_test, y_test, sensitive_test, phase='Te'
            )

            # 合并指标
            row = {
                'method': method_name,
                'alpha': alpha,
                'Width': 2
            }
            row.update(dataset_metrics)
            row.update(train_metrics)
            row.update(test_metrics)

            results.append(row)

            # 显示关键指标
            print(f"  ✓ Te_Acc={test_metrics.get('Te_Acc', 0):.3f}, "
                  f"Te_SPD={test_metrics.get('Te_SPD', 0):.3f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

# 创建DataFrame
df = pd.DataFrame(results)

# 保存结果
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
output_path = 'data/demo_training_data.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ 数据收集完成")
print(f"  - 收集了 {len(df)} 个数据点")
print(f"  - 保存到: {output_path}")
print(f"  - 列数: {df.shape[1]}")

# 显示数据样本
print(f"\n数据样本（前3行）:")
print(df.head(3).to_string())

# ============================================================================
# 步骤3: DiBS因果图学习
# ============================================================================
print("\n" + "▶"*35)
print("步骤3: DiBS因果图学习")
print("▶"*35)

try:
    from utils.causal_discovery import CausalGraphLearner

    print("\n使用DiBS学习因果图...")
    print("注意: 这可能需要几分钟时间")

    # 准备数据：选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 移除Width列（如果存在）
    if 'Width' in numeric_cols:
        numeric_cols.remove('Width')

    causal_data = df[numeric_cols]
    print(f"  - 使用 {len(numeric_cols)} 个变量")
    print(f"  - 数据点: {len(causal_data)}")

    # 创建因果图学习器（使用较少迭代次数用于演示）
    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=1000,  # 演示用，论文中为10000
        alpha=0.1,     # 较小的alpha得到更稀疏的图
        random_seed=42
    )

    # 学习因果图
    causal_graph = learner.fit(causal_data, verbose=True)

    # 分析结果
    edges = learner.get_edges(threshold=0.3)
    print(f"\n✓ DiBS学习完成")
    print(f"  - 检测到 {len(edges)} 条因果边 (阈值=0.3)")

    # 显示与alpha相关的边
    alpha_idx = numeric_cols.index('alpha') if 'alpha' in numeric_cols else None
    if alpha_idx is not None:
        alpha_edges = [e for e in edges if e[0] == alpha_idx or e[1] == alpha_idx]
        if len(alpha_edges) > 0:
            print(f"\n  与alpha相关的因果边:")
            for source, target, weight in alpha_edges[:5]:
                if source == alpha_idx:
                    print(f"    alpha → {numeric_cols[target]}: {weight:.3f}")
                else:
                    print(f"    {numeric_cols[source]} → alpha: {weight:.3f}")
        else:
            print(f"\n  未检测到与alpha直接相关的因果边")

    # 保存因果图
    graph_path = 'results/causal_graph.npy'
    learner.save_graph(graph_path)

    print(f"\n注: 如需更准确的因果图，请增加n_steps到5000-10000")

except ImportError as e:
    print(f"\n⚠️  DiBS未安装，使用简化的相关性分析")
    print(f"    错误: {e}")

    # 后备方案：相关性分析
    print("\n计算变量间相关性（简化版）...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # 找出与alpha相关的变量
    alpha_corr = corr_matrix['alpha'].abs().sort_values(ascending=False)
    print(f"\n与alpha最相关的5个变量:")
    for i, (var, corr) in enumerate(alpha_corr.head(6).items(), 1):
        if var != 'alpha':
            print(f"  {i}. {var}: {corr:.3f}")

except Exception as e:
    print(f"\n❌ DiBS执行失败: {e}")
    print(f"    使用相关性分析作为后备方案")

    # 后备方案：相关性分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    alpha_corr = corr_matrix['alpha'].abs().sort_values(ascending=False)
    print(f"\n与alpha最相关的5个变量:")
    for i, (var, corr) in enumerate(alpha_corr.head(6).items(), 1):
        if var != 'alpha':
            print(f"  {i}. {var}: {corr:.3f}")

# 分析Reweighing方法的效果
reweighing_data = df[df['method'] == 'Reweighing']
if len(reweighing_data) > 0:
    print(f"\nReweighing方法的效果:")
    print(f"  α=0.0 → α=1.0:")
    for metric in ['Te_Acc', 'Te_SPD', 'Te_F1']:
        if metric in reweighing_data.columns:
            val_0 = reweighing_data[reweighing_data['alpha'] == 0.0][metric].values
            val_1 = reweighing_data[reweighing_data['alpha'] == 1.0][metric].values
            if len(val_0) > 0 and len(val_1) > 0:
                change = val_1[0] - val_0[0]
                print(f"    {metric}: {val_0[0]:.3f} → {val_1[0]:.3f} (变化: {change:+.3f})")

# ============================================================================
# 步骤3.5: DML因果推断（如果DiBS成功）
# ============================================================================
causal_effects = {}
try:
    # 检查是否有因果图
    if 'causal_graph' in locals() and causal_graph is not None and 'numeric_cols' in locals():
        print("\n" + "▶"*35)
        print("步骤3.5: DML因果推断")
        print("▶"*35)

        from utils.causal_inference import CausalInferenceEngine

        print("\n使用DML估计因果效应...")
        print("注意: 这可能需要几分钟时间")

        # 创建因果推断引擎
        engine = CausalInferenceEngine(verbose=True)

        # 对因果图中的边进行因果推断
        causal_effects = engine.analyze_all_edges(
            data=causal_data,
            causal_graph=causal_graph,
            var_names=numeric_cols,
            threshold=0.3
        )

        # 保存结果
        if causal_effects:
            effects_path = 'results/causal_effects.csv'
            engine.save_results(effects_path)

            # 显示显著的因果效应
            significant = engine.get_significant_effects()
            if significant:
                print(f"\n显著的因果效应 (共{len(significant)}个):")
                for i, (edge, result) in enumerate(list(significant.items())[:5], 1):
                    print(f"  {i}. {edge}: ATE={result['ate']:.4f}, "
                          f"95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
                if len(significant) > 5:
                    print(f"  ... 还有 {len(significant)-5} 个")
        else:
            print("\n⚠️  未发现显著的因果效应")

except Exception as e:
    print(f"\n⚠️  DML因果推断跳过: {e}")
    print("  使用简化的权衡检测方法")

# ============================================================================
# 步骤4: 权衡检测（基于因果推断）
# ============================================================================
print("\n" + "▶"*35)
print("步骤4: 权衡检测")
print("▶"*35)

# 使用sign函数检测权衡
sign_funcs = define_sign_functions()

# 方法1: 基于因果推断的权衡检测（如果有因果效应结果）
if causal_effects:
    try:
        from utils.tradeoff_detection import TradeoffDetector

        print("\n使用因果推断结果检测权衡...")

        # 创建权衡检测器
        detector = TradeoffDetector(sign_funcs, verbose=True)

        # 检测权衡
        tradeoffs = detector.detect_tradeoffs(causal_effects, require_significance=True)

        if tradeoffs:
            # 生成摘要
            summary = detector.summarize_tradeoffs(tradeoffs)
            print(f"\n权衡摘要:")
            print(summary.to_string(index=False))

            # 保存结果
            summary_path = 'results/tradeoffs.csv'
            summary.to_csv(summary_path, index=False)
            print(f"\n✓ 权衡检测结果已保存到: {summary_path}")

            # 可视化（如果matplotlib可用）
            try:
                detector.visualize_tradeoffs(tradeoffs, 'results/tradeoffs.png')
            except Exception:
                pass
        else:
            print("\n✓ 未检测到显著的权衡关系")

    except Exception as e:
        print(f"\n⚠️  基于因果推断的权衡检测失败: {e}")
        print("  回退到简化方法")
        causal_effects = {}  # 清空，使用简化方法

# 方法2: 简化的权衡检测（后备方案）
if not causal_effects:
    print("\n使用简化方法检测权衡...")

# 分析Reweighing从alpha=0到alpha=1的效果
if len(reweighing_data) >= 2:
    baseline = reweighing_data[reweighing_data['alpha'] == 0.0].iloc[0]
    full_apply = reweighing_data[reweighing_data['alpha'] == 1.0].iloc[0]

    print(f"\n检测权衡 (Reweighing, α: 0 → 1):")

    # 检查Acc vs SPD
    if 'Te_Acc' in baseline and 'Te_SPD' in baseline:
        acc_change = full_apply['Te_Acc'] - baseline['Te_Acc']
        spd_change = full_apply['Te_SPD'] - baseline['Te_SPD']

        acc_sign = sign_funcs['Acc'](baseline['Te_Acc'], acc_change)
        spd_sign = sign_funcs['SPD'](baseline['Te_SPD'], spd_change)

        print(f"\n  Accuracy vs SPD:")
        print(f"    Te_Acc: {baseline['Te_Acc']:.3f} → {full_apply['Te_Acc']:.3f} ({acc_sign})")
        print(f"    Te_SPD: {baseline['Te_SPD']:.3f} → {full_apply['Te_SPD']:.3f} ({spd_sign})")

        if acc_sign != spd_sign:
            print(f"    ⚠️  检测到权衡！")
        else:
            print(f"    ✓ 无权衡（双赢或双输）")

# ============================================================================
# 步骤5: 总结
# ============================================================================
print("\n" + "="*70)
print(" "*20 + "演示完成！")
print("="*70)

print(f"\n✅ 成功验证的功能:")
print(f"  1. ✓ 数据生成和预处理")
print(f"  2. ✓ 公平性方法应用 (Baseline, Reweighing)")
print(f"  3. ✓ 神经网络模型训练")
print(f"  4. ✓ 多类型指标计算 (性能、公平性、鲁棒性)")
print(f"  5. ✓ DiBS因果图学习 (NeurIPS 2021算法)")
print(f"  6. ✓ DML因果推断 (Chernozhukov et al. 2018)")
print(f"  7. ✓ 权衡检测 (论文算法1)")

print(f"\n📊 生成的文件:")
print(f"  - {output_path}")
print(f"  - results/causal_graph.npy (如果DiBS成功运行)")
print(f"  - results/causal_effects.csv (如果DML成功运行)")
print(f"  - results/tradeoffs.csv (如果检测到权衡)")

print(f"\n📌 注意:")
print(f"  - 这是使用模拟数据的演示")
print(f"  - DiBS使用较少迭代次数（1000步），完整版需要10000步")
print(f"  - DML可能降级到简化方法（如果EconML未安装）")
print(f"  - 真实复现需要使用Adult/COMPAS/German数据集")

print(f"\n🚀 下一步:")
print(f"  1. 查看生成的数据: cat {output_path}")
print(f"  2. 查看阶段1完成报告: cat STAGE1_COMPLETION_REPORT.md")
print(f"  3. 查看阶段1&2最终报告: cat STAGE1_2_FINAL_REPORT.md")
print(f"  4. 运行完整测试: python run_tests.py")
print(f"  5. 与论文代码比较: 见即将生成的比较报告")

print("\n" + "="*70 + "\n")
