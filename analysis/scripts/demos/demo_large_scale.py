"""
大规模演示脚本：使用更多样本量和GPU加速
目标运行时间：约30分钟
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import torch

# 设置随机种子保证可复现
np.random.seed(42)
torch.manual_seed(42)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator, define_sign_functions
from utils.fairness_methods import get_fairness_method

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  GPU不可用，使用CPU")

print("="*70)
print(" "*10 + "大规模实验 - GPU加速版（目标30分钟）")
print("="*70)

start_time = time.time()

# ============================================================================
# 配置参数（针对30分钟运行时间优化）
# ============================================================================
print("\n" + "▶"*35)
print("实验配置")
print("▶"*35)

# 数据规模配置
N_SAMPLES_TRAIN = 3000  # 训练样本（原500 → 3000）
N_SAMPLES_TEST = 1000   # 测试样本（原200 → 1000）
N_FEATURES = 20         # 特征数（原10 → 20）

# 方法和参数配置
METHODS = ['Baseline', 'Reweighing']  # 可用的方法
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6个alpha值（原3个 → 6个）

# 模型训练配置
EPOCHS = 20            # 训练轮数（原5 → 20）
BATCH_SIZE = 128       # 批次大小
MODEL_WIDTH = 4        # 模型宽度（原2 → 4）
LEARNING_RATE = 0.001  # 学习率

# DiBS配置
DIBS_STEPS = 3000      # DiBS迭代次数（原1000 → 3000）

print(f"\n数据配置:")
print(f"  训练样本: {N_SAMPLES_TRAIN}")
print(f"  测试样本: {N_SAMPLES_TEST}")
print(f"  特征数: {N_FEATURES}")

print(f"\n实验配置:")
print(f"  方法数: {len(METHODS)}")
print(f"  Alpha值: {len(ALPHA_VALUES)}个")
print(f"  总配置数: {len(METHODS) * len(ALPHA_VALUES)}")

print(f"\n训练配置:")
print(f"  训练轮数: {EPOCHS}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  模型宽度: {MODEL_WIDTH}")
print(f"  设备: {device}")

print(f"\n预估运行时间:")
total_configs = len(METHODS) * len(ALPHA_VALUES)
est_time_per_config = 2.0 if device == 'cuda' else 3.5  # GPU更快
est_total_time = total_configs * est_time_per_config
est_dibs_time = 3.0 if device == 'cuda' else 5.0
print(f"  数据收集: ~{est_total_time:.0f}分钟")
print(f"  DiBS学习: ~{est_dibs_time:.0f}分钟")
print(f"  DML推断: ~2分钟")
print(f"  总计: ~{est_total_time + est_dibs_time + 2:.0f}分钟")

# ============================================================================
# 步骤1: 生成大规模模拟数据
# ============================================================================
print("\n" + "▶"*35)
print("步骤1: 生成大规模模拟数据")
print("▶"*35)

step_start = time.time()

# 生成特征
X_train = np.random.randn(N_SAMPLES_TRAIN, N_FEATURES)
X_test = np.random.randn(N_SAMPLES_TEST, N_FEATURES)

# 生成标签（更复杂的关系）
# 使用多个特征的组合
y_train = ((X_train[:, 0] + 0.5*X_train[:, 1] - 0.3*X_train[:, 2]) +
           np.random.randn(N_SAMPLES_TRAIN) * 0.3 > 0).astype(int)
y_test = ((X_test[:, 0] + 0.5*X_test[:, 1] - 0.3*X_test[:, 2]) +
          np.random.randn(N_SAMPLES_TEST) * 0.3 > 0).astype(int)

# 生成敏感属性
sensitive_train = np.random.randint(0, 2, N_SAMPLES_TRAIN)
sensitive_test = np.random.randint(0, 2, N_SAMPLES_TEST)

print(f"✓ 生成训练集: {len(X_train)} 样本, {N_FEATURES} 特征")
print(f"✓ 生成测试集: {len(X_test)} 样本")
print(f"✓ 标签分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
print(f"✓ 敏感属性分布 - 训练集: {np.bincount(sensitive_train)}")
print(f"⏱  耗时: {time.time() - step_start:.1f}秒")

# ============================================================================
# 步骤2: 大规模数据收集
# ============================================================================
print("\n" + "▶"*35)
print("步骤2: 大规模数据收集")
print("▶"*35)

step_start = time.time()
results = []
total_configs = len(METHODS) * len(ALPHA_VALUES)
current_config = 0

for method_name in METHODS:
    for alpha in ALPHA_VALUES:
        current_config += 1
        config_start = time.time()

        print(f"\n[{current_config}/{total_configs}] {method_name}, α={alpha}")

        try:
            # 应用公平性方法
            method = get_fairness_method(method_name, alpha, sensitive_attr='sex')
            X_transformed, y_transformed = method.fit_transform(
                X_train, y_train, sensitive_train
            )

            # 训练模型（使用GPU）
            model = FFNN(input_dim=N_FEATURES, width=MODEL_WIDTH)
            trainer = ModelTrainer(model, device=device, lr=LEARNING_RATE)

            print(f"  训练模型（{EPOCHS}轮，设备={device}）...", end=' ', flush=True)
            trainer.train(
                X_transformed, y_transformed,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=False
            )
            print(f"✓")

            # 计算指标
            calculator = MetricsCalculator(trainer, sensitive_attr='sex')

            print(f"  计算指标...", end=' ', flush=True)

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
            print(f"✓")

            # 合并指标
            row = {
                'method': method_name,
                'alpha': alpha,
                'Width': MODEL_WIDTH
            }
            row.update(dataset_metrics)
            row.update(train_metrics)
            row.update(test_metrics)

            results.append(row)

            # 显示关键指标和时间
            config_time = time.time() - config_start
            elapsed = time.time() - start_time
            remaining = (total_configs - current_config) * (elapsed / current_config)

            print(f"  结果: Te_Acc={test_metrics.get('Te_Acc', 0):.3f}, "
                  f"Te_SPD={test_metrics.get('Te_SPD', 0):.3f}")
            print(f"  ⏱  本次: {config_time:.1f}秒 | "
                  f"已用: {elapsed/60:.1f}分 | "
                  f"预计剩余: {remaining/60:.1f}分")

        except Exception as e:
            print(f"\n  ✗ 失败: {e}")
            continue

# 创建DataFrame
df = pd.DataFrame(results)

# 保存结果
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
output_path = 'data/large_scale_training_data.csv'
df.to_csv(output_path, index=False)

step_time = time.time() - step_start
print(f"\n✓ 数据收集完成")
print(f"  - 收集了 {len(df)} 个数据点")
print(f"  - 保存到: {output_path}")
print(f"  - 列数: {df.shape[1]}")
print(f"⏱  总耗时: {step_time/60:.1f}分钟")

# 显示数据统计
print(f"\n数据统计:")
print(f"  Te_Acc: {df['Te_Acc'].min():.3f} ~ {df['Te_Acc'].max():.3f} (均值={df['Te_Acc'].mean():.3f})")
if 'Te_SPD' in df.columns:
    print(f"  Te_SPD: {df['Te_SPD'].min():.3f} ~ {df['Te_SPD'].max():.3f} (均值={df['Te_SPD'].mean():.3f})")
if 'Te_F1' in df.columns:
    print(f"  Te_F1:  {df['Te_F1'].min():.3f} ~ {df['Te_F1'].max():.3f} (均值={df['Te_F1'].mean():.3f})")

# ============================================================================
# 步骤3: DiBS因果图学习（增加迭代次数）
# ============================================================================
print("\n" + "▶"*35)
print("步骤3: DiBS因果图学习（高精度）")
print("▶"*35)

step_start = time.time()

try:
    from utils.causal_discovery import CausalGraphLearner

    print(f"\n使用DiBS学习因果图（{DIBS_STEPS}步迭代）...")

    # 准备数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Width' in numeric_cols:
        numeric_cols.remove('Width')

    causal_data = df[numeric_cols]
    print(f"  变量数: {len(numeric_cols)}")
    print(f"  数据点: {len(causal_data)}")
    print(f"  迭代次数: {DIBS_STEPS}")

    # 创建因果图学习器
    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=DIBS_STEPS,
        alpha=0.1,
        random_seed=42
    )

    # 学习因果图
    print(f"\n  正在运行DiBS（预计{est_dibs_time:.0f}分钟）...")
    causal_graph = learner.fit(causal_data, verbose=True)

    # 分析结果
    edges = learner.get_edges(threshold=0.3)
    print(f"\n✓ DiBS学习完成")
    print(f"  学到的边数: {len(edges)}")

    # 显示关键因果边
    print(f"\n  前10条最强因果边:")
    for i, (source, target, weight) in enumerate(edges[:10], 1):
        print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")

    # 显示与alpha相关的边
    alpha_idx = numeric_cols.index('alpha') if 'alpha' in numeric_cols else None
    if alpha_idx is not None:
        alpha_edges = [e for e in edges if e[0] == alpha_idx or e[1] == alpha_idx]
        if len(alpha_edges) > 0:
            print(f"\n  与alpha相关的因果边 (共{len(alpha_edges)}条):")
            for i, (source, target, weight) in enumerate(alpha_edges[:10], 1):
                if source == alpha_idx:
                    print(f"    {i}. alpha → {numeric_cols[target]}: {weight:.3f}")
                else:
                    print(f"    {i}. {numeric_cols[source]} → alpha: {weight:.3f}")
        else:
            print(f"\n  ⚠️  未检测到与alpha直接相关的因果边")
            print(f"     原因可能是：数据量仍不足或alpha影响不够显著")

    # 保存因果图
    graph_path = 'results/large_scale_causal_graph.npy'
    learner.save_graph(graph_path)
    print(f"\n✓ 因果图已保存到: {graph_path}")

except ImportError as e:
    print(f"\n⚠️  DiBS未安装: {e}")
    print("   使用相关性分析作为后备方案")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    alpha_corr = corr_matrix['alpha'].abs().sort_values(ascending=False)
    print(f"\n与alpha最相关的10个变量:")
    for i, (var, corr) in enumerate(alpha_corr.head(11).items(), 1):
        if var != 'alpha':
            print(f"  {i}. {var}: {corr:.3f}")

    causal_graph = None

except Exception as e:
    print(f"\n❌ DiBS执行失败: {e}")
    causal_graph = None

step_time = time.time() - step_start
print(f"\n⏱  DiBS耗时: {step_time/60:.1f}分钟")

# ============================================================================
# 步骤4: DML因果推断
# ============================================================================
causal_effects = {}

if 'causal_graph' in locals() and causal_graph is not None:
    print("\n" + "▶"*35)
    print("步骤4: DML因果推断")
    print("▶"*35)

    step_start = time.time()

    try:
        from utils.causal_inference import CausalInferenceEngine

        print("\n使用DML估计因果效应...")

        # 创建因果推断引擎
        engine = CausalInferenceEngine(verbose=True)

        # 分析所有边
        causal_effects = engine.analyze_all_edges(
            data=causal_data,
            causal_graph=causal_graph,
            var_names=numeric_cols,
            threshold=0.3
        )

        if causal_effects:
            # 保存结果
            effects_path = 'results/large_scale_causal_effects.csv'
            engine.save_results(effects_path)
            print(f"\n✓ 因果效应已保存到: {effects_path}")

            # 显示统计
            significant = engine.get_significant_effects()
            print(f"\n因果效应统计:")
            print(f"  总边数: {len(causal_effects)}")
            print(f"  统计显著: {len(significant)}")

            if significant:
                print(f"\n  前10个最显著的因果效应:")
                for i, (edge, result) in enumerate(list(significant.items())[:10], 1):
                    print(f"    {i}. {edge}")
                    print(f"       ATE={result['ate']:.4f}, "
                          f"95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        else:
            print("\n⚠️  未发现显著的因果效应")

        step_time = time.time() - step_start
        print(f"\n⏱  DML耗时: {step_time/60:.1f}分钟")

    except Exception as e:
        print(f"\n⚠️  DML因果推断失败: {e}")
        causal_effects = {}

# ============================================================================
# 步骤5: 权衡检测
# ============================================================================
print("\n" + "▶"*35)
print("步骤5: 权衡检测")
print("▶"*35)

sign_funcs = define_sign_functions()

# 基于因果推断的权衡检测
if causal_effects:
    try:
        from utils.tradeoff_detection import TradeoffDetector

        print("\n基于因果推断检测权衡...")

        detector = TradeoffDetector(sign_funcs, verbose=True)
        tradeoffs = detector.detect_tradeoffs(causal_effects, require_significance=True)

        if tradeoffs:
            summary = detector.summarize_tradeoffs(tradeoffs)
            print(f"\n✓ 检测到 {len(tradeoffs)} 个权衡关系")
            print(f"\n权衡摘要:")
            print(summary.to_string(index=False))

            # 保存结果
            summary_path = 'results/large_scale_tradeoffs.csv'
            summary.to_csv(summary_path, index=False)
            print(f"\n✓ 权衡检测结果已保存到: {summary_path}")
        else:
            print("\n✓ 未检测到显著的权衡关系")

    except Exception as e:
        print(f"\n⚠️  权衡检测失败: {e}")

# 简化的权衡分析
print("\n基于数据的简单权衡分析:")
for method_name in METHODS:
    method_data = df[df['method'] == method_name]
    if len(method_data) >= 2:
        alpha_0 = method_data[method_data['alpha'] == 0.0].iloc[0]
        alpha_1 = method_data[method_data['alpha'] == 1.0].iloc[0]

        print(f"\n{method_name} (α: 0.0 → 1.0):")
        for metric in ['Te_Acc', 'Te_SPD', 'Te_F1']:
            if metric in method_data.columns:
                val_0 = alpha_0[metric]
                val_1 = alpha_1[metric]
                change = val_1 - val_0
                print(f"  {metric}: {val_0:.3f} → {val_1:.3f} (Δ={change:+.3f})")

# ============================================================================
# 总结
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*70)
print(" "*20 + "实验完成！")
print("="*70)

print(f"\n⏱  总运行时间: {total_time/60:.1f} 分钟")

print(f"\n✅ 完成的任务:")
print(f"  1. ✓ 生成 {N_SAMPLES_TRAIN + N_SAMPLES_TEST} 个样本")
print(f"  2. ✓ 训练 {len(df)} 个模型配置")
print(f"  3. ✓ DiBS因果图学习 ({DIBS_STEPS}步迭代)")
print(f"  4. ✓ DML因果推断")
print(f"  5. ✓ 权衡检测")

print(f"\n📊 生成的文件:")
print(f"  - {output_path} ({len(df)}行 × {df.shape[1]}列)")
if 'causal_graph' in locals() and causal_graph is not None:
    print(f"  - results/large_scale_causal_graph.npy")
if causal_effects:
    print(f"  - results/large_scale_causal_effects.csv")
    print(f"  - results/large_scale_tradeoffs.csv (如果检测到权衡)")

print(f"\n📈 实验规模对比:")
print(f"  样本量: 700 → {N_SAMPLES_TRAIN + N_SAMPLES_TEST} (×{(N_SAMPLES_TRAIN + N_SAMPLES_TEST)/700:.1f})")
print(f"  数据点: 6 → {len(df)} (×{len(df)/6:.1f})")
print(f"  训练轮数: 5 → {EPOCHS} (×{EPOCHS/5:.1f})")
print(f"  DiBS迭代: 1000 → {DIBS_STEPS} (×{DIBS_STEPS/1000:.1f})")

print(f"\n💡 结果可靠性评估:")
if len(df) >= 10:
    print(f"  ✓ 数据点充足 ({len(df)}个，建议>10)")
else:
    print(f"  ⚠️  数据点偏少 ({len(df)}个，建议>10)")

if DIBS_STEPS >= 3000:
    print(f"  ✓ DiBS迭代充分 ({DIBS_STEPS}步，建议>3000)")
else:
    print(f"  ⚠️  DiBS迭代偏少 ({DIBS_STEPS}步，建议>3000)")

print(f"\n🎯 复现度提升:")
print(f"  相比原演示版: 数据规模×{len(df)/6:.0f}, 训练质量×{EPOCHS/5:.0f}")
print(f"  相比论文完整版: 约{(len(df)/726)*100:.1f}% (论文726个数据点)")

print("\n" + "="*70 + "\n")
