"""
Adult数据集完整因果分析 - 优化版
适合长时间后台运行，包含完整的进度报告和检查点保存
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator, define_sign_functions
from utils.fairness_methods import get_fairness_method

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def save_checkpoint(data, filename):
    """保存检查点"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ 检查点已保存: {filename}")
    except Exception as e:
        print(f"  ⚠️  保存检查点失败: {e}")

def load_checkpoint(filename):
    """加载检查点"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  加载检查点失败: {e}")
    return None

print_header("Adult数据集完整因果分析 - 后台运行版")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"预计运行时间: 2-3小时")

start_time = time.time()
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================================
# 步骤1: 加载数据（或从检查点恢复）
# ============================================================================
print_header("步骤1: 加载Adult数据集")

checkpoint_file = 'results/adult_data_checkpoint.pkl'
checkpoint = load_checkpoint(checkpoint_file)

if checkpoint:
    print("  ✓ 从检查点恢复数据")
    X_train = checkpoint['X_train']
    X_test = checkpoint['X_test']
    y_train = checkpoint['y_train']
    y_test = checkpoint['y_test']
    sensitive_train = checkpoint['sensitive_train']
    sensitive_test = checkpoint['sensitive_test']
    n_features = checkpoint['n_features']
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  特征数: {n_features}")
else:
    print("  加载新数据...")
    try:
        from aif360.datasets import AdultDataset

        dataset = AdultDataset(
            protected_attribute_names=['sex'],
            privileged_classes=[['Male']],
            categorical_features=['workclass', 'education', 'marital-status',
                                 'occupation', 'relationship', 'race', 'native-country'],
            features_to_drop=['fnlwgt']
        )

        X_full = dataset.features
        y_full = dataset.labels.ravel()
        sensitive_full = dataset.protected_attributes.ravel()

        print(f"  ✓ 数据集加载成功: {len(X_full)} 样本, {X_full.shape[1]} 特征")

        # 分割数据
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X_full, y_full, sensitive_full,
            test_size=0.3,
            random_state=42,
            stratify=y_full
        )

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_features = X_train.shape[1]

        print(f"  ✓ 数据准备完成")
        print(f"    训练集: {len(X_train)} 样本")
        print(f"    测试集: {len(X_test)} 样本")

        # 保存检查点
        save_checkpoint({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'sensitive_train': sensitive_train,
            'sensitive_test': sensitive_test,
            'n_features': n_features
        }, checkpoint_file)

    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        sys.exit(1)

# ============================================================================
# 步骤2: 数据收集（或从检查点恢复）
# ============================================================================
print_header("步骤2: 数据收集（检测已有数据）")

# 配置
METHODS = ['Baseline', 'Reweighing']
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCHS = 50
MODEL_WIDTH = 2

data_file = 'data/adult_training_data.csv'

if os.path.exists(data_file):
    print(f"  ✓ 发现已有数据文件: {data_file}")
    df = pd.read_csv(data_file)
    print(f"  已有数据点: {len(df)}")
    print(f"  跳过数据收集步骤")
else:
    print(f"  未找到数据文件，开始数据收集...")
    print(f"  配置: {len(METHODS)}方法 × {len(ALPHA_VALUES)} alpha = {len(METHODS)*len(ALPHA_VALUES)}个")

    results = []
    total_configs = len(METHODS) * len(ALPHA_VALUES)

    for idx, (method_name, alpha) in enumerate(
        [(m, a) for m in METHODS for a in ALPHA_VALUES], 1
    ):
        config_start = time.time()
        print(f"\n  [{idx}/{total_configs}] {method_name}, α={alpha:.2f}")

        try:
            # 应用方法
            method = get_fairness_method(method_name, alpha, sensitive_attr='sex')
            X_transformed, y_transformed = method.fit_transform(
                X_train, y_train, sensitive_train
            )

            # 训练模型
            model = FFNN(input_dim=n_features, width=MODEL_WIDTH)
            trainer = ModelTrainer(model, device=device, lr=0.001)
            trainer.train(X_transformed, y_transformed, epochs=EPOCHS, batch_size=256, verbose=False)

            # 计算指标
            calculator = MetricsCalculator(trainer, sensitive_attr='sex')

            dataset_metrics = calculator.compute_all_metrics(X_train, y_train, sensitive_train, phase='D')
            train_metrics = calculator.compute_all_metrics(X_transformed, y_transformed, sensitive_train, phase='Tr')
            test_metrics = calculator.compute_all_metrics(X_test, y_test, sensitive_test, phase='Te')

            row = {'method': method_name, 'alpha': alpha, 'Width': MODEL_WIDTH}
            row.update(dataset_metrics)
            row.update(train_metrics)
            row.update(test_metrics)
            results.append(row)

            elapsed = time.time() - start_time
            eta = (total_configs - idx) * (elapsed / idx) / 60
            print(f"    ✓ Acc={test_metrics.get('Te_Acc', 0):.3f} | 耗时={time.time()-config_start:.0f}s | ETA={eta:.1f}min")

        except Exception as e:
            print(f"    ✗ 失败: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(data_file, index=False)
    print(f"\n  ✓ 数据收集完成，保存到: {data_file}")

# ============================================================================
# 步骤3: DiBS因果图学习（优化版）
# ============================================================================
print_header("步骤3: DiBS因果图学习（优化版）")

graph_file = 'results/adult_causal_graph.npy'
edges_file = 'results/adult_causal_edges.pkl'

if os.path.exists(graph_file) and os.path.exists(edges_file):
    print(f"  ✓ 发现已有因果图: {graph_file}")
    try:
        causal_graph = np.load(graph_file)
        with open(edges_file, 'rb') as f:
            edge_data = pickle.load(f)
        edges = edge_data['edges']
        numeric_cols = edge_data['numeric_cols']
        print(f"  因果边数: {len(edges)}")
        print(f"  跳过DiBS学习步骤")
    except Exception as e:
        print(f"  ⚠️  加载失败: {e}，重新学习...")
        causal_graph = None
else:
    print(f"  开始DiBS因果图学习...")
    print(f"  配置: 变量数={len(df.columns)-3}, 迭代=3000步（优化版）")
    print(f"  预计时间: 30-45分钟")

    try:
        from utils.causal_discovery import CausalGraphLearner

        # 准备数据
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Width' in numeric_cols:
            numeric_cols.remove('Width')

        causal_data = df[numeric_cols]
        print(f"  变量数: {len(numeric_cols)}")
        print(f"  数据点: {len(causal_data)}")

        # 创建学习器（降低迭代次数以加快速度）
        learner = CausalGraphLearner(
            n_vars=len(numeric_cols),
            n_steps=3000,  # 从5000降到3000
            alpha=0.1,
            random_seed=42
        )

        # 学习因果图
        dibs_start = time.time()
        print(f"\n  开始时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  正在运行DiBS（请耐心等待）...")

        # 每100步报告一次进度（如果可能）
        causal_graph = learner.fit(causal_data, verbose=True)

        dibs_time = time.time() - dibs_start
        print(f"\n  ✓ DiBS完成，耗时: {dibs_time/60:.1f}分钟")

        # 分析边
        edges = learner.get_edges(threshold=0.3)
        print(f"  检测到 {len(edges)} 条因果边")

        # 保存结果
        learner.save_graph(graph_file)
        with open(edges_file, 'wb') as f:
            pickle.dump({'edges': edges, 'numeric_cols': numeric_cols}, f)
        print(f"  ✓ 因果图已保存")

        # 显示关键边
        if len(edges) > 0:
            print(f"\n  前10条最强因果边:")
            for i, (source, target, weight) in enumerate(edges[:10], 1):
                print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")

    except Exception as e:
        print(f"  ✗ DiBS失败: {e}")
        import traceback
        traceback.print_exc()
        causal_graph = None
        edges = []
        numeric_cols = []

# ============================================================================
# 步骤4: DML因果推断
# ============================================================================
if causal_graph is not None and len(edges) > 0:
    print_header("步骤4: DML因果推断")

    effects_file = 'results/adult_causal_effects.csv'

    if os.path.exists(effects_file):
        print(f"  ✓ 发现已有因果效应文件: {effects_file}")
        effects_df = pd.read_csv(effects_file)
        print(f"  因果效应数: {len(effects_df)}")
        print(f"  跳过DML步骤")
    else:
        print(f"  开始DML因果推断...")
        print(f"  分析 {len(edges)} 条因果边")
        print(f"  预计时间: 10-15分钟")

        try:
            from utils.causal_inference import CausalInferenceEngine

            engine = CausalInferenceEngine(verbose=True)
            causal_data = df[numeric_cols]

            dml_start = time.time()
            print(f"\n  开始时间: {datetime.now().strftime('%H:%M:%S')}")

            causal_effects = engine.analyze_all_edges(
                data=causal_data,
                causal_graph=causal_graph,
                var_names=numeric_cols,
                threshold=0.3
            )

            dml_time = time.time() - dml_start
            print(f"\n  ✓ DML完成，耗时: {dml_time/60:.1f}分钟")

            if causal_effects:
                engine.save_results(effects_file)
                print(f"  ✓ 因果效应已保存到: {effects_file}")

                significant = engine.get_significant_effects()
                print(f"\n  因果效应统计:")
                print(f"    总边数: {len(causal_effects)}")
                print(f"    统计显著: {len(significant)}")

                if significant:
                    print(f"\n  显著的因果效应 (前5个):")
                    for i, (edge, result) in enumerate(list(significant.items())[:5], 1):
                        print(f"    {i}. {edge}")
                        print(f"       ATE={result['ate']:.4f}, CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

        except Exception as e:
            print(f"  ✗ DML失败: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# 步骤5: 权衡检测
# ============================================================================
if os.path.exists('results/adult_causal_effects.csv'):
    print_header("步骤5: 权衡检测")

    try:
        from utils.tradeoff_detection import TradeoffDetector
        from utils.causal_inference import CausalInferenceEngine

        # 重新加载因果效应
        engine = CausalInferenceEngine(verbose=False)
        # 这里需要重新构建causal_effects字典...
        print(f"  基于已保存的因果效应进行权衡检测...")
        print(f"  （需要完整实现）")

    except Exception as e:
        print(f"  ⚠️  权衡检测跳过: {e}")

# ============================================================================
# 总结
# ============================================================================
total_time = time.time() - start_time

print_header("分析完成！")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")

print(f"\n生成的文件:")
for file in ['data/adult_training_data.csv', 'results/adult_causal_graph.npy',
             'results/adult_causal_effects.csv', 'results/adult_causal_edges.pkl']:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"  ✓ {file} ({size:.1f} KB)")

print(f"\n查看结果:")
print(f"  python -c \"import pandas as pd; df=pd.read_csv('data/adult_training_data.csv'); print(df.describe())\"")
print(f"  python -c \"import numpy as np; g=np.load('results/adult_causal_graph.npy'); print('因果图:', g.shape)\"")

print("\n" + "="*70)
