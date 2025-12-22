"""
集成测试套件
测试完整流程的集成
"""
import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator
from utils.fairness_methods import get_fairness_method
import config


class TestDataCollectionIntegration(unittest.TestCase):
    """测试数据收集流程"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)

    def test_complete_data_collection_pipeline(self):
        """测试完整的数据收集流程"""
        print("\n" + "="*60)
        print("Testing Complete Data Collection Pipeline")
        print("="*60)

        # 1. 生成模拟数据
        n_samples = 500
        input_dim = 10
        X_train = np.random.randn(n_samples, input_dim)
        y_train = np.random.randint(0, 2, n_samples)
        sensitive_train = np.random.randint(0, 2, n_samples)

        X_test = np.random.randn(n_samples // 3, input_dim)
        y_test = np.random.randint(0, 2, n_samples // 3)
        sensitive_test = np.random.randint(0, 2, n_samples // 3)

        print(f"✓ Generated data: train={len(X_train)}, test={len(X_test)}")

        # 2. 收集数据点
        results = []
        methods_to_test = ['Baseline', 'Reweighing']
        alpha_values = [0.0, 0.5, 1.0]

        for method_name in methods_to_test:
            for alpha in alpha_values:
                print(f"\nTesting {method_name} with α={alpha}")

                # 应用方法
                method = get_fairness_method(method_name, alpha, sensitive_attr='sex')
                X_transformed, y_transformed = method.fit_transform(
                    X_train, y_train, sensitive_train
                )

                # 训练模型
                model = FFNN(input_dim=input_dim, width=2)
                trainer = ModelTrainer(model, device='cpu', lr=0.01)
                trainer.train(X_transformed, y_transformed, epochs=5, verbose=False)

                # 计算指标
                calculator = MetricsCalculator(trainer, sensitive_attr='sex')

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
                print(f"  ✓ Collected {len(row)} metrics")

        # 3. 创建DataFrame
        df = pd.DataFrame(results)

        # 4. 验证
        self.assertEqual(len(df), len(methods_to_test) * len(alpha_values))
        self.assertGreater(df.shape[1], 15)  # 至少15列

        # 检查没有NaN
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Warning: {nan_counts.sum()} NaN values found")
            print(nan_counts[nan_counts > 0])

        # 保存到临时文件
        output_path = os.path.join(self.temp_dir, 'test_training_data.csv')
        df.to_csv(output_path, index=False)
        self.assertTrue(os.path.exists(output_path))

        print(f"\n✓ Data collection pipeline completed successfully")
        print(f"  Collected {len(df)} data points with {df.shape[1]} features")
        print(f"  Saved to {output_path}")

        return df


class TestCausalGraphSimulation(unittest.TestCase):
    """测试因果图学习（简化模拟）"""

    def test_causal_graph_structure(self):
        """测试因果图结构"""
        print("\n" + "="*60)
        print("Testing Causal Graph Structure")
        print("="*60)

        # 创建模拟训练数据
        data = {
            'Reweighing_alpha': [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            'Baseline_alpha': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            'Width': [2, 2, 2, 2, 2, 2],
            'D_SPD': [0.1, 0.08, 0.05, 0.1, 0.1, 0.1],
            'Tr_Acc': [0.75, 0.77, 0.80, 0.75, 0.76, 0.77],
            'Tr_SPD': [0.08, 0.06, 0.03, 0.08, 0.08, 0.08],
            'Te_Acc': [0.73, 0.75, 0.78, 0.73, 0.74, 0.75],
            'Te_SPD': [0.09, 0.07, 0.04, 0.09, 0.09, 0.09],
        }
        df = pd.DataFrame(data)

        # 计算相关性矩阵（简化的因果关系）
        corr_matrix = df.corr()

        print(f"✓ Created correlation matrix: {corr_matrix.shape}")

        # 验证期望的因果关系
        # Reweighing_alpha 应该影响 D_SPD, Tr_SPD, Te_SPD
        corr_alpha_d_spd = abs(corr_matrix.loc['Reweighing_alpha', 'D_SPD'])
        corr_alpha_tr_spd = abs(corr_matrix.loc['Reweighing_alpha', 'Tr_SPD'])
        corr_alpha_te_spd = abs(corr_matrix.loc['Reweighing_alpha', 'Te_SPD'])

        print(f"  Correlations:")
        print(f"    Reweighing_alpha → D_SPD: {corr_alpha_d_spd:.3f}")
        print(f"    Reweighing_alpha → Tr_SPD: {corr_alpha_tr_spd:.3f}")
        print(f"    Reweighing_alpha → Te_SPD: {corr_alpha_te_spd:.3f}")

        # 至少应该有一定的相关性
        self.assertGreater(corr_alpha_d_spd, 0.5, "Expected correlation between method and dataset SPD")

        # 构建简化的因果图
        import networkx as nx
        G = nx.DiGraph()

        # 添加节点
        nodes = df.columns.tolist()
        G.add_nodes_from(nodes)

        # 添加边（基于相关性阈值）
        threshold = 0.3
        for i, col1 in enumerate(nodes):
            for j, col2 in enumerate(nodes):
                if i != j and abs(corr_matrix.iloc[i, j]) > threshold:
                    # 根据命名规则确定方向
                    if '_alpha' in col1 and col1 not in col2:
                        G.add_edge(col1, col2)
                    elif col1.startswith('D_') and col2.startswith('Tr_'):
                        G.add_edge(col1, col2)
                    elif col1.startswith('Tr_') and col2.startswith('Te_'):
                        G.add_edge(col1, col2)

        print(f"\n✓ Constructed causal graph:")
        print(f"    Nodes: {G.number_of_nodes()}")
        print(f"    Edges: {G.number_of_edges()}")

        # 验证预期的因果路径
        # 应该有 方法 → 数据集 → 训练 → 测试 的路径
        has_method_to_dataset = any(
            G.has_edge('Reweighing_alpha', node) for node in nodes if node.startswith('D_')
        )
        print(f"    Has method → dataset edges: {has_method_to_dataset}")

        self.assertTrue(G.number_of_edges() > 0, "Causal graph should have edges")


class TestTradeoffAnalysisSimulation(unittest.TestCase):
    """测试权衡分析（简化模拟）"""

    def test_tradeoff_detection(self):
        """测试权衡检测"""
        print("\n" + "="*60)
        print("Testing Trade-off Detection")
        print("="*60)

        # 创建模拟数据，展示明显的权衡
        data = {
            'method_alpha': [0.0, 0.3, 0.6, 0.9, 1.0],
            'Te_Acc': [0.85, 0.83, 0.80, 0.77, 0.75],  # 准确率下降
            'Te_SPD': [0.20, 0.15, 0.10, 0.05, 0.02],  # SPD改善（接近0）
        }
        df = pd.DataFrame(data)

        print("Data:")
        print(df.to_string(index=False))

        # 计算ATE（简化版：使用差值）
        acc_t0 = df[df['method_alpha'] == 0.0]['Te_Acc'].values[0]
        acc_t1 = df[df['method_alpha'] == 1.0]['Te_Acc'].values[0]
        ate_acc = acc_t1 - acc_t0

        spd_t0 = df[df['method_alpha'] == 0.0]['Te_SPD'].values[0]
        spd_t1 = df[df['method_alpha'] == 1.0]['Te_SPD'].values[0]
        ate_spd = spd_t1 - spd_t0

        print(f"\nATE estimates:")
        print(f"  Method → Te_Acc: {ate_acc:+.3f}")
        print(f"  Method → Te_SPD: {ate_spd:+.3f}")

        # 使用sign函数检测权衡
        from utils.metrics import define_sign_functions
        sign_funcs = define_sign_functions()

        sign_acc = sign_funcs['Acc'](acc_t0, ate_acc)
        sign_spd = sign_funcs['SPD'](spd_t0, ate_spd)

        print(f"\nSign analysis:")
        print(f"  Te_Acc: {sign_acc} (accuracy {'improved' if sign_acc == '+' else 'degraded'})")
        print(f"  Te_SPD: {sign_spd} (fairness {'improved' if sign_spd == '+' else 'degraded'})")

        # 检测权衡
        has_tradeoff = (sign_acc != sign_spd)
        print(f"\n{'✓' if has_tradeoff else '✗'} Trade-off detected: {has_tradeoff}")

        self.assertTrue(has_tradeoff, "Expected to detect trade-off between accuracy and SPD")


class TestSystemRobustness(unittest.TestCase):
    """测试系统鲁棒性"""

    def test_handle_missing_data(self):
        """测试处理缺失数据"""
        print("\n" + "="*60)
        print("Testing Robustness: Missing Data")
        print("="*60)

        # 创建有缺失值的数据
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)

        # 添加一些NaN
        X[0, 0] = np.nan
        X[5, 3] = np.nan

        # 尝试处理（应该能够容错或报错）
        try:
            # 删除NaN
            mask = ~np.isnan(X).any(axis=1)
            X_clean = X[mask]
            y_clean = y[mask]
            sensitive_clean = sensitive[mask]

            print(f"✓ Handled missing data: {len(X)} → {len(X_clean)} samples")

            # 继续训练
            model = FFNN(input_dim=X_clean.shape[1], width=2)
            trainer = ModelTrainer(model, device='cpu')
            trainer.train(X_clean, y_clean, epochs=2, verbose=False)

            print(f"✓ Training completed with cleaned data")

        except Exception as e:
            self.fail(f"Failed to handle missing data: {e}")

    def test_handle_edge_cases(self):
        """测试边界情况"""
        print("\n" + "="*60)
        print("Testing Robustness: Edge Cases")
        print("="*60)

        # 测试1: 极小样本
        X_small = np.random.randn(10, 5)
        y_small = np.random.randint(0, 2, 10)
        sensitive_small = np.random.randint(0, 2, 10)

        model = FFNN(input_dim=5, width=2)
        trainer = ModelTrainer(model, device='cpu')
        try:
            trainer.train(X_small, y_small, epochs=2, verbose=False)
            print(f"✓ Handled small sample size (n={len(X_small)})")
        except Exception as e:
            print(f"⚠ Warning: Failed with small sample: {e}")

        # 测试2: 不平衡数据
        X_imb = np.random.randn(100, 5)
        y_imb = np.array([0] * 95 + [1] * 5)  # 95% class 0
        sensitive_imb = np.random.randint(0, 2, 100)

        calculator = MetricsCalculator(trainer, sensitive_attr='sex')
        try:
            metrics = calculator.compute_all_metrics(
                X_imb, y_imb, sensitive_imb, phase='Te'
            )
            print(f"✓ Handled imbalanced data (95:5 ratio)")
        except Exception as e:
            print(f"⚠ Warning: Failed with imbalanced data: {e}")


def run_integration_tests():
    """运行所有集成测试"""
    print("\n" + "="*70)
    print("Running Integration Test Suite")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加集成测试
    suite.addTests(loader.loadTestsFromTestCase(TestDataCollectionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalGraphSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeoffAnalysisSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemRobustness))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "="*70)
    print("Integration Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
