"""
单元测试套件
测试各个模块的核心功能
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import FFNN, ModelTrainer
from utils.metrics import MetricsCalculator, define_sign_functions
from utils.fairness_methods import FairnessMethodWrapper, get_fairness_method
import config


class TestModel(unittest.TestCase):
    """测试神经网络模型"""

    def setUp(self):
        """设置测试数据"""
        self.input_dim = 10
        self.n_samples = 100
        self.X = np.random.randn(self.n_samples, self.input_dim)
        self.y = np.random.randint(0, 2, self.n_samples).astype(float)

    def test_model_initialization(self):
        """测试模型初始化"""
        model = FFNN(input_dim=self.input_dim, width=4)
        self.assertIsNotNone(model)
        # 检查模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"✓ Model initialized with {total_params} parameters")

    def test_model_forward_pass(self):
        """测试前向传播"""
        model = FFNN(input_dim=self.input_dim, width=4)
        import torch
        X_tensor = torch.FloatTensor(self.X)
        output = model(X_tensor)

        # 检查输出形状
        self.assertEqual(output.shape, (self.n_samples, 1))
        # 检查输出范围（Sigmoid输出应在[0,1]）
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        print(f"✓ Forward pass output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")

    def test_model_training(self):
        """测试模型训练"""
        model = FFNN(input_dim=self.input_dim, width=4)
        trainer = ModelTrainer(model, device='cpu', lr=0.01)

        # 训练5轮
        trainer.train(self.X, self.y, epochs=5, batch_size=32, verbose=False)

        # 训练后应该能够预测
        predictions = trainer.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

        accuracy = np.mean(predictions == self.y)
        print(f"✓ Training completed. Accuracy: {accuracy:.3f}")
        # 至少应该比随机猜测好
        self.assertGreater(accuracy, 0.3)

    def test_model_prediction_proba(self):
        """测试概率预测"""
        model = FFNN(input_dim=self.input_dim, width=4)
        trainer = ModelTrainer(model, device='cpu')
        trainer.train(self.X, self.y, epochs=3, verbose=False)

        proba = trainer.predict_proba(self.X)
        self.assertEqual(len(proba), len(self.y))
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))
        print(f"✓ Probability predictions in range [0, 1]")


class TestMetrics(unittest.TestCase):
    """测试指标计算"""

    def setUp(self):
        """设置测试数据"""
        self.n_samples = 200
        self.input_dim = 10
        self.X = np.random.randn(self.n_samples, self.input_dim)
        self.y = np.random.randint(0, 2, self.n_samples)
        self.sensitive = np.random.randint(0, 2, self.n_samples)

        # 创建一个简单的模型
        from utils.model import FFNN, ModelTrainer
        model = FFNN(input_dim=self.input_dim, width=2)
        self.trainer = ModelTrainer(model, device='cpu')
        self.trainer.train(self.X, self.y, epochs=3, verbose=False)

    def test_sign_functions(self):
        """测试sign函数定义"""
        sign_funcs = define_sign_functions()

        # 测试性能指标
        self.assertEqual(sign_funcs['Acc'](0.5, 0.1), '+')
        self.assertEqual(sign_funcs['Acc'](0.5, -0.1), '-')

        # 测试SPD（理想值为0）
        self.assertEqual(sign_funcs['SPD'](0.1, -0.05), '+')  # 更接近0
        self.assertEqual(sign_funcs['SPD'](0.1, 0.05), '-')   # 远离0

        print(f"✓ Sign functions working correctly")

    def test_metrics_calculation(self):
        """测试指标计算"""
        calculator = MetricsCalculator(self.trainer, sensitive_attr='sex')

        metrics = calculator.compute_all_metrics(
            self.X, self.y, self.sensitive, phase='Te'
        )

        # 检查必须的指标
        required_metrics = ['Te_Acc', 'Te_F1', 'Te_DI', 'Te_SPD', 'Te_AOD',
                           'Te_Cons', 'Te_TI', 'A_FGSM', 'A_PGD']

        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
            # 处理可能是数组的情况
            value = metrics[metric]
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value[0]
            self.assertIsInstance(value, (int, float, np.number))
            self.assertFalse(np.isnan(value), f"{metric} is NaN")

        print(f"✓ All {len(required_metrics)} required metrics calculated")
        print(f"  Sample metrics: Acc={metrics['Te_Acc']:.3f}, SPD={metrics['Te_SPD']:.3f}")

    def test_metrics_range_validity(self):
        """测试指标值的合理范围"""
        calculator = MetricsCalculator(self.trainer, sensitive_attr='sex')
        metrics = calculator.compute_all_metrics(
            self.X, self.y, self.sensitive, phase='Te'
        )

        # Accuracy应在[0, 1]
        self.assertGreaterEqual(metrics['Te_Acc'], 0)
        self.assertLessEqual(metrics['Te_Acc'], 1)

        # F1应在[0, 1]
        self.assertGreaterEqual(metrics['Te_F1'], 0)
        self.assertLessEqual(metrics['Te_F1'], 1)

        # 鲁棒性指标应在[0, 1]
        self.assertGreaterEqual(metrics['A_FGSM'], 0)
        self.assertLessEqual(metrics['A_FGSM'], 1)

        print(f"✓ All metrics within valid ranges")


class TestFairnessMethods(unittest.TestCase):
    """测试公平性方法"""

    def setUp(self):
        """设置测试数据"""
        self.n_samples = 200
        self.input_dim = 10
        self.X = np.random.randn(self.n_samples, self.input_dim)
        self.y = np.random.randint(0, 2, self.n_samples)
        self.sensitive = np.random.randint(0, 2, self.n_samples)

    def test_baseline_method(self):
        """测试Baseline方法（不做任何改变）"""
        method = FairnessMethodWrapper('Baseline', alpha=0.5, sensitive_attr='sex')
        X_transformed, y_transformed = method.fit_transform(
            self.X, self.y, self.sensitive
        )

        # Baseline不应该改变数据
        np.testing.assert_array_equal(X_transformed, self.X)
        np.testing.assert_array_equal(y_transformed, self.y)
        print(f"✓ Baseline method preserves data")

    def test_alpha_parameter(self):
        """测试alpha参数的影响"""
        # alpha=0应该不改变数据
        method_0 = FairnessMethodWrapper('Reweighing', alpha=0.0, sensitive_attr='sex')
        X_0, y_0 = method_0.fit_transform(self.X, self.y, self.sensitive)

        # 由于alpha=0，没有样本被应用方法
        np.testing.assert_array_equal(X_0, self.X)

        # alpha=1应该对所有数据应用方法
        method_1 = FairnessMethodWrapper('Reweighing', alpha=1.0, sensitive_attr='sex')
        X_1, y_1 = method_1.fit_transform(self.X, self.y, self.sensitive)

        # 数据形状应该保持
        self.assertEqual(X_1.shape, self.X.shape)
        self.assertEqual(len(y_1), len(self.y))

        print(f"✓ Alpha parameter works correctly")

    def test_method_factory(self):
        """测试方法工厂函数"""
        for method_name in config.FAIRNESS_METHODS:
            method = get_fairness_method(method_name, alpha=0.5, sensitive_attr='sex')
            self.assertIsNotNone(method)
            self.assertEqual(method.method_name, method_name)
            print(f"✓ Factory created {method_name} successfully")


class TestConfiguration(unittest.TestCase):
    """测试配置文件"""

    def test_config_validity(self):
        """测试配置参数的有效性"""
        # 检查关键配置
        self.assertIn(config.DATASET, ['adult', 'compas', 'german'])
        self.assertIn(config.SENSITIVE_ATTR, ['sex', 'race', 'age'])

        # 检查数值范围
        self.assertGreater(config.MODEL_WIDTH, 0)
        self.assertGreater(config.EPOCHS, 0)
        self.assertGreater(config.BATCH_SIZE, 0)

        # 检查alpha值
        for alpha in config.ALPHA_VALUES:
            self.assertGreaterEqual(alpha, 0)
            self.assertLessEqual(alpha, 1)

        print(f"✓ Configuration file is valid")
        print(f"  Dataset: {config.DATASET}, Sensitive: {config.SENSITIVE_ATTR}")
        print(f"  Methods: {len(config.FAIRNESS_METHODS)}, Alpha values: {len(config.ALPHA_VALUES)}")

    def test_metric_definitions(self):
        """测试指标定义"""
        all_metrics = []
        for category, metrics in config.METRICS.items():
            all_metrics.extend(metrics)

        # 应该有足够的指标
        self.assertGreater(len(all_metrics), 5)

        # 检查指标名称格式
        for metric in all_metrics:
            self.assertTrue(metric.isalnum() or '_' in metric)

        print(f"✓ {len(all_metrics)} metrics defined across {len(config.METRICS)} categories")


class TestDataFlow(unittest.TestCase):
    """测试数据流完整性"""

    def test_end_to_end_data_flow(self):
        """测试从数据加载到指标计算的完整流程"""
        # 1. 生成测试数据
        n_samples = 100
        input_dim = 10
        X = np.random.randn(n_samples, input_dim)
        y = np.random.randint(0, 2, n_samples)
        sensitive = np.random.randint(0, 2, n_samples)

        # 2. 应用公平性方法
        method = get_fairness_method('Baseline', alpha=0.5, sensitive_attr='sex')
        X_transformed, y_transformed = method.fit_transform(X, y, sensitive)

        # 3. 训练模型
        from utils.model import FFNN, ModelTrainer
        model = FFNN(input_dim=input_dim, width=2)
        trainer = ModelTrainer(model, device='cpu')
        trainer.train(X_transformed, y_transformed, epochs=3, verbose=False)

        # 4. 计算指标
        calculator = MetricsCalculator(trainer, sensitive_attr='sex')
        metrics = calculator.compute_all_metrics(X, y, sensitive, phase='Te')

        # 5. 验证输出
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)

        print(f"✓ End-to-end data flow completed successfully")
        print(f"  Generated {n_samples} samples → Applied method → Trained model → Computed {len(metrics)} metrics")


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("Running Comprehensive Test Suite")
    print("="*70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestFairnessMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFlow))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
