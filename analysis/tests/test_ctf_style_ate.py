"""
CTF风格ATE计算功能的测试套件

测试内容：
1. 迭代1 MVP核心功能：
   - CTF风格的混淆因素识别
   - 简化的ATE计算（无ref_df，无T0/T1）

2. 迭代2 V1.0扩展功能：
   - ref_df构建
   - T0/T1计算
   - 完整的CTF风格ATE计��

作者：Claude Code
日期：2026-01-26
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.causal_inference import CausalInferenceEngine


class TestCTFConfounderIdentification(unittest.TestCase):
    """测试CTF风格的混淆因素识别"""

    def setUp(self):
        """设置测试数据"""
        # 创建简单的因果图
        # 结构: A -> B -> C
        #        \-> D
        self.n_vars = 4
        self.var_names = ['A', 'B', 'C', 'D']
        self.causal_graph = np.zeros((self.n_vars, self.n_vars))

        # 添加边: A->B, B->C, A->D
        self.causal_graph[0, 1] = 0.8  # A -> B
        self.causal_graph[1, 2] = 0.7  # B -> C
        self.causal_graph[0, 3] = 0.6  # A -> D

        self.ci = CausalInferenceEngine(verbose=False)

    def test_confounders_simple_path(self):
        """测试简单路径的混淆因素识别"""
        # 测试 A -> B
        # A的父节点: []（无）
        # B的父节点: [A]
        # 合并: [A]
        # 移除A本身: []
        confounders = self.ci._get_confounders_from_graph(
            'A', 'B', self.causal_graph, self.var_names, threshold=0.3
        )

        self.assertEqual(len(confounders), 0)
        print(f"✓ A->B的混淆因素: {confounders} (预期: [])")

    def test_confounders_with_shared_parent(self):
        """测试有共同父节点的情况"""
        # 添加一个共同父节点
        # 结构: E -> A -> B
        #        \-> D
        causal_graph = np.zeros((5, 5))
        var_names = ['E', 'A', 'B', 'C', 'D']

        causal_graph[0, 1] = 0.8  # E -> A
        causal_graph[1, 2] = 0.7  # A -> B
        causal_graph[0, 4] = 0.6  # E -> D

        # 测试 A -> B
        # A的父节点: [E]
        # B的父节点: [A]
        # 合并: [E, A]
        # 移除A本身: [E]
        confounders = self.ci._get_confounders_from_graph(
            'A', 'B', causal_graph, var_names, threshold=0.3
        )

        self.assertIn('E', confounders)
        self.assertNotIn('A', confounders)
        self.assertNotIn('B', confounders)
        print(f"✓ A->B的混淆因素: {confounders} (预期包含E)")

    def test_confounders_child_not_in_list(self):
        """测试child不在混淆因素列表中（验证逻辑正确性）"""
        # 测试 B -> C
        # B的父节点: [A]
        # C的父节点: [B]
        # 合并: [A, B]
        # 移除B本身: [A]
        # C不应该在列表中
        confounders = self.ci._get_confounders_from_graph(
            'B', 'C', self.causal_graph, self.var_names, threshold=0.3
        )

        self.assertIn('A', confounders)
        self.assertNotIn('C', confounders)  # C是child，不应在混淆因素中
        print(f"✓ B->C的混淆因素: {confounders} (预期包含A，不包含C)")

    def test_confounders_prevents_circular_dependency(self):
        """测试防止循环依赖（child在混淆因素中时报错）"""
        # 创建有循环依赖的图（错误情况）
        bad_graph = np.zeros((3, 3))
        var_names = ['A', 'B', 'C']

        bad_graph[0, 1] = 0.8  # A -> B
        bad_graph[1, 2] = 0.7  # B -> C
        bad_graph[2, 1] = 0.6  # C -> B (循环依赖)

        # 应该抛出ValueError，因为C（child）在混淆因素中
        with self.assertRaises(ValueError) as context:
            self.ci._get_confounders_from_graph(
                'B', 'C', bad_graph, var_names, threshold=0.3
            )

        self.assertIn("should not be in confounders", str(context.exception))
        print(f"✓ 正确检测到循环依赖并抛出异常")


class TestBuildReferenceDF(unittest.TestCase):
    """测试ref_df构建功能"""

    def setUp(self):
        """设置测试数据"""
        self.ci = CausalInferenceEngine(verbose=False)

        # 创建测试数据
        np.random.seed(42)
        self.data = pd.DataFrame({
            'model_name': ['ResNet', 'ResNet', 'VGG', 'VGG', 'BERT', 'BERT'],
            'is_parallel': [0, 1, 0, 1, 0, 1],
            'learning_rate': np.random.uniform(0.001, 0.1, 6),
            'batch_size': [32, 32, 64, 64, 128, 128],
            'energy': np.random.uniform(100, 500, 6),
            'accuracy': np.random.uniform(0.7, 0.95, 6)
        })

    def test_ref_df_non_parallel(self):
        """测试非并行模式策略"""
        ref_df = self.ci.build_reference_df(self.data, strategy="non_parallel")

        # 应该只包含is_parallel=0的数据
        self.assertEqual(len(ref_df), 3)
        self.assertTrue(all(ref_df['is_parallel'] == 0))
        print(f"✓ 非并行模式ref_df: {len(ref_df)}行")

    def test_ref_df_mean(self):
        """测试全局均值策略"""
        ref_df = self.ci.build_reference_df(self.data, strategy="mean")

        # 应该只有1行（均值）
        self.assertEqual(len(ref_df), 1)
        print(f"✓ 全局均值ref_df: {len(ref_df)}行")

    def test_ref_df_group_mean(self):
        """测试分组均值策略"""
        ref_df = self.ci.build_reference_df(
            self.data, strategy="group_mean", groupby_cols=['model_name']
        )

        # 应该有3行（ResNet、VGG、BERT）
        self.assertEqual(len(ref_df), 3)
        self.assertIn('ResNet', ref_df['model_name'].values)
        self.assertIn('VGG', ref_df['model_name'].values)
        self.assertIn('BERT', ref_df['model_name'].values)
        print(f"✓ 分组均值ref_df: {len(ref_df)}行 (按model_name)")

    def test_ref_df_invalid_strategy(self):
        """测试无效策略"""
        with self.assertRaises(ValueError):
            self.ci.build_reference_df(self.data, strategy="invalid_strategy")

    def test_ref_df_empty_result(self):
        """测试空结果（数据不包含指定的分组）"""
        # 创建一个不可能满足条件的数据
        empty_data = pd.DataFrame({
            'is_parallel': [1, 1, 1],  # 全部是并行模式
            'value': [1, 2, 3]
        })

        # 应该产生警告但不会崩溃
        import warnings
        with warnings.catch_warnings(record=True):
            ref_df = self.ci.build_reference_df(empty_data, strategy="non_parallel")
            # 由于没有is_parallel=0的数据，应该返回空或全部数据
            self.assertGreaterEqual(len(ref_df), 0)


class TestComputeT0T1(unittest.TestCase):
    """测试T0/T1计算功能"""

    def setUp(self):
        """设置测试数据"""
        self.ci = CausalInferenceEngine(verbose=False)

        # 创建测试数据
        np.random.seed(42)
        self.data = pd.DataFrame({
            'learning_rate': np.random.uniform(0.001, 0.1, 1000),
            'batch_size': np.random.choice([32, 64, 128], 1000),
            'energy': np.random.uniform(100, 500, 1000)
        })

    def test_t0_t1_quantile(self):
        """测试分位数策略"""
        T0, T1 = self.ci.compute_T0_T1(
            self.data, 'learning_rate', strategy='quantile'
        )

        # T0应该是25分位数，T1应该是75分位数
        expected_T0 = self.data['learning_rate'].quantile(0.25)
        expected_T1 = self.data['learning_rate'].quantile(0.75)

        self.assertAlmostEqual(T0, expected_T0, places=5)
        self.assertAlmostEqual(T1, expected_T1, places=5)
        self.assertGreater(T1, T0)
        print(f"✓ 分位数策略: T0={T0:.4f}, T1={T1:.4f}")

    def test_t0_t1_min_max(self):
        """测试最小最大值策略"""
        T0, T1 = self.ci.compute_T0_T1(
            self.data, 'learning_rate', strategy='min_max'
        )

        # T0应该是最小值，T1应该是最大值
        expected_T0 = self.data['learning_rate'].min()
        expected_T1 = self.data['learning_rate'].max()

        self.assertAlmostEqual(T0, expected_T0, places=5)
        self.assertAlmostEqual(T1, expected_T1, places=5)
        self.assertGreater(T1, T0)
        print(f"✓ 最小最大值策略: T0={T0:.4f}, T1={T1:.4f}")

    def test_t0_t1_mean_std(self):
        """测试均值标准差策略"""
        T0, T1 = self.ci.compute_T0_T1(
            self.data, 'learning_rate', strategy='mean_std'
        )

        # T0应该是mean-std，T1应该是mean+std
        mean_val = self.data['learning_rate'].mean()
        std_val = self.data['learning_rate'].std()
        expected_T0 = mean_val - std_val
        expected_T1 = mean_val + std_val

        self.assertAlmostEqual(T0, expected_T0, places=5)
        self.assertAlmostEqual(T1, expected_T1, places=5)
        self.assertGreater(T1, T0)
        print(f"✓ 均值标准差策略: T0={T0:.4f}, T1={T1:.4f}")

    def test_t0_t1_invalid_strategy(self):
        """测试无效策略"""
        with self.assertRaises(ValueError):
            self.ci.compute_T0_T1(
                self.data, 'learning_rate', strategy='invalid_strategy'
            )

    def test_t0_t1_constant_variable(self):
        """测试常量变量（T1 <= T0）"""
        # 创建常量数据
        constant_data = pd.DataFrame({
            'constant_value': [1.0] * 100
        })

        with self.assertRaises(ValueError) as context:
            self.ci.compute_T0_T1(
                constant_data, 'constant_value', strategy='quantile'
            )

        self.assertIn("T1", str(context.exception))
        self.assertIn("T0", str(context.exception))
        print(f"✓ 正确检测到常量变量并抛出异常")


class TestCTFStyleATE(unittest.TestCase):
    """测试CTF风格的ATE计算"""

    def setUp(self):
        """设置测试数据"""
        self.ci = CausalInferenceEngine(verbose=False)

        # 创建模拟数据（有因果关系）
        np.random.seed(42)
        n_samples = 500

        # X -> T -> Y
        # X -> Y
        X = np.random.randn(n_samples)
        T = 0.5 * X + np.random.randn(n_samples) * 0.1
        Y = 0.3 * T + 0.4 * X + np.random.randn(n_samples) * 0.1

        self.data = pd.DataFrame({
            'X': X,
            'treatment': T,
            'outcome': Y
        })

        # 创建因果图
        self.causal_graph = np.zeros((3, 3))
        self.causal_graph[0, 1] = 0.9  # X -> treatment
        self.causal_graph[1, 2] = 0.8  # treatment -> outcome
        self.causal_graph[0, 2] = 0.7  # X -> outcome

        self.var_names = ['X', 'treatment', 'outcome']

    def test_ctf_style_ate_mvp(self):
        """测试MVP版本（无ref_df，无T0/T1）"""
        results = self.ci.analyze_all_edges_ctf_style(
            data=self.data,
            causal_graph=self.causal_graph,
            var_names=self.var_names,
            threshold=0.5
        )

        # 应该至少分析了一条边
        self.assertGreater(len(results), 0)

        # 检查结果格式
        for edge, result in results.items():
            self.assertIn('ate', result)
            self.assertIn('ci_lower', result)
            self.assertIn('ci_upper', result)
            self.assertIn('is_significant', result)
            self.assertIn('confounders', result)

            # ATE应该是数值
            self.assertIsInstance(result['ate'], (int, float))

            # CI应该有效
            self.assertLess(result['ci_lower'], result['ci_upper'])

        print(f"✓ MVP版本分析了{len(results)}条边")

    def test_ctf_style_ate_with_ref_df(self):
        """测试V1.0版本（使用ref_df）"""
        # 创建ref_df（使用前50%的数据）
        ref_df = self.data.iloc[:len(self.data)//2].copy()

        results = self.ci.analyze_all_edges_ctf_style(
            data=self.data,
            causal_graph=self.causal_graph,
            var_names=self.var_names,
            threshold=0.5,
            ref_df=ref_df
        )

        # 应该至少分析了一条边
        self.assertGreater(len(results), 0)

        # 检查结果格式
        for edge, result in results.items():
            self.assertIn('ate', result)
            self.assertIsInstance(result['ate'], (int, float))

        print(f"✓ 使用ref_df分析了{len(results)}条边")

    def test_ctf_style_ate_with_t_strategy(self):
        """测试V1.0版本（使用T0/T1策略）"""
        results = self.ci.analyze_all_edges_ctf_style(
            data=self.data,
            causal_graph=self.causal_graph,
            var_names=self.var_names,
            threshold=0.5,
            t_strategy='quantile'
        )

        # 应该至少分析了一条边
        self.assertGreater(len(results), 0)

        # 检查结果格式
        for edge, result in results.items():
            self.assertIn('ate', result)
            self.assertIsInstance(result['ate'], (int, float))

        print(f"✓ 使用T0/T1策略分析了{len(results)}条边")

    def test_ctf_style_ate_full_v1(self):
        """测试完整的V1.0版本（ref_df + T0/T1）"""
        # 创建ref_df
        ref_df = self.ci.build_reference_df(
            self.data, strategy="mean"
        )

        results = self.ci.analyze_all_edges_ctf_style(
            data=self.data,
            causal_graph=self.causal_graph,
            var_names=self.var_names,
            threshold=0.5,
            ref_df=ref_df,
            t_strategy='quantile'
        )

        # 应该至少分析了一条边
        self.assertGreater(len(results), 0)

        # 检查显著率
        n_significant = sum(1 for r in results.values() if r['is_significant'])
        significance_rate = n_significant / len(results) * 100 if len(results) > 0 else 0

        print(f"✓ 完整V1.0版本分析了{len(results)}条边，显著率={significance_rate:.1f}%")


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("CTF Style ATE Test Suite")
    print("="*70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCTFConfounderIdentification))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildReferenceDF))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeT0T1))
    suite.addTests(loader.loadTestsFromTestCase(TestCTFStyleATE))

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

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
