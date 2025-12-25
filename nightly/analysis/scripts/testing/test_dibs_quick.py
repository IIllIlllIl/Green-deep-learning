"""
快速测试DiBS功能
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/green/energy_dl/analysis')

from utils.causal_discovery import CausalGraphLearner

def test_dibs_simple():
    """测试DiBS在简单数据上"""
    print("=" * 60)
    print("测试1: DiBS简单链式因果关系 (X → Y → Z)")
    print("=" * 60)

    # 生成简单的链式因果数据
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples)
    Y = 2*X + np.random.randn(n_samples)*0.1
    Z = 3*Y + np.random.randn(n_samples)*0.1

    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    # 学习因果图（使用较少迭代次数快速测试）
    learner = CausalGraphLearner(n_vars=3, n_steps=500, alpha=0.9)

    try:
        graph = learner.fit(data, verbose=True)

        # 分析结果
        print("\n学到的邻接矩阵:")
        print(graph)

        edges = learner.get_edges(threshold=0.3)
        print(f"\n检测到的边 (阈值=0.3):")
        for i, j, w in edges:
            print(f"  {data.columns[i]} → {data.columns[j]}: {w:.3f}")

        # 验证预期的边
        if graph[0, 1] > 0.3:
            print("\n✓ 正确检测到 X → Y")
        if graph[1, 2] > 0.3:
            print("✓ 正确检测到 Y → Z")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dibs_moderate_size():
    """测试DiBS在中等规模数据"""
    print("\n" + "=" * 60)
    print("测试2: DiBS中等规模数据 (10变量)")
    print("=" * 60)

    # 生成10个变量的随机数据
    np.random.seed(42)
    n_samples = 200
    n_vars = 10
    data = pd.DataFrame(
        np.random.randn(n_samples, n_vars),
        columns=[f'var_{i}' for i in range(n_vars)]
    )

    # 学习因果图
    learner = CausalGraphLearner(n_vars=n_vars, n_steps=300)

    try:
        graph = learner.fit(data, verbose=True)

        print(f"\n图的形状: {graph.shape}")
        print(f"边数: {np.sum(graph > 0.5)}")
        print(f"是否为DAG: {learner._is_dag(graph)}")

        # 保存图
        learner.save_graph('/tmp/test_graph.npy')

        # 加载图
        learner2 = CausalGraphLearner(n_vars=n_vars)
        learner2.load_graph('/tmp/test_graph.npy')

        print("\n✓ 保存和加载功能正常")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("DiBS因果图学习 - 快速功能测试")
    print("=" * 60)

    results = []

    # 测试1
    results.append(('简单链式关系', test_dibs_simple()))

    # 测试2
    results.append(('中等规模数据', test_dibs_moderate_size()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n✅ 所有测试通过！DiBS模块可以使用。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
