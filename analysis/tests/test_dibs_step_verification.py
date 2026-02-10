#!/usr/bin/env python3
"""
DiBS步数验证测试

目的: 验证DiBS训练是否真的执行了设定的步数
背景: 之前出现过callback设置导致未执行设定步数的问题

测试策略:
1. 使用最小的组（group5_mrt_oast，60样本）快速测试
2. 设置n_steps=1000（较小步数）
3. 通过callback机制监控实际执行的步数
4. 验证实际步数是否匹配设定步数

创建日期: 2026-02-10
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.causal_discovery import CausalGraphLearner


class DiBSStepMonitor:
    """DiBS训练步数监控器"""

    def __init__(self, expected_steps: int, verbose: bool = True):
        self.expected_steps = expected_steps
        self.verbose = verbose
        self.actual_steps = 0
        self.step_times = []
        self.start_time = None

    def callback(self, t: int = None, dibs=None, zs=None, **kwargs):
        """
        DiBS callback函数

        DiBS会在每个callback_every间隔调用此函数
        参数（通过关键字参数传递）:
            t: 当前步数（DiBS使用't'参数名，不是'step'）
            dibs: DiBS实例
            zs: 当前粒子状态
            kwargs: 其他参数（忽略）
        """
        if self.start_time is None:
            self.start_time = time.time()

        # DiBS通过关键字参数't'传递步数
        step = t if t is not None else kwargs.get('t', 0)
        self.actual_steps = step
        self.step_times.append(time.time() - self.start_time)

        if self.verbose and step % 100 == 0:
            elapsed = time.time() - self.start_time
            print(f"  步数监控: {step}/{self.expected_steps} "
                  f"({step/self.expected_steps*100:.1f}%) "
                  f"- 已用时间: {elapsed:.1f}s")

    def verify(self) -> dict:
        """验证实际步数是否符合预期"""
        match = self.actual_steps == self.expected_steps
        discrepancy = abs(self.actual_steps - self.expected_steps)

        result = {
            'expected_steps': self.expected_steps,
            'actual_steps': self.actual_steps,
            'match': match,
            'discrepancy': discrepancy,
            'total_time': self.step_times[-1] if self.step_times else 0,
            'avg_time_per_step': np.mean(np.diff(self.step_times)) if len(self.step_times) > 1 else 0
        }

        return result


def load_test_group():
    """加载测试数据组（最小的组）"""
    data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "6groups_global_std"
    csv_file = data_dir / "group5_mrt_oast_global_std.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {csv_file}")

    df = pd.read_csv(csv_file)

    # 过滤非数值列（timestamp等）
    numeric_df = df.select_dtypes(include=['number'])
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    print(f"\n加载数据: {csv_file}")
    print(f"  样本数: {len(df)}")
    print(f"  原始列数: {len(df.columns)}")
    if non_numeric_cols:
        print(f"  已过滤非数值列: {non_numeric_cols}")
    print(f"  数值特征数: {len(numeric_df.columns)}")

    return numeric_df


def test_dibs_with_callback(n_steps: int = 1000,
                            n_particles: int = 20,
                            callback_every: int = 10,
                            verbose: bool = True):
    """
    测试DiBS是否执行了设定步数（使用callback监控）

    参数:
        n_steps: 训练步数
        n_particles: 粒子数
        callback_every: callback调用间隔
        verbose: 是否输出详细信息
    """
    print("="*80)
    print("DiBS步数验证测试（使用callback监控）")
    print("="*80)

    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    data_df = load_test_group()
    n_vars = len(data_df.columns)

    # 2. 创建监控器
    print(f"\n2. 创建步数监控器...")
    monitor = DiBSStepMonitor(expected_steps=n_steps, verbose=verbose)
    print(f"  预期步数: {n_steps}")
    print(f"  callback间隔: {callback_every}")

    # 3. 初始化DiBS学习器（使用较小参数以加快测试）
    print(f"\n3. 初始化DiBS学习器...")
    config = {
        'n_vars': n_vars,
        'n_steps': n_steps,
        'n_particles': n_particles,
        'alpha': 0.5,
        'beta': 1.0,
        'tau': 1.0,
        'variant': 'MarginalDiBS',
        'n_grad_mc_samples': 64,  # 减少以加快测试
        'n_acyclicity_mc_samples': 16  # 减少以加快测试
    }

    print(f"  配置参数:")
    for key, value in config.items():
        print(f"    {key}: {value}")

    learner = CausalGraphLearner(**config)

    # 4. 运行DiBS（需要修改以支持callback）
    print(f"\n4. 运行DiBS训练（预期{n_steps}步）...")
    start_time = time.time()

    try:
        # 修改fit方法以支持callback
        # 注意：这需要临时修改learner.model.sample的调用
        graph = learner.fit_with_callback(
            data=data_df,
            callback=monitor.callback,
            callback_every=callback_every,
            verbose=verbose
        )
    except AttributeError:
        # 如果fit_with_callback不存在，使用标准fit（无法监控）
        print("\n⚠️ 警告: CausalGraphLearner不支持fit_with_callback")
        print("  使用标准fit方法（无法监控步数）")
        graph = learner.fit(data=data_df, verbose=verbose)
        print("\n❌ 测试失败: 无法监控步数执行")
        return None

    elapsed_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {elapsed_time:.1f}秒")

    # 5. 验证步数
    print(f"\n5. 验证步数执行...")
    result = monitor.verify()

    print(f"\n验证结果:")
    print(f"  预期步数: {result['expected_steps']}")
    print(f"  实际步数: {result['actual_steps']}")
    print(f"  匹配状态: {'✅ 通过' if result['match'] else '❌ 失败'}")
    print(f"  步数差异: {result['discrepancy']}")
    print(f"  总时间: {result['total_time']:.1f}秒")
    print(f"  平均每步时间: {result['avg_time_per_step']*1000:.2f}ms")

    # 6. 输出图统计
    print(f"\n6. 因果图统计:")
    n_edges = np.sum(graph > 0.3)
    density = n_edges / (n_vars * (n_vars - 1))
    print(f"  强边数 (>0.3): {n_edges}")
    print(f"  图密度: {density:.3f}")

    return result


def test_dibs_without_callback(n_steps: int = 1000, n_particles: int = 20):
    """
    测试DiBS在无callback情况下的执行（基准测试）

    目的: 对比有无callback的训练时间差异
    """
    print("\n" + "="*80)
    print("DiBS步数验证测试（无callback，基准测试）")
    print("="*80)

    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    data_df = load_test_group()
    n_vars = len(data_df.columns)

    # 2. 初始化DiBS学习器
    print(f"\n2. 初始化DiBS学习器...")
    config = {
        'n_vars': n_vars,
        'n_steps': n_steps,
        'n_particles': n_particles,
        'alpha': 0.5,
        'beta': 1.0,
        'tau': 1.0,
        'variant': 'MarginalDiBS',
        'n_grad_mc_samples': 64,
        'n_acyclicity_mc_samples': 16
    }

    learner = CausalGraphLearner(**config)

    # 3. 运行DiBS（无callback）
    print(f"\n3. 运行DiBS训练（无callback监控）...")
    start_time = time.time()
    graph = learner.fit(data=data_df, verbose=True)
    elapsed_time = time.time() - start_time

    print(f"\n训练完成！总耗时: {elapsed_time:.1f}秒")

    # 4. 输出图统计
    print(f"\n4. 因果图统计:")
    n_edges = np.sum(graph > 0.3)
    density = n_edges / (n_vars * (n_vars - 1))
    print(f"  强边数 (>0.3): {n_edges}")
    print(f"  图密度: {density:.3f}")

    return {
        'elapsed_time': elapsed_time,
        'n_edges': n_edges,
        'density': density
    }


def main():
    parser = argparse.ArgumentParser(description='DiBS步数验证测试')
    parser.add_argument('--steps', type=int, default=1000,
                        help='训练步数（默认1000）')
    parser.add_argument('--particles', type=int, default=20,
                        help='粒子数（默认20）')
    parser.add_argument('--callback-every', type=int, default=10,
                        help='callback间隔（默认10）')
    parser.add_argument('--no-callback', action='store_true',
                        help='不使用callback（基准测试）')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.no_callback:
        # 基准测试（无callback）
        result = test_dibs_without_callback(
            n_steps=args.steps,
            n_particles=args.particles
        )
    else:
        # 步数监控测试（有callback）
        result = test_dibs_with_callback(
            n_steps=args.steps,
            n_particles=args.particles,
            callback_every=args.callback_every,
            verbose=verbose
        )

        if result and result['match']:
            print("\n" + "="*80)
            print("✅ 测试通过: DiBS正确执行了设定的步数")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("❌ 测试失败: DiBS未执行设定的步数")
            print("="*80)
            sys.exit(1)


if __name__ == "__main__":
    main()
