#!/usr/bin/env python3
"""
训练时间估算工具
根据数据集大小、模型类型和训练参数估算所需训练时间
"""

import argparse
from typing import Dict, Tuple

# 数据集样本数量
DATASET_SIZES = {
    'd2a': {'train': 4643, 'val': 596},
    'reveal': {'train': 18187, 'val': 2273},
    'devign': {'train': 21854, 'val': 2732},
    'vuldeepecker': {'train': 128118, 'val': 16015},
    'mvd': {'train': 123515, 'val': 21797},
    'draper': {'train': 1019471, 'val': 127476},
}

# 默认超参数
MLP_DEFAULTS = {
    'batch_size': 2,  # 推荐值，避免OOM
    'epochs': 10,
    'learning_rate': 3e-05,
    'time_per_step': 0.7,  # 秒/步 (中等估算)
}

CNN_DEFAULTS = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.0005,
    'time_per_step': 0.4,  # 秒/步 (中等估算)
}

def format_time(seconds: float) -> str:
    """将秒数格式化为易读的时间字符串"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{days:.1f}天 ({hours:.1f}小时)"

def estimate_training_time(
    dataset: str,
    model_type: str,
    batch_size: int = None,
    epochs: int = None,
    time_per_step: float = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    估算训练时间

    参数:
        dataset: 数据集名称
        model_type: 'mlp' 或 'cnn'
        batch_size: batch大小 (None使用默认值)
        epochs: 训练轮数 (None使用默认值)
        time_per_step: 每步时间(秒) (None使用默认值)
        verbose: 是否打印详细信息

    返回:
        包含训练时间估算的字典
    """
    if dataset not in DATASET_SIZES:
        raise ValueError(f"未知数据集: {dataset}. 可选: {list(DATASET_SIZES.keys())}")

    # 获取默认参数
    defaults = MLP_DEFAULTS if model_type == 'mlp' else CNN_DEFAULTS

    # 使用提供的参数或默认值
    batch_size = batch_size or defaults['batch_size']
    epochs = epochs or defaults['epochs']
    time_per_step = time_per_step or defaults['time_per_step']

    # 计算训练步数
    train_samples = DATASET_SIZES[dataset]['train']
    val_samples = DATASET_SIZES[dataset]['val']

    steps_per_epoch = train_samples // batch_size
    if train_samples % batch_size != 0:
        steps_per_epoch += 1

    total_train_steps = steps_per_epoch * epochs

    # 估算时间
    train_time = total_train_steps * time_per_step

    # 评估时间 (假设评估速度是训练的2倍快)
    val_steps = val_samples // batch_size + (1 if val_samples % batch_size else 0)
    eval_time_per_epoch = val_steps * (time_per_step / 2)
    total_eval_time = eval_time_per_epoch * epochs

    # 总时间 (包含5%开销用于模型保存、日志等)
    total_time = (train_time + total_eval_time) * 1.05

    # 保守和乐观估算
    conservative_time = total_time * 1.2  # +20%
    optimistic_time = total_time * 0.85   # -15%

    results = {
        'train_samples': train_samples,
        'val_samples': val_samples,
        'batch_size': batch_size,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_train_steps,
        'time_per_step': time_per_step,
        'train_time_seconds': train_time,
        'eval_time_seconds': total_eval_time,
        'total_time_seconds': total_time,
        'conservative_seconds': conservative_time,
        'optimistic_seconds': optimistic_time,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"训练时间估算: {model_type.upper()} on {dataset.upper()}")
        print(f"{'='*70}")
        print(f"\n数据集信息:")
        print(f"  训练样本: {train_samples:,}")
        print(f"  验证样本: {val_samples:,}")
        print(f"\n训练配置:")
        print(f"  模型类型: {model_type.upper()}")
        print(f"  Batch大小: {batch_size}")
        print(f"  训练轮数: {epochs}")
        print(f"  每步时间: {time_per_step:.2f}秒")
        print(f"\n训练步数:")
        print(f"  每轮步数: {steps_per_epoch:,}")
        print(f"  总训练步数: {total_train_steps:,}")
        print(f"\n时间估算:")
        print(f"  训练时间: {format_time(train_time)}")
        print(f"  评估时间: {format_time(total_eval_time)}")
        print(f"  总时间(中等): {format_time(total_time)}")
        print(f"\n估算范围:")
        print(f"  乐观估算: {format_time(optimistic_time)}")
        print(f"  保守估算: {format_time(conservative_time)}")
        print(f"{'='*70}\n")

    return results

def estimate_all_datasets(model_type: str, **kwargs):
    """估算所有数据集的训练时间"""
    print(f"\n{'='*70}")
    print(f"所有数据集的 {model_type.upper()} 模型训练时间估算")
    print(f"{'='*70}\n")

    results = {}
    for dataset in sorted(DATASET_SIZES.keys()):
        result = estimate_training_time(
            dataset=dataset,
            model_type=model_type,
            verbose=False,
            **kwargs
        )
        results[dataset] = result

    # 按训练时间排序
    sorted_datasets = sorted(
        results.items(),
        key=lambda x: x[1]['total_time_seconds']
    )

    # 打印表格
    print(f"{'数据集':<15} {'样本数':<12} {'步数/轮':<12} {'总步数':<15} "
          f"{'乐观估算':<18} {'中等估算':<18} {'保守估算':<18}")
    print("-" * 130)

    for dataset, result in sorted_datasets:
        print(f"{dataset:<15} "
              f"{result['train_samples']:<12,} "
              f"{result['steps_per_epoch']:<12,} "
              f"{result['total_steps']:<15,} "
              f"{format_time(result['optimistic_seconds']):<18} "
              f"{format_time(result['total_time_seconds']):<18} "
              f"{format_time(result['conservative_seconds']):<18}")

    # 计算总时间
    total_optimistic = sum(r['optimistic_seconds'] for r in results.values())
    total_medium = sum(r['total_time_seconds'] for r in results.values())
    total_conservative = sum(r['conservative_seconds'] for r in results.values())

    print("-" * 130)
    print(f"{'总计 (所有数据集)':<15} "
          f"{'':<12} {'':<12} {'':<15} "
          f"{format_time(total_optimistic):<18} "
          f"{format_time(total_medium):<18} "
          f"{format_time(total_conservative):<18}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='VulBERTa训练时间估算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 估算MLP模型在devign数据集上的训练时间
  python estimate_training_time.py --model mlp --dataset devign

  # 估算CNN模型在所有数据集上的训练时间
  python estimate_training_time.py --model cnn --dataset all

  # 自定义参数
  python estimate_training_time.py --model mlp --dataset draper --batch_size 1 --epochs 5

  # 指定每步时间（基于实际测试）
  python estimate_training_time.py --model mlp --dataset devign --time_per_step 0.8
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['mlp', 'cnn'],
        help='模型类型 (mlp 或 cnn)'
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='数据集名称 (devign, draper, reveal, mvd, vuldeepecker, d2a, 或 all)'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Batch大小 (默认: MLP=2, CNN=128)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='训练轮数 (默认: MLP=10, CNN=20)'
    )

    parser.add_argument(
        '--time_per_step', '-t',
        type=float,
        default=None,
        help='每步时间(秒) (默认: MLP=0.7, CNN=0.4)'
    )

    args = parser.parse_args()

    # 准备参数
    kwargs = {}
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.epochs:
        kwargs['epochs'] = args.epochs
    if args.time_per_step:
        kwargs['time_per_step'] = args.time_per_step

    # 执行估算
    if args.dataset.lower() == 'all':
        estimate_all_datasets(args.model, **kwargs)
    else:
        estimate_training_time(
            dataset=args.dataset,
            model_type=args.model,
            verbose=True,
            **kwargs
        )

if __name__ == '__main__':
    main()
