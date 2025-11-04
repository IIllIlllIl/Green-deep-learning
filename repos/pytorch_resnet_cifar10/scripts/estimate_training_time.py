#!/usr/bin/env python
"""
估算各个ResNet模型在当前设备上的训练时长
通过运行几个批次来测量每个epoch的平均时间
"""
import torch
import torch.nn as nn
import time
import sys
import os

# 添加父目录到路径以导入resnet模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import resnet

def estimate_epoch_time(model_name, num_batches=10, batch_size=128):
    """
    估算一个epoch的训练时间

    Args:
        model_name: 模型名称，如'resnet20'
        num_batches: 用于测量的批次数
        batch_size: 批次大小

    Returns:
        平均每批次时间（秒）
    """
    print(f"\n正在测试 {model_name}...")

    # 创建模型
    model = resnet.__dict__[model_name]()
    model = torch.nn.DataParallel(model)
    model.cuda()

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # 模拟数据
    dummy_input = torch.randn(batch_size, 3, 32, 32).cuda()
    dummy_target = torch.randint(0, 10, (batch_size,)).cuda()

    # 预热
    model.train()
    for _ in range(3):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 实际测量
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_batches):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    avg_batch_time = (end_time - start_time) / num_batches

    # 清理显存
    del model, optimizer, criterion, dummy_input, dummy_target
    torch.cuda.empty_cache()

    return avg_batch_time

def main():
    models = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

    # CIFAR-10训练集有50000个样本
    train_samples = 50000
    batch_size = 128
    batches_per_epoch = train_samples // batch_size  # 390个批次
    total_epochs = 200

    print(f"GPU信息: {torch.cuda.get_device_name(0)}")
    print(f"CIFAR-10训练配置:")
    print(f"  - 训练样本数: {train_samples}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 每个epoch的批次数: {batches_per_epoch}")
    print(f"  - 总epoch数: {total_epochs}")
    print("=" * 80)

    results = []

    for model_name in models:
        try:
            avg_batch_time = estimate_epoch_time(model_name)
            epoch_time = avg_batch_time * batches_per_epoch
            total_time = epoch_time * total_epochs

            result = {
                'model': model_name,
                'batch_time': avg_batch_time,
                'epoch_time': epoch_time,
                'total_time': total_time
            }
            results.append(result)

            print(f"  平均每批次时间: {avg_batch_time:.3f}秒")
            print(f"  预计每个epoch时间: {epoch_time/60:.1f}分钟")
            print(f"  预计总训练时间: {total_time/3600:.1f}小时")

        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'model': model_name,
                'error': str(e)
            })

    print("\n" + "=" * 80)
    print("汇总表格:")
    print("-" * 80)
    print(f"{'模型':<12} {'批次时间':<12} {'Epoch时间':<12} {'总时间':<12} {'是否>2h'}")
    print("-" * 80)

    for result in results:
        if 'error' in result:
            print(f"{result['model']:<12} ERROR: {result['error']}")
        else:
            model = result['model']
            batch_time = f"{result['batch_time']:.3f}s"
            epoch_time = f"{result['epoch_time']/60:.1f}min"
            total_time_hours = result['total_time']/3600
            total_time = f"{total_time_hours:.1f}h"
            over_2h = "是" if total_time_hours > 2 else "否"
            print(f"{model:<12} {batch_time:<12} {epoch_time:<12} {total_time:<12} {over_2h}")

    print("=" * 80)

if __name__ == "__main__":
    main()
