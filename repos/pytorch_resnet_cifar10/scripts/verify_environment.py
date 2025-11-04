#!/usr/bin/env python
"""
验证环境配置是否正确
检查PyTorch、CUDA、GPU等是否正常工作
"""
import sys

def check_python_version():
    """检查Python版本"""
    print("=" * 80)
    print("检查Python版本")
    print("-" * 80)
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("✗ Python版本过低，建议使用Python 3.8或更高版本")
        return False

def check_pytorch():
    """检查PyTorch安装"""
    print("\n" + "=" * 80)
    print("检查PyTorch")
    print("-" * 80)
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print("✓ PyTorch已安装")
        return True, torch
    except ImportError:
        print("✗ PyTorch未安装")
        print("请运行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False, None

def check_cuda(torch):
    """检查CUDA支持"""
    print("\n" + "=" * 80)
    print("检查CUDA")
    print("-" * 80)
    if torch is None:
        print("✗ 跳过CUDA检查（PyTorch未安装）")
        return False

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA可用")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  可用GPU数量: {torch.cuda.device_count()}")
        return True
    else:
        print("✗ CUDA不可用")
        print("请检查：")
        print("  1. 是否安装了NVIDIA驱动")
        print("  2. 是否安装了支持CUDA的PyTorch版本")
        return False

def check_gpu(torch):
    """检查GPU信息"""
    print("\n" + "=" * 80)
    print("检查GPU")
    print("-" * 80)
    if torch is None or not torch.cuda.is_available():
        print("✗ 跳过GPU检查（CUDA不可用）")
        return False

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")

        # 检查当前显存使用
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  已分配显存: {memory_allocated:.2f} GB")
        print(f"  已预留显存: {memory_reserved:.2f} GB")

    print("\n✓ GPU信息检查完成")
    return True

def check_torchvision():
    """检查torchvision"""
    print("\n" + "=" * 80)
    print("检查torchvision")
    print("-" * 80)
    try:
        import torchvision
        print(f"torchvision版本: {torchvision.__version__}")
        print("✓ torchvision已安装")
        return True
    except ImportError:
        print("✗ torchvision未安装")
        print("请运行: pip install torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_model_import():
    """检查能否导入resnet模块"""
    print("\n" + "=" * 80)
    print("检查ResNet模型")
    print("-" * 80)
    try:
        import os
        # 添加项目根目录到路径
        project_root = os.getcwd()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import resnet
        print("✓ 成功导入resnet模块")

        # 检查所有模型
        models = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
        print("\n可用的模型:")
        for model_name in models:
            if hasattr(resnet, model_name):
                print(f"  ��� {model_name}")
            else:
                print(f"  ✗ {model_name} (未找到)")

        return True
    except ImportError as e:
        print(f"✗ 无法导入resnet模块: {e}")
        print("请确保在项目根目录运行此脚本")
        return False

def test_simple_training(torch):
    """测试简单的训练流程"""
    print("\n" + "=" * 80)
    print("测试训练流程")
    print("-" * 80)

    if torch is None or not torch.cuda.is_available():
        print("✗ 跳过训练测试（CUDA不可用）")
        return False

    try:
        import os
        # 添加项目根目录到路径
        project_root = os.getcwd()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import resnet
        import torch.nn as nn

        print("创建ResNet20模型...")
        model = resnet.resnet20()
        model = torch.nn.DataParallel(model)
        model.cuda()

        print("创建虚拟数据...")
        dummy_input = torch.randn(4, 3, 32, 32).cuda()
        dummy_target = torch.randint(0, 10, (4,)).cuda()

        print("执行前向传播...")
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        print("执行反向传播...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}")
        print("✓ 训练流程测试成功")

        # 清理
        del model, optimizer, criterion, dummy_input, dummy_target
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ 训练流程测试失败: {e}")
        return False

def check_data_directory():
    """检查数据目录"""
    print("\n" + "=" * 80)
    print("检查数据目录")
    print("-" * 80)
    import os

    if os.path.exists('./data'):
        print("✓ data目录已存在")
        # 检查是否有CIFAR-10数据
        cifar_files = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5', 'test_batch']
        cifar_path = './data/cifar-10-batches-py'
        if os.path.exists(cifar_path):
            all_exist = all(os.path.exists(os.path.join(cifar_path, f)) for f in cifar_files)
            if all_exist:
                print("✓ CIFAR-10数据集已下载")
                return True
            else:
                print("! CIFAR-10数据集不完整，首次训练时会自动下载")
                return True
        else:
            print("! CIFAR-10数据集未下载，首次训练时会自动下载")
            return True
    else:
        print("! data目录不存在，首次训练时会自动创建并下载数据集")
        return True

def estimate_resnet1202_feasibility(torch):
    """估算是否能训练ResNet1202"""
    print("\n" + "=" * 80)
    print("ResNet1202可行性分析")
    print("-" * 80)

    if torch is None or not torch.cuda.is_available():
        print("✗ 跳过分析（CUDA不可用）")
        return

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU总显存: {total_memory:.2f} GB")

    if total_memory >= 16:
        print("✓ 显存充足，可以使用默认配置训练ResNet1202 (batch_size=128)")
    elif total_memory >= 10:
        print("! 显存适中，建议使用优化配置:")
        print("  - 方案1: --batch-size=32 --half")
        print("  - 方案2: --batch-size=16")
    else:
        print("✗ 显存不足，建议:")
        print("  - 使用预训练模型进行评估")
        print("  - 或使用更小的模型 (ResNet20-110)")

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("PyTorch ResNet CIFAR-10 环境验证")
    print("=" * 80)

    results = {}

    # 检查Python版本
    results['python'] = check_python_version()

    # 检查PyTorch
    pytorch_ok, torch = check_pytorch()
    results['pytorch'] = pytorch_ok

    # 检查CUDA
    results['cuda'] = check_cuda(torch)

    # 检查GPU
    results['gpu'] = check_gpu(torch)

    # 检查torchvision
    results['torchvision'] = check_torchvision()

    # 检查模型导入
    results['model'] = check_model_import()

    # 检查数据目录
    results['data'] = check_data_directory()

    # 测试训练流程
    results['training'] = test_simple_training(torch)

    # ResNet1202可行性分析
    estimate_resnet1202_feasibility(torch)

    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    all_passed = all(results.values())

    print("\n检查项目:")
    for item, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {item}")

    print("\n" + "-" * 80)
    if all_passed:
        print("✓ 所有检查通过！环境配置正确。")
        print("\n可以开始训练:")
        print("  python -u trainer.py --arch=resnet20 --save-dir=save_resnet20")
        print("  或运行 ./run.sh 训练所有模型")
    else:
        print("✗ 部分检查未通过，请根据上述提示修复问题。")
        print("\n参考文档:")
        print("  docs/environment_setup.md")

    print("=" * 80 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
