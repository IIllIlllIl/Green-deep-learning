#!/usr/bin/env python
# 快速验证seed参数的测试脚本
import sys
sys.path.insert(0, '/home/green/energy_dl/nightly/models/Person_reID_baseline_pytorch')

import argparse
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

# 模拟train.py的参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

print(f"Arguments: {sys.argv[1:]}")
print(f"Parsed seed value: {args.seed}")
print(f"Seed type: {type(args.seed)}")

# 测试seed设置逻辑
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"✅ Using seed: {args.seed} (deterministic mode)")
    print(f"✅ cudnn.deterministic: {cudnn.deterministic}")
    print(f"✅ cudnn.benchmark: {cudnn.benchmark}")
else:
    cudnn.benchmark = True
    print(f"✅ No seed set - using non-deterministic training (original behavior)")
    print(f"✅ cudnn.deterministic: {cudnn.deterministic}")
    print(f"✅ cudnn.benchmark: {cudnn.benchmark}")

# 测试随机数生成
print(f"\n随机数测试:")
print(f"  torch.rand(3): {torch.rand(3)}")
print(f"  np.random.rand(3): {np.random.rand(3)}")
print(f"  random.random(): {random.random()}")
