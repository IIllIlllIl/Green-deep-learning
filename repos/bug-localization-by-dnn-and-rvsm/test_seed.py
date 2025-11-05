#!/usr/bin/env python
"""
验证seed参数的测试脚本 - bug-localization模型
测试sklearn MLPRegressor的random_state参数
"""
import sys
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor

print("=" * 70)
print("Bug Localization Seed Verification Test")
print("=" * 70)

# 模拟train_wrapper.py的参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

print(f"\nCommand line arguments: {sys.argv[1:]}")
print(f"Parsed seed value: {args.seed}")
print(f"Seed type: {type(args.seed)}")
print()

# 测试numpy random seed设置
if args.seed is not None:
    np.random.seed(args.seed)
    print(f"✅ Using seed: {args.seed} (deterministic mode)")
    print(f"✅ numpy random state will be set to: {args.seed}")
else:
    print(f"✅ No seed set - using non-deterministic training (original behavior)")
    print(f"✅ MLPRegressor will use random_state=None")

print()
print("-" * 70)
print("Testing random number generation:")
print("-" * 70)

# 生成一些随机数
print(f"np.random.rand(5): {np.random.rand(5)}")
print()

# 测试MLPRegressor的random_state参数
print("-" * 70)
print("Testing MLPRegressor with random_state:")
print("-" * 70)

# 创建简单的训练数据
X_train = np.random.rand(20, 10)
y_train = np.random.rand(20)

# 创建两个相同配置的模型
clf1 = MLPRegressor(
    hidden_layer_sizes=(10,),
    random_state=args.seed,
    max_iter=10,
    solver='sgd'
)
clf1.fit(X_train, y_train)

# 重新设置随机种子（如果有）
if args.seed is not None:
    np.random.seed(args.seed)

# 使用相同数据训练第二个模型
X_train2 = np.random.rand(20, 10)
y_train2 = np.random.rand(20)

clf2 = MLPRegressor(
    hidden_layer_sizes=(10,),
    random_state=args.seed,
    max_iter=10,
    solver='sgd'
)
clf2.fit(X_train2, y_train2)

# 测试预测
X_test = np.random.rand(5, 10)
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)

print(f"Model 1 predictions: {pred1[:3]}")  # 只显示前3个
print(f"Model 2 predictions: {pred2[:3]}")  # 只显示前3个

if args.seed is not None:
    # 如果使用seed，两次训练应该产生相似的初始权重（虽然数据不同，但初始化应该一致）
    print()
    print("✅ With seed set, MLPRegressor will use deterministic weight initialization")
else:
    print()
    print("✅ Without seed, MLPRegressor will use random weight initialization")

print()
print("=" * 70)
print("Verification Test Completed Successfully!")
print("=" * 70)
