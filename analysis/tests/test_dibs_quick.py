#!/usr/bin/env python3
"""
DiBS快速步数验证（最小测试）

目的: 最小化测试，快速验证DiBS是否执行设定步数
测试配置:
- 最小组（group5，60样本）
- 最小步数（100步）
- 最小粒子数（10个）
- 预期时间：< 30秒

用法:
    python3 tests/test_dibs_quick.py
"""

import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_dibs_step_verification import test_dibs_with_callback


def main():
    print("="*80)
    print("DiBS快速步数验证（最小测试）")
    print("="*80)
    print("\n配置:")
    print("  训练步数: 100")
    print("  粒子数: 10")
    print("  callback间隔: 10")
    print("  预期时间: < 30秒")
    print("")

    start = time.time()
    result = test_dibs_with_callback(
        n_steps=100,
        n_particles=10,
        callback_every=10,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\n总测试时间: {elapsed:.1f}秒")

    if result and result['match']:
        print("\n" + "="*80)
        print("✅ 快速测试通过")
        print(f"   DiBS正确执行了 {result['actual_steps']} 步（预期 {result['expected_steps']} 步）")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("❌ 快速测试失败")
        if result:
            print(f"   实际步数: {result['actual_steps']}")
            print(f"   预期步数: {result['expected_steps']}")
            print(f"   差异: {result['discrepancy']}")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
