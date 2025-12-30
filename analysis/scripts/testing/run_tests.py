"""
测试运行器 - 运行所有测试并生成报告
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试模块
from tests import test_units, test_integration


def main():
    """运行所有测试"""
    print("="*80)
    print(" "*20 + "COMPREHENSIVE TEST SUITE")
    print("="*80)

    all_success = True

    # 1. 运行单元测试
    print("\n" + "▶"*40)
    print("PHASE 1: UNIT TESTS")
    print("▶"*40)
    unit_success = test_units.run_all_tests()
    all_success = all_success and unit_success

    # 2. 运行集成测试
    print("\n" + "▶"*40)
    print("PHASE 2: INTEGRATION TESTS")
    print("▶"*40)
    integration_success = test_integration.run_integration_tests()
    all_success = all_success and integration_success

    # 最终报告
    print("\n" + "="*80)
    print(" "*25 + "FINAL REPORT")
    print("="*80)
    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print("="*80)

    if all_success:
        print("\n🎉 ALL TESTS PASSED! Your implementation is ready.")
        print("\nNext steps:")
        print("  1. Run code review: python run_code_review.py")
        print("  2. Start data collection: python 1_data_collection.py")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the errors above.")

    return 0 if all_success else 1


if __name__ == '__main__':
    exit(main())
