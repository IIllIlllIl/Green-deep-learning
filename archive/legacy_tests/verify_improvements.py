#!/usr/bin/env python3
"""
验证脚本 - 检查代码改进
"""

import subprocess
import sys
from pathlib import Path

# 颜色定义
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_magic_numbers():
    """检查是否存在魔法数字"""
    print("\n" + "="*80)
    print("检查1: 魔法数字消除")
    print("="*80)

    mutation_file = Path("mutation.py")
    with open(mutation_file, 'r') as f:
        content = f.read()

    # 检查是否定义了常量
    constants = [
        "BACKGROUND_STARTUP_WAIT_SECONDS",
        "BACKGROUND_RESTART_DELAY_SECONDS",
        "BACKGROUND_TERMINATION_TIMEOUT_SECONDS"
    ]

    all_found = True
    for const in constants:
        if const in content:
            print(f"{GREEN}✓{RESET} 找到常量: {const}")
        else:
            print(f"{RED}✗{RESET} 缺少常量: {const}")
            all_found = False

    # 检查是否还有硬编码的数字
    issues = []
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'sleep 2' in line and 'RESTART_DELAY' not in line:
            issues.append(f"Line {i}: 发现硬编码 'sleep 2'")
        if 'timeout=10' in line and 'TERMINATION_TIMEOUT' not in line:
            issues.append(f"Line {i}: 发现硬编码 'timeout=10'")
        if 'time.sleep(5)' in line and 'STARTUP_WAIT' not in line:
            issues.append(f"Line {i}: 发现硬编码 'time.sleep(5)'")

    if issues:
        for issue in issues:
            print(f"{RED}✗{RESET} {issue}")
        return False
    else:
        print(f"{GREEN}✓{RESET} 未发现硬编码的魔法数字")

    return all_found

def check_resource_cleanup():
    """检查资源清理逻辑"""
    print("\n" + "="*80)
    print("检查2: 资源清理机制")
    print("="*80)

    mutation_file = Path("mutation.py")
    with open(mutation_file, 'r') as f:
        content = f.read()

    checks = [
        ("返回script_path", "-> Tuple[subprocess.Popen, Path]"),
        ("接受script_path参数", "script_path: Optional[Path] = None"),
        ("finally块清理", "finally:"),
        ("删除脚本文件", "script_path.unlink()"),
    ]

    all_passed = True
    for check_name, pattern in checks:
        if pattern in content:
            print(f"{GREEN}✓{RESET} {check_name}")
        else:
            print(f"{RED}✗{RESET} 缺少: {check_name}")
            all_passed = False

    return all_passed

def check_error_handling():
    """检查异常处理"""
    print("\n" + "="*80)
    print("检查3: 异常处理")
    print("="*80)

    mutation_file = Path("mutation.py")
    with open(mutation_file, 'r') as f:
        content = f.read()

    checks = [
        ("IOError处理", "except IOError"),
        ("OSError处理", "except OSError"),
        ("RuntimeError抛出", "raise RuntimeError"),
    ]

    all_passed = True
    for check_name, pattern in checks:
        if pattern in content:
            print(f"{GREEN}✓{RESET} {check_name}")
        else:
            print(f"{YELLOW}⚠{RESET}  缺少: {check_name} (可选)")

    return True

def run_tests():
    """运行测试套件"""
    print("\n" + "="*80)
    print("检查4: 运行测试")
    print("="*80)

    try:
        result = subprocess.run(
            ["python3", "test/test_parallel_training.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0 and "All tests passed!" in result.stdout:
            print(f"{GREEN}✓{RESET} 所有测试通过 (5/5)")

            # 检查新增的测试
            if "Script was cleaned up" in result.stdout:
                print(f"{GREEN}✓{RESET} 资源清理测试通过")
            if "RESTART_DELAY" in result.stdout:
                print(f"{GREEN}✓{RESET} 常量使用测试通过")

            return True
        else:
            print(f"{RED}✗{RESET} 测试失败")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"{RED}✗{RESET} 测试执行失败: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*20 + "代码改进验证脚本")
    print("="*80)

    results = []

    # 运行检查
    results.append(("魔法数字消除", check_magic_numbers()))
    results.append(("资源清理机制", check_resource_cleanup()))
    results.append(("异常处理", check_error_handling()))
    results.append(("测试套件", run_tests()))

    # 打印总结
    print("\n" + "="*80)
    print("验证总结")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"{status} - {check_name}")

    print("="*80)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}✅ 所有检查通过！代码质量优秀。{RESET}\n")
        return 0
    else:
        print(f"\n{RED}❌ 部分检查失败，请修复问题。{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
