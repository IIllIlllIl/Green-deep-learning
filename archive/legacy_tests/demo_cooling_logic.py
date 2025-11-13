#!/usr/bin/env python3
"""
快速演示：并行训练的GPU冷却逻辑

此脚本演示新的并行训练行为:
1. 后台启动等待 30秒（确保完全运行）
2. 60秒冷却期（GPU完全空闲）

运行方式:
    python3 test/demo_cooling_logic.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mutation import MutationRunner


def demo_cooling_logic():
    """演示GPU冷却逻辑"""
    print("=" * 80)
    print("并行训练 GPU 冷却逻辑演示")
    print("=" * 80)
    print()

    runner = MutationRunner()

    print("📊 配置信息:")
    print(f"   后台启动等待时间: {runner.BACKGROUND_STARTUP_WAIT_SECONDS} 秒")
    print(f"   运行间隔冷却时间: {runner.RUN_SLEEP_SECONDS} 秒")
    print(f"   后台重启延迟: {runner.BACKGROUND_RESTART_DELAY_SECONDS} 秒")
    print()

    # 模拟2次并行运行
    num_runs = 2
    experiment_id_base = f"demo_{int(time.time())}"

    for run in range(1, num_runs + 1):
        print("\n" + "=" * 80)
        print(f"模拟并行运行 {run}/{num_runs}")
        print("=" * 80)

        experiment_id = f"{experiment_id_base}_run{run}"

        # 1. 启动后台训练
        print(f"\n[步骤1] 启动后台训练...")
        bg_process, bg_script = runner._start_background_training(
            repo="pytorch_resnet_cifar10",
            model="resnet20",
            hyperparams={"epochs": 1, "learning_rate": 0.01},
            experiment_id=experiment_id
        )

        # 2. 等待后台训练稳定
        print(f"\n[步骤2] 等待后台训练完全启动...")
        print(f"⏳ 等待 {runner.BACKGROUND_STARTUP_WAIT_SECONDS} 秒...")
        for i in range(runner.BACKGROUND_STARTUP_WAIT_SECONDS):
            if i % 5 == 0:
                print(f"   {i}/{runner.BACKGROUND_STARTUP_WAIT_SECONDS} 秒 (后台训练运行中...)")
            time.sleep(1)
        print(f"✓ 后台训练已完全启动")

        # 3. 模拟前景训练（这里只等待10秒以节省时间）
        print(f"\n[步骤3] 运行前景训练...")
        print("🚀 前景训练开始（模拟：10秒）...")
        for i in range(1, 11):
            print(f"   前景训练进度: {i}/10 秒")
            time.sleep(1)
        print("✅ 前景训练完成")

        # 4. 停止后台训练
        print(f"\n[步骤4] 停止后台训练...")
        runner._stop_background_training(bg_process, bg_script)
        print("✓ 后台训练已停止")

        # 5. GPU冷却期
        if run < num_runs:
            print(f"\n[步骤5] GPU 冷却期")
            print("❄️  所有训练已停止，GPU进入冷却模式")
            print(f"⏳ 冷却 {runner.RUN_SLEEP_SECONDS} 秒...")

            # 显示冷却倒计时
            for i in range(runner.RUN_SLEEP_SECONDS):
                if i % 10 == 0:
                    remaining = runner.RUN_SLEEP_SECONDS - i
                    print(f"   冷却中... 剩余 {remaining} 秒 (GPU 空闲)")
                time.sleep(1)

            print("✓ GPU冷却完成，准备下一次运行")
        else:
            print(f"\n✨ 所有运行完成！")

    print("\n" + "=" * 80)
    print("演示总结")
    print("=" * 80)
    print(f"✅ 模拟了 {num_runs} 次并行运行")
    print(f"✅ 每次运行:")
    print(f"   1. 启动后台 → 等待 {runner.BACKGROUND_STARTUP_WAIT_SECONDS}秒")
    print(f"   2. 前景训练 (完整监控)")
    print(f"   3. 停止后台")
    print(f"   4. GPU冷却 {runner.RUN_SLEEP_SECONDS}秒")
    print()
    print("💡 关键特性:")
    print("   • 后台训练每次运行都重新启动")
    print("   • 60秒冷却期内GPU完全空闲")
    print("   • 30秒等待确保后台训练完全运行")
    print("=" * 80)


if __name__ == "__main__":
    try:
        demo_cooling_logic()
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
        sys.exit(1)
