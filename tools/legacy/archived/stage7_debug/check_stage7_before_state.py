#!/usr/bin/env python3
"""
使用Stage7运行前的历史数据重新模拟

关键：summary_all.csv在Stage7运行前只有381行，运行后是388行
我们需要用381行的数据来模拟Stage7的生成过程
"""

import csv
import sys
from pathlib import Path

# 添加mutation模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 读取当前的summary_all.csv（388行）
csv_path = Path(__file__).parent.parent / "results" / "summary_all.csv"

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

print(f"当前summary_all.csv总行数: {len(all_rows)}")
print(f"Stage7添加的实验数: 7")
print(f"Stage7运行前的历史数据: {len(all_rows) - 7} 行")

# 移除最后7行（Stage7添加的），重建Stage7运行前的状态
historical_rows = all_rows[:-7]

print(f"\n最后7行（Stage7添加的实验）:")
print("=" * 80)
for i, row in enumerate(all_rows[-7:], 1):
    print(f"{i}. {row['experiment_id']}: {row['repository']}/{row['model']}")
    # 显示关键参数
    params = []
    for key in ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_batch_size', 'hyperparam_seed']:
        if row.get(key):
            params.append(f"{key.replace('hyperparam_', '')}={row[key][:10]}")
    print(f"   参数: {', '.join(params[:3])}")

print("\n" + "=" * 80)
print("关键发现：每个配置只运行了1个实验，而不是7个！")
print("=" * 80)
