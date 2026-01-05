#!/usr/bin/env python3
"""
比较 data.csv 和 raw_data.csv 的数据一致性

功能:
1. 检查基本信息（行数、列数）
2. 检查关键标识字段（experiment_id, timestamp）
3. 检查非并行模式下顶层字段一致性
4. 检查并行模式下fg_字段整合正确性
5. 检查所有数值字段的数值一致性
6. 生成详细的差异报告

版本: v1.0
日期: 2025-12-29
"""

import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class DataComparator:
    """数据比较器"""

    def __init__(self, data_file: str, raw_data_file: str):
        self.data_file = data_file
        self.raw_data_file = raw_data_file

        # 存储数据
        self.data_rows = []
        self.raw_data_rows = []

        # 统计信息
        self.stats = {
            'total_rows': 0,
            'data_rows': 0,
            'raw_data_rows': 0,
            'parallel_rows': 0,
            'nonparallel_rows': 0,
            'mismatches': defaultdict(int),
            'errors': [],
            'warnings': []
        }

        # 定义需要检查的字段映射
        # data.csv字段 -> raw_data.csv字段（非并行） / fg_字段（并行）
        self.field_mappings = {
            # 基础信息
            'experiment_id': 'experiment_id',
            'timestamp': 'timestamp',
            'repository': 'repository',
            'model': 'model',
            'training_success': 'training_success',
            'duration_seconds': 'duration_seconds',
            'retries': 'retries',
            'error_message': 'error_message',

            # 超参数
            'hyperparam_alpha': 'hyperparam_alpha',
            'hyperparam_batch_size': 'hyperparam_batch_size',
            'hyperparam_dropout': 'hyperparam_dropout',
            'hyperparam_epochs': 'hyperparam_epochs',
            'hyperparam_kfold': 'hyperparam_kfold',
            'hyperparam_learning_rate': 'hyperparam_learning_rate',
            'hyperparam_max_iter': 'hyperparam_max_iter',
            'hyperparam_seed': 'hyperparam_seed',
            'hyperparam_weight_decay': 'hyperparam_weight_decay',

            # 性能指标
            'perf_accuracy': 'perf_accuracy',
            'perf_best_val_accuracy': 'perf_best_val_accuracy',
            'perf_map': 'perf_map',
            'perf_precision': 'perf_precision',
            'perf_rank1': 'perf_rank1',
            'perf_rank5': 'perf_rank5',
            'perf_recall': 'perf_recall',
            'perf_test_accuracy': 'perf_test_accuracy',
            'perf_test_loss': 'perf_test_loss',
            'perf_eval_loss': 'perf_eval_loss',
            'perf_final_training_loss': 'perf_final_training_loss',
            'perf_eval_samples_per_second': 'perf_eval_samples_per_second',
            'perf_top1_accuracy': 'perf_top1_accuracy',
            'perf_top5_accuracy': 'perf_top5_accuracy',
            'perf_top10_accuracy': 'perf_top10_accuracy',
            'perf_top20_accuracy': 'perf_top20_accuracy',

            # 能耗指标
            'energy_cpu_pkg_joules': 'energy_cpu_pkg_joules',
            'energy_cpu_ram_joules': 'energy_cpu_ram_joules',
            'energy_cpu_total_joules': 'energy_cpu_total_joules',
            'energy_gpu_avg_watts': 'energy_gpu_avg_watts',
            'energy_gpu_max_watts': 'energy_gpu_max_watts',
            'energy_gpu_min_watts': 'energy_gpu_min_watts',
            'energy_gpu_total_joules': 'energy_gpu_total_joules',
            'energy_gpu_temp_avg_celsius': 'energy_gpu_temp_avg_celsius',
            'energy_gpu_temp_max_celsius': 'energy_gpu_temp_max_celsius',
            'energy_gpu_util_avg_percent': 'energy_gpu_util_avg_percent',
            'energy_gpu_util_max_percent': 'energy_gpu_util_max_percent',
        }

    def load_data(self):
        """加载两个CSV文件"""
        print("加载数据文件...")

        # 加载 data.csv
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            self.data_rows = list(reader)
            self.stats['data_rows'] = len(self.data_rows)

        # 加载 raw_data.csv
        with open(self.raw_data_file, 'r') as f:
            reader = csv.DictReader(f)
            self.raw_data_rows = list(reader)
            self.stats['raw_data_rows'] = len(self.raw_data_rows)

        print(f"✓ data.csv: {self.stats['data_rows']} 行")
        print(f"✓ raw_data.csv: {self.stats['raw_data_rows']} 行")
        print()

        # 检查行数是否一致
        if self.stats['data_rows'] != self.stats['raw_data_rows']:
            self.stats['errors'].append(
                f"行数不一致: data.csv有{self.stats['data_rows']}行, "
                f"raw_data.csv有{self.stats['raw_data_rows']}行"
            )

    def check_basic_info(self):
        """检查基本信息一致性"""
        print("=" * 80)
        print("1. 检查基本信息一致性")
        print("=" * 80)

        if self.stats['data_rows'] != self.stats['raw_data_rows']:
            print(f"✗ 行数不一致!")
            print(f"  data.csv: {self.stats['data_rows']} 行")
            print(f"  raw_data.csv: {self.stats['raw_data_rows']} 行")
            return False
        else:
            print(f"✓ 行数一致: {self.stats['data_rows']} 行")

        print()
        return True

    def check_key_fields(self):
        """检查关键标识字段"""
        print("=" * 80)
        print("2. 检查关键标识字段 (experiment_id, timestamp)")
        print("=" * 80)

        mismatches = []

        for i, (data_row, raw_row) in enumerate(zip(self.data_rows, self.raw_data_rows)):
            row_num = i + 2  # +2 因为有header，而且从1开始计数

            # 检查 experiment_id
            if data_row['experiment_id'] != raw_row['experiment_id']:
                mismatches.append({
                    'row': row_num,
                    'field': 'experiment_id',
                    'data_value': data_row['experiment_id'],
                    'raw_value': raw_row['experiment_id']
                })

            # 检查 timestamp
            if data_row['timestamp'] != raw_row['timestamp']:
                mismatches.append({
                    'row': row_num,
                    'field': 'timestamp',
                    'data_value': data_row['timestamp'],
                    'raw_value': raw_row['timestamp']
                })

        if mismatches:
            print(f"✗ 发现 {len(mismatches)} 处不一致:")
            for m in mismatches[:10]:  # 只显示前10个
                print(f"  行{m['row']}, 字段'{m['field']}': "
                      f"data='{m['data_value']}', raw='{m['raw_value']}'")
            if len(mismatches) > 10:
                print(f"  ... 还有 {len(mismatches) - 10} 处不一致")
            self.stats['errors'].extend(mismatches)
        else:
            print("✓ 所有关键标识字段完全一致")

        print()
        return len(mismatches) == 0

    def get_raw_field_value(self, raw_row: Dict, field: str, is_parallel: bool) -> str:
        """
        从raw_data获取字段值（模拟create_unified_data_csv.py的逻辑）

        并行模式: 优先使用fg_字段，fallback到顶层字段
        非并行模式: 直接使用顶层字段
        """
        if is_parallel:
            # 优先使用 fg_ 前缀字段
            fg_value = raw_row.get(f'fg_{field}', '').strip()
            if fg_value:
                return fg_value
        # fallback 到顶层字段
        return raw_row.get(field, '').strip()

    def check_field_consistency(self):
        """检查所有字段的一致性"""
        print("=" * 80)
        print("3. 检查字段值一致性")
        print("=" * 80)

        field_mismatches = defaultdict(list)
        total_checks = 0
        total_mismatches = 0

        # 统计并行/非并行数量
        for raw_row in self.raw_data_rows:
            if raw_row['mode'] == 'parallel':
                self.stats['parallel_rows'] += 1
            else:
                self.stats['nonparallel_rows'] += 1

        print(f"数据分布: 非并行={self.stats['nonparallel_rows']}, "
              f"并行={self.stats['parallel_rows']}")
        print()

        # 逐行逐字段检查
        for i, (data_row, raw_row) in enumerate(zip(self.data_rows, self.raw_data_rows)):
            row_num = i + 2
            is_parallel = (raw_row['mode'] == 'parallel')

            # 检查每个映射字段
            for data_field, raw_field in self.field_mappings.items():
                total_checks += 1

                # 从data.csv获取值
                data_value = data_row.get(data_field, '').strip()

                # 从raw_data.csv获取值（使用相同的逻辑）
                raw_value = self.get_raw_field_value(raw_row, raw_field, is_parallel)

                # 比较
                if data_value != raw_value:
                    total_mismatches += 1
                    field_mismatches[data_field].append({
                        'row': row_num,
                        'mode': 'parallel' if is_parallel else 'nonparallel',
                        'data_value': data_value,
                        'raw_value': raw_value,
                        'raw_toplevel': raw_row.get(raw_field, '').strip(),
                        'raw_fg': raw_row.get(f'fg_{raw_field}', '').strip() if is_parallel else 'N/A'
                    })

        # 报告结果
        print(f"总检查次数: {total_checks:,}")
        print(f"不一致次数: {total_mismatches:,} ({total_mismatches/total_checks*100:.2f}%)")
        print()

        if field_mismatches:
            print(f"发现 {len(field_mismatches)} 个字段有不一致:")
            print()

            # 按字段分组显示
            for field, mismatches in sorted(field_mismatches.items(),
                                           key=lambda x: len(x[1]),
                                           reverse=True):
                print(f"字段 '{field}': {len(mismatches)} 处不一致")

                # 显示前3个示例
                for m in mismatches[:3]:
                    print(f"  行{m['row']} ({m['mode']}): "
                          f"data='{m['data_value']}', raw='{m['raw_value']}'")
                    if m['mode'] == 'parallel':
                        print(f"    (raw顶层='{m['raw_toplevel']}', raw_fg='{m['raw_fg']}')")

                if len(mismatches) > 3:
                    print(f"  ... 还有 {len(mismatches) - 3} 处")
                print()

                self.stats['mismatches'][field] = len(mismatches)
        else:
            print("✓ 所有字段值完全一致!")

        print()
        return total_mismatches == 0

    def check_parallel_specific_fields(self):
        """检查并行模式特有字段"""
        print("=" * 80)
        print("4. 检查并行模式特有字段")
        print("=" * 80)

        parallel_fields = [
            'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
            'fg_duration_seconds', 'fg_retries', 'fg_error_message'
        ]

        mismatches = []

        for i, (data_row, raw_row) in enumerate(zip(self.data_rows, self.raw_data_rows)):
            row_num = i + 2
            is_parallel = (raw_row['mode'] == 'parallel')

            for field in parallel_fields:
                data_value = data_row.get(field, '').strip()
                raw_value = raw_row.get(field, '').strip()

                # 非并行模式下，这些字段应该为空
                if not is_parallel:
                    if data_value:
                        mismatches.append({
                            'row': row_num,
                            'field': field,
                            'issue': f'非并行模式但有值: {data_value}'
                        })
                else:
                    # 并行模式下，检查值是否一致
                    if data_value != raw_value:
                        mismatches.append({
                            'row': row_num,
                            'field': field,
                            'issue': f'值不一致: data={data_value}, raw={raw_value}'
                        })

        if mismatches:
            print(f"✗ 发现 {len(mismatches)} 处问题:")
            for m in mismatches[:10]:
                print(f"  行{m['row']}, 字段'{m['field']}': {m['issue']}")
            if len(mismatches) > 10:
                print(f"  ... 还有 {len(mismatches) - 10} 处")
            self.stats['warnings'].extend(mismatches)
        else:
            print("✓ 并行模式特有字段全部正确")

        print()
        return len(mismatches) == 0

    def check_is_parallel_field(self):
        """检查 is_parallel 字段正确性"""
        print("=" * 80)
        print("5. 检查 is_parallel 字段正确性")
        print("=" * 80)

        mismatches = []

        for i, (data_row, raw_row) in enumerate(zip(self.data_rows, self.raw_data_rows)):
            row_num = i + 2

            is_parallel_data = data_row.get('is_parallel', '').strip()
            mode_raw = raw_row.get('mode', '').strip()

            expected = 'True' if mode_raw == 'parallel' else 'False'

            if is_parallel_data != expected:
                mismatches.append({
                    'row': row_num,
                    'is_parallel': is_parallel_data,
                    'mode': mode_raw,
                    'expected': expected
                })

        if mismatches:
            print(f"✗ 发现 {len(mismatches)} 处不一致:")
            for m in mismatches[:10]:
                print(f"  行{m['row']}: is_parallel={m['is_parallel']}, "
                      f"mode={m['mode']}, 应为{m['expected']}")
            self.stats['errors'].extend(mismatches)
        else:
            print("✓ is_parallel 字段全部正确")

        print()
        return len(mismatches) == 0

    def generate_report(self):
        """生成汇总报告"""
        print("=" * 80)
        print("汇总报告")
        print("=" * 80)
        print()

        print("文件信息:")
        print(f"  data.csv: {self.stats['data_rows']} 行, 56 列")
        print(f"  raw_data.csv: {self.stats['raw_data_rows']} 行, 87 列")
        print()

        print("数据分布:")
        print(f"  非并行模式: {self.stats['nonparallel_rows']} 行")
        print(f"  并行模式: {self.stats['parallel_rows']} 行")
        print()

        print("检查结果:")
        if self.stats['errors']:
            print(f"  ✗ 严重错误: {len(self.stats['errors'])} 个")
        else:
            print("  ✓ 无严重错误")

        if self.stats['warnings']:
            print(f"  ⚠ 警告: {len(self.stats['warnings'])} 个")
        else:
            print("  ✓ 无警告")

        if self.stats['mismatches']:
            print(f"  ✗ 字段不一致: {len(self.stats['mismatches'])} 个字段")
            total_field_mismatches = sum(self.stats['mismatches'].values())
            print(f"    总不一致次数: {total_field_mismatches:,}")
        else:
            print("  ✓ 所有字段值一致")

        print()

        # 最终结论
        if not self.stats['errors'] and not self.stats['mismatches']:
            print("=" * 80)
            print("✅ 结论: data.csv 和 raw_data.csv 完全一致!")
            print("=" * 80)
            return True
        else:
            print("=" * 80)
            print("❌ 结论: 发现数据不一致，需要进一步调查")
            print("=" * 80)
            return False

    def run(self):
        """运行完整检查流程"""
        print("=" * 80)
        print("data.csv vs raw_data.csv 数据一致性检查")
        print("=" * 80)
        print()

        # 1. 加载数据
        self.load_data()

        # 2. 检查基本信息
        self.check_basic_info()

        # 3. 检查关键标识字段
        self.check_key_fields()

        # 4. 检查is_parallel字段
        self.check_is_parallel_field()

        # 5. 检查所有字段一致性
        self.check_field_consistency()

        # 6. 检查并行模式特有字段
        self.check_parallel_specific_fields()

        # 7. 生成报告
        result = self.generate_report()

        return 0 if result else 1

def main():
    """主函数"""
    data_file = 'data/data.csv'
    raw_data_file = 'data/raw_data.csv'

    comparator = DataComparator(data_file, raw_data_file)
    return comparator.run()

if __name__ == '__main__':
    sys.exit(main())
