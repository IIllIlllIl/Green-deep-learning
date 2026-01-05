#!/usr/bin/env python3
"""
通用脚本：从任意session目录提取实验数据并追加到raw_data.csv

用途：
1. 自动从指定session目录提取所有实验
2. 从experiment.json和terminal_output.txt提取完整数据
3. 去重检查，只追加新实验
4. 自动备份raw_data.csv
5. 验证数据完整性

用法：
    python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS
    python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS --dry-run
    python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS --no-backup

版本：1.0
创建日期：2025-12-13
"""

import json
import csv
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 默认配置
DEFAULT_RAW_DATA_CSV = Path('data/raw_data.csv')
DEFAULT_MODELS_CONFIG = Path('mutation/models_config.json')


class SessionDataAppender:
    """Session数据追加器"""

    def __init__(self, session_dir, raw_data_csv=None, models_config_path=None,
                 dry_run=False, create_backup=True, verbose=True):
        """
        初始化

        Args:
            session_dir: session目录路径
            raw_data_csv: raw_data.csv路径（默认：data/raw_data.csv）
            models_config_path: models_config.json路径（默认：mutation/models_config.json）
            dry_run: 是否为测试运行（不实际写入）
            create_backup: 是否创建备份
            verbose: 是否显示详细信息
        """
        self.session_dir = Path(session_dir)
        self.raw_data_csv = Path(raw_data_csv) if raw_data_csv else DEFAULT_RAW_DATA_CSV
        self.models_config_path = Path(models_config_path) if models_config_path else DEFAULT_MODELS_CONFIG
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.verbose = verbose

        # 加载配置
        self.models_config = self._load_models_config()

        # 统计信息
        self.stats = {
            'total_found': 0,
            'skipped_no_json': 0,
            'skipped_unknown_repo': 0,
            'skipped_duplicate': 0,
            'added': 0
        }

    def _log(self, message):
        """打印日志"""
        if self.verbose:
            print(message)

    def _load_models_config(self):
        """加载models_config.json"""
        if not self.models_config_path.exists():
            raise FileNotFoundError(f"Models config not found: {self.models_config_path}")

        with open(self.models_config_path, 'r') as f:
            return json.load(f)['models']

    def _extract_performance_from_terminal_output(self, terminal_output_path, log_patterns):
        """从terminal_output.txt提取性能指标"""
        if not terminal_output_path.exists():
            return {}

        try:
            with open(terminal_output_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self._log(f"   ⚠️  读取terminal_output失败: {e}")
            return {}

        metrics = {}
        for metric_name, pattern in log_patterns.items():
            try:
                match = re.search(pattern, content)
                if match:
                    value = float(match.group(1))
                    metrics[f'perf_{metric_name}'] = value
            except (ValueError, IndexError, AttributeError):
                pass

        return metrics

    def _load_experiment_json(self, exp_dir):
        """加载experiment.json"""
        json_path = exp_dir / 'experiment.json'
        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"   ⚠️  加载experiment.json失败: {e}")
            return None

    def _build_row_from_experiment(self, exp_data, perf_metrics, fieldnames):
        """从实验数据构建CSV行"""
        # 初始化所有字段为空
        row = {key: '' for key in fieldnames}

        # 检查是否为并行模式
        is_parallel = exp_data.get('mode') == 'parallel'

        if is_parallel:
            # 并行模式：从foreground中提取数据
            fg_data = exp_data.get('foreground', {})

            # 填充基础字段
            row['experiment_id'] = exp_data.get('experiment_id', '')
            row['timestamp'] = exp_data.get('timestamp', '')
            row['repository'] = fg_data.get('repository', '')
            row['model'] = fg_data.get('model', '')
            row['training_success'] = str(fg_data.get('training_success', ''))
            row['duration_seconds'] = str(fg_data.get('duration_seconds', ''))
            row['retries'] = str(fg_data.get('retries', 0))
            row['experiment_source'] = exp_data.get('experiment_source', '')  # 顶层字段
            row['num_mutated_params'] = str(exp_data.get('num_mutated_params', ''))  # 顶层字段
            row['mutated_param'] = exp_data.get('mutated_param', '')  # 顶层字段
            row['mode'] = exp_data.get('mode', '')
            row['error_message'] = fg_data.get('error_message', '')

            # 填充超参数（从foreground）
            hyperparams = fg_data.get('hyperparameters', {})
            for key, value in hyperparams.items():
                col_name = f'hyperparam_{key}'
                if col_name in fieldnames:
                    row[col_name] = str(value)

            # 填充能耗数据（从foreground.energy_metrics）
            energy = fg_data.get('energy_metrics', {})
            # 映射energy_metrics的字段名到CSV列名
            energy_mapping = {
                'cpu_energy_pkg_joules': 'energy_cpu_pkg_joules',
                'cpu_energy_ram_joules': 'energy_cpu_ram_joules',
                'cpu_energy_total_joules': 'energy_cpu_total_joules',
                'gpu_power_avg_watts': 'energy_gpu_avg_watts',
                'gpu_power_max_watts': 'energy_gpu_max_watts',
                'gpu_power_min_watts': 'energy_gpu_min_watts',
                'gpu_energy_total_joules': 'energy_gpu_total_joules',
                'gpu_temp_avg_celsius': 'energy_gpu_temp_avg_celsius',
                'gpu_temp_max_celsius': 'energy_gpu_temp_max_celsius',
                'gpu_util_avg_percent': 'energy_gpu_util_avg_percent',
                'gpu_util_max_percent': 'energy_gpu_util_max_percent'
            }
            for src_key, dst_key in energy_mapping.items():
                if src_key in energy and dst_key in fieldnames:
                    row[dst_key] = str(energy[src_key])

            # 填充性能数据（从foreground.performance_metrics和从terminal_output提取的）
            fg_perf = fg_data.get('performance_metrics', {})
            # 先填充foreground中的性能指标
            perf_mapping = {
                'eval_loss': 'perf_eval_loss',
                'final_training_loss': 'perf_final_training_loss',
                'eval_samples_per_second': 'perf_eval_samples_per_second',
                'accuracy': 'perf_accuracy',
                'precision': 'perf_precision',
                'recall': 'perf_recall',
                'f1': 'perf_f1',
                'top1_accuracy': 'perf_top1_accuracy',
                'top5_accuracy': 'perf_top5_accuracy',
                'top10_accuracy': 'perf_top10_accuracy',
                'top20_accuracy': 'perf_top20_accuracy',
                'test_accuracy': 'perf_test_accuracy',
                'test_error': 'perf_test_error',
                'train_error': 'perf_train_error'
            }
            for src_key, dst_key in perf_mapping.items():
                if src_key in fg_perf and dst_key in fieldnames:
                    row[dst_key] = str(fg_perf[src_key])

            # 再填充从terminal_output提取的性能指标（可能会覆盖）
            for key, value in perf_metrics.items():
                if key in fieldnames:
                    row[key] = str(value)

        else:
            # 非并行模式：直接从顶层提取数据
            row['experiment_id'] = exp_data.get('experiment_id', '')
            row['timestamp'] = exp_data.get('timestamp', '')
            row['repository'] = exp_data.get('repository', '')
            row['model'] = exp_data.get('model', '')
            row['training_success'] = str(exp_data.get('training_success', ''))
            row['duration_seconds'] = str(exp_data.get('duration_seconds', ''))
            row['retries'] = str(exp_data.get('retries', 0))
            row['experiment_source'] = exp_data.get('experiment_source', '')
            row['num_mutated_params'] = str(exp_data.get('num_mutated_params', ''))
            row['mutated_param'] = exp_data.get('mutated_param', '')
            row['mode'] = exp_data.get('mode', '')
            row['error_message'] = exp_data.get('error_message', '')

            # 填充超参数
            hyperparams = exp_data.get('hyperparameters', {})
            for key, value in hyperparams.items():
                col_name = f'hyperparam_{key}'
                if col_name in fieldnames:
                    row[col_name] = str(value)

            # 填充能耗数据（使用energy_metrics）
            energy = exp_data.get('energy_metrics', {})
            # 映射energy_metrics的字段名到CSV列名
            energy_mapping = {
                'cpu_energy_pkg_joules': 'energy_cpu_pkg_joules',
                'cpu_energy_ram_joules': 'energy_cpu_ram_joules',
                'cpu_energy_total_joules': 'energy_cpu_total_joules',
                'gpu_power_avg_watts': 'energy_gpu_avg_watts',
                'gpu_power_max_watts': 'energy_gpu_max_watts',
                'gpu_power_min_watts': 'energy_gpu_min_watts',
                'gpu_energy_total_joules': 'energy_gpu_total_joules',
                'gpu_temp_avg_celsius': 'energy_gpu_temp_avg_celsius',
                'gpu_temp_max_celsius': 'energy_gpu_temp_max_celsius',
                'gpu_util_avg_percent': 'energy_gpu_util_avg_percent',
                'gpu_util_max_percent': 'energy_gpu_util_max_percent'
            }
            for src_key, dst_key in energy_mapping.items():
                if src_key in energy and dst_key in fieldnames:
                    row[dst_key] = str(energy[src_key])

            # 填充性能数据（从experiment.json的performance_metrics和terminal_output）
            exp_perf = exp_data.get('performance_metrics', {})
            # 先填充experiment.json中的性能指标
            perf_mapping = {
                'eval_loss': 'perf_eval_loss',
                'final_training_loss': 'perf_final_training_loss',
                'eval_samples_per_second': 'perf_eval_samples_per_second',
                'accuracy': 'perf_accuracy',
                'precision': 'perf_precision',
                'recall': 'perf_recall',
                'f1': 'perf_f1',
                'top1_accuracy': 'perf_top1_accuracy',
                'top5_accuracy': 'perf_top5_accuracy',
                'top10_accuracy': 'perf_top10_accuracy',
                'top20_accuracy': 'perf_top20_accuracy',
                'test_accuracy': 'perf_test_accuracy',
                'test_error': 'perf_test_error',
                'train_error': 'perf_train_error'
            }
            for src_key, dst_key in perf_mapping.items():
                if src_key in exp_perf and dst_key in fieldnames:
                    row[dst_key] = str(exp_perf[src_key])

            # 再填充从terminal_output提取的性能指标（可能会覆盖）
            for key, value in perf_metrics.items():
                if key in fieldnames:
                    row[key] = str(value)

        return row

    def _is_duplicate(self, exp_data, existing_keys):
        """
        检查是否为重复实验

        使用复合键：experiment_id + timestamp
        这样可以避免不同批次产生相同 experiment_id 的问题

        Args:
            exp_data: 实验数据字典
            existing_keys: 现有实验的复合键集合

        Returns:
            bool: 是否为重复实验
        """
        exp_id = exp_data.get('experiment_id', '')
        timestamp = exp_data.get('timestamp', '')

        # 创建复合键
        composite_key = f"{exp_id}|{timestamp}"

        return composite_key in existing_keys

    def extract_experiments(self):
        """从session目录提取所有实验"""
        if not self.session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {self.session_dir}")

        self._log('=' * 80)
        self._log(f'从Session提取实验: {self.session_dir.name}')
        self._log('=' * 80)
        self._log('')

        # 读取现有raw_data.csv
        if not self.raw_data_csv.exists():
            raise FileNotFoundError(f"raw_data.csv not found: {self.raw_data_csv}")

        with open(self.raw_data_csv, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)

        # 构建复合键集合（experiment_id + timestamp）
        existing_keys = set()
        for row in existing_rows:
            exp_id = row.get('experiment_id', '')
            timestamp = row.get('timestamp', '')
            composite_key = f"{exp_id}|{timestamp}"
            existing_keys.add(composite_key)

        self._log(f'✅ 加载现有数据: {len(existing_rows)}行')
        self._log(f'   现有实验唯一键: {len(existing_keys)}个')
        self._log('')

        # 遍历session目录
        new_experiments = []

        for exp_dir in sorted(self.session_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name in ['__pycache__', '.git']:
                continue

            self.stats['total_found'] += 1

            # 加载experiment.json
            exp_data = self._load_experiment_json(exp_dir)
            if not exp_data:
                self._log(f'⚠️  跳过 {exp_dir.name}: 无experiment.json')
                self.stats['skipped_no_json'] += 1
                continue

            # 检查是否重复
            if self._is_duplicate(exp_data, existing_keys):
                exp_id = exp_data.get('experiment_id', '')
                self._log(f'⚠️  跳过 {exp_dir.name}: 重复实验 ({exp_id})')
                self.stats['skipped_duplicate'] += 1
                continue

            # 获取log_patterns
            # 对于并行模式，repository和model在foreground中
            if exp_data.get('mode') == 'parallel':
                repo = exp_data.get('foreground', {}).get('repository')
                model = exp_data.get('foreground', {}).get('model')
            else:
                repo = exp_data.get('repository')
                model = exp_data.get('model')

            if repo not in self.models_config:
                self._log(f'⚠️  跳过 {exp_dir.name}: 仓库配置未找到 ({repo})')
                self.stats['skipped_unknown_repo'] += 1
                continue

            log_patterns = self.models_config[repo].get('performance_metrics', {}).get('log_patterns', {})

            # 提取性能数据
            terminal_output = exp_dir / 'terminal_output.txt'
            perf_metrics = self._extract_performance_from_terminal_output(terminal_output, log_patterns)

            # 构建行数据
            row = self._build_row_from_experiment(exp_data, perf_metrics, fieldnames)
            new_experiments.append(row)
            self.stats['added'] += 1

            exp_id = exp_data.get('experiment_id', '')
            self._log(f'✅ {exp_dir.name}:')
            self._log(f'   实验ID: {exp_id}')
            self._log(f'   训练成功: {row["training_success"]}')
            self._log(f'   性能指标: {list(perf_metrics.keys()) if perf_metrics else "无"}')
            self._log('')

        return new_experiments, existing_rows, fieldnames

    def append_to_raw_data(self, new_experiments, existing_rows, fieldnames):
        """追加新实验到raw_data.csv"""
        if not new_experiments:
            self._log('⚠️  未找到新实验，无需更新')
            return False

        self._log(f'=== 总结 ===')
        self._log(f'找到目录: {self.stats["total_found"]}个')
        self._log(f'跳过（无JSON）: {self.stats["skipped_no_json"]}个')
        self._log(f'跳过（未知仓库）: {self.stats["skipped_unknown_repo"]}个')
        self._log(f'跳过（重复）: {self.stats["skipped_duplicate"]}个')
        self._log(f'新增实验: {self.stats["added"]}个')
        self._log('')

        if self.dry_run:
            self._log('🔍 [DRY RUN] 测试运行，不实际写入文件')
            self._log(f'   将添加 {len(new_experiments)} 个实验到 raw_data.csv')
            return True

        # 备份
        if self.create_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.raw_data_csv.parent / f'raw_data.csv.backup_{timestamp}'
            import shutil
            shutil.copy(self.raw_data_csv, backup_path)
            self._log(f'✅ 已备份: {backup_path}')

        # 追加新实验
        all_rows = existing_rows + new_experiments

        with open(self.raw_data_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)

        self._log(f'✅ 已更新: {self.raw_data_csv}')
        self._log(f'   原始: {len(existing_rows)}行')
        self._log(f'   新增: {len(new_experiments)}行')
        self._log(f'   总计: {len(all_rows)}行')
        self._log('')

        # 验证
        with open(self.raw_data_csv, 'r') as f:
            reader = csv.DictReader(f)
            final_rows = list(reader)

        self._log(f'✅ 验证: {len(final_rows)}行 (预期{len(all_rows)}行)')

        if len(final_rows) == len(all_rows):
            self._log('✅ 数据完整性验证通过')
            return True
        else:
            self._log('❌ 数据完整性验证失败')
            return False

    def run(self):
        """执行完整流程"""
        try:
            new_experiments, existing_rows, fieldnames = self.extract_experiments()
            success = self.append_to_raw_data(new_experiments, existing_rows, fieldnames)

            self._log('')
            self._log('=' * 80)
            if success:
                self._log('✅ 完成')
            else:
                self._log('⚠️  完成（有警告）')
            self._log('=' * 80)

            return success

        except Exception as e:
            self._log(f'❌ 错误: {e}')
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从session目录提取实验数据并追加到raw_data.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 追加最新session的实验
  python3 tools/data_management/append_session_to_raw_data.py results/run_20251213_100000

  # 测试运行（不实际写入）
  python3 tools/data_management/append_session_to_raw_data.py results/run_20251213_100000 --dry-run

  # 不创建备份
  python3 tools/data_management/append_session_to_raw_data.py results/run_20251213_100000 --no-backup

  # 静默模式
  python3 tools/data_management/append_session_to_raw_data.py results/run_20251213_100000 --quiet
"""
    )

    parser.add_argument('session_dir', type=str,
                        help='Session目录路径 (例如: results/run_20251213_100000)')
    parser.add_argument('--raw-data-csv', type=str, default=None,
                        help=f'raw_data.csv路径 (默认: {DEFAULT_RAW_DATA_CSV})')
    parser.add_argument('--models-config', type=str, default=None,
                        help=f'models_config.json路径 (默认: {DEFAULT_MODELS_CONFIG})')
    parser.add_argument('--dry-run', action='store_true',
                        help='测试运行，不实际写入文件')
    parser.add_argument('--no-backup', action='store_true',
                        help='不创建备份文件')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，减少输出')

    args = parser.parse_args()

    # 创建追加器
    appender = SessionDataAppender(
        session_dir=args.session_dir,
        raw_data_csv=args.raw_data_csv,
        models_config_path=args.models_config,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
        verbose=not args.quiet
    )

    # 运行
    success = appender.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
