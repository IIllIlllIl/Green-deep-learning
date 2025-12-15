#!/usr/bin/env python3
"""
测试套件：append_session_to_raw_data.py

测试覆盖：
1. 基本功能测试 - 提取和追加新实验
2. 去重测试 - 跳过已存在的实验
3. 错误处理 - 缺失文件、损坏数据
4. 备份功能 - 验证备份创建
5. Dry-run模式 - 测试运行不写入
6. 数据完整性 - 验证所有字段正确填充
7. 性能数据提取 - terminal_output.txt解析

版本：1.0
创建日期：2025-12-13
"""

import unittest
import tempfile
import shutil
import json
import csv
from pathlib import Path
import sys

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from append_session_to_raw_data import SessionDataAppender


class TestSessionDataAppender(unittest.TestCase):
    """SessionDataAppender测试类"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.test_dir = Path(tempfile.mkdtemp())

        # 创建测试用的models_config.json
        self.models_config = self.test_dir / 'models_config.json'
        self._create_test_models_config()

        # 创建测试用的raw_data.csv
        self.raw_data_csv = self.test_dir / 'raw_data.csv'
        self._create_test_raw_data_csv()

        # 创建测试session目录
        self.session_dir = self.test_dir / 'run_20251213_test'
        self.session_dir.mkdir()

    def tearDown(self):
        """清理测试环境"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_models_config(self):
        """创建测试用的models_config.json"""
        config = {
            "models": {
                "examples": {
                    "performance_metrics": {
                        "log_patterns": {
                            "test_accuracy": "Test Accuracy[:\\s]+([0-9.]+)",
                            "train_accuracy": "Train Accuracy[:\\s]+([0-9.]+)"
                        }
                    }
                },
                "VulBERTa": {
                    "performance_metrics": {
                        "log_patterns": {
                            "accuracy": "Accuracy[:\\s]+([0-9.]+)",
                            "f1": "F1[:\\s]+([0-9.]+)"
                        }
                    }
                }
            }
        }

        with open(self.models_config, 'w') as f:
            json.dump(config, f)

    def _create_test_raw_data_csv(self):
        """创建测试用的raw_data.csv（带有1个现有实验）"""
        fieldnames = [
            'experiment_id', 'timestamp', 'repository', 'model',
            'training_success', 'duration_seconds', 'retries',
            'experiment_source', 'num_mutated_params', 'mutated_param',
            'mode', 'error_message',
            'hyperparam_learning_rate', 'hyperparam_epochs',
            'energy_cpu_total_joules', 'energy_gpu_total_joules',
            'perf_test_accuracy', 'perf_train_accuracy',
            'perf_accuracy', 'perf_f1'
        ]

        # 创建一个现有实验（用于测试去重）
        existing_row = {
            'experiment_id': 'existing_exp_001',
            'timestamp': '2025-12-13T10:00:00',  # 固定时间戳
            'repository': 'examples',
            'model': 'mnist',
            'training_success': 'True',
            'duration_seconds': '100',
            'retries': '0',
            'experiment_source': 'test',
            'num_mutated_params': '1',
            'mutated_param': 'learning_rate',
            'mode': 'mutation',
            'error_message': '',
            'hyperparam_learning_rate': '0.001',
            'hyperparam_epochs': '10',
            'energy_cpu_total_joules': '1000',
            'energy_gpu_total_joules': '2000',
            'perf_test_accuracy': '95.5',
            'perf_train_accuracy': '98.0',
            'perf_accuracy': '',
            'perf_f1': ''
        }

        with open(self.raw_data_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(existing_row)

    def _create_experiment_dir(self, exp_id, repo, model, has_terminal=True,
                               training_success=True, perf_data=None):
        """创建测试用的实验目录"""
        exp_dir = self.session_dir / f'{repo}_{model}_{exp_id}'
        exp_dir.mkdir()

        # 创建experiment.json
        exp_json = {
            'experiment_id': exp_id,
            'timestamp': '2025-12-13T12:00:00',
            'repository': repo,
            'model': model,
            'training_success': training_success,
            'duration_seconds': 200,
            'retries': 0,
            'experiment_source': 'test',
            'num_mutated_params': 1,
            'mutated_param': 'learning_rate',
            'mode': 'mutation',
            'error_message': '' if training_success else 'Test error',
            'hyperparameters': {
                'learning_rate': 0.002,
                'epochs': 20
            },
            'energy_consumption': {
                'cpu_total_joules': 1500,
                'gpu_total_joules': 3000
            }
        }

        with open(exp_dir / 'experiment.json', 'w') as f:
            json.dump(exp_json, f)

        # 创建terminal_output.txt（如果需要）
        if has_terminal and perf_data:
            terminal_output = exp_dir / 'terminal_output.txt'
            content_lines = []

            if repo == 'examples':
                if 'test_accuracy' in perf_data:
                    content_lines.append(f"Test Accuracy: {perf_data['test_accuracy']}")
                if 'train_accuracy' in perf_data:
                    content_lines.append(f"Train Accuracy: {perf_data['train_accuracy']}")
            elif repo == 'VulBERTa':
                if 'accuracy' in perf_data:
                    content_lines.append(f"Accuracy: {perf_data['accuracy']}")
                if 'f1' in perf_data:
                    content_lines.append(f"F1: {perf_data['f1']}")

            with open(terminal_output, 'w') as f:
                f.write('\n'.join(content_lines))

        return exp_dir

    # ==================== 测试用例 ====================

    def test_01_basic_extraction(self):
        """测试1: 基本提取功能"""
        print("\n=== 测试1: 基本提取功能 ===")

        # 创建一个新实验
        self._create_experiment_dir(
            'new_exp_001', 'examples', 'mnist',
            has_terminal=True,
            perf_data={'test_accuracy': 96.5, 'train_accuracy': 99.0}
        )

        # 运行追加器（dry-run模式）
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证
        self.assertEqual(len(new_exps), 1, "应该提取到1个新实验")
        self.assertEqual(appender.stats['added'], 1)
        self.assertEqual(appender.stats['skipped_duplicate'], 0)

        # 验证数据完整性
        exp = new_exps[0]
        self.assertEqual(exp['experiment_id'], 'new_exp_001')
        self.assertEqual(exp['repository'], 'examples')
        self.assertEqual(exp['model'], 'mnist')
        self.assertEqual(exp['perf_test_accuracy'], '96.5')
        self.assertEqual(exp['perf_train_accuracy'], '99.0')

        print("✅ 基本提取功能测试通过")

    def test_02_deduplication(self):
        """测试2: 去重功能（相同ID + 相同timestamp）"""
        print("\n=== 测试2: 去重功能（相同ID + 相同timestamp） ===")

        # 创建一个与现有实验ID和timestamp都相同的实验
        exp_dir = self.session_dir / 'examples_mnist_existing_exp_001'
        exp_dir.mkdir()

        # 使用相同的 ID 和 timestamp
        exp_json = {
            'experiment_id': 'existing_exp_001',
            'timestamp': '2025-12-13T10:00:00',  # 与现有实验相同
            'repository': 'examples',
            'model': 'mnist',
            'training_success': True,
            'duration_seconds': 200,
            'retries': 0,
            'experiment_source': 'test',
            'num_mutated_params': 1,
            'mutated_param': 'learning_rate',
            'mode': 'mutation',
            'error_message': '',
            'hyperparameters': {
                'learning_rate': 0.002,
                'epochs': 20
            },
            'energy_consumption': {
                'cpu_total_joules': 1500,
                'gpu_total_joules': 3000
            }
        }

        with open(exp_dir / 'experiment.json', 'w') as f:
            json.dump(exp_json, f)

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证
        self.assertEqual(len(new_exps), 0, "应该跳过重复实验（相同ID+timestamp）")
        self.assertEqual(appender.stats['skipped_duplicate'], 1)
        self.assertEqual(appender.stats['added'], 0)

        print("✅ 去重功能测试通过（相同ID+timestamp）")

    def test_02b_different_timestamp_same_id(self):
        """测试2b: 相同ID但不同timestamp - 应该被添加"""
        print("\n=== 测试2b: 相同ID但不同timestamp - 应该被添加 ===")

        # 创建一个与现有实验ID相同但timestamp不同的实验
        exp_dir = self.session_dir / 'examples_mnist_existing_exp_001_new'
        exp_dir.mkdir()

        # 使用相同的 ID 但不同的 timestamp
        exp_json = {
            'experiment_id': 'existing_exp_001',  # 相同ID
            'timestamp': '2025-12-13T11:00:00',   # 不同timestamp
            'repository': 'examples',
            'model': 'mnist',
            'training_success': True,
            'duration_seconds': 200,
            'retries': 0,
            'experiment_source': 'test',
            'num_mutated_params': 1,
            'mutated_param': 'learning_rate',
            'mode': 'mutation',
            'error_message': '',
            'hyperparameters': {
                'learning_rate': 0.002,
                'epochs': 20
            },
            'energy_consumption': {
                'cpu_total_joules': 1500,
                'gpu_total_joules': 3000
            }
        }

        with open(exp_dir / 'experiment.json', 'w') as f:
            json.dump(exp_json, f)

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证 - 应该被添加，因为timestamp不同
        self.assertEqual(len(new_exps), 1, "应该添加实验（相同ID但不同timestamp）")
        self.assertEqual(appender.stats['skipped_duplicate'], 0)
        self.assertEqual(appender.stats['added'], 1)

        # 验证数据
        exp = new_exps[0]
        self.assertEqual(exp['experiment_id'], 'existing_exp_001')
        self.assertEqual(exp['timestamp'], '2025-12-13T11:00:00')

        print("✅ 相同ID不同timestamp测试通过（正确添加）")

    def test_03_multiple_experiments(self):
        """测试3: 多个实验提取"""
        print("\n=== 测试3: 多个实验提取 ===")

        # 创建3个新实验
        self._create_experiment_dir(
            'new_exp_002', 'examples', 'mnist',
            has_terminal=True,
            perf_data={'test_accuracy': 96.0}
        )
        self._create_experiment_dir(
            'new_exp_003', 'VulBERTa', 'mlp',
            has_terminal=True,
            perf_data={'accuracy': 88.5, 'f1': 0.87}
        )
        self._create_experiment_dir(
            'new_exp_004', 'examples', 'mnist_rnn',
            has_terminal=True,
            perf_data={'test_accuracy': 94.0}
        )

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证
        self.assertEqual(len(new_exps), 3, "应该提取到3个新实验")
        self.assertEqual(appender.stats['added'], 3)

        print("✅ 多个实验提取测试通过")

    def test_04_missing_terminal_output(self):
        """测试4: 缺失terminal_output.txt"""
        print("\n=== 测试4: 缺失terminal_output.txt ===")

        # 创建没有terminal_output.txt的实验
        self._create_experiment_dir(
            'new_exp_005', 'examples', 'mnist',
            has_terminal=False
        )

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证 - 应该能提取实验，但性能数据为空
        self.assertEqual(len(new_exps), 1, "应该提取到实验（即使没有性能数据）")
        exp = new_exps[0]
        self.assertEqual(exp['experiment_id'], 'new_exp_005')
        self.assertEqual(exp['perf_test_accuracy'], '', "性能数据应为空")

        print("✅ 缺失terminal_output测试通过")

    def test_05_missing_experiment_json(self):
        """测试5: 缺失experiment.json"""
        print("\n=== 测试5: 缺失experiment.json ===")

        # 创建目录但不创建experiment.json
        exp_dir = self.session_dir / 'incomplete_exp'
        exp_dir.mkdir()

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证
        self.assertEqual(len(new_exps), 0, "应该跳过没有experiment.json的目录")
        self.assertEqual(appender.stats['skipped_no_json'], 1)

        print("✅ 缺失experiment.json测试通过")

    def test_06_unknown_repository(self):
        """测试6: 未知仓库"""
        print("\n=== 测试6: 未知仓库 ===")

        # 创建使用未知仓库的实验
        self._create_experiment_dir(
            'new_exp_006', 'unknown_repo', 'unknown_model',
            has_terminal=False
        )

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证
        self.assertEqual(len(new_exps), 0, "应该跳过未知仓库的实验")
        self.assertEqual(appender.stats['skipped_unknown_repo'], 1)

        print("✅ 未知仓库测试通过")

    def test_07_actual_write(self):
        """测试7: 实际写入文件"""
        print("\n=== 测试7: 实际写入文件 ===")

        # 创建新实验
        self._create_experiment_dir(
            'new_exp_007', 'examples', 'mnist',
            has_terminal=True,
            perf_data={'test_accuracy': 97.5}
        )

        # 记录原始行数
        with open(self.raw_data_csv, 'r') as f:
            reader = csv.DictReader(f)
            original_count = len(list(reader))

        # 运行追加器（实际写入）
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=False,
            create_backup=True,
            verbose=False
        )

        success = appender.run()

        # 验证
        self.assertTrue(success, "写入应该成功")

        # 验证文件内容
        with open(self.raw_data_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), original_count + 1, "应该增加1行")

        # 验证新增的行
        new_row = rows[-1]
        self.assertEqual(new_row['experiment_id'], 'new_exp_007')
        self.assertEqual(new_row['perf_test_accuracy'], '97.5')

        # 验证备份文件存在
        backup_files = list(self.test_dir.glob('raw_data.csv.backup_*'))
        self.assertGreater(len(backup_files), 0, "应该创建备份文件")

        print("✅ 实际写入文件测试通过")

    def test_08_no_backup_option(self):
        """测试8: 不创建备份选项"""
        print("\n=== 测试8: 不创建备份选项 ===")

        # 创建新实验
        self._create_experiment_dir(
            'new_exp_008', 'examples', 'mnist',
            has_terminal=False
        )

        # 清除已有备份
        for backup in self.test_dir.glob('raw_data.csv.backup_*'):
            backup.unlink()

        # 运行追加器（不创建备份）
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=False,
            create_backup=False,
            verbose=False
        )

        appender.run()

        # 验证没有备份文件
        backup_files = list(self.test_dir.glob('raw_data.csv.backup_*'))
        self.assertEqual(len(backup_files), 0, "不应该创建备份文件")

        print("✅ 不创建备份选项测试通过")

    def test_09_performance_data_extraction(self):
        """测试9: 性能数据提取准确性"""
        print("\n=== 测试9: 性能数据提取准确性 ===")

        # 创建带有多种性能指标的实验
        self._create_experiment_dir(
            'new_exp_009', 'VulBERTa', 'mlp',
            has_terminal=True,
            perf_data={'accuracy': 89.23, 'f1': 0.8945}
        )

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证性能数据
        exp = new_exps[0]
        self.assertEqual(exp['perf_accuracy'], '89.23')
        self.assertEqual(exp['perf_f1'], '0.8945')

        print("✅ 性能数据提取准确性测试通过")

    def test_10_mixed_scenario(self):
        """测试10: 混合场景（新、重复、缺失）"""
        print("\n=== 测试10: 混合场景 ===")

        # 创建多种情况的实验
        # 1. 新实验
        self._create_experiment_dir(
            'new_exp_010', 'examples', 'mnist',
            has_terminal=True,
            perf_data={'test_accuracy': 95.0}
        )

        # 2. 重复实验
        self._create_experiment_dir(
            'existing_exp_001', 'examples', 'mnist',
            has_terminal=True,
            perf_data={'test_accuracy': 96.0}
        )

        # 3. 缺失JSON
        incomplete_dir = self.session_dir / 'incomplete'
        incomplete_dir.mkdir()

        # 4. 未知仓库
        self._create_experiment_dir(
            'new_exp_011', 'unknown', 'model',
            has_terminal=False
        )

        # 5. 另一个新实验
        self._create_experiment_dir(
            'new_exp_012', 'VulBERTa', 'mlp',
            has_terminal=True,
            perf_data={'accuracy': 87.0}
        )

        # 运行追加器
        appender = SessionDataAppender(
            self.session_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            dry_run=True,
            verbose=False
        )

        new_exps, existing_rows, fieldnames = appender.extract_experiments()

        # 验证统计
        self.assertEqual(appender.stats['total_found'], 5)
        self.assertEqual(appender.stats['added'], 3)  # new_exp_010, existing_exp_001(不同timestamp), new_exp_012
        self.assertEqual(appender.stats['skipped_duplicate'], 0)  # 无重复（因为timestamp不同）
        self.assertEqual(appender.stats['skipped_no_json'], 1)  # incomplete
        self.assertEqual(appender.stats['skipped_unknown_repo'], 1)  # unknown

        print("✅ 混合场景测试通过")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSessionDataAppender)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
