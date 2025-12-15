#!/usr/bin/env python3
"""
回归测试：并行模式数据提取

测试目标：
1. 验证append_session_to_raw_data.py能正确处理并行实验的experiment.json
2. 防止并行模式数据提取bug重现（Phase 4发现的问题）

问题历史：
- 2025-12-14: 发现append_session_to_raw_data.py无法提取并行实验数据
- 原因: 并行模式的experiment.json结构不同（repository/model在foreground中）
- 修复: 区分并行/非并行模式，从foreground子对象提取数据

测试方法：
- 创建模拟的experiment.json（并行和非并行模式）
- 调用_build_row_from_experiment方法
- 验证提取的数据正确性
"""

import unittest
import sys
import os
import tempfile
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入待测试的类
from scripts.append_session_to_raw_data import SessionDataAppender


class TestParallelDataExtraction(unittest.TestCase):
    """并行模式数据提取测试"""

    def setUp(self):
        """测试初始化"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.raw_data_csv = Path(self.temp_dir) / 'raw_data.csv'
        self.models_config = Path(self.temp_dir) / 'models_config.json'

        # 创建模拟的models_config.json
        models_config_data = {
            "models": {
                "VulBERTa": {
                    "mlp": {
                        "performance_metrics": {
                            "log_patterns": {
                                "eval_loss": r"eval_loss:\s*([\d.]+)",
                                "accuracy": r"accuracy:\s*([\d.]+)"
                            }
                        }
                    }
                },
                "MRT-OAST": {
                    "default": {
                        "performance_metrics": {
                            "log_patterns": {
                                "accuracy": r"accuracy:\s*([\d.]+)",
                                "f1": r"f1:\s*([\d.]+)"
                            }
                        }
                    }
                },
                "examples": {
                    "mnist": {
                        "performance_metrics": {
                            "log_patterns": {}
                        }
                    }
                }
            }
        }
        with open(self.models_config, 'w') as f:
            json.dump(models_config_data, f)

        # 创建空的raw_data.csv（仅表头）
        with open(self.raw_data_csv, 'w') as f:
            f.write('experiment_id,timestamp,repository,model,training_success,duration_seconds,retries,'
                   'hyperparam_learning_rate,hyperparam_dropout,hyperparam_epochs,hyperparam_seed,'
                   'perf_accuracy,perf_eval_loss,perf_f1,'
                   'energy_cpu_total_joules,energy_gpu_total_joules,'
                   'energy_gpu_avg_watts,energy_gpu_max_watts,energy_gpu_min_watts,'
                   'energy_gpu_temp_avg_celsius,energy_gpu_temp_max_celsius,'
                   'energy_gpu_util_avg_percent,energy_gpu_util_max_percent,'
                   'experiment_source,num_mutated_params,mutated_param,mode,error_message\n')

        # 创建appender实例
        self.appender = SessionDataAppender(
            session_dir=self.temp_dir,
            raw_data_csv=self.raw_data_csv,
            models_config_path=self.models_config,
            verbose=False
        )

        # 标准fieldnames
        self.fieldnames = [
            'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
            'duration_seconds', 'retries',
            'hyperparam_learning_rate', 'hyperparam_dropout', 'hyperparam_epochs', 'hyperparam_seed',
            'perf_accuracy', 'perf_eval_loss', 'perf_f1',
            'energy_cpu_total_joules', 'energy_gpu_total_joules',
            'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
            'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
            'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',
            'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message'
        ]

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_nonparallel_data_extraction(self):
        """测试1：非并行模式数据提取"""
        # 模拟非并行实验的experiment.json
        exp_data = {
            "experiment_id": "VulBERTa_mlp_001",
            "timestamp": "2025-12-14T10:00:00.000000",
            "repository": "VulBERTa",
            "model": "mlp",
            "hyperparameters": {
                "learning_rate": 3e-05,
                "epochs": 10
            },
            "duration_seconds": 3136.0,
            "energy_consumption": {
                "cpu_total_joules": 98529.22,
                "gpu_total_joules": 749138.76,
                "gpu_avg_watts": 245.38,
                "gpu_max_watts": 319.4,
                "gpu_min_watts": 4.13
            },
            "training_success": True,
            "retries": 0,
            "error_message": "Training completed successfully"
        }

        # 提取数据
        row = self.appender._build_row_from_experiment(exp_data, {}, self.fieldnames)

        # 验证基础字段
        self.assertEqual(row['experiment_id'], 'VulBERTa_mlp_001')
        self.assertEqual(row['repository'], 'VulBERTa')
        self.assertEqual(row['model'], 'mlp')
        self.assertEqual(row['training_success'], 'True')
        self.assertEqual(row['duration_seconds'], '3136.0')

        # 验证超参数
        self.assertEqual(row['hyperparam_learning_rate'], '3e-05')
        self.assertEqual(row['hyperparam_epochs'], '10')

        # 验证能耗数据
        self.assertEqual(row['energy_cpu_total_joules'], '98529.22')
        self.assertEqual(row['energy_gpu_total_joules'], '749138.76')
        self.assertEqual(row['energy_gpu_avg_watts'], '245.38')

    def test_parallel_data_extraction(self):
        """测试2：并行模式数据提取（核心测试）"""
        # 模拟并行实验的experiment.json（真实结构）
        exp_data = {
            "experiment_id": "VulBERTa_mlp_012_parallel",
            "timestamp": "2025-12-14T06:41:39.489342",
            "mode": "parallel",
            "foreground": {
                "repository": "VulBERTa",
                "model": "mlp",
                "hyperparameters": {
                    "learning_rate": 3e-05
                },
                "duration_seconds": 3760.08,
                "energy_metrics": {
                    "cpu_energy_pkg_joules": 135011.86,
                    "cpu_energy_ram_joules": 9256.11,
                    "cpu_energy_total_joules": 144267.97,
                    "gpu_power_avg_watts": 239.44,
                    "gpu_power_max_watts": 319.12,
                    "gpu_power_min_watts": 87.65,
                    "gpu_energy_total_joules": 877323.23,
                    "gpu_temp_avg_celsius": 79.47,
                    "gpu_temp_max_celsius": 84.0,
                    "gpu_util_avg_percent": 91.41,
                    "gpu_util_max_percent": 100.0
                },
                "performance_metrics": {
                    "eval_loss": 0.6839,
                    "final_training_loss": 0.7466
                },
                "training_success": True,
                "retries": 0,
                "error_message": "Training completed successfully"
            },
            "background": {
                "repository": "examples",
                "model": "mnist",
                "hyperparameters": {}
            }
        }

        # 提取数据
        row = self.appender._build_row_from_experiment(exp_data, {}, self.fieldnames)

        # 验证基础字段（从foreground提取）
        self.assertEqual(row['experiment_id'], 'VulBERTa_mlp_012_parallel')
        self.assertEqual(row['repository'], 'VulBERTa',
                        "并行模式应从foreground提取repository")
        self.assertEqual(row['model'], 'mlp',
                        "并行模式应从foreground提取model")
        self.assertEqual(row['training_success'], 'True')
        self.assertEqual(row['duration_seconds'], '3760.08')
        self.assertEqual(row['mode'], 'parallel')

        # 验证超参数（从foreground.hyperparameters提取）
        self.assertEqual(row['hyperparam_learning_rate'], '3e-05',
                        "并行模式应从foreground.hyperparameters提取超参数")

        # 验证能耗数据（从foreground.energy_metrics提取，需要字段名映射）
        self.assertEqual(row['energy_cpu_total_joules'], '144267.97',
                        "并行模式应从foreground.energy_metrics提取能耗")
        self.assertEqual(row['energy_gpu_total_joules'], '877323.23')
        self.assertEqual(row['energy_gpu_avg_watts'], '239.44')
        self.assertEqual(row['energy_gpu_max_watts'], '319.12')
        self.assertEqual(row['energy_gpu_min_watts'], '87.65')
        self.assertEqual(row['energy_gpu_temp_avg_celsius'], '79.47')
        self.assertEqual(row['energy_gpu_util_avg_percent'], '91.41')

        # 验证性能数据（从foreground.performance_metrics提取，需要字段名映射）
        self.assertEqual(row['perf_eval_loss'], '0.6839',
                        "并行模式应从foreground.performance_metrics提取性能指标")

    def test_parallel_vs_nonparallel_mode_detection(self):
        """测试3：并行/非并行模式检测"""
        # 非并行模式
        nonpar_data = {
            "experiment_id": "test_001",
            "timestamp": "2025-12-14T10:00:00",
            "repository": "VulBERTa",
            "model": "mlp",
            "hyperparameters": {},
            "energy_consumption": {},
            "training_success": True,
            "retries": 0
        }

        row_nonpar = self.appender._build_row_from_experiment(nonpar_data, {}, self.fieldnames)
        self.assertEqual(row_nonpar['repository'], 'VulBERTa',
                        "非并行模式应从顶层提取repository")
        self.assertEqual(row_nonpar['mode'], '')

        # 并行模式
        par_data = {
            "experiment_id": "test_002_parallel",
            "timestamp": "2025-12-14T10:00:00",
            "mode": "parallel",
            "foreground": {
                "repository": "MRT-OAST",
                "model": "default",
                "hyperparameters": {},
                "energy_metrics": {},
                "training_success": True,
                "retries": 0
            },
            "background": {
                "repository": "examples",
                "model": "mnist"
            }
        }

        row_par = self.appender._build_row_from_experiment(par_data, {}, self.fieldnames)
        self.assertEqual(row_par['repository'], 'MRT-OAST',
                        "并行模式应从foreground提取repository")
        self.assertEqual(row_par['mode'], 'parallel')

    def test_energy_metrics_field_mapping(self):
        """测试4：能耗字段名映射（并行模式特有）"""
        # 并行模式的energy_metrics字段名与CSV列名不同，需要映射
        exp_data = {
            "experiment_id": "test_parallel",
            "timestamp": "2025-12-14T10:00:00",
            "mode": "parallel",
            "foreground": {
                "repository": "MRT-OAST",
                "model": "default",
                "hyperparameters": {},
                "energy_metrics": {
                    "cpu_energy_pkg_joules": 56957.55,
                    "cpu_energy_ram_joules": 3701.46,
                    "cpu_energy_total_joules": 60659.01,
                    "gpu_power_avg_watts": 239.11,      # 注意：power vs energy
                    "gpu_power_max_watts": 315.76,
                    "gpu_power_min_watts": 89.74,
                    "gpu_energy_total_joules": 364647.02,
                    "gpu_temp_avg_celsius": 80.47,
                    "gpu_temp_max_celsius": 84.0,
                    "gpu_util_avg_percent": 94.42,
                    "gpu_util_max_percent": 100.0
                },
                "training_success": True,
                "retries": 0
            },
            "background": {}
        }

        row = self.appender._build_row_from_experiment(exp_data, {}, self.fieldnames)

        # 验证字段名正确映射
        mappings = [
            ('energy_cpu_total_joules', '60659.01'),
            ('energy_gpu_total_joules', '364647.02'),
            ('energy_gpu_avg_watts', '239.11'),  # gpu_power_avg_watts -> energy_gpu_avg_watts
            ('energy_gpu_max_watts', '315.76'),
            ('energy_gpu_min_watts', '89.74'),
            ('energy_gpu_temp_avg_celsius', '80.47'),
            ('energy_gpu_temp_max_celsius', '84.0'),
            ('energy_gpu_util_avg_percent', '94.42'),
            ('energy_gpu_util_max_percent', '100.0')
        ]

        for csv_field, expected_value in mappings:
            self.assertEqual(row[csv_field], expected_value,
                           f"字段 {csv_field} 映射错误")

    def test_performance_metrics_field_mapping(self):
        """测试5：性能指标字段名映射（并行模式）"""
        exp_data = {
            "experiment_id": "test_parallel",
            "timestamp": "2025-12-14T10:00:00",
            "mode": "parallel",
            "foreground": {
                "repository": "VulBERTa",
                "model": "mlp",
                "hyperparameters": {},
                "energy_metrics": {},
                "performance_metrics": {
                    "eval_loss": 0.6839,
                    "final_training_loss": 0.7466,
                    "accuracy": 0.85,
                    "f1": 0.82
                },
                "training_success": True,
                "retries": 0
            },
            "background": {}
        }

        row = self.appender._build_row_from_experiment(exp_data, {}, self.fieldnames)

        # 验证性能指标映射
        mappings = [
            ('perf_eval_loss', '0.6839'),
            ('perf_accuracy', '0.85'),
            ('perf_f1', '0.82')
        ]

        for csv_field, expected_value in mappings:
            self.assertEqual(row[csv_field], expected_value,
                           f"性能指标 {csv_field} 映射错误")

    def test_missing_foreground_data_handling(self):
        """测试6：处理缺失foreground数据的情况"""
        # 并行模式但foreground为空（异常情况）
        exp_data = {
            "experiment_id": "test_broken_parallel",
            "timestamp": "2025-12-14T10:00:00",
            "mode": "parallel",
            "foreground": {},  # 空的foreground
            "background": {}
        }

        row = self.appender._build_row_from_experiment(exp_data, {}, self.fieldnames)

        # 应该返回空值而不是崩溃
        self.assertEqual(row['repository'], '')
        self.assertEqual(row['model'], '')
        self.assertEqual(row['mode'], 'parallel')


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestParallelDataExtraction)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
