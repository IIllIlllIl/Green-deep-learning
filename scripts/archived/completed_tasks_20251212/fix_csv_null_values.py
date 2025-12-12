#!/usr/bin/env python3
"""
CSV空值修复脚本 - 修订版

根据用户需求修改:
1. experiment_source: 从results目录下的实际文件夹名读取
2. mutated_param: 新增列，通过比较超参数与默认值识别变异参数
3. 超参数默认值: 从models_config.json填充空值

日期: 2025-12-11
版本: v2.0
"""

import csv
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class CSVNullValueFixer:
    def __init__(self, csv_path: str, config_path: str, dry_run: bool = False):
        self.csv_path = csv_path
        self.config_path = config_path
        self.dry_run = dry_run
        self.models_config = {}
        self.stats = {
            'total_rows': 0,
            'experiment_source_fixed': 0,
            'mutated_param_added': 0,
            'hyperparams_filled': 0,
            'errors': []
        }

    def load_models_config(self) -> None:
        """加载模型配置文件"""
        print(f"📖 加载配置文件: {self.config_path}")
        with open(self.config_path, 'r') as f:
            data = json.load(f)
            self.models_config = data.get('models', {})
        print(f"✓ 加载了 {len(self.models_config)} 个模型配置\n")

    def get_default_hyperparams(self, repository: str, model: str) -> Dict[str, any]:
        """获取指定模型的默认超参数"""
        if repository not in self.models_config:
            return {}

        repo_config = self.models_config[repository]
        supported = repo_config.get('supported_hyperparams', {})

        defaults = {}
        for param, config in supported.items():
            default_value = config.get('default')
            if default_value is not None:
                defaults[param] = default_value

        return defaults

    def find_experiment_directory(self, experiment_id: str) -> Optional[str]:
        """
        根据experiment_id查找实际的目录名

        命名规则:
        - default__repo_model_001 -> repo_model_001
        - mutation_1x__repo_model_002 -> repo_model_002
        - repo_model_003 -> repo_model_003 (已经是目录名)
        """
        # 尝试去掉前缀
        possible_names = [experiment_id]

        if '__' in experiment_id:
            # 去掉 default__ 或 mutation_*__ 前缀
            folder_name = experiment_id.split('__', 1)[1]
            possible_names.append(folder_name)

        # 在results下所有run_*目录中查找
        for name in possible_names:
            pattern = f"results/run_*/{name}"
            matches = glob.glob(pattern)
            if matches:
                # 返回目录名（不含路径）
                return os.path.basename(matches[0])

        return None

    def extract_experiment_source_from_directory(self, folder_name: str, experiment_id: str) -> str:
        """
        从目录名推断experiment_source

        逻辑:
        - 如果experiment_id以"default__"开头 -> "default"
        - 如果experiment_id以"mutation_"开头 -> 提取mutation部分 (如"mutation_1x")
        - 如果目录名包含"_parallel" -> 检查是否并行实验
        - 否则 -> 空字符串（表示后期添加的实验）
        """
        if experiment_id.startswith('default__'):
            return 'default'
        elif experiment_id.startswith('mutation_'):
            # 提取mutation_1x, mutation_2x_safe等
            parts = experiment_id.split('__')
            if len(parts) >= 2:
                return parts[0]  # mutation_1x, mutation_2x_safe等

        # 对于没有前缀的，返回空字符串
        return ''

    def identify_mutated_param(self, row: Dict, defaults: Dict[str, any]) -> Optional[str]:
        """
        识别被变异的超参数

        逻辑: 比较实际值与默认值，找出唯一不同的参数
        """
        mutated_params = []

        for param, default_value in defaults.items():
            col_name = f'hyperparam_{param}'
            actual_value = row.get(col_name, '').strip()

            if not actual_value:
                continue

            # 类型转换并比较
            try:
                if isinstance(default_value, int):
                    if int(float(actual_value)) != default_value:
                        mutated_params.append(param)
                elif isinstance(default_value, float):
                    if abs(float(actual_value) - default_value) > 1e-9:
                        mutated_params.append(param)
                else:
                    if str(actual_value) != str(default_value):
                        mutated_params.append(param)
            except (ValueError, TypeError):
                # 无法转换，跳过比较
                pass

        # 只返回单参数变异的情况
        if len(mutated_params) == 1:
            return mutated_params[0]
        elif len(mutated_params) > 1:
            # 多参数变异，记录警告
            return f"MULTIPLE:[{','.join(mutated_params)}]"

        return None

    def fill_default_hyperparams(self, row: Dict, defaults: Dict[str, any]) -> int:
        """填充空的超参数为默认值"""
        filled = 0

        for param, default_value in defaults.items():
            col_name = f'hyperparam_{param}'
            if col_name in row and not row[col_name].strip():
                # 只填充空值
                row[col_name] = str(default_value)
                filled += 1

        return filled

    def process_csv(self) -> List[Dict]:
        """处理CSV文件"""
        print(f"📊 读取CSV文件: {self.csv_path}")

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        self.stats['total_rows'] = len(rows)
        print(f"✓ 读取了 {len(rows)} 行数据")
        print(f"✓ 当前列数: {len(fieldnames)}\n")

        # 添加mutated_param列（如果不存在）
        if 'mutated_param' not in fieldnames:
            fieldnames = list(fieldnames)
            # 在experiment_source后面插入
            if 'experiment_source' in fieldnames:
                idx = fieldnames.index('experiment_source') + 1
                fieldnames.insert(idx, 'mutated_param')
            else:
                fieldnames.append('mutated_param')
            print(f"✓ 新增列: mutated_param\n")

        print("🔧 开始处理数据...")
        print("="*70)

        # 处理每一行
        for i, row in enumerate(rows, 1):
            exp_id = row['experiment_id']
            repo = row['repository']
            model = row['model']

            if i <= 5 or i % 50 == 0:
                print(f"\n[{i}/{len(rows)}] {exp_id}")

            # 1. 修复experiment_source（从目录名推断）
            current_source = row.get('experiment_source', '').strip()
            folder_name = self.find_experiment_directory(exp_id)

            if folder_name:
                new_source = self.extract_experiment_source_from_directory(folder_name, exp_id)
                if new_source != current_source:
                    if i <= 5 or i % 50 == 0:
                        print(f"  experiment_source: '{current_source}' -> '{new_source}'")
                    row['experiment_source'] = new_source
                    self.stats['experiment_source_fixed'] += 1
            else:
                # 找不到目录，保持原样
                if not current_source and (i <= 5 or i % 50 == 0):
                    print(f"  ⚠️  找不到实验目录")

            # 2. 识别变异参数
            defaults = self.get_default_hyperparams(repo, model)
            if defaults:
                mutated = self.identify_mutated_param(row, defaults)
                if mutated:
                    row['mutated_param'] = mutated
                    self.stats['mutated_param_added'] += 1
                    if i <= 5 or i % 50 == 0:
                        print(f"  mutated_param: {mutated}")
                else:
                    row['mutated_param'] = ''
            else:
                row['mutated_param'] = ''
                if i <= 5 or i % 50 == 0:
                    print(f"  ⚠️  模型 {repo}/{model} 无配置")

            # 3. 填充默认超参数
            if defaults:
                filled = self.fill_default_hyperparams(row, defaults)
                if filled > 0:
                    self.stats['hyperparams_filled'] += filled
                    if i <= 5 or i % 50 == 0:
                        print(f"  填充了 {filled} 个默认超参数")

        print("\n" + "="*70)
        return rows, fieldnames

    def save_csv(self, rows: List[Dict], fieldnames: List[str]) -> None:
        """保存修复后的CSV"""
        if self.dry_run:
            print("\n🔍 DRY-RUN模式：不实际写入文件")
            return

        output_path = self.csv_path
        print(f"\n💾 写入修复后的CSV: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ 已保存 {len(rows)} 行数据")

    def print_summary(self) -> None:
        """打印修复总结"""
        print("\n" + "="*70)
        print("📈 修复总结")
        print("="*70)
        print(f"总行数:              {self.stats['total_rows']}")
        print(f"修复experiment_source: {self.stats['experiment_source_fixed']}")
        print(f"添加mutated_param:    {self.stats['mutated_param_added']}")
        print(f"填充默认超参数:        {self.stats['hyperparams_filled']}")

        if self.stats['errors']:
            print(f"\n⚠️  错误数: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")

        print("="*70)

    def run(self) -> bool:
        """执行修复流程"""
        try:
            # 1. 加载配置
            self.load_models_config()

            # 2. 处理CSV
            rows, fieldnames = self.process_csv()

            # 3. 保存结果
            self.save_csv(rows, fieldnames)

            # 4. 打印总结
            self.print_summary()

            return True

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='修复CSV空值')
    parser.add_argument('--input', default='results/summary_all.csv',
                      help='输入CSV文件路径')
    parser.add_argument('--config', default='mutation/models_config.json',
                      help='模型配置文件路径')
    parser.add_argument('--dry-run', action='store_true',
                      help='预览模式，不实际修改文件')

    args = parser.parse_args()

    print("="*70)
    print("CSV空值修复脚本 v2.0")
    print("="*70)
    print(f"输入文件: {args.input}")
    print(f"配置文件: {args.config}")
    print(f"模式: {'DRY-RUN（预览）' if args.dry_run else '实际执行'}")
    print("="*70 + "\n")

    fixer = CSVNullValueFixer(args.input, args.config, args.dry_run)
    success = fixer.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
