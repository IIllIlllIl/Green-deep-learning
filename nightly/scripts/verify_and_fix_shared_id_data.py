#!/usr/bin/env python3
"""
验证并修复所有共享experiment_id的数据行

功能：
1. 查找所有共享experiment_id的数据行
2. 通过时间戳匹配对应的JSON和terminal_output文件
3. 验证每行的能耗指标、性能指标、超参数
4. 生成详细的验证报告
5. 修复错误的数据

用法: python3 scripts/verify_and_fix_shared_id_data.py

版本: 1.0
创建日期: 2025-12-15
"""

import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil

class DataVerifier:
    """数据验证器"""

    def __init__(self, csv_path='results/raw_data.csv'):
        self.csv_path = Path(csv_path)
        self.results_dir = Path('results')
        self.rows = []
        self.fieldnames = []
        self.issues = []
        self.stats = {
            'total_rows': 0,
            'shared_id_rows': 0,
            'verified_ok': 0,
            'energy_mismatch': 0,
            'performance_mismatch': 0,
            'hyperparam_mismatch': 0,
            'json_not_found': 0,
            'terminal_not_found': 0
        }

    def load_csv(self):
        """加载CSV数据"""
        print('=' * 80)
        print('加载CSV数据')
        print('=' * 80)
        print('')

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            self.rows = list(reader)

        self.stats['total_rows'] = len(self.rows)
        print(f'总数据行数: {len(self.rows)}')
        print(f'列数: {len(self.fieldnames)}')
        print('')

    def find_shared_ids(self):
        """查找所有共享experiment_id的数据行"""
        print('=' * 80)
        print('查找共享experiment_id的数据行')
        print('=' * 80)
        print('')

        by_exp_id = defaultdict(list)
        for i, row in enumerate(self.rows):
            exp_id = row.get('experiment_id', '')
            by_exp_id[exp_id].append((i, row))

        shared_ids = {exp_id: rows for exp_id, rows in by_exp_id.items()
                     if len(rows) > 1}

        self.stats['shared_id_rows'] = sum(len(rows) for rows in shared_ids.values())

        print(f'发现 {len(shared_ids)} 个共享experiment_id')
        print(f'涉及 {self.stats["shared_id_rows"]} 行数据')
        print('')

        return shared_ids

    def find_experiment_files(self, exp_id, timestamp):
        """通过experiment_id和timestamp查找对应的实验文件"""
        json_path = None
        terminal_path = None
        json_data = None

        # 遍历所有session目录
        for session_dir in sorted(self.results_dir.glob('run_*')):
            if not session_dir.is_dir():
                continue

            # 查找匹配experiment_id的目录
            for exp_dir in session_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                # 检查是否包含experiment_id
                if exp_id not in exp_dir.name:
                    continue

                # 检查JSON文件
                json_file = exp_dir / 'experiment.json'
                if not json_file.exists():
                    continue

                # 读取JSON验证时间戳
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    if data.get('timestamp') == timestamp:
                        json_path = json_file
                        json_data = data

                        # 查找terminal_output.txt
                        terminal_file = exp_dir / 'terminal_output.txt'
                        if terminal_file.exists():
                            terminal_path = terminal_file

                        return json_path, terminal_path, json_data
                except:
                    continue

        return None, None, None

    def extract_json_data(self, json_data):
        """从JSON中提取所有数据"""
        extracted = {
            'hyperparameters': {},
            'energy_metrics': {},
            'performance_metrics': {}
        }

        if not json_data:
            return extracted

        # 检查是否为并行模式
        mode = json_data.get('mode', '')

        if mode == 'parallel':
            # 从foreground提取
            fg = json_data.get('foreground', {})
            extracted['hyperparameters'] = fg.get('hyperparameters', {})
            extracted['energy_metrics'] = fg.get('energy_metrics', {})
            extracted['performance_metrics'] = fg.get('performance_metrics', {})
        else:
            # 从顶层提取
            extracted['hyperparameters'] = json_data.get('hyperparameters', {})
            extracted['energy_metrics'] = json_data.get('energy_metrics', {}) or json_data.get('energy_consumption', {})
            extracted['performance_metrics'] = json_data.get('performance_metrics', {})

        return extracted

    def compare_hyperparameters(self, csv_row, json_data):
        """比较超参数"""
        mismatches = []

        json_hyperparams = json_data['hyperparameters']

        for csv_key, csv_value in csv_row.items():
            if not csv_key.startswith('hyperparam_'):
                continue
            if not csv_value:
                continue

            param_name = csv_key.replace('hyperparam_', '')
            json_value = json_hyperparams.get(param_name)

            if json_value is None:
                continue

            # 转换为字符串比较
            csv_str = str(csv_value)
            json_str = str(json_value)

            if csv_str != json_str:
                # 尝试数值比较
                try:
                    if abs(float(csv_str) - float(json_str)) > 1e-6:
                        mismatches.append({
                            'param': param_name,
                            'csv_value': csv_str,
                            'json_value': json_str
                        })
                except:
                    mismatches.append({
                        'param': param_name,
                        'csv_value': csv_str,
                        'json_value': json_str
                    })

        return mismatches

    def compare_energy(self, csv_row, json_data):
        """比较能耗指标"""
        mismatches = []

        json_energy = json_data['energy_metrics']

        # 能耗字段映射
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

        for json_key, csv_key in energy_mapping.items():
            csv_value = csv_row.get(csv_key, '')
            json_value = json_energy.get(json_key)

            if not csv_value and json_value is None:
                continue

            if csv_value and json_value is None:
                mismatches.append({
                    'metric': csv_key,
                    'csv_value': csv_value,
                    'json_value': 'None',
                    'type': 'missing_in_json'
                })
                continue

            if not csv_value and json_value is not None:
                mismatches.append({
                    'metric': csv_key,
                    'csv_value': 'empty',
                    'json_value': str(json_value),
                    'type': 'missing_in_csv'
                })
                continue

            # 数值比较
            try:
                csv_num = float(csv_value)
                json_num = float(json_value)

                # 允许小的浮点误差
                if abs(csv_num - json_num) > max(abs(csv_num), abs(json_num)) * 0.001:
                    mismatches.append({
                        'metric': csv_key,
                        'csv_value': csv_value,
                        'json_value': str(json_value),
                        'type': 'value_mismatch'
                    })
            except:
                if str(csv_value) != str(json_value):
                    mismatches.append({
                        'metric': csv_key,
                        'csv_value': csv_value,
                        'json_value': str(json_value),
                        'type': 'value_mismatch'
                    })

        return mismatches

    def compare_performance(self, csv_row, json_data):
        """比较性能指标"""
        mismatches = []

        json_perf = json_data['performance_metrics']

        for csv_key, csv_value in csv_row.items():
            if not csv_key.startswith('perf_'):
                continue
            if not csv_value:
                continue

            metric_name = csv_key.replace('perf_', '')
            json_value = json_perf.get(metric_name)

            if json_value is None:
                # 可能从terminal_output提取，跳过
                continue

            # 数值比较
            try:
                csv_num = float(csv_value)
                json_num = float(json_value)

                # 允许小的浮点误差
                if abs(csv_num - json_num) > max(abs(csv_num), abs(json_num)) * 0.001:
                    mismatches.append({
                        'metric': metric_name,
                        'csv_value': csv_value,
                        'json_value': str(json_value),
                        'type': 'value_mismatch'
                    })
            except:
                if str(csv_value) != str(json_value):
                    mismatches.append({
                        'metric': metric_name,
                        'csv_value': csv_value,
                        'json_value': str(json_value),
                        'type': 'value_mismatch'
                    })

        return mismatches

    def verify_row(self, row_index, row, json_path, terminal_path, json_data):
        """验证单行数据"""
        verification = {
            'row_index': row_index,
            'experiment_id': row.get('experiment_id', ''),
            'timestamp': row.get('timestamp', ''),
            'json_path': str(json_path) if json_path else None,
            'terminal_path': str(terminal_path) if terminal_path else None,
            'has_json': json_path is not None,
            'has_terminal': terminal_path is not None,
            'hyperparam_mismatches': [],
            'energy_mismatches': [],
            'performance_mismatches': [],
            'status': 'ok'
        }

        if not json_data:
            verification['status'] = 'json_not_found'
            return verification

        # 提取JSON数据
        extracted = self.extract_json_data(json_data)

        # 比较超参数
        hp_mismatches = self.compare_hyperparameters(row, extracted)
        if hp_mismatches:
            verification['hyperparam_mismatches'] = hp_mismatches
            verification['status'] = 'mismatch'

        # 比较能耗
        energy_mismatches = self.compare_energy(row, extracted)
        if energy_mismatches:
            verification['energy_mismatches'] = energy_mismatches
            verification['status'] = 'mismatch'

        # 比较性能
        perf_mismatches = self.compare_performance(row, extracted)
        if perf_mismatches:
            verification['performance_mismatches'] = perf_mismatches
            verification['status'] = 'mismatch'

        return verification

    def verify_shared_ids(self, shared_ids):
        """验证所有共享ID的数据行"""
        print('=' * 80)
        print('验证共享ID的数据行')
        print('=' * 80)
        print('')

        all_verifications = []

        for exp_id, rows in sorted(shared_ids.items()):
            print(f'\n验证 experiment_id: {exp_id} ({len(rows)}行)')
            print('-' * 80)

            for row_index, row in rows:
                timestamp = row.get('timestamp', '')
                print(f'\n  行 {row_index + 2}: timestamp = {timestamp[:19]}...')

                # 查找文件
                json_path, terminal_path, json_data = self.find_experiment_files(exp_id, timestamp)

                if not json_path:
                    print(f'    ❌ 未找到JSON文件')
                    self.stats['json_not_found'] += 1
                    verification = {
                        'row_index': row_index,
                        'experiment_id': exp_id,
                        'timestamp': timestamp,
                        'status': 'json_not_found'
                    }
                    all_verifications.append(verification)
                    continue

                print(f'    ✅ JSON: {json_path.relative_to(self.results_dir)}')
                if terminal_path:
                    print(f'    ✅ Terminal: {terminal_path.relative_to(self.results_dir)}')
                else:
                    print(f'    ⚠️  Terminal: 未找到（历史实验）')

                # 验证数据
                verification = self.verify_row(row_index, row, json_path, terminal_path, json_data)
                all_verifications.append(verification)

                # 打印验证结果
                if verification['status'] == 'ok':
                    print(f'    ✅ 数据验证通过')
                    self.stats['verified_ok'] += 1
                else:
                    print(f'    ❌ 发现数据不匹配')

                    if verification['hyperparam_mismatches']:
                        print(f'       超参数不匹配: {len(verification["hyperparam_mismatches"])}个')
                        for m in verification['hyperparam_mismatches'][:2]:
                            print(f'         {m["param"]}: CSV={m["csv_value"]}, JSON={m["json_value"]}')
                        self.stats['hyperparam_mismatch'] += 1

                    if verification['energy_mismatches']:
                        print(f'       能耗指标不匹配: {len(verification["energy_mismatches"])}个')
                        for m in verification['energy_mismatches'][:2]:
                            print(f'         {m["metric"]}: CSV={m["csv_value"]}, JSON={m["json_value"]}')
                        self.stats['energy_mismatch'] += 1

                    if verification['performance_mismatches']:
                        print(f'       性能指标不匹配: {len(verification["performance_mismatches"])}个')
                        for m in verification['performance_mismatches'][:2]:
                            print(f'         {m["metric"]}: CSV={m["csv_value"]}, JSON={m["json_value"]}')
                        self.stats['performance_mismatch'] += 1

        return all_verifications

    def generate_report(self, verifications):
        """生成验证报告"""
        print('\n' + '=' * 80)
        print('验证报告')
        print('=' * 80)
        print('')

        print('统计信息:')
        print(f'  总数据行数: {self.stats["total_rows"]}')
        print(f'  共享ID行数: {self.stats["shared_id_rows"]}')
        print(f'  验证通过: {self.stats["verified_ok"]}')
        print(f'  JSON未找到: {self.stats["json_not_found"]}')
        print(f'  超参数不匹配: {self.stats["hyperparam_mismatch"]}')
        print(f'  能耗不匹配: {self.stats["energy_mismatch"]}')
        print(f'  性能不匹配: {self.stats["performance_mismatch"]}')
        print('')

        # 统计问题类型
        issues_by_type = defaultdict(list)
        for v in verifications:
            if v['status'] != 'ok':
                issues_by_type[v['status']].append(v)

        if issues_by_type:
            print('问题详情:')
            print('')

            for issue_type, issue_list in sorted(issues_by_type.items()):
                print(f'{issue_type}: {len(issue_list)}行')

            print('')

        return issues_by_type

    def fix_data(self, verifications):
        """修复错误数据"""
        print('=' * 80)
        print('修复错误数据')
        print('=' * 80)
        print('')

        # 统计需要修复的行
        rows_to_fix = [v for v in verifications if v['status'] == 'mismatch']

        if not rows_to_fix:
            print('✅ 没有需要修复的数据')
            return False

        print(f'发现 {len(rows_to_fix)} 行需要修复')
        print('')

        # 创建备份
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.csv_path.parent / f'raw_data.csv.backup_before_fix_{timestamp}'
        shutil.copy(self.csv_path, backup_path)
        print(f'✅ 已备份: {backup_path.name}')
        print('')

        # 修复数据
        fixed_count = 0

        for verification in rows_to_fix:
            row_index = verification['row_index']
            json_path = Path(verification['json_path'])

            # 重新读取JSON
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            extracted = self.extract_json_data(json_data)

            # 修复超参数
            for mismatch in verification.get('hyperparam_mismatches', []):
                param = mismatch['param']
                correct_value = mismatch['json_value']
                csv_key = f'hyperparam_{param}'

                if csv_key in self.rows[row_index]:
                    self.rows[row_index][csv_key] = correct_value

            # 修复能耗
            for mismatch in verification.get('energy_mismatches', []):
                if mismatch['type'] == 'missing_in_csv':
                    csv_key = mismatch['metric']
                    correct_value = mismatch['json_value']
                    self.rows[row_index][csv_key] = correct_value
                elif mismatch['type'] == 'value_mismatch':
                    csv_key = mismatch['metric']
                    correct_value = mismatch['json_value']
                    self.rows[row_index][csv_key] = correct_value

            # 修复性能指标
            for mismatch in verification.get('performance_mismatches', []):
                metric = mismatch['metric']
                correct_value = mismatch['json_value']
                csv_key = f'perf_{metric}'

                if csv_key in self.rows[row_index]:
                    self.rows[row_index][csv_key] = correct_value

            fixed_count += 1
            print(f'✅ 已修复行 {row_index + 2}: {verification["experiment_id"]}')

        print('')
        print(f'总计修复: {fixed_count} 行')
        print('')

        # 写回CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows)

        print(f'✅ 已更新: {self.csv_path}')
        print('')

        return True

    def run(self, fix=False):
        """执行完整验证流程"""
        self.load_csv()
        shared_ids = self.find_shared_ids()

        if not shared_ids:
            print('✅ 没有共享experiment_id的数据行')
            return

        verifications = self.verify_shared_ids(shared_ids)
        issues = self.generate_report(verifications)

        if fix and issues:
            self.fix_data(verifications)

        # 生成详细报告文件
        self.save_detailed_report(verifications)

    def save_detailed_report(self, verifications):
        """保存详细报告到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path('docs/results_reports') / f'SHARED_ID_VERIFICATION_REPORT_{timestamp}.md'

        with open(report_path, 'w') as f:
            f.write('# 共享ID数据验证报告\n\n')
            f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write('---\n\n')

            f.write('## 统计摘要\n\n')
            f.write(f'- 总数据行数: {self.stats["total_rows"]}\n')
            f.write(f'- 共享ID行数: {self.stats["shared_id_rows"]}\n')
            f.write(f'- 验证通过: {self.stats["verified_ok"]}\n')
            f.write(f'- JSON未找到: {self.stats["json_not_found"]}\n')
            f.write(f'- 超参数不匹配: {self.stats["hyperparam_mismatch"]}\n')
            f.write(f'- 能耗不匹配: {self.stats["energy_mismatch"]}\n')
            f.write(f'- 性能不匹配: {self.stats["performance_mismatch"]}\n\n')

            # 详细验证结果
            f.write('## 详细验证结果\n\n')

            for v in verifications:
                if v['status'] == 'ok':
                    continue

                f.write(f'### 行 {v["row_index"] + 2}: {v["experiment_id"]}\n\n')
                f.write(f'- **timestamp**: {v["timestamp"]}\n')
                f.write(f'- **状态**: {v["status"]}\n')

                if v.get('json_path'):
                    f.write(f'- **JSON**: `{v["json_path"]}`\n')

                if v.get('hyperparam_mismatches'):
                    f.write(f'\n**超参数不匹配** ({len(v["hyperparam_mismatches"])}个):\n\n')
                    for m in v['hyperparam_mismatches']:
                        f.write(f'- `{m["param"]}`: CSV=`{m["csv_value"]}`, JSON=`{m["json_value"]}`\n')

                if v.get('energy_mismatches'):
                    f.write(f'\n**能耗不匹配** ({len(v["energy_mismatches"])}个):\n\n')
                    for m in v['energy_mismatches']:
                        f.write(f'- `{m["metric"]}`: CSV=`{m["csv_value"]}`, JSON=`{m["json_value"]}` ({m["type"]})\n')

                if v.get('performance_mismatches'):
                    f.write(f'\n**性能不匹配** ({len(v["performance_mismatches"])}个):\n\n')
                    for m in v['performance_mismatches']:
                        f.write(f'- `{m["metric"]}`: CSV=`{m["csv_value"]}`, JSON=`{m["json_value"]}`\n')

                f.write('\n---\n\n')

        print(f'✅ 详细报告已保存: {report_path}')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='验证并修复所有共享experiment_id的数据行',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--fix', action='store_true',
                       help='自动修复发现的错误数据')

    args = parser.parse_args()

    print('=' * 80)
    print('共享ID数据验证工具')
    print('=' * 80)
    print('')

    verifier = DataVerifier()
    verifier.run(fix=args.fix)

    print('')
    print('=' * 80)
    print('完成')
    print('=' * 80)


if __name__ == '__main__':
    main()
