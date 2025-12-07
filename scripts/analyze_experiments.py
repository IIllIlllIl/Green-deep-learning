#!/usr/bin/env python3
"""
统一的实验分析工具

支持多种数据源:
- CSV文件 (summary_all.csv)
- JSON文件 (遍历experiment.json)

功能:
1. 统计参数-模式组合的唯一值数量
2. 生成完成度报告
3. 列出缺失的参数-模式组合
4. 估算需要补充的实验数

使用示例:
  # 从CSV分析
  python3 scripts/analyze_experiments.py --source csv --file results/summary_all.csv

  # 从JSON分析
  python3 scripts/analyze_experiments.py --source json --dir results/

  # 仅显示缺失组合
  python3 scripts/analyze_experiments.py --source csv --missing-only

  # 导出Markdown报告
  python3 scripts/analyze_experiments.py --source csv --output report.md

作者: Green
日期: 2025-12-06
版本: 1.0
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

# 超参数列映射
HYPERPARAM_COLUMNS = {
    "hyperparam_alpha": "alpha",
    "hyperparam_batch_size": "batch_size",
    "hyperparam_dropout": "dropout",
    "hyperparam_epochs": "epochs",
    "hyperparam_kfold": "kfold",
    "hyperparam_learning_rate": "learning_rate",
    "hyperparam_max_iter": "max_iter",
    "hyperparam_seed": "seed",
    "hyperparam_weight_decay": "weight_decay",
}


class ExperimentAnalyzer:
    """实验分析器"""

    def __init__(self):
        # 数据结构：{(repo, model, param, mode): set of unique values}
        self.unique_values: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)
        self.total_experiments = 0

    def analyze_from_csv(self, csv_path: Path):
        """从CSV文件分析实验"""
        print(f"读取 {csv_path}...")

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.total_experiments += 1

                exp_id = row.get('experiment_id', '')
                repo = row.get('repository', '')
                model = row.get('model', '')

                # 确定模式
                mode = 'parallel' if '_parallel' in exp_id else 'nonparallel'

                # 提取所有超参数
                for csv_col, param_name in HYPERPARAM_COLUMNS.items():
                    value_str = row.get(csv_col, '').strip()

                    if not value_str:
                        continue

                    # 标准化数值
                    try:
                        value = float(value_str)
                        normalized_value = f"{value:.6f}"
                    except ValueError:
                        normalized_value = value_str

                    self.unique_values[(repo, model, param_name, mode)].add(normalized_value)

        print(f"总共读取 {self.total_experiments} 个实验\n")

    def analyze_from_json(self, results_dir: Path):
        """从JSON文件分析实验"""
        print(f"扫描 {results_dir}...")

        # 遍历所有session目录
        session_dirs = [d for d in results_dir.iterdir()
                       if d.is_dir() and d.name.startswith("run_")]

        print(f"发现 {len(session_dirs)} 个session目录...")

        for session_dir in sorted(session_dirs):
            # 遍历session中的所有实验目录
            for exp_dir in session_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                json_file = exp_dir / "experiment.json"
                if not json_file.exists():
                    continue

                self.total_experiments += 1

                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    exp_id = data.get("experiment_id", "")

                    # 确定模式
                    if "parallel" in exp_id or data.get("mode") == "parallel":
                        mode = "parallel"
                    else:
                        mode = "nonparallel"

                    # 提取超参数（处理并行和非并行两种结构）
                    if "foreground" in data:
                        # 并行实验
                        repo = data["foreground"].get("repository", "")
                        model = data["foreground"].get("model", "")
                        hyperparams = data["foreground"].get("hyperparameters", {})
                    else:
                        # 非并行实验
                        repo = data.get("repository", "")
                        model = data.get("model", "")
                        hyperparams = data.get("hyperparameters", {})

                    # 记录超参数
                    for param, value in hyperparams.items():
                        normalized_value = (f"{float(value):.6f}"
                                          if isinstance(value, (int, float))
                                          else str(value))
                        self.unique_values[(repo, model, param, mode)].add(normalized_value)

                except Exception as e:
                    print(f"  警告: 读取 {json_file} 失败: {e}")
                    continue

        print(f"总共扫描 {self.total_experiments} 个实验\n")

    def generate_report(self, missing_only=False):
        """生成分析报告"""
        # 按repo/model组织
        by_model = defaultdict(lambda: defaultdict(dict))

        for (repo, model, param, mode), values in self.unique_values.items():
            model_key = f"{repo}/{model}"
            if mode not in by_model[model_key][param]:
                by_model[model_key][param] = {}
            by_model[model_key][param][mode] = len(values)

        # 统计
        total_combinations = 0
        complete_combinations = 0
        missing_list = []

        print("=" * 100)
        print("实验完成情况分析（区分并行/非并行模式）")
        print("=" * 100)

        if not missing_only:
            print(f"{'模型':<40} {'参数':<20} {'非并行':<10} {'并行':<10} {'状态':<10}")
            print("-" * 100)

        for model_key in sorted(by_model.keys()):
            params = by_model[model_key]

            for param in sorted(params.keys()):
                modes = params[param]
                nonparallel_count = modes.get('nonparallel', 0)
                parallel_count = modes.get('parallel', 0)

                # 统计两种模式
                for mode in ['nonparallel', 'parallel']:
                    total_combinations += 1
                    count = modes.get(mode, 0)

                    if count >= 5:
                        complete_combinations += 1
                    else:
                        needed = 5 - count
                        missing_list.append({
                            'model': model_key,
                            'param': param,
                            'mode': mode,
                            'current': count,
                            'needed': needed
                        })

                # 打印行（仅在非missing_only模式）
                if not missing_only:
                    nonparallel_status = "✓" if nonparallel_count >= 5 else f"{nonparallel_count}/5"
                    parallel_status = "✓" if parallel_count >= 5 else f"{parallel_count}/5"
                    overall_status = "✓" if nonparallel_count >= 5 and parallel_count >= 5 else "待补充"

                    print(f"{model_key:<40} {param:<20} {nonparallel_status:<10} "
                          f"{parallel_status:<10} {overall_status:<10}")

        print("=" * 100)
        print(f"\n总计参数-模式组合: {total_combinations}")
        print(f"已完成组合: {complete_combinations} ({complete_combinations/total_combinations*100:.1f}%)")
        print(f"待补充组合: {len(missing_list)}")

        # 详细缺失列表
        print("\n" + "=" * 100)
        print("待补充的参数-模式组合详细列表")
        print("=" * 100)

        # 按模式分组
        nonparallel_missing = [x for x in missing_list if x['mode'] == 'nonparallel']
        parallel_missing = [x for x in missing_list if x['mode'] == 'parallel']

        print(f"\n非并行模式缺失: {len(nonparallel_missing)}个")
        print("-" * 100)
        for item in nonparallel_missing:
            print(f"  {item['model']:<40} {item['param']:<20} "
                  f"当前:{item['current']} 需补充:{item['needed']}")

        print(f"\n并行模式缺失: {len(parallel_missing)}个")
        print("-" * 100)
        for item in parallel_missing:
            print(f"  {item['model']:<40} {item['param']:<20} "
                  f"当前:{item['current']} 需补充:{item['needed']}")

        # 计算需要的实验数
        total_needed_values = sum(x['needed'] for x in missing_list)
        print(f"\n总共需补充唯一值: {total_needed_values}个")

        # 考虑去重率，估算实际需要的实验数
        if total_needed_values > 0:
            estimated_experiments_50 = int(total_needed_values / 0.5)
            estimated_experiments_60 = int(total_needed_values / 0.6)

            print(f"\n预估需要的实验数（考虑去重）:")
            print(f"  - 保守估计（50%成功率）: {estimated_experiments_50}个")
            print(f"  - 乐观估计（60%成功率）: {estimated_experiments_60}个")

        return missing_list, total_needed_values

    def export_markdown(self, output_path: Path, missing_list, total_needed_values):
        """导出Markdown报告"""
        from datetime import datetime

        with open(output_path, 'w') as f:
            f.write("# 实验完成情况分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总实验数**: {self.total_experiments}\n\n")

            f.write("## 总体统计\n\n")

            # 按repo/model组织
            by_model = defaultdict(lambda: defaultdict(dict))
            for (repo, model, param, mode), values in self.unique_values.items():
                model_key = f"{repo}/{model}"
                if mode not in by_model[model_key][param]:
                    by_model[model_key][param] = {}
                by_model[model_key][param][mode] = len(values)

            total_combinations = sum(2 for _ in by_model.values() for _ in _.keys())  # 每个参数2个模式
            complete_combinations = sum(
                1 for params in by_model.values()
                for modes in params.values()
                for mode in ['nonparallel', 'parallel']
                if modes.get(mode, 0) >= 5
            )

            f.write(f"- 总参数-模式组合: {total_combinations}\n")
            f.write(f"- 已完成组合: {complete_combinations} ({complete_combinations/total_combinations*100:.1f}%)\n")
            f.write(f"- 待补充组合: {len(missing_list)}\n")
            f.write(f"- 需补充唯一值: {total_needed_values}个\n\n")

            f.write("## 详细完成情况\n\n")
            f.write("| 模型 | 参数 | 非并行 | 并行 | 状态 |\n")
            f.write("|------|------|--------|------|------|\n")

            for model_key in sorted(by_model.keys()):
                params = by_model[model_key]
                for param in sorted(params.keys()):
                    modes = params[param]
                    nonparallel_count = modes.get('nonparallel', 0)
                    parallel_count = modes.get('parallel', 0)

                    nonparallel_status = "✓" if nonparallel_count >= 5 else f"{nonparallel_count}/5"
                    parallel_status = "✓" if parallel_count >= 5 else f"{parallel_count}/5"
                    overall_status = "✓" if nonparallel_count >= 5 and parallel_count >= 5 else "待补充"

                    f.write(f"| {model_key} | {param} | {nonparallel_status} | "
                           f"{parallel_status} | {overall_status} |\n")

            f.write("\n## 缺失组合详情\n\n")

            nonparallel_missing = [x for x in missing_list if x['mode'] == 'nonparallel']
            parallel_missing = [x for x in missing_list if x['mode'] == 'parallel']

            f.write(f"### 非并行模式缺失 ({len(nonparallel_missing)}个)\n\n")
            if nonparallel_missing:
                f.write("| 模型 | 参数 | 当前 | 需补充 |\n")
                f.write("|------|------|------|--------|\n")
                for item in nonparallel_missing:
                    f.write(f"| {item['model']} | {item['param']} | "
                           f"{item['current']} | {item['needed']} |\n")

            f.write(f"\n### 并行模式缺失 ({len(parallel_missing)}个)\n\n")
            if parallel_missing:
                f.write("| 模型 | 参数 | 当前 | 需补充 |\n")
                f.write("|------|------|------|--------|\n")
                for item in parallel_missing:
                    f.write(f"| {item['model']} | {item['param']} | "
                           f"{item['current']} | {item['needed']} |\n")

        print(f"\n✓ Markdown报告已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="统一的实验分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从CSV分析
  python3 scripts/analyze_experiments.py --source csv --file results/summary_all.csv

  # 从JSON分析
  python3 scripts/analyze_experiments.py --source json --dir results/

  # 仅显示缺失组合
  python3 scripts/analyze_experiments.py --source csv --missing-only

  # 导出Markdown报告
  python3 scripts/analyze_experiments.py --source csv --output report.md
        """
    )

    parser.add_argument(
        '--source',
        choices=['csv', 'json'],
        required=True,
        help='数据源类型 (csv 或 json)'
    )

    parser.add_argument(
        '--file',
        type=str,
        help='CSV文件路径 (source=csv时必需)'
    )

    parser.add_argument(
        '--dir',
        type=str,
        default='results',
        help='JSON文件目录 (source=json时使用，默认: results/)'
    )

    parser.add_argument(
        '--missing-only',
        action='store_true',
        help='仅显示缺失的参数-模式组合'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='导出Markdown报告路径'
    )

    args = parser.parse_args()

    # 验证参数
    if args.source == 'csv' and not args.file:
        parser.error("--source csv 需要指定 --file 参数")

    # 创建分析器
    analyzer = ExperimentAnalyzer()

    # 分析数据
    if args.source == 'csv':
        csv_path = Path(args.file)
        if not csv_path.exists():
            print(f"错误: 找不到文件 {csv_path}")
            sys.exit(1)
        analyzer.analyze_from_csv(csv_path)
    else:  # json
        results_dir = Path(args.dir)
        if not results_dir.exists():
            print(f"错误: 找不到目录 {results_dir}")
            sys.exit(1)
        analyzer.analyze_from_json(results_dir)

    # 生成报告
    missing_list, total_needed = analyzer.generate_report(missing_only=args.missing_only)

    # 导出Markdown
    if args.output:
        output_path = Path(args.output)
        analyzer.export_markdown(output_path, missing_list, total_needed)

    print("\n" + "=" * 100)
    print("分析完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
