#!/usr/bin/env python3
"""
生成带交互项的DiBS输入数据

功能:
1. 读取6groups_final数据
2. 标准化超参数和能耗变量
3. 转换is_parallel为0/1
4. 创建交互项 (超参数 × is_parallel)
5. 保存为6groups_interaction数据

作者: Claude
创建日期: 2026-01-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import sys
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class InteractionTermsGenerator:
    """交互项数据生成器"""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 标准化器字典（每组独立）
        self.scalers = {}

        # 统计信息
        self.stats = {
            'generation_time': datetime.now().isoformat(),
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'groups': {}
        }

    def identify_column_types(self, df):
        """识别列类型"""
        columns = df.columns.tolist()

        # 能耗列（所有energy_开头的列）
        energy_cols = [col for col in columns if col.startswith('energy_')]

        # 超参数列（排除seed）
        hyperparam_cols = [col for col in columns
                          if col.startswith('hyperparam_')
                          and not col.endswith('_seed')]

        # 性能指标列
        perf_cols = [col for col in columns if col.startswith('perf_')]

        # 模型列（model_开头的列）
        model_cols = [col for col in columns if col.startswith('model_')]

        # 控制列
        control_cols = ['is_parallel', 'timestamp']

        # 保留seed但不标准化
        seed_cols = [col for col in columns if col.endswith('_seed')]

        return {
            'energy': energy_cols,
            'hyperparam': hyperparam_cols,
            'perf': perf_cols,
            'model': model_cols,
            'control': control_cols,
            'seed': seed_cols
        }

    def transform_group(self, group_name, df):
        """转换单个分组数据"""
        print(f"\n{'='*60}")
        print(f"处理 {group_name}")
        print(f"{'='*60}")

        # 创建副本
        df_transformed = df.copy()

        # 识别列类型
        col_types = self.identify_column_types(df)
        print(f"\n列类型识别:")
        print(f"  能耗变量: {len(col_types['energy'])}个")
        print(f"  超参数: {len(col_types['hyperparam'])}个")
        print(f"  性能指标: {len(col_types['perf'])}个")
        print(f"  模型变量: {len(col_types['model'])}个")
        print(f"  种子变量: {len(col_types['seed'])}个")

        # 步骤1: 转换is_parallel为0/1
        print(f"\n步骤1: 转换is_parallel")
        original_is_parallel = df_transformed['is_parallel'].value_counts()
        print(f"  原始分布: {original_is_parallel.to_dict()}")

        df_transformed['is_parallel'] = df_transformed['is_parallel'].map({
            True: 1, False: 0,
            'True': 1, 'False': 0,
            1: 1, 0: 0
        })

        # 验证转换
        if df_transformed['is_parallel'].isnull().any():
            raise ValueError(f"is_parallel转换失败，存在空值")

        print(f"  转换后分布: {df_transformed['is_parallel'].value_counts().to_dict()}")

        # 步骤2: 标准化能耗变量
        print(f"\n步骤2: 标准化能耗变量")
        if col_types['energy']:
            scaler_energy = StandardScaler()
            df_transformed[col_types['energy']] = scaler_energy.fit_transform(
                df_transformed[col_types['energy']]
            )

            # 验证标准化
            for col in col_types['energy'][:3]:  # 检查前3个
                mean = df_transformed[col].mean()
                std = df_transformed[col].std()
                print(f"  {col}: mean={mean:.6f}, std={std:.6f}")

            self.scalers[f"{group_name}_energy"] = {
                'mean': scaler_energy.mean_.tolist(),
                'std': scaler_energy.scale_.tolist(),
                'columns': col_types['energy']
            }

        # 步骤3: 标准化超参数
        print(f"\n步骤3: 标准化超参数")
        if col_types['hyperparam']:
            scaler_hyperparam = StandardScaler()
            df_transformed[col_types['hyperparam']] = scaler_hyperparam.fit_transform(
                df_transformed[col_types['hyperparam']]
            )

            # 验证标准化
            for col in col_types['hyperparam']:
                mean = df_transformed[col].mean()
                std = df_transformed[col].std()
                print(f"  {col}: mean={mean:.6f}, std={std:.6f}")

            self.scalers[f"{group_name}_hyperparam"] = {
                'mean': scaler_hyperparam.mean_.tolist(),
                'std': scaler_hyperparam.scale_.tolist(),
                'columns': col_types['hyperparam']
            }

        # 步骤4: 标准化性能指标
        print(f"\n步骤4: 标准化性能指标")
        if col_types['perf']:
            scaler_perf = StandardScaler()
            df_transformed[col_types['perf']] = scaler_perf.fit_transform(
                df_transformed[col_types['perf']]
            )

            # 验证标准化
            for col in col_types['perf']:
                mean = df_transformed[col].mean()
                std = df_transformed[col].std()
                print(f"  {col}: mean={mean:.6f}, std={std:.6f}")

            self.scalers[f"{group_name}_perf"] = {
                'mean': scaler_perf.mean_.tolist(),
                'std': scaler_perf.scale_.tolist(),
                'columns': col_types['perf']
            }

        # 步骤5: 创建交互项
        print(f"\n步骤5: 创建交互项")
        interaction_cols = []
        for hp in col_types['hyperparam']:
            interaction_col = f"{hp}_x_is_parallel"
            df_transformed[interaction_col] = (
                df_transformed[hp] * df_transformed['is_parallel']
            )
            interaction_cols.append(interaction_col)

            # 验证交互项
            nonparallel_count = (df_transformed['is_parallel'] == 0).sum()
            parallel_count = (df_transformed['is_parallel'] == 1).sum()

            nonparallel_zeros = (
                df_transformed[df_transformed['is_parallel'] == 0][interaction_col] == 0
            ).sum()

            print(f"  {interaction_col}:")
            print(f"    非并行记录: {nonparallel_count}条, 交互项=0的: {nonparallel_zeros}条")
            print(f"    并行记录: {parallel_count}条")

        # 保存统计信息
        self.stats['groups'][group_name] = {
            'original_rows': len(df),
            'transformed_rows': len(df_transformed),
            'original_cols': len(df.columns),
            'transformed_cols': len(df_transformed.columns),
            'energy_vars': col_types['energy'],
            'hyperparam_vars': col_types['hyperparam'],
            'perf_vars': col_types['perf'],
            'interaction_vars': interaction_cols,
            'parallel_count': int(parallel_count),
            'nonparallel_count': int(nonparallel_count)
        }

        print(f"\n✅ {group_name} 转换完成:")
        print(f"  原始: {len(df)}行 × {len(df.columns)}列")
        print(f"  转换后: {len(df_transformed)}行 × {len(df_transformed.columns)}列")
        print(f"  新增交互项: {len(interaction_cols)}个")

        return df_transformed

    def process_all_groups(self):
        """处理所有分组"""
        print("="*60)
        print("开始生成交互项数据")
        print("="*60)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")

        # 查找所有group文件
        group_files = sorted(self.input_dir.glob("group*.csv"))

        if not group_files:
            raise FileNotFoundError(f"在 {self.input_dir} 中未找到group*.csv文件")

        print(f"\n找到 {len(group_files)} 个分组文件")

        # 处理每个分组
        for group_file in group_files:
            group_name = group_file.stem  # 例如 "group1_examples"

            # 读取原始数据
            df = pd.read_csv(group_file)

            # 转换数据
            df_transformed = self.transform_group(group_name, df)

            # 保存转换后的数据
            output_file = self.output_dir / f"{group_name}_interaction.csv"
            df_transformed.to_csv(output_file, index=False)
            print(f"  保存到: {output_file}")

        # 保存标准化参数
        params_file = self.output_dir / "standardization_params.json"
        with open(params_file, 'w') as f:
            json.dump(self.scalers, f, indent=2)
        print(f"\n✅ 标准化参数保存到: {params_file}")

        # 保存统计信息
        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✅ 统计信息保存到: {stats_file}")

        # 生成人类可读的统计报告
        self.generate_report()

    def generate_report(self):
        """生成人类可读的统计报告"""
        report_file = self.output_dir / "generation_report.txt"

        with open(report_file, 'w') as f:
            f.write("交互项数据生成报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"生成时间: {self.stats['generation_time']}\n")
            f.write(f"输入目录: {self.stats['input_dir']}\n")
            f.write(f"输出目录: {self.stats['output_dir']}\n\n")

            f.write(f"处理分组数: {len(self.stats['groups'])}\n\n")

            for group_name, group_stats in self.stats['groups'].items():
                f.write(f"\n{group_name}:\n")
                f.write(f"  数据行数: {group_stats['original_rows']} → {group_stats['transformed_rows']}\n")
                f.write(f"  列数: {group_stats['original_cols']} → {group_stats['transformed_cols']}\n")
                f.write(f"  并行/非并行: {group_stats['parallel_count']}/{group_stats['nonparallel_count']}\n")
                f.write(f"  能耗变量: {len(group_stats['energy_vars'])}个\n")
                f.write(f"  超参数: {len(group_stats['hyperparam_vars'])}个\n")
                f.write(f"  性能指标: {len(group_stats['perf_vars'])}个\n")
                f.write(f"  交互项: {len(group_stats['interaction_vars'])}个\n")
                f.write(f"    {', '.join(group_stats['interaction_vars'])}\n")

        print(f"✅ 生成报告保存到: {report_file}")


def main():
    """主函数"""
    # 路径配置
    input_dir = PROJECT_ROOT / "data" / "energy_research" / "6groups_final"
    output_dir = PROJECT_ROOT / "data" / "energy_research" / "6groups_interaction"

    # 检查输入目录
    if not input_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 创建生成器
    generator = InteractionTermsGenerator(input_dir, output_dir)

    # 处理所有分组
    try:
        generator.process_all_groups()
        print("\n" + "="*60)
        print("✅ 所有数据处理完成！")
        print("="*60)
        print(f"\n输出目录: {output_dir}")
        print("\n生成的文件:")
        for f in sorted(output_dir.glob("*")):
            print(f"  - {f.name}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
