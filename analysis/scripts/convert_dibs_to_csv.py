#!/usr/bin/env python3
"""
DiBS结果转换为CSV格式

功能:
- 将DiBS因果图矩阵(.npy)转换为可读的CSV文件
- 生成4个文件：edges_all, paths, edges_0.3, paths_0.3
- 不修改原始文件

使用:
    python scripts/convert_dibs_to_csv.py \\
        --input-dir results/energy_research/dibs_interaction/20260117_000522 \\
        --output-dir results/energy_research/dibs_interaction/20260117_000522_readable
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime


def convert_dibs_results_to_csv(input_dir, output_dir):
    """
    将DiBS结果转换为可读的CSV格式

    参数:
        input_dir: 原始DiBS输出目录（如 20260117_000522）
        output_dir: CSV输出目录（如 20260117_000522_readable）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 验证输入目录存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 防止修改原始目录（在创建目录后检查）
    if output_path.samefile(input_path):
        raise ValueError("输出目录不能与输入目录相同！")
    print(f"输出目录: {output_path}")

    # 查找所有任务组
    npy_files = list(input_path.glob('*_causal_graph.npy'))

    if len(npy_files) == 0:
        raise FileNotFoundError(f"在{input_path}中未找到DiBS结果文件")

    print(f"\n找到 {len(npy_files)} 个任务组")

    for npy_file in sorted(npy_files):
        task_id = npy_file.stem.replace('_causal_graph', '')
        print(f"\n{'='*60}")
        print(f"处理任务组: {task_id}")
        print(f"{'='*60}")

        # 读取数据
        causal_graph = np.load(npy_file)
        feature_names_file = input_path / f"{task_id}_feature_names.json"

        if not feature_names_file.exists():
            print(f"  ⚠️  警告: 未找到{feature_names_file.name}，跳过")
            continue

        with open(feature_names_file) as f:
            feature_names = json.load(f)

        print(f"  变量数: {len(feature_names)}")
        print(f"  矩阵形状: {causal_graph.shape}")

        # 1. 生成 causal_edges_all.csv
        print(f"\n  [1/4] 生成所有边...")
        edges_all = generate_all_edges(causal_graph, feature_names)
        output_file = output_path / f"{task_id}_causal_edges_all.csv"
        edges_all.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(edges_all)} 行)")

        # 2. 生成 causal_paths.csv
        print(f"\n  [2/4] 生成间接路径...")
        paths_all = generate_all_paths(causal_graph, feature_names, min_strength=0.05)
        output_file = output_path / f"{task_id}_causal_paths.csv"
        paths_all.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(paths_all)} 行)")

        # 3. 生成 causal_edges_0.3.csv
        print(f"\n  [3/4] 生成强边...")
        edges_strong = edges_all[edges_all['strength'] > 0.3]
        output_file = output_path / f"{task_id}_causal_edges_0.3.csv"
        edges_strong.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(edges_strong)} 行)")

        # 4. 生成 causal_paths_0.3.csv
        print(f"\n  [4/4] 生成强路径...")
        paths_strong = paths_all[paths_all['path_strength'] > 0.3] if len(paths_all) > 0 else paths_all
        output_file = output_path / f"{task_id}_causal_paths_0.3.csv"
        paths_strong.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(paths_strong)} 行)")

    # 生成README.md
    print(f"\n{'='*60}")
    print(f"生成README...")
    print(f"{'='*60}")
    generate_readme(output_path, len(npy_files))
    print(f"✓ 生成 README.md")

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"{'='*60}")
    print(f"共生成 {len(npy_files) * 4} 个CSV文件 + 1个README")
    print(f"输出目录: {output_path}")


def generate_all_edges(causal_graph, feature_names):
    """
    生成所有边的DataFrame（无筛选）

    参数:
        causal_graph: (n, n) numpy数组，因果图矩阵
        feature_names: 变量名列表

    返回:
        DataFrame包含所有529条边
    """
    n = len(feature_names)
    edges = []

    # 遍历所有可能的边（包括强度=0）
    for i in range(n):
        for j in range(n):
            source = feature_names[i]
            target = feature_names[j]
            strength = float(causal_graph[i, j])

            # 分类边类型
            edge_type = classify_edge_type(source, target)

            # 判断是否显著
            is_significant = 'yes' if strength > 0.1 else 'no'

            # 强度等级
            if strength > 0.5:
                strength_level = 'very_strong'
            elif strength > 0.3:
                strength_level = 'strong'
            elif strength > 0.1:
                strength_level = 'moderate'
            elif strength > 0.01:
                strength_level = 'weak'
            elif strength > 0.001:
                strength_level = 'very_weak'
            else:
                strength_level = 'zero'

            # 变量类别
            source_category = get_variable_category(source)
            target_category = get_variable_category(target)

            # 研究问题相关性
            question_relevance = get_question_relevance(source, target, edge_type)

            # 人类可读解释
            interpretation = generate_interpretation(source, target, strength, edge_type)

            edges.append({
                'source': source,
                'target': target,
                'strength': strength,
                'edge_type': edge_type,
                'is_significant': is_significant,
                'strength_level': strength_level,
                'source_category': source_category,
                'target_category': target_category,
                'question_relevance': question_relevance,
                'interpretation': interpretation
            })

    # 转换为DataFrame并按强度降序排序
    df = pd.DataFrame(edges)
    df = df.sort_values('strength', ascending=False)

    return df


def classify_edge_type(source, target):
    """
    分类边类型

    规则:
    1. 如果source包含'_x_is_parallel' → moderation（调节效应）
    2. 如果source是超参数 且 target是能耗 → main_effect
    3. 如果source是能耗相关 且 target是能耗 → mediator
    4. 如果source是'model_' → control_effect
    5. 如果strength≈0 → irrelevant
    """
    if '_x_is_parallel' in source:
        return 'moderation'

    if source.startswith('hyperparam_') and '_x_' not in source:
        if target.startswith('energy_'):
            return 'main_effect'
        elif target.startswith('perf_'):
            return 'main_effect'

    if source.startswith('energy_') and target.startswith('energy_') and source != target:
        return 'mediator'

    if source.startswith('model_'):
        return 'control_effect'

    if source == 'is_parallel':
        return 'mode_effect'

    return 'irrelevant'


def get_variable_category(var_name):
    """获取变量类别

    变量分类规则 (v2.0 - 2026-02-07更新):
    - energy: 能耗结果变量（包括焦耳和功率指标）
    - mediator: 机制中介变量（仅温度和利用率，物理状态变量）
    - hyperparam: 超参数处理变量
    - interaction: 交互项调节变量
    - performance: 性能结果变量
    - control: 模型控制变量
    - mode: 并行模式变量

    修订说明:
    - 功率变量(energy_gpu_*_watts)从mediator改为energy
    - 中间变量仅保留物理状态变量(温度、利用率)
    """
    if '_x_is_parallel' in var_name:
        return 'interaction'
    elif var_name.startswith('hyperparam_'):
        return 'hyperparam'
    # 能耗指标：焦耳能耗 + 功率指标
    elif var_name in ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
                      'energy_cpu_total_joules', 'energy_gpu_total_joules',
                      'energy_gpu_avg_watts', 'energy_gpu_min_watts',
                      'energy_gpu_max_watts']:
        return 'energy'
    # 中间变量：仅温度和利用率（物理状态变量）
    elif var_name in ['energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
                      'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent']:
        return 'mediator'
    elif var_name.startswith('perf_'):
        return 'performance'
    elif var_name.startswith('model_'):
        return 'control'
    elif var_name == 'is_parallel':
        return 'mode'
    else:
        return 'other'


def get_question_relevance(source, target, edge_type):
    """判断与研究问题的相关性"""
    relevance = []

    # Q1: 超参数对能耗的影响
    if (source.startswith('hyperparam_') or '_x_is_parallel' in source) and \
       target in ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
                  'energy_cpu_total_joules', 'energy_gpu_total_joules']:
        relevance.append('Q1')

    # Q2: 能耗-性能权衡
    if (source.startswith('energy_') and target.startswith('perf_')) or \
       (source.startswith('perf_') and target.startswith('energy_')):
        relevance.append('Q2')

    # Q3: 中介效应
    if edge_type == 'mediator':
        relevance.append('Q3')

    return ','.join(relevance) if relevance else 'other'


def generate_interpretation(source, target, strength, edge_type):
    """生成人类可读解释"""
    # 简化变量名用于显示
    source_simple = simplify_variable_name(source)
    target_simple = simplify_variable_name(target)

    if strength < 0.001:
        return f"{source_simple}对{target_simple}无影响"

    if edge_type == 'moderation':
        base_param = source.replace('_x_is_parallel', '').replace('hyperparam_', '')
        return f"并行模式调节{base_param}对{target_simple}的效应"
    elif edge_type == 'main_effect':
        return f"{source_simple}直接影响{target_simple}"
    elif edge_type == 'mediator':
        return f"{source_simple}通过某种机制影响{target_simple}"
    elif edge_type == 'control_effect':
        return f"模型控制变量的影响"
    else:
        return f"{source_simple} → {target_simple}"


def generate_all_paths(causal_graph, feature_names, min_strength=0.05):
    """
    生成所有间接路径的DataFrame

    参数:
        causal_graph: (n, n) numpy数组
        feature_names: 变量名列表
        min_strength: 最小路径强度阈值

    返回:
        DataFrame包含所有2步和3步路径
    """
    n = len(feature_names)
    paths = []

    # ========== 1. 发现2步路径: source → mediator → target ==========
    print(f"    发现2步路径...")
    for source in range(n):
        # 只关注关键起点（超参数、交互项）
        source_name = feature_names[source]
        if not is_key_variable(source_name):
            continue

        for target in range(n):
            if source == target:
                continue  # 跳过自环

            for mediator in range(n):
                if mediator in [source, target]:
                    continue  # 跳过直接边

                strength1 = float(causal_graph[source, mediator])
                strength2 = float(causal_graph[mediator, target])

                if strength1 > 0 and strength2 > 0:
                    path_strength = strength1 * strength2
                    if path_strength > min_strength:
                        path_id = f"P2_{len([p for p in paths if p['path_length'] == 2]) + 1:03d}"

                        paths.append({
                            'path_id': path_id,
                            'path_length': 2,
                            'source': feature_names[source],
                            'target': feature_names[target],
                            'path': format_path([feature_names[source],
                                               feature_names[mediator],
                                               feature_names[target]]),
                            'path_strength': path_strength,
                            'step1_strength': strength1,
                            'step2_strength': strength2,
                            'step3_strength': None,
                            'path_type': classify_path_type(feature_names[source],
                                                           feature_names[mediator],
                                                           feature_names[target]),
                            'question_relevance': get_path_question_relevance(
                                feature_names[source], feature_names[target]),
                            'interpretation': generate_path_interpretation(
                                feature_names[source], feature_names[mediator],
                                feature_names[target], 2)
                        })

    print(f"      找到 {len([p for p in paths if p['path_length'] == 2])} 条2步路径")

    # ========== 2. 发现3步路径: source → med1 → med2 → target ==========
    print(f"    发现3步路径...")
    for source in range(n):
        source_name = feature_names[source]
        if not is_key_variable(source_name):
            continue

        for target in range(n):
            if source == target:
                continue

            for med1 in range(n):
                if med1 in [source, target]:
                    continue

                for med2 in range(n):
                    if med2 in [source, target, med1]:
                        continue  # 避免环路

                    s1 = float(causal_graph[source, med1])
                    s2 = float(causal_graph[med1, med2])
                    s3 = float(causal_graph[med2, target])

                    if s1 > 0 and s2 > 0 and s3 > 0:
                        path_strength = s1 * s2 * s3
                        if path_strength > min_strength:
                            path_id = f"P3_{len([p for p in paths if p['path_length'] == 3]) + 1:03d}"

                            paths.append({
                                'path_id': path_id,
                                'path_length': 3,
                                'source': feature_names[source],
                                'target': feature_names[target],
                                'path': format_path([feature_names[source],
                                                   feature_names[med1],
                                                   feature_names[med2],
                                                   feature_names[target]]),
                                'path_strength': path_strength,
                                'step1_strength': s1,
                                'step2_strength': s2,
                                'step3_strength': s3,
                                'path_type': classify_path_type(feature_names[source],
                                                               feature_names[med1],
                                                               feature_names[target],
                                                               feature_names[med2]),
                                'question_relevance': get_path_question_relevance(
                                    feature_names[source], feature_names[target]),
                                'interpretation': generate_path_interpretation(
                                    feature_names[source], feature_names[med1],
                                    feature_names[target], 3, feature_names[med2])
                            })

    print(f"      找到 {len([p for p in paths if p['path_length'] == 3])} 条3步路径")

    # 3. 转换为DataFrame并排序
    df = pd.DataFrame(paths)
    if len(df) > 0:
        df = df.sort_values('path_strength', ascending=False)

    return df


def is_key_variable(var_name):
    """判断是否为关键起点变量（用于路径搜索）"""
    # 关键起点：超参数、交互项
    if var_name.startswith('hyperparam_') or '_x_is_parallel' in var_name:
        return True
    # 排除：控制变量、种子、模型变量
    if var_name.startswith('model_') or 'seed' in var_name:
        return False
    return False


def format_path(nodes):
    """格式化路径为可读字符串"""
    simplified = [simplify_variable_name(n) for n in nodes]
    return ' → '.join(simplified)


def simplify_variable_name(var_name):
    """
    简化变量名（用于path列显示）

    规则:
    - 删除 'hyperparam_', 'energy_', 'perf_' 前缀
    - '_x_is_parallel' → '_x_parallel'
    - 保留 'model_', 'is_parallel' 等特殊变量
    """
    # 特殊变量不简化
    if var_name in ['is_parallel'] or var_name.startswith('model_'):
        return var_name

    # 删除前缀
    prefixes = ['hyperparam_', 'energy_', 'perf_']
    for prefix in prefixes:
        if var_name.startswith(prefix):
            var_name = var_name.replace(prefix, '')
            break

    # 简化交互项标记
    var_name = var_name.replace('_x_is_parallel', '_x_parallel')

    return var_name


def classify_path_type(source, mediator, target, mediator2=None):
    """分类路径类型"""
    if source.startswith('hyperparam_') and '_x_' not in source:
        if target.startswith('energy_'):
            return 'mediation_to_energy'
        elif target.startswith('perf_'):
            return 'mediation_to_performance'

    if '_x_is_parallel' in source:
        return 'moderation_mediated'

    if source.startswith('energy_') and target.startswith('perf_'):
        return 'energy_perf_mediated'

    return 'other_mediation'


def get_path_question_relevance(source, target):
    """获取路径的研究问题相关性"""
    relevance = []

    if (source.startswith('hyperparam_') or '_x_is_parallel' in source) and \
       target.startswith('energy_'):
        relevance.extend(['Q1', 'Q3'])  # 超参数影响能耗，有中介效应

    if source.startswith('energy_') and target.startswith('perf_'):
        relevance.extend(['Q2', 'Q3'])

    if source.startswith('hyperparam_') and target.startswith('perf_'):
        relevance.append('Q3')

    return ','.join(set(relevance)) if relevance else 'other'


def generate_path_interpretation(source, mediator1, target, steps, mediator2=None):
    """生成路径解释"""
    source_simple = simplify_variable_name(source)
    med1_simple = simplify_variable_name(mediator1)
    target_simple = simplify_variable_name(target)

    if steps == 2:
        return f"{source_simple}通过{med1_simple}间接影响{target_simple}"
    else:  # steps == 3
        med2_simple = simplify_variable_name(mediator2)
        return f"{source_simple}通过{med1_simple}和{med2_simple}间接影响{target_simple}"


def generate_readme(output_path, num_groups):
    """生成README.md"""
    readme_content = f"""# DiBS因果分析可读结果

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**原始数据**: ../20260117_000522/
**文件数量**: {num_groups * 4}个CSV文件（{num_groups}组 × 4文件）

---

## 文件说明

每个任务组生成4个CSV文件：

1. **{{group}}_causal_edges_all.csv** (529行)
   - 所有直接因果边（无筛选）
   - 包括强度=0的边
   - 用于验证"边不存在"

2. **{{group}}_causal_paths.csv** (约100-200行)
   - 所有间接因果路径（2步和3步）
   - 筛选: path_strength > 0.05
   - 用于发现中介效应

3. **{{group}}_causal_edges_0.3.csv** (约42-78行)
   - 强直接边（strength > 0.3）
   - 快速查看最重要的因果关系

4. **{{group}}_causal_paths_0.3.csv** (约0-5行)
   - 强间接路径（path_strength > 0.3）
   - 发现极强的中介效应

---

## 使用示例

### Excel快速查看

1. 打开 `group1_examples_causal_edges_0.3.csv`
2. 查看所有强因果关系（一目了然）

### 查看调节效应

```bash
cat group1_examples_causal_edges_all.csv | grep "moderation"
```

### 查看间接效应

```bash
cat group1_examples_causal_paths.csv | grep "batch_size"
```

### Python分析

```python
import pandas as pd

# 读取所有边
edges = pd.read_csv('group1_examples_causal_edges_all.csv')

# 筛选调节效应
moderation = edges[edges['edge_type'] == 'moderation']
print(moderation[['source', 'target', 'strength', 'interpretation']])

# 读取间接路径
paths = pd.read_csv('group1_examples_causal_paths.csv')

# 查找batch_size相关的路径
batch_paths = paths[paths['source'].str.contains('batch_size')]
print(batch_paths[['path', 'path_strength', 'interpretation']])
```

---

## 原始数据位置

完整的DiBS输出（包括.npy矩阵和.json结果）位于：
`../20260117_000522/`

---

## 数据字典

### causal_edges_all.csv 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| source | string | 源变量（因） |
| target | string | 目标变量（果） |
| strength | float | 边强度（0-1） |
| edge_type | string | 边类型（main_effect/moderation/mediator等） |
| is_significant | string | 是否显著（yes: >0.1, no: ≤0.1） |
| strength_level | string | 强度等级（very_strong/strong/moderate/weak/zero） |
| source_category | string | 源变量类别 |
| target_category | string | 目标变量类别 |
| question_relevance | string | 相关研究问题（Q1/Q2/Q3） |
| interpretation | string | 人类可读解释 |

### causal_paths.csv 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| path_id | string | 路径唯一ID |
| path_length | int | 路径步数（2或3） |
| source | string | 起点变量 |
| target | string | 终点变量 |
| path | string | 完整路径（简化变量名） |
| path_strength | float | 路径总强度（各步相乘） |
| step1_strength | float | 第1步强度 |
| step2_strength | float | 第2步强度 |
| step3_strength | float | 第3步强度（2步路径为空） |
| path_type | string | 路径类型 |
| question_relevance | string | 相关研究问题 |
| interpretation | string | 人类可读解释 |

---

**生成脚本**: `scripts/convert_dibs_to_csv.py`
**方案文档**: `docs/DIBS_RESULTS_REPRESENTATION_FINAL.md`
"""

    with open(output_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(description='将DiBS结果转换为CSV格式')
    parser.add_argument('--input-dir', required=True,
                        help='原始DiBS输出目录（如 results/energy_research/dibs_interaction/20260117_000522）')
    parser.add_argument('--output-dir', required=True,
                        help='CSV输出目录（如 results/energy_research/dibs_interaction/20260117_000522_readable）')

    args = parser.parse_args()

    print(f"DiBS结果转换为CSV")
    print(f"{'='*60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    convert_dibs_results_to_csv(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
