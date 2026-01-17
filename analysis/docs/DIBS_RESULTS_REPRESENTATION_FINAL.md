# DiBS结果展示最终方案 v3.0

**创建日期**: 2026-01-17
**版本**: v3.0 (最终版，用户确认)
**状态**: 待实施

---

## 📋 用户确认的需求

### 需求1: 生成的文件清单

每个任务组生成4个CSV文件：

1. **causal_edges_all.csv** - 所有边（529行，无筛选）
2. **causal_paths.csv** - 所有间接路径（path_strength > 0.05）
3. **causal_edges_0.3.csv** - 强边（strength > 0.3）
4. **causal_paths_0.3.csv** - 强路径（path_strength > 0.3）

### 需求2: 文件存放位置

**不修改原有文件**，在上一层目录创建新文件夹：

```
results/energy_research/dibs_interaction/
├── 20260117_000522/                    # 原始DiBS输出（不修改）⭐
│   ├── group1_examples_causal_graph.npy
│   ├── group1_examples_feature_names.json
│   ├── group1_examples_result.json
│   └── ...
│
└── 20260117_000522_readable/           # 新增：可读结果（CSV）⭐⭐⭐
    ├── group1_examples_causal_edges_all.csv
    ├── group1_examples_causal_paths.csv
    ├── group1_examples_causal_edges_0.3.csv
    ├── group1_examples_causal_paths_0.3.csv
    ├── group2_vulberta_causal_edges_all.csv
    ├── group2_vulberta_causal_paths.csv
    ├── ... (6组 × 4文件 = 24个CSV文件)
    └── README.md                       # 文件说明
```

**关键设计**:
- ✅ 原始目录 `20260117_000522/` **不被修改**
- ✅ 新目录 `20260117_000522_readable/` 存放所有CSV文件
- ✅ 清晰的命名约定

---

## 📄 文件格式详细说明

### 文件1: causal_edges_all.csv（所有直接边）

**行数**: 529行（23×23所有可能的边）

**列定义**:
```csv
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation
```

**字段说明**:
- `source`: 源变量（完整名称）
- `target`: 目标变量（完整名称）
- `strength`: 边强度（0-1，包括0.00）
- `edge_type`: 边类型（main_effect/moderation/mediator/control_effect/irrelevant）
- `is_significant`: 是否显著（yes: >0.1, no: ≤0.1）
- `strength_level`: 强度等级（very_strong/strong/moderate/weak/very_weak/zero）
- `source_category`: 源类别（hyperparam/interaction/energy/performance/mediator/control）
- `target_category`: 目标类别（同上）
- `question_relevance`: 相关研究问题（Q1/Q2/Q3/other）
- `interpretation`: 人类可读解释

**排序**: 按`strength`降序（强边在前）

**示例**:
```csv
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation
hyperparam_epochs,energy_gpu_total_joules,0.40,main_effect,yes,strong,hyperparam,energy,Q1,epochs直接影响GPU总能耗
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,yes,strong,interaction,energy,Q1,并行模式调节batch_size对CPU能耗的效应
hyperparam_batch_size,energy_cpu_total_joules,0.00,main_effect,no,zero,hyperparam,energy,Q1,batch_size对CPU能耗无直接影响
hyperparam_seed,perf_test_accuracy,0.00,irrelevant,no,zero,hyperparam,performance,other,随机种子不影响性能（预期）
```

---

### 文件2: causal_paths.csv（所有间接路径）

**筛选条件**: path_strength > 0.05（过滤弱路径）

**列定义**:
```csv
path_id,path_length,source,target,path,path_strength,step1_strength,step2_strength,step3_strength,path_type,question_relevance,interpretation
```

**字段说明**:
- `path_id`: 路径唯一ID（P2_001表示2步路径第1条，P3_001表示3步路径第1条）
- `path_length`: 路径步数（2或3）
- `source`: 起点变量（完整名称）
- `target`: 终点变量（完整名称）
- `path`: **完整路径（简化变量名）** ⭐⭐⭐⭐⭐
  - 格式: "简化源 → 简化中介1 → 简化中介2 → 简化目标"
  - 简化规则:
    - `hyperparam_` → 删除
    - `energy_` → 删除
    - `perf_` → 删除
    - `_x_is_parallel` → `_x_parallel`
- `path_strength`: 路径总强度（各步强度相乘）
- `step1_strength`, `step2_strength`, `step3_strength`: 各步强度
- `path_type`: 路径类型
- `question_relevance`: 相关研究问题
- `interpretation`: 人类可读解释

**路径类型**:
- `mediation_to_energy`: 超参数 → 中介 → 能耗
- `mediation_to_performance`: 超参数 → 中介 → 性能
- `moderation_mediated`: 调节效应 → 中介 → 能耗/性能
- `energy_perf_mediated`: 能耗 → 中介 → 性能（或反向）
- `other_mediation`: 其他中介路径

**排序**: 按`path_strength`降序

**示例**:
```csv
path_id,path_length,source,target,path,path_strength,step1_strength,step2_strength,step3_strength,path_type,question_relevance,interpretation
P2_001,2,hyperparam_batch_size,energy_gpu_total_joules,batch_size → gpu_max_watts → gpu_total,0.12,0.30,0.40,,mediation_to_energy,Q1-Q3,batch_size通过GPU峰值功率间接影响GPU总能耗
P3_001,3,hyperparam_batch_size,perf_test_accuracy,batch_size → gpu_max_watts → gpu_util_max → test_accuracy,0.042,0.30,0.40,0.35,mediation_to_performance,Q3,batch_size通过GPU功率和利用率间接影响性能
P2_002,2,hyperparam_epochs_x_is_parallel,energy_cpu_total_joules,epochs_x_parallel → gpu_temp_max → cpu_total,0.09,0.30,0.30,,moderation_mediated,Q1-Q3,并行调节效应通过GPU温度影响CPU能耗
```

---

### 文件3: causal_edges_0.3.csv（强边）

**筛选条件**: strength > 0.3

**格式**: 与`causal_edges_all.csv`完全相同，只是行数更少

**行数**: 约42-78行（根据组不同）

**用途**: 快速查看最重要的因果关系，无需手动筛选

**示例**:
```csv
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation
hyperparam_epochs,energy_gpu_total_joules,0.40,main_effect,yes,strong,hyperparam,energy,Q1,epochs直接影响GPU总能耗
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,yes,strong,interaction,energy,Q1,并行模式调节batch_size对CPU能耗的效应
hyperparam_epochs_x_is_parallel,energy_gpu_total_joules,0.40,moderation,yes,strong,interaction,energy,Q1,并行模式调节epochs对GPU能耗的效应
```

---

### 文件4: causal_paths_0.3.csv（强路径）

**筛选条件**: path_strength > 0.3

**格式**: 与`causal_paths.csv`完全相同，只是行数更少

**行数**: 约0-5行（强路径非常少见）

**用途**: 发现极强的间接效应

**注意**: 如果某组没有强路径（path_strength > 0.3），该文件可能为空（仅包含header）

---

## 🗂️ 目录结构和文件命名

### 完整目录结构

```
results/energy_research/dibs_interaction/
├── 20260117_000522/                          # 原始DiBS输出（保持不变）
│   ├── group1_examples_causal_graph.npy
│   ├── group1_examples_feature_names.json
│   ├── group1_examples_result.json
│   ├── group2_vulberta_causal_graph.npy
│   ├── group2_vulberta_feature_names.json
│   ├── group2_vulberta_result.json
│   ├── ... (共18个文件: 6组 × 3文件)
│   └── DIBS_INTERACTION_ANALYSIS_REPORT.md
│
└── 20260117_000522_readable/                 # 新增：可读结果
    ├── README.md                             # 文件说明 ⭐
    │
    ├── group1_examples_causal_edges_all.csv
    ├── group1_examples_causal_paths.csv
    ├── group1_examples_causal_edges_0.3.csv
    ├── group1_examples_causal_paths_0.3.csv
    │
    ├── group2_vulberta_causal_edges_all.csv
    ├── group2_vulberta_causal_paths.csv
    ├── group2_vulberta_causal_edges_0.3.csv
    ├── group2_vulberta_causal_paths_0.3.csv
    │
    ├── group3_person_reid_causal_edges_all.csv
    ├── group3_person_reid_causal_paths.csv
    ├── group3_person_reid_causal_edges_0.3.csv
    ├── group3_person_reid_causal_paths_0.3.csv
    │
    ├── group4_bug_localization_causal_edges_all.csv
    ├── group4_bug_localization_causal_paths.csv
    ├── group4_bug_localization_causal_edges_0.3.csv
    ├── group4_bug_localization_causal_paths_0.3.csv
    │
    ├── group5_mrt_oast_causal_edges_all.csv
    ├── group5_mrt_oast_causal_paths.csv
    ├── group5_mrt_oast_causal_edges_0.3.csv
    ├── group5_mrt_oast_causal_paths_0.3.csv
    │
    ├── group6_resnet_causal_edges_all.csv
    ├── group6_resnet_causal_paths.csv
    ├── group6_resnet_causal_edges_0.3.csv
    └── group6_resnet_causal_paths_0.3.csv
```

**总文件数**: 24个CSV文件（6组 × 4文件）+ 1个README = 25个文件

### README.md内容

```markdown
# DiBS因果分析可读结果

**生成时间**: 2026-01-17
**原始数据**: ../20260117_000522/
**文件数量**: 24个CSV文件（6组 × 4文件）

---

## 文件说明

每个任务组生成4个CSV文件：

1. **{group}_causal_edges_all.csv** (529行)
   - 所有直接因果边（无筛选）
   - 包括强度=0的边
   - 用于验证"边不存在"

2. **{group}_causal_paths.csv** (约100-200行)
   - 所有间接因果路径（2步和3步）
   - 筛选: path_strength > 0.05
   - 用于发现中介效应

3. **{group}_causal_edges_0.3.csv** (约42-78行)
   - 强直接边（strength > 0.3）
   - 快速查看最重要的因果关系

4. **{group}_causal_paths_0.3.csv** (约0-5行)
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

---

## 原始数据位置

完整的DiBS输出（包括.npy矩阵和.json结果）位于：
`../20260117_000522/`
```

---

## 🛠️ 实施方案

### 脚本设计

**脚本名称**: `scripts/convert_dibs_to_csv.py`

**输入参数**:
```bash
python scripts/convert_dibs_to_csv.py \
  --input-dir results/energy_research/dibs_interaction/20260117_000522 \
  --output-dir results/energy_research/dibs_interaction/20260117_000522_readable
```

**功能**:
1. 读取原始目录中的所有`.npy`和`.json`文件
2. 对每个任务组生成4个CSV文件
3. 在输出目录生成README.md
4. **不修改**原始目录中的任何文件

### 核心函数

```python
import numpy as np
import pandas as pd
import json
from pathlib import Path

def convert_dibs_results_to_csv(input_dir, output_dir):
    """
    将DiBS结果转换为可读的CSV格式

    参数:
        input_dir: 原始DiBS输出目录（如 20260117_000522）
        output_dir: CSV输出目录（如 20260117_000522_readable）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有任务组
    npy_files = list(input_path.glob('*_causal_graph.npy'))

    print(f"找到 {len(npy_files)} 个任务组")

    for npy_file in npy_files:
        task_id = npy_file.stem.replace('_causal_graph', '')
        print(f"\n处理任务组: {task_id}")

        # 读取数据
        causal_graph = np.load(npy_file)
        feature_names_file = input_path / f"{task_id}_feature_names.json"
        with open(feature_names_file) as f:
            feature_names = json.load(f)

        # 1. 生成 causal_edges_all.csv
        edges_all = generate_all_edges(causal_graph, feature_names)
        output_file = output_path / f"{task_id}_causal_edges_all.csv"
        edges_all.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(edges_all)} 行)")

        # 2. 生成 causal_paths.csv
        paths_all = generate_all_paths(causal_graph, feature_names, min_strength=0.05)
        output_file = output_path / f"{task_id}_causal_paths.csv"
        paths_all.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(paths_all)} 行)")

        # 3. 生成 causal_edges_0.3.csv
        edges_strong = edges_all[edges_all['strength'] > 0.3]
        output_file = output_path / f"{task_id}_causal_edges_0.3.csv"
        edges_strong.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(edges_strong)} 行)")

        # 4. 生成 causal_paths_0.3.csv
        paths_strong = paths_all[paths_all['path_strength'] > 0.3]
        output_file = output_path / f"{task_id}_causal_paths_0.3.csv"
        paths_strong.to_csv(output_file, index=False)
        print(f"  ✓ 生成 {output_file.name} ({len(paths_strong)} 行)")

    # 生成README.md
    generate_readme(output_path, len(npy_files))
    print(f"\n✓ 生成 README.md")

    print(f"\n完成！共生成 {len(npy_files) * 4} 个CSV文件")

def generate_all_edges(causal_graph, feature_names):
    """
    生成所有边的DataFrame（无筛选）

    参数:
        causal_graph: (n, n) numpy数组，因果图矩阵
        feature_names: 变量名列表

    返回:
        DataFrame包含所有529条边
    """
    import pandas as pd

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
    """获取变量类别"""
    if '_x_is_parallel' in var_name:
        return 'interaction'
    elif var_name.startswith('hyperparam_'):
        return 'hyperparam'
    elif var_name in ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
                      'energy_cpu_total_joules', 'energy_gpu_total_joules']:
        return 'energy'
    elif var_name.startswith('energy_gpu'):
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
    import pandas as pd

    n = len(feature_names)
    paths = []

    # ========== 1. 发现2步路径: source → mediator → target ==========
    print(f"  发现2步路径...")
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

    print(f"    找到 {len([p for p in paths if p['path_length'] == 2])} 条2步路径")

    # ========== 2. 发现3步路径: source → med1 → med2 → target ==========
    print(f"  发现3步路径...")
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

    print(f"    找到 {len([p for p in paths if p['path_length'] == 3])} 条3步路径")

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

**生成时间**: 2026-01-17
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

---

## 原始数据位置

完整的DiBS输出（包括.npy矩阵和.json结果）位于：
`../20260117_000522/`
"""

    with open(output_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
```

---

## ✅ 验证清单

实施前检查：
- [ ] 方案符合用户需求（4个CSV文件）
- [ ] 文件命名约定清晰
- [ ] 原始目录不被修改
- [ ] 新目录命名合理（`_readable`后缀）
- [ ] README.md说明完整

实施后检查：
- [ ] 6组 × 4文件 = 24个CSV生成成功
- [ ] 所有CSV文件可用Excel打开
- [ ] `path`列格式正确（简化变量名 + 箭头）
- [ ] 强度筛选正确（0.3阈值）
- [ ] 原始目录未被修改
- [ ] README.md生成成功

---

## 📊 预期输出统计

| 文件类型 | 平均行数 | 文件大小 | 总数 |
|---------|---------|---------|------|
| causal_edges_all.csv | 529 | ~80 KB | 6 |
| causal_paths.csv | ~150 | ~30 KB | 6 |
| causal_edges_0.3.csv | ~60 | ~10 KB | 6 |
| causal_paths_0.3.csv | ~2 | ~2 KB | 6 |
| README.md | - | ~3 KB | 1 |
| **总计** | - | **~750 KB** | **25** |

---

## 🎯 与v2.0方案的差异

| 项目 | v2.0方案 | v3.0方案（最终） |
|------|---------|----------------|
| 文件数量 | 每组4个（all + summary.json + npy + paths） | 每组4个CSV |
| analysis_summary.json | ✅ 增强 | ❌ 不生成（保持原始） |
| 强边筛选文件 | ❌ 无 | ✅ 有（0.3阈值） |
| 文件位置 | 原目录 | 新目录（_readable） |
| 原始文件修改 | 可能修改 | **不修改** ⭐ |

---

## 📝 总结

### 核心要点

✅ **4个CSV文件**: edges_all, paths, edges_0.3, paths_0.3
✅ **不修改原始**: 新建`_readable`目录
✅ **完整信息**: 包含所有边（包括强度=0）
✅ **间接路径**: 自动发现，`path`列直观展示
✅ **快速筛选**: 0.3阈值文件

### 实施步骤

1. 创建 `scripts/convert_dibs_to_csv.py` 脚本
2. 实现边生成和路径发现算法
3. 执行转换（约5-6小时开发 + 10分钟运行）
4. 验证输出文件
5. 提交Subagent独立检查

---

**下一步**: 启动Subagent检查方案
