"""
能耗研究数据因果分析 - 4任务组分层版本
基于Adult数据集成功的DiBS+DML方法，适配能耗数据
适合长时间后台运行，包含完整的进度报告和检查点保存
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import pickle
from datetime import datetime

# 设置随机种子
np.random.seed(42)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def save_checkpoint(data, filename):
    """保存检查点"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ 检查点已保存: {filename}")
    except Exception as e:
        print(f"  ⚠️  保存检查点失败: {e}")

def load_checkpoint(filename):
    """加载检查点"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  加载检查点失败: {e}")
    return None

# ============================================================================
# 任务组定义（按数据质量优先级排序）
# ============================================================================
TASK_GROUPS = [
    {
        'name': 'image_classification',
        'display_name': '图像分类',
        'data_file': 'data/energy_research/training/training_data_image_classification.csv',
        'samples': 258,
        'features': 13,
        'priority': 1
    },
    {
        'name': 'person_reid',
        'display_name': 'Person_reID',
        'data_file': 'data/energy_research/training/training_data_person_reid.csv',
        'samples': 116,
        'features': 16,
        'priority': 2
    },
    {
        'name': 'vulberta',
        'display_name': 'VulBERTa',
        'data_file': 'data/energy_research/training/training_data_vulberta.csv',
        'samples': 142,
        'features': 10,
        'priority': 3
    },
    {
        'name': 'bug_localization',
        'display_name': 'Bug定位',
        'data_file': 'data/energy_research/training/training_data_bug_localization.csv',
        'samples': 132,
        'features': 11,
        'priority': 4
    }
]

# DiBS参数（与Adult分析保持一致 - 已验证有效）
DIBS_N_STEPS = 3000  # 优化版（Adult分析: 5000→3000, 速度提升>97%）
DIBS_ALPHA = 0.1
DIBS_THRESHOLD = 0.3
DIBS_RANDOM_SEED = 42

print_header("能耗研究因果分析 - 4任务组分层版")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"DiBS配置: n_steps={DIBS_N_STEPS}, alpha={DIBS_ALPHA}, threshold={DIBS_THRESHOLD}")
print(f"任务组数: {len(TASK_GROUPS)}")
print(f"预计总时间: 60-120分钟（取决于变量数和样本量）")

start_time = time.time()
os.makedirs('results/energy_research/task_specific', exist_ok=True)
os.makedirs('logs/energy_research/experiments', exist_ok=True)

# 统计信息
completed_tasks = []
failed_tasks = []

# ============================================================================
# 循环处理每个任务组
# ============================================================================
for task_idx, task in enumerate(TASK_GROUPS, 1):
    task_start_time = time.time()
    task_name = task['name']
    display_name = task['display_name']
    data_file = task['data_file']

    print_header(f"任务组 {task_idx}/{len(TASK_GROUPS)}: {display_name} ({task_name})")
    print(f"数据文件: {data_file}")
    print(f"预期: {task['samples']}样本, {task['features']}特征")

    # ------------------------------------------------------------------------
    # 步骤1: 加载数据
    # ------------------------------------------------------------------------
    print(f"\n  [步骤1] 加载数据...")

    if not os.path.exists(data_file):
        print(f"  ✗ 数据文件不存在: {data_file}")
        failed_tasks.append({'task': task_name, 'reason': '数据文件不存在'})
        continue

    try:
        df = pd.read_csv(data_file)
        print(f"  ✓ 数据加载成功: {len(df)}行 × {len(df.columns)}列")

        # 检查数据完整性
        if len(df) < 10:
            print(f"  ⚠️  样本量过少({len(df)}), 建议至少10个样本")
            failed_tasks.append({'task': task_name, 'reason': f'样本量过少({len(df)})'})
            continue

        # 准备数值型数据（DiBS要求）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        causal_data = df[numeric_cols].copy()

        # 移除全为NaN的列
        causal_data = causal_data.dropna(axis=1, how='all')
        numeric_cols = causal_data.columns.tolist()

        print(f"  数值型变量: {len(numeric_cols)}个")
        print(f"  有效样本: {len(causal_data)}行")

        # 检查缺失率
        missing_rate = causal_data.isna().sum().sum() / (len(causal_data) * len(numeric_cols))
        print(f"  总体缺失率: {missing_rate*100:.2f}%")

    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        failed_tasks.append({'task': task_name, 'reason': f'数据加载失败: {e}'})
        continue

    # ------------------------------------------------------------------------
    # 步骤2: DiBS因果图学习（优化版 - 与Adult分析相同配置）
    # ------------------------------------------------------------------------
    print(f"\n  [步骤2] DiBS因果图学习...")

    graph_file = f'results/energy_research/task_specific/{task_name}_causal_graph.npy'
    edges_file = f'results/energy_research/task_specific/{task_name}_causal_edges.pkl'

    # 检查是否已存在结果
    if os.path.exists(graph_file) and os.path.exists(edges_file):
        print(f"  ✓ 发现已有因果图: {graph_file}")
        try:
            causal_graph = np.load(graph_file)
            with open(edges_file, 'rb') as f:
                edge_data = pickle.load(f)
            edges = edge_data['edges']
            print(f"  因果边数: {len(edges)}")
            print(f"  跳过DiBS学习（使用已有结果）")
        except Exception as e:
            print(f"  ⚠️  加载失败: {e}，重新学习...")
            causal_graph = None
    else:
        causal_graph = None

    if causal_graph is None:
        print(f"  开始DiBS学习...")
        print(f"  配置: 变量数={len(numeric_cols)}, 样本数={len(causal_data)}, 迭代={DIBS_N_STEPS}步")
        print(f"  预计时间: 15-30分钟（取决于变量数）")

        try:
            from utils.causal_discovery import CausalGraphLearner

            # 创建学习器（与Adult分析相同配置）
            learner = CausalGraphLearner(
                n_vars=len(numeric_cols),
                n_steps=DIBS_N_STEPS,
                alpha=DIBS_ALPHA,
                random_seed=DIBS_RANDOM_SEED
            )

            # 学习因果图
            dibs_start = time.time()
            print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")
            print(f"  正在运行DiBS（请耐心等待）...")

            causal_graph = learner.fit(causal_data, verbose=True)

            dibs_time = time.time() - dibs_start
            print(f"\n  ✓ DiBS完成，耗时: {dibs_time/60:.1f}分钟")

            # 分析边
            edges = learner.get_edges(threshold=DIBS_THRESHOLD)
            print(f"  检测到 {len(edges)} 条因果边（阈值={DIBS_THRESHOLD}）")

            # 保存结果
            learner.save_graph(graph_file)
            with open(edges_file, 'wb') as f:
                pickle.dump({
                    'edges': edges,
                    'numeric_cols': numeric_cols,
                    'task_name': task_name,
                    'dibs_params': {
                        'n_steps': DIBS_N_STEPS,
                        'alpha': DIBS_ALPHA,
                        'threshold': DIBS_THRESHOLD,
                        'random_seed': DIBS_RANDOM_SEED
                    },
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f)
            print(f"  ✓ 因果图已保存")

            # 显示关键边
            if len(edges) > 0:
                print(f"\n  前10条最强因果边:")
                for i, (source, target, weight) in enumerate(edges[:10], 1):
                    print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")
            else:
                print(f"  ⚠️  未检测到置信度>{DIBS_THRESHOLD}的因果边")

        except Exception as e:
            print(f"  ✗ DiBS失败: {e}")
            import traceback
            traceback.print_exc()
            failed_tasks.append({'task': task_name, 'reason': f'DiBS失败: {e}'})
            continue

    # ------------------------------------------------------------------------
    # 步骤3: DML因果推断（与Adult分析相同方法）
    # ------------------------------------------------------------------------
    if causal_graph is not None and len(edges) > 0:
        print(f"\n  [步骤3] DML因果推断...")

        effects_file = f'results/energy_research/task_specific/{task_name}_causal_effects.csv'

        if os.path.exists(effects_file):
            print(f"  ✓ 发现已有因果效应文件: {effects_file}")
            effects_df = pd.read_csv(effects_file)
            print(f"  因果效应数: {len(effects_df)}")
            print(f"  跳过DML步骤（使用已有结果）")
        else:
            print(f"  开始DML因果推断...")
            print(f"  分析 {len(edges)} 条因果边")
            print(f"  预计时间: 5-10分钟")

            try:
                from utils.causal_inference import CausalInferenceEngine

                engine = CausalInferenceEngine(verbose=True)

                dml_start = time.time()
                print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")

                causal_effects = engine.analyze_all_edges(
                    data=causal_data,
                    causal_graph=causal_graph,
                    var_names=numeric_cols,
                    threshold=DIBS_THRESHOLD
                )

                dml_time = time.time() - dml_start
                print(f"\n  ✓ DML完成，耗时: {dml_time/60:.1f}分钟")

                if causal_effects:
                    engine.save_results(effects_file)
                    print(f"  ✓ 因果效应已保存到: {effects_file}")

                    significant = engine.get_significant_effects()
                    print(f"\n  因果效应统计:")
                    print(f"    总边数: {len(causal_effects)}")
                    print(f"    统计显著(p<0.05): {len(significant)}")

                    if significant:
                        print(f"\n  显著的因果效应 (前5个):")
                        for i, (edge, result) in enumerate(list(significant.items())[:5], 1):
                            print(f"    {i}. {edge}")
                            print(f"       ATE={result['ate']:.4f}, CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], p={result.get('p_value', 'N/A')}")
                else:
                    print(f"  ⚠️  DML未返回有效结果")

            except Exception as e:
                print(f"  ✗ DML失败: {e}")
                import traceback
                traceback.print_exc()
                failed_tasks.append({'task': task_name, 'reason': f'DML失败: {e}'})
                continue
    else:
        print(f"\n  [步骤3] DML因果推断: 跳过（无有效因果边）")

    # ------------------------------------------------------------------------
    # 任务完成
    # ------------------------------------------------------------------------
    task_time = time.time() - task_start_time
    print(f"\n  ✓ 任务组 '{display_name}' 完成，耗时: {task_time/60:.1f}分钟")

    completed_tasks.append({
        'task': task_name,
        'display_name': display_name,
        'samples': len(df),
        'features': len(numeric_cols),
        'causal_edges': len(edges) if 'edges' in locals() else 0,
        'time_minutes': task_time / 60
    })

    # 保存中间进度
    progress_file = 'logs/energy_research/experiments/dibs_progress.txt'
    with open(progress_file, 'w') as f:
        f.write(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"已完成: {len(completed_tasks)}/{len(TASK_GROUPS)}\n")
        f.write(f"当前任务: {task_name}\n")

    # 进度报告
    elapsed_total = time.time() - start_time
    if task_idx < len(TASK_GROUPS):
        avg_time_per_task = elapsed_total / task_idx
        eta = avg_time_per_task * (len(TASK_GROUPS) - task_idx) / 60
        print(f"\n  总体进度: {task_idx}/{len(TASK_GROUPS)}, ETA: {eta:.1f}分钟")

# ============================================================================
# 总结
# ============================================================================
total_time = time.time() - start_time

print_header("分析完成！")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")

print(f"\n任务组完成情况:")
print(f"  成功: {len(completed_tasks)}/{len(TASK_GROUPS)}")
print(f"  失败: {len(failed_tasks)}/{len(TASK_GROUPS)}")

if completed_tasks:
    print(f"\n成功的任务组:")
    for task in completed_tasks:
        print(f"  ✓ {task['display_name']} ({task['task']})")
        print(f"    样本数: {task['samples']}, 特征数: {task['features']}, 因果边: {task['causal_edges']}")
        print(f"    耗时: {task['time_minutes']:.1f}分钟")

if failed_tasks:
    print(f"\n失败的任务组:")
    for task in failed_tasks:
        print(f"  ✗ {task['task']}: {task['reason']}")

print(f"\n生成的文件:")
for task in TASK_GROUPS:
    task_name = task['name']
    files = [
        f"results/energy_research/task_specific/{task_name}_causal_graph.npy",
        f"results/energy_research/task_specific/{task_name}_causal_edges.pkl",
        f"results/energy_research/task_specific/{task_name}_causal_effects.csv"
    ]
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"  ✓ {file} ({size:.1f} KB)")

print(f"\n查看结果示例:")
print(f"  # 因果图")
print(f"  python -c \"import numpy as np; g=np.load('results/energy_research/task_specific/image_classification_causal_graph.npy'); print('因果图形状:', g.shape)\"")
print(f"  # 因果效应")
print(f"  python -c \"import pandas as pd; df=pd.read_csv('results/energy_research/task_specific/image_classification_causal_effects.csv'); print(df.head())\"")

print("\n" + "="*70)

# 保存最终摘要
summary_file = 'results/energy_research/task_specific/analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write(f"能耗研究因果分析摘要\n")
    f.write(f"{'='*70}\n")
    f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总运行时间: {total_time/60:.1f}分钟\n\n")
    f.write(f"成功任务组: {len(completed_tasks)}/{len(TASK_GROUPS)}\n")
    for task in completed_tasks:
        f.write(f"  - {task['display_name']}: {task['causal_edges']}条因果边, {task['time_minutes']:.1f}分钟\n")
    if failed_tasks:
        f.write(f"\n失败任务组: {len(failed_tasks)}/{len(TASK_GROUPS)}\n")
        for task in failed_tasks:
            f.write(f"  - {task['task']}: {task['reason']}\n")

print(f"摘要已保存: {summary_file}")
