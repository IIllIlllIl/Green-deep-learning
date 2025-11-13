#!/usr/bin/env python3
"""
并发训练超参数影响实验设计 - 模型组合选取方法分析
Concurrent Training Hyperparameter Study - Model Pair Sampling Analysis
"""
import json
import random
from pathlib import Path
from itertools import combinations
import math

# Load model data
with open('config/models_config.json') as f:
    config = json.load(f)

# Load concurrent feasibility analysis results
# Simplified model data based on previous analysis
model_data = {
    'examples_mnist': {'memory': 450, 'gpu_util': 12.0, 'duration': 155, 'category': 'low', 'success_count': 16},
    'examples_mnist_rnn': {'memory': 450, 'gpu_util': 50, 'duration': 180, 'category': 'low', 'success_count': 0},
    'examples_mnist_ff': {'memory': 450, 'gpu_util': 50, 'duration': 180, 'category': 'low', 'success_count': 0},
    'pytorch_resnet_cifar10_resnet20': {'memory': 720, 'gpu_util': 50, 'duration': 300, 'category': 'low', 'success_count': 0},
    'pytorch_resnet_cifar10_resnet32': {'memory': 900, 'gpu_util': 50, 'duration': 350, 'category': 'low', 'success_count': 0},
    'pytorch_resnet_cifar10_resnet44': {'memory': 1080, 'gpu_util': 50, 'duration': 400, 'category': 'low', 'success_count': 0},
    'pytorch_resnet_cifar10_resnet56': {'memory': 1350, 'gpu_util': 50, 'duration': 500, 'category': 'low', 'success_count': 0},
    'Person_reID_baseline_pytorch_densenet121': {'memory': 3300, 'gpu_util': 71.9, 'duration': 2329, 'category': 'high', 'success_count': 7},
    'Person_reID_baseline_pytorch_hrnet18': {'memory': 2250, 'gpu_util': 50, 'duration': 1500, 'category': 'medium', 'success_count': 0},
    'Person_reID_baseline_pytorch_pcb': {'memory': 1800, 'gpu_util': 50, 'duration': 1200, 'category': 'medium', 'success_count': 0},
    'MRT-OAST_default': {'memory': 1950, 'gpu_util': 92.9, 'duration': 1386, 'category': 'medium', 'success_count': 9},
    'VulBERTa_mlp': {'memory': 1350, 'gpu_util': 50, 'duration': 600, 'category': 'low', 'success_count': 0},
    'VulBERTa_cnn': {'memory': 1350, 'gpu_util': 50, 'duration': 600, 'category': 'low', 'success_count': 0},
    'bug-localization-by-dnn-and-rvsm_default': {'memory': 1800, 'gpu_util': 50, 'duration': 800, 'category': 'medium', 'success_count': 0},
    'examples_siamese': {'memory': 1350, 'gpu_util': 50, 'duration': 400, 'category': 'low', 'success_count': 0},
    'examples_word_lm': {'memory': 1350, 'gpu_util': 50, 'duration': 500, 'category': 'low', 'success_count': 0},
}

print("="*100)
print("并发训练超参数影响研究 - 模型组合选取方法分析")
print("Concurrent Training Hyperparameter Study - Sampling Method Analysis")
print("="*100)

print("\n📋 研究目标:")
print("""
1. 研究并发训练时，超参数变异对能耗和性能的影响
2. 对比：单独训练 vs 并发训练下的超参数敏感性
3. 发现：并发是否改变超参数-能耗-性能的关系
""")

print("\n🔬 实验设计:")
print("""
对于每个模型组合 (ModelA, ModelB):
  - 控制组: ModelA固定默认超参数 + ModelB固定默认超参数 (并发)
  - 实验组: ModelA固定默认超参数 + ModelB变异超参数 (并发)
  - 对照组: ModelB变异超参数 (单独训练)

测量指标:
  - 能耗: GPU能耗、CPU能耗、总能耗
  - 性能: 准确率、mAP等
  - 时间: 训练时长

分析维度:
  - 并发对超参数敏感性的影响
  - 不同模型组合下的能耗干扰
  - 最优超参数在并发场景下是否改变
""")

# Generate all feasible pairs
all_pairs = []
models = list(model_data.keys())

for i, model1 in enumerate(models):
    for model2 in models[i+1:]:
        data1 = model_data[model1]
        data2 = model_data[model2]

        total_memory = data1['memory'] + data2['memory']
        total_gpu_util = data1['gpu_util'] + data2['gpu_util']

        # Feasibility check
        if total_memory <= 9000:  # 9GB safe limit
            is_complementary = (data1['gpu_util'] > 70 and data2['gpu_util'] < 30) or \
                             (data1['gpu_util'] < 30 and data2['gpu_util'] > 70)

            is_safe = total_gpu_util < 150

            # Calculate diversity score
            memory_diff = abs(data1['memory'] - data2['memory'])
            gpu_diff = abs(data1['gpu_util'] - data2['gpu_util'])

            all_pairs.append({
                'model1': model1,
                'model2': model2,
                'total_memory': total_memory,
                'total_gpu_util': total_gpu_util,
                'is_complementary': is_complementary,
                'is_safe': is_safe,
                'category_combo': f"{data1['category']}+{data2['category']}",
                'memory_diff': memory_diff,
                'gpu_diff': gpu_diff,
                'has_data': data1['success_count'] > 0 or data2['success_count'] > 0,
                'both_have_data': data1['success_count'] > 0 and data2['success_count'] > 0,
            })

print(f"\n📊 可行的模型组合总数: {len(all_pairs)}")

# Analyze category combinations
category_counts = {}
for pair in all_pairs:
    cat = pair['category_combo']
    category_counts[cat] = category_counts.get(cat, 0) + 1

print(f"\n按类别组合分布:")
for cat, count in sorted(category_counts.items()):
    print(f"  {cat:<20} {count:>3}个组合")

print("\n" + "="*100)
print("方法1: 随机抽样 (Random Sampling)")
print("="*100)
print("""
原理: 从所有可行组合中完全随机选取12个
优点:
  ✅ 无偏差，统计意义明确
  ✅ 简单易实现
  ✅ 可重复（设置随机种子）
缺点:
  ❌ 可能遗漏重要组合
  ❌ 可能集中在某些类型
  ❌ 不保证覆盖多样性
适用场景: 初步探索，样本量大时
""")

random.seed(42)
random_sample = random.sample(all_pairs, 12)
print("\n随机抽样结果 (seed=42):")
for i, pair in enumerate(random_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45} "
          f"[{pair['category_combo']:15}] {pair['total_memory']:4}MB")

print("\n" + "="*100)
print("方法2: 分层抽样 (Stratified Sampling)")
print("="*100)
print("""
原理: 按模型组合类型分层，从每层按比例抽取
优点:
  ✅ 保证各类型组合都有代表
  ✅ 覆盖度高
  ✅ 统计可靠性好
缺点:
  ❌ 需要事先定义分层标准
  ❌ 可能错过极端案例
适用场景: 需要全面覆盖不同类型时
""")

# Stratify by category combination
stratified_sample = []
target_per_category = {
    'low+low': 3,
    'low+medium': 3,
    'low+high': 2,
    'medium+medium': 2,
    'medium+high': 2,
}

for cat, target in target_per_category.items():
    cat_pairs = [p for p in all_pairs if p['category_combo'] == cat]
    if cat_pairs:
        sample_size = min(target, len(cat_pairs))
        random.seed(42)
        stratified_sample.extend(random.sample(cat_pairs, sample_size))

print(f"\n分层抽样结果 (目标12个，实际{len(stratified_sample)}个):")
for i, pair in enumerate(stratified_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45} "
          f"[{pair['category_combo']:15}] {pair['total_memory']:4}MB")

print("\n" + "="*100)
print("方法3: 代表性抽样 (Representative Sampling)")
print("="*100)
print("""
原理: 选择能代表不同维度的典型组合
维度:
  - 显存: 低+低, 低+中, 低+高, 中+中, 中+高
  - GPU利用率: 互补型, 均衡型, 竞争型
  - 训练时长: 快+快, 快+慢, 慢+慢
优点:
  ✅ 覆盖关键场景
  ✅ 结果可解释性强
  ✅ 针对性强
缺点:
  ❌ 主观性较强
  ❌ 可能遗漏未知模式
适用场景: 需要深入理解特定场景时
""")

# Select representative samples
representative_sample = []

# 1. Best complementary pairs (high + low GPU)
complementary = [p for p in all_pairs if p['is_complementary']]
if complementary:
    representative_sample.append(max(complementary, key=lambda x: x['gpu_diff']))  # Most complementary

# 2. Safe balanced pairs (both medium)
balanced = [p for p in all_pairs if 40 < p['total_gpu_util'] < 120 and p['is_safe']]
if balanced:
    representative_sample.append(balanced[0])

# 3. High competition pair (both high GPU, but safe memory)
competitive = [p for p in all_pairs if p['total_gpu_util'] > 120 and p['is_safe']]
if competitive:
    representative_sample.append(competitive[0])

# 4. Extreme memory difference
memory_diverse = sorted(all_pairs, key=lambda x: x['memory_diff'], reverse=True)
representative_sample.append(memory_diverse[0])

# 5. Similar memory
memory_similar = sorted(all_pairs, key=lambda x: x['memory_diff'])
representative_sample.append(memory_similar[0])

# 6-12. Fill with diverse category combinations
remaining_categories = set(['low+low', 'low+medium', 'low+high', 'medium+medium', 'medium+high'])
for cat in remaining_categories:
    cat_pairs = [p for p in all_pairs if p['category_combo'] == cat and p not in representative_sample]
    if cat_pairs and len(representative_sample) < 12:
        # Prefer pairs with existing data
        with_data = [p for p in cat_pairs if p['has_data']]
        if with_data:
            representative_sample.append(with_data[0])
        else:
            representative_sample.append(cat_pairs[0])

# Fill remaining spots
while len(representative_sample) < 12:
    remaining = [p for p in all_pairs if p not in representative_sample]
    if not remaining:
        break
    # Prefer pairs with data
    with_data = [p for p in remaining if p['has_data']]
    if with_data:
        representative_sample.append(with_data[0])
    else:
        representative_sample.append(remaining[0])

print(f"\n代表性抽样结果 ({len(representative_sample)}个):")
for i, pair in enumerate(representative_sample, 1):
    complementary_mark = "✅互补" if pair['is_complementary'] else ""
    safe_mark = "✅安全" if pair['is_safe'] else "⚠️竞争"
    data_mark = "📊有数据" if pair['has_data'] else "❓无数据"
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45}")
    print(f"      [{pair['category_combo']:15}] {pair['total_memory']:4}MB, GPU:{pair['total_gpu_util']:.1f}% "
          f"{complementary_mark} {safe_mark} {data_mark}")

print("\n" + "="*100)
print("方法4: 正交设计 (Orthogonal Design)")
print("="*100)
print("""
原理: 系统性地覆盖多个因素的不同水平组合
因素:
  - 因素A (显存): 低 (L1), 中 (L2), 高 (L3)
  - 因素B (GPU利用率): 低 (<30%), 中 (30-70%), 高 (>70%)
  - 因素C (训练时长): 快 (<500s), 中 (500-1500s), 慢 (>1500s)
优点:
  ✅ 最小实验次数获得最大信息
  ✅ 可以分析因素交互作用
  ✅ 统计效率高
缺点:
  ❌ 设计复杂
  ❌ 可能选到不现实的组合
适用场景: 需要分析多因素影响时
""")

# Simplified orthogonal design: Cover key combinations
orthogonal_sample = []

# Factor combinations (simplified L9 orthogonal array)
factor_combinations = [
    ('low', 'low', 'fast'),      # 1
    ('low', 'medium', 'medium'), # 2
    ('low', 'high', 'slow'),     # 3
    ('medium', 'low', 'medium'), # 4
    ('medium', 'medium', 'slow'),# 5
    ('medium', 'high', 'fast'),  # 6
    ('high', 'low', 'slow'),     # 7
    ('high', 'medium', 'fast'),  # 8
    ('high', 'high', 'medium'),  # 9
]

def classify_duration(duration):
    if duration < 500:
        return 'fast'
    elif duration < 1500:
        return 'medium'
    else:
        return 'slow'

def classify_gpu(gpu_util):
    if gpu_util < 30:
        return 'low'
    elif gpu_util < 70:
        return 'medium'
    else:
        return 'high'

# Match pairs to factor combinations
for mem_cat, gpu_cat, dur_cat in factor_combinations:
    candidates = []
    for pair in all_pairs:
        # Get combined characteristics
        m1_data = model_data[pair['model1']]
        m2_data = model_data[pair['model2']]

        # Average duration category
        avg_dur = (m1_data['duration'] + m2_data['duration']) / 2
        dur_class = classify_duration(avg_dur)

        # Check if matches (roughly)
        if pair['category_combo'].startswith(mem_cat) or pair['category_combo'].endswith(mem_cat):
            # Check GPU util classification
            avg_gpu = pair['total_gpu_util'] / 2
            gpu_class = classify_gpu(avg_gpu)

            if dur_class == dur_cat:
                candidates.append(pair)

    if candidates and len(orthogonal_sample) < 12:
        # Prefer pairs with data
        with_data = [p for p in candidates if p['has_data']]
        if with_data:
            orthogonal_sample.append(with_data[0])
        elif candidates:
            orthogonal_sample.append(candidates[0])

# Fill to 12 if needed
while len(orthogonal_sample) < 12:
    remaining = [p for p in all_pairs if p not in orthogonal_sample]
    if not remaining:
        break
    orthogonal_sample.append(remaining[0])

print(f"\n正交设计抽样结果 ({len(orthogonal_sample)}个):")
for i, pair in enumerate(orthogonal_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45} "
          f"[{pair['category_combo']:15}] {pair['total_memory']:4}MB")

print("\n" + "="*100)
print("方法5: 实用性抽样 (Practical Sampling)")
print("="*100)
print("""
原理: 选择实际应用中最有价值的组合
优先级:
  1. 已有训练数据的模型 (便于对比分析)
  2. 训练时长适中的组合 (不要太快或太慢)
  3. 安全的并发组合 (避免OOM或严重竞争)
  4. 不同应用领域的组合 (提高泛化性)
优点:
  ✅ 结果实用性强
  ✅ 便于后续分析
  ✅ 减少失败风险
缺点:
  ❌ 可能偏向已知模型
  ❌ 探索性不足
适用场景: 资源有限，追求稳妥结果时
""")

practical_sample = []

# Priority 1: Both models have training data
both_data = [p for p in all_pairs if p['both_have_data']]
practical_sample.extend(both_data[:4])  # Top 4

# Priority 2: At least one model has data + safe
one_data_safe = [p for p in all_pairs if p['has_data'] and p['is_safe'] and p not in practical_sample]
practical_sample.extend(one_data_safe[:4])  # Next 4

# Priority 3: Complementary pairs
complementary_remaining = [p for p in all_pairs if p['is_complementary'] and p not in practical_sample]
practical_sample.extend(complementary_remaining[:2])  # Next 2

# Priority 4: Fill with diverse safe pairs
diverse_safe = [p for p in all_pairs if p['is_safe'] and p not in practical_sample]
practical_sample.extend(diverse_safe[:2])  # Final 2

print(f"\n实用性抽样结果 ({len(practical_sample)}个):")
for i, pair in enumerate(practical_sample, 1):
    data_status = "✅✅双方有数据" if pair['both_have_data'] else ("✅单方有数据" if pair['has_data'] else "❓无数据")
    safe_status = "✅安全" if pair['is_safe'] else "⚠️竞争"
    comp_status = "⭐互补" if pair['is_complementary'] else ""
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45}")
    print(f"      [{pair['category_combo']:15}] {data_status} {safe_status} {comp_status}")

print("\n" + "="*100)
print("💡 推荐方案对比")
print("="*100)

methods = [
    ("随机抽样", "无偏差，简单", "可能遗漏关键组合", "⭐⭐⭐"),
    ("分层抽样", "覆盖全面，代表性强", "需要分层标准", "⭐⭐⭐⭐"),
    ("代表性抽样", "针对性强，可解释", "主观性强", "⭐⭐⭐⭐"),
    ("正交设计", "统计效率高", "设计复杂", "⭐⭐⭐"),
    ("实用性抽样", "稳妥，便于分析", "探索性不足", "⭐⭐⭐⭐⭐")
]

print(f"\n{'方法':<15} {'优点':<25} {'缺点':<25} {'推荐度':<10}")
print("-" * 80)
for method, pros, cons, rating in methods:
    print(f"{method:<15} {pros:<25} {cons:<25} {rating:<10}")

print("\n" + "="*100)
print("🎯 最终推荐")
print("="*100)
print("""
推荐使用: 实用性抽样 + 分层抽样混合策略

理由:
1. 实用性抽样保证实验成功率和分析可行性
2. 分层抽样保证覆盖不同类型组合
3. 两者结合兼顾稳妥性和全面性

具体方案:
- 6个组合: 实用性抽样 (优先有数据、安全、互补的组合)
- 6个组合: 分层抽样 (覆盖不同类型组合)
- 总计12个组合

预期效果:
✅ 有足够的已知模型数据作为baseline
✅ 覆盖高/中/低显存的不同组合
✅ 包含互补型和竞争型组合
✅ 降低实验失败风险
✅ 结果具有泛化性
""")

# Generate final recommendation
print("\n" + "="*100)
print("📋 最终推荐的12个模型组合")
print("="*100)

final_recommendation = []

# From practical sampling: Top 6
final_recommendation.extend(practical_sample[:6])

# From stratified sampling: 6 different ones
stratified_remaining = [p for p in stratified_sample if p not in final_recommendation]
final_recommendation.extend(stratified_remaining[:6])

print(f"\n共{len(final_recommendation)}个组合:\n")
for i, pair in enumerate(final_recommendation, 1):
    data_status = "✅✅" if pair['both_have_data'] else ("✅" if pair['has_data'] else "❓")
    safe_status = "✅" if pair['is_safe'] else "⚠️"
    comp_status = "⭐" if pair['is_complementary'] else "  "

    print(f"{i:2}. {pair['model1']:<50}")
    print(f"    + {pair['model2']:<50}")
    print(f"    类型:[{pair['category_combo']:15}] 显存:{pair['total_memory']:4}MB GPU:{pair['total_gpu_util']:5.1f}% "
          f"{data_status}{safe_status}{comp_status}")
    print()

print("图例:")
print("  ✅✅ = 双方都有训练数据")
print("  ✅   = 至少一方有训练数据")
print("  ❓   = 双方都无训练数据")
print("  ✅   = 安全并发 (GPU<150%)")
print("  ⚠️   = 有竞争 (GPU>150%)")
print("  ⭐   = GPU利用率互补")

print("\n" + "="*100)
print("分析完成！使用 -o 参数可以导出JSON配置")
print("="*100)

# Export to JSON if needed
export_data = {
    'methods': {
        'random': [{'model1': p['model1'], 'model2': p['model2']} for p in random_sample],
        'stratified': [{'model1': p['model1'], 'model2': p['model2']} for p in stratified_sample],
        'representative': [{'model1': p['model1'], 'model2': p['model2']} for p in representative_sample],
        'orthogonal': [{'model1': p['model1'], 'model2': p['model2']} for p in orthogonal_sample],
        'practical': [{'model1': p['model1'], 'model2': p['model2']} for p in practical_sample],
    },
    'recommended': [{'model1': p['model1'], 'model2': p['model2'], 'details': p} for p in final_recommendation]
}

with open('model_pair_sampling_analysis.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"\n结果已导出到: model_pair_sampling_analysis.json")
