#!/usr/bin/env python3
"""
分层抽样策略深度分析
Stratified Sampling Strategy Analysis for Concurrent Training Study
"""
import json
import random
from collections import defaultdict
from pathlib import Path

# Model data (simplified from previous analysis)
model_data = {
    'examples_mnist': {'memory': 450, 'gpu_util': 12.0, 'duration': 155, 'category': 'low', 'domain': 'vision', 'arch': 'cnn', 'task': 'classification'},
    'examples_mnist_rnn': {'memory': 450, 'gpu_util': 50, 'duration': 180, 'category': 'low', 'domain': 'vision', 'arch': 'rnn', 'task': 'classification'},
    'examples_mnist_ff': {'memory': 450, 'gpu_util': 50, 'duration': 180, 'category': 'low', 'domain': 'vision', 'arch': 'mlp', 'task': 'classification'},
    'pytorch_resnet_cifar10_resnet20': {'memory': 720, 'gpu_util': 50, 'duration': 300, 'category': 'low', 'domain': 'vision', 'arch': 'resnet', 'task': 'classification'},
    'pytorch_resnet_cifar10_resnet32': {'memory': 900, 'gpu_util': 50, 'duration': 350, 'category': 'low', 'domain': 'vision', 'arch': 'resnet', 'task': 'classification'},
    'pytorch_resnet_cifar10_resnet44': {'memory': 1080, 'gpu_util': 50, 'duration': 400, 'category': 'low', 'domain': 'vision', 'arch': 'resnet', 'task': 'classification'},
    'pytorch_resnet_cifar10_resnet56': {'memory': 1350, 'gpu_util': 50, 'duration': 500, 'category': 'low', 'domain': 'vision', 'arch': 'resnet', 'task': 'classification'},
    'Person_reID_baseline_pytorch_densenet121': {'memory': 3300, 'gpu_util': 71.9, 'duration': 2329, 'category': 'high', 'domain': 'vision', 'arch': 'densenet', 'task': 'reid'},
    'Person_reID_baseline_pytorch_hrnet18': {'memory': 2250, 'gpu_util': 50, 'duration': 1500, 'category': 'medium', 'domain': 'vision', 'arch': 'hrnet', 'task': 'reid'},
    'Person_reID_baseline_pytorch_pcb': {'memory': 1800, 'gpu_util': 50, 'duration': 1200, 'category': 'medium', 'domain': 'vision', 'arch': 'pcb', 'task': 'reid'},
    'MRT-OAST_default': {'memory': 1950, 'gpu_util': 92.9, 'duration': 1386, 'category': 'medium', 'domain': 'vision', 'arch': 'custom', 'task': 'tracking'},
    'VulBERTa_mlp': {'memory': 1350, 'gpu_util': 50, 'duration': 600, 'category': 'low', 'domain': 'code', 'arch': 'transformer', 'task': 'vulnerability'},
    'VulBERTa_cnn': {'memory': 1350, 'gpu_util': 50, 'duration': 600, 'category': 'low', 'domain': 'code', 'arch': 'cnn', 'task': 'vulnerability'},
    'bug-localization-by-dnn-and-rvsm_default': {'memory': 1800, 'gpu_util': 50, 'duration': 800, 'category': 'medium', 'domain': 'code', 'arch': 'dnn', 'task': 'localization'},
    'examples_siamese': {'memory': 1350, 'gpu_util': 50, 'duration': 400, 'category': 'low', 'domain': 'vision', 'arch': 'siamese', 'task': 'similarity'},
    'examples_word_lm': {'memory': 1350, 'gpu_util': 50, 'duration': 500, 'category': 'low', 'domain': 'nlp', 'arch': 'rnn', 'task': 'lm'},
}

# Generate all feasible pairs
all_pairs = []
models = list(model_data.keys())

for i, model1 in enumerate(models):
    for model2 in models[i+1:]:
        data1 = model_data[model1]
        data2 = model_data[model2]

        total_memory = data1['memory'] + data2['memory']
        if total_memory <= 9000:
            total_gpu_util = data1['gpu_util'] + data2['gpu_util']

            all_pairs.append({
                'model1': model1,
                'model2': model2,
                'data1': data1,
                'data2': data2,
                'total_memory': total_memory,
                'total_gpu_util': total_gpu_util,
            })

print("="*100)
print("分层抽样策略深度分析")
print("Stratified Sampling Strategy Deep Analysis")
print("="*100)
print(f"\n总可行组合数: {len(all_pairs)}")

print("\n" + "="*100)
print("策略1: 按显存占用分层 (Memory-based Stratification)")
print("="*100)
print("""
原理: 根据模型组合的总显存占用分层
理由: 显存是并发训练的关键资源约束

分层标准:
  - 超低显存层 (<1.5GB): 两个小模型
  - 低显存层 (1.5-3GB): 一大一小或两个中等
  - 中显存层 (3-5GB): 一大一中
  - 高显存层 (5-7GB): 两个大模型或大+中
  - 极高显存层 (>7GB): 接近硬件上限

优点: 直接对应硬件约束，OOM风险分层明确
缺点: 未考虑GPU算力竞争
""")

memory_strata = {
    'ultra_low': [],  # <1500MB
    'low': [],        # 1500-3000MB
    'medium': [],     # 3000-5000MB
    'high': [],       # 5000-7000MB
    'ultra_high': [], # >7000MB
}

for pair in all_pairs:
    mem = pair['total_memory']
    if mem < 1500:
        memory_strata['ultra_low'].append(pair)
    elif mem < 3000:
        memory_strata['low'].append(pair)
    elif mem < 5000:
        memory_strata['medium'].append(pair)
    elif mem < 7000:
        memory_strata['high'].append(pair)
    else:
        memory_strata['ultra_high'].append(pair)

print("\n各层统计:")
for stratum, pairs in memory_strata.items():
    print(f"  {stratum:<15} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
sample_sizes = {
    'ultra_low': 2,  # 20个 -> 2个
    'low': 4,        # 65个 -> 4个
    'medium': 4,     # 27个 -> 4个
    'high': 2,       # 8个 -> 2个
    'ultra_high': 0, # 0个 -> 0个
}

memory_sample = []
random.seed(42)
for stratum, target in sample_sizes.items():
    if memory_strata[stratum] and target > 0:
        sample = random.sample(memory_strata[stratum], min(target, len(memory_strata[stratum])))
        memory_sample.extend(sample)
        print(f"  {stratum:<15} 抽取{len(sample)}个")

print(f"\n实际抽样结果 ({len(memory_sample)}个):")
for i, pair in enumerate(memory_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45} [{pair['total_memory']:4}MB]")

print("\n" + "="*100)
print("策略2: 按GPU利用率分层 (GPU Utilization-based Stratification)")
print("="*100)
print("""
原理: 根据GPU利用率总和分层
理由: GPU算力竞争影响训练效率和能耗

分层标准:
  - 低竞争层 (<80%): GPU资源充裕，几乎无竞争
  - 中竞争层 (80-120%): 适度竞争，可接受
  - 高竞争层 (120-150%): 明显竞争，会相互拖慢
  - 极高竞争层 (>150%): 严重竞争，可能不如串行

优点: 直接反映资源竞争程度
缺点: 估算可能不准确
""")

gpu_strata = {
    'low_competition': [],     # <80%
    'medium_competition': [],  # 80-120%
    'high_competition': [],    # 120-150%
    'extreme_competition': [], # >150%
}

for pair in all_pairs:
    gpu = pair['total_gpu_util']
    if gpu < 80:
        gpu_strata['low_competition'].append(pair)
    elif gpu < 120:
        gpu_strata['medium_competition'].append(pair)
    elif gpu < 150:
        gpu_strata['high_competition'].append(pair)
    else:
        gpu_strata['extreme_competition'].append(pair)

print("\n各层统计:")
for stratum, pairs in gpu_strata.items():
    print(f"  {stratum:<25} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
gpu_sample_sizes = {
    'low_competition': 4,
    'medium_competition': 5,
    'high_competition': 2,
    'extreme_competition': 1,
}

gpu_sample = []
for stratum, target in gpu_sample_sizes.items():
    if gpu_strata[stratum] and target > 0:
        sample = random.sample(gpu_strata[stratum], min(target, len(gpu_strata[stratum])))
        gpu_sample.extend(sample)
        print(f"  {stratum:<25} 抽取{len(sample)}个")

print(f"\n实际抽样结果 ({len(gpu_sample)}个):")
for i, pair in enumerate(gpu_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45} [GPU:{pair['total_gpu_util']:5.1f}%]")

print("\n" + "="*100)
print("策略3: 按训练时长分层 (Duration-based Stratification)")
print("="*100)
print("""
原理: 根据预期训练时长分层
理由: 不同时长组合对实验设计影响不同

分层标准:
  - 快速层 (<500s): 快速验证，适合大量测试
  - 中速层 (500-1500s): 平衡速度和代表性
  - 慢速层 (1500-3000s): 大模型，训练时间长
  - 极慢层 (>3000s): 超长训练，资源密集

优点: 便于实验时间管理
缺点: 与研究目标关联较弱
""")

duration_strata = {
    'fast': [],      # <500s
    'medium': [],    # 500-1500s
    'slow': [],      # 1500-3000s
    'ultra_slow': [],# >3000s
}

for pair in all_pairs:
    avg_duration = (pair['data1']['duration'] + pair['data2']['duration']) / 2
    if avg_duration < 500:
        duration_strata['fast'].append(pair)
    elif avg_duration < 1500:
        duration_strata['medium'].append(pair)
    elif avg_duration < 3000:
        duration_strata['slow'].append(pair)
    else:
        duration_strata['ultra_slow'].append(pair)

print("\n各层统计:")
for stratum, pairs in duration_strata.items():
    print(f"  {stratum:<15} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
duration_sample_sizes = {
    'fast': 4,
    'medium': 5,
    'slow': 3,
    'ultra_slow': 0,
}

duration_sample = []
for stratum, target in duration_sample_sizes.items():
    if duration_strata[stratum] and target > 0:
        sample = random.sample(duration_strata[stratum], min(target, len(duration_strata[stratum])))
        duration_sample.extend(sample)
        print(f"  {stratum:<15} 抽取{len(sample)}个")

print("\n" + "="*100)
print("策略4: 按应用领域分层 (Domain-based Stratification)")
print("="*100)
print("""
原理: 根据模型所属应用领域组合分层
理由: 不同领域模型的并发特性可能不同

领域分类:
  - vision: 计算机视觉 (MNIST, ResNet, DenseNet, HRNet, etc.)
  - nlp: 自然语言处理 (Word LM, etc.)
  - code: 代码分析 (VulBERTa, Bug Localization)

分层标准:
  - vision+vision: 同领域，数据加载模式相似
  - vision+nlp: 跨领域，互补性强
  - vision+code: 跨领域，I/O模式可能不同
  - code+code: 同领域
  - code+nlp: 跨领域

优点: 考虑应用场景多样性
缺点: 某些组合样本少
""")

domain_strata = defaultdict(list)

for pair in all_pairs:
    domain1 = pair['data1']['domain']
    domain2 = pair['data2']['domain']
    # Sort to avoid duplicates like vision+nlp vs nlp+vision
    domain_combo = '+'.join(sorted([domain1, domain2]))
    domain_strata[domain_combo].append(pair)

print("\n各层统计:")
for stratum, pairs in sorted(domain_strata.items()):
    print(f"  {stratum:<20} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
domain_sample_sizes = {
    'vision+vision': 8,  # 主要领域
    'code+vision': 2,    # 跨领域
    'nlp+vision': 1,     # 跨领域
    'code+code': 1,      # 同领域
}

domain_sample = []
for stratum, target in domain_sample_sizes.items():
    if stratum in domain_strata and target > 0:
        sample = random.sample(domain_strata[stratum], min(target, len(domain_strata[stratum])))
        domain_sample.extend(sample)
        print(f"  {stratum:<20} 抽取{len(sample)}个")

print("\n" + "="*100)
print("策略5: 按互补性分层 (Complementarity-based Stratification)")
print("="*100)
print("""
原理: 根据模型间资源利用的互补程度分层
理由: 互补性影响并发效率和能耗

互补性定义:
  互补分数 = |GPU_util_1 - GPU_util_2| / 100

分层标准:
  - 高度互补层 (分数>0.6): 一高一低，最优配对
  - 中度互补层 (0.3-0.6): 有一定差异
  - 低度互补层 (0.1-0.3): 相似利用率
  - 无互补层 (<0.1): 几乎相同

优点: 直接针对并发效率优化
缺点: 高度互补的组合可能较少
""")

complementarity_strata = {
    'high_complementary': [],   # >0.6
    'medium_complementary': [], # 0.3-0.6
    'low_complementary': [],    # 0.1-0.3
    'no_complementary': [],     # <0.1
}

for pair in all_pairs:
    gpu_diff = abs(pair['data1']['gpu_util'] - pair['data2']['gpu_util'])
    comp_score = gpu_diff / 100

    if comp_score > 0.6:
        complementarity_strata['high_complementary'].append(pair)
    elif comp_score > 0.3:
        complementarity_strata['medium_complementary'].append(pair)
    elif comp_score > 0.1:
        complementarity_strata['low_complementary'].append(pair)
    else:
        complementarity_strata['no_complementary'].append(pair)

print("\n各层统计:")
for stratum, pairs in complementarity_strata.items():
    print(f"  {stratum:<25} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
comp_sample_sizes = {
    'high_complementary': 4,   # 重点研究
    'medium_complementary': 4,
    'low_complementary': 3,
    'no_complementary': 1,
}

comp_sample = []
for stratum, target in comp_sample_sizes.items():
    if complementarity_strata[stratum] and target > 0:
        sample = random.sample(complementarity_strata[stratum], min(target, len(complementarity_strata[stratum])))
        comp_sample.extend(sample)
        print(f"  {stratum:<25} 抽取{len(sample)}个")

print("\n" + "="*100)
print("策略6: 按模型架构分层 (Architecture-based Stratification)")
print("="*100)
print("""
原理: 根据模型架构类型组合分层
理由: 不同架构的并发特性可能不同

架构分类:
  - CNN: 卷积神经网络 (ResNet, DenseNet, etc.)
  - RNN: 循环神经网络 (LSTM, etc.)
  - Transformer: 注意力机制 (VulBERTa, etc.)
  - MLP: 全连接网络
  - Custom: 自定义架构

分层标准:
  - CNN+CNN: 同类架构，计算模式相似
  - CNN+RNN: 异构，互补性可能强
  - CNN+Transformer: 异构
  - 等等...

优点: 反映底层计算模式差异
缺点: 分层过细，某些组合样本少
""")

arch_strata = defaultdict(list)

for pair in all_pairs:
    arch1 = pair['data1']['arch']
    arch2 = pair['data2']['arch']
    arch_combo = '+'.join(sorted([arch1, arch2]))
    arch_strata[arch_combo].append(pair)

print("\n各层统计:")
for stratum, pairs in sorted(arch_strata.items()):
    print(f"  {stratum:<30} {len(pairs):>3}个组合")

print("\n抽样方案示例 (简化为主要类别):")
# Simplify to main categories
simplified_arch_strata = {
    'cnn+cnn': [],
    'cnn+other': [],
    'other+other': [],
}

for pair in all_pairs:
    arch1 = pair['data1']['arch']
    arch2 = pair['data2']['arch']

    if arch1 == 'cnn' or arch1 in ['resnet', 'densenet']:
        arch1 = 'cnn'
    else:
        arch1 = 'other'

    if arch2 == 'cnn' or arch2 in ['resnet', 'densenet']:
        arch2 = 'cnn'
    else:
        arch2 = 'other'

    if arch1 == 'cnn' and arch2 == 'cnn':
        simplified_arch_strata['cnn+cnn'].append(pair)
    elif arch1 == 'cnn' or arch2 == 'cnn':
        simplified_arch_strata['cnn+other'].append(pair)
    else:
        simplified_arch_strata['other+other'].append(pair)

for stratum, pairs in simplified_arch_strata.items():
    print(f"  {stratum:<20} {len(pairs):>3}个组合")

print("\n" + "="*100)
print("策略7: 多维度交叉分层 (Multi-dimensional Stratification)")
print("="*100)
print("""
原理: 同时考虑多个维度进行交叉分层
理由: 单一维度可能不足以捕捉复杂的并发特性

推荐组合:
  维度1 (主): 显存占用 (low/medium/high)
  维度2 (次): GPU利用率互补性 (complementary/competitive)

分层矩阵:
  ┌─────────────┬──────────────┬──────────────┐
  │             │ Complementary│ Competitive  │
  ├─────────────┼──────────────┼──────────────┤
  │ Low Memory  │  优先抽样     │  适量抽样     │
  │ Med Memory  │  优先抽样     │  少量抽样     │
  │ High Memory │  适量抽样     │  必须抽样     │
  └─────────────┴──────────────┴──────────────┘

优点: 全面考虑多个因素，覆盖更均衡
缺点: 分层复杂，某些格子可能样本少
""")

# Create 2D stratification
multi_strata = defaultdict(list)

for pair in all_pairs:
    # Dimension 1: Memory
    mem = pair['total_memory']
    if mem < 2000:
        mem_cat = 'low'
    elif mem < 4000:
        mem_cat = 'medium'
    else:
        mem_cat = 'high'

    # Dimension 2: Complementarity
    gpu_diff = abs(pair['data1']['gpu_util'] - pair['data2']['gpu_util'])
    if gpu_diff > 40:
        comp_cat = 'complementary'
    else:
        comp_cat = 'competitive'

    key = f"{mem_cat}_{comp_cat}"
    multi_strata[key].append(pair)

print("\n各格子统计:")
for stratum in ['low_complementary', 'low_competitive',
                'medium_complementary', 'medium_competitive',
                'high_complementary', 'high_competitive']:
    count = len(multi_strata[stratum])
    print(f"  {stratum:<25} {count:>3}个组合")

print("\n抽样方案 (总12个):")
multi_sample_sizes = {
    'low_complementary': 3,
    'low_competitive': 2,
    'medium_complementary': 3,
    'medium_competitive': 2,
    'high_complementary': 1,
    'high_competitive': 1,
}

multi_sample = []
for stratum, target in multi_sample_sizes.items():
    if multi_strata[stratum] and target > 0:
        sample = random.sample(multi_strata[stratum], min(target, len(multi_strata[stratum])))
        multi_sample.extend(sample)
        print(f"  {stratum:<25} 抽取{len(sample)}个")

print("\n实际抽样结果:")
for i, pair in enumerate(multi_sample, 1):
    print(f"  {i:2}. {pair['model1']:<45} + {pair['model2']:<45}")

print("\n" + "="*100)
print("策略8: 按风险等级分层 (Risk-based Stratification)")
print("="*100)
print("""
原理: 根据并发风险程度分层
理由: 平衡探索性和稳妥性

风险评估因素:
  - 显存接近上限 (>7GB): 高风险
  - GPU竞争激烈 (>150%): 中风险
  - 无历史数据: 中风险
  - 训练时间极长 (>3000s): 低风险但成本高

分层标准:
  - 低风险层: 显存<3GB, GPU<120%, 有数据
  - 中风险层: 显存3-5GB 或 GPU 120-150%
  - 高风险层: 显存>5GB 或 GPU>150% 或 无数据

优点: 便于实验资源和风险管理
缺点: 可能错过高风险但有价值的组合
""")

risk_strata = {
    'low_risk': [],
    'medium_risk': [],
    'high_risk': [],
}

for pair in all_pairs:
    risk_score = 0

    # Memory risk
    if pair['total_memory'] > 7000:
        risk_score += 3
    elif pair['total_memory'] > 5000:
        risk_score += 2
    elif pair['total_memory'] > 3000:
        risk_score += 1

    # GPU competition risk
    if pair['total_gpu_util'] > 150:
        risk_score += 2
    elif pair['total_gpu_util'] > 120:
        risk_score += 1

    # Assign to strata
    if risk_score <= 1:
        risk_strata['low_risk'].append(pair)
    elif risk_score <= 3:
        risk_strata['medium_risk'].append(pair)
    else:
        risk_strata['high_risk'].append(pair)

print("\n各层统计:")
for stratum, pairs in risk_strata.items():
    print(f"  {stratum:<15} {len(pairs):>3}个组合")

print("\n抽样方案 (总12个):")
risk_sample_sizes = {
    'low_risk': 6,      # 保守为主
    'medium_risk': 4,   # 适度探索
    'high_risk': 2,     # 少量高风险
}

risk_sample = []
for stratum, target in risk_sample_sizes.items():
    if risk_strata[stratum] and target > 0:
        sample = random.sample(risk_strata[stratum], min(target, len(risk_strata[stratum])))
        risk_sample.extend(sample)
        print(f"  {stratum:<15} 抽取{len(sample)}个")

print("\n" + "="*100)
print("📊 各策略对比总结")
print("="*100)

strategies = [
    ("显存占用分层", "直接对应硬件约束", "未考虑GPU竞争", "⭐⭐⭐⭐⭐", "资源受限场景"),
    ("GPU利用率分层", "反映资源竞争", "估算可能不准", "⭐⭐⭐⭐", "性能优化场景"),
    ("训练时长分层", "便于时间管理", "关联较弱", "⭐⭐⭐", "时间敏感场景"),
    ("应用领域分层", "考虑应用多样性", "某些组合少", "⭐⭐⭐⭐", "泛化性研究"),
    ("互补性分层", "针对并发效率", "互补组合少", "⭐⭐⭐⭐⭐", "效率优化场景"),
    ("模型架构分层", "反映计算模式", "分层过细", "⭐⭐⭐", "架构研究"),
    ("多维交叉分层", "全面均衡", "复杂度高", "⭐⭐⭐⭐⭐", "全面研究"),
    ("风险等级分层", "便于风险管理", "可能保守", "⭐⭐⭐⭐", "稳妥实施"),
]

print(f"\n{'策略':<18} {'优点':<22} {'缺点':<18} {'推荐度':<12} {'适用场景':<15}")
print("-" * 95)
for name, pros, cons, rating, scenario in strategies:
    print(f"{name:<15} {pros:<20} {cons:<15} {rating:<10} {scenario:<12}")

print("\n" + "="*100)
print("💡 最终推荐")
print("="*100)
print("""
综合推荐策略: 多维交叉分层 (显存 × 互补性)

理由:
1. ✅ 显存是硬件硬约束，必须考虑
2. ✅ 互补性直接影响并发效率，是研究核心
3. ✅ 两个维度交叉覆盖最全面
4. ✅ 6个格子易于管理和解释

备选策略（根据研究侧重）:
- 如果重点研究资源利用: 优先"显存+GPU利用率分层"
- 如果重点研究应用场景: 优先"应用领域分层"
- 如果追求稳妥实施: 优先"风险等级分层"
- 如果时间有限: 优先"训练时长分层"

混合使用:
可以先用"多维交叉分层"划分大类，
再在每个格子内用"实用性原则"（优先有数据、安全）选择具体组合。
""")

print("\n分析完成！")
print("="*100)
