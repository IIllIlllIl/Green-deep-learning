#!/usr/bin/env python3
"""
分析各模型的资源占用情况，评估并发训练的可行性
Analyze resource usage of each model to evaluate concurrent training feasibility
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

# Hardware configuration
GPU_TOTAL_MEMORY_MB = 10240  # RTX 3080: 10GB
CPU_CORES = 20
RAM_TOTAL_GB = 30

# Load model config to get all models
with open('config/models_config.json') as f:
    config = json.load(f)

all_models = []
for repo_name, repo_info in config['models'].items():
    for model_name in repo_info['models']:
        all_models.append({
            'repo': repo_name,
            'model': model_name,
            'key': f"{repo_name}_{model_name}"
        })

print("="*100)
print("模型并发训练可行性分析")
print("Concurrent Training Feasibility Analysis")
print("="*100)
print(f"\n硬件配置:")
print(f"  GPU: RTX 3080, {GPU_TOTAL_MEMORY_MB}MB显存")
print(f"  CPU: {CPU_CORES}核")
print(f"  RAM: {RAM_TOTAL_GB}GB\n")

# Analyze results
results_dir = Path('results')
model_stats = defaultdict(lambda: {
    'gpu_util': [], 'gpu_power': [], 'duration': [],
    'gpu_memory_est': [], 'success_count': 0, 'total_count': 0
})

print("正在分析历史训练数据...\n")

for result_file in sorted(results_dir.glob('*.json')):
    try:
        with open(result_file) as f:
            data = json.load(f)

        repo = data.get('repository', '')
        model = data.get('model', '')
        model_key = f"{repo}_{model}"

        model_stats[model_key]['total_count'] += 1

        if data.get('training_success'):
            model_stats[model_key]['success_count'] += 1
            energy = data.get('energy_metrics', {})

            if energy:
                model_stats[model_key]['gpu_util'].append(energy.get('gpu_util_avg_percent', 0))
                model_stats[model_key]['gpu_power'].append(energy.get('gpu_power_avg_watts', 0))
                model_stats[model_key]['duration'].append(data.get('duration_seconds', 0))
    except:
        pass

# Estimate GPU memory usage based on model characteristics and GPU utilization
# Higher utilization usually correlates with higher memory usage
def estimate_gpu_memory(gpu_util_avg, model_key):
    """Estimate GPU memory based on model type and utilization"""
    # Base estimates from typical model sizes
    base_estimates = {
        'mnist': 500,      # Small CNN
        'resnet20': 800,   # Small ResNet
        'resnet32': 1000,
        'resnet44': 1200,
        'resnet56': 1500,
        'resnet110': 2500,
        'densenet121': 3000,  # Larger model
        'hrnet18': 2500,
        'pcb': 2000,
        'MRT-OAST': 3500,     # Object tracking, complex
        'VulBERTa': 4000,     # Transformer-based
        'bug-localization': 2000,
    }

    # Find base estimate
    base = 1500  # default
    for key, mem in base_estimates.items():
        if key in model_key.lower():
            base = mem
            break

    # Adjust based on utilization (high util often means high memory)
    if gpu_util_avg > 80:
        multiplier = 1.3
    elif gpu_util_avg > 50:
        multiplier = 1.1
    else:
        multiplier = 0.9

    return int(base * multiplier)

# Print detailed analysis
print("="*100)
print(f"{'模型':<45} {'训练次数':<10} {'成功率':<10} {'GPU利用率%':<12} {'功率(W)':<12} {'时长(s)':<12} {'显存估计(MB)':<15}")
print("="*100)

model_analysis = {}
for model_info in all_models:
    model_key = model_info['key']
    stats = model_stats[model_key]

    if stats['success_count'] > 0:
        avg_util = statistics.mean(stats['gpu_util'])
        avg_power = statistics.mean(stats['gpu_power'])
        avg_duration = statistics.mean(stats['duration'])
        max_util = max(stats['gpu_util']) if stats['gpu_util'] else 0
        success_rate = stats['success_count'] / stats['total_count'] * 100

        # Estimate memory
        est_memory = estimate_gpu_memory(avg_util, model_key)

        model_analysis[model_key] = {
            'avg_gpu_util': avg_util,
            'max_gpu_util': max_util,
            'avg_power': avg_power,
            'avg_duration': avg_duration,
            'est_memory_mb': est_memory,
            'success_count': stats['success_count'],
            'success_rate': success_rate,
            'repo': model_info['repo'],
            'model': model_info['model']
        }

        print(f"{model_key:<45} {stats['success_count']:<10} {success_rate:<10.1f} "
              f"{avg_util:<12.1f} {avg_power:<12.1f} {avg_duration:<12.0f} {est_memory:<15}")
    else:
        # No data available
        # Use conservative estimates
        est_memory = estimate_gpu_memory(50, model_key)
        model_analysis[model_key] = {
            'avg_gpu_util': 50,  # assume moderate
            'max_gpu_util': 70,
            'avg_power': 150,
            'avg_duration': 600,
            'est_memory_mb': est_memory,
            'success_count': 0,
            'success_rate': 0,
            'repo': model_info['repo'],
            'model': model_info['model']
        }
        print(f"{model_key:<45} {'无数据':<10} {'N/A':<10} "
              f"{'估计:50':<12} {'估计:150':<12} {'估计:600':<12} {est_memory:<15} (估计)")

# Analyze concurrent training feasibility
print("\n" + "="*100)
print("并发训练可行性分析")
print("="*100)

# Sort models by memory usage
sorted_models = sorted(model_analysis.items(), key=lambda x: x[1]['est_memory_mb'], reverse=True)

print("\n按显存占用排序:")
print(f"{'排名':<6} {'模型':<45} {'估计显存(MB)':<15} {'GPU利用率%':<15}")
print("-"*90)
for i, (model_key, data) in enumerate(sorted_models, 1):
    print(f"{i:<6} {model_key:<45} {data['est_memory_mb']:<15} {data['avg_gpu_util']:<15.1f}")

# Identify incompatible pairs
print("\n" + "="*100)
print("并发冲突分析 - 不能同时训练的模型对")
print("="*100)

high_memory_models = []
medium_memory_models = []
low_memory_models = []

for model_key, data in model_analysis.items():
    mem = data['est_memory_mb']
    if mem > 2500:
        high_memory_models.append((model_key, mem))
    elif mem > 1500:
        medium_memory_models.append((model_key, mem))
    else:
        low_memory_models.append((model_key, mem))

print(f"\n高显存模型 (>2500MB): {len(high_memory_models)}个")
for model, mem in sorted(high_memory_models, key=lambda x: x[1], reverse=True):
    print(f"  - {model:<45} {mem:>6}MB")

print(f"\n中显存模型 (1500-2500MB): {len(medium_memory_models)}个")
for model, mem in sorted(medium_memory_models, key=lambda x: x[1], reverse=True):
    print(f"  - {model:<45} {mem:>6}MB")

print(f"\n低显存模型 (<1500MB): {len(low_memory_models)}个")
for model, mem in sorted(low_memory_models, key=lambda x: x[1], reverse=True):
    print(f"  - {model:<45} {mem:>6}MB")

# Check incompatible pairs
print("\n" + "="*100)
print("❌ 不能同时训练的模型组合 (总显存 > 9GB, 留1GB缓冲)")
print("="*100)

incompatible_pairs = []
for i, (model1, data1) in enumerate(sorted_models):
    for model2, data2 in sorted_models[i+1:]:
        total_memory = data1['est_memory_mb'] + data2['est_memory_mb']
        if total_memory > 9000:  # Leave 1GB buffer
            incompatible_pairs.append((model1, model2, total_memory))

if incompatible_pairs:
    print(f"\n发现 {len(incompatible_pairs)} 个不兼容的模型对:\n")
    for model1, model2, total in sorted(incompatible_pairs, key=lambda x: x[2], reverse=True):
        print(f"  ❌ {model1:<40} + {model2:<40} = {total:>5}MB (超出)")
else:
    print("\n✅ 未发现不兼容的模型对（基于估计）")

# Check compatible pairs
print("\n" + "="*100)
print("✅ 可以同时训练的模型组合 (总显存 <= 9GB)")
print("="*100)

compatible_pairs = []
for i, (model1, data1) in enumerate(sorted_models):
    for model2, data2 in sorted_models[i+1:]:
        total_memory = data1['est_memory_mb'] + data2['est_memory_mb']
        if total_memory <= 9000:
            # Also check if combined GPU utilization is reasonable
            total_util = data1['avg_gpu_util'] + data2['avg_gpu_util']
            compatible_pairs.append((model1, model2, total_memory, total_util))

if compatible_pairs:
    # Sort by memory usage
    compatible_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n发现 {len(compatible_pairs)} 个兼容的模型对:\n")
    print(f"{'模型1':<40} {'模型2':<40} {'总显存(MB)':<12} {'总GPU利用率%':<15}")
    print("-"*110)
    for model1, model2, total_mem, total_util in compatible_pairs[:20]:  # Show top 20
        status = "⚠️ 高负载" if total_util > 150 else "✅ 合理"
        print(f"{model1:<40} {model2:<40} {total_mem:<12} {total_util:<15.1f} {status}")

    if len(compatible_pairs) > 20:
        print(f"\n... 还有 {len(compatible_pairs) - 20} 个兼容组合未显示")

# Best concurrent pairs
print("\n" + "="*100)
print("💡 推荐的并发训练组合 (显存安全 + GPU利用率互补)")
print("="*100)

# Find pairs where one is high-util and one is low-util (good complementary)
recommended_pairs = []
for model1, data1 in model_analysis.items():
    for model2, data2 in model_analysis.items():
        if model1 >= model2:  # Avoid duplicates
            continue

        total_memory = data1['est_memory_mb'] + data2['est_memory_mb']
        if total_memory > 9000:
            continue

        # Check if GPU utilization is complementary (one high, one low)
        util1 = data1['avg_gpu_util']
        util2 = data2['avg_gpu_util']

        is_complementary = (util1 > 70 and util2 < 30) or (util1 < 30 and util2 > 70)
        total_util = util1 + util2

        if is_complementary or total_util < 120:
            recommended_pairs.append({
                'model1': model1,
                'model2': model2,
                'total_memory': total_memory,
                'total_util': total_util,
                'util1': util1,
                'util2': util2,
                'complementary': is_complementary
            })

if recommended_pairs:
    recommended_pairs.sort(key=lambda x: (x['complementary'], -x['total_util']), reverse=True)
    print(f"\n发现 {len(recommended_pairs)} 个推荐组合:\n")
    print(f"{'模型1':<40} {'GPU%1':<8} {'模型2':<40} {'GPU%2':<8} {'总显存MB':<12} {'总GPU%':<10} {'互补':<8}")
    print("-"*125)
    for pair in recommended_pairs[:15]:
        comp_mark = "✅是" if pair['complementary'] else "  -"
        print(f"{pair['model1']:<40} {pair['util1']:<8.1f} {pair['model2']:<40} "
              f"{pair['util2']:<8.1f} {pair['total_memory']:<12} {pair['total_util']:<10.1f} {comp_mark}")

# Summary and recommendations
print("\n" + "="*100)
print("📊 总结与建议")
print("="*100)

high_mem_count = len([m for m in model_analysis.values() if m['est_memory_mb'] > 2500])
medium_mem_count = len([m for m in model_analysis.values() if 1500 < m['est_memory_mb'] <= 2500])
low_mem_count = len([m for m in model_analysis.values() if m['est_memory_mb'] <= 1500])

print(f"""
1. 模型分类:
   - 高显存模型 (>2.5GB): {high_mem_count}个 - 这些模型相互之间可能无法并发
   - 中显存模型 (1.5-2.5GB): {medium_mem_count}个 - 可以与低显存模型并发
   - 低显存模型 (<1.5GB): {low_mem_count}个 - 可以与大多数模型并发

2. 并发策略:
   ✅ 推荐: 1个高显存模型 + 1个低显存模型
   ✅ 推荐: 2个低显存模型 + 1个中显存模型
   ⚠️  谨慎: 2个中显存模型
   ❌ 避免: 2个高显存模型

3. GPU利用率互补原则:
   - 高GPU利用率模型 (>70%): {len([m for m in model_analysis.values() if m['avg_gpu_util'] > 70])}个
   - 低GPU利用率模型 (<30%): {len([m for m in model_analysis.values() if m['avg_gpu_util'] < 30])}个
   - 配对高利用率+低利用率可以提高GPU整体利用率

4. 潜在问题:
   - 显存不足 (OOM): 高显存模型之间并发会导致Out of Memory
   - GPU算力竞争: 两个高利用率模型并发会相互拖慢
   - CPU/磁盘I/O瓶颈: 数据加载密集的模型可能竞争I/O

5. 避免措施:
   ✅ 使用mutation.py的max_parallel参数限制并发数
   ✅ 为高显存模型单独分组，避免并发
   ✅ 监控nvidia-smi，动态调整并发策略
   ✅ 使用较小的batch_size减少显存占用

6. 最佳实践:
   - 使用--max-parallel 2 限制最多2个模型同时训练
   - 优先串行训练高显存模型
   - 并发训练互补的模型组合（如MNIST + MRT-OAST）
""")

print("="*100)
print("分析完成！")
print("="*100)
