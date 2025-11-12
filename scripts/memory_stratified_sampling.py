#!/usr/bin/env python3
"""
显存分层抽样工具 - 实际可执行版本
Memory-based Stratified Sampling Tool
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import statistics

def collect_memory_data_from_results():
    """从历史训练结果中提取显存数据（如果有）"""
    memory_data = {}
    results_dir = Path('results')

    if not results_dir.exists():
        return None

    for result_file in results_dir.glob('*.json'):
        try:
            with open(result_file) as f:
                data = json.load(f)

            if data.get('training_success'):
                model_key = f"{data['repository']}_{data['model']}"

                # 检查是否有显存数据
                energy = data.get('energy_metrics', {})
                if 'gpu_memory_peak_mb' in energy:
                    if model_key not in memory_data:
                        memory_data[model_key] = []
                    memory_data[model_key].append(energy['gpu_memory_peak_mb'])
        except:
            pass

    # 计算平均值
    if memory_data:
        return {k: statistics.mean(v) for k, v in memory_data.items()}
    return None

def use_estimated_memory():
    """使用保守的显存估算值"""
    return {
        'examples_mnist': 500,
        'examples_mnist_rnn': 500,
        'examples_mnist_ff': 500,
        'pytorch_resnet_cifar10_resnet20': 800,
        'pytorch_resnet_cifar10_resnet32': 1000,
        'pytorch_resnet_cifar10_resnet44': 1200,
        'pytorch_resnet_cifar10_resnet56': 1500,
        'Person_reID_baseline_pytorch_densenet121': 3500,
        'Person_reID_baseline_pytorch_hrnet18': 2500,
        'Person_reID_baseline_pytorch_pcb': 2000,
        'MRT-OAST_default': 2200,
        'VulBERTa_mlp': 1500,
        'VulBERTa_cnn': 1500,
        'bug-localization-by-dnn-and-rvsm_default': 2000,
        'examples_siamese': 1500,
        'examples_word_lm': 1500,
    }

def define_memory_strata():
    """定义显存分层标准"""
    return {
        'ultra_low': {
            'range': (0, 1500),
            'description': '超低显存 (<1.5GB): 两个小模型，极安全',
            'oom_risk': 'very_low',
            'target_ratio': 0.15
        },
        'low': {
            'range': (1500, 3000),
            'description': '低显存 (1.5-3GB): 一大一小或两中等，安全',
            'oom_risk': 'low',
            'target_ratio': 0.35
        },
        'medium': {
            'range': (3000, 5000),
            'description': '中显存 (3-5GB): 一大一中，可接受',
            'oom_risk': 'medium',
            'target_ratio': 0.30
        },
        'high': {
            'range': (5000, 7000),
            'description': '高显存 (5-7GB): 接近上限，需谨慎',
            'oom_risk': 'high',
            'target_ratio': 0.15
        },
        'ultra_high': {
            'range': (7000, 10000),
            'description': '极高显存 (>7GB): 超过安全界限',
            'oom_risk': 'very_high',
            'target_ratio': 0.05
        }
    }

def generate_and_stratify_pairs(memory_data, strata_def, max_memory=9000):
    """生成所有可行的模型对并按显存分层"""
    models = list(memory_data.keys())
    strata = {name: [] for name in strata_def.keys()}

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            total_memory = memory_data[model1] + memory_data[model2]

            if total_memory <= max_memory:
                pair = {
                    'model1': model1,
                    'model2': model2,
                    'memory1': memory_data[model1],
                    'memory2': memory_data[model2],
                    'total_memory': total_memory
                }

                # 分配到对应层
                for stratum_name, stratum_info in strata_def.items():
                    min_mem, max_mem = stratum_info['range']
                    if min_mem <= total_memory < max_mem:
                        strata[stratum_name].append(pair)
                        break

    return strata

def calculate_sample_sizes(strata, strata_def, total_samples=12, method='proportional'):
    """计算每层应抽取的样本数"""

    if method == 'proportional':
        # 按各层实际样本数比例分配
        total_pairs = sum(len(pairs) for pairs in strata.values())
        if total_pairs == 0:
            return {}

        sample_sizes = {}
        for stratum_name, pairs in strata.items():
            if len(pairs) > 0:
                ratio = len(pairs) / total_pairs
                size = max(1, round(ratio * total_samples))
                sample_sizes[stratum_name] = min(size, len(pairs))
            else:
                sample_sizes[stratum_name] = 0

        # 调整到总数
        current_total = sum(sample_sizes.values())
        if current_total != total_samples and sample_sizes:
            largest_stratum = max(sample_sizes, key=sample_sizes.get)
            sample_sizes[largest_stratum] += (total_samples - current_total)

    elif method == 'equal':
        # 每层相等数量
        non_empty_strata = [name for name, pairs in strata.items() if len(pairs) > 0]
        if not non_empty_strata:
            return {}

        per_stratum = total_samples // len(non_empty_strata)
        sample_sizes = {name: per_stratum for name in strata.keys()}

        remainder = total_samples % len(non_empty_strata)
        for i, name in enumerate(non_empty_strata[:remainder]):
            sample_sizes[name] += 1

    elif method == 'target_ratio':
        # 按预设目标比例
        sample_sizes = {}
        for stratum_name, pairs in strata.items():
            if len(pairs) > 0:
                target = strata_def[stratum_name]['target_ratio']
                size = max(1, round(target * total_samples))
                sample_sizes[stratum_name] = min(size, len(pairs))
            else:
                sample_sizes[stratum_name] = 0

    return sample_sizes

def perform_stratified_sampling(strata, sample_sizes, seed=42):
    """执行分层抽样"""
    random.seed(seed)
    sampled_pairs = []

    for stratum_name, pairs in strata.items():
        size = sample_sizes.get(stratum_name, 0)

        if size > 0 and len(pairs) > 0:
            sample = random.sample(pairs, min(size, len(pairs)))
            sampled_pairs.extend(sample)

    return sampled_pairs

def main():
    print("="*100)
    print("显存分层抽样工具")
    print("Memory-based Stratified Sampling Tool")
    print("="*100)

    # 步骤1: 收集显存数据
    print("\n步骤1: 收集模型显存数据...")
    memory_data = collect_memory_data_from_results()

    if memory_data:
        print(f"  ✅ 从历史记录中找到 {len(memory_data)} 个模型的显存数据")
    else:
        print("  ⚠️  未找到历史显存数据，使用估算值")
        memory_data = use_estimated_memory()
        print(f"  📊 使用 {len(memory_data)} 个模型的估算显存值")

    print("\n  模型显存占用:")
    for model, mem in sorted(memory_data.items(), key=lambda x: x[1]):
        print(f"    {model:<50} {mem:>6.0f} MB")

    # 步骤2: 定义分层标准
    print("\n步骤2: 定义分层标准...")
    strata_def = define_memory_strata()
    print("  分层定义:")
    for name, info in strata_def.items():
        print(f"    {name:15} {info['description']}")

    # 步骤3: 生成并分层
    print("\n步骤3: 生成并分层所有模型对...")
    strata = generate_and_stratify_pairs(memory_data, strata_def)
    total_pairs = sum(len(pairs) for pairs in strata.values())
    print(f"  总可行组合数: {total_pairs}")
    print("\n  各层分布:")
    for name, pairs in strata.items():
        if len(pairs) > 0:
            pct = len(pairs) / total_pairs * 100
            print(f"    {name:15} {len(pairs):3}个组合 ({pct:5.1f}%)")

    # 步骤4: 计算样本数（对比三种方法）
    print("\n步骤4: 计算每层抽样数 (对比三种方法)...")

    methods = [
        ('proportional', '按比例分配'),
        ('equal', '均等分配'),
        ('target_ratio', '目标比例分配')
    ]

    for method, desc in methods:
        print(f"\n  {desc} ({method}):")
        sizes = calculate_sample_sizes(strata, strata_def, total_samples=12, method=method)
        for name, size in sizes.items():
            if size > 0:
                print(f"    {name:15} {size:2}个")

    # 步骤5: 执行抽样（使用按比例分配）
    print("\n步骤5: 执行抽样 (使用按比例分配)...")
    sizes_prop = calculate_sample_sizes(strata, strata_def, total_samples=12, method='proportional')
    sampled_pairs = perform_stratified_sampling(strata, sizes_prop, seed=42)

    print(f"\n  ✅ 最终抽取 {len(sampled_pairs)} 个模型组合")

    print(f"\n{'序号':<5} {'模型A':<50} {'模型B':<50} {'总显存(MB)':<12} {'层级':<15}")
    print("-" * 132)

    # 确定每个样本所属层级
    for i, pair in enumerate(sampled_pairs, 1):
        layer = ''
        for name, info in strata_def.items():
            min_mem, max_mem = info['range']
            if min_mem <= pair['total_memory'] < max_mem:
                layer = name
                break

        print(f"{i:<5} {pair['model1']:<50} {pair['model2']:<50} {pair['total_memory']:<12.0f} {layer:<15}")

    # 导出结果
    output = {
        'method': 'memory_stratified_sampling',
        'total_samples': len(sampled_pairs),
        'strata_definition': strata_def,
        'sample_sizes': sizes_prop,
        'sampled_pairs': sampled_pairs,
        'memory_data_source': 'historical' if collect_memory_data_from_results() else 'estimated'
    }

    output_file = 'memory_stratified_sample.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ 结果已保存到: {output_file}")

    # 生成统计报告
    print("\n" + "="*100)
    print("抽样质量报告")
    print("="*100)

    print("\n层级覆盖情况:")
    for name, pairs in strata.items():
        sampled = sum(1 for p in sampled_pairs if any(
            info['range'][0] <= p['total_memory'] < info['range'][1]
            for sname, info in strata_def.items() if sname == name
        ))
        if len(pairs) > 0:
            coverage = sampled / len(pairs) * 100
            print(f"  {name:15} 总数:{len(pairs):3} 抽取:{sampled:2} 抽样率:{coverage:5.1f}%")

    print("\n显存分布:")
    memories = [p['total_memory'] for p in sampled_pairs]
    print(f"  最小: {min(memories):.0f} MB")
    print(f"  最大: {max(memories):.0f} MB")
    print(f"  平均: {statistics.mean(memories):.0f} MB")
    print(f"  中位数: {statistics.median(memories):.0f} MB")

    print("\n="*100)
    print("完成！")
    print("="*100)

if __name__ == "__main__":
    main()
