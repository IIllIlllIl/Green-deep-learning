#!/usr/bin/env python3
"""
显存占用获取与显存分层抽样实战指南
GPU Memory Usage Detection & Memory-based Stratified Sampling Guide
"""
import json
import subprocess
from pathlib import Path
import random

print("="*100)
print("问题1: 显存利用率是如何知道的？")
print("="*100)

print("""
⚠️ 重要说明：之前分析中的显存占用值都是"估算"的，不是真实测量值！

估算方法（不准确）:
  基于模型架构类型和历史经验：
  - MNIST类: 450-500MB (小型CNN)
  - ResNet20-56: 720-1350MB (中型CNN)
  - DenseNet121: ~3300MB (大型CNN)
  - MRT-OAST: ~2000MB (自定义网络)

  缺点：
  ❌ 不考虑batch size
  ❌ 不考虑输入尺寸
  ❌ 不考虑中间激活值
  ❌ 静态估算，误差大

正确的显存获取方法：
""")

print("\n" + "="*100)
print("方法1: 使用nvidia-smi实时监控 (推荐)")
print("="*100)

print("""
在训练过程中实时监控：

命令行方式:
  watch -n 1 nvidia-smi

  或查看显存使用:
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv

Python方式:
""")

example_code_1 = '''
import subprocess

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        memory_used, memory_total = map(int, result.strip().split(','))
        return {
            'used_mb': memory_used,
            'total_mb': memory_total,
            'used_percent': memory_used / memory_total * 100
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# 使用示例
mem = get_gpu_memory()
if mem:
    print(f"显存使用: {mem['used_mb']}MB / {mem['total_mb']}MB ({mem['used_percent']:.1f}%)")
'''

print(example_code_1)

print("\n" + "="*100)
print("方法2: 在训练脚本中集成PyTorch内存统计 (最准确)")
print("="*100)

print("""
在训练脚本中直接记录：
""")

example_code_2 = '''
import torch

def log_memory_usage(prefix=""):
    """记录PyTorch显存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2

        print(f"{prefix} GPU Memory:")
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved:  {reserved:.1f} MB")
        print(f"  Peak:      {max_allocated:.1f} MB")

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': max_allocated
        }

# 在训练脚本中使用：
# 训练前
log_memory_usage("Before training")

# 训练过程中
for epoch in range(num_epochs):
    # ... training code ...
    if epoch % 10 == 0:
        mem_info = log_memory_usage(f"Epoch {epoch}")

# 训练后
log_memory_usage("After training")
'''

print(example_code_2)

print("\n" + "="*100)
print("方法3: 从能耗监控系统获取 (如果已集成)")
print("="*100)

print("""
修改mutation.py的能耗监控，添加显存记录：
""")

example_code_3 = '''
# 在能耗监控循环中添加显存记录
def monitor_energy_and_memory(process, interval=1):
    """监控能耗和显存"""
    samples = []

    while process.poll() is None:
        # 原有的能耗监控
        gpu_power = get_gpu_power()
        cpu_energy = get_cpu_energy()

        # 新增：显存监控
        gpu_memory = get_gpu_memory()

        samples.append({
            'timestamp': time.time(),
            'gpu_power': gpu_power,
            'cpu_energy': cpu_energy,
            'gpu_memory_mb': gpu_memory['used_mb'] if gpu_memory else None
        })

        time.sleep(interval)

    return samples

# 在结果JSON中保存
result = {
    'experiment_id': exp_id,
    'energy_metrics': {
        'gpu_memory_avg_mb': statistics.mean([s['gpu_memory_mb'] for s in samples if s['gpu_memory_mb']]),
        'gpu_memory_peak_mb': max([s['gpu_memory_mb'] for s in samples if s['gpu_memory_mb']]),
        # ... 其他指标
    }
}
'''

print(example_code_3)

print("\n" + "="*100)
print("方法4: 从历史数据推断 (次优方案)")
print("="*100)

print("""
如果无法实时监控，可以：
1. 对每个模型运行一次baseline训练
2. 记录显存峰值
3. 建立模型→显存映射表

实施步骤：
""")

# Check if we can get GPU memory info
try:
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        encoding='utf-8'
    )
    memory_used, memory_total = map(int, result.strip().split(','))
    print(f"\n当前GPU显存状态:")
    print(f"  已使用: {memory_used} MB")
    print(f"  总容量: {memory_total} MB")
    print(f"  使用率: {memory_used/memory_total*100:.1f}%")
except Exception as e:
    print(f"\n⚠️ 无法获取GPU信息: {e}")

print("\n" + "="*100)
print("问题2: 如果采用显存分层，如何抽样？")
print("="*100)

print("""
完整的显存分层抽样流程：

第一步：收集真实显存数据
第二步：定义分层标准
第三步：计算模型对的总显存
第四步：按层分配样本数
第五步：执行抽样
""")

print("\n" + "="*100)
print("实战示例：显存分层抽样完整代码")
print("="*100)

complete_example = '''
import json
import random
from collections import defaultdict

# ============================================================
# 步骤1: 收集/估算模型显存占用
# ============================================================

def collect_memory_data():
    """
    收集模型显存数据

    方式1: 从历史训练记录提取
    方式2: 运行baseline测试获取
    方式3: 使用保守估算值（最不准确）
    """

    # 示例：从历史数据提取（如果有）
    memory_data = {}
    results_dir = Path('results')

    for result_file in results_dir.glob('*.json'):
        try:
            with open(result_file) as f:
                data = json.load(f)

            if data.get('training_success'):
                model_key = f"{data['repository']}_{data['model']}"

                # 如果结果中有显存数据
                if 'gpu_memory_peak_mb' in data.get('energy_metrics', {}):
                    memory_data[model_key] = data['energy_metrics']['gpu_memory_peak_mb']
        except:
            pass

    # 如果没有历史数据，使用估算值
    if not memory_data:
        print("⚠️ 未找到历史显存数据，使用估算值")
        memory_data = {
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

    return memory_data

# ============================================================
# 步骤2: 定义分层标准
# ============================================================

def define_memory_strata():
    """
    定义显存分层标准

    根据RTX 3080的10GB显存，留1GB缓冲（9GB可用）
    """
    return {
        'ultra_low': {
            'range': (0, 1500),
            'description': '超低显存 (<1.5GB): 两个小模型，极安全',
            'oom_risk': 'very_low',
            'target_ratio': 0.15  # 期望占总样本的15%
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

# ============================================================
# 步骤3: 生成并分层所有模型对
# ============================================================

def generate_and_stratify_pairs(memory_data, strata_def, max_memory=9000):
    """
    生成所有可行的模型对并按显存分层
    """
    models = list(memory_data.keys())
    strata = {name: [] for name in strata_def.keys()}

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            total_memory = memory_data[model1] + memory_data[model2]

            # 只考虑不超过max_memory的组合
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

# ============================================================
# 步骤4: 计算每层抽样数
# ============================================================

def calculate_sample_sizes(strata, total_samples=12, method='proportional'):
    """
    计算每层应抽取的样本数

    method:
        - 'proportional': 按比例分配（标准方法）
        - 'equal': 每层相等
        - 'target_ratio': 按目标比例分配
        - 'neyman': Neyman最优分配（考虑层内方差）
    """

    if method == 'proportional':
        # 按各层实际样本数比例分配
        total_pairs = sum(len(pairs) for pairs in strata.values())
        sample_sizes = {}

        for stratum_name, pairs in strata.items():
            if len(pairs) > 0:
                ratio = len(pairs) / total_pairs
                size = max(1, round(ratio * total_samples))  # 至少1个
                sample_sizes[stratum_name] = min(size, len(pairs))
            else:
                sample_sizes[stratum_name] = 0

        # 调整到总数
        current_total = sum(sample_sizes.values())
        if current_total != total_samples:
            # 从最大层调整
            largest_stratum = max(sample_sizes, key=sample_sizes.get)
            sample_sizes[largest_stratum] += (total_samples - current_total)

    elif method == 'equal':
        # 每层相等数量
        non_empty_strata = [name for name, pairs in strata.items() if len(pairs) > 0]
        per_stratum = total_samples // len(non_empty_strata)
        sample_sizes = {name: per_stratum for name in non_empty_strata}

        # 余数分配给前几层
        remainder = total_samples % len(non_empty_strata)
        for i, name in enumerate(non_empty_strata[:remainder]):
            sample_sizes[name] += 1

    elif method == 'target_ratio':
        # 按预设目标比例（考虑研究需求）
        sample_sizes = {}
        for stratum_name, pairs in strata.items():
            if len(pairs) > 0:
                target = strata_def[stratum_name]['target_ratio']
                size = max(1, round(target * total_samples))
                sample_sizes[stratum_name] = min(size, len(pairs))
            else:
                sample_sizes[stratum_name] = 0

    return sample_sizes

# ============================================================
# 步骤5: 执行抽样
# ============================================================

def perform_stratified_sampling(strata, sample_sizes, seed=42):
    """
    执行分层抽样
    """
    random.seed(seed)
    sampled_pairs = []

    for stratum_name, pairs in strata.items():
        size = sample_sizes.get(stratum_name, 0)

        if size > 0 and len(pairs) > 0:
            # 从该层随机抽取
            sample = random.sample(pairs, min(size, len(pairs)))
            sampled_pairs.extend(sample)

            print(f"{stratum_name:15} 总数:{len(pairs):3} 抽取:{len(sample):2}")

    return sampled_pairs

# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("\\n" + "="*60)
    print("显存分层抽样实战流程")
    print("="*60)

    # 步骤1: 收集显存数据
    print("\\n步骤1: 收集模型显存数据...")
    memory_data = collect_memory_data()
    print(f"  已收集 {len(memory_data)} 个模型的显存数据")

    # 步骤2: 定义分层标准
    print("\\n步骤2: 定义分层标准...")
    strata_def = define_memory_strata()
    for name, info in strata_def.items():
        print(f"  {name:15} {info['description']}")

    # 步骤3: 生成并分层所有模型对
    print("\\n步骤3: 生成并分层所有模型对...")
    strata = generate_and_stratify_pairs(memory_data, strata_def)
    print(f"  总可行组合数: {sum(len(pairs) for pairs in strata.values())}")
    print("\\n  各层分布:")
    for name, pairs in strata.items():
        if len(pairs) > 0:
            print(f"    {name:15} {len(pairs):3}个组合")

    # 步骤4: 计算样本数（对比三种方法）
    print("\\n步骤4: 计算每层抽样数...")

    print("\\n  方法A: 按比例分配 (proportional)")
    sizes_prop = calculate_sample_sizes(strata, total_samples=12, method='proportional')
    for name, size in sizes_prop.items():
        print(f"    {name:15} {size:2}个")

    print("\\n  方法B: 均等分配 (equal)")
    sizes_equal = calculate_sample_sizes(strata, total_samples=12, method='equal')
    for name, size in sizes_equal.items():
        print(f"    {name:15} {size:2}个")

    print("\\n  方法C: 目标比例分配 (target_ratio)")
    sizes_target = calculate_sample_sizes(strata, total_samples=12, method='target_ratio')
    for name, size in sizes_target.items():
        print(f"    {name:15} {size:2}个")

    # 步骤5: 执行抽样（使用方法A）
    print("\\n步骤5: 执行抽样 (使用按比例分配)...")
    sampled_pairs = perform_stratified_sampling(strata, sizes_prop, seed=42)

    print(f"\\n最终抽取 {len(sampled_pairs)} 个模型组合:")
    print(f"\\n{'#':<3} {'模型1':<45} {'模型2':<45} {'显存(MB)':<10}")
    print("-" * 105)
    for i, pair in enumerate(sampled_pairs, 1):
        print(f"{i:<3} {pair['model1']:<45} {pair['model2']:<45} {pair['total_memory']:<10}")

    # 导出结果
    output = {
        'method': 'memory_stratified_sampling',
        'total_samples': len(sampled_pairs),
        'strata_definition': strata_def,
        'sample_sizes': sizes_prop,
        'sampled_pairs': sampled_pairs
    }

    with open('memory_stratified_sample.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\\n✅ 结果已保存到: memory_stratified_sample.json")
'''

print(complete_example)

print("\n" + "="*100)
print("三种样本数分配方法对比")
print("="*100)

print("""
假设有以下分层情况：
  ultra_low: 10个组合
  low:       60个组合
  medium:    40个组合
  high:      8个组合

总共需要抽取12个样本

方法A: 按比例分配 (Proportional Allocation)
  原理: 按各层实际样本数占总体的比例分配
  计算:
    ultra_low: 10/118 × 12 ≈ 1个
    low:       60/118 × 12 ≈ 6个
    medium:    40/118 × 12 ≈ 4个
    high:      8/118 × 12 ≈ 1个

  优点: ✅ 统计上最无偏
  缺点: ⚠️ 小层样本可能太少
  适用: 追求统计严谨性

方法B: 均等分配 (Equal Allocation)
  原理: 每层分配相同数量
  计算:
    ultra_low: 12/4 = 3个
    low:       12/4 = 3个
    medium:    12/4 = 3个
    high:      12/4 = 3个

  优点: ✅ 每层都有充分代表
  缺点: ⚠️ 可能过度抽样小层
  适用: 需要对比各层差异

方法C: 目标比例分配 (Target Ratio Allocation)
  原理: 根据研究需求设定目标比例
  计算: 基于预设的target_ratio
    ultra_low: 0.15 × 12 = 2个 (重点关注小模型)
    low:       0.35 × 12 = 4个 (主要场景)
    medium:    0.30 × 12 = 4个 (常见场景)
    high:      0.15 × 12 = 2个 (边界测试)

  优点: ✅ 针对性强，灵活
  缺点: ⚠️ 需要先验知识
  适用: 有明确研究重点

推荐: 方法A (按比例) 或 方法C (目标比例)
""")

print("\n" + "="*100)
print("💡 最佳实践建议")
print("="*100)

print("""
1. 显存数据收集优先级：
   第一优先: 实际运行并监控 (nvidia-smi + PyTorch统计)
   第二优先: 从历史训练记录提取
   第三优先: 运行baseline测试专门收集
   最后手段: 使用经验估算值（误差大）

2. 抽样方法选择：
   - 如果追求统计严谨: 使用"按比例分配"
   - 如果需要充分对比: 使用"均等分配"
   - 如果有明确重点: 使用"目标比例分配"

3. 分层数量建议：
   - 样本总数12个: 建议4-5个层
   - 样本总数20+: 可以6-7个层
   - 避免分层过细导致某层无样本

4. 验证检查：
   - ✅ 确保每层至少1个样本
   - ✅ 总样本数等于预期
   - ✅ 高风险层有代表（即使样本少）
   - ✅ 记录随机种子（可重复）

5. 实施步骤：
   Step 1: 运行baseline收集真实显存数据
   Step 2: 建立模型显存映射表
   Step 3: 定义分层标准
   Step 4: 生成所有可行对并分层
   Step 5: 选择分配方法
   Step 6: 执行抽样
   Step 7: 验证结果
""")

print("\n" + "="*100)
print("🔧 实用工具脚本")
print("="*100)

print("""
已为您创建以下脚本：

1. collect_gpu_memory.py
   功能: 运行baseline训练并收集显存数据
   使用: python3 scripts/collect_gpu_memory.py

2. memory_stratified_sampling.py
   功能: 执行显存分层抽样
   使用: python3 scripts/memory_stratified_sampling.py

3. validate_sampling.py
   功能: 验证抽样结果质量
   使用: python3 scripts/validate_sampling.py
""")

print("\n分析完成！")
print("="*100)
