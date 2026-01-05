#!/usr/bin/env python3
"""
变异实验配置生成器

功能：
为11个模型生成变异实验配置，每个超参数变异3次（低、中、高）
只改变一个超参数，其他保持默认值（单因素实验设计）

使用方法：
    python3 generate_mutation_config.py [--output CONFIG.json]
"""

import json
import argparse
from pathlib import Path

# 默认值配置（来自基线测试）
DEFAULT_CONFIGS = {
    "MRT-OAST": {
        "repo": "MRT-OAST",
        "model": "default",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 0.0001,
            "dropout": 0.2,
            "weight_decay": 0.0,
            "seed": 1334
        },
        "mutable": ["epochs", "learning_rate", "dropout", "weight_decay"]
    },
    "bug-localization": {
        "repo": "bug-localization-by-dnn-and-rvsm",
        "model": "default",
        "hyperparameters": {
            "max_iter": 10000,
            "alpha": 1e-05,
            "kfold": 10,
            "seed": 42
        },
        "mutable": ["max_iter", "alpha"]  # 特殊：max_iter=epochs, alpha=weight_decay(L2 penalty); 不支持learning_rate
    },
    "resnet20": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "hyperparameters": {
            "epochs": 200,
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "seed": 1334
        },
        "mutable": ["epochs", "learning_rate", "weight_decay"]
    },
    "VulBERTa_mlp": {
        "repo": "VulBERTa",
        "model": "mlp",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 3e-05,
            "weight_decay": 0.0,
            "seed": 42
        },
        "mutable": ["epochs", "learning_rate", "weight_decay"]
    },
    "densenet121": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "densenet121",
        "hyperparameters": {
            "epochs": 60,
            "learning_rate": 0.05,
            "dropout": 0.5,
            "seed": 1334
        },
        "mutable": ["epochs", "learning_rate", "dropout"]
    },
    "hrnet18": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "hrnet18",
        "hyperparameters": {
            "epochs": 60,
            "learning_rate": 0.05,
            "dropout": 0.5,
            "seed": 1334
        },
        "mutable": ["epochs", "learning_rate", "dropout"]
    },
    "pcb": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "pcb",
        "hyperparameters": {
            "epochs": 60,
            "learning_rate": 0.05,
            "dropout": 0.5,
            "seed": 1334
        },
        "mutable": ["epochs", "learning_rate", "dropout"]
    },
    "mnist": {
        "repo": "examples",
        "model": "mnist",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "seed": 1
        },
        "mutable": ["epochs", "learning_rate"]
    },
    "mnist_rnn": {
        "repo": "examples",
        "model": "mnist_rnn",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "seed": 1
        },
        "mutable": ["epochs", "learning_rate"]
    },
    "mnist_ff": {
        "repo": "examples",
        "model": "mnist_ff",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "seed": 1
        },
        "mutable": ["epochs", "learning_rate"]
    },
    "siamese": {
        "repo": "examples",
        "model": "siamese",
        "hyperparameters": {
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "seed": 1
        },
        "mutable": ["epochs", "learning_rate"]
    }
}

# 并行训练背景任务配置
PARALLEL_BACKGROUNDS = {
    "MRT-OAST": {"repo": "examples", "model": "mnist_rnn"},
    "bug-localization": {"repo": "Person_reID_baseline_pytorch", "model": "pcb"},
    "resnet20": {"repo": "examples", "model": "mnist_ff"},
    "VulBERTa_mlp": {"repo": "examples", "model": "mnist"},
    "densenet121": {"repo": "VulBERTa", "model": "mlp"},
    "hrnet18": {"repo": "examples", "model": "mnist_rnn"},
    "pcb": {"repo": "examples", "model": "mnist_rnn"},
    "mnist": {"repo": "VulBERTa", "model": "mlp"},
    "mnist_rnn": {"repo": "Person_reID_baseline_pytorch", "model": "pcb"},
    "mnist_ff": {"repo": "Person_reID_baseline_pytorch", "model": "densenet121"},
    "siamese": {"repo": "Person_reID_baseline_pytorch", "model": "pcb"}
}

def generate_mutation_values(param_name, default_value):
    """
    为超参数生成3个变异值（低、中、高）

    返回: [(value, multiplier_description), ...]
    """
    if param_name in ["epochs", "max_iter"]:
        # Epochs: [0.5×, 1.0×, 1.5×]
        low = max(1, int(default_value * 0.5))
        mid = default_value
        high = int(default_value * 1.5)
        return [
            (low, "0.5×"),
            (mid, "1.0×"),
            (high, "1.5×")
        ]

    elif param_name == "learning_rate" or param_name == "alpha":
        # Learning rate: [0.5×, 1.0×, 2.0×] (log-uniform)
        low = default_value * 0.5
        mid = default_value
        high = default_value * 2.0
        return [
            (low, "0.5×"),
            (mid, "1.0×"),
            (high, "2.0×")
        ]

    elif param_name == "dropout":
        # Dropout: [max(0, default-0.2), default, min(0.4, default+0.2)]
        low = max(0.0, default_value - 0.2)
        mid = default_value
        high = min(0.7, default_value + 0.2)  # 0.7 是常见上限
        return [
            (low, f"{low:.1f}"),
            (mid, f"{mid:.1f}"),
            (high, f"{high:.1f}")
        ]

    elif param_name == "weight_decay":
        # Weight decay: [0.1×, 1.0×, 10×] (log-uniform)
        if default_value == 0.0:
            # 如果默认值是0，则测试 [0, 0.0001, 0.001]
            return [
                (0.0, "0"),
                (0.0001, "1e-4"),
                (0.001, "1e-3")
            ]
        else:
            low = default_value * 0.1
            mid = default_value
            high = min(0.1, default_value * 10)  # 上限0.1
            return [
                (low, "0.1×"),
                (mid, "1.0×"),
                (high, "10×")
            ]

    else:
        # 其他参数保持不变
        return [(default_value, "default")]

def generate_experiments():
    """生成所有变异实验"""
    experiments = []
    exp_counter = 1

    # 遍历11个模型
    for model_name, config in DEFAULT_CONFIGS.items():
        repo = config["repo"]
        model = config["model"]
        default_params = config["hyperparameters"].copy()
        mutable_params = config["mutable"]

        # 遍历每个可变异超参数
        for param_name in mutable_params:
            default_value = default_params[param_name]
            mutation_values = generate_mutation_values(param_name, default_value)

            # 为��个变异值生成一个实验（顺序训练）
            for value, description in mutation_values:
                # 创建变异后的超参数
                mutated_params = default_params.copy()
                mutated_params[param_name] = value

                # 顺序训练实验
                seq_exp = {
                    "mode": "default",
                    "repo": repo,
                    "model": model,
                    "hyperparameters": mutated_params,
                    "note": f"Sequential {exp_counter}: {model_name} - mutate {param_name}={description}"
                }
                experiments.append(seq_exp)
                exp_counter += 1

        # 并行训练实验（每个可变异超参数）
        for param_name in mutable_params:
            default_value = default_params[param_name]
            mutation_values = generate_mutation_values(param_name, default_value)

            # 为每个变异值生成一个并行实验
            for value, description in mutation_values:
                # 创建变异后的超参数
                mutated_params = default_params.copy()
                mutated_params[param_name] = value

                # 获取背景任务配置
                bg_config = DEFAULT_CONFIGS.get(
                    PARALLEL_BACKGROUNDS[model_name]["model"],
                    DEFAULT_CONFIGS["mnist"]  # fallback
                )

                # 背景任务使用默认超参数
                if PARALLEL_BACKGROUNDS[model_name]["model"] == "pcb":
                    bg_params = DEFAULT_CONFIGS["pcb"]["hyperparameters"].copy()
                elif PARALLEL_BACKGROUNDS[model_name]["model"] == "densenet121":
                    bg_params = DEFAULT_CONFIGS["densenet121"]["hyperparameters"].copy()
                elif PARALLEL_BACKGROUNDS[model_name]["model"] == "mnist_rnn":
                    bg_params = DEFAULT_CONFIGS["mnist_rnn"]["hyperparameters"].copy()
                elif PARALLEL_BACKGROUNDS[model_name]["model"] == "mnist_ff":
                    bg_params = DEFAULT_CONFIGS["mnist_ff"]["hyperparameters"].copy()
                elif PARALLEL_BACKGROUNDS[model_name]["model"] == "mnist":
                    bg_params = DEFAULT_CONFIGS["mnist"]["hyperparameters"].copy()
                elif PARALLEL_BACKGROUNDS[model_name]["model"] == "mlp":
                    bg_params = DEFAULT_CONFIGS["VulBERTa_mlp"]["hyperparameters"].copy()
                else:
                    bg_params = {"epochs": 10, "learning_rate": 0.01, "seed": 1}

                # 并行训练实验
                par_exp = {
                    "mode": "parallel",
                    "foreground": {
                        "repo": repo,
                        "model": model,
                        "mode": "default",
                        "hyperparameters": mutated_params
                    },
                    "background": {
                        "repo": PARALLEL_BACKGROUNDS[model_name]["repo"],
                        "model": PARALLEL_BACKGROUNDS[model_name]["model"],
                        "hyperparameters": bg_params
                    },
                    "note": f"Parallel {exp_counter}: {model_name} - mutate {param_name}={description}"
                }
                experiments.append(par_exp)
                exp_counter += 1

    return experiments

def count_experiments_by_model():
    """统计每个模型的实验数量"""
    counts = {}
    total_seq = 0
    total_par = 0

    for model_name, config in DEFAULT_CONFIGS.items():
        mutable_count = len(config["mutable"])
        exp_per_model = mutable_count * 3  # 每个参数3个变异值
        counts[model_name] = {
            "mutable_params": config["mutable"],
            "experiments_per_mode": exp_per_model,
            "total_experiments": exp_per_model * 2  # sequential + parallel
        }
        total_seq += exp_per_model
        total_par += exp_per_model

    return counts, total_seq, total_par

def estimate_time(baseline_times):
    """
    估算总运行时间

    baseline_times: dict with model_name -> {"sequential": minutes, "parallel": minutes}
    """
    counts, total_seq, total_par = count_experiments_by_model()

    total_time_seq = 0
    total_time_par = 0

    for model_name, config in DEFAULT_CONFIGS.items():
        exp_count = counts[model_name]["experiments_per_mode"]

        # 获取该模型的基线时间
        base_time_seq = baseline_times.get(model_name, {}).get("sequential", 30)  # 默认30分钟
        base_time_par = baseline_times.get(model_name, {}).get("parallel", 35)

        # Epochs变异会影响时间：0.5×约减半，1.5×约增加50%
        # 其他超参数变异时间基本不变
        # 平均时间系数：(0.5 + 1.0 + 1.5) / 3 = 1.0 对于epochs
        # 对于每个模型，假设epochs变异占1/3的实验（如果有epochs）

        # 简化估算：使用基线时间的平均值
        total_time_seq += exp_count * base_time_seq
        total_time_par += exp_count * base_time_par

    total_time = total_time_seq + total_time_par

    return {
        "sequential_minutes": total_time_seq,
        "parallel_minutes": total_time_par,
        "total_minutes": total_time,
        "total_hours": total_time / 60,
        "total_days": total_time / 60 / 24
    }

def main():
    parser = argparse.ArgumentParser(description="生成变异实验配置文件")
    parser.add_argument("--output", "-o",
                       default="settings/mutation_all_models_3x.json",
                       help="输出配置文件路径")
    parser.add_argument("--stats-only", action="store_true",
                       help="只显示统计信息，不生成文件")

    args = parser.parse_args()

    # 统计实验数量
    counts, total_seq, total_par = count_experiments_by_model()

    print("=" * 80)
    print("变异实验配置生成器")
    print("=" * 80)
    print()
    print("📊 实验数量统计")
    print("-" * 80)
    print(f"{'模型':<20} {'可变异参数':<30} {'顺序':<8} {'并行':<8} {'小计':<8}")
    print("-" * 80)

    for model_name, count in counts.items():
        params_str = ", ".join(count["mutable_params"])
        exp_per = count["experiments_per_mode"]
        total = count["total_experiments"]
        print(f"{model_name:<20} {params_str:<30} {exp_per:<8} {exp_per:<8} {total:<8}")

    print("-" * 80)
    print(f"{'总计':<20} {'':<30} {total_seq:<8} {total_par:<8} {total_seq + total_par:<8}")
    print()

    # 基于基线测试的实际时间估算
    baseline_times = {
        "MRT-OAST": {"sequential": 21, "parallel": 22},
        "bug-localization": {"sequential": 15, "parallel": 20},
        "resnet20": {"sequential": 19, "parallel": 19},
        "VulBERTa_mlp": {"sequential": 52, "parallel": 63},
        "densenet121": {"sequential": 54, "parallel": 59},
        "hrnet18": {"sequential": 71, "parallel": 83},
        "pcb": {"sequential": 72, "parallel": 72},
        "mnist": {"sequential": 2, "parallel": 2},
        "mnist_rnn": {"sequential": 4, "parallel": 4},
        "mnist_ff": {"sequential": 0.13, "parallel": 0.12},
        "siamese": {"sequential": 5, "parallel": 8}
    }

    time_est = estimate_time(baseline_times)

    print("⏱️  运行时间估算")
    print("-" * 80)
    print(f"顺序训练总时长:       {time_est['sequential_minutes']:.1f} 分钟 ({time_est['sequential_minutes']/60:.1f} 小时)")
    print(f"并行训练总时长:       {time_est['parallel_minutes']:.1f} 分钟 ({time_est['parallel_minutes']/60:.1f} 小时)")
    print(f"总计:               {time_est['total_minutes']:.1f} 分钟 ({time_est['total_hours']:.1f} 小时)")
    print(f"预计天数:            {time_est['total_days']:.1f} 天")
    print()
    print("⚠️  注意:")
    print("  - Epochs变异会影响运行时间（0.5×约快一半，1.5×约慢50%）")
    print("  - 以上估算基于默认epochs的平均时间")
    print("  - 实际时间可能在 ±30% 范围内波动")
    print()

    if args.stats_only:
        print("ℹ️  仅显示统计信息，未生成配置文件")
        print("    使用 --output 参数生成配置文件")
        return

    # 生成实验配置
    print("🔧 生成实验配置...")
    experiments = generate_experiments()

    # 创建完整配置
    config = {
        "experiment_name": "mutation_all_models_3x",
        "description": "完整变异测试：11个模型，每个超参数变异3次（低、中、高），顺序+并行训练",
        "governor": "performance",
        "runs_per_config": 1,
        "max_retries": 2,
        "mode": "mixed",
        "experiments": experiments
    }

    # 保存到文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 配置文件已生成: {output_path}")
    print(f"   总实验数: {len(experiments)}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    print("💡 使用方法:")
    print(f"   export HF_HUB_OFFLINE=1")
    print(f"   sudo -E python3 mutation.py -ec {output_path}")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
