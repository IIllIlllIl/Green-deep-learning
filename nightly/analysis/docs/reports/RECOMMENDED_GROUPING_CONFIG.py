# 推荐的分组配置
# 生成时间: 2025-12-24
# 策略: strategy2_task_then_hyperparam
# 数据保留率: 82.7%

TASK_GROUPS = {
    'image_classification_examples': {
        'repos': ['examples'],
        'models': {
            'examples': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'],
        },
    },
    'image_classification_resnet': {
        'repos': ['pytorch_resnet_cifar10'],
        'models': {
            'pytorch_resnet_cifar10': ['resnet20'],
        },
    },
    'person_reid': {
        'repos': ['Person_reID_baseline_pytorch'],
        'models': {
            'Person_reID_baseline_pytorch': ['densenet121', 'hrnet18', 'pcb'],
        },
    },
    'vulberta': {
        'repos': ['VulBERTa'],
        'models': {
            'VulBERTa': ['mlp'],
        },
    },
    'bug_localization': {
        'repos': ['bug-localization-by-dnn-and-rvsm'],
        'models': {
            'bug-localization-by-dnn-and-rvsm': ['default'],
        },
    },
}
