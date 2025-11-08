# 配置文件功能总结

## 🎉 新增功能

### 1. 实验配置文件支持

mutation.py现在支持通过JSON配置文件批量运行实验，比命令行模式更方便。

### 2. 两种运行模式

#### Mode 1: Mutation模式（变异）
- 自动生成随机的超参数变体
- 用于探索超参数空间
- 示例：`all.json`

#### Mode 2: Default模式（默认）
- 使用指定的固定超参数值
- 用于复现原始训练过程
- 示例：`default.json`

---

## 📁 ��创建的文件

### settings/ 目录

```
settings/
├── all.json                      # 变异所有模型的所有超参数
├── default.json                  # 复现所有模型的原始训练
├── resnet_all_models.json        # ResNet家族实验
├── learning_rate_study.json      # 学习率影响研究
├── mixed_mode_demo.json          # 混合模式演示
└── README.md                     # 完整使用说明
```

### mutation.py 修改

**新增方法**:
- `run_from_experiment_config()` - 从配置文件运行实验

**新增参数**:
- `--experiment-config PATH` - 指定实验配置文件路径

---

## 🚀 使用方式

### all.json - 全面变异实验

**内容**:
- 所有16个模型（6个仓库）
- 每个模型变异所有支持的超参数
- 每个配置运行5次

**运行**:
```bash
sudo python3 mutation.py --experiment-config settings/all.json
```

**预计时间**: 10-50小时

**用途**:
- 全面探索超参数空间
- 建立完整的性能-能耗数据库

---

### default.json - 基线复现实验 ⭐

**内容**:
- 所有16个模型
- 使用各模型的原始默认超参数
- 每个模型运行1次
- **mode: "default"** - 不进行随机变异

**运行**:
```bash
sudo python3 mutation.py --experiment-config settings/default.json
```

**预计时间**: 5-20小时

**用途**:
- **这是唯一能复现原始训练过程的方式**
- 建立性能基线（baseline）
- 与变异实验对比
- 能耗基准测试

**示例配置片段**:
```json
{
  "repo": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "mode": "default",
  "hyperparameters": {
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 0.0001
  },
  "comment": "ResNet20原始默认配置（不设置seed）"
}
```

---

## 🔑 关键特性

### 1. 支持两种模式混合

可以在同一个配置文���中混合使用mutation和default模式：

```json
{
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mode": "default",
      "hyperparameters": {"epochs": 200, "learning_rate": 0.1}
    },
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mode": "mutation",
      "mutate": ["learning_rate"]
    }
  ]
}
```

### 2. 灵活的runs_per_config

每个配置可以运行多次：
- default模式：重复运行相同超参数，评估训练稳定性
- mutation模式：生成多个不同的变异

### 3. 自动休眠防干扰

- 同一配置的runs之间：60秒
- 不同配置之间：120秒

### 4. Governor支持

配置文件中可以指定CPU调度器：
```json
{
  "governor": "performance"
}
```

---

## 📊 实验对比

### 命令行模式 vs 配置文件模式

| 特性 | 命令行模式 | 配置文件模式 |
|------|-----------|-------------|
| 适用场景 | 单次快速测试 | 批量长期实验 |
| 复现能力 | ❌ 无法复现原始训练 | ✅ default模式可以 |
| 批量运行 | ❌ 需要脚本循环 | ✅ 原生支持 |
| 可维护性 | ❌ 命令难以保存 | ✅ 配置文件易管理 |
| 灵活性 | ✅ 适合临时测试 | ✅ 适合系统研究 |

---

## 💡 使用建议

### 推荐工作流程

1. **第一步：建立基线**
   ```bash
   sudo python3 mutation.py --experiment-config settings/default.json
   ```

2. **第二步：运行变异实验**
   ```bash
   sudo python3 mutation.py --experiment-config settings/all.json
   ```

3. **第三步：分析对比**
   ```bash
   cd results
   # 对比基线和变异的能耗差异
   cat *.json | jq -r '[.mode, .hyperparameters, .energy_metrics.cpu_energy_total_joules] | @csv'
   ```

### 自定义配置

可以基于预设配置创建自己的实验：

```bash
# 复制模板
cp settings/default.json settings/my_experiment.json

# 修改配置
vim settings/my_experiment.json

# 运行
python3 mutation.py --experiment-config settings/my_experiment.json
```

---

## 🎯 解决的问题

### 问题1: 如何复现原始训练？

**解决方案**: `default.json` 配置文件

**原因**:
- 命令行模式总是变异超参数，无法复现原始训练
- default模式使用固定的超参数值
- 配置文件明确记录了所有使用的默认值

### 问题2: 批量实验太麻烦

**解决方案**: 配置文件支持

**原因**:
- 不需要写bash循环脚本
- 配置文件更易维护和分享
- 自动处理休眠和错误

### 问题3: 实验难以重现

**解决方案**: JSON配置文件

**原因**:
- 配置文件完整记录实验设置
- 可以版本控制
- 易于分享和复现

---

## 📖 文档

### 核心文档

1. **settings/README.md** - 配置文件使用说明
   - 预设配置详解
   - 配置文件格式
   - 使用示例

2. **主README.md** - 更新了使用方式
   - 添加了配置文件模式说明
   - 推荐工作流程

3. **docs/CONFIG_EXPLANATION.md** - 配置说明
   - 仍然适用，解释models_config.json

---

## 🔬 技术实现

### 核心逻辑

```python
# mutation.py

def run_from_experiment_config(self, config_file: str):
    # 1. 加载配置文件
    exp_config = json.load(open(config_file))

    # 2. 遍历所有实验配置
    for exp in exp_config["experiments"]:
        repo = exp["repo"]
        model = exp["model"]
        mode = exp.get("mode", "mutation")

        if mode == "default":
            # Default模式：使用指定的超参数
            hyperparams = exp["hyperparameters"]
            self.run_experiment(repo, model, hyperparams)
        else:
            # Mutation模式：生成随机变异
            mutate_params = exp["mutate"]
            mutations = self.generate_mutations(repo, model, mutate_params)
            for mutation in mutations:
                self.run_experiment(repo, model, mutation)
```

### 向后兼容

- 旧的命令行模式完全保留
- 可以混合使用两种模式
- 不影响现有功能

---

## ✅ 测试

### 快速验证

```bash
# 1. 验证配置文件语法
python3 -c "import json; print(json.load(open('settings/default.json'))['experiment_name'])"

# 2. 快速测试（使用demo配置）
python3 mutation.py --experiment-config settings/mixed_mode_demo.json

# 3. 检查生成的结果
ls -lh results/*.json
cat results/*.json | jq '.mode'
```

---

## 🎓 示例命令

```bash
# 查看帮助
python3 mutation.py --help

# 列出模型
python3 mutation.py --list

# 命令行模式（旧方式，仍然支持）
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate all --runs 5

# 配置文件模式（新方式）
python3 mutation.py --experiment-config settings/all.json

# 复现基线（新功能）
python3 mutation.py --experiment-config settings/default.json
```

---

## 🌟 总结

### 已实现

✅ 配置文件支持（--experiment-config）
✅ Default模式（复现原始训练）
✅ Mutation模式（变异探索）
✅ 混合模式（同一配置文件中混合使用）
✅ 5个预设配置文件
✅ 完整文档
✅ 向后兼容

### 核心优势

1. **更方便** - 不需要写bash脚本循环
2. **可复现** - default模式真正复现原始训练
3. **易维护** - JSON配置文件易于编辑和版本控制
4. **更灵活** - 支持混合模式
5. **更可靠** - 自动处理休眠和重试

### 使用建议

**推荐**: 先运行`default.json`建立基线，再运行`all.json`进行变异探索。
