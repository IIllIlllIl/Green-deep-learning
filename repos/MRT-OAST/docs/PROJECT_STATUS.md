# MRT-OAST 项目配置完成报告

## 环境配置状态 ✓

### Conda环境：mrt-oast
- Python 3.7
- PyTorch 1.13.1 (CUDA 11.7)
- NumPy 1.21.6
- Matplotlib 3.5.3
- 其他依赖：tqdm, javalang, tensorboard

### 硬件支持
- GPU: NVIDIA GeForce RTX 3080
- CUDA 11.7 已启用

## 数据集状态 ✓

### 数据集概览
总大小：273MB
总文件数：31个文件

### 三个主要数据集

#### 1. OJClone数据集
- 主文件：OJClone_with_AST+OAST.csv (283,990 行)
- 训练集：OJClone_train.csv (40,001 行)
- 验证集：OJClone_valid.csv (5,001 行)
- 测试集：OJClone_test.csv (5,001 行)
- AST字典：46 个词
- OAST字典：112 个词

#### 2. GCJ (Google Code Jam) 数据集
- 主文件：GCJ_with_AST+OAST.csv (103,569 行)
- 训练集：GCJ_train11.csv (353,399 行)
- 验证集：GCJ_valid.csv (13,696 行)
- 测试集：GCJ_test.csv (13,696 行)
- AST字典：54 个词
- OAST字典：414 个词

#### 3. BCB (BigCloneBench) 数据集
- 主文件：BCB_with_AST+OAST.csv (309,524 行)
- 训练集：BCB_train.csv (450,863 行)
- 验证集：BCB_valid.csv (416,329 行)
- 测试集：BCB_test.csv (416,329 行)
- AST字典：55 个词
- OAST字典：4,397 个词

### 额外文件
- 源代码文件：bigclonebenchdata_src.zip (5.9M)
- 源代码文件：googlejam4_src.zip (1.3M)

## 快速开始训练

### 1. 激活环境
```bash
conda activate mrt-oast
```

### 2. 训练不同数据集

#### 训练OJClone数据集
```bash
python main_batch.py --cuda \
    --data origindata/OJClone_with_AST+OAST.csv \
    --train_pair origindata/OJClone_train.csv \
    --valid_pair origindata/OJClone_valid.csv \
    --test_pair origindata/OJClone_test.csv \
    --dictionary origindata/OJClone_OAST_dictionary.txt \
    --ast_type OAST \
    --tag OJClone_OAST
```

#### 训练GCJ数据集（默认配置）
```bash
python main_batch.py --cuda \
    --data origindata/GCJ_with_AST+OAST.csv \
    --train_pair origindata/GCJ_train11.csv \
    --valid_pair origindata/GCJ_valid.csv \
    --test_pair origindata/GCJ_test.csv \
    --dictionary origindata/GCJ_OAST_dictionary.txt \
    --ast_type OAST \
    --tag GCJ_OAST
```

#### 训练BCB数据集
```bash
python main_batch.py --cuda \
    --data origindata/BCB_with_AST+OAST.csv \
    --train_pair origindata/BCB_train.csv \
    --valid_pair origindata/BCB_valid.csv \
    --test_pair origindata/BCB_test.csv \
    --dictionary origindata/BCB_OAST_dictionary.txt \
    --ast_type OAST \
    --tag BCB_OAST
```

### 3. 快速测试训练好的模型
```bash
# 假设已经训练好了GCJ数据集的模型
python main_batch.py --cuda --is_test --quick_test \
    --save model/BTransfrom_MRT_len256_batch64_GCJ_OAST
```

## 常用训练参数调整

### 调整学习率
```bash
python main_batch.py --cuda --lr 0.0005
```

### 调整批次大小（内存不足时）
```bash
python main_batch.py --cuda --batch_size 32
```

### 调整训练轮数
```bash
python main_batch.py --cuda --epochs 20
```

### 调整序列长度
```bash
python main_batch.py --cuda --sen_max_len 512
```

### 调整模型参数
```bash
python main_batch.py --cuda \
    --transformer_nlayers 4 \
    --d_model 256 \
    --d_ff 1024 \
    --h 16
```

## 监控训练过程

### 使用TensorBoard查看训练日志
```bash
# 激活环境后
conda activate mrt-oast

# 启动TensorBoard
tensorboard --logdir model/
```

然后在浏览器中访问：http://localhost:6006

## 项目文件结构

```
MRT-OAST/
├── environment.yml              # Conda环境配置
├── SETUP_CN.md                 # 环境配置说明
├── README.md                   # 项目说明
├── main_batch.py               # 训练入口
├── preprocess_data.py          # 数据预处理
├── dataset.py                  # 数据集类
├── tutils.py                   # 训练工具
├── quick_test.py               # 快速测试
├── nnet/
│   └── btransform.py          # MRT模型定义
├── origindata/                 # 数据集目录 ✓
│   ├── OJClone_*.csv
│   ├── GCJ_*.csv
│   ├── BCB_*.csv
│   └── *_dictionary.txt
├── model/                      # 模型保存目录（训练时自动创建）
└── AST构建工具
    ├── ast_builder.py
    ├── sast_builder.py
    ├── dast_builder.py
    ├── oast_builder.py
    └── build_java_ast.py
```

## 预期训练时间（参考）

根据不同数据集大小：
- OJClone (小): 约2-4小时 (10 epochs)
- GCJ (中): 约4-8小时 (10 epochs)
- BCB (大): 约8-16小时 (10 epochs)

注：实际时间取决于GPU性能和参数设置

## 下一步建议

1. **开始小规模测试**：先用OJClone数据集快速验证环境
2. **监控资源**：使用 `nvidia-smi` 监控GPU使用
3. **保存日志**：训练时重定向输出到日志文件
4. **定期检查**：通过TensorBoard监控训练进度

## 故障排除

### GPU内存不足
- 减小batch_size（如：32, 16）
- 减小sen_max_len（如：128）
- 减小模型维度（d_model, d_ff）

### 训练速度慢
- 检查是否使用了--cuda参数
- 检查GPU是否被其他进程占用：`nvidia-smi`

### 数据加载错误
- 确认数据文件路径正确
- 确认dictionary文件路径正确
- 检查CSV文件格式是否完整

## 相关文档

- 详细环境配置：SETUP_CN.md
- 原始README：README.md

---
配置时间：2025-10-13
GPU：NVIDIA GeForce RTX 3080
状态：✓ 已就绪，可以开始训练
