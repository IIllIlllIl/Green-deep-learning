# 实验仓库GitHub链接

**版本**: v4.3.0
**最后更新**: 2025-11-18

本文档列出了实验中使用的6个代码仓库的GitHub链接和相关信息。

---

## 📦 仓库列表

| # | 仓库名称 | GitHub链接 | Stars | 主要论文/作者 |
|---|---------|-----------|-------|-------------|
| 1 | **pytorch_resnet_cifar10** | [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) | ![Stars](https://img.shields.io/github/stars/akamaster/pytorch_resnet_cifar10) | Yerlan Idelbayev |
| 2 | **Person_reID_baseline_pytorch** | [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) | ![Stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch) | Zheng et al., CVPR 2019 |
| 3 | **VulBERTa** | [ICL-ml4csec/VulBERTa](https://github.com/ICL-ml4csec/VulBERTa) | ![Stars](https://img.shields.io/github/stars/ICL-ml4csec/VulBERTa) | Hanif & Maffeis, IJCNN 2022 |
| 4 | **examples** | [pytorch/examples](https://github.com/pytorch/examples) | ![Stars](https://img.shields.io/github/stars/pytorch/examples) | PyTorch Team |
| 5 | **MRT-OAST** | [UnbSky/MRT-OAST](https://github.com/UnbSky/MRT-OAST) | ![Stars](https://img.shields.io/github/stars/UnbSky/MRT-OAST) | Code Clone Detection |
| 6 | **bug-localization-by-dnn-and-rvsm** | [emredogan7/bug-localization-by-dnn-and-rvsm](https://github.com/emredogan7/bug-localization-by-dnn-and-rvsm) | ![Stars](https://img.shields.io/github/stars/emredogan7/bug-localization-by-dnn-and-rvsm) | Emre Dogan & Hamdi Alperen Cetin |

---

## 🔍 详细信息

### 1. pytorch_resnet_cifar10 (ResNet for CIFAR-10)

**GitHub**: https://github.com/akamaster/pytorch_resnet_cifar10

**描述**: Proper ResNet Implementation for CIFAR10/CIFAR100 in PyTorch

**特点**:
- 严格按照[He et al., 2016](https://arxiv.org/abs/1512.03385)原始论文实现
- 提供ResNet-20/32/44/56/110/1202预训练模型
- 比原论文更好的测试错误率

**引用**:
```bibtex
@misc{Idelbayev18a,
  author       = "Yerlan Idelbayev",
  title        = "Proper {ResNet} Implementation for {CIFAR10/CIFAR100} in {PyTorch}",
  howpublished = "\url{https://github.com/akamaster/pytorch_resnet_cifar10}",
  year         = "2018"
}
```

**许可证**: MIT

**预训练模型**:
- ResNet20: [下载链接](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th)
- ResNet32: [下载链接](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32.th)
- ResNet44: [下载链接](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44.th)
- ResNet56: [下载链接](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56.th)

---

### 2. Person_reID_baseline_pytorch (行人重识别)

**GitHub**: https://github.com/layumi/Person_reID_baseline_pytorch

**描述**: Strong, Small, Friendly baseline for Person Re-identification

**特点**:
- 2500+ 引用
- 支持多种backbone: ResNet, DenseNet, HRNet, Swin Transformer, EfficientNet
- 支持多种损失函数: Circle Loss, Triplet Loss, Contrastive Loss等
- BF16/FP16支持，仅需2GB显存
- PCB (Part-based Convolutional Baseline)
- GPU Re-Ranking

**主要论文**:
```bibtex
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

**许可证**: MIT

**性能** (Market-1501):
- ResNet-50: Rank@1=88.84%, mAP=71.59%
- DenseNet-121: Rank@1=90.17%, mAP=74.02%
- HRNet-18: Rank@1=90.83%, mAP=76.65%
- PCB: Rank@1=92.64%, mAP=77.47%
- Swin (all tricks): Rank@1=94.12%, mAP=84.39%

**相关资源**:
- [8分钟教程](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/README.md)
- [中文视频简介](https://www.bilibili.com/video/BV11K4y1f7eQ)
- [Google Colab](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab)

---

### 3. VulBERTa (代码漏洞检测)

**GitHub**: https://github.com/ICL-ml4csec/VulBERTa

**描述**: Simplified Source Code Pre-Training for Vulnerability Detection

**特点**:
- 基于RoBERTa的代码漏洞检测模型
- 自定义tokenization pipeline
- 在多个数据集上达到SOTA性能: Vuldeepecker, Draper, REVEAL, muVuldeepecker

**主要论文**:
```bibtex
@INPROCEEDINGS{hanif2022vulberta,
  author={Hanif, Hazim and Maffeis, Sergio},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  title={VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection},
  year={2022},
  pages={1-8},
  doi={10.1109/IJCNN55064.2022.9892280}
}
```

**会议**: IJCNN 2022 (Oral Presentation)

**论文链接**: https://ieeexplore.ieee.org/document/9892280

**数据集**:
- Tokenizer训练数据
- 预训练数据 (DrapGH)
- Fine-tuning数据 (Devign, d2a等)

**模型**:
- VulBERTa-MLP
- VulBERTa-CNN

---

### 4. examples (PyTorch官方示例)

**GitHub**: https://github.com/pytorch/examples

**官方网站**: https://pytorch.org/examples/

**描述**: PyTorch官方示例仓库，包含各种高质量、易理解的示例代码

**使用的模型**:
- **MNIST (CNN)**: [mnist/](https://github.com/pytorch/examples/tree/main/mnist)
- **MNIST RNN**: [mnist_rnn/](https://github.com/pytorch/examples/tree/main/mnist_rnn)
- **MNIST Forward-Forward**: [mnist_forward_forward/](https://github.com/pytorch/examples/tree/main/mnist_forward_forward)
- **Siamese Network**: [siamese_network/](https://github.com/pytorch/examples/tree/main/siamese_network)
- **Word Language Model**: [word_language_model/](https://github.com/pytorch/examples/tree/main/word_language_model)

**其他资源**:
- PyTorch教程: https://github.com/pytorch/tutorials
- PyTorch Hub: https://pytorch.org/hub/
- 生产环境recipes: https://github.com/facebookresearch/recipes
- 社区支持: https://discuss.pytorch.org/

**许可证**: BSD-3-Clause

---

### 5. MRT-OAST (代码克隆检测)

**GitHub**: https://github.com/UnbSky/MRT-OAST

**描述**: MRT-OAST for Code Clone Detection

**全称**: Multiple Representation Transformer with Optimized Abstract Syntax Tree

**特点**:
- 基于Transformer的代码克隆检测
- 使用优化的抽象语法树(OAST)
- 支持OJClone、GCJ、BCB数据集

**数据集**:
- OJClone with AST+OAST
- Google Code Jam (GCJ)
- BigCloneBench (BCB)

**技术栈**:
- PyTorch 1.13.1
- Python 3.7
- Javalang (Java代码解析)

**作者**: UnbSky

**相关资源**:
- [快速开始指南](https://github.com/UnbSky/MRT-OAST/blob/main/docs/QUICKSTART.md)
- [环境配置](https://github.com/UnbSky/MRT-OAST/blob/main/docs/SETUP_CN.md)
- [训练脚本文档](https://github.com/UnbSky/MRT-OAST/blob/main/docs/SCRIPTS_GUIDE.md)

---

### 6. bug-localization-by-dnn-and-rvsm (软件缺陷定位)

**GitHub**: https://github.com/emredogan7/bug-localization-by-dnn-and-rvsm

**描述**: Bug Localization with Combination of Deep Learning and Information Retrieval

**作者**: Emre Dogan & Hamdi Alperen Cetin

**参考论文**: [Bug Localization with Combination of Deep Learning and Information Retrieval](https://ieeexplore.ieee.org/document/7961519)

**数据集**:
- Eclipse Platform UI
- 源代码: [eclipse/eclipse.platform.ui](https://github.com/eclipse/eclipse.platform.ui)
- Bug报告: [logpai/bugrepo/EclipsePlatform](https://github.com/logpai/bugrepo/tree/master/EclipsePlatform)

**方法**:
- rVSM (Revised Vector Space Model) - 信息检索
- DNN (Deep Neural Network) - 深度学习
- 混合模型计算bug报告与源文件的相关性

**性能**:
- Top-20准确率: ~79% (原论文: ~85%)
- Top-10准确率: ~65%
- Top-5准确率: ~50%

**支持的项目**:
- AspectJ
- Eclipse
- SWT
- Tomcat
- JDT

---

## 📊 仓库统计

### 按Star数排序 (截至2025-11)

1. **pytorch/examples**: 20,000+ ⭐
2. **layumi/Person_reID_baseline_pytorch**: 4,500+ ⭐
3. **akamaster/pytorch_resnet_cifar10**: 2,200+ ⭐
4. **ICL-ml4csec/VulBERTa**: 200+ ⭐
5. **UnbSky/MRT-OAST**: 10+ ⭐
6. **emredogan7/bug-localization-by-dnn-and-rvsm**: 5+ ⭐

### 按主要语言

| 语言 | 仓库 |
|------|------|
| **Python** | 全部 (6个) |
| **C++** | examples (C++ Frontend示例) |
| **Java** | MRT-OAST (AST解析), bug-localization (Java项目分析) |

### 按应用领域

| 领域 | 仓库数 | 仓库列表 |
|------|--------|---------|
| **计算机视觉** | 2 | pytorch_resnet_cifar10, Person_reID_baseline_pytorch |
| **代码分析** | 3 | VulBERTa, MRT-OAST, bug-localization |
| **通用ML/DL** | 1 | examples |

---

## 🔗 相关链接

### 数据集

**计算机视觉**:
- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- Market-1501: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
- MNIST: http://yann.lecun.com/exdb/mnist/

**代码分析**:
- Devign: https://github.com/microsoft/CodeXGLUE
- BigCloneBench: https://github.com/clonebench/BigCloneBench
- Eclipse Bug Repository: https://github.com/logpai/bugrepo

### 预训练模型

**HuggingFace**:
- CodeBERT: https://huggingface.co/microsoft/codebert-base
- RoBERTa: https://huggingface.co/roberta-base

**Timm (PyTorch Image Models)**:
- HRNet: `timm.create_model('hrnet_w18', pretrained=True)`
- EfficientNet: `timm.create_model('efficientnet_b4', pretrained=True)`

**Torchvision**:
- ResNet-50: `torchvision.models.resnet50(pretrained=True)`
- DenseNet-121: `torchvision.models.densenet121(pretrained=True)`

---

## 📝 引用格式

如果您在研究中使用了这些仓库，请引用相应的论文：

### ResNet on CIFAR-10
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}
```

### Person Re-ID Baseline
```bibtex
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={CVPR},
  year={2019}
}
```

### VulBERTa
```bibtex
@inproceedings{hanif2022vulberta,
  author={Hanif, Hazim and Maffeis, Sergio},
  booktitle={IJCNN},
  title={VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection},
  year={2022}
}
```

---

## ⚠️ 许可证信息

| 仓库 | 许可证 | 商业使用 |
|------|--------|---------|
| pytorch_resnet_cifar10 | MIT | ✅ |
| Person_reID_baseline_pytorch | MIT | ✅ |
| VulBERTa | *MIT (推测)* | ✅ |
| examples | BSD-3-Clause | ✅ |
| MRT-OAST | *无明确许可证* | ⚠️ |
| bug-localization | *无明确许可证* | ⚠️ |

**注意**: 使用代码前请查看各仓库的LICENSE文件。

---

## 🔄 更新日志

- **2025-11-18 19:20**: 补充MRT-OAST和bug-localization的GitHub链接
- **2025-11-18 16:00**: 初始版本，收集6个仓库的GitHub链接和基本信息

---

## 📧 联系方式

如果您发现链接失效或有更新信息，请通过以下方式联系：

- 项目Issue: [提交Issue](https://github.com/your-repo/issues)
- 邮件: green@example.com

---

**文档版本**: v4.3.0
**维护者**: Green
**最后更新**: 2025-11-18
