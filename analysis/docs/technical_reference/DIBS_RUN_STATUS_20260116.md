# DiBS分析运行状态报告

**开始时间**: 2026-01-16 00:43
**状态**: ✅ 正常运行中
**当前任务**: Group 1 - examples（图像分类-小型）

---

## ✅ 已完成的准备工作

### 1. 环境修复
- ✅ NumPy降级: 2.4.1 → 1.26.4
- ✅ JAX版本: 0.4.25（兼容）
- ✅ 环境验证通过

### 2. 脚本配置
- ✅ 脚本路径: `scripts/run_dibs_6groups_final.py`
- ✅ Python解释器: `/home/green/miniconda3/envs/causal-research/bin/python`
- ✅ 运行模式: 后台无缓冲 (`-u` 参数)
- ✅ 日志文件: `dibs_run.log`

### 3. 数据验证
- ✅ 数据源: `data/energy_research/6groups_final/`
- ✅ 总记录数: 818条
- ✅ 任务组数: 6个
- ✅ 缺失值填充: 自动处理

---

## 📊 运行配置

### DiBS参数（最优配置）
```python
alpha_linear: 0.05        # DiBS默认值
beta_linear: 0.1          # ⭐ 关键参数 - 低无环约束
n_particles: 20           # 最佳性价比
n_steps: 5000             # 足够收敛
tau: 1.0                  # Gumbel-softmax温度
```

**配置来源**: `DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md`

### 任务组列表（6组）
1. **group1_examples** - examples（图像分类-小型）
   - 样本数: 304条
   - 特征数: 20个（4超参数, 1性能, 4能耗, 7中介, 4控制）
   - 状态: ⏳ 正在运行

2. **group2_vulberta** - VulBERTa（代码漏洞检测）
   - 预期样本: 72条
   - 预期特征: 20个

3. **group3_person_reid** - Person_reID（行人重识别）
   - 预期样本: 206条
   - 预期特征: 22个

4. **group4_bug_localization** - bug-localization（缺陷定位）
   - 预期样本: 90条
   - 预期特征: 21个

5. **group5_mrt_oast** - MRT-OAST（缺陷定位）
   - 预期样本: 72条
   - 预期特征: 21个

6. **group6_resnet** - pytorch_resnet（图像分类-ResNet）
   - 预期样本: 74条
   - 预期特征: 19个

---

## 📈 当前进度

### 进程信息
- **PID**: 3383845
- **CPU使用率**: 491%（多核）
- **内存使用**: 1.5%
- **运行时长**: 5分25秒（截至检查时）
- **状态**: Sl（Sleeping - interruptible）

### 日志输出（最新）
```
执行DiBS因果发现...
  alpha_linear: 0.05
  beta_linear: 0.1 ⭐ 关键参数
  n_particles: 20
  n_steps: 5000

开始DiBS因果图学习...
  变量数: 20
  样本数: 304
  迭代次数: 5000
  Alpha参数: 0.05
  粒子数: 20
  Beta参数: 0.1
  Tau参数: 1.0

正在运行DiBS算法（这可能需要几分钟）...
```

### 备注
- ⚠️ JAX使用CPU而非GPU（JAX未安装CUDA版本）
- ℹ️ 这不影响DiBS运行，只是速度可能较慢
- ✅ 所有缺失值已自动填充
- ✅ Timestamp列已移除（DiBS不支持字符串）

---

## 🔍 监控方法

### 快速监控
```bash
# 进入分析目录
cd /home/green/energy_dl/nightly/analysis

# 运行监控脚本
bash check_dibs_progress.sh
```

### 手动监控
```bash
# 查看实时日志
tail -f dibs_run.log

# 查看进程状态
ps aux | grep run_dibs_6groups_final

# 检查输出目录
ls -lh results/energy_research/dibs_6groups_final/
```

---

## 📝 预期结果

### 输出文件（每个任务组）
1. `{group_id}_causal_graph.npy` - 因果图矩阵
2. `{group_id}_feature_names.json` - 特征名称列表
3. `{group_id}_result.json` - 完整分析结果（包含3个研究问题的证据）

### 总结报告
- `DIBS_6GROUPS_FINAL_REPORT.md` - 综合分析报告

### 研究问题证据
**问题1**: 超参数对能耗的影响
- 直接因果边（超参数→能耗）
- 间接路径（超参数→中介→能耗）

**问题2**: 能耗-性能权衡关系
- 直接因果边（性能↔能耗）
- 共同超参数（同时影响能耗和性能）
- 中介权衡路径

**问题3**: 中介效应路径
- 超参数→中介→能耗
- 超参数→中介→性能
- 多步路径（≥4节点）

---

## ⏱️ 预计完成时间

### 基于历史数据（2026-01-05调优结果）
- **单组平均时间**: 15-20分钟
- **6组总时间**: 90-120分钟
- **预计完成**: 2026-01-16 02:00 - 02:30

### 实际可能更快（CPU优化）
- JAX在CPU上运行可能更稳定
- 数据质量高（99.5%）
- 缺失值处理完善

---

## 🚨 故障排查

### 如果进程停止
```bash
# 检查日志最后的错误
tail -100 dibs_run.log

# 重新启动
cd /home/green/energy_dl/nightly/analysis
/home/green/miniconda3/envs/causal-research/bin/python -u scripts/run_dibs_6groups_final.py > dibs_run.log 2>&1 &
```

### 如果输出无更新
```bash
# 检查日志文件大小变化
watch -n 10 'ls -lh dibs_run.log'

# 或查看最新几行
watch -n 10 'tail -5 dibs_run.log'
```

---

## 📋 下一步检查清单（下次对话）

- [ ] 检查进程是否完成
- [ ] 查看完整日志输出
- [ ] 验证6个任务组结果文件
- [ ] 阅读总结报告
- [ ] 检查因果图统计（边数、强度等）
- [ ] 提取3个研究问题的证据
- [ ] 生成可视化图表

---

**报告生成时间**: 2026-01-16 00:45
**下次检查建议**: 2026-01-16 02:00（约1小时15分钟后）
