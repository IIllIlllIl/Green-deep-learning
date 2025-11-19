# 解决sudo运行后的文件权限问题

## 问题

使用sudo运行mutation.py后，生成的文件归root用户所有，导致普通用户无法直接访问和处理实验结果。

## 解决方案

### ✅ 方案1：自动权限恢复（已实现，推荐）

mutation.py现已内置自动权限恢复功能。当检测到使用sudo运行时，会在实验完成后自动将所有文件所有权恢复给原始用户。

**使用方法**：
```bash
# 正常使用sudo运行
sudo python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance

# 实验完成后会自动显示：
# 🔧 Restoring file ownership to user 'green'...
# ✅ File ownership restored: results/run_YYYYMMDD_HHMMSS
#    User 'green' can now access all files without sudo
```

**特点**：
- ✅ 完全自动，无需手动干预
- ✅ 使用环境变量`$SUDO_USER`获取原始用户
- ✅ 仅在检测到sudo运行时才执行
- ✅ 递归修改整个session目录的所有权
- ✅ 错误安全，失败不影响实验结果

**实现原理**：
```python
# 在session.py中新增restore_permissions()方法
def restore_permissions(self):
    if os.geteuid() != 0:  # 不是root，无需处理
        return

    sudo_user = os.environ.get('SUDO_USER')  # 获取原始用户
    # 使用chown -R递归修改所有权
    subprocess.run(['chown', '-R', f'{uid}:{gid}', str(self.session_dir)])
```

---

### 方案2：手动恢复权限（备用方案）

如果需要手动恢复旧数据的权限：

```bash
# 恢复单个session的权限
sudo chown -R green:green results/run_20251117_123456/

# 恢复所有results的权限
sudo chown -R green:green results/

# 或者使用当前用户
sudo chown -R $USER:$USER results/
```

---

### 方案3：配置perf权限（避免使用sudo）

**注意**：此方案可能有安全风险，仅适用于受信任的开发环境。

```bash
# 1. 设置perf_event_paranoid
sudo sysctl -w kernel.perf_event_paranoid=-1

# 2. 创建用户组并赋予perf权限
sudo groupadd perfusers
sudo usermod -a -G perfusers green

# 3. 设置perf工具权限
sudo chown root:perfusers /usr/bin/perf
sudo chmod 750 /usr/bin/perf
sudo setcap cap_sys_admin,cap_sys_ptrace,cap_syslog=eip /usr/bin/perf

# 4. 重新登录使组权限生效
# 然后可以不使用sudo运行
python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance
```

**缺点**：
- ❌ governor脚本仍需要root权限
- ❌ 安全性降低
- ❌ 设置复杂

---

### 方案4：使用umask控制权限（不推荐）

通过设置umask让创建的文件对所有人可读写：

```bash
# 临时设置umask
sudo bash -c 'umask 0000 && python3 mutation.py -ec settings/all.json'
```

**缺点**：
- ❌ 文件仍归root所有，只是权限更开放
- ❌ 安全性问题
- ❌ 不够优雅

---

## 推荐使用

### 新实验：方案1（自动恢复）
```bash
# 直接使用sudo运行，权限会自动恢复
sudo python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance
```

### 旧数据：方案2（手动恢复）
```bash
# 恢复之前实验的文件权限
sudo chown -R green:green results/single_default/
sudo chown -R green:green results/run_20251116_184943/
```

---

## 验证权限恢复

```bash
# 检查文件所有者
ls -l results/run_*/

# 应该看到：
# drwxr-xr-x green green ...
# （而不是 root root）

# 测试访问权限
cat results/run_*/summary.csv
# 应该能正常读取，无需sudo
```

---

## 实现细节

### 修改的文件

1. **mutation/session.py**
   - 新增`restore_permissions()`方法
   - 检测sudo运行并自动恢复权限

2. **mutation/runner.py**
   - 在两处`generate_summary_csv()`之后调用`restore_permissions()`
   - 确保所有实验模式都自动恢复权限

### 向后兼容

- ✅ 不使用sudo运行时不受影响
- ✅ 不改变任何现有功能
- ✅ 权限恢复失败不影响实验结果

---

**版本**: v4.2.0
**更新日期**: 2025-11-17
**状态**: ✅ 已实现并测试
