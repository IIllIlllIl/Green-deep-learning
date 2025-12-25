# 共享ID数据验证报告

**生成时间**: 2025-12-15 18:28:24

---

## 统计摘要

- 总数据行数: 584
- 共享ID行数: 77
- 验证通过: 46
- JSON未找到: 0
- 超参数不匹配: 0
- 能耗不匹配: 13
- 性能不匹配: 19

## 详细验证结果

### 行 212: MRT-OAST_default_001

- **timestamp**: 2025-11-26T23:05:10.088419
- **状态**: mismatch
- **JSON**: `results/run_20251126_224751/MRT-OAST_default_001/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4396.0`, JSON=`4583.0`
- `precision`: CSV=`0.9419`, JSON=`0.976605`
- `recall`: CSV=`0.8089`, JSON=`0.742656`

---

### 行 320: MRT-OAST_default_001

- **timestamp**: 2025-12-02T19:20:10.686027
- **状态**: mismatch
- **JSON**: `results/run_20251202_185830/MRT-OAST_default_001/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4396.0`, JSON=`4934.0`
- `precision`: CSV=`0.9419`, JSON=`0.999124`
- `recall`: CSV=`0.8089`, JSON=`0.944146`

---

### 行 332: MRT-OAST_default_001

- **timestamp**: 2025-12-03T23:20:40.746427
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/MRT-OAST_default_001/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4396.0`, JSON=`4745.0`
- `precision`: CSV=`0.9419`, JSON=`0.986171`
- `recall`: CSV=`0.8089`, JSON=`0.855606`

---

### 行 215: MRT-OAST_default_004

- **timestamp**: 2025-11-27T00:17:51.991618
- **状态**: mismatch
- **JSON**: `results/run_20251126_224751/MRT-OAST_default_004/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4413.0`, JSON=`4738.0`
- `precision`: CSV=`0.9396`, JSON=`0.985039`
- `recall`: CSV=`0.8185`, JSON=`0.844435`

---

### 行 477: MRT-OAST_default_004

- **timestamp**: 2025-12-13T00:56:10.836826
- **状态**: mismatch
- **JSON**: `results/run_20251212_224937/MRT-OAST_default_004/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`34763.75` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`2673.32` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`37437.07` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`249.8013808267369` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`317.0` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`71.47` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`284024.16999999987` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`79.12225153913808` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`84.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`92.53737906772207` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

**性能不匹配** (3个):

- `accuracy`: CSV=`4285.0`, JSON=`4580.0`
- `precision`: CSV=`0.9137`, JSON=`0.979006`
- `recall`: CSV=`0.7894`, JSON=`0.73314`

---

### 行 216: MRT-OAST_default_005

- **timestamp**: 2025-11-27T00:40:39.697491
- **状态**: mismatch
- **JSON**: `results/run_20251126_224751/MRT-OAST_default_005/experiment.json`

**性能不匹配** (2个):

- `accuracy`: CSV=`4294.0`, JSON=`4622.0`
- `precision`: CSV=`0.9258`, JSON=`0.98178`

---

### 行 382: MRT-OAST_default_005

- **timestamp**: 2025-12-05T21:40:59.101497
- **状态**: mismatch
- **JSON**: `results/run_20251205_211726/MRT-OAST_default_005/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4294.0`, JSON=`4460.0`
- `precision`: CSV=`0.9258`, JSON=`0.948565`
- `recall`: CSV=`0.781`, JSON=`0.724866`

---

### 行 485: MRT-OAST_default_030_parallel

- **timestamp**: 2025-12-14T13:38:50.509943
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/MRT-OAST_default_030_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4303.0`, JSON=`0.883`
- `precision`: CSV=`0.9253`, JSON=`0.9826`
- `recall`: CSV=`0.7854`, JSON=`0.7719`

---

### 行 523: MRT-OAST_default_030_parallel

- **timestamp**: 2025-12-15T14:00:13.339026
- **状态**: mismatch
- **JSON**: `results/run_20251214_160925/MRT-OAST_default_030_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4364.0`, JSON=`0.8926`
- `precision`: CSV=`0.9325`, JSON=`0.9831`
- `recall`: CSV=`0.8045`, JSON=`0.7917`

---

### 行 486: MRT-OAST_default_031_parallel

- **timestamp**: 2025-12-14T14:06:34.663941
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/MRT-OAST_default_031_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4185.0`, JSON=`0.8422`
- `precision`: CSV=`0.8983`, JSON=`0.9744`
- `recall`: CSV=`0.7611`, JSON=`0.6921`

---

### 行 524: MRT-OAST_default_031_parallel

- **timestamp**: 2025-12-15T14:28:09.569087
- **状态**: mismatch
- **JSON**: `results/run_20251214_160925/MRT-OAST_default_031_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4306.0`, JSON=`0.8578`
- `precision`: CSV=`0.9315`, JSON=`0.9776`
- `recall`: CSV=`0.7806`, JSON=`0.7227`

---

### 行 487: MRT-OAST_default_032_parallel

- **timestamp**: 2025-12-14T14:34:16.631664
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/MRT-OAST_default_032_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4329.0`, JSON=`0.8776`
- `precision`: CSV=`0.9135`, JSON=`0.9788`
- `recall`: CSV=`0.8089`, JSON=`0.7636`

---

### 行 525: MRT-OAST_default_032_parallel

- **timestamp**: 2025-12-15T14:56:05.268245
- **状态**: mismatch
- **JSON**: `results/run_20251214_160925/MRT-OAST_default_032_parallel/experiment.json`

**性能不匹配** (3个):

- `accuracy`: CSV=`4314.0`, JSON=`0.8648`
- `precision`: CSV=`0.9249`, JSON=`0.9796`
- `recall`: CSV=`0.7906`, JSON=`0.736`

---

### 行 333: Person_reID_baseline_pytorch_hrnet18_002

- **timestamp**: 2025-12-04T00:29:59.987074
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/Person_reID_baseline_pytorch_hrnet18_002/experiment.json`

**性能不匹配** (3个):

- `map`: CSV=`0.781272`, JSON=`0.626317`
- `rank1`: CSV=`0.917162`, JSON=`0.832245`
- `rank5`: CSV=`0.967637`, JSON=`0.936461`

---

### 行 334: Person_reID_baseline_pytorch_hrnet18_003

- **timestamp**: 2025-12-04T01:39:40.473332
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/Person_reID_baseline_pytorch_hrnet18_003/experiment.json`

**性能不匹配** (1个):

- `map`: CSV=`0.765036`, JSON=`0.770546`

---

### 行 335: Person_reID_baseline_pytorch_hrnet18_004

- **timestamp**: 2025-12-04T02:52:04.281180
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/Person_reID_baseline_pytorch_hrnet18_004/experiment.json`

**性能不匹配** (3个):

- `map`: CSV=`0.738058`, JSON=`0.763961`
- `rank1`: CSV=`0.894002`, JSON=`0.908551`
- `rank5`: CSV=`0.95962`, JSON=`0.964964`

---

### 行 336: Person_reID_baseline_pytorch_pcb_005

- **timestamp**: 2025-12-04T04:00:08.109631
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/Person_reID_baseline_pytorch_pcb_005/experiment.json`

**性能不匹配** (3个):

- `map`: CSV=`0.77076`, JSON=`0.77942`
- `rank1`: CSV=`0.92785`, JSON=`0.925178`
- `rank5`: CSV=`0.969418`, JSON=`0.976247`

---

### 行 337: Person_reID_baseline_pytorch_pcb_006

- **timestamp**: 2025-12-04T05:12:21.266716
- **状态**: mismatch
- **JSON**: `results/run_20251203_225507/Person_reID_baseline_pytorch_pcb_006/experiment.json`

**性能不匹配** (2个):

- `map`: CSV=`0.7765`, JSON=`0.772052`
- `rank1`: CSV=`0.924287`, JSON=`0.922803`

---

### 行 489: VulBERTa_mlp_001

- **timestamp**: 2025-12-13T21:28:09.540399
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_001/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`91348.39` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`7180.83` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`98529.22` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`245.37791025221026` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.4` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`4.13` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`749138.7599999979` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`77.40550278414675` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`88.86275794300688` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 478: VulBERTa_mlp_002

- **timestamp**: 2025-12-13T00:25:14.656413
- **状态**: mismatch
- **JSON**: `results/run_20251212_224937/VulBERTa_mlp_002/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`164563.83` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`12846.76` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`177410.59` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`230.1179148550723` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.3` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`4.01` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`1270250.8899999992` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`77.59420289855072` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`84.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`89.64021739130435` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 490: VulBERTa_mlp_002

- **timestamp**: 2025-12-13T22:22:21.582872
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_002/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`92552.06` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`7222.72` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`99774.78` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`232.74800257731957` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.46` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`4.82` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`722449.7999999999` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`77.97164948453609` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`88.13595360824742` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 491: VulBERTa_mlp_003

- **timestamp**: 2025-12-13T23:16:23.606582
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_003/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`92183.6` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`7219.03` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`99402.63` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`234.6684389140272` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.18` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`4.76` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`726064.1500000001` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`78.02714932126698` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`88.31480284421461` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 492: VulBERTa_mlp_004

- **timestamp**: 2025-12-13T23:49:33.150977
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_004/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`56325.5` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`4376.12` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`60701.62` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`237.99857066666715` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.29` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`4.73` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`446247.3200000009` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`78.22986666666667` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`87.2528` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 493: VulBERTa_mlp_005

- **timestamp**: 2025-12-14T00:16:28.616054
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_005/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`47275.08` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`3650.02` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`50925.1` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`236.92819630337803` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`318.69` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`5.51` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`371740.34000000014` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`78.61376673040154` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`86.94263862332696` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 494: VulBERTa_mlp_006

- **timestamp**: 2025-12-14T01:10:07.965678
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_006/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`92156.27` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`7126.83` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`99283.1` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`238.11630448926473` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`318.76` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`5.09` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`731969.5199999998` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`78.14541314248537` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`84.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`88.29440468445023` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 495: VulBERTa_mlp_007

- **timestamp**: 2025-12-14T02:02:48.323090
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/VulBERTa_mlp_007/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`92338.81` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`7192.76` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`99531.57` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`237.51318802862718` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`319.24` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`5.38` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`730115.5399999999` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`78.16688353936239` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`83.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`88.22576447625244` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`100.0` (missing_in_csv)

---

### 行 537: VulBERTa_mlp_012_parallel

- **timestamp**: 2025-12-15T06:44:34.129122
- **状态**: mismatch
- **JSON**: `results/run_20251214_160925/VulBERTa_mlp_012_parallel/experiment.json`

**性能不匹配** (1个):

- `eval_loss`: CSV=`0.7177466154098511`, JSON=`0.7020828127861023`

---

### 行 503: bug-localization-by-dnn-and-rvsm_default_015

- **timestamp**: 2025-12-14T09:05:15.151891
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/bug-localization-by-dnn-and-rvsm_default_015/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`55706.56` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`2138.66` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`57845.22` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`74.10883852691218` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`84.36` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`56.92` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`26160.420000000002` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`54.93767705382436` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`62.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`0.32577903682719545` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`3.0` (missing_in_csv)

---

### 行 504: bug-localization-by-dnn-and-rvsm_default_016

- **timestamp**: 2025-12-14T09:20:26.360751
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/bug-localization-by-dnn-and-rvsm_default_016/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`55075.72` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`2124.77` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`57200.49` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`70.80417366946776` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`79.62` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`39.96` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`25277.08999999999` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`47.450980392156865` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`49.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`0.2801120448179272` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`3.0` (missing_in_csv)

---

### 行 505: bug-localization-by-dnn-and-rvsm_default_017

- **timestamp**: 2025-12-14T09:35:39.066123
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/bug-localization-by-dnn-and-rvsm_default_017/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`55106.38` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`2104.56` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`57210.94` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`69.77523809523807` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`77.83` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`41.55` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`24909.759999999987` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`45.47899159663866` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`46.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`0.24929971988795518` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`3.0` (missing_in_csv)

---

### 行 506: bug-localization-by-dnn-and-rvsm_default_018

- **timestamp**: 2025-12-14T09:49:41.938863
- **状态**: mismatch
- **JSON**: `results/run_20251213_203552/bug-localization-by-dnn-and-rvsm_default_018/experiment.json`

**能耗不匹配** (11个):

- `energy_cpu_pkg_joules`: CSV=`empty`, JSON=`54529.01` (missing_in_csv)
- `energy_cpu_ram_joules`: CSV=`empty`, JSON=`2056.87` (missing_in_csv)
- `energy_cpu_total_joules`: CSV=`empty`, JSON=`56585.88` (missing_in_csv)
- `energy_gpu_avg_watts`: CSV=`empty`, JSON=`69.55847025495747` (missing_in_csv)
- `energy_gpu_max_watts`: CSV=`empty`, JSON=`78.0` (missing_in_csv)
- `energy_gpu_min_watts`: CSV=`empty`, JSON=`41.1` (missing_in_csv)
- `energy_gpu_total_joules`: CSV=`empty`, JSON=`24554.139999999985` (missing_in_csv)
- `energy_gpu_temp_avg_celsius`: CSV=`empty`, JSON=`45.01416430594901` (missing_in_csv)
- `energy_gpu_temp_max_celsius`: CSV=`empty`, JSON=`46.0` (missing_in_csv)
- `energy_gpu_util_avg_percent`: CSV=`empty`, JSON=`0.2096317280453258` (missing_in_csv)
- `energy_gpu_util_max_percent`: CSV=`empty`, JSON=`3.0` (missing_in_csv)

---

