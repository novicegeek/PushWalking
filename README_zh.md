# 侧向扰动行走的生物力学分析

[English Version](./README.md) | [中文版本](./README_zh.md)

## 概述

本项目研究**正常行走过程中施加侧向推力产生的运动学与动力学效应**。实验使用定制的测力拳击靶，在特定步态时相向受试者施加不同强度的侧向推力，同步采集三维运动捕捉（Qualisys）、测力台（Kistler）和拳击靶传感器数据。数据分析流程涵盖原始信号处理、肌肉骨骼建模（OpenSim 逆运动学与逆动力学）、统计建模与可视化。

## 文件结构

### 配置与入口

| 文件 | 说明 |
|------|------|
| `main.py` | 流水线入口。定义受试者元数据并遍历所有受试者，启动 `analyze_subject()` 分析流程。 |
| `measuresheet.ipynb` | Jupyter notebook，生成实验用随机化试验列表，控制推击条件顺序与强度。 |

### 核心模块

| 文件 | 说明 |
|------|------|
| `basic.py` | **核心信号处理库。** 包含 `VoltConverter`（原始电压→力/力矩转换）、`spectral_subtraction` / `spectral_subtraction_same_trial`（行走噪声抑制）、`Mask`（冲击门控与静默检测）、`Filter`（Butterworth、中值、小波、陷波等滤波器组合）、`transform_pad_to_global`（坐标系转换）及各类常用工具函数。 |
| `subject.py` | **受试者级数据管理。** `Subject` 类负责读取试验列表、加载动作捕捉与拳击靶数据、批量处理全部试验以及汇总结果。 |
| `trial.py` | **单次试验分析流水线。** `Trial` 类协调数据去偏移、滤波、基于步态事件的裁剪、步态周期归一化、OpenSim IK/ID 求解以及运动学/动力学指标提取（速度、推击冲量、关节力矩等）。 |
| `file.py` | **文件 I/O 模块。** 提供 `.trc` / `.sto` 格式导出（供 OpenSim 使用）、通过子进程读取 `.c3d`、拳击靶数据读取与预处理（电压转换、坐标变换）以及 `.mot` / `.sto` 解析工具。 |

### 数据读取

| 文件 | 说明 |
|------|------|
| `readc3d.py` | `C3DReader` 类——通过持久性子进程读取 `.c3d` 文件，以规避同一进程中同时加载 `ezc3d` 和 `opensim` 导致的 DLL 冲突。 |
| `readc3d_subprocess.py` | 子进程工作脚本，使用 `ezc3d` 解析 `.c3d` 文件并通过 pickle 序列化在 stdin/stdout 上返回数据。 |

### QTM 处理

| 文件 | 说明 |
|------|------|
| `qtm_process.py` | **Qualisys QTM Python 脚本接口** 自动化脚本。实现步态事件检测（基于力阈值与跟骨标记点极小值的足跟着地识别）、时间轴裁剪、三维追踪、间隙填充、AIM（标记点自动识别）、C3D 导出及文件保存设置——支持单次试验与批处理两种模式。 |

### 统计与可视化

| 文件 | 说明 |
|------|------|
| `stats.py` | **统计建模。** 使用线性混合效应模型（LMM）分析行走速度和推击冲量对位移、速度变化量和关节力矩的影响。生成个体斜率图、Q-Q 诊断图并导出统计结果汇总表。 |
| `descriptive_stats.py` | 生成基线速度、推击冲量和峰值推击力的箱线图（按速度/强度条件分组），含合并数据叠加。 |
| `descriptive_stats_subplots.py` | 将三个描述性箱线图合并为一个水平排列的子图图像。 |

### 工具与杂项

| 文件 | 说明 |
|------|------|
| `test.py` | 临时测试脚本，用于测试拳击靶的坐标轴方向。 |
| `tmp.py` | 临时工具（如批量文件重命名）。 |

### 早期脚本

| 文件 | 说明 |
|------|------|
| `MATLAB/voltToMechanics.m` | MATLAB 版电压-力学量转换原型。 |
| `MATLAB/getSpeedTrialInd.m` | MATLAB 工具，从文件名中提取速度与试验序号。 |
| `MATLAB/testDataInspection.m` | MATLAB 数据检查测试脚本。 |

## 依赖

- Python 3.11+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `pywt`（小波去噪）
- `sympy`（符号计算）
- `opensim`（肌肉骨骼仿真——IK/ID）
- `ezc3d`（C3D 文件解析，子进程中使用）
- `statsmodels`（线性混合模型）
- `qtm`（Qualisys QTM 脚本 API）

## 分析流程

1. **实验设计** → `measuresheet.ipynb` 生成随机化试验列表。
2. **数据采集** → Qualisys（动作捕捉）、Kistler 测力台和测量拳击靶同步记录。
3. **QTM 处理** → `qtm_process.py` 检测步态事件、追踪标记点、填充间隙并导出 C3D。
4. **数据读取** → `readc3d.py` + `readc3d_subprocess.py` 解析 C3D 文件；`file.py` 读取拳击靶电压数据。
5. **信号处理** → `basic.py` 将电压转换为力、通过谱减法去除行走噪声、对推击阶段进行门控、滤波所有信号。
6. **分析** → `subject.py` + `trial.py` 编排每个受试者/每次试验的分析流程，包括 OpenSim IK/ID。
7. **统计与可视化** → `stats.py`、`descriptive_stats.py` 和 `descriptive_stats_subplots.py` 产出 LMM 结果和易读的图表。

## 注意事项

- 脚本中的所有文件路径均为绝对路径，指向特定的本地目录。在另一台机器上运行前，请修改 `DATA_DIR`、`ANALYSIS_DIR` 等相关路径。
- `readc3d.py` 采用子进程架构以避免 `ezc3d` 与 `opensim` 的 DLL 冲突——工作进程在多次请求间保持驻留以提高效率。
- 试验列表（`.csv`）定义了录制文件索引与实验条件之间的映射关系。
