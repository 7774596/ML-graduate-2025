<div align="center">

# Assignment 4: 强化学习

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Deadline](https://img.shields.io/badge/Deadline-Dec%2014-red.svg)](http://101.132.193.95:3000)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

**📅 截止日期：12月14日** | **🏆 [查看排行榜](http://101.132.193.95:3000)** | **📋 Assignment ID: 04**

### 📤 提交方式: 通过可执行文件提交(可执行文件位于仓库的Releases下)

---

</div>

## 📋 任务概述

> 本次任务要求通过强化学习算法解决一个动态位置优化问题。在一个磁性球环境中，智能体（P）需要学习调整其位置，以最大化与动态变化的目标球（q）相关的奖励信号。目标球会根据智能体的移动而响应性地移动，形成一个复杂的动态系统。

## 🎯 环境描述

### 环境参数
- **轨道范围**: 0-99
- **动作范围**: -99到99
- **每轮步数**: 200步

### 状态和动作
- **状态**: 当前智能体位置 P
- **动作**: 移动距离（整数，范围-99到99）
- **位置更新逻辑**: 当前位置 P 执行动作 action 后，新位置为 `P_new = P + action`。如果新位置超出轨道边界 [0, 99]，会触发反弹机制：
  - 如果 `P_new < 0`：超出边界的距离除以2后反弹，即 `P_final = 0 + abs(P_new) // 2`
  - 如果 `P_new > 99`：超出边界的距离除以2后反弹，即 `P_final = 99 - (P_new - 99) // 2`
  - 最终位置会被限制在 [0, 99] 范围内
- **轨迹**: 包含历史状态、动作和奖励的序列，每个元素为 (P, action, reward) 三元组
- **奖励计算**: 奖励基于智能体P与目标球q之间的距离计算，距离越近奖励值越大


#### 轨迹示例

**初始状态示例**（第0步）：
```python
# 轨迹为空列表 []
state = 50  # 当前状态（执行动作前的位置P=50）
trajectory = []  # 空轨迹（初始状态）
```

**中间状态示例**（第5步）：
```python
# 轨迹包含前4步的历史记录，每个元素为(新状态, 动作, 奖励)
state = 67  # 当前状态（执行动作前的位置P=67）
trajectory = [
    (60, 10, 0.15),   # 第0步：从50移动到60，执行动作10，奖励0.15
    (55, -5, 0.23),   # 第1步：从60移动到55，执行动作-5，奖励0.23
    (63, 8, 0.18),    # 第2步：从55移动到63，执行动作8，奖励0.18
    (67, 4, 0.12)     # 第3步：从63移动到67，执行动作4，奖励0.12
]
```

### 📊 离线数据集说明

项目提供了`offline_data.csv`离线数据集，包含1000个trial的完整轨迹数据，可用于策略分析和算法调试。

#### 数据集字段说明
| 字段名 | 说明 | 
|--------|------|
| **trial_id** | 试验编号 |
| **step_id** | 当前试验中的步数 | 
| **P** | 智能体当前位置 | 
| **action** | 执行的动作 |
| **reward** | 获得的奖励 |

#### 数据集统计信息
- **总trial数**: 1000个
- **每个trial步数**: 200步
- **总数据量**: 200,000条记录

> 💡 **提示**: 离线数据可以帮助你理解环境动态，但最终评测使用的是实时环境，策略需要能够适应环境的随机性。

### 🎯 数据收集与策略优化建议

除了使用提供的离线数据，你还可以在 `policy` 函数中添加数据收集逻辑来生成自己的训练数据。

**数据收集策略建议**：
- **数据收集**: 在 `policy` 函数中，将当前状态和轨迹保存到文件中，以便后续分析。
- **数据分析**: 使用离线数据集进行策略分析，了解环境特性。
- **策略调整**: 根据数据分析结果，调整策略以适应环境变化。
通过收集自己的数据，你可以更好地理解环境特性，并针对性地优化策略。

## 💻 代码要求

### solution.py 代码要求
在 `solution.py` 中实现 `Solution` 类的policy函数：

```python
def policy(self, state, trajectory):
    """
    实现你的策略函数
    
    参数:
        state: 当前状态（智能体位置P）
        trajectory: 历史轨迹列表，每个元素为(P, action, reward)三元组
    
    返回:
        action: 要执行的动作（整数，范围-99到99）
    """
    # 数据收集示例：保存轨迹数据到文件
    # if len(trajectory) > 0:
    #     # 记录上一个状态、动作、奖励和当前状态
    #     with open("collected_data.csv", "a") as f:
    #         P, action, reward = trajectory[-1]
    #         f.write(f"{P},{action},{reward},{state}\n")
    
    # 你的策略实现
    action = np.random.randint(-99, 99)  # 示例动作，实际策略需要根据状态和轨迹进行计算
    # 可以根据收集的数据进行分析和策略调整
    
    return action
```

## 📈 评分标准 (总分 20分)

本次作业总分为 20 分。

### 评分规则（基于TotalReward）
根据TotalReward得分进行线性插值计算基础分数：

#### 评分基准点
- **基准线1**: TotalReward = 20 → **6分**
- **基准线2**: TotalReward = 94.61 → **10分**
- **排行榜前10%线**: TotalReward = FullScoreReward（排行榜前10%的最低分） → **20分**

#### 基础得分规则

| 场景 | 条件说明 | 基础得分 |
| :--- | :--- | :--- |
| **1** | TotalReward ≤ 20 | **6 分** |
| **2** | 20 < TotalReward < 94.61 | **6 ~ 10 分** (线性插值) |
| **3** | 94.61 ≤ TotalReward < FullScoreReward | **10 ~ 20 分** (线性插值) |
| **4** | TotalReward ≥ FullScoreReward | **20 分** |

---

## ⚙️ 环境要求

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-2.31+-green?style=for-the-badge&logo=python&logoColor=white)

</div>

## 🚀 运行评测

### 📦 1. 依赖安装:

推荐使用conda环境：

```bash
conda create -n ML python=3.8
conda activate ML
```

### ⚙️ 2. 设置环境变量

**🐧 Linux/ 🍎 macOS:**

```bash
export STUDENT_ID='你的学号'
export STUDENT_NAME='你的姓名'
export STUDENT_NICKNAME='你的昵称'
export MAIN_CONTRIBUTOR='AI' 
```

**🪟 Windows PowerShell:**

```powershell
$env:STUDENT_ID="你的学号"
$env:STUDENT_NAME="你的姓名"
$env:STUDENT_NICKNAME="你的昵称"
$env:MAIN_CONTRIBUTOR="human"
```

### ▶️ 3. 运行评测

**多平台可执行文件**：

根据你的操作系统选择相应的可执行文件：

**🐧 Linux/ 🍎 macOS:**
```bash

chmod +x evaluate-linux     # Linux版本
./evaluate-linux

chmod +x evaluate-macos     # macOS版本
./evaluate-macos
```


**🪟 Windows:**
```cmd
evaluate-win.exe
```
或者使用PowerShell:
```powershell
.\evaluate-win.exe
```

**评测配置**：
- **时间限制**: 60秒
- **步数**: 200步
- **评估指标**: 200步总奖励值
- **提交平台**: 支持水杉平台提交

**输出示例**：
```
Starting reinforcement learning assignment evaluation
============================================================
Evaluation configuration:
  Total steps: 200
  Time limit: 60 seconds
  Reward metric: 200-step total reward
============================================================
Found solution.py file
Running mode: 1

Starting reinforcement learning policy evaluation...
--------------------------------------------------
Evaluation completed:
  Steps completed: 200/200
  Total reward: 15.234567
  Evaluation time: 2.34 seconds
  Status: Completed normally

Submitting to leaderboard...
Submission successful!

Evaluation completed!
Final reward: 15.234567
```

### ⏱️ 超时处理说明

**重要**：评测系统有**60秒时间限制**。如果评估超时：

- **自动截断**：系统会立即停止后续步骤的执行
- **评估机制**：使用**当前已获得的总奖励**作为最终得分
- **正常计分**：超时不会影响得分计算，直接使用已完成的奖励值

---

## 🏆 Leaderboard

<div align="center">

### 🌐 访问地址

**🔗 [http://101.132.193.95:3000](http://101.132.193.95:3000)**

---

</div>

### 📋 提交说明

1. **提交方式**: 运行对应平台的可执行文件进行提交
2. **提交频率**: 支持多次提交，系统记录最佳成绩
3. **提交时间**: 截止日期前的所有提交均有效
4. **成绩计算**: 以截止日期前获得的最高TotalReward为准

---

### 🎉 祝你取得好成绩！

**📅 记得在截止日期前提交你的最佳成绩！**

---

Made with ❤️ for Machine Learning Education

</div>
