# 从迷宫开始的 RL

simple maze 是 Q 表  
feature maze 是 DQN

---

## 代码版本介绍

本仓库按学习阶段组织：`simple_maze` 使用 Q 表，`feature_maze` 使用 CNN-DQN。下表汇总当前已实现版本与文件路径。

| 版本 | 文件 | 算法 | 状态 |
|------|------|------|------|
| Simple Maze | `simple_maze/main.py` | Q 表 | 已实现 |
| Feature Maze v1 | `feature_maze/v1.py` | CNN-DQN | 已实现 |
| Feature Maze v1.1 | `feature_maze/v1.1.py` | CNN-DQN | 已实现 |
| Feature Maze v1_debug | `feature_maze/v1_debug.py` | CNN-DQN（诊断版） | 已实现 |
| v2.0.0 – v2.13.0 | 待新增 | CNN-DQN 实验系列 | 规划中 |

### simple_maze/main.py

固定 4×4 迷宫上的 **Q 表** 强化学习入门实现。状态为 `(x, y, has_key, got_reward)`，包含钥匙（K）、奖励格（R）、陷阱（T）等元素；动作以 85% 概率按意图执行（可设为 1.0 变为确定性）。使用 ε-greedy 策略训练约 3000 个 episode，训练前后分别展示随机策略与贪心策略路径。

```bash
python simple_maze/main.py
```

### feature_maze/v1.py

最早的 **CNN-DQN** 版本。8×8 随机迷宫生成（BFS 验证可达性），CNN 输入 3 通道（墙 / 智能体 / 终点）；支持手画迷宫（`S` 起点、`E` 终点）。训练、评估、可视化集成在同一脚本中，通过顶部常量开关控制。

```bash
python feature_maze/v1.py
```

### feature_maze/v1.1.py

在 v1 基础上重构的 **基础 DQN** 版本。提供 `train` / `eval` / `test` / `stats` / `all` 命令行模式；用手绘迷宫集合（`open_shortest`、`multi_open_paths` 等）与随机迷宫混合训练；内置 BFS 最短路径用于诊断。奖励仅含基础项（`goal +100, wall -1, step -0.05`），**不含 BFS shaping、visited 惩罚、action mask**。

```bash
# 训练
python feature_maze/v1.1.py --mode train

# 评估
python feature_maze/v1.1.py --mode eval --load-model cnn_dqn_maze_8x8_basic.pt

# 在手绘测试迷宫上逐步可视化
python feature_maze/v1.1.py --mode test --load-model cnn_dqn_maze_8x8_basic.pt
```

### feature_maze/v1_debug.py

面向对照实验的 **诊断版 DQN**，最接近下文 v2.0.0 基线设定。主要特性：

- **Stratified Replay**：按 wall / success / progress 等元数据分层采样
- **Curriculum**：easy / medium / hard / manual 四档难度，随 episode 渐进
- **Reward Presets**：`g10_w10_s001`、`g10_w5_s001` 等可切换，支持 `--mode reward_ablate` 批量对比
- **Epsilon Presets**：含 `two_phase` 两阶段衰减
- **诊断指标**：wall_hit_rate、repeat_count、progress_rate、path_ratio 等
- 明确 **无 action mask、无 visited mask、无 BFS reward shaping**

```bash
# 单次训练 + 评估
python feature_maze/v1_debug.py --mode single --reward-preset g10_w5_s001 --epsilon-preset two_phase

# 奖励预设消融实验
python feature_maze/v1_debug.py --mode reward_ablate

# 加载模型评估，首个失败 episode 输出 debug trace
python feature_maze/v1_debug.py --mode eval --model diagnose_cnn_dqn.pt --debug
```

### v2.n.0 系列（规划中）

下文 v2.0.0 – v2.13.0 为统一参数下的策略对照实验，每个版本独立可运行，逐个验证单一学习机制的效果。当前代码以 `v1_debug.py` 为实验平台基础，后续将按推荐顺序实现。

---

## 统一基础设定：v2.n 系列共同参数

所有 v2.n.0 模型都使用同一套基础参数，方便对照：

- reward_preset = goal +10, wall -10, step -0.01
- epsilon = two_phase
- episodes = 3000
- max_steps = 64
- replay_type = stratified
- 无 action mask
- 无 visited mask
- 无直接禁止重复
- CNN 结构不改
- Double DQN target 不改
- optimizer / loss / target network 不改

课程难度改成固定三阶段：

- Episode 1-1000: easy
- Episode 1001-2000: medium
- Episode 2001-3000: hard

epsilon 设计：

- epsilon_start = 1.0
- epsilon_mid = 0.20
- epsilon_end = 0.05
- epsilon 在 episode 2500 收缩到最低值 0.05
- episode 2500-3000 保持 0.05

也就是说，所有模型只比较"学习策略"的差异，不再混入 wall、epsilon、episode、网络结构等额外变量。

---

## v2.0.0：Baseline Curriculum DQN（基线课程 DQN）

这是新的基础对照组。

### 思路

只做你要求的统一基础设定：

- wall = -10
- two-phase epsilon
- 3000 episodes
- 每 1000 episode 进入 easy / medium / hard

不加入任何额外引导。

### 目的

用它回答：

单靠更清晰的课程难度 + 强撞墙惩罚，DQN 能不能自己学会走迷宫？

### 预计现象

它可能会继续学成：

- 少撞墙
- 少亏损
- 安全循环

但 success 不一定高。

这个模型是所有后续 v2.n.0 的对照基准。

---

## v2.1.0：BFS Demonstration Replay（BFS 专家示范回放）

这是我建议第一个实现的增强策略。

### 思路

每次生成一个迷宫后，用 BFS 找一条成功路径，然后把这条路径对应的 transition 放进 replay buffer。

例如：

```
s0 -> s1 -> s2 -> ... -> goal
```

转换成：

```
(s0, a0, r0, s1)
(s1, a1, r1, s2)
...
(sk, ak, +10, goal)
```

这些 transition 进入一个单独的：

demo_buffer

训练 batch 中固定抽一部分 demo 样本。

### 直觉解释

现在模型最大问题是：

它很少见过成功路径

所以 v2.1.0 的目标是：

- 先让模型知道终点确实可达
- 并且给它看一些真正能到终点的路径

不是替它走，也不是 mask 掉错误动作，而是让 replay buffer 里有"正确经验"。

### 专业术语

- demonstration replay（专家示范回放）
- learning from demonstrations，从示范中学习
- DQfD，Deep Q-learning from Demonstrations，基于专家示范的深度 Q 学习

### 验证问题

如果 v2.1.0 比 v2.0.0 明显提升 success，说明：

当前失败确实主要来自成功经验太少。

---

## v2.2.0：N-step DQN（n 步 DQN）

### 思路

普通 DQN 只用一步 target：

当前奖励 + 下一状态 Q

n-step DQN（n 步 DQN）会把后面 n 步真实奖励一起放进 target。

比如 5-step：

```
r0 + gamma*r1 + gamma^2*r2 + gamma^3*r3 + gamma^4*r4 + gamma^5*Q(s5)
```

### 直觉解释

原来终点 +10 要一格一格慢慢传回来：

goal -> 前一步 -> 再前一步 -> 再再前一步

n-step 相当于一次往前传多几格。

它不是凭空加奖励，而是：

把真实发生的后面几步结果一起拿来训练当前动作

### 专业术语

- n-step return（n 步回报）
- multi-step return（多步回报）
- n-step DQN（n 步 DQN）
- temporal-difference learning，TD learning（时序差分学习）

### 推荐参数

先做：

- n = 3

再做：

- n = 5

不建议一开始 n=10，因为失败轨迹太多时，太长的 n-step 会把很多乱走结果也强行打包进 target。

### 验证问题

如果 v2.2.0 比 v2.0.0 更好，说明：

当前主要问题之一是终点奖励通过 1-step TD 传播太慢。

---

## v2.3.0：BFS Progress Reward Shaping（BFS 进度奖励塑形）

### 思路

在原 reward 基础上加入一点短期引导：

- 如果 BFS 距离变小：+ small_reward
- 如果 BFS 距离变大：- small_penalty
- 如果距离不变：0

例如：

- progress_reward = +0.03
- regress_penalty = -0.03

### 直觉解释

现在模型只在终点拿 +10。

但很多动作离终点还很远，它不知道：

这一步到底有没有让事情变好

BFS progress shaping（BFS 进度奖励塑形）就是告诉它：

- 你离目标更近了一点，这一步稍微值得鼓励
- 你离目标更远了一点，这一步稍微应该扣分

### 重要边界

这不是 action mask。

它没有说：

你只能走 BFS 动作

它只是给短期反馈：

你这一步让离目标的最短路径距离变短了

### 专业术语

- reward shaping（奖励塑形）
- BFS progress reward（BFS 进度奖励）
- 更规范版本叫 potential-based reward shaping（基于势函数的奖励塑形）

### 验证问题

如果 v2.3.0 成功率上升、repeat 下降，说明：

当前模型确实缺少中间过程反馈。

---

## v2.4.0：Potential-Based Reward Shaping（基于势函数的奖励塑形）

这是 v2.3.0 的更规范版本。

### 思路

不用简单的：

- 距离变小 +0.03
- 距离变大 -0.03

而是定义一个势函数：

```
Phi(s) = - BFS_distance(s, goal)
```

然后 shaping reward 为：

```
F(s, s') = eta * (gamma * Phi(s') - Phi(s))
```

### 直觉解释

你可以把它想成：

- 状态本身有一个势能
- 越接近 goal，势能越高
- 这一步的奖励来自势能变化

这样比普通 progress reward 更稳定，因为它不是粗暴判断"近了/远了"，而是用一个连续的状态势能差。

### 专业术语

- potential-based reward shaping（基于势函数的奖励塑形）
- potential function（势函数）
- policy invariance（策略不变性）

### 验证问题

和 v2.3.0 对照：

简单 BFS 进度奖励和更规范的势函数奖励，哪个更稳定？

---

## v2.5.0：BFS Demonstration Replay + N-step DQN（专家示范回放 + n 步 DQN）

这是一个组合策略。

### 思路

v2.1.0 解决：

有没有成功路径经验？

v2.2.0 解决：

成功路径里的 +10 能不能更快往前传？

所以 v2.5.0 把两者合起来：

- BFS 成功路径进入 demo_buffer
- 同时训练 target 使用 n-step return

### 直觉解释

如果只放 BFS 成功路径，但仍然用 1-step DQN，+10 还是要慢慢往前传。

如果只用 n-step，但成功轨迹太少，n-step 也没有多少 +10 可以传。

组合后：

- 有成功路径
- 并且成功路径的价值能更快传到前面状态

### 专业术语

- demonstration replay（专家示范回放）
- n-step return（n 步回报）
- multi-step bootstrapping（多步自举）

### 验证问题

如果这个模型明显超过 v2.1.0 和 v2.2.0，说明：

成功经验数量和奖励传播速度都重要。

---

## v2.6.0：Behavior Cloning Pretraining + DQN（行为克隆预训练 + DQN）

### 思路

先不用强化学习。

先生成很多 BFS 状态-动作对：

```
state -> BFS 下一步动作
```

用监督学习训练 CNN，让它先模仿 BFS。

然后再接 DQN 训练。

### 直觉解释

这相当于先让模型学会：

像老师一样走几步

再让它用 DQN 自己调整价值判断。

### 专业术语

- behavior cloning（行为克隆）
- supervised pretraining（监督预训练）
- imitation learning（模仿学习）

### 优点

模型不再从纯随机开始。

### 风险

它可能变成"模仿 BFS"，而不是纯粹通过 reward 学出 Q 函数。

但这对诊断很有价值：

- 如果行为克隆都学不好，说明 CNN 表达能力或输入状态有问题
- 如果行为克隆学得好，DQN 学不好，说明问题在强化学习训练信号

---

## v2.7.0：Prioritized Experience Replay（优先经验回放）

### 思路

普通 replay 是随机抽样。

PER（优先经验回放）根据 TD error 决定样本优先级：

TD error 越大，越容易被抽到

TD error 是：

```
target - 当前 Q
```

直觉上就是：

错得越离谱的题，多复习几遍

### 专业术语

- Prioritized Experience Replay，PER（优先经验回放）
- TD error（时序差分误差）
- importance sampling（重要性采样）

### 在你项目中的简化版

可以不一开始做完整 PER，而是先做：

- demo samples
- success samples
- wall samples
- progress samples
- high TD-error samples

组合采样。

### 验证问题

如果 PER 有帮助，说明：

纯 stratified replay 还不够，模型需要更频繁复盘当前学不好的关键样本。

---

## v2.8.0：Hindsight Experience Replay（事后经验回放）

### 思路

如果 agent 没到真正 goal，但它到达了某个位置 p。

原任务失败：

目标是 G，没到

HER（事后经验回放）会重新解释这条轨迹：

假设目标本来是 p，那么这条轨迹成功了

因为你的状态里 goal 是一个通道，所以理论上可以把失败轨迹中的某个已访问位置改成新的 goal，重新生成状态和 reward。

### 直觉解释

就像：

你本来想去北京，结果只走到天津  
那我们至少把这段路当成"如何到天津"的成功经验复盘

### 专业术语

- Hindsight Experience Replay，HER（事后经验回放）
- goal relabeling（目标重标记）
- goal-conditioned RL（目标条件强化学习）

### 适用性

你的迷宫任务很适合 HER，因为 goal 是状态输入的一部分。

### 风险

实现复杂度比 demonstration replay 高，因为要重写状态里的 goal channel 和 reward。

建议放后面。

---

## v2.9.0：Monte Carlo Return DQN Variant（蒙特卡洛回报 DQN 变体）

### 思路

一条 episode 结束后，从每一步开始计算完整未来回报：

```
G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
```

然后用 G_t 训练 Q(s_t, a_t)。

### 直觉解释

不是只看下一步，也不是只看 n 步，而是完整复盘：

从这一步开始，后来整局到底过得怎么样？

### 专业术语

- Monte Carlo return（蒙特卡洛回报）
- episodic return（整局回报）
- return-to-go（从当前步到结束的回报）

### 优点

终点奖励可以直接影响整条成功轨迹。

### 缺点

方差大。失败轨迹多时，会让很多动作都被标成低价值。

建议作为理解实验，不一定作为主力方案。

---

## v2.10.0：AlphaGo-lite BFS Policy Target（简化 AlphaGo：BFS 策略目标）

这是 AlphaGo 思想的最简化版本。

### 思路

AlphaGo 的核心不是给每步人工 reward，而是用搜索产生更强的动作目标。

在迷宫里，BFS 就是一个搜索老师。

对每个状态：

BFS 找出能让最短路径变短的动作

然后训练一个 policy head 或者用辅助 loss，让网络更倾向这些动作。

如果保持 DQN 网络不改，可以加一个辅助损失：

```
Q(s, BFS_action) 应该高于其他动作
```

### 直觉解释

不是直接让 BFS 代替模型走。

而是：

BFS 搜索告诉模型：  
在这个状态下，哪些动作更有希望通向终点

这就类似 AlphaGo 里 MCTS 搜索给 policy network 提供更强动作分布。

### 专业术语

- search policy target（搜索策略目标）
- policy distillation（策略蒸馏）
- auxiliary loss（辅助损失）
- AlphaGo-style search supervision（AlphaGo 风格搜索监督）

### 注意

这个已经不再是纯 DQN，而是：

DQN + 搜索监督

所以建议放在后面。

---

## v2.11.0：AlphaGo-lite Value Target（简化 AlphaGo：状态价值目标）

### 思路

AlphaGo 有 value network（价值网络），预测当前局面最终输赢。

你的迷宫可以训练一个辅助 value：

从当前状态出发，最终是否能到 goal / 预计离 goal 多远

如果不改网络结构，也可以暂时不做。

如果改网络结构，可以让 CNN 输出：

4 个 Q 值 + 1 个 V(s)

### 直觉解释

DQN 的 Q 是：

这个动作之后长期好不好

Value 是：

这个状态本身有没有希望

对迷宫来说，value 可以帮助模型建立：

- 哪些区域更接近可达终点
- 哪些区域是死胡同

### 专业术语

- value network（价值网络）
- state value function，V(s)（状态价值函数）
- auxiliary value loss（辅助价值损失）

### 注意

这会改网络结构，所以不应放在最前。

---

## v2.12.0：AlphaGo-lite Search Replay（简化 AlphaGo：搜索生成经验）

### 思路

每次训练时，不只是用 agent 自己探索，还用 BFS / A* 生成一些搜索轨迹。

这和 v2.1.0 的 demonstration replay 很像，但更进一步：

- 不仅存一条最短路径
- 还可以存多个候选路径、绕路路径、失败路径的对比

### 直觉解释

AlphaGo 用 MCTS 生成比当前网络更强的训练目标。

迷宫里可以用 BFS/A* 生成比当前随机探索更强的经验。

### 专业术语

- planning-guided replay（规划引导回放）
- search-generated experience（搜索生成经验）
- model-based guidance（基于模型的引导）

### 注意

这已经比较接近"用规划帮助学习"，不再是纯 trial-and-error DQN。

---

## v2.13.0：Combined Strong Agent（综合强模型）

最后做一个组合模型，不用于理解单因素，而用于验证上限。

可能组合：

BFS demonstration replay  
+ n-step return  
+ potential-based reward shaping  
+ prioritized replay

### 目的

不是分析单个机制，而是验证：

如果把成功经验、奖励传播、中间反馈、样本优先级都处理好，当前 CNN-DQN 是否能学会泛化迷宫？

如果这个模型仍然失败，才更应该怀疑：

- CNN 表达能力不足
- 状态输入不足
- DQN 框架本身不适合这个泛化任务

---

## 推荐实现顺序

我建议不要一次实现所有，而是逐个 v2.n.0 写代码，每个都能独立运行。

顺序如下：

| 版本 | 名称 |
|------|------|
| v2.0.0 | Baseline Curriculum DQN（基线课程 DQN） |
| v2.1.0 | BFS Demonstration Replay（BFS 专家示范回放） |
| v2.2.0 | N-step DQN（n 步 DQN） |
| v2.3.0 | BFS Progress Reward Shaping（BFS 进度奖励塑形） |
| v2.5.0 | Demonstration + N-step（专家示范 + n 步） |
| v2.6.0 | Behavior Cloning + DQN（行为克隆 + DQN） |
| v2.7.0 | Prioritized Replay（优先经验回放） |
| v2.8.0 | HER（事后经验回放） |
| v2.10.0 | AlphaGo-lite BFS Policy Target（简化 AlphaGo：BFS 策略目标） |
| v2.13.0 | Combined Strong Agent（综合强模型） |

其中我最建议你马上实现的前四个是：

- v2.0.0 baseline
- v2.1.0 demonstration replay
- v2.2.0 n-step DQN
- v2.3.0 reward shaping

因为它们分别对应当前失败的四个核心问题：

- 成功经验太少
- 终点奖励传太慢
- 短期反馈缺失
- 局部安全循环

---

## 当前清单的核心逻辑

你可以把所有策略分成四大类：

**第一类：让模型看到成功**

- BFS demonstration replay（BFS 专家示范回放）
- behavior cloning（行为克隆）
- AlphaGo-lite search replay（简化 AlphaGo 搜索经验）

**第二类：让成功奖励传得更远**

- n-step return（n 步回报）
- Monte Carlo return（蒙特卡洛回报）

**第三类：给中间过程一点方向感**

- reward shaping（奖励塑形）
- potential-based shaping（基于势函数的奖励塑形）
- BFS progress reward（BFS 进度奖励）

**第四类：让模型更频繁复盘关键样本**

- prioritized replay（优先经验回放）
- stratified replay（分层经验回放）
- hindsight experience replay（事后经验回放）

AlphaGo 的思想可以归入第五类：

**第五类：用搜索把未来价值提前变成当前训练目标**

- MCTS / BFS search target（搜索目标）
- policy target（策略目标）
- value target（价值目标）
- planning-guided learning（规划引导学习）

---

下一步我建议从 v2.0.0 Baseline Curriculum DQN（基线课程 DQN） 开始，给你完整代码，然后再逐个增加 v2.1.0、v2.2.0、v2.3.0。
