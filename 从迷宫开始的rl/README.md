# 项目背景与近期技术路线：从迷宫强化学习到二进制漏洞挖掘

本项目的近期目标不是直接追求复杂环境下的最终智能体，而是从一个足够小、足够可控的 8×8 迷宫开始，逐步建立强化学习任务中的数据生成、奖励建模、策略学习、模型诊断和泛化评估方法。迷宫在这里不是最终任务，而是一个可解释、可度量、可逐步复杂化的实验载体。

项目的长期迁移方向是二进制漏洞挖掘。二进制程序分析同样具有路径搜索、状态爆炸、局部选择、长期目标、无效动作、高代价反馈和探索收益等问题。因此，迷宫任务承担的是“方法训练场”的角色：先在二维迷宫中学会如何定义状态、动作、奖励、偏好、探索、失败类型和调试指标，再逐步迁移到更复杂、更概率化、更接近真实程序执行状态空间的任务中。

---

## 一、2.0.n 阶段的工程经验总结

2.0.n 阶段的核心收获是：强化学习失败时，不能只看 success rate，也不能只调 reward 数字。必须把问题拆成四层：

1. 环境是否定义清楚；
2. reward 是否表达了正确行为；
3. 模型是否有能力拟合目标函数；
4. 策略学习是否能把局部 reward 转化成长期规划。

早期版本从 DQN baseline 开始，逐步尝试 reward scale、CEM 搜索、repeat penalty、visited state、trajectory preference RM、normalized margin、trajectory family、local segment preference、direction × visit × wall 因子化语义。每一次失败都暴露了一个新的工程约束。

第一类经验是：手写 reward 很难表达“正确过程”。goal reward、wall penalty、step penalty 和 repeat penalty 可以让模型学会某些局部偏好，但不能保证它学到 BFS 最短路径。repeat penalty 尤其容易误伤 recovery 行为，因为 revisit 既可能是循环，也可能是死胡同回撤后的恢复。

第二类经验是：轨迹偏好 RM 能学粗粒度偏好，但默认不会自动学到局部动作价值。只给 success > timeout 的偏好，RM 很容易成为轨迹排序器，而不是单步 reward model。BTL accuracy 很高并不等价于 greedy rollout 能成功。

第三类经验是：数据生成比 loss 更重要。RM 的质量主要由 pair 的语义纯度、分布平衡、margin 设计和 debug probe 决定。尤其是 visit、wall、away 这类行为，如果样本分布不平衡，模型会学到错误相关性。例如 visit 可能被学成 failure 信号，away 可能被 safe > wall 间接抬高，wall 可能只在 repeated wall 场景中被学坏，而普通 wall step 没学稳。

第四类经验是：必须区分 margin 和 primitive pressure。margin 是 BTL 中 Score(x+) - Score(x-) 的分离阈值；primitive pressure 是按样本占比、平均 margin 和语义差值聚合后得到的方向性诊断向量。二者量纲不同，不能直接比较。primitive pressure 的价值不是告诉我们 loss 有多大，而是告诉我们 toward、away、wall、visit 分别被数据往哪个方向推。

第五类经验是：RM 训练必须有独立诊断链路。最终我们形成了比较完整的 RM-only debug 流程，包括 sample profile distribution、primitive distribution、local bucket context distribution、observed primitive pressure、RM primitive reward score report、contextual probe、pure argmax greedy、BFS tie-break greedy、stuck timeout analysis。这套报告比单纯训练曲线更重要。

第六类经验是：当前瓶颈已经从 RM 转向策略模型。最后的 v2.0.9 系列中，RM 已经能够较好地区分 toward、away、wall、stuck_escape、stuck_loop，但纯 CNN/QCNN 策略仍然容易出现 wall chosen、away choice、stuck timeout 和 action saturation。这说明 CNN 更像在识别局部纹理，而不是执行 BFS 式规划。后续必须先验证模型本身是否具备拟合 BFS 的能力，再继续强化学习。

---

## 二、近期总路线

近期路线分三步：

第一步，进入 1.x.y 系列，暂时不做 DQN，不做 RM，不做 exploration，只研究“模型能否拟合 BFS”。目标是找到一个能够学习 BFS 本质的模型结构。验收指标是：只用 BFS 监督训练，模型在随机迷宫上达到 95% 以上 success，并达到 80% 以上最短路径规划成功率。

第二步，回到 2.0.10，把 2.0.9 已经整理好的 RM 数据工程和 probe 体系保留，但把 RM / Q 模型替换为 1.x.y 中表现最好的模型结构。2.0.10 的目标不是引入新环境，而是验收：优秀模型结构是否能同时提升 RM 局部 reward、Q model 策略执行和 greedy rollout。

第三步，进入 2.1.x、2.2.x，逐步把迷宫复杂化、概率化、可探索化。每个版本只学习一个新的方法，不急于加入 demo，不急于堆功能。目标是让 DQN 在逐步复杂化的环境中真的学会，而不是靠人工规则硬推。

---

## 三、1.x.y：BFS 拟合能力评估路线

1.x.y 的定位是模型能力评估，不是强化学习。它的任务是回答一个更基础的问题：给定迷宫、当前位置和目标点，模型能不能学出 BFS 的本质？

统一数据来源是 BFS。输入可以包含 wall map、agent position、goal position、BFS distance map、reachable mask、visit map 的不同组合；输出可以是最优动作分类、BFS distance 回归、next-distance delta 分类、policy + value 多任务。所有 1.x.y 版本都不训练 DQN，只做监督学习或简单回归。

版本组织原则是：每一个 x 对应一种模型结构，每一个 y 只做参数调整、输入通道调整、loss 权重调整或数据规模调整。

### 1.0.y：CNN Policy Classifier Baseline

任务：输入 wall / agent / goal，输出 BFS 最优动作分类。

目标：确认普通 CNN 的上限。

预期问题：CNN 可能学到局部方向偏好，但难以表达全局路径绕行。它在简单迷宫上成功率可能高，在复杂障碍上最短路径率不足。

验收指标：action accuracy、rollout success、shortest path success、BFS gap、按 easy / medium / hard 分布统计。

### 1.1.y：CNN + BFS Distance Regression

任务：输入状态，回归当前位置到 goal 的 BFS distance，或者预测四个动作后的 distance delta。

目标：测试 value-like supervision 是否比 policy classification 更稳定。

关键思想：BFS 本质上是一个全局距离场。如果模型能拟合 distance field，再从 distance field 中选择下降最快的动作，就更接近规划。

验收指标：distance MAE、distance rank accuracy、greedy success、shortest path success。

### 1.2.y：U-Net Distance Field Predictor

任务：输入 wall / goal，输出整张图每个 cell 到 goal 的 BFS distance field。

目标：让模型学习全局结构，而不是只对当前位置做分类。

关键思想：BFS 的本质不是“当前位置该往哪走”，而是“整张可达空间的距离传播”。U-Net 比普通 CNN 更适合 dense prediction。

验收指标：全图 distance MAE、reachable cell MAE、argmin neighbor action accuracy、rollout success、shortest path success。

### 1.3.y：ResNet / Deep CNN Policy-Value Model

任务：输入 wall / agent / goal，联合输出 action logits 和 value / distance。

目标：测试加深 CNN 是否足够。

预期：如果普通 CNN 的问题只是容量不足，ResNet 应显著提升；如果仍然失败，说明需要显式图结构或迭代推理。

### 1.4.y：Value Iteration Network / Differentiable Planning

任务：引入可微分 value iteration 或类似迭代规划结构。

目标：测试“带规划归纳偏置”的模型是否能逼近 BFS。

关键判断：如果 VIN 类模型显著超过 CNN/U-Net，说明迷宫任务确实需要迭代规划归纳偏置，而不是单纯视觉模式识别。

### 1.5.y：Graph Neural Network BFS Model

任务：把 free cell 当作 graph node，用 GNN message passing 学 BFS distance 或最优动作。

目标：直接测试图搜索归纳偏置。

关键判断：BFS 本质是图上的最短路传播。如果 GNN 成功，说明后续二进制漏洞挖掘中的 CFG / state graph 建模也应优先考虑图网络。

### 1.6.y：Transformer / Attention Planner

任务：把网格 cell token 化，使用 attention 学全局可达关系和动作策略。

目标：测试全局注意力是否能替代显式图搜索。

适合后续迁移：二进制程序状态、basic block、函数调用、路径约束都可以 token 化，因此这个方向与二进制漏洞挖掘更接近。

### 1.7.y：Hybrid Planner

任务：CNN / U-Net 提取空间特征，GNN / Transformer 做全局传播，最后输出 policy + distance。

目标：找到最适合后续 2.0.10 的模型结构。

最终 1.x.y 选择标准不是训练集 accuracy，而是 rollout 指标：

* success >= 95%；
* shortest path success >= 80%；
* hard maze success 稳定；
* BFS gap 小；
* 对随机 start / goal 泛化；
* 对 obstacle density 泛化；
* 不依赖手绘迷宫模板。

---

## 四、2.0.10：用优秀 BFS 模型替换 RM / Q 模型

2.0.10 不急着改任务。它的目标是把 2.0.9 中已经成熟的数据工程、RM probe、primitive reward report、greedy probe、stuck analysis 保留下来，只替换模型结构。

如果 1.x.y 证明 U-Net / VIN / GNN / Transformer 明显优于 CNN，那么 2.0.10 应该做两件事：

第一，把 Reward Model 从普通 transition CNN 换成更强结构。例如输入 transition，但内部用 distance-field-like encoder 或 graph encoder 处理全局结构。

第二，把 Q model 从普通 QCNN 换成同类优秀结构，验证同一个模型是否既能拟合 BFS，也能从 RM reward 中学策略。

2.0.10 的验收指标包括：

* RM primitive reward 顺序正确；
* toward > away > wall；
* stuck_escape > stuck_loop；
* pure argmax success 提升；
* BFS tie-break greedy 与 pure argmax 差距缩小；
* Q model success 提升；
* Q model shortest path ratio 提升；
* hard maze 不崩；
* action saturation 下降。

---

## 五、2.1.x：复杂化迷宫，但不急于 demo

2.1.x 的目标是环境复杂化，而不是加 demonstration。

建议按“每个版本只学习一个新方法”的原则推进。

### 2.1.0：随机 start / goal + BFS 难度控制

不再固定角落，不再让 difficulty 由 wall probability 决定，而是由 BFS length、reachable ratio、branching factor、dead-end density 控制。

学习目标：环境分布工程。

### 2.1.1：多 profile maze generation

显式生成 open、corridor、dead-end、rooms、bottleneck、loop-rich 等 profile。

学习目标：训练分布覆盖与泛化评估。

### 2.1.2：局部可观测迷宫

agent 只能看到局部窗口，或全局地图不完整。

学习目标：partial observability 与 memory。

### 2.1.3：动态障碍 / 概率墙

墙或通道有概率变化，动作结果不再完全确定。

学习目标：stochastic transition 与 risk-aware policy。

### 2.1.4：奖励延迟与探索任务

goal 不再总是可见，或需要先探索才能知道 goal / key / door。

学习目标：exploration、uncertainty、information gain。

---

## 六、2.2.x：从确定性迷宫到概率化可探索环境

2.2.x 开始接近真实漏洞挖掘的抽象：

* 路径不是完全确定的；
* 某些状态只有探索后才知道；
* 某些动作有失败概率；
* 局部选择可能影响长期可达性；
* 有些路径高风险但高收益；
* 有些路径短期无效但能打开新区域。

对应到二进制漏洞挖掘，这些概念可以映射为：

* maze cell -> 程序状态 / basic block / symbolic state；
* wall -> 不可行路径 / crash / 约束不可满足；
* revisit -> 状态重复 / 路径爆炸；
* goal -> 漏洞触发点 / sanitizer target / crash target；
* stochastic transition -> 输入变异的不确定效果；
* exploration reward -> 新路径覆盖率 / 新状态发现；
* stuck timeout -> fuzzer 卡在局部路径簇；
* wall recovery -> 错误输入后仍能恢复到有效路径搜索。

---

## 七、近期执行顺序

近期不要同时推进太多方向。建议顺序是：

1. 冻结 2.0.9 作为 RM 工程总结版本；
2. 开始 1.0.0，做 CNN BFS policy classifier；
3. 做 1.1.0，distance / delta regression；
4. 做 1.2.0，U-Net distance field；
5. 做 1.3.0，ResNet policy-value；
6. 做 1.4.0，VIN / differentiable planning；
7. 做 1.5.0，GNN BFS；
8. 做 1.6.0，Transformer planner；
9. 选择最优模型进入 2.0.10；
10. 在 2.0.10 验收 RM 和 Q model；
11. 再进入 2.1.x 的复杂迷宫与概率化环境。

这一阶段的核心目标是建立“模型能力基准”。只有当模型在纯 BFS 监督下已经能达到 95% success 和 80% shortest path success，继续讨论 DQN、RM、exploration 才有意义。

---

## 八、近期项目结论

2.0.n 已经证明：RM 数据工程可以把局部 reward 学到比较合理，但普通 CNN/QCNN 不是天然规划器。下一阶段必须先找到能拟合 BFS 本质的模型，再把它放回 RM 和 Q learning 框架中。

所以近期路线不是“继续调 DQN”，而是：

先做 BFS 模型能力评估，再做 RM/Q model 替换，最后再复杂化迷宫和概率化环境。

这条路线能够避免继续在 reward 和 DQN 超参数里盲目搜索，也能让项目从迷宫任务自然过渡到二进制漏洞挖掘中的路径搜索、状态探索和概率决策问题。
